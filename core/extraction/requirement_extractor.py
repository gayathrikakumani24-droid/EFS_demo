"""
Generic Semantic & Atomic Requirement Extractor (Sections 4, 5, 6, 7, 8, 14, 17, 18, 19, 23, 24, 27).

Extracts atomic requirements, hardware entities, instructions, opcodes, memory regions,
data structures, errors, performance requirements, state transitions, conditions, and
constraints from Document IR blocks. Uses dynamic category discovery and multi-pass
context-aware processing with ZERO hardcoded protocol terminology.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from config import CONFIG
from core.extraction.llm_client import get_llm_client
from core.extraction.semantic_candidate import SemanticCandidate
from core.requirement_ir.models import (
    AtomicRequirement, DiscoveredEntity, DocumentBlock, DocumentIR,
    RequirementGraphEdge, RequirementIR, SourceLocation, new_req_id
)
from utils.logger import get_logger

from core.extraction.pre_classifier import classify_chunk, ClassificationResult
from core.extraction.heuristic_extractor import extract_heuristically, HeuristicExtractionResult
from core.extraction.decision_engine import needs_llm
from core.extraction.chunk_cache import get_chunk_cache, build_cache_key, compute_chunk_hash
from core.efs_ir.validation_rules import (
    is_valid_register_candidate,
    has_sufficient_register_evidence,
    normalize_hardware_offset,
    is_explicit_address,
    VALID_ACCESS_MODES,
    is_fsm_state_name,
    is_fsm_transition_or_condition,
    is_physical_signal_name,
    is_hardware_component_candidate,
    is_valid_interface_candidate,
    DISQUALIFIED_COMPONENT_NAMES,
)

logger = get_logger("extraction.requirement_extractor")

COMPACT_EXTRACTION_PROMPT = """Extract hardware protocol entities, requirements, and inter-module relationships into structured JSON. Return ONLY non-empty keys among:
{
  "atomic_requirements": [{"type": "BEHAVIOR|INTERFACE|REGISTER|TIMING|CONSTRAINT|PERFORMANCE", "statement": "...", "subject": "...", "action": "...", "condition": null, "confidence": 0.9}],
  "entities": [{"name": "...", "type": "signal|port|state|fsm|register|interface|clock|reset|instruction|opcode|performance_requirement|component", "description": "...", "source_quote": "exact span from source text", "attributes": {"direction": "input|output|inout"}}],
  "relationships": [{"source": "...", "target": "...", "signals": ["..."], "direction": "... -> ...", "relationship": "sends_to|returns_to|connects_to|interfaces_with|communicates_with|reads_writes"}]
}

EXTRACTION GUIDELINES:
- SIGNAL & PORT DIRECTION: Extract signal/port direction ('input', 'output', 'inout') in attributes. Lamps, LEDs, indicators, status outputs, and driven signals MUST be marked as 'output'. Never classify all signals as 'input' when outputs are explicitly declared in the text.
- STATE & FSM INFORMATION: Extract states and FSM entities only when explicitly mentioned in the text. Do NOT invent synthetic states (such as IDLE, FAULT, or additional phases) if absent from the specification.
- RELATIONSHIP INFORMATION: Extract inter-module relationships, component connectivity, and interface signals (e.g. 'A sends X to B', 'B returns Y to A', 'A connects to B', 'A interfaces with B', 'A communicates with B', 'A reads/writes B'). Include source component, target component, signals list, direction ('A -> B'), and relationship verb.
- REGISTER INFORMATION: Extract named registers, register groups/categories (e.g. Configuration Registers, Status Registers), and register ranges (e.g. Matrix Registers M0-M7). Include register addresses when explicitly specified, register width, access type (RW/RO/WO), register fields, evidence/provenance, and specify address as null if not specified in source (distinguish whether address is explicit or unspecified). Do NOT discard registers or categories lacking hex addresses.
- CONSTRAINT & PERFORMANCE INFORMATION: Extract clock frequency, clock period, throughput, latency, power, area, bandwidth, timing requirements, reset timing/behavior, operating limits, and performance targets. Include source evidence and normalized values where possible.
- STRICT FIDELITY: Extract ONLY information directly supported by the source text. Require a source span/quote for every entity. Do NOT invent states, signals, registers, timing values, or defaults not present in the text.
Respond strictly in JSON without explanations or prose."""

SYSTEM_EXTRACTION_PROMPT = COMPACT_EXTRACTION_PROMPT

_HEADER_RESERVED_WORDS = {
    "signal", "port", "pin", "name", "direction", "dir", "width", "size", "description",
    "function", "register", "field", "offset", "address", "access", "reset", "default",
    "opcode", "encoding", "instruction", "mnemonic", "type", "bit", "value", "unit",
    "notes", "protocol/description", "signal name", "port name", "signal_name", "port_name",
    "pin name", "register name", "field name", "details", "meaning", "table", "total", "code"
}


class GenericRequirementExtractor:
    """Extraction engine for converting Document IR into Requirement IR with token optimization & rate-limit safety."""

    def __init__(self, doc_ir: DocumentIR):
        self.doc_ir = doc_ir
        self.llm = get_llm_client()
        self.cache = get_chunk_cache()
        self.stats = {
            "total_chunks": 0,
            "relevant_chunks": 0,
            "skipped_chunks": 0,
            "heuristic_extractions": 0,
            "llm_extractions": 0,
            "llm_fallback_extractions": 0,
            "llm_failures": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "duplicate_chunks": 0,
            "average_confidence": 0.0,
            "estimated_input_tokens": 0,
            "estimated_output_tokens": 0,
            "num_llm_calls": 0,
        }

    def extract_requirement_ir(self) -> RequirementIR:
        """Execute context-aware semantic candidate extraction with deterministic pre-classification & adaptive LLM escalation."""
        req_ir = RequirementIR(doc_ir=self.doc_ir)

        blocks = self.doc_ir.blocks
        if not blocks:
            blocks = self._synthesize_blocks_from_sections()

        self.stats["total_chunks"] = len(blocks)

        # Stage 1: Extract Semantic Candidates from structured tables
        table_reqs, table_ents, table_edges = self._extract_from_structured_tables(blocks)

        # Stage 2: Adaptive Chunk Processing (Classification -> Heuristics -> Cache/Dup -> LLM Escalation Batching)
        all_candidates: List[SemanticCandidate] = []
        blocks_needing_llm: List[DocumentBlock] = []

        total_conf_sum = 0.0
        processed_count = 0

        for b in blocks:
            # 1. Pre-classification
            class_res = classify_chunk(b.text, heading=b.section_title or "")
            if not class_res.relevant:
                self.stats["skipped_chunks"] += 1
                continue
            self.stats["relevant_chunks"] += 1

            # 2. Check Deduplication
            dup_res = self.cache.check_duplicate(b.text, page=b.page, section=b.section_title or "")
            if dup_res and dup_res.get("candidates"):
                self.stats["duplicate_chunks"] += 1
                self.stats["cache_hits"] += 1
                all_candidates.extend(dup_res["candidates"])
                continue

            # 3. Check Cache by composite key
            doc_hash = getattr(self.doc_ir, "document_hash", "doc_default")
            model_name = getattr(CONFIG.llm, "extraction_model", "model_default")
            ckey = build_cache_key(doc_hash, b.text, model_name)
            cached_data = self.cache.get(ckey)
            if cached_data and cached_data.get("candidates"):
                self.stats["cache_hits"] += 1
                all_candidates.extend(cached_data["candidates"])
                continue
            else:
                self.stats["cache_misses"] += 1

            # 4. Heuristic Extraction
            h_res = extract_heuristically(b.text, chunk_id=b.block_id, doc_id=self.doc_ir.document_id)

            # 5. Decision Engine: Does chunk need LLM?
            if needs_llm(b.text, class_res, h_res):
                blocks_needing_llm.append(b)
            else:
                # Accept Heuristic Extraction directly without calling LLM
                self.stats["heuristic_extractions"] += 1
                h_candidates = self._convert_heuristic_res_to_candidates(b, h_res)
                all_candidates.extend(h_candidates)
                total_conf_sum += h_res.confidence
                processed_count += 1
                # Save to cache & deduplicator
                self.cache.put(ckey, {"candidates": h_candidates})
                self.cache.register_chunk(b.text, {"candidates": h_candidates})

        # Stage 3: Adaptive Token-Aware Batching for Escalated Blocks
        if blocks_needing_llm:
            llm_candidates = self._process_llm_adaptive_batches(blocks_needing_llm)
            all_candidates.extend(llm_candidates)

        # Stage 4: Global Reconciliation & Context-Aware Classification
        reconciled_reqs, reconciled_ents, reconciled_edges = self._reconcile_semantic_candidates(
            all_candidates, table_reqs, table_ents, table_edges
        )

        req_ir.candidates = all_candidates
        req_ir.requirements = reconciled_reqs
        req_ir.entities = reconciled_ents
        req_ir.edges = reconciled_edges

        # Store pipeline observability statistics inside doc_ir/req_ir context
        avg_conf = (total_conf_sum / max(processed_count, 1)) if processed_count > 0 else 0.90
        self.stats["average_confidence"] = avg_conf
        req_ir.stats = self.stats
        logger.info(f"Extraction complete: {self.stats}")

        return req_ir

    def _extract_from_structured_tables(
        self, blocks: List[DocumentBlock]
    ) -> Tuple[List[AtomicRequirement], List[DiscoveredEntity], List[RequirementGraphEdge]]:
        """Pass 1: Structurally parse tables (Markdown/Text) into typed hardware entities & requirements."""
        reqs: List[AtomicRequirement] = []
        ents: List[DiscoveredEntity] = []
        edges: List[RequirementGraphEdge] = []

        for b in blocks:
            text = b.text
            if "|" not in text:
                continue

            lines = [l.strip() for l in text.split("\n") if "|" in l]
            if len(lines) < 2:
                continue

            header_line = lines[0]
            headers = [h.strip().lower() for h in header_line.split("|") if h.strip()]
            row_lines = [l for l in lines[1:] if not re.match(r"^\|?\s*:?-+:?\s*\|", l)]

            for row_idx, r_line in enumerate(row_lines):
                cells = [c.strip() for c in r_line.split("|") if c.strip()]
                if not cells or len(cells) < 2:
                    continue

                row_dict = {}
                for idx, h in enumerate(headers):
                    if idx < len(cells):
                        row_dict[h] = cells[idx]

                first_val = cells[0]
                if first_val.strip().lower() in _HEADER_RESERVED_WORDS:
                    continue

                source_loc = b.source_location
                h_str = " ".join(headers)

                # 1. Instruction / Opcode Table
                if any(k in h_str for k in ("instruction", "opcode", "mnemonic", "format", "operation", "encoding")):
                    inst_name = row_dict.get("instruction") or row_dict.get("mnemonic") or row_dict.get("name") or first_val
                    if inst_name.strip().lower() in _HEADER_RESERVED_WORDS:
                        continue
                    opcode_val = row_dict.get("opcode") or row_dict.get("encoding") or row_dict.get("code") or ""
                    fmt_val = row_dict.get("format") or row_dict.get("width") or ""
                    desc_val = row_dict.get("description") or row_dict.get("function") or ""

                    ents.append(DiscoveredEntity(
                        entity_id=new_req_id("ent"),
                        name=inst_name,
                        type="instruction",
                        description=desc_val,
                        attributes={"opcode": opcode_val, "format": fmt_val},
                        source_references=[source_loc]
                    ))
                    if opcode_val:
                        ents.append(DiscoveredEntity(
                            entity_id=new_req_id("ent"),
                            name=f"{inst_name}_OPCODE",
                            type="opcode",
                            description=f"Opcode encoding for instruction {inst_name}",
                            attributes={"opcode_hex": opcode_val, "instruction": inst_name},
                            source_references=[source_loc]
                        ))
                    reqs.append(AtomicRequirement(
                        requirement_id=new_req_id("req"),
                        type="INSTRUCTION",
                        statement=f"Instruction '{inst_name}' (opcode {opcode_val}) format {fmt_val}: {desc_val}",
                        entities=[inst_name],
                        subject=inst_name,
                        action="EXECUTE",
                        source=source_loc,
                        confidence=1.0
                    ))

                # 2. Register / Bitfield Table
                # Require explicit register headers or address/offset combined with access/width/reset/bit/field
                elif (
                    any(k in h_str for k in ("register", "reg name", "reg_name"))
                    or (
                        any(k in h_str for k in ("offset", "address"))
                        and any(k in h_str for k in ("access", "width", "reset", "rw", "ro", "wo", "r/w", "bit", "field"))
                    )
                ):
                    r_name = row_dict.get("register") or row_dict.get("reg") or row_dict.get("reg_name") or row_dict.get("name")
                    if not r_name:
                        cand_attrs = {"offset": row_dict.get("offset") or row_dict.get("address"), "access": row_dict.get("access")}
                        if is_valid_register_candidate(first_val, cand_attrs):
                            r_name = first_val
                        else:
                            r_name = None

                    if not r_name or r_name.strip().lower() in _HEADER_RESERVED_WORDS:
                        continue

                    r_offset = normalize_hardware_offset(row_dict.get("offset") or row_dict.get("address"))
                    raw_access = str(row_dict.get("access") or row_dict.get("type") or "RW").strip().upper()
                    r_access = raw_access if raw_access in VALID_ACCESS_MODES else "RW"
                    r_reset = normalize_hardware_offset(row_dict.get("reset") or row_dict.get("default"))
                    r_desc = row_dict.get("description") or ""

                    cand_attrs = {"offset": r_offset, "access": r_access, "reset_value": r_reset}
                    if not is_valid_register_candidate(r_name, cand_attrs, context=h_str):
                        continue

                    is_field = ("field" in h_str or "bit" in h_str) and not ("register" in h_str or "reg name" in h_str) and r_offset is None
                    ent_type = "field" if is_field else "register"

                    ents.append(DiscoveredEntity(
                        entity_id=new_req_id("ent"),
                        name=r_name,
                        type=ent_type,
                        description=r_desc,
                        attributes=cand_attrs,
                        source_references=[source_loc]
                    ))
                    if ent_type == "register":
                        reqs.append(AtomicRequirement(
                            requirement_id=new_req_id("req"),
                            type="REGISTER",
                            statement=f"Register '{r_name}' offset={r_offset or 'unassigned'}, access={r_access}, reset={r_reset or 'unassigned'}: {r_desc}",
                            entities=[r_name],
                            subject=r_name,
                            action="READ_WRITE",
                            source=source_loc,
                            confidence=1.0 if r_offset is not None else 0.85
                        ))

                # 3. Port / Signal Table
                elif any(k in h_str for k in ("signal", "port", "pin", "direction", "dir", "in/out")):
                    p_name = row_dict.get("signal") or row_dict.get("port") or row_dict.get("name") or first_val
                    p_name_clean = str(p_name).strip().lower()
                    if p_name_clean in _HEADER_RESERVED_WORDS:
                        continue
                    p_dir = row_dict.get("direction") or row_dict.get("dir") or row_dict.get("in/out") or "input"
                    p_width = row_dict.get("width") or row_dict.get("size") or "1"
                    p_desc = row_dict.get("description") or row_dict.get("function") or ""

                    stype = "clock" if "clock" in p_name_clean or "clk" in p_name_clean else ("reset" if "reset" in p_name_clean or "rst" in p_name_clean else ("port" if "port" in h_str else "signal"))

                    ents.append(DiscoveredEntity(
                        entity_id=new_req_id("ent"),
                        name=p_name,
                        type=stype,
                        description=p_desc,
                        attributes={"direction": p_dir, "width": p_width},
                        source_references=[source_loc]
                    ))
                    reqs.append(AtomicRequirement(
                        requirement_id=new_req_id("req"),
                        type="INTERFACE",
                        statement=f"Interface signal '{p_name}' direction={p_dir}, width={p_width}: {p_desc}",
                        entities=[p_name],
                        subject=p_name,
                        action="TRANSMIT",
                        source=source_loc,
                        confidence=1.0
                    ))

                # 4. Error Code / Exception Table
                elif any(k in h_str for k in ("error", "code", "exception", "fault")):
                    err_name = row_dict.get("error") or row_dict.get("name") or first_val
                    if err_name.strip().lower() in _HEADER_RESERVED_WORDS:
                        continue
                    err_code = row_dict.get("code") or row_dict.get("id") or ""
                    err_desc = row_dict.get("description") or row_dict.get("condition") or ""

                    ents.append(DiscoveredEntity(
                        entity_id=new_req_id("ent"),
                        name=err_name,
                        type="error",
                        description=err_desc,
                        attributes={"code": err_code},
                        source_references=[source_loc]
                    ))
                    reqs.append(AtomicRequirement(
                        requirement_id=new_req_id("req"),
                        type="ERROR",
                        statement=f"Error condition '{err_name}' (code {err_code}): {err_desc}",
                        entities=[err_name],
                        subject=err_name,
                        action="RAISE_ERROR",
                        source=source_loc,
                        confidence=1.0
                    ))

                # 5. Generic / Custom Entity Table Fallback
                else:
                    item_name = first_val
                    if item_name.strip().lower() in _HEADER_RESERVED_WORDS:
                        continue
                    item_desc = row_dict.get("description") or row_dict.get("notes") or row_dict.get("function") or " ".join(cells[1:])
                    ents.append(DiscoveredEntity(
                        entity_id=new_req_id("ent"),
                        name=item_name,
                        type="custom_entity",
                        description=item_desc,
                        attributes=row_dict,
                        source_references=[source_loc]
                    ))
                    reqs.append(AtomicRequirement(
                        requirement_id=new_req_id("req"),
                        type="CONSTRAINT",
                        statement=f"Entity '{item_name}' defined in table: {row_dict}.",
                        entities=[item_name],
                        subject=item_name,
                        action="DEFINE",
                        source=source_loc,
                        confidence=0.85
                    ))

        return reqs, ents, edges

    def _convert_heuristic_res_to_candidates(
        self, b: DocumentBlock, h_res: HeuristicExtractionResult
    ) -> List[SemanticCandidate]:
        candidates: List[SemanticCandidate] = []
        source_loc = b.source_location
        context = {"section_heading": b.section_title, "block_id": b.block_id, "page": b.page}

        for ent in h_res.entities:
            candidates.append(SemanticCandidate(
                document_id=self.doc_ir.document_id,
                chunk_id=b.block_id,
                source_text=ent.get("description", ent["name"]),
                context=context,
                candidate_type=ent.get("type", "entity").upper(),
                name=ent["name"],
                attributes=ent.get("attributes", {}),
                confidence=h_res.confidence,
                source=source_loc
            ))

        for req in h_res.requirements:
            candidates.append(SemanticCandidate(
                document_id=self.doc_ir.document_id,
                chunk_id=b.block_id,
                source_text=req["statement"],
                context=context,
                candidate_type=req.get("type", "FUNCTIONAL_REQUIREMENT").upper(),
                name=req.get("subject") or f"REQ_{b.block_id}",
                attributes=req,
                confidence=req.get("confidence", h_res.confidence),
                source=source_loc
            ))

        return candidates

    def _process_llm_adaptive_batches(
        self, blocks: List[DocumentBlock]
    ) -> List[SemanticCandidate]:
        candidates: List[SemanticCandidate] = []
        max_batch_tokens = getattr(CONFIG, "max_extraction_batch_tokens", 3000)

        current_batch: List[DocumentBlock] = []
        current_tokens = 0

        for b in blocks:
            b_tokens = max(1, int(len(b.text) / 3.5))
            if current_batch and (current_tokens + b_tokens > max_batch_tokens):
                cands = self._extract_semantic_candidates_for_batch(current_batch, 0, len(blocks))
                candidates.extend(cands)
                current_batch = [b]
                current_tokens = b_tokens
            else:
                current_batch.append(b)
                current_tokens += b_tokens

        if current_batch:
            cands = self._extract_semantic_candidates_for_batch(current_batch, 0, len(blocks))
            candidates.extend(cands)

        return candidates

    def _extract_semantic_candidates_for_batch(
        self, batch: List[DocumentBlock], index: int, total: int
    ) -> List[SemanticCandidate]:
        """Stage 1: Extract uncommitted SemanticCandidates per batch using LLM when available, falling back gracefully to heuristics."""
        candidates: List[SemanticCandidate] = []

        batch_text = "\n\n".join([
            f"[Block {b.block_id} | Section: {b.section_title} | Page: {b.page}]\n{b.text}"
            for b in batch
        ])

        est_prompt_tokens = int(len(batch_text) / 3.5)
        self.stats["estimated_input_tokens"] += est_prompt_tokens

        llm_success = False
        if self.llm and self.llm.available:
            try:
                self.stats["num_llm_calls"] += 1
                user_prompt = f"Extract hardware entities and requirements from the following document batch:\n\n{batch_text}"
                res = self.llm.complete_json(COMPACT_EXTRACTION_PROMPT, user_prompt, model=CONFIG.llm.extraction_model)
                if res and isinstance(res, dict):
                    raw_ents = res.get("entities", [])
                    raw_reqs = res.get("atomic_requirements", [])
                    raw_rels = res.get("relationships", [])

                    if raw_ents or raw_reqs or raw_rels:
                        llm_success = True
                        self.stats["llm_extractions"] += len(batch)
                        for rrel in raw_rels:
                            if not isinstance(rrel, dict):
                                continue
                            r_src = (rrel.get("source") or "").strip()
                            r_dst = (rrel.get("target") or "").strip()
                            if not r_src or not r_dst or r_src.lower() == r_dst.lower():
                                continue
                            r_sigs = rrel.get("signals", [])
                            if isinstance(r_sigs, str):
                                from core.extraction.dedicated_extractors import extract_signals_from_phrase
                                r_sigs = extract_signals_from_phrase(r_sigs)
                            r_dir = rrel.get("direction") or f"{r_src} -> {r_dst}"
                            r_rel = rrel.get("relationship", "connects_to")
                            matched_blocks = [b for b in batch if r_src.lower() in b.text.lower() or r_dst.lower() in b.text.lower()]
                            if not matched_blocks:
                                matched_blocks = [batch[0]]
                            for b_ref in matched_blocks:
                                candidates.append(SemanticCandidate(
                                    document_id=self.doc_ir.document_id,
                                    chunk_id=b_ref.block_id,
                                    source_text=f"{r_src} {r_rel} {r_dst}",
                                    context={"section_heading": b_ref.section_title, "block_id": b_ref.block_id, "page": b_ref.page},
                                    candidate_type="RELATIONSHIP",
                                    name=f"{r_src}_{r_dst}",
                                    attributes={
                                        "source": r_src,
                                        "target": r_dst,
                                        "signals": r_sigs,
                                        "direction": r_dir,
                                        "relation": r_rel
                                    },
                                    confidence=rrel.get("confidence", 0.95),
                                    source=b_ref.source_location
                                ))

                        for rent in raw_ents:
                            if not isinstance(rent, dict):
                                continue
                            ename = rent.get("name")
                            etype = rent.get("type", "custom_entity")
                            if not ename or ename.strip().lower() in _HEADER_RESERVED_WORDS:
                                continue
                            matched_blocks = [b for b in batch if re.search(r'\b' + re.escape(ename.lower()) + r'\b', b.text.lower())]
                            if not matched_blocks:
                                s_quote = rent.get("source_quote", "")
                                if s_quote and any(s_quote.lower() in b.text.lower() for b in batch):
                                    matched_blocks = [b for b in batch if s_quote.lower() in b.text.lower()]
                                else:
                                    logger.debug(f"Dropping ungrounded entity '{ename}' (type '{etype}') not found in source text.")
                                    continue
                            for b_ref in matched_blocks:
                                candidates.append(SemanticCandidate(
                                    document_id=self.doc_ir.document_id,
                                    chunk_id=b_ref.block_id,
                                    source_text=rent.get("description", ename),
                                    context={"section_heading": b_ref.section_title, "block_id": b_ref.block_id, "page": b_ref.page},
                                    candidate_type=etype.upper(),
                                    name=ename,
                                    attributes=rent.get("attributes", {}),
                                    confidence=rent.get("confidence", 0.95),
                                    source=b_ref.source_location
                                ))

                        for rreq in raw_reqs:
                            if not isinstance(rreq, dict):
                                continue
                            stmt = rreq.get("statement")
                            if not stmt:
                                continue
                            subj = rreq.get("subject") or ""
                            matched_blocks = [b for b in batch if subj and subj.lower() in b.text.lower()]
                            if not matched_blocks:
                                matched_blocks = [batch[0]]
                            for b_ref in matched_blocks:
                                candidates.append(SemanticCandidate(
                                    document_id=self.doc_ir.document_id,
                                    chunk_id=b_ref.block_id,
                                    source_text=stmt,
                                    context={"section_heading": b_ref.section_title, "block_id": b_ref.block_id, "page": b_ref.page},
                                    candidate_type=rreq.get("type", "FUNCTIONAL_REQUIREMENT").upper(),
                                    name=rreq.get("subject") or f"REQ_{b_ref.block_id}",
                                    attributes=rreq,
                                    confidence=rreq.get("confidence", 0.90),
                                    source=b_ref.source_location
                                ))
            except Exception as ex:
                logger.warning(f"LLM extraction for batch failed: {ex}. Falling back to heuristic extraction.")
                self.stats["llm_failures"] += 1

        if not llm_success:
            self.stats["llm_fallback_extractions"] += len(batch)
            for b in batch:
                h_res = extract_heuristically(b.text, chunk_id=b.block_id, doc_id=self.doc_ir.document_id)
                h_cands = self._convert_heuristic_res_to_candidates(b, h_res)
                candidates.extend(h_cands)

        # Generic Rule-Based Extraction (Specification-Agnostic) - Always runs to ensure complete extraction
        for b in batch:
            lines = b.text.split("\n")
            sec_title = b.section_title or ""
            sec_lower = sec_title.lower()

            for line in lines:
                l_strip = line.strip()
                if not l_strip or l_strip.startswith("|"):
                    continue

                source_loc = b.source_location
                context = {
                    "section_heading": sec_title,
                    "block_id": b.block_id,
                    "page": b.page
                }

                # 0. Inter-Module Relationships:
                # - "A sends X to B"
                # - "B returns X to A"
                # - "A connects to B"
                # - "A interfaces with B"
                # - "A communicates with B"
                # - "A reads/writes B"
                from core.extraction.dedicated_extractors import extract_signals_from_phrase, is_abstract_or_action_phrase
                m_send = re.match(r"^[-*]?\s*([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)\s+sends\s+(.+?)\s+to\s+([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)[.;\s]*$", l_strip, re.I)
                m_ret = re.match(r"^[-*]?\s*([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)\s+returns\s+(.+?)\s+to\s+([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)[.;\s]*$", l_strip, re.I)
                m_conn = re.match(r"^[-*]?\s*([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)\s+(?:connects\s+to|connected\s+to)\s+([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)(?:\s+(?:via|using|with)\s+(.+))?[.;\s]*$", l_strip, re.I)
                m_iface = re.match(r"^[-*]?\s*([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)\s+interfaces\s+with\s+([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)(?:\s+(?:via|using|with)\s+(.+))?[.;\s]*$", l_strip, re.I)
                m_comm = re.match(r"^[-*]?\s*([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)\s+communicates\s+with\s+([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)(?:\s+(?:via|using|with)\s+(.+))?[.;\s]*$", l_strip, re.I)
                m_rw = re.match(r"^[-*]?\s*([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)\s+(?:reads/writes\s+(?:to\s+)?|reads\s+(?:from\s+)?and\s+writes\s+to\s+|reads\s+from\s+|writes\s+to\s+)([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)(?:\s+(?:via|using|with)\s+(.+))?[.;\s]*$", l_strip, re.I)

                rel_item_match = None
                if m_send:
                    rel_item_match = (m_send.group(1).strip(), "sends_to", m_send.group(3).strip(), extract_signals_from_phrase(m_send.group(2).strip()), f"{m_send.group(1).strip()} -> {m_send.group(3).strip()}")
                elif m_ret:
                    rel_item_match = (m_ret.group(1).strip(), "returns_to", m_ret.group(3).strip(), extract_signals_from_phrase(m_ret.group(2).strip()), f"{m_ret.group(1).strip()} -> {m_ret.group(3).strip()}")
                elif m_conn:
                    sigs = extract_signals_from_phrase(m_conn.group(3).strip()) if m_conn.group(3) else []
                    rel_item_match = (m_conn.group(1).strip(), "connects_to", m_conn.group(2).strip(), sigs, f"{m_conn.group(1).strip()} -> {m_conn.group(2).strip()}")
                elif m_iface:
                    sigs = extract_signals_from_phrase(m_iface.group(3).strip()) if m_iface.group(3) else []
                    rel_item_match = (m_iface.group(1).strip(), "interfaces_with", m_iface.group(2).strip(), sigs, f"{m_iface.group(1).strip()} -> {m_iface.group(2).strip()}")
                elif m_comm:
                    sigs = extract_signals_from_phrase(m_comm.group(3).strip()) if m_comm.group(3) else []
                    rel_item_match = (m_comm.group(1).strip(), "communicates_with", m_comm.group(2).strip(), sigs, f"{m_comm.group(1).strip()} -> {m_comm.group(2).strip()}")
                elif m_rw:
                    sigs = extract_signals_from_phrase(m_rw.group(3).strip()) if m_rw.group(3) else []
                    rel_item_match = (m_rw.group(1).strip(), "reads_writes", m_rw.group(2).strip(), sigs, f"{m_rw.group(1).strip()} -> {m_rw.group(2).strip()}")

                if rel_item_match:
                    r_s, r_v, r_d, r_sigs, r_dir = rel_item_match
                    if not is_abstract_or_action_phrase(r_s) and not is_abstract_or_action_phrase(r_d) and r_s.lower() != r_d.lower():
                        candidates.append(SemanticCandidate(
                            document_id=self.doc_ir.document_id,
                            chunk_id=b.block_id,
                            source_text=l_strip,
                            context=context,
                            candidate_type="RELATIONSHIP",
                            name=f"{r_s}_{r_d}",
                            attributes={"source": r_s, "target": r_d, "signals": r_sigs, "direction": r_dir, "relation": r_v},
                            confidence=0.98,
                            source=source_loc
                        ))
                        candidates.append(SemanticCandidate(
                            document_id=self.doc_ir.document_id,
                            chunk_id=b.block_id,
                            source_text=l_strip,
                            context=context,
                            candidate_type="COMPONENT",
                            name=r_s,
                            attributes={"description": f"Component {r_s}", "section": sec_title},
                            confidence=0.95,
                            source=source_loc
                        ))
                        candidates.append(SemanticCandidate(
                            document_id=self.doc_ir.document_id,
                            chunk_id=b.block_id,
                            source_text=l_strip,
                            context=context,
                            candidate_type="COMPONENT",
                            name=r_d,
                            attributes={"description": f"Component {r_d}", "section": sec_title},
                            confidence=0.95,
                            source=source_loc
                        ))
                        continue

                # 1. Instruction Fields (e.g. Field Name: 8 bits OR Field [msb:lsb]: description)
                has_explicit_opcode = bool(re.search(r"\bopcode\s*(?:0x[0-9a-f]+|[01]{4,32}|\d+)", l_strip, re.IGNORECASE))
                if not has_explicit_opcode and (any(k in sec_lower for k in ("format", "instruction format", "field", "encoding")) or "bit" in l_strip.lower() or re.search(r"\[\d+:\d+\]", l_strip)):
                    fld_match = re.search(r"^[-*]?\s*([A-Za-z0-9_\s]+)\s*[:|-]\s*(\d+)\s*bits?", l_strip, re.IGNORECASE)
                    fld_range_match = re.search(r"^[-*]?\s*([A-Za-z0-9_]+)\s*\[(\d+):(\d+)\]\s*[:|-]?\s*(.*)", l_strip, re.IGNORECASE)
                    if fld_match:
                        f_name = fld_match.group(1).strip()
                        f_width = int(fld_match.group(2))
                        if f_name.lower() not in _HEADER_RESERVED_WORDS:
                            candidates.append(SemanticCandidate(
                                document_id=self.doc_ir.document_id,
                                chunk_id=b.block_id,
                                source_text=l_strip,
                                context=context,
                                candidate_type="INSTRUCTION_FIELD",
                                name=f_name,
                                attributes={"width": f_width},
                                confidence=0.95,
                                source=source_loc
                            ))
                            continue
                    elif fld_range_match:
                        f_name = fld_range_match.group(1).strip()
                        msb_val = int(fld_range_match.group(2))
                        lsb_val = int(fld_range_match.group(3))
                        f_width = abs(msb_val - lsb_val) + 1
                        desc_text = fld_range_match.group(4).strip()
                        if f_name.lower() not in _HEADER_RESERVED_WORDS:
                            candidates.append(SemanticCandidate(
                                document_id=self.doc_ir.document_id,
                                chunk_id=b.block_id,
                                source_text=l_strip,
                                context=context,
                                candidate_type="INSTRUCTION_FIELD",
                                name=f_name,
                                attributes={"width": f_width, "msb": msb_val, "lsb": lsb_val, "description": desc_text},
                                confidence=0.95,
                                source=source_loc
                            ))
                            continue

                # 2. Instruction / Opcode Definition
                # Format A (standard table format): 000001 : MADD (Matrix Addition) OR 0x04 : READ
                op_table_match = re.search(r"^\s*([01]{4,32}|0x[0-9A-Fa-f]+|\b\d{1,4}\b)\s*:\s*([A-Za-z0-9_]+)\s*(?:\((.*?)\)|[-–—]\s*(.*?))?$", l_strip, re.IGNORECASE)
                # Format B: MLOAD [63:0]: Opcode 0x01, Load tile...
                inst_match = re.search(r"^[-*]?\s*\b([A-Z0-9_]{2,16})\b\s*(?:\[[^\]]+\])?\s*:\s*(?:opcode\s*)?(0x[0-9A-Fa-f]+|[01]{4,32}|\d+)?\s*,?\s*(.*)", l_strip, re.IGNORECASE)

                if op_table_match:
                    op_hex = op_table_match.group(1).strip()
                    mnemonic = op_table_match.group(2).strip().upper()
                    desc = (op_table_match.group(3) or op_table_match.group(4) or "").strip()
                    if mnemonic.lower() not in _HEADER_RESERVED_WORDS and mnemonic not in ("THE", "FOR", "AND", "NOT", "NOTE", "RULE", "SECTION"):
                        candidates.append(SemanticCandidate(
                            document_id=self.doc_ir.document_id,
                            chunk_id=b.block_id,
                            source_text=l_strip,
                            context=context,
                            candidate_type="INSTRUCTION",
                            name=mnemonic,
                            attributes={"opcode": op_hex, "description": desc},
                            confidence=0.95,
                            source=source_loc
                        ))
                        candidates.append(SemanticCandidate(
                            document_id=self.doc_ir.document_id,
                            chunk_id=b.block_id,
                            source_text=l_strip,
                            context=context,
                            candidate_type="OPCODE",
                            name=f"{mnemonic}_OPCODE",
                            attributes={"opcode_hex": op_hex, "binary_encoding": op_hex, "instruction": mnemonic, "mnemonic": mnemonic},
                            confidence=0.95,
                            source=source_loc
                        ))
                        continue
                elif inst_match and any(k in sec_lower or k in l_strip.lower() for k in ("instruction", "isa", "opcode", "command", "mnemonic")):
                    mnemonic = inst_match.group(1).upper()
                    op_hex = inst_match.group(2) or ""
                    desc = inst_match.group(3) or ""

                    # If line describes a field/operand without an opcode, skip treating it as an instruction definition
                    is_field_desc = (not op_hex) and any(k in desc.lower() for k in ("field", "bit", "bits", "register operand", "immediate"))
                    if is_field_desc:
                        continue

                    if mnemonic.lower() not in _HEADER_RESERVED_WORDS and mnemonic not in ("THE", "FOR", "AND", "NOT", "NOTE", "RULE", "SECTION"):
                        candidates.append(SemanticCandidate(
                            document_id=self.doc_ir.document_id,
                            chunk_id=b.block_id,
                            source_text=l_strip,
                            context=context,
                            candidate_type="INSTRUCTION",
                            name=mnemonic,
                            attributes={"opcode": op_hex, "description": desc},
                            confidence=0.95,
                            source=source_loc
                        ))
                        if op_hex:
                            candidates.append(SemanticCandidate(
                                document_id=self.doc_ir.document_id,
                                chunk_id=b.block_id,
                                source_text=l_strip,
                                context=context,
                                candidate_type="OPCODE",
                                name=f"{mnemonic}_OPCODE",
                                attributes={"opcode_hex": op_hex, "binary_encoding": op_hex, "instruction": mnemonic, "mnemonic": mnemonic},
                                confidence=0.95,
                                source=source_loc
                            ))
                        continue

                # 3. Signals & Ports (Format A: input wire [31:0] data_in | Format B: - str_valid: input, 1 bit | Format C: Inputs: clk, reset_n)
                sig_match_a = re.search(r"\b(input|output|inout)\s+(?:wire|reg|logic)?\s*(\[\d+:\d+\]|\d+)?\s*([A-Za-z0-9_]+)\b", l_strip, re.IGNORECASE)
                sig_match_b = re.search(r"^[-*]?\s*([A-Za-z0-9_]+)\s*:\s*(input|output|inout)\b\s*,?\s*(?:(\d+|\[[^\]]+\])\s*bits?)?\s*,?\s*(.*)", l_strip, re.IGNORECASE)
                in_out_hdr = re.search(r"^(?:[-*]?\s*)?(inputs?|outputs?|inouts?)\s*:\s*(.+)$", l_strip, re.IGNORECASE)

                if sig_match_a or sig_match_b:
                    if sig_match_a:
                        p_dir = sig_match_a.group(1).lower()
                        p_width = sig_match_a.group(2) or "1"
                        p_name = sig_match_a.group(3)
                        p_desc = l_strip
                    else:
                        p_name = sig_match_b.group(1)
                        p_dir = sig_match_b.group(2).lower()
                        p_width = sig_match_b.group(3) or "1"
                        p_desc = sig_match_b.group(4) or l_strip

                    if p_name.lower() not in _HEADER_RESERVED_WORDS and p_name.upper() not in ("SECTION", "RULE", "NOTE", "THE", "FOR"):
                        stype = "clock" if "clk" in p_name.lower() or "clock" in p_name.lower() else ("reset" if "rst" in p_name.lower() or "reset" in p_name.lower() else "SIGNAL")
                        candidates.append(SemanticCandidate(
                            document_id=self.doc_ir.document_id,
                            chunk_id=b.block_id,
                            source_text=l_strip,
                            context=context,
                            candidate_type=stype,
                            name=p_name,
                            attributes={"direction": p_dir, "width": p_width, "description": p_desc},
                            confidence=0.95,
                            source=source_loc
                        ))
                        continue

                elif in_out_hdr:
                    h_dir = "input" if "in" in in_out_hdr.group(1).lower() and "out" not in in_out_hdr.group(1).lower() else ("inout" if "inout" in in_out_hdr.group(1).lower() else "output")
                    rest = in_out_hdr.group(2)
                    raw_sig_names = [s.strip("` \t\r\n.,;") for s in re.split(r"[,;]", rest)]
                    found_any = False
                    for r_item in raw_sig_names:
                        if not r_item:
                            continue
                        sig_tokens = [w for w in re.split(r"\s+", r_item) if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", w)]
                        if not sig_tokens:
                            continue
                        sig_cand_name = sig_tokens[-1] if (len(sig_tokens) > 1 and sig_tokens[0].lower() in ("active-low", "active_low", "active-high", "active_high")) else sig_tokens[0]
                        if sig_cand_name.lower() not in _HEADER_RESERVED_WORDS and sig_cand_name.upper() not in ("THE", "FOR", "AND", "NOT", "NOTE", "RULE", "SECTION"):
                            stype = "clock" if any(k in sig_cand_name.lower() for k in ("clk", "clock")) else ("reset" if any(k in sig_cand_name.lower() for k in ("rst", "reset")) else "SIGNAL")
                            candidates.append(SemanticCandidate(
                                document_id=self.doc_ir.document_id,
                                chunk_id=b.block_id,
                                source_text=l_strip,
                                context=context,
                                candidate_type=stype,
                                name=sig_cand_name,
                                attributes={"direction": h_dir, "width": "1", "description": r_item},
                                confidence=0.95,
                                source=source_loc
                            ))
                            found_any = True
                    if found_any:
                        continue

                # 3b. FSM States & Durations (e.g. - IDLE: Waiting ... OR RED lasts 5 cycles -> GREEN lasts 10 cycles)
                fsm_dur_matches = list(re.finditer(r"\b([A-Z0-9_]{2,16})\s+lasts\s+(\d+)\s+cycles?\b", l_strip, re.IGNORECASE))
                if fsm_dur_matches:
                    for fdm in fsm_dur_matches:
                        st_name = fdm.group(1).upper()
                        dur_val = int(fdm.group(2))
                        if st_name.lower() not in _HEADER_RESERVED_WORDS and st_name not in ("THE", "FOR", "AND", "NOT", "NOTE", "RULE", "SECTION"):
                            candidates.append(SemanticCandidate(
                                document_id=self.doc_ir.document_id,
                                chunk_id=b.block_id,
                                source_text=fdm.group(0),
                                context=context,
                                candidate_type="STATE",
                                name=st_name,
                                attributes={"duration_cycles": dur_val, "description": f"State {st_name} active for {dur_val} cycles"},
                                confidence=0.98,
                                source=source_loc
                            ))
                    continue

                if any(k in sec_lower for k in ("state", "fsm", "transition")):
                    st_match = re.search(r"^[-*]?\s*\b([A-Z0-9_]{2,16})\b\s*:\s*(.*)", l_strip)
                    if st_match:
                        st_name = st_match.group(1).upper()
                        st_desc = st_match.group(2)
                        if st_name.lower() not in _HEADER_RESERVED_WORDS and st_name not in ("THE", "FOR", "AND", "NOT", "NOTE", "RULE", "SECTION"):
                            candidates.append(SemanticCandidate(
                                document_id=self.doc_ir.document_id,
                                chunk_id=b.block_id,
                                source_text=l_strip,
                                context=context,
                                candidate_type="STATE",
                                name=st_name,
                                attributes={"description": st_desc},
                                confidence=0.95,
                                source=source_loc
                            ))
                            continue

                # 4. Memory Resource / Region Definitions
                mem_match = re.search(r"^[-*]?\s*([A-Za-z0-9_]+)\s*:\s*(.*)", l_strip)
                if mem_match and any(k in l_strip.lower() for k in ("sram", "dram", "cache", "buffer", "fifo", "memory", "storage", "base address")):
                    m_name = mem_match.group(1)
                    rest = mem_match.group(2)
                    if m_name.lower() not in _HEADER_RESERVED_WORDS and m_name.upper() not in ("SECTION", "RULE", "NOTE", "THE", "FOR", "OPCODE"):
                        addr_m = re.search(r"0x[0-9A-Fa-f]+", rest)
                        size_m = re.search(r"\b\d+\s*(?:KB|MB|GB|TB|bytes|bits)\b", rest, re.IGNORECASE)
                        b_addr = addr_m.group(0) if addr_m else None
                        m_size = size_m.group(0) if size_m else ""

                        candidates.append(SemanticCandidate(
                            document_id=self.doc_ir.document_id,
                            chunk_id=b.block_id,
                            source_text=l_strip,
                            context=context,
                            candidate_type="MEMORY_RESOURCE",
                            name=m_name,
                            attributes={"base_address": b_addr, "size": m_size, "description": rest},
                            confidence=0.95,
                            source=source_loc
                        ))
                        continue

                # 5. Performance Targets
                perf_match = re.search(r"\b(maximum\s+clock\s+frequency|clock\s+frequency|frequency|throughput|matrix\s+multiplies\s+throughput|power\s+consumption|power|bandwidth|capacity|latency)\b\s*[:=]\s*([^\n\r]+)", l_strip, re.IGNORECASE)
                if perf_match:
                    p_metric = perf_match.group(1).strip()
                    p_val_raw = perf_match.group(2).strip()
                    candidates.append(SemanticCandidate(
                        document_id=self.doc_ir.document_id,
                        chunk_id=b.block_id,
                        source_text=l_strip,
                        context=context,
                        candidate_type="PERFORMANCE_REQUIREMENT",
                        name=f"{p_metric.upper().replace(' ', '_')}_TARGET",
                        attributes={"metric": p_metric, "target_value": p_val_raw, "value": p_val_raw},
                        confidence=0.95,
                        source=source_loc
                    ))
                else:
                    num_match = re.search(r"\b(frequency|latency|throughput|power|bandwidth|capacity|clock)\b\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(MHz|GHz|ns|ps|cycles|bits|bytes|KB|MB|GB|mW|W|TFLOPS|GFLOPS)?", l_strip, re.IGNORECASE)
                    if num_match:
                        p_metric = num_match.group(1).upper()
                        p_val = num_match.group(2)
                        p_unit = num_match.group(3) or ""
                        candidates.append(SemanticCandidate(
                            document_id=self.doc_ir.document_id,
                            chunk_id=b.block_id,
                            source_text=l_strip,
                            context=context,
                            candidate_type="PERFORMANCE_REQUIREMENT",
                            name=f"{p_metric}_TARGET",
                            attributes={"metric": p_metric, "target_value": f"{p_val} {p_unit}".strip(), "unit": p_unit},
                            confidence=0.95,
                            source=source_loc
                        ))

                # 6. Register Definitions
                # 6a. Register Range (e.g. Matrix Registers: M0-M7 (8 registers in total))
                range_reg_m = re.search(r"([A-Za-z0-9_]+)\s*Registers?\s*:\s*([A-Za-z0-9_]+)\s*-\s*([A-Za-z0-9_]+)", l_strip, re.IGNORECASE)
                if range_reg_m:
                    cat_name = range_reg_m.group(1).strip()
                    s_name = range_reg_m.group(2).strip()
                    e_name = range_reg_m.group(3).strip()
                    m_start = re.match(r"^([A-Za-z]+)(\d+)$", s_name)
                    m_end = re.match(r"^([A-Za-z]+)(\d+)$", e_name)
                    if m_start and m_end and m_start.group(1).upper() == m_end.group(1).upper():
                        prefix = m_start.group(1).upper()
                        s_idx = int(m_start.group(2))
                        e_idx = int(m_end.group(2))
                        for r_num in range(s_idx, e_idx + 1):
                            r_id = f"{prefix}{r_num}"
                            candidates.append(SemanticCandidate(
                                document_id=self.doc_ir.document_id,
                                chunk_id=b.block_id,
                                source_text=l_strip,
                                context=context,
                                candidate_type="REGISTER",
                                name=r_id,
                                attributes={"access": "RW", "category": f"{cat_name} Registers", "range": f"{s_name}-{e_name}"},
                                confidence=0.95,
                                source=source_loc
                            ))
                        # Also preserve the named register range category itself
                        candidates.append(SemanticCandidate(
                            document_id=self.doc_ir.document_id,
                            chunk_id=b.block_id,
                            source_text=l_strip,
                            context=context,
                            candidate_type="REGISTER",
                            name=f"{cat_name} Registers {s_name}-{e_name}",
                            attributes={"access": "RW", "category": f"{cat_name} Registers", "range": f"{s_name}-{e_name}", "description": l_strip},
                            confidence=0.95,
                            source=source_loc
                        ))
                        continue

                # 6b. Register Category (e.g. Configuration Registers: Control matrix engine parameters)
                cat_reg_m = re.search(r"^[-*]?\s*([A-Za-z0-9_\s]{2,30}\s*Registers?)\s*:\s*(.+)$", l_strip, re.IGNORECASE)
                if cat_reg_m and cat_reg_m.group(1).strip().lower() not in ("register addresses", "registers", "external interfaces"):
                    cat_name = cat_reg_m.group(1).strip()
                    cat_desc = cat_reg_m.group(2).strip()
                    candidates.append(SemanticCandidate(
                        document_id=self.doc_ir.document_id,
                        chunk_id=b.block_id,
                        source_text=l_strip,
                        context=context,
                        candidate_type="REGISTER",
                        name=cat_name,
                        attributes={"access": "RW", "description": cat_desc},
                        confidence=0.90,
                        source=source_loc
                    ))
                    continue

                # 6c. Explicit Register Definition with offset
                reg_match = re.search(r"^[-*]?\s*([A-Za-z0-9_]+)\s*:\s*offset\s*(0x[0-9A-Fa-f]+|\d+)?\s*,?\s*(?:access\s*([A-Za-z]+))?\s*,?\s*(.*)", l_strip, re.IGNORECASE)
                if reg_match:
                    r_name = reg_match.group(1)
                    r_off = normalize_hardware_offset(reg_match.group(2))
                    r_acc = reg_match.group(3) or "RW"
                    r_desc = reg_match.group(4) or ""
                    if r_name.lower() not in _HEADER_RESERVED_WORDS and is_valid_register_candidate(r_name, {"offset": r_off, "access": r_acc}):
                        candidates.append(SemanticCandidate(
                            document_id=self.doc_ir.document_id,
                            chunk_id=b.block_id,
                            source_text=l_strip,
                            context=context,
                            candidate_type="REGISTER",
                            name=r_name,
                            attributes={"offset": r_off, "access": r_acc, "description": r_desc},
                            confidence=0.95 if r_off is not None else 0.8,
                            source=source_loc
                        ))
                        continue

                # 6d. Prose Named Registers (e.g. "exposes configuration registers CTRL and STATUS" or "CTRL configuration register")
                if "register" in l_strip.lower() or "csr" in l_strip.lower():
                    prose_reg_m = re.finditer(r"\b(?:exposes|contains|defines|includes|provides|has)\s+(?:(?:configuration|status|control|hardware)\s+)?registers?\s+([A-Za-z0-9_,\s]+?)(?:\.|\n|$)", l_strip, re.I)
                    matched_prose = False
                    for prm in prose_reg_m:
                        tokens = [t.strip() for t in re.split(r",|\band\b", prm.group(1)) if t.strip()]
                        for tok in tokens:
                            if is_valid_register_candidate(tok) and tok.lower() not in _HEADER_RESERVED_WORDS:
                                acc = "RO" if ("read by software" in b.text.lower() and f"{tok.lower()} is read" in b.text.lower()) else ("WO" if ("written by software" in b.text.lower() and f"{tok.lower()} is written" in b.text.lower()) else "RW")
                                candidates.append(SemanticCandidate(
                                    document_id=self.doc_ir.document_id,
                                    chunk_id=b.block_id,
                                    source_text=l_strip,
                                    context=context,
                                    candidate_type="REGISTER",
                                    name=tok,
                                    attributes={"access": acc, "category": "Configuration Registers", "description": l_strip},
                                    confidence=0.95,
                                    source=source_loc
                                ))
                                matched_prose = True

                    # Also check single register mention: "CTRL configuration register", "STATUS register"
                    if not matched_prose:
                        single_m = re.finditer(r"\b([A-Za-z0-9_]+)\s+(?:configuration\s+register|status\s+register|control\s+register)\b", l_strip, re.I)
                        for sm in single_m:
                            tok = sm.group(1).strip()
                            if is_valid_register_candidate(tok) and tok.lower() not in _HEADER_RESERVED_WORDS:
                                acc = "RO" if "status" in sm.group(0).lower() or ("read by software" in b.text.lower() and f"{tok.lower()} is read" in b.text.lower()) else ("WO" if ("written by software" in b.text.lower() and f"{tok.lower()} is written" in b.text.lower()) else "RW")
                                candidates.append(SemanticCandidate(
                                    document_id=self.doc_ir.document_id,
                                    chunk_id=b.block_id,
                                    source_text=l_strip,
                                    context=context,
                                    candidate_type="REGISTER",
                                    name=tok,
                                    attributes={"access": acc, "category": "Registers", "description": l_strip},
                                    confidence=0.95,
                                    source=source_loc
                                ))
                                matched_prose = True

                # Check for register fields in prose e.g. "CTRL contains an enable field"
                field_m = re.finditer(r"\b([A-Za-z0-9_]+)\s+contains\s+(?:an?|the)?\s*([A-Za-z0-9_]+)\s+field\b", l_strip, re.I)
                for fm in field_m:
                    reg_name = fm.group(1).strip()
                    fld_name = fm.group(2).strip()
                    if is_valid_register_candidate(reg_name) and fld_name.lower() not in _HEADER_RESERVED_WORDS:
                        candidates.append(SemanticCandidate(
                            document_id=self.doc_ir.document_id,
                            chunk_id=b.block_id,
                            source_text=l_strip,
                            context=context,
                            candidate_type="FIELD",
                            name=fld_name,
                            attributes={"register": reg_name, "parent": reg_name, "width": 1, "access": "RW", "description": l_strip},
                            confidence=0.95,
                            source=source_loc
                        ))

                # 6b. Architectural Components (e.g., Core Components, Modules, Subsystems)
                comp_section_keywords = (
                    "core components", "components", "architecture", "blocks", "modules",
                    "subsystems", "processing elements", "interfaces", "functional units",
                    "block diagram", "top level", "overview", "structure"
                )
                is_comp_sec = any(k in sec_lower for k in comp_section_keywords)
                comp_match = re.search(r"^[-*]?\s*([A-Za-z0-9_\-\s\(\)]{2,50})\s*[:|-]\s*(.+)$", l_strip)
                if comp_match:
                    c_name = comp_match.group(1).strip()
                    c_desc = comp_match.group(2).strip()
                    c_name_lower = c_name.lower()
                    if (is_comp_sec or "component" in l_strip.lower() or "module" in l_strip.lower() or "unit" in l_strip.lower() or "decoder" in l_strip.lower() or "alu" in l_strip.lower() or "processor" in l_strip.lower() or "scheduler" in l_strip.lower() or "network" in l_strip.lower() or "elements" in l_strip.lower()) and c_name_lower not in _HEADER_RESERVED_WORDS and not any(kw in c_name_lower for kw in ("note", "section", "rule", "table", "figure", "page", "step", "total", "instruction format", "format", "dimension", "dimensions")):
                        candidates.append(SemanticCandidate(
                            document_id=self.doc_ir.document_id,
                            chunk_id=b.block_id,
                            source_text=l_strip,
                            context=context,
                            candidate_type="COMPONENT",
                            name=c_name,
                            attributes={"description": c_desc, "section": sec_title},
                            confidence=0.95,
                            source=source_loc
                        ))
                        continue

                # 7. General Requirement Sentences
                if any(kw in l_strip.lower() for kw in ("shall", "must", "should", "assert", "operates", "executes", "performs", "requires", "supports", "wide", "bits")) or l_strip.lower().startswith("rule"):
                    candidates.append(SemanticCandidate(
                        document_id=self.doc_ir.document_id,
                        chunk_id=b.block_id,
                        source_text=l_strip,
                        context=context,
                        candidate_type="FUNCTIONAL_REQUIREMENT",
                        name=f"REQ_{b.block_id}",
                        attributes={"statement": l_strip},
                        confidence=0.85,
                        source=source_loc
                    ))

        return candidates

    def _reconcile_semantic_candidates(
        self,
        candidates: List[SemanticCandidate],
        table_reqs: List[AtomicRequirement],
        table_ents: List[DiscoveredEntity],
        table_edges: List[RequirementGraphEdge]
    ) -> Tuple[List[AtomicRequirement], List[DiscoveredEntity], List[RequirementGraphEdge]]:
        """Stage 2: Reconcile candidates across chunks, building clean Requirement IR entities & requirements."""
        entity_reg: Dict[Tuple[str, str], DiscoveredEntity] = {}
        atomic_reqs: List[AtomicRequirement] = list(table_reqs)
        edges: List[RequirementGraphEdge] = list(table_edges)

        # Pre-seed table entities
        for te in table_ents:
            key = (te.type.lower(), te.name.lower())
            entity_reg[key] = te

        # Group candidates by candidate_type & name
        grouped: Dict[Tuple[str, str], List[SemanticCandidate]] = {}
        for c in candidates:
            if c.candidate_type and c.name:
                key = (c.candidate_type.lower(), c.name.lower())
                grouped.setdefault(key, []).append(c)

        # Process Instruction Fields into Instruction Format dynamically if present
        instruction_fields = []
        for (ctype, cname), cand_list in grouped.items():
            if ctype == "instruction_field":
                width_val = cand_list[0].attributes.get("width", 0)
                instruction_fields.append({"name": cand_list[0].name, "width": width_val})

        if instruction_fields:
            tot_w = sum(f["width"] for f in instruction_fields if isinstance(f.get("width"), int)) or 32
            fmt_entity = DiscoveredEntity(
                entity_id=new_req_id("ent"),
                name=f"{tot_w}-Bit Instruction Format",
                type="instruction_format",
                description=f"Discovered {tot_w}-bit instruction layout",
                attributes={"total_width": tot_w, "fields": instruction_fields},
                source_references=[candidates[0].source] if candidates else []
            )
            entity_reg[("instruction_format", f"{tot_w}-bit instruction format")] = fmt_entity

        # Process all remaining candidate types
        for (ctype, cname), cand_list in grouped.items():
            first_c = cand_list[0]
            canonical_name = first_c.name or cname
            src_loc = first_c.source

            # Accumulate all distinct source references across mentions
            all_sources: List[SourceLocation] = []
            seen_locs = set()
            for c in cand_list:
                if c.source:
                    loc_tuple = (c.source.document_id, c.source.block_id, c.source.page)
                    if loc_tuple not in seen_locs:
                        seen_locs.add(loc_tuple)
                        all_sources.append(c.source)
            if not all_sources and src_loc:
                all_sources = [src_loc]

            # Merge attributes across multiple mentions so cross-chunk properties are associated
            merged_attributes: Dict[str, Any] = {}
            for c in cand_list:
                if isinstance(c.attributes, dict):
                    for k, v in c.attributes.items():
                        if v and (k not in merged_attributes or not merged_attributes[k]):
                            merged_attributes[k] = v

            if ctype in ("signal", "port", "clock", "reset"):
                explicit_dir = None
                for c in cand_list:
                    if isinstance(c.attributes, dict) and c.attributes.get("direction"):
                        d_val = str(c.attributes["direction"]).lower().strip()
                        if d_val in ("input", "output", "inout"):
                            explicit_dir = d_val
                            break

                ent_attrs = {**first_c.attributes, **merged_attributes}
                if explicit_dir:
                    ent_attrs["direction"] = explicit_dir

                cur_dir = str(ent_attrs.get("direction", "")).lower()
                c_low = canonical_name.lower()
                if not cur_dir or cur_dir not in ("input", "output", "inout"):
                    desc_s = " ".join([
                        str(canonical_name),
                        str(ent_attrs.get("description", "")),
                        str(getattr(first_c, "mention", "")),
                        str(getattr(first_c, "source_text", ""))
                    ]).lower()
                    if re.search(r"\b(output|outputs|out|lamp|led|indicator|drives?|asserts?|transmits?)\b", desc_s):
                        ent_attrs["direction"] = "output"
                    elif re.search(r"\b(input|inputs|in|receives?|clock|reset|clk|rst)\b", desc_s):
                        ent_attrs["direction"] = "input"
                    elif c_low in ("busy", "done", "error", "err", "ack", "resp") or any(c_low.endswith(sfx) for sfx in ("_out", "_tx", "_led", "_light", "_pwm", "_done", "_busy", "_err", "_error", "_resp", "_ack")):
                        ent_attrs["direction"] = "output"
                    elif any(c_low.endswith(sfx) for sfx in ("_in", "_rx", "_req", "_cmd", "_en", "_enable", "_clk", "_rst", "_rst_n", "_reset_n")):
                        ent_attrs["direction"] = "input"
                    else:
                        ent_attrs["direction"] = "input"

                target_type = ctype
                if c_low in ("clk", "clock") or "clk" in c_low:
                    target_type = "clock"
                    ent_attrs["semantic_role"] = "clock"
                elif "rst" in c_low or "reset" in c_low:
                    target_type = "reset"
                    ent_attrs["semantic_role"] = "reset"
                elif c_low in ("start", "busy", "done", "error"):
                    ent_attrs["semantic_role"] = "control"

                existing_key = None
                for candidate_k in [("signal", cname), ("clock", cname), ("reset", cname), ("port", cname)]:
                    if candidate_k in entity_reg:
                        existing_key = candidate_k
                        break

                if existing_key:
                    existing_ent = entity_reg[existing_key]
                    merged_all = {**existing_ent.attributes, **ent_attrs}
                    if existing_ent.attributes.get("direction") and not ent_attrs.get("direction"):
                        merged_all["direction"] = existing_ent.attributes["direction"]
                    existing_ent.attributes = merged_all
                    existing_ent.source_references.extend(all_sources)
                else:
                    ent = DiscoveredEntity(
                        entity_id=new_req_id("ent"),
                        name=canonical_name,
                        type=target_type,
                        description=ent_attrs.get("description", f"Signal {canonical_name}"),
                        attributes=ent_attrs,
                        source_references=all_sources
                    )
                    entity_reg[(target_type, cname)] = ent
                    atomic_reqs.append(AtomicRequirement(
                        requirement_id=new_req_id("req"),
                        type="INTERFACE",
                        statement=f"Signal '{canonical_name}' direction={ent_attrs.get('direction', 'input')}, width={ent_attrs.get('width', '1')}",
                        entities=[canonical_name],
                        subject=canonical_name,
                        action="TRANSMIT",
                        source=src_loc,
                        confidence=first_c.confidence or 0.95
                    ))

            elif ctype == "instruction":
                op_hex = merged_attributes.get("opcode") or merged_attributes.get("opcode_hex") or first_c.attributes.get("opcode", "")
                desc = merged_attributes.get("description") or first_c.attributes.get("description", f"Instruction {canonical_name}")
                ent_attrs = {"opcode": op_hex, "mnemonic": canonical_name, "description": desc, **merged_attributes}
                ent = DiscoveredEntity(
                    entity_id=new_req_id("ent"),
                    name=canonical_name,
                    type="instruction",
                    description=desc,
                    attributes=ent_attrs,
                    source_references=all_sources
                )
                entity_reg[(ctype, cname)] = ent

                if op_hex:
                    op_ent = DiscoveredEntity(
                        entity_id=new_req_id("ent"),
                        name=f"{canonical_name}_OPCODE",
                        type="opcode",
                        description=f"Opcode for {canonical_name}",
                        attributes={"opcode_hex": op_hex, "binary_encoding": op_hex, "instruction": canonical_name, "mnemonic": canonical_name},
                        source_references=all_sources
                    )
                    entity_reg[("opcode", f"{cname}_opcode")] = op_ent

                atomic_reqs.append(AtomicRequirement(
                    requirement_id=new_req_id("req"),
                    type="INSTRUCTION",
                    statement=f"Instruction '{canonical_name}' opcode={op_hex}: {desc}",
                    entities=[canonical_name],
                    subject=canonical_name,
                    action="EXECUTE",
                    source=src_loc,
                    confidence=first_c.confidence or 0.95
                ))

            elif ctype == "opcode":
                op_hex = merged_attributes.get("opcode_hex") or merged_attributes.get("opcode") or merged_attributes.get("binary_encoding") or first_c.attributes.get("opcode_hex", "")
                inst_name = merged_attributes.get("instruction") or canonical_name.replace("_OPCODE", "").replace("_opcode", "")
                ent_attrs = {"opcode_hex": op_hex, "binary_encoding": op_hex, "instruction": inst_name, "mnemonic": inst_name, **merged_attributes}
                ent = DiscoveredEntity(
                    entity_id=new_req_id("ent"),
                    name=canonical_name,
                    type="opcode",
                    description=f"Opcode encoding {op_hex}" if op_hex else f"Opcode for {inst_name}",
                    attributes=ent_attrs,
                    source_references=all_sources
                )
                entity_reg[(ctype, cname)] = ent

            elif ctype in ("interface", "channel"):
                ent_attrs = {**first_c.attributes, **merged_attributes}
                # Check if this candidate is actually a physical signal / port / clock / reset
                if is_physical_signal_name(canonical_name, ent_attrs):
                    c_low = canonical_name.lower()
                    sig_role = "clock" if "clk" in c_low or "clock" in c_low else (
                        "reset" if "rst" in c_low or "reset" in c_low else ("control" if c_low in ("start", "busy", "done", "error") else "data")
                    )
                    target_type = "clock" if sig_role == "clock" else ("reset" if sig_role == "reset" else "signal")
                    existing_key = None
                    for candidate_k in [("signal", cname), ("clock", cname), ("reset", cname), ("port", cname)]:
                        if candidate_k in entity_reg:
                            existing_key = candidate_k
                            break

                    if existing_key:
                        existing_ent = entity_reg[existing_key]
                        for k, v in ent_attrs.items():
                            if v and k not in existing_ent.attributes:
                                existing_ent.attributes[k] = v
                        existing_ent.source_references.extend(all_sources)
                    else:
                        ent = DiscoveredEntity(
                            entity_id=new_req_id("ent"),
                            name=canonical_name,
                            type=target_type,
                            description=ent_attrs.get("description", f"Signal {canonical_name}"),
                            attributes={"semantic_role": sig_role, **ent_attrs},
                            source_references=all_sources
                        )
                        entity_reg[(target_type, cname)] = ent
                        atomic_reqs.append(AtomicRequirement(
                            requirement_id=new_req_id("req"),
                            type="INTERFACE",
                            statement=f"Signal '{canonical_name}' direction={ent_attrs.get('direction', 'input')}, width={ent_attrs.get('width', '1')}",
                            entities=[canonical_name],
                            subject=canonical_name,
                            action="TRANSMIT",
                            source=src_loc,
                            confidence=first_c.confidence or 0.95
                        ))
                elif not is_fsm_state_name(canonical_name, ent_attrs) and not is_fsm_transition_or_condition(canonical_name):
                    ent = DiscoveredEntity(
                        entity_id=new_req_id("ent"),
                        name=canonical_name,
                        type="interface",
                        description=ent_attrs.get("description", f"Interface {canonical_name}"),
                        attributes=ent_attrs,
                        source_references=all_sources
                    )
                    entity_reg[(ctype, cname)] = ent

            elif ctype in ("parameter", "param"):
                ent_attrs = {**first_c.attributes, **merged_attributes}
                ent = DiscoveredEntity(
                    entity_id=new_req_id("ent"),
                    name=canonical_name,
                    type="parameter",
                    description=ent_attrs.get("description", f"Parameter {canonical_name}"),
                    attributes=ent_attrs,
                    source_references=all_sources
                )
                entity_reg[(ctype, cname)] = ent

            elif ctype in ("error", "exception", "fault"):
                ent_attrs = {**first_c.attributes, **merged_attributes}
                ent = DiscoveredEntity(
                    entity_id=new_req_id("ent"),
                    name=canonical_name,
                    type="error",
                    description=ent_attrs.get("description", f"Error condition {canonical_name}"),
                    attributes=ent_attrs,
                    source_references=all_sources
                )
                entity_reg[(ctype, cname)] = ent
                atomic_reqs.append(AtomicRequirement(
                    requirement_id=new_req_id("req"),
                    type="ERROR",
                    statement=ent_attrs.get("description", f"Error condition {canonical_name}"),
                    entities=[canonical_name],
                    subject=canonical_name,
                    action="HANDLE_ERROR",
                    source=src_loc,
                    confidence=0.90
                ))

            elif ctype in ("instruction_format", "format"):
                ent_attrs = {**first_c.attributes, **merged_attributes}
                ent = DiscoveredEntity(
                    entity_id=new_req_id("ent"),
                    name=canonical_name,
                    type="instruction_format",
                    description=ent_attrs.get("description", f"Instruction Format {canonical_name}"),
                    attributes=ent_attrs,
                    source_references=all_sources
                )
                entity_reg[(ctype, cname)] = ent

            elif ctype in ("component", "module", "subsystem", "block", "unit", "functional_unit", "processing_element", "custom_entity", "core_component", "architectural_component"):
                ent_attrs = {**first_c.attributes, **merged_attributes}
                # Hard Semantic Boundary: FSM states, physical signals, and meta-labels must NEVER become components!
                if (
                    is_fsm_state_name(canonical_name, ent_attrs)
                    or is_physical_signal_name(canonical_name, ent_attrs)
                    or is_fsm_transition_or_condition(canonical_name)
                    or not is_hardware_component_candidate(canonical_name, ent_attrs)
                    or canonical_name.strip().lower() in DISQUALIFIED_COMPONENT_NAMES
                ):
                    # Route to signal if appropriate
                    if is_physical_signal_name(canonical_name, ent_attrs):
                        ent = DiscoveredEntity(
                            entity_id=new_req_id("ent"),
                            name=canonical_name,
                            type="signal",
                            description=ent_attrs.get("description", f"Signal {canonical_name}"),
                            attributes=ent_attrs,
                            source_references=all_sources
                        )
                        entity_reg[("signal", cname)] = ent
                    # FSM states and disqualified placeholders are rejected
                    continue

                ent_type = "component" if ctype in ("component", "core_component", "architectural_component", "block", "unit", "processing_element") else "module"
                ent = DiscoveredEntity(
                    entity_id=new_req_id("ent"),
                    name=canonical_name,
                    type=ent_type,
                    description=ent_attrs.get("description", f"Component {canonical_name}"),
                    attributes=ent_attrs,
                    source_references=all_sources
                )
                entity_reg[(ctype, cname)] = ent

            elif ctype in ("relationship", "interface_relationship", "module_relationship"):
                for c in cand_list:
                    r_src = c.attributes.get("source") or c.name.split("_")[0]
                    r_dst = c.attributes.get("target") or (c.name.split("_")[1] if "_" in c.name else "")
                    r_sigs = c.attributes.get("signals", [])
                    r_dir = c.attributes.get("direction") or f"{r_src} -> {r_dst}"
                    r_rel = c.attributes.get("relation", "connects_to")
                    if not (r_src and r_dst):
                        continue

                    # 1. Hard Semantic Boundary: FSM transitions must ONLY be FSM transitions, NEVER interfaces!
                    if is_fsm_state_name(r_src) or is_fsm_state_name(r_dst):
                        cond = r_sigs[0] if (r_sigs and isinstance(r_sigs, list)) else c.attributes.get("condition", "")
                        atomic_reqs.append(AtomicRequirement(
                            requirement_id=new_req_id("req"),
                            type="FSM",
                            statement=f"FSM transition from '{r_src}' to '{r_dst}'" + (f" when {cond}" if cond else ""),
                            entities=[r_src, r_dst],
                            subject=r_src,
                            action="TRANSITION",
                            condition=str(cond) if cond else None,
                            source=src_loc,
                            confidence=0.95
                        ))
                        # Do NOT create interface entity or interface requirement
                        continue

                    # 2. Hard Semantic Boundary: Signal-to-Component associations (clk -> controller, controller -> busy)
                    known_sig_names = {k[1] for k in entity_reg if k[0] in ("signal", "port", "clock", "reset")}
                    s_sig_ent = entity_reg.get(("signal", r_src.lower())) or entity_reg.get(("port", r_src.lower())) or entity_reg.get(("clock", r_src.lower())) or entity_reg.get(("reset", r_src.lower()))
                    d_sig_ent = entity_reg.get(("signal", r_dst.lower())) or entity_reg.get(("port", r_dst.lower())) or entity_reg.get(("clock", r_dst.lower())) or entity_reg.get(("reset", r_dst.lower()))

                    s_attrs = s_sig_ent.attributes if s_sig_ent else ({"type": "signal"} if r_src.lower() in known_sig_names else {})
                    d_attrs = d_sig_ent.attributes if d_sig_ent else ({"type": "signal"} if r_dst.lower() in known_sig_names else {})

                    s_is_sig = bool(s_sig_ent) or is_physical_signal_name(r_src, s_attrs, known_signals=known_sig_names)
                    d_is_sig = bool(d_sig_ent) or is_physical_signal_name(r_dst, d_attrs, known_signals=known_sig_names)

                    if s_is_sig and not d_is_sig:
                        # r_src is an input signal to r_dst
                        sig_ent = s_sig_ent or entity_reg.get(("interface", r_src.lower()))
                        if sig_ent:
                            sig_ent.type = "signal"
                            sig_ent.attributes["owner"] = r_dst
                            sig_ent.attributes["component"] = r_dst
                            sig_ent.attributes["direction"] = "input"
                            entity_reg[("signal", r_src.lower())] = sig_ent
                        else:
                            sig_role = "clock" if "clk" in r_src.lower() or "clock" in r_src.lower() else (
                                "reset" if "rst" in r_src.lower() or "reset" in r_src.lower() else "data"
                            )
                            ent = DiscoveredEntity(
                                entity_id=new_req_id("ent"),
                                name=r_src,
                                type="signal",
                                description=f"Signal {r_src} input to {r_dst}",
                                attributes={"direction": "input", "owner": r_dst, "component": r_dst, "semantic_role": sig_role},
                                source_references=all_sources
                            )
                            entity_reg[("signal", r_src.lower())] = ent
                        atomic_reqs.append(AtomicRequirement(
                            requirement_id=new_req_id("req"),
                            type="INTERFACE",
                            statement=f"Signal '{r_src}' direction=input, owner={r_dst}",
                            entities=[r_src, r_dst],
                            subject=r_src,
                            action="TRANSMIT",
                            source=src_loc,
                            confidence=0.95
                        ))
                        continue

                    if not s_is_sig and d_is_sig:
                        # r_dst is an output signal from r_src
                        sig_ent = d_sig_ent or entity_reg.get(("interface", r_dst.lower()))
                        if sig_ent:
                            sig_ent.type = "signal"
                            sig_ent.attributes["owner"] = r_src
                            sig_ent.attributes["component"] = r_src
                            sig_ent.attributes["direction"] = "output"
                            entity_reg[("signal", r_dst.lower())] = sig_ent
                        else:
                            ent = DiscoveredEntity(
                                entity_id=new_req_id("ent"),
                                name=r_dst,
                                type="signal",
                                description=f"Signal {r_dst} output from {r_src}",
                                attributes={"direction": "output", "owner": r_src, "component": r_src},
                                source_references=all_sources
                            )
                            entity_reg[("signal", r_dst.lower())] = ent
                        atomic_reqs.append(AtomicRequirement(
                            requirement_id=new_req_id("req"),
                            type="INTERFACE",
                            statement=f"Signal '{r_dst}' direction=output, owner={r_src}",
                            entities=[r_dst, r_src],
                            subject=r_dst,
                            action="TRANSMIT",
                            source=src_loc,
                            confidence=0.95
                        ))
                        continue

                    # If either is a signal, it's not a multi-signal component interface
                    if s_is_sig or d_is_sig:
                        continue

                    # 3. Check for component descriptions (e.g. Control Unit : Manages execution...)
                    if r_rel in ("description", "is_description", "manages", "orchestrates", "controls") or len(r_dst.split()) > 4:
                        comp_ent = entity_reg.get(("component", r_src.lower())) or entity_reg.get(("module", r_src.lower()))
                        if comp_ent and not comp_ent.description:
                            comp_ent.description = r_dst
                        continue

                    # 4. Actual EFSInterface validation
                    is_valid_if, rej_reason = is_valid_interface_candidate(r_src, r_dst, c.attributes)
                    if not is_valid_if:
                        logger.info(f"Skipping non-interface candidate relationship '{r_src} -> {r_dst}': {rej_reason}")
                        continue

                    # Genuine Hardware Interface
                    if_name = f"{r_src}_{r_dst}_interface"
                    ent = DiscoveredEntity(
                        entity_id=new_req_id("ent"),
                        name=if_name,
                        type="interface",
                        description=f"Inter-module relationship: {r_src} {r_rel} {r_dst}",
                        attributes={
                            "source": r_src,
                            "target": r_dst,
                            "signals": r_sigs,
                            "direction": r_dir,
                            "relation": r_rel,
                            "role": "interconnect"
                        },
                        source_references=all_sources
                    )
                    entity_reg[("interface", if_name.lower())] = ent
                    atomic_reqs.append(AtomicRequirement(
                        requirement_id=new_req_id("req"),
                        type="INTERFACE",
                        statement=f"Relationship '{r_src}' -> '{r_dst}' via signals {r_sigs}",
                        entities=[r_src, r_dst],
                        subject=r_src,
                        action="CONNECT",
                        source=src_loc,
                        confidence=0.95
                    ))

            elif ctype in ("memory", "memory_region", "memory_resource", "sram", "dram", "cache", "buffer"):
                ent_attrs = {**first_c.attributes, **merged_attributes}
                b_addr = normalize_hardware_offset(ent_attrs.get("base_address")) or ""
                sz = ent_attrs.get("size", "")
                ent = DiscoveredEntity(
                    entity_id=new_req_id("ent"),
                    name=canonical_name,
                    type="memory_region",
                    description=ent_attrs.get("description", f"Memory domain {canonical_name}"),
                    attributes={"base_address": b_addr, "size": sz, "type": ent_attrs.get("type", "sram"), **ent_attrs},
                    source_references=all_sources
                )
                entity_reg[(ctype, cname)] = ent
                atomic_reqs.append(AtomicRequirement(
                    requirement_id=new_req_id("req"),
                    type="MEMORY",
                    statement=f"Memory region '{canonical_name}' base_address={b_addr or 'unassigned'}, size={sz}",
                    entities=[canonical_name],
                    subject=canonical_name,
                    action="ALLOCATE",
                    source=src_loc,
                    confidence=0.95
                ))

            elif ctype == "data_structure":
                ent_attrs = {**first_c.attributes, **merged_attributes}
                ent = DiscoveredEntity(
                    entity_id=new_req_id("ent"),
                    name=canonical_name,
                    type="data_structure",
                    description=ent_attrs.get("description", f"Data structure {canonical_name}"),
                    attributes=ent_attrs,
                    source_references=all_sources
                )
                entity_reg[(ctype, cname)] = ent

            elif ctype == "performance_requirement":
                ent_attrs = {**first_c.attributes, **merged_attributes}
                metric = ent_attrs.get("metric", "PERFORMANCE")
                val = ent_attrs.get("target_value", "")
                unit = ent_attrs.get("unit", "")
                ent = DiscoveredEntity(
                    entity_id=new_req_id("ent"),
                    name=canonical_name,
                    type="performance_requirement",
                    description=f"Performance target {metric} = {val} {unit}",
                    attributes={"metric": metric, "target_value": val, "unit": unit, **ent_attrs},
                    source_references=all_sources
                )
                entity_reg[(ctype, cname)] = ent
                atomic_reqs.append(AtomicRequirement(
                    requirement_id=new_req_id("req"),
                    type="PERFORMANCE",
                    statement=f"Performance constraint for {metric}: {val} {unit}.",
                    entities=[canonical_name],
                    subject=canonical_name,
                    action="CONSTRAIN",
                    value=val,
                    unit=unit,
                    source=src_loc,
                    confidence=0.95
                ))

            elif ctype == "timing":
                stmt = first_c.attributes.get("statement", first_c.source_text)
                t_expr = first_c.attributes.get("timing", "")
                atomic_reqs.append(AtomicRequirement(
                    requirement_id=new_req_id("req"),
                    type="TIMING",
                    statement=stmt,
                    timing=t_expr,
                    entities=[canonical_name] if canonical_name else [],
                    subject=canonical_name,
                    action="CONSTRAIN",
                    source=src_loc,
                    confidence=first_c.confidence or 0.90
                ))

            elif ctype == "register":
                ent_attrs = {**first_c.attributes, **merged_attributes}
                r_off = normalize_hardware_offset(ent_attrs.get("offset"))
                r_acc = ent_attrs.get("access", "RW")
                if is_valid_register_candidate(canonical_name, {"offset": r_off, "access": r_acc}):
                    ent = DiscoveredEntity(
                        entity_id=new_req_id("ent"),
                        name=canonical_name,
                        type="register",
                        description=ent_attrs.get("description", f"Register {canonical_name}"),
                        attributes={"offset": r_off, "access": r_acc, **ent_attrs},
                        source_references=all_sources
                    )
                    entity_reg[(ctype, cname)] = ent
                    atomic_reqs.append(AtomicRequirement(
                        requirement_id=new_req_id("req"),
                        type="REGISTER",
                        statement=f"Register '{canonical_name}' offset={r_off or 'unassigned'}, access={r_acc}: {ent_attrs.get('description', '')}",
                        entities=[canonical_name],
                        subject=canonical_name,
                        action="READ_WRITE",
                        source=src_loc,
                        confidence=first_c.confidence or 0.95
                    ))

            elif ctype in ("functional_requirement", "behavior", "flow", "execution_model"):
                stmt = first_c.attributes.get("statement", first_c.source_text)
                subj = first_c.attributes.get("subject", canonical_name)
                act = first_c.attributes.get("action", "EXECUTE")
                cond = first_c.attributes.get("condition", "")
                atomic_reqs.append(AtomicRequirement(
                    requirement_id=new_req_id("req"),
                    type="BEHAVIOR",
                    statement=stmt,
                    entities=[subj] if subj else [],
                    subject=subj,
                    action=act,
                    condition=cond,
                    source=src_loc,
                    confidence=0.85
                ))

            elif ctype in ("state", "fsm_state"):
                ent_attrs = {**first_c.attributes, **merged_attributes}
                dur_c = ent_attrs.get("duration_cycles")
                desc = ent_attrs.get("description", f"FSM State {canonical_name}" + (f" (duration: {dur_c} cycles)" if dur_c else ""))
                ent = DiscoveredEntity(
                    entity_id=new_req_id("ent"),
                    name=canonical_name,
                    type="state",
                    description=desc,
                    attributes={"duration_cycles": dur_c, **ent_attrs},
                    source_references=all_sources
                )
                entity_reg[("state", cname)] = ent
                atomic_reqs.append(AtomicRequirement(
                    requirement_id=new_req_id("req"),
                    type="FSM",
                    statement=f"FSM State '{canonical_name}'" + (f" duration={dur_c} cycles" if dur_c else "") + f": {desc}",
                    entities=[canonical_name],
                    subject=canonical_name,
                    state=canonical_name,
                    action="STATE_DEFINITION",
                    timing=str(dur_c) if dur_c else None,
                    source=src_loc,
                    confidence=first_c.confidence or 0.95
                ))

            logger.debug(
                f"[ENTITY_TRACE] mention='{first_c.source_text[:40]}' | name='{canonical_name}' | "
                f"type='{ctype}' | mentions={len(all_sources)} | "
                f"chunks={[s.block_id for s in all_sources]}"
            )

        return atomic_reqs, list(entity_reg.values()), edges

    def _build_batch_context(self, batch: List[DocumentBlock], index: int, total: int) -> str:
        ctx_parts = [f"=== DOCUMENT BATCH {index//5 + 1} ({len(batch)} blocks) ==="]
        for b in batch:
            ctx_parts.append(
                f"[Block {b.block_id} | Section: {b.section_title} | Page: {b.page}]\n{b.text}\n"
            )
        return "\n".join(ctx_parts)

    def _synthesize_blocks_from_sections(self) -> List[DocumentBlock]:
        blocks = []
        for idx, sec in enumerate(self.doc_ir.sections):
            b_id = f"BLOCK_{idx+1:03d}"
            text = sec.get("text", "") or sec.get("title", "")
            loc = SourceLocation(
                document_id=self.doc_ir.document_id,
                section=sec.get("title", ""),
                section_id=sec.get("section_id"),
                page=sec.get("page_start", 1),
                block_id=b_id
            )
            blocks.append(DocumentBlock(
                block_id=b_id,
                type="paragraph",
                section_id=sec.get("section_id", ""),
                section_title=sec.get("title", ""),
                page=sec.get("page_start", 1),
                text=text,
                source_location=loc
            ))
        return blocks
