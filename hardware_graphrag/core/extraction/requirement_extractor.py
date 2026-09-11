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

logger = get_logger("extraction.requirement_extractor")

SYSTEM_EXTRACTION_PROMPT = """You are a Universal Hardware Compiler and Semantic Extraction Engine.
Your task is to analyze specification document blocks and extract ALL hardware concepts into structured, protocol-independent JSON.

DO NOT use hardcoded assumptions about the protocol or hardware architecture. Discover the terminology, signal names, register names, instruction mnemonics, opcodes, memory structures, error codes, performance targets, states, and transaction concepts dynamically from the text.

For each block batch, extract:
1. "atomic_requirements": List of atomic requirement objects:
   - "type": Generic type (e.g. BEHAVIOR, HOLD, TRANSACTION, INTERFACE, STATE, TIMING, CONSTRAINT, REGISTER, INSTRUCTION, MEMORY, ERROR, PERFORMANCE, POWER, SECURITY, PROTOCOL_RULE, UNKNOWN).
   - "statement": The exact or normalized requirement sentence.
   - "entities": List of discovered entity names referenced in this requirement.
   - "subject": Discovered entity name performing or owning the action (or null).
   - "action": Action performed (e.g. ASSERT, DEASSERT, TRANSMIT, READ, WRITE, EXECUTE, TRANSITION, DECODE) (or null).
   - "condition": Trigger or guard condition string (or null).
   - "event": Triggering event string (or null).
   - "state": Current state string (or null).
   - "next_state": Target state string after transition (or null).
   - "timing": Timing rule or latency string (or null).
   - "value": Numerical or parametric value (or null).
   - "unit": Physical/logical unit (e.g. MHz, ns, cycles, bits, bytes, mW) (or null).
   - "constraints": List of constraint strings.
   - "knowledge_status": "EXPLICIT" if directly stated, "DERIVED" if logically deduced, "INFERRED" if inferred with ambiguity.
   - "confidence": Float between 0.0 and 1.0.

2. "entities": List of discovered entities across ALL semantic categories:
   - "name": Discovered canonical name (e.g. signal name, instruction mnemonic, register name, memory region, error code, state).
   - "type": Entity category (e.g. signal, port, register, field, module, interface, instruction, opcode, memory_region, data_structure, error, performance_requirement, power_requirement, security_requirement, transaction, state, event, parameter, clock, reset, custom_entity).
   - "description": Detailed summary of function or semantics.
   - "attributes": Dictionary of attributes (width, direction, offset, reset_value, opcode_hex, mnemonic, format, depth, base_address, unit, value, constraint, etc.).

3. "relationships": List of relationship triples:
   - "source": Source entity name or requirement ID.
   - "relationship": Generic relationship type (DEPENDS_ON, REFERENCES, USES, PRODUCES, CONSUMES, TRIGGERS, FOLLOWED_BY, PRECEDES, CAUSES, TRANSITIONS_TO, CONSTRAINS, CONFIGURES, CONTROLS, BELONGS_TO, PART_OF, ASSOCIATED_WITH).
   - "target": Target entity name or requirement ID.

Return strictly a JSON object with keys "atomic_requirements", "entities", "relationships".
"""

_HEADER_RESERVED_WORDS = {
    "signal", "port", "pin", "name", "direction", "dir", "width", "size", "description",
    "function", "register", "field", "offset", "address", "access", "reset", "default",
    "opcode", "encoding", "instruction", "mnemonic", "type", "bit", "value", "unit",
    "notes", "protocol/description", "signal name", "port name", "signal_name", "port_name",
    "pin name", "register name", "field name", "details", "meaning", "table", "total", "code"
}


class GenericRequirementExtractor:
    """Extraction engine for converting Document IR into Requirement IR."""

    def __init__(self, doc_ir: DocumentIR):
        self.doc_ir = doc_ir
        self.llm = get_llm_client()

    def extract_requirement_ir(self) -> RequirementIR:
        """Execute two-stage context-aware semantic candidate extraction & global reconciliation."""
        req_ir = RequirementIR(doc_ir=self.doc_ir)

        blocks = self.doc_ir.blocks
        if not blocks:
            blocks = self._synthesize_blocks_from_sections()

        # Stage 1: Extract Semantic Candidates from structured tables & text blocks
        table_reqs, table_ents, table_edges = self._extract_from_structured_tables(blocks)

        all_candidates: List[SemanticCandidate] = []
        batch_size = 3
        for i in range(0, max(len(blocks), 1), batch_size):
            batch = blocks[i:i + batch_size]
            batch_candidates = self._extract_semantic_candidates_for_batch(batch, i, len(blocks))
            all_candidates.extend(batch_candidates)

        # Stage 2: Global Reconciliation & Context-Aware Classification
        reconciled_reqs, reconciled_ents, reconciled_edges = self._reconcile_semantic_candidates(
            all_candidates, table_reqs, table_ents, table_edges
        )

        req_ir.candidates = all_candidates
        req_ir.requirements = reconciled_reqs
        req_ir.entities = reconciled_ents
        req_ir.edges = reconciled_edges

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

                # 2. Port / Signal Table
                elif any(k in h_str for k in ("signal", "port", "pin", "direction", "width", "dir", "in/out")):
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

                # 3. Register / Bitfield Table
                elif any(k in h_str for k in ("register", "offset", "address", "bit", "field", "reset")):
                    r_name = row_dict.get("register") or row_dict.get("field") or row_dict.get("name") or first_val
                    if r_name.strip().lower() in _HEADER_RESERVED_WORDS:
                        continue
                    r_offset = row_dict.get("offset") or row_dict.get("address") or "0x0"
                    r_access = row_dict.get("access") or row_dict.get("type") or "RW"
                    r_reset = row_dict.get("reset") or row_dict.get("default") or "0x0"
                    r_desc = row_dict.get("description") or ""

                    ents.append(DiscoveredEntity(
                        entity_id=new_req_id("ent"),
                        name=r_name,
                        type="register" if "register" in h_str else "field",
                        description=r_desc,
                        attributes={"offset": r_offset, "access": r_access, "reset_value": r_reset},
                        source_references=[source_loc]
                    ))
                    reqs.append(AtomicRequirement(
                        requirement_id=new_req_id("req"),
                        type="REGISTER",
                        statement=f"Register '{r_name}' offset={r_offset}, access={r_access}, reset={r_reset}: {r_desc}",
                        entities=[r_name],
                        subject=r_name,
                        action="READ_WRITE",
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

    def _extract_semantic_candidates_for_batch(
        self, batch: List[DocumentBlock], index: int, total: int
    ) -> List[SemanticCandidate]:
        """Stage 1: Extract uncommitted SemanticCandidates per batch using LLM when available, falling back to generic regex."""
        candidates: List[SemanticCandidate] = []

        batch_text = "\n\n".join([
            f"[Block {b.block_id} | Section: {b.section_title} | Page: {b.page}]\n{b.text}"
            for b in batch
        ])

        if self.llm and self.llm.available:
            try:
                user_prompt = f"Extract hardware entities and requirements from the following document batch:\n\n{batch_text}"
                res = self.llm.complete_json(SYSTEM_EXTRACTION_PROMPT, user_prompt, model=CONFIG.llm.extraction_model)
                if res and isinstance(res, dict):
                    raw_ents = res.get("entities", [])
                    raw_reqs = res.get("atomic_requirements", [])
                    
                    for rent in raw_ents:
                        if not isinstance(rent, dict):
                            continue
                        ename = rent.get("name")
                        etype = rent.get("type", "custom_entity")
                        if not ename or ename.strip().lower() in _HEADER_RESERVED_WORDS:
                            continue
                        b_ref = batch[0]
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
                        b_ref = batch[0]
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
                    pass
            except Exception as ex:
                logger.warning(f"LLM extraction for batch {index} failed: {ex}. Using generic specification-agnostic heuristics.")

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

                # 1. Instruction Fields (e.g. Field Name: 8 bits)
                if any(k in sec_lower for k in ("format", "instruction format", "field", "encoding")) or "bit" in l_strip.lower():
                    fld_match = re.search(r"^[-*]?\s*([A-Za-z0-9_\s]+)\s*[:|-]\s*(\d+)\s*bits?", l_strip, re.IGNORECASE)
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

                # 2. Instruction / Opcode Definition (e.g. - MLOAD [63:0]: Opcode 0x01, Load tile...)
                inst_match = re.search(r"^[-*]?\s*\b([A-Z0-9_]{2,16})\b\s*(?:\[[^\]]+\])?\s*:\s*(?:opcode\s*)?(0x[0-9A-Fa-f]+|\d+)?\s*,?\s*(.*)", l_strip, re.IGNORECASE)
                if inst_match and any(k in sec_lower or k in l_strip.lower() for k in ("instruction", "isa", "opcode", "command", "mnemonic")):
                    mnemonic = inst_match.group(1).upper()
                    op_hex = inst_match.group(2) or ""
                    desc = inst_match.group(3) or ""

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
                                attributes={"opcode_hex": op_hex, "instruction": mnemonic},
                                confidence=0.95,
                                source=source_loc
                            ))
                        continue

                # 3. Signals & Ports (Format A: input wire [31:0] data_in | Format B: - str_valid: input, 1 bit, Description)
                sig_match_a = re.search(r"\b(input|output|inout)\s+(?:wire|reg|logic)?\s*(\[\d+:\d+\]|\d+)?\s*([A-Za-z0-9_]+)\b", l_strip, re.IGNORECASE)
                sig_match_b = re.search(r"^[-*]?\s*([A-Za-z0-9_]+)\s*:\s*(input|output|inout)\b\s*,?\s*(?:(\d+|\[[^\]]+\])\s*bits?)?\s*,?\s*(.*)", l_strip, re.IGNORECASE)
                
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

                # 3b. FSM States (e.g. - IDLE: Waiting for trigger signal compute_start)
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
                        b_addr = addr_m.group(0) if addr_m else "0x0"
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
                        attributes={"metric": p_metric, "target_value": p_val, "unit": p_unit},
                        confidence=0.95,
                        source=source_loc
                    ))

                # 6. Register Definitions
                reg_match = re.search(r"^[-*]?\s*([A-Za-z0-9_]+)\s*:\s*offset\s*(0x[0-9A-Fa-f]+|\d+)?\s*,?\s*(?:access\s*([A-Za-z]+))?\s*,?\s*(.*)", l_strip, re.IGNORECASE)
                if reg_match:
                    r_name = reg_match.group(1)
                    r_off = reg_match.group(2) or "0x0"
                    r_acc = reg_match.group(3) or "RW"
                    r_desc = reg_match.group(4) or ""
                    if r_name.lower() not in _HEADER_RESERVED_WORDS:
                        candidates.append(SemanticCandidate(
                            document_id=self.doc_ir.document_id,
                            chunk_id=b.block_id,
                            source_text=l_strip,
                            context=context,
                            candidate_type="REGISTER",
                            name=r_name,
                            attributes={"offset": r_off, "access": r_acc, "description": r_desc},
                            confidence=0.95,
                            source=source_loc
                        ))

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

            if ctype in ("signal", "port", "clock", "reset"):
                ent = DiscoveredEntity(
                    entity_id=new_req_id("ent"),
                    name=canonical_name,
                    type=ctype,
                    description=first_c.attributes.get("description", f"Signal {canonical_name}"),
                    attributes=first_c.attributes,
                    source_references=[src_loc]
                )
                entity_reg[(ctype, cname)] = ent
                atomic_reqs.append(AtomicRequirement(
                    requirement_id=new_req_id("req"),
                    type="INTERFACE",
                    statement=f"Signal '{canonical_name}' direction={first_c.attributes.get('direction', 'input')}, width={first_c.attributes.get('width', '1')}",
                    entities=[canonical_name],
                    subject=canonical_name,
                    action="TRANSMIT",
                    source=src_loc,
                    confidence=first_c.confidence or 0.95
                ))

            elif ctype == "instruction":
                op_hex = first_c.attributes.get("opcode", "")
                desc = first_c.attributes.get("description", f"Instruction {canonical_name}")
                ent = DiscoveredEntity(
                    entity_id=new_req_id("ent"),
                    name=canonical_name,
                    type="instruction",
                    description=desc,
                    attributes={"opcode": op_hex, "mnemonic": canonical_name},
                    source_references=[src_loc]
                )
                entity_reg[(ctype, cname)] = ent

                if op_hex:
                    op_ent = DiscoveredEntity(
                        entity_id=new_req_id("ent"),
                        name=f"{canonical_name}_OPCODE",
                        type="opcode",
                        description=f"Opcode for {canonical_name}",
                        attributes={"opcode_hex": op_hex, "instruction": canonical_name},
                        source_references=[src_loc]
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
                op_hex = first_c.attributes.get("opcode_hex", "")
                ent = DiscoveredEntity(
                    entity_id=new_req_id("ent"),
                    name=canonical_name,
                    type="opcode",
                    description=f"Opcode encoding {op_hex}",
                    attributes={"opcode_hex": op_hex},
                    source_references=[src_loc]
                )
                entity_reg[(ctype, cname)] = ent

            elif ctype in ("component", "module"):
                ent = DiscoveredEntity(
                    entity_id=new_req_id("ent"),
                    name=canonical_name,
                    type="module",
                    description=first_c.attributes.get("description", f"Component {canonical_name}"),
                    attributes={},
                    source_references=[src_loc]
                )
                entity_reg[(ctype, cname)] = ent

            elif ctype == "memory_resource":
                b_addr = first_c.attributes.get("base_address", "0x0")
                sz = first_c.attributes.get("size", "")
                ent = DiscoveredEntity(
                    entity_id=new_req_id("ent"),
                    name=canonical_name,
                    type="memory_region",
                    description=first_c.attributes.get("description", f"Memory domain {canonical_name}"),
                    attributes={"base_address": b_addr, "size": sz},
                    source_references=[src_loc]
                )
                entity_reg[(ctype, cname)] = ent
                atomic_reqs.append(AtomicRequirement(
                    requirement_id=new_req_id("req"),
                    type="MEMORY",
                    statement=f"Memory region '{canonical_name}' base_address={b_addr}, size={sz}",
                    entities=[canonical_name],
                    subject=canonical_name,
                    action="ALLOCATE",
                    source=src_loc,
                    confidence=0.95
                ))

            elif ctype == "data_structure":
                ent = DiscoveredEntity(
                    entity_id=new_req_id("ent"),
                    name=canonical_name,
                    type="data_structure",
                    description=first_c.attributes.get("description", f"Data structure {canonical_name}"),
                    attributes=first_c.attributes,
                    source_references=[src_loc]
                )
                entity_reg[(ctype, cname)] = ent

            elif ctype == "performance_requirement":
                metric = first_c.attributes.get("metric", "PERFORMANCE")
                val = first_c.attributes.get("target_value", "")
                unit = first_c.attributes.get("unit", "")
                ent = DiscoveredEntity(
                    entity_id=new_req_id("ent"),
                    name=canonical_name,
                    type="performance_requirement",
                    description=f"Performance target {metric} = {val} {unit}",
                    attributes={"metric": metric, "target_value": val, "unit": unit},
                    source_references=[src_loc]
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

            elif ctype == "register":
                r_off = first_c.attributes.get("offset", "0x0")
                r_acc = first_c.attributes.get("access", "RW")
                ent = DiscoveredEntity(
                    entity_id=new_req_id("ent"),
                    name=canonical_name,
                    type="register",
                    description=first_c.attributes.get("description", f"Register {canonical_name}"),
                    attributes={"offset": r_off, "access": r_acc},
                    source_references=[src_loc]
                )
                entity_reg[(ctype, cname)] = ent

            elif ctype == "functional_requirement":
                stmt = first_c.attributes.get("statement", "")
                atomic_reqs.append(AtomicRequirement(
                    requirement_id=new_req_id("req"),
                    type="BEHAVIOR",
                    statement=stmt,
                    entities=[],
                    subject=canonical_name,
                    action="EXECUTE",
                    source=src_loc,
                    confidence=0.85
                ))

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
