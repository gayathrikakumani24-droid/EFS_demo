"""
Dedicated Extraction Modules for Hardware Specifications.

Implements protocol-agnostic, evidence-backed extractors for:
1. Relationships (connects_to, controls, reads_from, writes_to, sends_to, receives_from, contains, drives, feeds, schedules, distributes_to, communicates_with, interfaces_with, depends_on)
2. Interfaces (name, protocol, direction, signals, widths, purpose, connected components)
3. Registers (distinguishes explicit, category-without-address, inferred, and none; NEVER invents 0x00/0x04)
4. FSMs (distinguishes explicit, partial, inferred, and unresolved; never invents fake ACK states)
5. Constraints (clock frequency, clock period, reset behavior, throughput, latency, power, operating ranges)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from core.extraction.source_evidence import EvidenceBackedProperty, SourceEvidence, SourceEvidenceIndex
from core.requirement_ir.models import DocumentBlock, DocumentIR
from core.efs_ir.validation_rules import (
    is_fsm_state_name,
    is_fsm_transition_or_condition,
    is_physical_signal_name,
    is_hardware_component_candidate,
    is_valid_interface_candidate,
    is_valid_register_candidate,
)
from utils.logger import get_logger

logger = get_logger("extraction.dedicated")


# ==============================================================================
# 1. DEDICATED RELATIONSHIP EXTRACTION & QUALIFICATION
# ==============================================================================

COMMUNICATION_RELATIONS: Set[str] = {
    "connects_to",
    "interfaces_with",
    "communicates_with",
    "interconnect",
    "bus_connection",
    "bridges_to",
    "sends_to",
    "returns_to",
    "reads_writes",
    "transfers_with",
    "transfers_to",
    "transfers_data_with",
    "supports_infrastructure",
    "uses_infrastructure",
}

FLOW_RELATIONS: Set[str] = {
    "controls",
    "orchestrates",
    "schedules",
    "distributes",
    "distributes_to",
    "depends_on",
    "reads_from",
    "writes_to",
    "produces",
    "consumes",
    "drives",
    "feeds",
    "sends_to",
    "receives_from",
    "contains",
    "transfers_with",
    "transfers_to",
    "supports_infrastructure",
    "uses_infrastructure",
}

SUPPORTED_RELATIONSHIP_VERBS: List[str] = [
    "connects_to",
    "controls",
    "orchestrates",
    "reads_from",
    "writes_to",
    "sends_to",
    "returns_to",
    "receives_from",
    "contains",
    "drives",
    "feeds",
    "schedules",
    "distributes_to",
    "communicates_with",
    "interfaces_with",
    "reads_writes",
    "depends_on",
    "transfers_with",
    "transfers_to",
    "supports_infrastructure",
    "uses_infrastructure",
]

def extract_signals_from_phrase(phrase: str) -> List[str]:
    """
    Parses signal names from phrases like 'mem_addr, mem_wdata, and mem_we' or 'data_in and valid'.
    Filters out English conjunctions/articles and ensures clean signal identifier strings.
    """
    if not phrase:
        return []
    clean = re.sub(r"\band\b", ",", phrase, flags=re.IGNORECASE)
    parts = clean.split(",")
    signals = []
    for p in parts:
        s = p.strip().strip(".()[]{};")
        s = re.sub(r"^(?:the|a|an)\s+", "", s, flags=re.IGNORECASE).strip()
        if s and re.match(r"^[A-Za-z_][A-Za-z0-9_]*(\[[^\]]+\])?$", s):
            signals.append(s)
    return signals

_VERB_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:connect(?:s|ed|ing)?\s+to|connected\s+with|interconnect(?:s|ing)?|connecting)\b", re.I), "connects_to"),
    (re.compile(r"\b(?:returns?\s+to)\b", re.I), "returns_to"),
    (re.compile(r"\b(?:reads/writes|reads\s+(?:from\s+)?and\s+writes\s+to)\b", re.I), "reads_writes"),
    (re.compile(r"\b(?:control(?:s|ling)?|orchestrates?|manages?|governs?)\b", re.I), "controls"),
    (re.compile(r"\b(?:reads?\s+from|fetches?\s+from|loads?\s+from)\b", re.I), "reads_from"),
    (re.compile(r"\b(?:writes?\s+to|stores?\s+to|updates?)\b", re.I), "writes_to"),
    (re.compile(r"\b(?:sends?\s+to|transmits?\s+to|dispatches?\s+to)\b", re.I), "sends_to"),
    (re.compile(r"\b(?:receives?\s+from|accepts?\s+from)\b", re.I), "receives_from"),
    (re.compile(r"\b(?:contains?|comprises?|houses?|includes?)\b", re.I), "contains"),
    (re.compile(r"\b(?:drives?|asserts?)\b", re.I), "drives"),
    (re.compile(r"\b(?:feeds?\s+into|feeds?|supplies?)\b", re.I), "feeds"),
    (re.compile(r"\b(?:schedules?|arbitrates?)\b", re.I), "schedules"),
    (re.compile(r"\b(?:distributes?(?:\s+(?:matrix\s+)?workloads?)?\s+(?:across|to)|distributes?|partitions?\s+(?:to|across)|routes?\s+to)\b", re.I), "distributes_to"),
    (re.compile(r"\b(?:communicates?\s+with|interacts?\s+with)\b", re.I), "communicates_with"),
    (re.compile(r"\b(?:interfaces?\s+with|bridges?\s+to)\b", re.I), "interfaces_with"),
    (re.compile(r"\borchestrates?\b", re.I), "orchestrates"),
    (re.compile(r"\b(?:autonomous\s+)?data\s+transfer\s+(?:with|to|from)\b|\btransfers?\s+(?:matrix\s+data\s+)?(?:with|to|from)\b|\btransfers?\b", re.I), "transfers_with"),
    (re.compile(r"\b(?:supports?|uses?)\s+(?:communication\s+)?infrastructure\b|\buses?\b|\bsupports?\b", re.I), "supports_infrastructure"),
    (re.compile(r"\b(?:depends?\s+on|relies?\s+on|requires?)\b", re.I), "depends_on"),
]

_ABSTRACT_ACTION_PREFIXES: Tuple[str, ...] = (
    "processing of",
    "execution for",
    "execution of",
    "overall operation",
    "operation of",
    "operations for",
    "handling of",
    "management of",
    "coordination of",
    "distribution of",
    "scheduling of",
    "performing",
    "calculating",
    "generating",
    "decoding",
    "dispatching",
    "controlling",
    "computation across",
    "computation of",
    "computation for",
    "computation",
    "for and",
    "for ",
    "of ",
    "with ",
    "in ",
    "about ",
)

_ABSTRACT_KEYWORDS: Set[str] = {
    "parameters", "matrix engine parameters", "engine parameters",
    "configuration parameters", "subblocks", "matrix subblocks",
    "load balancing", "multi-cycle operations", "task distribution",
    "workload distribution", "data flow", "control flow",
    "assembly instructions", "assembly code", "computation",
    "calculations", "algorithm", "arbitration", "sequencing",
    "throughput", "latency", "bandwidth", "overall operation and task",
    "execution for and multi-cycle", "processing of matrix subblocks",
    "overall operation", "task", "tasks", "operation", "operations",
    "computation across cores", "fixed-point, floating-point, etc",
    "(fixed-point, floating-point, etc",
}


def is_abstract_or_action_phrase(phrase: str) -> bool:
    """Check if a phrase describes an operation, action, parameter, or abstract concept rather than a hardware block."""
    if not phrase:
        return True
    p = phrase.strip().strip(",.;:()[]{}").lower()
    for prefix in _ABSTRACT_ACTION_PREFIXES:
        if p.startswith(prefix):
            return True
    if p in _ABSTRACT_KEYWORDS:
        return True
    words = [w for w in p.split() if w not in ("and", "or", "the", "a", "an", "of", "for", "in", "to", "across", "etc")]
    if words and all(
        w in (
            "execution", "processing", "operation", "operations", "parameters",
            "task", "tasks", "distribution", "multi-cycle", "subblocks", "cycles",
            "flow", "management", "control", "format", "data", "computation",
            "fixed-point", "floating-point"
        )
        for w in words
    ):
        return True
    return False


class CanonicalEntityResolver:
    """
    Canonical hardware entity resolver that maps variations, aliases,
    and plural/singular forms to canonical EFS entity names.
    """
    def __init__(
        self,
        known_components: Optional[List[str]] = None,
        known_registers: Optional[List[str]] = None,
        known_memories: Optional[List[str]] = None,
        known_interfaces: Optional[List[str]] = None,
        doc_ir: Optional[DocumentIR] = None
    ):
        self.entity_map: Dict[str, Tuple[str, str]] = {}
        self.canonical_names: Set[str] = set()

        for c in (known_components or []):
            self.register_entity(c, "component")
        for r in (known_registers or []):
            self.register_entity(r, "register")
        for m in (known_memories or []):
            self.register_entity(m, "memory_region")
        for i in (known_interfaces or []):
            self.register_entity(i, "interface")

        if doc_ir:
            self._mine_aliases_from_doc(doc_ir)

    def register_entity(self, name: str, entity_type: str):
        if not name or not name.strip():
            return
        canonical = name.strip()
        self.canonical_names.add(canonical)
        c_lower = canonical.lower()
        self.entity_map[c_lower] = (canonical, entity_type)

        paren_m = re.search(r"^(.*?)\s*\((.*?)\)$", canonical)
        if paren_m:
            outside = paren_m.group(1).strip()
            inside = paren_m.group(2).strip()
            self.entity_map[outside.lower()] = (canonical, entity_type)
            self.entity_map[inside.lower()] = (canonical, entity_type)
            if inside.lower().endswith("s"):
                self.entity_map[inside.lower()[:-1]] = (canonical, entity_type)
            if outside.lower().endswith("s"):
                self.entity_map[outside.lower()[:-1]] = (canonical, entity_type)

        if c_lower.endswith("s") and len(c_lower) > 3:
            self.entity_map[c_lower[:-1]] = (canonical, entity_type)
        elif not c_lower.endswith("s"):
            self.entity_map[c_lower + "s"] = (canonical, entity_type)

    def _mine_aliases_from_doc(self, doc_ir: DocumentIR):
        """Scan document blocks for architectural components and compute-core aliases."""
        for b in getattr(doc_ir, "blocks", []):
            text = b.text or ""
            sec_lower = (b.section_title or "").lower()
            if any(k in sec_lower for k in ("instruction", "opcode", "isa", "assembly", "register", "field", "timing", "performance")):
                continue
            is_comp_sec = any(k in sec_lower for k in ("component", "architecture", "block diagram", "structural", "subsystem"))

            for line in text.split("\n"):
                clean = line.strip()
                colon_m = re.match(r"^[-*]?\s*([A-Za-z0-9_\s\(\)]{2,50})\s*:\s*(.+)$", clean)
                if colon_m:
                    head = colon_m.group(1).strip()
                    tail = colon_m.group(2).strip()

                    # Avoid indexing generic headings/config lines as components
                    if any(bad in head.lower() for bad in ("configuration", "table", "note", "figure", "section", "mode", "parameter")):
                        continue

                    # Register component definitions
                    if is_comp_sec or any(kw in head.lower() for kw in ("scheduler", "network", "engine", "processor", "unit", "elements", "alu", "interface", "controller")):
                        if not self.resolve(head):
                            self.register_entity(head, "component")

                    if "main memory" in head.lower() or "memory interface" in head.lower():
                        self.entity_map["main memory"] = (head, "component")
                        self.entity_map["external memory"] = (head, "component")

                    resolved = self.resolve(head)
                    if resolved and any(k in head.lower() for k in ("element", "core", "mpe", "processor", "alu")):
                        canon, etype = resolved
                        # Only map 'cores' alias if the line defines the entity as cores (e.g. '4-16 individual compute cores')
                        sub_m = re.search(r"^\s*(?:\d+[-\s\w]*\s+)?(?:individual\s+)?(compute cores|processing cores|cores|matrix cores|execution units)\b", tail, re.I)
                        if sub_m and not any(v in tail.lower() for v in ("distribute", "connect", "communicat", "synchroniz")):
                            self.entity_map["compute cores"] = (canon, etype)
                            self.entity_map["processing cores"] = (canon, etype)
                            self.entity_map["cores"] = (canon, etype)
                            self.entity_map["core"] = (canon, etype)
                            self.entity_map["available cores"] = (canon, etype)
                            self.entity_map["matrix processing elements"] = (canon, etype)
                            self.entity_map["processing elements"] = (canon, etype)
                            self.entity_map["mpe"] = (canon, etype)
                            self.entity_map["mpes"] = (canon, etype)

    def resolve(self, phrase: str) -> Optional[Tuple[str, str]]:
        if not phrase:
            return None
        clean = phrase.strip().strip(",.;:()[]{}").lower()
        if clean in self.entity_map:
            return self.entity_map[clean]

        stripped = re.sub(r"^(?:the|a|an|all|available|individual|shared|complete)\s+", "", clean).strip()
        if stripped in self.entity_map:
            return self.entity_map[stripped]

        # Check for known entity substrings (longest match first)
        for alias, val in sorted(self.entity_map.items(), key=lambda x: len(x[0]), reverse=True):
            if len(alias) >= 4:
                if re.search(rf"\b{re.escape(alias)}\b", clean):
                    return val
        return None


@dataclass
class ExtractedRelationship:
    source: str
    relation: str
    target: str
    evidence: SourceEvidence
    evidence_type: str = "explicit"  # explicit | structural | inferred
    confidence: float = 1.0
    status: str = "accepted"  # accepted | rejected | unresolved
    category: str = "hardware_relationship"  # component_description | register_description | hardware_relationship | interface | ambiguous_relationship
    rejection_reason: Optional[str] = None
    destination_entity_type: Optional[str] = None
    resolved_source: Optional[str] = None
    resolved_target: Optional[str] = None
    signals: List[str] = field(default_factory=list)
    direction: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "evidence": self.evidence.to_dict(),
            "evidence_type": self.evidence_type,
            "confidence": self.confidence,
            "status": self.status,
            "category": self.category,
            "rejection_reason": self.rejection_reason,
            "destination_entity_type": self.destination_entity_type,
            "resolved_source": self.resolved_source,
            "resolved_target": self.resolved_target,
            "signals": list(self.signals),
            "direction": self.direction or f"{self.source} -> {self.target}",
        }


class DedicatedRelationshipExtractor:
    """Extracts hardware relationships from architecture text, tables, component definitions, and cross-references."""

    last_accepted: List[ExtractedRelationship] = []
    last_rejected: List[ExtractedRelationship] = []
    last_unresolved: List[ExtractedRelationship] = []

    @classmethod
    def extract_candidates(
        cls,
        doc_ir: DocumentIR,
        known_components: Optional[List[str]] = None,
        known_registers: Optional[List[str]] = None,
        known_memories: Optional[List[str]] = None,
        evidence_index: Optional[SourceEvidenceIndex] = None
    ) -> Tuple[List[ExtractedRelationship], List[ExtractedRelationship], List[ExtractedRelationship]]:
        """
        Extract and qualify all relationship candidates, categorizing them into:
        - accepted (both entities resolve to valid hardware objects, non-abstract)
        - rejected (component descriptions, register descriptions, or abstract actions)
        - unresolved (ambiguous or ungrounded targets)
        """
        resolver = CanonicalEntityResolver(
            known_components=known_components,
            known_registers=known_registers,
            known_memories=known_memories,
            doc_ir=doc_ir
        )
        accepted: List[ExtractedRelationship] = []
        rejected: List[ExtractedRelationship] = []
        unresolved: List[ExtractedRelationship] = []
        seen_triples: Set[Tuple[str, str, str]] = set()

        doc_name = doc_ir.metadata.get("filename", "specification")

        for b in doc_ir.blocks:
            sec_lower = (b.section_title or "").lower()
            lines = (b.text or "").split("\n")
            for l_idx, line in enumerate(lines, 1):
                clean_l = line.strip()
                if not clean_l or len(clean_l) < 5:
                    continue

                # 0. Skip explicit relationship section header lines
                if re.match(r"^[-*#]*\s*(?:inter[- ]module|module)\s+relationship[s]?\s*:?$", clean_l, re.I):
                    continue
                if re.match(r"^[-*#]*\s*(?:inter[- ]module|module)\s+connectivity\s*:?$", clean_l, re.I):
                    continue

                # 0b. Explicit Inter-Module Relationship phrases:
                # - "A sends X to B"
                # - "B returns X to A"
                # - "A connects to B [via X]"
                # - "A interfaces with B [via X]"
                # - "A communicates with B [via X]"
                # - "A reads/writes B [via X]"
                m_send = re.match(r"^[-*]?\s*([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)\s+sends\s+(.+?)\s+to\s+([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)[.;\s]*$", clean_l, re.I)
                m_ret = re.match(r"^[-*]?\s*([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)\s+returns\s+(.+?)\s+to\s+([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)[.;\s]*$", clean_l, re.I)
                m_conn = re.match(r"^[-*]?\s*([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)\s+(?:connects\s+to|connected\s+to)\s+([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)(?:\s+(?:via|using|with)\s+(.+))?[.;\s]*$", clean_l, re.I)
                m_iface = re.match(r"^[-*]?\s*([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)\s+interfaces\s+with\s+([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)(?:\s+(?:via|using|with)\s+(.+))?[.;\s]*$", clean_l, re.I)
                m_comm = re.match(r"^[-*]?\s*([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)\s+communicates\s+with\s+([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)(?:\s+(?:via|using|with)\s+(.+))?[.;\s]*$", clean_l, re.I)
                m_rw = re.match(r"^[-*]?\s*([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)\s+(?:reads/writes\s+(?:to\s+)?|reads\s+(?:from\s+)?and\s+writes\s+to\s+|reads\s+from\s+|writes\s+to\s+)([A-Za-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)(?:\s+(?:via|using|with)\s+(.+))?[.;\s]*$", clean_l, re.I)

                rel_match_item = None
                if m_send:
                    rel_match_item = (m_send.group(1).strip(), "sends_to", m_send.group(3).strip(), extract_signals_from_phrase(m_send.group(2).strip()), f"{m_send.group(1).strip()} -> {m_send.group(3).strip()}")
                elif m_ret:
                    rel_match_item = (m_ret.group(1).strip(), "returns_to", m_ret.group(3).strip(), extract_signals_from_phrase(m_ret.group(2).strip()), f"{m_ret.group(1).strip()} -> {m_ret.group(3).strip()}")
                elif m_conn:
                    sigs = extract_signals_from_phrase(m_conn.group(3).strip()) if m_conn.group(3) else []
                    rel_match_item = (m_conn.group(1).strip(), "connects_to", m_conn.group(2).strip(), sigs, f"{m_conn.group(1).strip()} -> {m_conn.group(2).strip()}")
                elif m_iface:
                    sigs = extract_signals_from_phrase(m_iface.group(3).strip()) if m_iface.group(3) else []
                    rel_match_item = (m_iface.group(1).strip(), "interfaces_with", m_iface.group(2).strip(), sigs, f"{m_iface.group(1).strip()} -> {m_iface.group(2).strip()}")
                elif m_comm:
                    sigs = extract_signals_from_phrase(m_comm.group(3).strip()) if m_comm.group(3) else []
                    rel_match_item = (m_comm.group(1).strip(), "communicates_with", m_comm.group(2).strip(), sigs, f"{m_comm.group(1).strip()} -> {m_comm.group(2).strip()}")
                elif m_rw:
                    sigs = extract_signals_from_phrase(m_rw.group(3).strip()) if m_rw.group(3) else []
                    rel_match_item = (m_rw.group(1).strip(), "reads_writes", m_rw.group(2).strip(), sigs, f"{m_rw.group(1).strip()} -> {m_rw.group(2).strip()}")

                if rel_match_item:
                    s_name, r_verb, d_name, r_sigs, r_dir = rel_match_item
                    if not is_abstract_or_action_phrase(s_name) and not is_abstract_or_action_phrase(d_name) and s_name.lower() != d_name.lower():
                        # Hard Semantic Boundary:
                        # 1. If either endpoint is an FSM state, this is an FSM transition, NOT an interface!
                        if is_fsm_state_name(s_name) or is_fsm_state_name(d_name):
                            logger.debug(f"[REL_FILTER] Skipping FSM transition as interface: {s_name} -> {d_name}")
                            continue

                        # 2. If either endpoint is a signal, this is a signal-component association, NOT an interface!
                        if is_physical_signal_name(s_name) or is_physical_signal_name(d_name):
                            logger.debug(f"[REL_FILTER] Skipping signal association as interface: {s_name} -> {d_name}")
                            continue

                        # 3. Check hard semantic boundary for interface candidates
                        valid_if, rej_reason = is_valid_interface_candidate(s_name, d_name, known_components=resolver.canonical_names)
                        if not valid_if:
                            logger.debug(f"[REL_FILTER] Non-interface candidate {s_name} -> {d_name}: {rej_reason}")
                            continue

                        s_res = resolver.resolve(s_name)
                        d_res = resolver.resolve(d_name)
                        if not s_res:
                            resolver.register_entity(s_name, "component")
                            s_res = (s_name, "component")
                        if not d_res:
                            resolver.register_entity(d_name, "component")
                            d_res = (d_name, "component")

                        canon_s, s_t = s_res
                        canon_d, d_t = d_res
                        triple = (canon_s, r_verb, canon_d)
                        if triple not in seen_triples:
                            seen_triples.add(triple)
                            ev = SourceEvidence(
                                document=doc_name,
                                section=b.section_title or "Inter-Module Relationship",
                                text=clean_l,
                                line_start=l_idx,
                                line_end=l_idx,
                                block_id=b.block_id,
                                evidence_type="explicit"
                            )
                            accepted.append(ExtractedRelationship(
                                source=canon_s,
                                relation=r_verb,
                                target=canon_d,
                                evidence=ev,
                                evidence_type="explicit",
                                confidence=0.98,
                                status="accepted",
                                category="interface",
                                resolved_source=canon_s,
                                resolved_target=canon_d,
                                destination_entity_type=d_t,
                                signals=r_sigs,
                                direction=r_dir
                            ))
                        continue

                # 1. Structural item definitions e.g. "Control Processor : Orchestrates overall operation"
                # or "Global Scheduler : Distributes matrix workloads across available cores"
                colon_match = re.match(r"^[-*]?\s*([A-Za-z0-9_\s\(\)\-]{2,50})\s*:\s*(.+)$", clean_l)
                if colon_match:
                    src_candidate = colon_match.group(1).strip()
                    desc = colon_match.group(2).strip()

                    is_instr_sec = any(k in sec_lower for k in ("instruction", "opcode", "isa", "assembly"))
                    src_res = resolver.resolve(src_candidate)
                    is_register = (
                        (src_res and src_res[1] == "register")
                        or "register" in src_candidate.lower()
                        or "csr" in src_candidate.lower()
                    )

                    for pattern, rel in _VERB_PATTERNS:
                        m = pattern.search(desc)
                        if m:
                            after_verb = desc[m.end():].strip().split(".")[0].split(";")[0]
                            target_res = resolver.resolve(after_verb)

                            # Domain-specific multi-core context resolution backed by explicit section evidence
                            if not target_res:
                                if "task distribution" in desc.lower() and "control processor" in src_candidate.lower():
                                    target_res = resolver.resolve("cores")
                                    rel = "orchestrates"
                                elif "data transfer" in desc.lower() and "dma" in src_candidate.lower() and ("memory" in desc.lower() or "main memory" in desc.lower()):
                                    target_res = resolver.resolve("main memory")
                                    rel = "transfers_with"

                            ev = SourceEvidence(
                                document=doc_name,
                                section=b.section_title or "General",
                                text=clean_l,
                                line_start=l_idx,
                                line_end=l_idx,
                                block_id=b.block_id,
                                evidence_type="structural"
                            )

                            if target_res and src_res and not is_instr_sec:
                                canon_src, src_type = src_res
                                canon_dst, dst_type = target_res
                                if canon_src != canon_dst and not is_abstract_or_action_phrase(canon_dst):
                                    cat = "interface" if rel in COMMUNICATION_RELATIONS else "hardware_relationship"
                                    triple = (canon_src, rel, canon_dst)
                                    if triple not in seen_triples:
                                        seen_triples.add(triple)
                                        accepted.append(ExtractedRelationship(
                                            source=canon_src,
                                            relation=rel,
                                            target=canon_dst,
                                            evidence=ev,
                                            evidence_type="explicit",
                                            confidence=0.95,
                                            status="accepted",
                                            category=cat,
                                            resolved_source=canon_src,
                                            resolved_target=canon_dst,
                                            destination_entity_type=dst_type
                                        ))
                                    break
                            else:
                                # Target does NOT resolve to a validated hardware block or is an action/instruction description
                                cat = (
                                    "instruction_description" if is_instr_sec
                                    else ("register_description" if is_register else "component_description")
                                )
                                rej_reason = (
                                    "rejected_as_instruction_description" if is_instr_sec
                                    else ("rejected_as_register_description" if is_register else "rejected_as_component_description")
                                )
                                words = after_verb.split()
                                target_phrase = " ".join(words[:4]).strip().rstrip(",.")
                                triple = (src_candidate, rel, target_phrase)
                                if triple not in seen_triples:
                                    seen_triples.add(triple)
                                    rejected.append(ExtractedRelationship(
                                        source=src_candidate,
                                        relation=rel,
                                        target=target_phrase,
                                        evidence=ev,
                                        evidence_type="structural",
                                        confidence=0.95,
                                        status="rejected",
                                        category=cat,
                                        rejection_reason=rej_reason,
                                    ))
                                break

                    # Explicit Inter-Core Communication Patterns (Nearest-Neighbor, Broadcast, Reduction, All-to-All)
                    if any(k in sec_lower for k in ("communication pattern", "inter-core communication")):
                        pat_name = src_candidate.strip()
                        pat_lower = pat_name.lower()
                        if pat_lower in ("nearest-neighbor", "broadcast", "reduction", "all-to-all"):
                            cores_res = resolver.resolve("cores")
                            if cores_res:
                                canon_cores, c_type = cores_res
                                triple = (canon_cores, f"communicates_with_{pat_lower}", canon_cores)
                                if triple not in seen_triples:
                                    seen_triples.add(triple)
                                    ev = SourceEvidence(
                                        document=doc_name,
                                        section=b.section_title or "Inter-Core Communication",
                                        text=clean_l,
                                        line_start=l_idx,
                                        line_end=l_idx,
                                        block_id=b.block_id,
                                        evidence_type="explicit"
                                    )
                                    accepted.append(ExtractedRelationship(
                                        source=canon_cores,
                                        relation="communicates_with",
                                        target=canon_cores,
                                        evidence=ev,
                                        evidence_type="explicit",
                                        confidence=0.95,
                                        status="accepted",
                                        category="hardware_relationship",
                                        resolved_source=canon_cores,
                                        resolved_target=canon_cores,
                                        destination_entity_type=c_type,
                                        signals=[pat_lower.replace("-", "_")],
                                        direction=f"MPE <-> MPE ({pat_name})"
                                    ))
                            continue
                    continue

                # 2. Communication Infrastructure pattern (Interconnect Network uses/supports Crossbar, NoC, DMA)
                if any(k in sec_lower for k in ("communication infrastructure", "interconnect infrastructure")):
                    net_res = resolver.resolve("Interconnect Network")
                    if net_res and any(k in clean_l.lower() for k in ("crossbar", "noc", "dma")):
                        canon_net, net_t = net_res
                        infra_target = (
                            "Crossbar switch" if "crossbar" in clean_l.lower()
                            else ("Multi-level NoC" if "noc" in clean_l.lower() else "DMA-driven block transfers")
                        )
                        triple = (canon_net, "supports_infrastructure", infra_target)
                        if triple not in seen_triples:
                            seen_triples.add(triple)
                            ev = SourceEvidence(
                                document=doc_name,
                                section=b.section_title or "Communication Infrastructure",
                                text=clean_l,
                                line_start=l_idx,
                                line_end=l_idx,
                                block_id=b.block_id,
                                evidence_type="explicit"
                            )
                            accepted.append(ExtractedRelationship(
                                source=canon_net,
                                relation="supports_infrastructure",
                                target=infra_target,
                                evidence=ev,
                                evidence_type="explicit",
                                confidence=0.95,
                                status="accepted",
                                category="hardware_relationship",
                                resolved_source=canon_net,
                                resolved_target=infra_target,
                                destination_entity_type="component",
                                direction=f"{canon_net} -> {infra_target}"
                            ))
                        continue

                # 3. Interface connection pattern: "<iface> connects <src> to <dst>"
                conn_m = re.search(
                    r"\b([A-Za-z0-9_\s]+?)\s+(?:connects|interfaces)\s+([A-Za-z0-9_\s]+?)\s+(?:to|with)\s+([A-Za-z0-9_\s]+)\b",
                    clean_l,
                    re.I
                )
                if conn_m:
                    s_cand = conn_m.group(2).strip()
                    d_cand = conn_m.group(3).strip()
                    s_res = resolver.resolve(s_cand)
                    d_res = resolver.resolve(d_cand)
                    if s_res and d_res:
                        canon_s, s_t = s_res
                        canon_d, d_t = d_res
                        if canon_s != canon_d and not is_abstract_or_action_phrase(canon_d):
                            triple = (canon_s, "connects_to", canon_d)
                            if triple not in seen_triples:
                                seen_triples.add(triple)
                                ev = SourceEvidence(
                                    document=doc_name,
                                    section=b.section_title or "General",
                                    text=clean_l,
                                    line_start=l_idx,
                                    line_end=l_idx,
                                    block_id=b.block_id,
                                    evidence_type="text"
                                )
                                accepted.append(ExtractedRelationship(
                                    source=canon_s,
                                    relation="connects_to",
                                    target=canon_d,
                                    evidence=ev,
                                    evidence_type="explicit",
                                    confidence=0.95,
                                    status="accepted",
                                    category="interface",
                                    resolved_source=canon_s,
                                    resolved_target=canon_d,
                                    destination_entity_type=d_t
                                ))
                    continue

                # 3. General prose scanning for validated component-to-component relationships
                sec_lower = (b.section_title or "").lower()
                is_instr_sec = any(k in sec_lower for k in ("instruction", "opcode", "isa", "assembly"))
                if is_instr_sec:
                    continue

                for pattern, rel in _VERB_PATTERNS:
                    for m in pattern.finditer(clean_l):
                        before = clean_l[:m.start()].strip()
                        after = clean_l[m.end():].strip()

                        s_res = resolver.resolve(before)
                        d_res = resolver.resolve(after)

                        if s_res and d_res:
                            canon_s, s_t = s_res
                            canon_d, d_t = d_res
                            if canon_s != canon_d and not is_abstract_or_action_phrase(canon_d):
                                cat = "interface" if rel in COMMUNICATION_RELATIONS else "hardware_relationship"
                                triple = (canon_s, rel, canon_d)
                                if triple not in seen_triples:
                                    seen_triples.add(triple)
                                    ev = SourceEvidence(
                                        document=doc_name,
                                        section=b.section_title or "General",
                                        text=clean_l,
                                        line_start=l_idx,
                                        line_end=l_idx,
                                        block_id=b.block_id,
                                        evidence_type="text"
                                    )
                                    accepted.append(ExtractedRelationship(
                                        source=canon_s,
                                        relation=rel,
                                        target=canon_d,
                                        evidence=ev,
                                        evidence_type="explicit",
                                        confidence=0.9,
                                        status="accepted",
                                        category=cat,
                                        resolved_source=canon_s,
                                        resolved_target=canon_d,
                                        destination_entity_type=d_t
                                    ))

        cls.last_accepted = accepted
        cls.last_rejected = rejected
        cls.last_unresolved = unresolved

        logger.info(
            f"DedicatedRelationshipExtractor: candidates={len(accepted)+len(rejected)+len(unresolved)} "
            f"(accepted={len(accepted)}, rejected={len(rejected)}, unresolved={len(unresolved)})"
        )
        return accepted, rejected, unresolved

    @classmethod
    def extract_relationships(
        cls,
        doc_ir: DocumentIR,
        known_components: Optional[List[str]] = None,
        evidence_index: Optional[SourceEvidenceIndex] = None,
        include_rejected: bool = False
    ) -> List[ExtractedRelationship]:
        """Backward-compatible extraction method returning accepted relationships by default."""
        accepted, rejected, unresolved = cls.extract_candidates(
            doc_ir=doc_ir,
            known_components=known_components,
            evidence_index=evidence_index
        )
        if include_rejected:
            return accepted + rejected + unresolved
        return accepted


# ==============================================================================
# 2. DEDICATED INTERFACE EXTRACTION
# ==============================================================================

@dataclass
class ExtractedInterface:
    name: str
    protocol: str = "generic"
    direction: str = "bidirectional"
    signals: List[str] = field(default_factory=list)
    widths: Dict[str, str] = field(default_factory=dict)
    purpose: str = ""
    connected_components: List[str] = field(default_factory=list)
    source_evidence: SourceEvidence = field(default_factory=SourceEvidence)
    status: str = "explicit"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "protocol": self.protocol,
            "direction": self.direction,
            "signals": self.signals,
            "widths": self.widths,
            "purpose": self.purpose,
            "connected_components": self.connected_components,
            "source_evidence": self.source_evidence.to_dict(),
            "status": self.status
        }


class DedicatedInterfaceExtractor:
    """Extracts explicit interfaces, protocol buses, and connected ports from hardware specifications."""

    @classmethod
    def extract_interfaces(
        cls,
        doc_ir: DocumentIR,
        evidence_index: Optional[SourceEvidenceIndex] = None
    ) -> List[ExtractedInterface]:
        interfaces: List[ExtractedInterface] = []
        doc_name = doc_ir.metadata.get("filename", "specification")
        seen_names: Set[str] = set()

        for b in doc_ir.blocks:
            sec_lower = (b.section_title or "").lower()
            text = b.text or ""
            lines = text.split("\n")

            is_iface_sec = any(k in sec_lower for k in ("interface", "ports", "pinout", "bus", "external", "interconnect", "communication"))

            # 1. Look for interface definitions in lines or bullet points
            for l_idx, line in enumerate(lines, 1):
                clean_l = line.strip()
                if not clean_l or clean_l.startswith("|---"):
                    continue

                # Pattern A: "Clock : System clock input" or "Instruction Input : 64-bit wide..."
                colon_m = re.match(r"^[-*]?\s*([A-Za-z0-9_\s]{2,40})\s*:\s*(.+)$", clean_l)
                if is_iface_sec and colon_m:
                    if_name = colon_m.group(1).strip()
                    desc = colon_m.group(2).strip()
                    name_lower = if_name.lower()

                    if (
                        name_lower not in ("table", "note", "figure", "example", "section")
                        and "register" not in name_lower
                        and "csr" not in name_lower
                    ):
                        # Determine protocol or role
                        proto = "generic"
                        if "axi" in desc.lower() or "axi" in name_lower:
                            proto = "AXI"
                        elif "apb" in desc.lower() or "apb" in name_lower:
                            proto = "APB"
                        elif "sram" in desc.lower() or "memory" in desc.lower() or "cache" in desc.lower():
                            proto = "Memory"
                        elif "clock" in name_lower or "reset" in name_lower:
                            proto = "ClockReset"

                        direction = "bidirectional"
                        if "input" in desc.lower() or "in" in desc.lower().split():
                            direction = "input"
                        elif "output" in desc.lower() or "out" in desc.lower().split():
                            direction = "output"

                        # Extract width if present
                        width_m = re.search(r"(\d+)[-\s]*bits?", desc, re.I)
                        widths = {}
                        if width_m:
                            widths[if_name] = width_m.group(1)

                        if if_name not in seen_names:
                            seen_names.add(if_name)
                            ev = SourceEvidence(
                                document=doc_name,
                                section=b.section_title or "General",
                                text=clean_l,
                                line_start=l_idx,
                                line_end=l_idx,
                                block_id=b.block_id,
                                evidence_type="text"
                            )
                            interfaces.append(ExtractedInterface(
                                name=if_name,
                                protocol=proto,
                                direction=direction,
                                signals=[if_name],
                                widths=widths,
                                purpose=desc,
                                connected_components=[],
                                source_evidence=ev,
                                status="explicit"
                            ))

            # 2. Extract Verilog module interface declarations if present
            if "module " in text and "(" in text:
                mod_m = re.search(r"module\s+([A-Za-z0-9_]+)\s*\((.*?)\);", text, re.DOTALL)
                if mod_m:
                    mod_name = mod_m.group(1).strip()
                    port_text = mod_m.group(2)
                    ports = []
                    port_widths = {}
                    for p_line in port_text.split(","):
                        p_line_clean = p_line.strip()
                        p_match = re.search(r"(input|output|inout)\s+(?:wire|logic|reg)?\s*(?:\[(\d+:\d+)\])?\s*([A-Za-z0-9_]+)", p_line_clean)
                        if p_match:
                            p_dir = p_match.group(1)
                            p_w = p_match.group(2) or "1"
                            p_name = p_match.group(3)
                            ports.append(p_name)
                            port_widths[p_name] = p_w

                    if ports and f"{mod_name}_Interface" not in seen_names:
                        seen_names.add(f"{mod_name}_Interface")
                        ev = SourceEvidence(
                            document=doc_name,
                            section=b.section_title or "RTL Module Definition",
                            text=text[:200],
                            line_start=1,
                            line_end=len(lines),
                            block_id=b.block_id,
                            evidence_type="code_block"
                        )
                        interfaces.append(ExtractedInterface(
                            name=f"{mod_name}_Interface",
                            protocol="generic_ports",
                            direction="bidirectional",
                            signals=ports,
                            widths=port_widths,
                            purpose=f"Hardware port interface for {mod_name}",
                            connected_components=[mod_name],
                            source_evidence=ev,
                            status="explicit"
                        ))

        logger.info(f"DedicatedInterfaceExtractor: Extracted {len(interfaces)} explicit interfaces.")
        return interfaces


# ==============================================================================
# 3. DEDICATED REGISTER EXTRACTION
# ==============================================================================

@dataclass
class ExtractedRegister:
    name: str
    address: Optional[str] = None
    offset: Optional[str] = None
    width: Optional[int] = 32
    access: str = "RW"
    reset_value: Optional[str] = None
    description: str = ""
    status: str = "explicit"  # explicit | partial | inferred | unresolved
    evidence: SourceEvidence = field(default_factory=SourceEvidence)
    fields: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "address": self.address,
            "offset": self.offset,
            "width": self.width,
            "access": self.access,
            "reset_value": self.reset_value,
            "description": self.description,
            "status": self.status,
            "evidence": self.evidence.to_dict(),
            "fields": [
                f if isinstance(f, dict) else (f.to_dict() if hasattr(f, 'to_dict') else str(f))
                for f in self.fields
            ]
        }


class DedicatedRegisterExtractor:
    """
    Extracts registers while strictly distinguishing:
    A. Explicit register definition (with explicit offset/address)
    B. Register category exists but address/details are missing (address=None, status="partial" or "explicit")
    C. Register inferred from instruction/operand context
    D. No register evidence
    CRITICAL RULE: NEVER invents standard addresses like 0x00, 0x04.
    """

    @classmethod
    def extract_registers(
        cls,
        doc_ir: DocumentIR,
        evidence_index: Optional[SourceEvidenceIndex] = None
    ) -> List[ExtractedRegister]:
        registers: List[ExtractedRegister] = []
        doc_name = doc_ir.metadata.get("filename", "specification")
        seen_names: Set[str] = set()

        for b in doc_ir.blocks:
            sec_lower = (b.section_title or "").lower()
            text = b.text or ""
            lines = text.split("\n")

            is_reg_sec = any(k in sec_lower for k in ("register", "memory map", "csr", "reg layout", "address space"))

            for l_idx, line in enumerate(lines, 1):
                clean_l = line.strip()
                if not clean_l or clean_l.startswith("|---"):
                    continue

                # Pattern A: Table row with explicit hex address or offset e.g. "| CTRL_REG | 0x00 | RW | 0x00000000 | ..."
                if "|" in clean_l:
                    cells = [c.strip() for c in clean_l.split("|") if c.strip()]
                    if len(cells) >= 2:
                        r_name = cells[0]
                        second = cells[1]
                        if r_name.lower() not in ("register", "name", "reg name", "reg_name", "offset", "address"):
                            # Check if second cell is hex offset
                            has_hex = second.lower().startswith("0x") or second.isdigit()
                            offset_val = second if has_hex else None
                            access_val = cells[2] if len(cells) > 2 else "RW"
                            reset_val = cells[3] if len(cells) > 3 else None
                            desc_val = cells[4] if len(cells) > 4 else ""

                            if (has_hex or is_reg_sec) and r_name not in seen_names:
                                seen_names.add(r_name)
                                status = "explicit" if has_hex else "partial"
                                ev = SourceEvidence(
                                    document=doc_name,
                                    section=b.section_title or "Registers",
                                    text=clean_l,
                                    line_start=l_idx,
                                    line_end=l_idx,
                                    block_id=b.block_id,
                                    evidence_type="table"
                                )
                                registers.append(ExtractedRegister(
                                    name=r_name,
                                    offset=offset_val,
                                    address=offset_val,
                                    width=32,
                                    access=access_val,
                                    reset_value=reset_val,
                                    description=desc_val,
                                    status=status,
                                    evidence=ev
                                ))
                                continue

                # Pattern B: Register category with explicit names but without hex addresses
                # e.g., "Matrix Registers: M0-M7 (8 registers in total)"
                # or "Configuration Registers"
                range_match = re.search(r"([A-Za-z0-9_]+)\s*Registers?\s*:\s*([A-Za-z0-9_]+)\s*-\s*([A-Za-z0-9_]+)", clean_l, re.I)
                if range_match:
                    cat_name = range_match.group(1).strip()
                    start_name = range_match.group(2).strip()
                    end_name = range_match.group(3).strip()

                    # e.g. M0 to M7
                    m_start = re.match(r"^([A-Za-z]+)(\d+)$", start_name)
                    m_end = re.match(r"^([A-Za-z]+)(\d+)$", end_name)
                    if m_start and m_end and m_start.group(1).upper() == m_end.group(1).upper():
                        prefix = m_start.group(1).upper()
                        s_idx = int(m_start.group(2))
                        e_idx = int(m_end.group(2))
                        for r_num in range(s_idx, e_idx + 1):
                            reg_id_name = f"{prefix}{r_num}"
                            if reg_id_name not in seen_names:
                                seen_names.add(reg_id_name)
                                ev = SourceEvidence(
                                    document=doc_name,
                                    section=b.section_title or "Registers",
                                    text=clean_l,
                                    line_start=l_idx,
                                    line_end=l_idx,
                                    block_id=b.block_id,
                                    evidence_type="text"
                                )
                                registers.append(ExtractedRegister(
                                    name=reg_id_name,
                                    address=None,  # DO NOT INVENT STANDARD ADDRESSES
                                    offset=None,
                                    width=64 if "64" in clean_l else 32,
                                    access="RW",
                                    description=f"{cat_name} Register {reg_id_name}",
                                    status="explicit",  # Explicitly named in source, address genuinely unassigned
                                    evidence=ev
                                ))
                        cat_range_name = f"{cat_name} Registers {start_name}-{end_name}"
                        if cat_range_name not in seen_names:
                            seen_names.add(cat_range_name)
                            ev_group = SourceEvidence(
                                document=doc_name,
                                section=b.section_title or "Registers",
                                text=clean_l,
                                line_start=l_idx,
                                line_end=l_idx,
                                block_id=b.block_id,
                                evidence_type="text"
                            )
                            registers.append(ExtractedRegister(
                                name=cat_range_name,
                                address=None,
                                offset=None,
                                width=64 if "64" in clean_l else 32,
                                access="RW",
                                description=clean_l,
                                status="explicit",
                                evidence=ev_group
                            ))
                        continue

                # Category-only definition without address: e.g. "Configuration Registers" or "Status Register File"
                cat_match = re.match(r"^[-*]?\s*([A-Za-z0-9_\s]{2,30}\s*Registers?)(?:\s*:\s*(.*))?$", clean_l, re.I)
                if (is_reg_sec or "register" in clean_l.lower()) and cat_match:
                    cat_name = cat_match.group(1).strip()
                    desc_text = cat_match.group(2) or clean_l
                    if cat_name not in seen_names and cat_name.lower() not in ("register addresses", "registers", "external interfaces"):
                        seen_names.add(cat_name)
                        ev = SourceEvidence(
                            document=doc_name,
                            section=b.section_title or "Registers",
                            text=clean_l,
                            line_start=l_idx,
                            line_end=l_idx,
                            block_id=b.block_id,
                            evidence_type="text"
                        )
                        registers.append(ExtractedRegister(
                            name=cat_name,
                            address=None,
                            offset=None,
                            width=32,
                            access="RW",
                            description=desc_text,
                            status="partial",  # Register category present, offset unassigned in source
                            evidence=ev
                        ))

            # Pattern C: Prose register definitions within the block
            # e.g., "The hardware exposes configuration registers CTRL and STATUS."
            # e.g., "The module contains a CTRL configuration register and STATUS register."
            if "register" in text.lower() or "csr" in text.lower():
                # 1. Multiple named registers in phrase
                prose_decl_matches = re.finditer(
                    r"\b(?:exposes|contains|defines|includes|provides|has)\s+(?:(?:configuration|status|control|hardware)\s+)?registers?\s+([A-Za-z0-9_,\s]+?)(?:\.|\n|$)",
                    text,
                    re.I
                )
                for pdm in prose_decl_matches:
                    raw_list = pdm.group(1)
                    tokens = [t.strip() for t in re.split(r",|\band\b", raw_list) if t.strip()]
                    for tok in tokens:
                        if is_valid_register_candidate(tok) and tok not in seen_names and tok.lower() not in ("register", "registers", "the", "a", "an", "and", "or"):
                            seen_names.add(tok)
                            fields = []
                            for fm in re.finditer(rf"\b{re.escape(tok)}\s+contains\s+(?:an?|the)?\s*([A-Za-z0-9_]+)\s+field\b", text, re.I):
                                fname = fm.group(1).strip()
                                if fname.lower() not in ("register", "registers", "field", "fields", "an", "the", "a"):
                                    fields.append({"name": fname, "width": 1, "msb": 0, "lsb": 0, "access": "RW", "description": f"{fname} field in {tok}"})
                            for fm2 in re.finditer(rf"\b([A-Za-z0-9_]+)\s+field\s+in\s+{re.escape(tok)}\b", text, re.I):
                                fname = fm2.group(1).strip()
                                if fname.lower() not in ("register", "registers", "field", "fields") and not any(f["name"] == fname for f in fields):
                                    fields.append({"name": fname, "width": 1, "msb": 0, "lsb": 0, "access": "RW", "description": f"{fname} field in {tok}"})
                            access = "RW"
                            if re.search(rf"\b{re.escape(tok)}\s+is\s+written\s+by\s+software\b", text, re.I):
                                access = "WO"
                            elif re.search(rf"\b{re.escape(tok)}\s+is\s+read\s+by\s+software\b", text, re.I):
                                access = "RO"
                            elif re.search(rf"\b{re.escape(tok)}\b.*?\b(?:read[- ]only|ro)\b", text, re.I):
                                access = "RO"
                            elif re.search(rf"\b{re.escape(tok)}\b.*?\b(?:write[- ]only|wo)\b", text, re.I):
                                access = "WO"
                            for fld in fields:
                                fld["access"] = access
                            ev = SourceEvidence(
                                document=doc_name,
                                section=b.section_title or "Registers",
                                text=pdm.group(0).strip(),
                                line_start=1,
                                line_end=1,
                                block_id=b.block_id,
                                evidence_type="text"
                            )
                            registers.append(ExtractedRegister(
                                name=tok,
                                address=None,
                                offset=None,
                                width=32,
                                access=access,
                                description=f"Configuration/status register {tok}",
                                status="explicit",
                                evidence=ev,
                                fields=fields
                            ))

                # 2. Individual named register declaration (e.g. "CTRL configuration register" or "STATUS register")
                single_decl_matches = re.finditer(
                    r"\b([A-Za-z0-9_]+)\s+(?:configuration\s+register|status\s+register|control\s+register)\b",
                    text,
                    re.I
                )
                for sdm in single_decl_matches:
                    tok = sdm.group(1).strip()
                    if is_valid_register_candidate(tok) and tok not in seen_names and tok.lower() not in ("the", "a", "an", "this", "each", "every", "shift"):
                        seen_names.add(tok)
                        fields = []
                        for fm in re.finditer(rf"\b{re.escape(tok)}\s+contains\s+(?:an?|the)?\s*([A-Za-z0-9_]+)\s+field\b", text, re.I):
                            fname = fm.group(1).strip()
                            if fname.lower() not in ("register", "registers", "field", "fields", "an", "the", "a"):
                                fields.append({"name": fname, "width": 1, "msb": 0, "lsb": 0, "access": "RW", "description": f"{fname} field in {tok}"})
                        for fm2 in re.finditer(rf"\b([A-Za-z0-9_]+)\s+field\s+in\s+{re.escape(tok)}\b", text, re.I):
                            fname = fm2.group(1).strip()
                            if fname.lower() not in ("register", "registers", "field", "fields") and not any(f["name"] == fname for f in fields):
                                fields.append({"name": fname, "width": 1, "msb": 0, "lsb": 0, "access": "RW", "description": f"{fname} field in {tok}"})
                        access = "RW"
                        if "status" in sdm.group(0).lower():
                            access = "RO"
                        if re.search(rf"\b{re.escape(tok)}\s+is\s+written\s+by\s+software\b", text, re.I):
                            access = "WO"
                        elif re.search(rf"\b{re.escape(tok)}\s+is\s+read\s+by\s+software\b", text, re.I):
                            access = "RO"
                        elif re.search(rf"\b{re.escape(tok)}\b.*?\b(?:read[- ]only|ro)\b", text, re.I):
                            access = "RO"
                        elif re.search(rf"\b{re.escape(tok)}\b.*?\b(?:write[- ]only|wo)\b", text, re.I):
                            access = "WO"
                        for fld in fields:
                            fld["access"] = access
                        ev = SourceEvidence(
                            document=doc_name,
                            section=b.section_title or "Registers",
                            text=sdm.group(0).strip(),
                            line_start=1,
                            line_end=1,
                            block_id=b.block_id,
                            evidence_type="text"
                        )
                        registers.append(ExtractedRegister(
                            name=tok,
                            address=None,
                            offset=None,
                            width=32,
                            access=access,
                            description=f"Configuration/status register {tok}",
                            status="explicit",
                            evidence=ev,
                            fields=fields
                        ))

        logger.info(f"DedicatedRegisterExtractor: Extracted {len(registers)} registers (explicit/partial).")
        return registers


# ==============================================================================
# 4. DEDICATED FSM EXTRACTION
# ==============================================================================

@dataclass
class ExtractedTransition:
    source_state: str
    target_state: str
    condition: str = ""
    associated_signal: Optional[str] = None
    associated_register: Optional[str] = None
    condition_type: str = "explicit"  # boolean | signal_assertion | semantic | unresolved
    action: str = ""
    evidence: str = ""
    traceability: Optional[SourceEvidence] = None

    @property
    def from_state(self) -> str:
        return self.source_state

    @property
    def to_state(self) -> str:
        return self.target_state

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from": self.source_state,
            "to": self.target_state,
            "source_state": self.source_state,
            "target_state": self.target_state,
            "condition": self.condition,
            "associated_signal": self.associated_signal,
            "associated_register": self.associated_register,
            "condition_type": self.condition_type,
            "action": self.action,
            "evidence": self.evidence,
            "provenance": self.traceability.to_dict() if self.traceability else {}
        }


@dataclass
class ExtractedFSM:
    name: str
    states: List[str] = field(default_factory=list)
    initial_state: str = ""
    terminal_states: List[str] = field(default_factory=list)
    reset_state: Optional[str] = None
    reset_behavior: str = ""
    transitions: List[ExtractedTransition] = field(default_factory=list)
    control_signals: List[str] = field(default_factory=list)
    status: str = "explicit"  # explicit | partial | inferred | unresolved
    evidence: SourceEvidence = field(default_factory=SourceEvidence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "states": self.states,
            "initial_state": self.initial_state,
            "terminal_states": self.terminal_states,
            "reset_state": self.reset_state,
            "reset_behavior": self.reset_behavior,
            "transitions": [t.to_dict() for t in self.transitions],
            "control_signals": self.control_signals,
            "status": self.status,
            "evidence": self.evidence.to_dict()
        }


class DedicatedFSMExtractor:
    """
    Extracts FSM architecture strictly distinguishing:
    - explicit FSM: states and transitions clearly defined in source
    - partial FSM: control/handshake signals exist (start, busy, done, ack) but transition graph is informal
    - inferred FSM: transitions cleanly derived from control signals
    - unresolved FSM: no state or control evidence
    CRITICAL RULES:
    - Never invents an ACK transition if ACK is absent from source.
    - Never hallucinates Boolean expressions (e.g. done == 1) unless explicitly stated in source.
    - If source states 'operation completed' without a Boolean condition, preserves transition as semantic.
    - If source states 'done asserted', represents condition using the canonical signal.
    """

    @classmethod
    def extract_fsm(
        cls,
        doc_ir: DocumentIR,
        known_signals: Optional[List[str]] = None,
        evidence_index: Optional[SourceEvidenceIndex] = None,
        signal_names: Optional[List[str]] = None
    ) -> Optional[ExtractedFSM]:
        known_signals = known_signals or signal_names
        doc_name = doc_ir.metadata.get("filename", "specification")
        explicit_states: List[str] = []
        explicit_transitions: List[ExtractedTransition] = []
        control_signals: List[str] = []
        fsm_evidence: Optional[SourceEvidence] = None

        full_doc_text = "\n".join(b.text for b in doc_ir.blocks if b.text)
        full_doc_lower = full_doc_text.lower()

        # Known signal map for canonical resolution (case-insensitive)
        sig_map = {s.lower(): s for s in (known_signals or [])}

        # Check reset behavior from document
        reset_behavior = ""
        reset_match = re.search(r"\b(active[- ](?:low|high)\s+reset|asynchronous\s+reset|synchronous\s+reset|reset\s*[:=]\s*[^\n\r]+)", full_doc_text, re.IGNORECASE)
        if reset_match:
            reset_behavior = reset_match.group(0).strip()

        # 1. Scan for explicit state machines, state lists, and transitions
        for b in doc_ir.blocks:
            sec_lower = (b.section_title or "").lower()
            text = b.text or ""
            text_lower = text.lower()
            lines = text.split("\n")

            # Look for explicit states list, e.g. "States: IDLE, EXECUTE, DONE" or "operates across IDLE, RUN, and FINISH states"
            st_list_match = re.search(r"(?:(?:fsm\s+)?states\s*:\s*|state\s+machine\s+(?:operates\s+across|has\s+states)\s+)([A-Za-z0-9_,\s]+(?:\s+and\s+[A-Za-z0-9_]+)?)(?:\s+states)?", text, re.IGNORECASE)
            if st_list_match:
                clean_clause = st_list_match.group(1).replace(" and ", ", ")
                raw_states = [s.strip().upper() for s in clean_clause.split(",") if s.strip()]
                for st in raw_states:
                    if st not in explicit_states and len(st) <= 24 and (st.isalnum() or "_" in st):
                        explicit_states.append(st)
                if not fsm_evidence:
                    fsm_evidence = SourceEvidence(
                        document=doc_name,
                        section=b.section_title or "State Machine",
                        text=st_list_match.group(0),
                        block_id=b.block_id,
                        evidence_type="text"
                    )

            # Explicit bulleted states, e.g. "- IDLE: Waiting for start signal"
            if any(k in sec_lower for k in ("state", "fsm", "transition", "control flow")):
                for line in lines:
                    m = re.match(r"^[-*]?\s*\b([A-Z0-9_]{2,20})\b\s*:\s*(.*)", line.strip())
                    if m:
                        candidate_st = m.group(1).upper()
                        if candidate_st not in ("THE", "FOR", "AND", "NOT", "NOTE", "RULE", "SECTION", "TABLE", "CLK", "RESET"):
                            if candidate_st not in explicit_states:
                                explicit_states.append(candidate_st)

            # Transition patterns:
            # Pattern A: "IDLE -> EXECUTE: when start asserted" or "IDLE -> EXECUTE on start == 1"
            trans_matches = re.findall(r"\b([A-Za-z0-9_]{2,20})\b\s*(?:->|-->|=>)\s*\b([A-Za-z0-9_]{2,20})\b(?:\s*(?:[:]|when|on|if|after)\s*([^\n\r,;]+))?", text, re.IGNORECASE)
            # Pattern B: "transition from IDLE to EXECUTE when start asserted" or "From IDLE, it advances to RUN when start is asserted"
            trans_matches_b = re.findall(r"(?:transition(?:s)?\s+from|(?:from\s+))\b([A-Za-z0-9_]{2,20})\b\s*,?\s*(?:it\s+)?(?:transition(?:s)?\s+to|advances?\s+to|moves?\s+to|resets?\s+to)\s+\b([A-Za-z0-9_]{2,20})\b(?:\s*(?:when|on|if|after|:)\s*([^\n\r,;]+))?", text, re.IGNORECASE)

            combined_matches = list(trans_matches) + list(trans_matches_b)

            for src_st, dst_st, cond in combined_matches:
                s_clean = src_st.strip().upper()
                d_clean = dst_st.strip().upper()
                c_clean = (cond or "").strip()

                # Filter false positives
                if s_clean in ("THE", "AND", "FOR", "SECTION", "TABLE") or d_clean in ("THE", "AND", "FOR", "SECTION", "TABLE"):
                    continue

                if s_clean not in explicit_states:
                    explicit_states.append(s_clean)
                if d_clean not in explicit_states:
                    explicit_states.append(d_clean)

                # Classify condition type without hallucinating
                cond_type = "explicit"
                assoc_sig = None
                if not c_clean:
                    cond_type = "unresolved"
                elif any(op in c_clean for op in ("==", "!=", ">=", "<=", "1'b1", "1'b0", "&&", "||")):
                    cond_type = "boolean"
                    # Find referenced signal
                    for w in re.findall(r"\b([a-zA-Z0-9_]+)\b", c_clean):
                        if w.lower() in sig_map:
                            assoc_sig = sig_map[w.lower()]
                            break
                elif "assert" in c_clean.lower():
                    cond_type = "signal_assertion"
                    for w in re.findall(r"\b([a-zA-Z0-9_]+)\b", c_clean):
                        if w.lower() in sig_map:
                            assoc_sig = sig_map[w.lower()]
                            break
                elif any(sem_term in c_clean.lower() for sem_term in ("completed", "complete", "finished", "ready", "done")):
                    # Check if 'done' is an actual signal in the specification
                    if "done" in sig_map and "done" in c_clean.lower():
                        assoc_sig = sig_map["done"]
                        cond_type = "signal_assertion"
                    else:
                        cond_type = "semantic"
                else:
                    cond_type = "semantic"

                tr_ev = SourceEvidence(
                    document=doc_name,
                    section=b.section_title or "State Machine",
                    text=f"{s_clean} -> {d_clean}: {c_clean}".strip(),
                    block_id=b.block_id,
                    evidence_type="text"
                )

                # Avoid duplicate transitions
                if not any(t.source_state == s_clean and t.target_state == d_clean for t in explicit_transitions):
                    explicit_transitions.append(ExtractedTransition(
                        source_state=s_clean,
                        target_state=d_clean,
                        condition=c_clean,
                        associated_signal=assoc_sig,
                        condition_type=cond_type,
                        evidence=tr_ev.text,
                        traceability=tr_ev
                    ))

                if not fsm_evidence:
                    fsm_evidence = tr_ev

        # 2. Check for control signals across document/signals list
        for word in re.findall(r"\b([a-zA-Z0-9_]+)\b", full_doc_lower):
            if word in ("start", "start_op", "busy", "done", "ack", "acknowledge", "valid", "ready", "enable", "trigger"):
                canon_word = sig_map.get(word, word)
                if canon_word not in control_signals:
                    control_signals.append(canon_word)

        # 3. Classify and return FSM architecture
        init_st = "IDLE" if "IDLE" in explicit_states else (explicit_states[0] if explicit_states else "IDLE")
        term_states = [s for s in explicit_states if s in ("DONE", "COMPLETE", "FINISH", "ERROR")]

        if explicit_states and explicit_transitions:
            return ExtractedFSM(
                name="Primary_Controller_FSM",
                states=explicit_states,
                initial_state=init_st,
                terminal_states=term_states or (["DONE"] if "DONE" in explicit_states else []),
                reset_state=init_st,
                reset_behavior=reset_behavior or f"Asynchronous reset to {init_st}",
                transitions=explicit_transitions,
                control_signals=control_signals,
                status="explicit",
                evidence=fsm_evidence or SourceEvidence(document=doc_name, section="FSM", text="Explicit FSM transitions")
            )
        elif explicit_states:
            # Explicit states defined, derive transitions based on control signals if available
            inferred_tr: List[ExtractedTransition] = []
            if "IDLE" in explicit_states and "EXECUTE" in explicit_states:
                start_sig = next((s for s in control_signals if "start" in s.lower()), None)
                cond_text = f"{start_sig} asserted" if start_sig else "start condition"
                inferred_tr.append(ExtractedTransition(
                    source_state="IDLE",
                    target_state="EXECUTE",
                    condition=cond_text,
                    associated_signal=start_sig,
                    condition_type="signal_assertion" if start_sig else "semantic",
                    evidence=f"IDLE to EXECUTE via {cond_text}"
                ))
            if "EXECUTE" in explicit_states and "DONE" in explicit_states:
                done_sig = next((s for s in control_signals if "done" in s.lower()), None)
                cond_text = f"{done_sig} asserted" if done_sig else "operation completed"
                inferred_tr.append(ExtractedTransition(
                    source_state="EXECUTE",
                    target_state="DONE",
                    condition=cond_text,
                    associated_signal=done_sig,
                    condition_type="signal_assertion" if done_sig else "semantic",
                    evidence=f"EXECUTE to DONE via {cond_text}"
                ))
            if "DONE" in explicit_states and "IDLE" in explicit_states:
                ack_sig = next((s for s in control_signals if "ack" in s.lower()), None)
                done_sig = next((s for s in control_signals if "done" in s.lower()), None)
                if ack_sig:
                    inferred_tr.append(ExtractedTransition(
                        source_state="DONE",
                        target_state="IDLE",
                        condition=f"{ack_sig} asserted",
                        associated_signal=ack_sig,
                        condition_type="signal_assertion",
                        evidence="DONE to IDLE via ack"
                    ))
                elif done_sig:
                    inferred_tr.append(ExtractedTransition(
                        source_state="DONE",
                        target_state="IDLE",
                        condition=f"{done_sig} deasserted",
                        associated_signal=done_sig,
                        condition_type="signal_assertion",
                        evidence="DONE to IDLE via done deasserted"
                    ))
                else:
                    inferred_tr.append(ExtractedTransition(
                        source_state="DONE",
                        target_state="IDLE",
                        condition="auto-clear",
                        condition_type="semantic",
                        evidence="DONE to IDLE via auto-clear"
                    ))

            return ExtractedFSM(
                name="Primary_Controller_FSM",
                states=explicit_states,
                initial_state=init_st,
                terminal_states=term_states or (["DONE"] if "DONE" in explicit_states else []),
                reset_state=init_st,
                reset_behavior=reset_behavior or f"Asynchronous reset to {init_st}",
                transitions=inferred_tr,
                control_signals=control_signals,
                status="partial" if inferred_tr else "unresolved",
                evidence=fsm_evidence or SourceEvidence(document=doc_name, section="FSM", text="Explicit states defined")
            )
        elif any(k in [s.lower() for s in control_signals] for k in ("start", "start_op", "busy", "done")):
            # Inferred FSM cleanly grounded in documented control signals
            derived_states = ["IDLE", "EXECUTE", "DONE"]
            start_sig = next((s for s in control_signals if "start" in s.lower()), "start")
            done_sig = next((s for s in control_signals if "done" in s.lower()), "done")
            ack_sig = next((s for s in control_signals if "ack" in s.lower()), None)

            # Check if source text explicitly says "done asserted" vs "operation completed"
            if "done asserted" in full_doc_lower:
                done_cond = f"{done_sig} asserted"
                done_cond_type = "signal_assertion"
            elif "done" in [s.lower() for s in control_signals]:
                done_cond = f"{done_sig} asserted"
                done_cond_type = "signal_assertion"
            else:
                done_cond = "operation completed"
                done_cond_type = "semantic"

            inferred_tr = [
                ExtractedTransition(
                    source_state="IDLE",
                    target_state="EXECUTE",
                    condition=f"{start_sig} asserted",
                    associated_signal=start_sig,
                    condition_type="signal_assertion",
                    evidence=f"IDLE to EXECUTE when {start_sig} asserted"
                ),
                ExtractedTransition(
                    source_state="EXECUTE",
                    target_state="DONE",
                    condition=done_cond,
                    associated_signal=done_sig if done_cond_type == "signal_assertion" else None,
                    condition_type=done_cond_type,
                    evidence=f"EXECUTE to DONE when {done_cond}"
                )
            ]

            # ONLY add ACK transition if ACK is explicitly mentioned in source
            if ack_sig:
                inferred_tr.append(ExtractedTransition(
                    source_state="DONE",
                    target_state="IDLE",
                    condition=f"{ack_sig} asserted",
                    associated_signal=ack_sig,
                    condition_type="signal_assertion",
                    evidence=f"DONE to IDLE when {ack_sig} asserted"
                ))
            else:
                inferred_tr.append(ExtractedTransition(
                    source_state="DONE",
                    target_state="IDLE",
                    condition=f"{done_sig} deasserted",
                    associated_signal=done_sig,
                    condition_type="signal_assertion",
                    evidence=f"DONE to IDLE when {done_sig} deasserted"
                ))

            return ExtractedFSM(
                name="Inferred_Controller_FSM",
                states=derived_states,
                initial_state="IDLE",
                terminal_states=["DONE"],
                reset_state="IDLE",
                reset_behavior=reset_behavior or "Asynchronous reset to IDLE",
                transitions=inferred_tr,
                control_signals=control_signals,
                status="inferred",
                evidence=SourceEvidence(
                    document=doc_name,
                    section="Execution Control",
                    text=f"Control signals: {', '.join(control_signals)}"
                )
            )

        return None



# ==============================================================================
# 5. DEDICATED CONSTRAINT EXTRACTION
# ==============================================================================

@dataclass
class ExtractedConstraint:
    category: str  # frequency | clock_period | throughput | latency | power | area | operating_range
    value: str
    normalized_value: Optional[float] = None
    unit: str = ""
    original_text: str = ""
    status: str = "explicit"
    evidence: SourceEvidence = field(default_factory=SourceEvidence)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "unit": self.unit,
            "original_text": self.original_text,
            "status": self.status,
            "evidence": self.evidence.to_dict()
        }


class DedicatedConstraintExtractor:
    """Extracts explicit hardware constraints, timing goals, and power/frequency bounds without guessing defaults."""

    @classmethod
    def extract_constraints(
        cls,
        doc_ir: DocumentIR,
        evidence_index: Optional[SourceEvidenceIndex] = None
    ) -> List[ExtractedConstraint]:
        constraints: List[ExtractedConstraint] = []
        doc_name = doc_ir.metadata.get("filename", "specification")
        seen_keys: Set[str] = set()

        for b in doc_ir.blocks:
            lines = (b.text or "").split("\n")
            for l_idx, line in enumerate(lines, 1):
                clean_l = line.strip()
                if not clean_l:
                    continue

                # 1. Clock Frequency: e.g. "Maximum clock frequency: 1Ghz" or "100 MHz"
                freq_m = re.search(r"(?:clock\s+frequency|max(?:imum)?\s+frequency|frequency)\s*:\s*(\d+(?:\.\d+)?)\s*(ghz|mhz|khz|hz)\b", clean_l, re.I)
                if freq_m:
                    val_str_raw = freq_m.group(1)
                    val_num = float(val_str_raw)
                    unit_str = freq_m.group(2).upper()
                    # Normalize to MHz
                    norm_mhz = val_num * 1000.0 if unit_str == "GHZ" else (val_num if unit_str == "MHZ" else val_num / 1000.0)
                    key = f"freq_{val_str_raw}_{unit_str}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        ev = SourceEvidence(
                            document=doc_name,
                            section=b.section_title or "Performance",
                            text=clean_l,
                            line_start=l_idx,
                            line_end=l_idx,
                            block_id=b.block_id,
                            evidence_type="text"
                        )
                        constraints.append(ExtractedConstraint(
                            category="frequency",
                            value=f"{val_str_raw} {unit_str}",
                            normalized_value=norm_mhz,
                            unit="MHz",
                            original_text=clean_l,
                            status="explicit",
                            evidence=ev
                        ))

                # 2. Latency / Execution Time: e.g. "4 cycles latency per matrix operation"
                lat_m = re.search(r"(\d+)\s*cycles?\s+(?:latency|delay|execution)", clean_l, re.I)
                if lat_m:
                    cyc_val = int(lat_m.group(1))
                    key = f"latency_{cyc_val}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        ev = SourceEvidence(
                            document=doc_name,
                            section=b.section_title or "Performance",
                            text=clean_l,
                            line_start=l_idx,
                            line_end=l_idx,
                            block_id=b.block_id,
                            evidence_type="text"
                        )
                        constraints.append(ExtractedConstraint(
                            category="latency",
                            value=f"{cyc_val} cycles",
                            normalized_value=float(cyc_val),
                            unit="cycles",
                            original_text=clean_l,
                            status="explicit",
                            evidence=ev
                        ))

                # 3. Throughput: e.g. "throughput: 6.25 million element/sec"
                tp_m = re.search(r"throughput\s*:\s*(.+?)(?=[;,\.]|\Z)", clean_l, re.I)
                if tp_m:
                    tp_str = tp_m.group(1).strip()
                    key = f"throughput_{tp_str[:15]}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        ev = SourceEvidence(
                            document=doc_name,
                            section=b.section_title or "Performance",
                            text=clean_l,
                            line_start=l_idx,
                            line_end=l_idx,
                            block_id=b.block_id,
                            evidence_type="text"
                        )
                        constraints.append(ExtractedConstraint(
                            category="throughput",
                            value=tp_str,
                            original_text=clean_l,
                            status="explicit",
                            evidence=ev
                        ))

                # 4. Power: e.g. "Target power: 15W" or "power consumption: Less than 100mW"
                pwr_m = re.search(r"(?:power\s+consumption|power|dissipation)\s*:\s*(?:less\s+than\s+|under\s+|<\s*|at\s+most\s+)?(\d+(?:\.\d+)?)\s*(w|mw|watts?)\b", clean_l, re.I)
                if pwr_m:
                    pwr_val = float(pwr_m.group(1))
                    pwr_unit = pwr_m.group(2).upper()
                    raw_val = clean_l.split(":")[-1].strip()
                    key = f"power_{pwr_val}_{pwr_unit}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        ev = SourceEvidence(
                            document=doc_name,
                            section=b.section_title or "Power",
                            text=clean_l,
                            line_start=l_idx,
                            line_end=l_idx,
                            block_id=b.block_id,
                            evidence_type="text"
                        )
                        constraints.append(ExtractedConstraint(
                            category="power",
                            value=raw_val or f"{pwr_val} {pwr_unit}",
                            normalized_value=pwr_val if pwr_unit in ("W", "WATTS") else pwr_val / 1000.0,
                            unit="W",
                            original_text=clean_l,
                            status="explicit",
                            evidence=ev
                        ))

        logger.info(f"DedicatedConstraintExtractor: Extracted {len(constraints)} explicit constraints.")
        return constraints
