import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from core.parsing.docx_parser import DocxParser
from core.requirement_ir.models import DocumentIR, DocumentBlock, SourceLocation
from core.extraction.source_evidence import SourceEvidence, SourceEvidenceIndex

COMMUNICATION_RELATIONS = {
    "connects_to",
    "interfaces_with",
    "communicates_with",
    "interconnect",
    "bus_connection",
    "bridges_to",
}

FLOW_RELATIONS = {
    "controls",
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
}

SUPPORTED_RELATIONSHIP_VERBS = [
    "connects_to",
    "controls",
    "reads_from",
    "writes_to",
    "sends_to",
    "receives_from",
    "contains",
    "drives",
    "feeds",
    "schedules",
    "distributes_to",
    "communicates_with",
    "interfaces_with",
    "depends_on",
]

_VERB_PATTERNS = [
    (re.compile(r"\b(?:connect(?:s|ed|ing)?\s+to|connected\s+with|interconnect(?:s|ing)?)\b", re.I), "connects_to"),
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
    (re.compile(r"\b(?:depends?\s+on|relies?\s+on|requires?)\b", re.I), "depends_on"),
]

_ABSTRACT_ACTION_PREFIXES = (
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
    "for and",
    "for ",
    "of ",
    "with ",
    "in ",
    "about ",
)

_ABSTRACT_KEYWORDS = {
    "parameters", "matrix engine parameters", "engine parameters",
    "configuration parameters", "subblocks", "matrix subblocks",
    "load balancing", "multi-cycle operations", "task distribution",
    "workload distribution", "data flow", "control flow",
    "assembly instructions", "assembly code", "computation",
    "calculations", "algorithm", "arbitration", "sequencing",
    "throughput", "latency", "bandwidth", "overall operation and task",
    "execution for and multi-cycle", "processing of matrix subblocks",
    "overall operation", "task", "tasks", "operation", "operations"
}


def is_abstract_or_action_phrase(phrase: str) -> bool:
    if not phrase:
        return True
    p_lower = phrase.strip().strip(",.;:").lower()
    
    for prefix in _ABSTRACT_ACTION_PREFIXES:
        if p_lower.startswith(prefix):
            return True
            
    if p_lower in _ABSTRACT_KEYWORDS:
        return True
        
    words = [w for w in p_lower.split() if w not in ("and", "or", "the", "a", "an", "of", "for", "in", "to", "across")]
    if words and all(w in ("execution", "processing", "operation", "operations", "parameters", "task", "tasks", "distribution", "multi-cycle", "subblocks", "cycles", "flow", "management", "control") for w in words):
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
        # Scan blocks for components and definitions
        for b in getattr(doc_ir, "blocks", []):
            text = b.text or ""
            sec_lower = (b.section_title or "").lower()
            is_comp_sec = any(k in sec_lower for k in ("component", "architecture", "block", "elements"))
            
            for line in text.split("\n"):
                clean = line.strip()
                colon_m = re.match(r"^[-*]?\s*([A-Za-z0-9_\s\(\)]{2,50})\s*:\s*(.+)$", clean)
                if colon_m:
                    head = colon_m.group(1).strip()
                    tail = colon_m.group(2).strip()
                    # If line defines a component, e.g. "Matrix Processing Elements (MPEs) : 4-16 individual compute cores"
                    if is_comp_sec or any(kw in head.lower() for kw in ("scheduler", "network", "engine", "processor", "unit", "elements", "alu")):
                        if not self.resolve(head):
                            self.register_entity(head, "component")
                    
                    resolved = self.resolve(head)
                    if resolved:
                        canon, etype = resolved
                        sub_m = re.search(r"\b(\d+[-\s\w]*\s+)?(compute cores|processing cores|cores|matrix cores|execution units)\b", tail, re.I)
                        if sub_m:
                            core_term = sub_m.group(2).lower()
                            self.entity_map[core_term] = (canon, etype)
                            self.entity_map["cores"] = (canon, etype)
                            self.entity_map["core"] = (canon, etype)
                            self.entity_map["available cores"] = (canon, etype)

    def resolve(self, phrase: str) -> Optional[Tuple[str, str]]:
        if not phrase:
            return None
        clean = phrase.strip().strip(",.;:()[]{}").lower()
        if clean in self.entity_map:
            return self.entity_map[clean]
            
        stripped = re.sub(r"^(?:the|a|an|all|available|individual|shared|complete)\s+", "", clean).strip()
        if stripped in self.entity_map:
            return self.entity_map[stripped]
            
        # Try to find a known entity inside the phrase
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
    evidence_type: str = "explicit"
    confidence: float = 1.0
    status: str = "accepted"  # accepted | rejected | unresolved
    category: str = "hardware_relationship"  # component_description | register_description | hardware_relationship | interface | ambiguous_relationship
    rejection_reason: Optional[str] = None
    destination_entity_type: Optional[str] = None
    resolved_source: Optional[str] = None
    resolved_target: Optional[str] = None


def extract_all(doc_ir, known_comps=None, known_regs=None, known_mems=None):
    resolver = CanonicalEntityResolver(known_comps, known_regs, known_mems, doc_ir=doc_ir)
    accepted = []
    rejected = []
    unresolved = []
    seen = set()
    doc_name = doc_ir.metadata.get("filename", "specification")

    for b in doc_ir.blocks:
        lines = (b.text or "").split("\n")
        for l_idx, line in enumerate(lines, 1):
            clean_l = line.strip()
            if not clean_l or len(clean_l) < 5:
                continue

            # 1. Colon items "<Source> : <Desc>"
            colon_m = re.match(r"^[-*]?\s*([A-Za-z0-9_\s\(\)]{2,50})\s*:\s*(.+)$", clean_l)
            if colon_m:
                src_cand = colon_m.group(1).strip()
                desc = colon_m.group(2).strip()

                src_res = resolver.resolve(src_cand)
                is_register = (src_res and src_res[1] == "register") or "register" in src_cand.lower() or "csr" in src_cand.lower()

                # Look for verbs
                verb_found = False
                for pattern, rel in _VERB_PATTERNS:
                    m = pattern.search(desc)
                    if m:
                        verb_found = True
                        after_verb = desc[m.end():].strip().split(".")[0].split(";")[0]
                        # Check target entity
                        target_res = resolver.resolve(after_verb)
                        
                        ev = SourceEvidence(
                            document=doc_name,
                            section=b.section_title or "General",
                            text=clean_l,
                            line_start=l_idx,
                            line_end=l_idx,
                            block_id=b.block_id,
                            evidence_type="structural"
                        )

                        if target_res and src_res:
                            # Both entities resolve!
                            canon_src, src_type = src_res
                            canon_dst, dst_type = target_res
                            if canon_src != canon_dst and not is_abstract_or_action_phrase(canon_dst):
                                cat = "interface" if rel in COMMUNICATION_RELATIONS else "hardware_relationship"
                                trip = (canon_src, rel, canon_dst)
                                if trip not in seen:
                                    seen.add(trip)
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
                        else:
                            # Target is NOT a resolved hardware entity
                            # Qualify as component description or register description
                            cat = "register_description" if is_register else "component_description"
                            rej_reason = "rejected_as_register_description" if is_register else "rejected_as_component_description"
                            words = after_verb.split()
                            target_phrase = " ".join(words[:4]).strip().rstrip(",.")
                            trip = (src_cand, rel, target_phrase)
                            if trip not in seen:
                                seen.add(trip)
                                rejected.append(ExtractedRelationship(
                                    source=src_cand,
                                    relation=rel,
                                    target=target_phrase,
                                    evidence=ev,
                                    evidence_type="structural",
                                    confidence=0.95,
                                    status="rejected",
                                    category=cat,
                                    rejection_reason=rej_reason,
                                    resolved_source=src_res[0] if src_res else src_cand
                                ))
                        break

            # 2. Interface connection pattern: "<iface> connects <src> to <dst>"
            conn_m = re.search(r"\b([A-Za-z0-9_\s]+?)\s+(?:connects|interfaces)\s+([A-Za-z0-9_\s]+?)\s+(?:to|with)\s+([A-Za-z0-9_\s]+)\b", clean_l, re.I)
            if conn_m:
                iface_cand = conn_m.group(1).strip()
                s_cand = conn_m.group(2).strip()
                d_cand = conn_m.group(3).strip()
                s_res = resolver.resolve(s_cand)
                d_res = resolver.resolve(d_cand)
                if s_res and d_res:
                    canon_s, s_t = s_res
                    canon_d, d_t = d_res
                    trip = (canon_s, "connects_to", canon_d)
                    if trip not in seen:
                        seen.add(trip)
                        ev = SourceEvidence(document=doc_name, section=b.section_title or "General", text=clean_l, line_start=l_idx, line_end=l_idx, block_id=b.block_id, evidence_type="text")
                        accepted.append(ExtractedRelationship(
                            source=canon_s, relation="connects_to", target=canon_d, evidence=ev,
                            status="accepted", category="interface", resolved_source=canon_s, resolved_target=canon_d, destination_entity_type=d_t
                        ))

            # 3. General prose scanning
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
                            trip = (canon_s, rel, canon_d)
                            if trip not in seen:
                                seen.add(trip)
                                ev = SourceEvidence(document=doc_name, section=b.section_title or "General", text=clean_l, line_start=l_idx, line_end=l_idx, block_id=b.block_id, evidence_type="text")
                                accepted.append(ExtractedRelationship(
                                    source=canon_s, relation=rel, target=canon_d, evidence=ev,
                                    status="accepted", category=cat, resolved_source=canon_s, resolved_target=canon_d, destination_entity_type=d_t
                                ))

    return accepted, rejected, unresolved


# Test with matrix_compute_unit.docx
parser = DocxParser("data/uploads/matrix_compute_unit.docx")
parsed_doc = parser.parse()
doc_ir = DocumentIR(
    document_id=parsed_doc.doc_id, document_version="1.0", source_type=parsed_doc.file_type,
    document_hash="h", ingestion_id="i", metadata={"filename": parsed_doc.filename},
    sections=[s.to_dict() for s in parsed_doc.sections],
    blocks=[DocumentBlock(block_id=f"BLK_{i+1:03d}", type=s.content_type, section_id=s.section_id, section_title=s.title, page=s.page_start, text=s.text, source_location=SourceLocation(document_id=parsed_doc.doc_id, document_hash="h", section=s.title, section_id=s.section_id, page=s.page_start, block_id=f"BLK_{i+1:03d}", original_text=s.text[:200])) for i, s in enumerate(parsed_doc.sections) if s.text and s.text.strip()]
)

known_comps = ["Register File", "Matrix ALU", "Control Unit", "Memory Interface", "Local Register File", "Local Tiling Engine", "L1 Matrix Cache", "DMA Engine"]
known_regs = ["Configuration Registers", "Status Registers", "Matrix Registers", "M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "Matrix Registers M0-M7"]

acc, rej, unres = extract_all(doc_ir, known_comps, known_regs)

print(f"\nACCEPTED RELATIONSHIPS ({len(acc)}):")
for r in acc:
    print(f"  [{r.category}] {r.source} --[{r.relation}]--> {r.target} (dest_type: {r.destination_entity_type})")
    print(f"    evidence: {r.evidence.text}")

print(f"\nREJECTED RELATIONSHIPS ({len(rej)}):")
for r in rej:
    print(f"  [{r.category}] {r.source} -> target candidate: {r.target!r} (reason: {r.rejection_reason})")
    print(f"    evidence: {r.evidence.text}")

print(f"\nUNRESOLVED RELATIONSHIPS ({len(unres)}):")
for r in unres:
    print(f"  {r.source} -> {r.target} (reason: {r.rejection_reason})")
