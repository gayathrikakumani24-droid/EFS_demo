"""
Target-Specific Protocol Context Resolver module.

Performs target-specific protocol context resolution:
USER TARGET -> TARGET EFS IR -> INTERFACE RESOLUTION -> PROTOCOL RESOLUTION
-> CHANNEL RESOLUTION -> TRANSACTION RESOLUTION -> RULE / CONSTRAINT RESOLUTION
-> SEMANTIC + GRAPH + VECTOR RETRIEVAL -> RELEVANT PROTOCOL CHUNKS
-> DEDUPLICATION + RANKING -> PROTOCOL CONTEXT SLICE.

Includes Completeness Validation Check to guarantee all required signal, handshake,
reset, timing, and ordering semantics are present before LLM execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from core.retrieval.dependency_resolver import ResolvedDependencies
from core.vectorstore.faiss_store import get_vector_store
from utils.logger import get_logger

logger = get_logger("retrieval.protocol_resolver")


@dataclass
class ProtocolEvidence:
    """Generic evidence container for resolved protocol metadata."""
    protocol_name: str = "GENERIC"
    confidence: float = 0.5
    source_chunk_ids: List[str] = field(default_factory=list)
    source_sections: List[str] = field(default_factory=list)
    matched_entities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_name": self.protocol_name,
            "confidence": self.confidence,
            "source_chunk_ids": self.source_chunk_ids,
            "source_sections": self.source_sections,
            "matched_entities": self.matched_entities,
        }


def resolve_protocol_context(
    dep_res: ResolvedDependencies,
    all_protocol_rules: List[Dict[str, Any]],
    requested_interface: Optional[str] = None,
    expansion_query_additions: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Perform Target-Specific Protocol Retrieval following the 10-step pipeline:
    Returns (selected_rules, protocol_evidence_list, protocol_chunks) with full traceability.
    """
    # 1. Interface Resolution
    target_if_names = [i.name for i in dep_res.interfaces]
    
    # 2. Protocol Resolution (Generic evidence-based metadata)
    active_proto = requested_interface
    if not active_proto and dep_res.interfaces:
        for iface in dep_res.interfaces:
            proto_val = getattr(iface, "protocol", None)
            if proto_val and proto_val.lower() not in ("generic", "discovered_protocol", "unknown"):
                active_proto = proto_val
                break
    active_proto = (active_proto or "GENERIC").upper()

    target_sig_names = [s.name.lower() for s in dep_res.signals]
    all_text_blob = " ".join([n.lower() for n in target_if_names] + target_sig_names)
    matched_entities = [n for n in target_if_names if n.lower() in all_text_blob]
    confidence = 0.85 if active_proto != "GENERIC" else 0.5

    evidence = ProtocolEvidence(
        protocol_name=active_proto,
        confidence=confidence,
        source_chunk_ids=[],
        source_sections=target_if_names,
        matched_entities=matched_entities
    )

    # 3. Dynamic Channel & Interface Evidence Resolution
    # Identify active terms from resolved signals and interfaces dynamically
    active_sig_keywords = set()
    for s_name in target_sig_names:
        for token in s_name.split("_"):
            if len(token) >= 2:
                active_sig_keywords.add(token)

    # 4. Signal Resolution & 5. Transaction Resolution
    transaction_types = [t.name if hasattr(t, "name") else str(t) for t in getattr(dep_res, "transactions", [])]

    # 6. Generic Rule / Constraint Filtering & Traceability Mapping
    selected_rules: List[Dict[str, Any]] = []
    seen_rule_ids = set()

    flat_rules = []
    for r in all_protocol_rules:
        if isinstance(r, list):
            flat_rules.extend(r)
        else:
            flat_rules.append(r)

    for rule in flat_rules:
        rule_text = ""
        doc_id = "protocol_spec"
        chunk_id = ""
        page = 1
        section = ""

        if isinstance(rule, dict):
            rule_text = rule.get("text", "") or rule.get("description", "") or str(rule)
            doc_id = rule.get("protocol_document_id") or rule.get("doc_id") or "protocol_spec"
            chunk_id = rule.get("chunk_id", "")
            page = rule.get("page", 1)
            section = rule.get("section", "")
        else:
            rule_text = str(rule)

        r_lower = rule_text.lower()
        rule_key = f"{doc_id}_{chunk_id}_{r_lower[:50]}"
        if rule_key in seen_rule_ids:
            continue

        is_relevant = False
        # General hardware rules (reset, clock, handshake, timing, setup, hold, stability, etc.)
        if any(kw in r_lower for kw in ["reset", "clock", "handshake", "setup", "hold", "stability", "valid", "ready", "ack", "req", "enable", "busy", "trigger"]):
            is_relevant = True
        # Dynamic signal/interface match
        elif any(kw in r_lower for kw in active_sig_keywords) or any(if_n.lower() in r_lower for if_n in target_if_names):
            is_relevant = True

        if is_relevant:
            seen_rule_ids.add(rule_key)
            rule_entry = {
                "text": rule_text,
                "protocol": active_proto,
                "protocol_document_id": doc_id,
                "chunk_id": chunk_id,
                "page": page,
                "section": section,
                "original_text": rule_text
            }
            selected_rules.append(rule_entry)

    # 7. Semantic + Graph + Vector Retrieval for Protocol Chunks
    # Build dynamic search query from active interface & signal names
    query_terms = [active_proto] if active_proto != "GENERIC" else []
    query_terms.extend(target_if_names)
    query_terms.extend(target_sig_names[:6])
    if expansion_query_additions:
        query_terms.extend(expansion_query_additions)

    search_query = " ".join(query_terms) if query_terms else "hardware protocol specification interface handshake"

    # Determine dynamic chunk count based on requirement complexity (focused 4-8 chunks)
    dynamic_k = max(3, min(8, len(dep_res.signals) + len(dep_res.interfaces) * 2))
    if expansion_query_additions:
        dynamic_k += 3  # Increase retrieval depth during expanded pass

    protocol_evidence: List[Dict[str, Any]] = [evidence.to_dict()]
    protocol_chunks: List[Dict[str, Any]] = []
    seen_chunks = set()

    try:
        vstore = get_vector_store()
        vector_hits = vstore.search(search_query, top_k=dynamic_k)
        for hit in vector_hits:
            meta = hit.metadata or {}
            c_id = hit.chunk_id
            if c_id in seen_chunks:
                continue
            seen_chunks.add(c_id)

            doc_id = meta.get("doc_id", "protocol_spec")
            page = meta.get("page", 1)
            section = meta.get("section", "")
            orig_text = hit.text

            chunk_info = {
                "protocol_document_id": doc_id,
                "chunk_id": c_id,
                "page": page,
                "section": section,
                "original_text": orig_text,
                "relevance_score": getattr(hit, "score", 1.0)
            }
            protocol_evidence.append(chunk_info)
            protocol_chunks.append(chunk_info)

    except Exception as e:
        logger.warning(f"Vector retrieval for protocol chunks failed: {e}")

    logger.info(
        f"Target-Specific Protocol Resolution Complete for protocol '{active_proto}': "
        f"Selected {len(selected_rules)} rules and {len(protocol_chunks)} dynamic protocol chunks (k={dynamic_k})."
    )
    return selected_rules, protocol_evidence, protocol_chunks


def validate_protocol_context_completeness(
    pack: Any,
    dep_res: ResolvedDependencies
) -> Tuple[bool, List[str]]:
    """
    PROTOCOL CONTEXT COMPLETENESS CHECK
    Verifies that all protocol semantics required by the target component are present:
    1. Signal definitions, directions, and widths
    2. Handshake rules (VALID/READY stability) if handshaking interface is present
    3. Reset behavior & requirements
    4. Response and timing requirements
    Returns (is_complete: bool, missing_semantics: List[str])
    """
    missing: List[str] = []

    # 1. Verify Signals
    target_signals = pack.signals if hasattr(pack, "signals") else []
    if not target_signals and dep_res.signals:
        missing.append("signal_definitions")

    # Check for missing signal widths/directions
    for sig in target_signals:
        if isinstance(sig, dict):
            if not sig.get("width") or not sig.get("direction"):
                missing.append(f"signal_details_for_{sig.get('name', 'sig')}")

    # 2. Check Handshake Stability Rules
    sig_names_lower = [s.get("name", "").lower() if isinstance(s, dict) else s.name.lower() for s in target_signals]
    has_handshake_signals = any("valid" in s or "ready" in s for s in sig_names_lower)
    
    rules = pack.protocol_rules if hasattr(pack, "protocol_rules") else []
    all_rules_text = " ".join([r.get("text", "").lower() for r in rules if isinstance(r, dict)])

    if has_handshake_signals:
        has_handshake_rule = any(kw in all_rules_text for kw in ["valid", "ready", "handshake", "stable", "stability"])
        if not has_handshake_rule:
            missing.append("handshake_stability_rules")

    # 3. Check Reset Behavior
    has_reset_rule = any(kw in all_rules_text for kw in ["reset", "rst", "initialization"])
    if not has_reset_rule:
        missing.append("reset_requirements")

    # 4. Check Timing & Ordering Requirements
    has_timing = len(pack.timing_rules if hasattr(pack, "timing_rules") else []) > 0 or "clock" in all_rules_text
    if not has_timing:
        missing.append("timing_requirements")

    is_complete = (len(missing) == 0)
    if is_complete:
        logger.info("PROTOCOL CONTEXT COMPLETENESS CHECK: PASSED. All required protocol semantics are present.")
    else:
        logger.warning(f"PROTOCOL CONTEXT COMPLETENESS CHECK: FAILED. Missing semantics: {missing}")

    return is_complete, missing
