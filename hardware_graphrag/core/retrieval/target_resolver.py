"""
Target Resolver module.

Resolves a requested target component against the complete EFS IR using strict priority:
1. Exact canonical ID / Name match
2. Exact registered alias match
3. Normalized canonical name match
4. Normalized alias match
5. Hierarchical / Path-aware match
6. Specification & Knowledge Graph entity match
7. Controlled fuzzy / semantic match (with strict similarity threshold)

Returns the resolved target component or explicit error status (TARGET_NOT_FOUND, AMBIGUOUS_TARGET).
Never silently substitutes an unrelated component.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple
from core.efs_ir.models import EFSIR, EFSComponent
from core.normalization.target_normalizer import normalize_target_name
from core.registry import get_registry
from utils.logger import get_logger, truncate_for_log

logger = get_logger("retrieval.target_resolver")


@dataclass
class TargetResolutionResult:
    status: str  # "SUCCESS" | "TARGET_NOT_FOUND" | "AMBIGUOUS_TARGET"
    target_component: Optional[EFSComponent] = None
    matching_components: List[EFSComponent] = None
    error_message: str = ""
    match_method: str = ""  # "exact_id" | "exact_name" | "exact_alias" | "normalized_name" | "normalized_alias" | "hierarchical" | "graph_entity" | "controlled_fuzzy" | "semantic"
    diagnostic_details: Optional[Dict[str, Any]] = None

    def is_success(self) -> bool:
        return self.status == "SUCCESS" and self.target_component is not None


def _stem_norm(name: str) -> str:
    """Normalize string and strip common inflections for stem-resilient hardware component lookup."""
    norm = normalize_target_name(name)
    return (norm.replace("registered", "register")
                .replace("controlled", "control")
                .replace("engines", "engine")
                .replace("modules", "module")
                .replace("controllers", "controller"))


def resolve_target(target_name: str, efs_ir: EFSIR) -> TargetResolutionResult:
    """
    Resolve target component against the canonical complete EFS IR.
    
    Priority:
    1. Generic top-level query terms (explicit system-wide requests)
    2. Exact Canonical Component ID / Name Match
    3. Exact Registered Alias Match
    4. Normalized Canonical Name Match
    5. Normalized Alias Match
    6. Hierarchical Target Match (e.g. Top/Sub)
    7. Specification / Knowledge Graph Entity Match
    8. Controlled Fuzzy / Semantic Match (with strict ratio >= 0.80)
    9. Structured Failure Diagnostic (TARGET_NOT_FOUND) - NO UNSAFE FALLBACK!
    """
    # Strict generic top-level query terms (excluding specific terms like "module", "controller", "unit")
    generic_query_terms = (
        "efs", "efsir", "design", "completedesign", "full", "system", "all",
        "top", "toplevel", "main", "plantuml", "puml", "diagram", "diagrams",
        "architecture", "overview", "structure", "flow", "schematic", "blockdiagram"
    )
    
    raw_target = str(target_name or "").strip()
    target_norm = normalize_target_name(raw_target)
    target_stem = _stem_norm(raw_target)
    
    if not efs_ir:
        return TargetResolutionResult(
            status="TARGET_NOT_FOUND",
            error_message=f"TARGET_RESOLUTION_FAILED: EFS IR is empty or uninitialized for target '{truncate_for_log(raw_target, 60)}'."
        )

    components = efs_ir.components or []

    # Helper to get top-level design component when explicitly requested
    def get_top_component() -> EFSComponent:
        top_name = getattr(efs_ir.metadata, "display_name", "") or getattr(efs_ir.metadata, "design_name", "top_module")
        if components:
            for c in components:
                if c.type in ("top_level", "top_level_design") or c.name.lower() in (top_name.lower(), getattr(efs_ir.metadata, "design_name", "").lower()):
                    return c
            return components[0]
        return EFSComponent(
            component_id="COMP_TOP",
            name=top_name,
            type="top_level_design",
            description=f"Top level component {top_name}"
        )

    # 0. Generic query check
    if not raw_target or target_norm in generic_query_terms:
        top_comp = get_top_component()
        logger.info(f"Generic target query '{truncate_for_log(raw_target, 60)}' resolved to top-level design component -> {top_comp.name}")
        return TargetResolutionResult(
            status="SUCCESS",
            target_component=top_comp,
            match_method="primary_component_default"
        )

    # Bounded target length check
    if len(raw_target) > 80 or "\n" in raw_target:
        logger.warning(f"Requested target '{truncate_for_log(raw_target, 60)}' exceeds target length bounds (>80 chars). Returning TARGET_NOT_FOUND.")
        return TargetResolutionResult(
            status="TARGET_NOT_FOUND",
            error_message=f"TARGET_RESOLUTION_FAILED: Requested target '{truncate_for_log(raw_target, 60)}' exceeds target name bounds."
        )

    # 1. Exact Canonical Component ID / Name Match
    id_matches = [c for c in components if c.component_id == raw_target or c.name == raw_target]
    if len(id_matches) == 1:
        logger.info(f"Target '{truncate_for_log(raw_target, 60)}' resolved via EXACT match -> {id_matches[0].name}")
        return TargetResolutionResult(status="SUCCESS", target_component=id_matches[0], match_method="exact_name")
    elif len(id_matches) > 1:
        logger.warning(f"Target '{truncate_for_log(raw_target, 60)}' matches multiple components exactly: {[c.name for c in id_matches]}")
        return TargetResolutionResult(
            status="AMBIGUOUS_TARGET",
            matching_components=id_matches,
            error_message=f"Ambiguous target '{truncate_for_log(raw_target, 60)}': multiple components match exactly ({', '.join([c.name for c in id_matches])}). User clarification required."
        )

    # 2. Exact Registered Alias Match
    alias_matches = []
    for c in components:
        aliases = getattr(c, "aliases", []) or []
        if raw_target in aliases:
            alias_matches.append(c)
    if len(alias_matches) == 1:
        logger.info(f"Target '{truncate_for_log(raw_target, 60)}' resolved via EXACT ALIAS match -> {alias_matches[0].name}")
        return TargetResolutionResult(status="SUCCESS", target_component=alias_matches[0], match_method="exact_alias")

    # 3. Normalized Canonical Name Match
    norm_matches = [c for c in components if normalize_target_name(c.name) == target_norm]
    if len(norm_matches) == 1:
        logger.info(f"Target '{truncate_for_log(raw_target, 60)}' resolved via NORMALIZED NAME match -> {norm_matches[0].name}")
        return TargetResolutionResult(status="SUCCESS", target_component=norm_matches[0], match_method="normalized_name")
    elif len(norm_matches) > 1:
        logger.warning(f"Target '{truncate_for_log(raw_target, 60)}' matches multiple components via normalized search: {[c.name for c in norm_matches]}")
        return TargetResolutionResult(
            status="AMBIGUOUS_TARGET",
            matching_components=norm_matches,
            error_message=f"Ambiguous target '{truncate_for_log(raw_target, 60)}': multiple normalized components match ({', '.join([c.name for c in norm_matches])}). User clarification required."
        )

    # 4. Normalized Alias Match
    norm_alias_matches = []
    for c in components:
        aliases = getattr(c, "aliases", []) or []
        if any(normalize_target_name(a) == target_norm for a in aliases):
            norm_alias_matches.append(c)
    if len(norm_alias_matches) == 1:
        logger.info(f"Target '{truncate_for_log(raw_target, 60)}' resolved via NORMALIZED ALIAS match -> {norm_alias_matches[0].name}")
        return TargetResolutionResult(status="SUCCESS", target_component=norm_alias_matches[0], match_method="normalized_alias")

    # 5. Hierarchical Target Match (e.g., TopLevel.SubComponent or TopLevel/SubComponent)
    if "." in raw_target or "/" in raw_target:
        parts = [p.strip() for p in re.split(r"[\./]", raw_target) if p.strip()]
        if len(parts) == 2:
            parent_query, child_query = parts[0], parts[1]
            parent_res = resolve_target(parent_query, efs_ir)
            if parent_res.is_success():
                child_norm = normalize_target_name(child_query)
                child_matches = [c for c in components if normalize_target_name(c.name) == child_norm]
                if child_matches:
                    logger.info(f"Target '{truncate_for_log(raw_target, 60)}' resolved via HIERARCHICAL match -> {child_matches[0].name}")
                    return TargetResolutionResult(status="SUCCESS", target_component=child_matches[0], match_method="hierarchical")
                return TargetResolutionResult(status="SUCCESS", target_component=parent_res.target_component, match_method="hierarchical_parent")

    # 6. Top-Level Design Metadata & Alias Matching
    meta = getattr(efs_ir, "metadata", None)
    if meta:
        meta_design_name = str(getattr(meta, "design_name", "") or "").strip()
        meta_canonical_id = str(getattr(meta, "canonical_id", "") or "").strip()
        meta_display_name = str(getattr(meta, "display_name", "") or "").strip()
        meta_aliases = getattr(meta, "aliases", []) or []
        
        design_names = [meta_design_name, meta_canonical_id, meta_display_name] + list(meta_aliases)
        design_norms = [normalize_target_name(n) for n in design_names if n]
        
        if raw_target in design_names or target_norm in design_norms:
            top_comp = get_top_component()
            logger.info(f"Target '{truncate_for_log(raw_target, 60)}' resolved via DESIGN ALIAS match -> {top_comp.name}")
            return TargetResolutionResult(status="SUCCESS", target_component=top_comp, match_method="design_alias")

    # 7. Specification / Knowledge Graph Entity Match (Check Registry & Sub-Entities)
    registry = get_registry()
    graph_entities = registry.all_entities()
    for ent in graph_entities:
        ent_norm = normalize_target_name(ent.name)
        ent_aliases = [normalize_target_name(a) for a in getattr(ent, "aliases", [])]
        if ent_norm == target_norm or target_norm in ent_aliases:
            # Look for corresponding component in EFS IR
            comp_match = next((c for c in components if normalize_target_name(c.name) == ent_norm), None)
            if comp_match:
                logger.info(f"Target '{truncate_for_log(raw_target, 60)}' matched Graph Entity '{ent.name}' -> {comp_match.name}")
                return TargetResolutionResult(status="SUCCESS", target_component=comp_match, match_method="graph_entity")

    # Check EFS IR sub-entities (Instructions, FSMs, Interfaces, Registers)
    sub_entities: List[str] = []
    for inst in getattr(efs_ir, "instructions", []) or []:
        inst_name = getattr(inst, "name", "") or getattr(inst, "mnemonic", "")
        if inst_name:
            sub_entities.append(f"Instruction:{inst_name}")
            if normalize_target_name(inst_name) == target_norm:
                top_comp = get_top_component()
                logger.info(f"Target '{truncate_for_log(raw_target, 60)}' matched EFS IR Instruction '{inst_name}' -> container {top_comp.name}")
                return TargetResolutionResult(status="SUCCESS", target_component=top_comp, match_method="sub_entity_instruction")

    for fsm in getattr(efs_ir, "fsms", []) or []:
        fsm_name = getattr(fsm, "fsm_id", "") or getattr(fsm, "name", "")
        if fsm_name:
            sub_entities.append(f"FSM:{fsm_name}")
            if normalize_target_name(fsm_name) == target_norm:
                parent_comp_id = getattr(fsm, "component_id", "")
                parent_comp = next((c for c in components if c.component_id == parent_comp_id), get_top_component())
                logger.info(f"Target '{truncate_for_log(raw_target, 60)}' matched EFS IR FSM '{fsm_name}' -> {parent_comp.name}")
                return TargetResolutionResult(status="SUCCESS", target_component=parent_comp, match_method="sub_entity_fsm")

    # 8. Controlled Fuzzy / Semantic Match (with strict ratio threshold >= 0.80)
    fuzzy_matches: List[Tuple[float, EFSComponent]] = []
    for c in components:
        c_norm = normalize_target_name(c.name)
        if not c_norm:
            continue
        ratio = SequenceMatcher(None, target_norm, c_norm).ratio()
        if ratio >= 0.80:
            fuzzy_matches.append((ratio, c))

    if fuzzy_matches:
        fuzzy_matches.sort(key=lambda x: x[0], reverse=True)
        best_score, best_comp = fuzzy_matches[0]
        # Only resolve if single clear top candidate or unambiguous top score
        if len(fuzzy_matches) == 1 or (len(fuzzy_matches) > 1 and fuzzy_matches[0][0] > fuzzy_matches[1][0] + 0.05):
            logger.info(f"Target '{truncate_for_log(raw_target, 60)}' resolved via CONTROLLED FUZZY match (ratio={best_score:.2f}) -> {best_comp.name}")
            return TargetResolutionResult(status="SUCCESS", target_component=best_comp, match_method="controlled_fuzzy")

    # 9. Structured Failure Diagnostic (NO UNSAFE FALLBACK!)
    # Collect candidate suggestions by similarity
    candidates_scored = []
    for c in components:
        score = SequenceMatcher(None, target_norm, normalize_target_name(c.name)).ratio()
        candidates_scored.append((score, c))
    candidates_scored.sort(key=lambda x: x[0], reverse=True)
    closest_candidates = [c.name for score, c in candidates_scored if score > 0.3][:5]

    avail_comp_names = [c.name for c in components]
    meta_name = getattr(meta, "display_name", "") or getattr(meta, "design_name", "N/A") if meta else "N/A"
    meta_canonical = getattr(meta, "canonical_id", "N/A") if meta else "N/A"
    meta_aliases = getattr(meta, "aliases", []) if meta else []

    diag_msg = (
        f"TARGET_RESOLUTION_FAILED\n\n"
        f"Requested Target: '{truncate_for_log(raw_target, 60)}'\n"
        f"Normalized Target: '{target_norm}'\n"
        f"Closest Candidates: {closest_candidates if closest_candidates else 'None'}\n"
        f"Design Canonical ID: '{meta_canonical}'\n"
        f"Design Display Name: '{meta_name}'\n"
        f"Registered Design Aliases: {meta_aliases}\n"
        f"Available EFS IR Components ({len(components)}): {avail_comp_names}\n"
        f"Reason: The requested target '{truncate_for_log(raw_target, 60)}' (normalized: '{target_norm}') "
        f"does not match any canonical component, alias, or graph entity in the current EFS IR."
    )

    diagnostic_details = {
        "type": "TARGET_RESOLUTION_FAILED",
        "requested_target": raw_target,
        "normalized_target": target_norm,
        "closest_candidates": closest_candidates,
        "canonical_design_id": meta_canonical,
        "design_display_name": meta_name,
        "aliases": meta_aliases,
        "available_components": avail_comp_names,
        "reason": (
            f"The requested target '{truncate_for_log(raw_target, 60)}' (normalized: '{target_norm}') "
            f"does not match any canonical component, alias, or graph entity in the current EFS IR."
        )
    }

    logger.error(f"Target '{truncate_for_log(raw_target, 60)}' NOT FOUND in complete EFS IR.")
    return TargetResolutionResult(
        status="TARGET_NOT_FOUND",
        matching_components=[c for _, c in candidates_scored[:5]],
        error_message=diag_msg,
        diagnostic_details=diagnostic_details
    )
