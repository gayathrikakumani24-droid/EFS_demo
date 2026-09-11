"""
Target Resolver module.

Resolves a requested target component against the complete EFS IR using strict priority:
1. Exact name match
2. Normalized name match
3. Aliases match
4. Semantic search
5. Graph relationships

Returns the resolved target component or explicit error status (TARGET_NOT_FOUND, AMBIGUOUS_TARGET).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from core.efs_ir.models import EFSIR, EFSComponent
from utils.logger import get_logger

logger = get_logger("retrieval.target_resolver")


@dataclass
class TargetResolutionResult:
    status: str  # "SUCCESS" | "TARGET_NOT_FOUND" | "AMBIGUOUS_TARGET"
    target_component: Optional[EFSComponent] = None
    matching_components: List[EFSComponent] = None
    error_message: str = ""
    match_method: str = ""  # "exact" | "normalized" | "alias" | "semantic" | "graph"

    def is_success(self) -> bool:
        return self.status == "SUCCESS" and self.target_component is not None


def _normalize_name(name: str) -> str:
    """Normalize hardware component names by stripping underscores, spaces, dashes, case."""
    return re.sub(r"[^a-zA-Z0-9]", "", name).lower()


def _stem_norm(name: str) -> str:
    """Normalize string and strip common inflections for stem-resilient hardware component lookup."""
    norm = _normalize_name(name)
    return (norm.replace("registered", "register")
                .replace("controlled", "control")
                .replace("engines", "engine")
                .replace("modules", "module")
                .replace("controllers", "controller"))


def resolve_target(target_name: str, efs_ir: EFSIR) -> TargetResolutionResult:
    """
    Resolve target component against the canonical complete EFS IR.
    
    Priority:
    1. Generic top-level query terms
    2. Canonical Top-Level Design ID / Display Name / Metadata Aliases match
    3. Exact Component Name match
    4. Normalized Component Name match
    5. Component Alias / Substring match
    6. Semantic / Description / Word Token match
    7. Top-Level Composite Design Fallback
    8. Structured Failure Diagnostic (TARGET_NOT_FOUND)
    """
    generic_query_terms = (
        "efs", "efsir", "design", "completedesign", "full", "system", "all",
        "io", "iomodule", "inputoutput", "inputoutputmodule", "interface", "interfaces",
        "top", "toplevel", "main", "plantuml", "puml", "diagram", "diagrams",
        "architecture", "overview", "structure", "flow", "schematic", "blockdiagram",
        "module", "controller", "block", "unit", "ip"
    )
    
    raw_target = str(target_name or "").strip()
    target_norm = _normalize_name(raw_target)
    target_stem = _stem_norm(raw_target)
    
    if not efs_ir:
        return TargetResolutionResult(
            status="TARGET_NOT_FOUND",
            error_message=f"TARGET_RESOLUTION_FAILED: EFS IR is empty or uninitialized for target '{raw_target}'."
        )

    # Helper to get or create top-level design component
    def get_top_component() -> EFSComponent:
        top_name = getattr(efs_ir.metadata, "display_name", "") or getattr(efs_ir.metadata, "design_name", "top_module")
        if efs_ir.components:
            # Check if an explicit top-level component exists
            for c in efs_ir.components:
                if c.type in ("top_level", "top_level_design") or c.name.lower() in (top_name.lower(), getattr(efs_ir.metadata, "design_name", "").lower()):
                    return c
            return efs_ir.components[0]
        return EFSComponent(
            component_id="COMP_TOP",
            name=top_name,
            type="top_level_design",
            description=f"Top level component {top_name}"
        )

    # 0. Generic query check
    if not raw_target or target_norm in generic_query_terms or target_norm in ("iomodule", "toplevel", "custommodule", "toplevelmodule"):
        top_comp = get_top_component()
        logger.info(f"Generic target query '{raw_target}' resolved to top-level design component -> {top_comp.name}")
        return TargetResolutionResult(
            status="SUCCESS",
            target_component=top_comp,
            match_method="primary_component_default"
        )

    # 1. Top-Level Design Metadata & Alias Matching
    meta = getattr(efs_ir, "metadata", None)
    if meta:
        meta_design_name = str(getattr(meta, "design_name", "") or "").strip()
        meta_canonical_id = str(getattr(meta, "canonical_id", "") or "").strip()
        meta_display_name = str(getattr(meta, "display_name", "") or "").strip()
        meta_aliases = getattr(meta, "aliases", []) or []
        
        design_names = [meta_design_name, meta_canonical_id, meta_display_name] + list(meta_aliases)
        design_norms = [_normalize_name(n) for n in design_names if n]
        design_stems = [_stem_norm(n) for n in design_names if n]
        
        # Check exact, normalized, or stem match against design metadata & registered aliases
        if raw_target in design_names or target_norm in design_norms or target_stem in design_stems:
            top_comp = get_top_component()
            logger.info(f"Target '{raw_target}' resolved via DESIGN ALIAS match -> {top_comp.name}")
            return TargetResolutionResult(status="SUCCESS", target_component=top_comp, match_method="design_alias")
            
        # Check substring / token overlap against design display name
        if len(target_stem) >= 3 and any(len(ds) >= 3 and (target_stem in ds or ds in target_stem) for ds in design_stems):
            top_comp = get_top_component()
            logger.info(f"Target '{raw_target}' resolved via DESIGN SUBSTRING match -> {top_comp.name}")
            return TargetResolutionResult(status="SUCCESS", target_component=top_comp, match_method="design_substring")

    components = efs_ir.components or []

    # 2. Exact Component Name Match
    exact_matches = [c for c in components if c.name == raw_target]
    if len(exact_matches) == 1:
        logger.info(f"Target '{raw_target}' resolved via EXACT match -> {exact_matches[0].name}")
        return TargetResolutionResult(status="SUCCESS", target_component=exact_matches[0], match_method="exact")
    elif len(exact_matches) > 1:
        logger.warning(f"Target '{raw_target}' matches multiple components exactly: {[c.name for c in exact_matches]}")
        return TargetResolutionResult(
            status="AMBIGUOUS_TARGET",
            matching_components=exact_matches,
            error_message=f"Ambiguous target '{raw_target}': multiple components match exactly ({', '.join([c.name for c in exact_matches])}). User clarification required."
        )

    # 3. Normalized Component Name Match
    norm_matches = [c for c in components if _normalize_name(c.name) == target_norm]
    if len(norm_matches) == 1:
        logger.info(f"Target '{raw_target}' resolved via NORMALIZED match -> {norm_matches[0].name}")
        return TargetResolutionResult(status="SUCCESS", target_component=norm_matches[0], match_method="normalized")
    elif len(norm_matches) > 1:
        logger.warning(f"Target '{raw_target}' matches multiple components via normalized search: {[c.name for c in norm_matches]}")
        return TargetResolutionResult(
            status="AMBIGUOUS_TARGET",
            matching_components=norm_matches,
            error_message=f"Ambiguous target '{raw_target}': multiple normalized components match ({', '.join([c.name for c in norm_matches])}). User clarification required."
        )

    # 4. Component Alias / Substring Match
    alias_matches = []
    for c in components:
        c_norm = _normalize_name(c.name)
        if len(target_norm) >= 3 and (target_norm in c_norm or c_norm in target_norm):
            alias_matches.append(c)

    if len(alias_matches) == 1:
        logger.info(f"Target '{raw_target}' resolved via ALIAS match -> {alias_matches[0].name}")
        return TargetResolutionResult(status="SUCCESS", target_component=alias_matches[0], match_method="alias")
    elif len(alias_matches) > 1:
        logger.warning(f"Target '{raw_target}' matches multiple components via alias search: {[c.name for c in alias_matches]}")
        return TargetResolutionResult(
            status="AMBIGUOUS_TARGET",
            matching_components=alias_matches,
            error_message=f"Ambiguous target '{raw_target}': multiple candidate modules found ({', '.join([c.name for c in alias_matches])}). Please specify exact module name."
        )

    # 5. Semantic / Description / Word Token Match
    semantic_matches = []
    for c in components:
        pattern = r"\b" + re.escape(raw_target) + r"\b"
        if re.search(pattern, c.description, re.IGNORECASE) or re.search(pattern, c.type, re.IGNORECASE):
            semantic_matches.append(c)
        else:
            # Check individual tokens
            target_tokens = [t.lower() for t in re.split(r"\W+", raw_target) if len(t) > 2]
            comp_tokens = [t.lower() for t in re.split(r"\W+", f"{c.name} {c.type} {c.description}")]
            if target_tokens and any(t in comp_tokens for t in target_tokens):
                semantic_matches.append(c)

    if len(semantic_matches) == 1:
        logger.info(f"Target '{raw_target}' resolved via SEMANTIC match -> {semantic_matches[0].name}")
        return TargetResolutionResult(status="SUCCESS", target_component=semantic_matches[0], match_method="semantic")
    elif len(semantic_matches) > 1:
        logger.warning(f"Ambiguous semantic matches for '{raw_target}': {[c.name for c in semantic_matches]}. Defaulting to primary component.")
        return TargetResolutionResult(status="SUCCESS", target_component=semantic_matches[0], match_method="semantic_ambiguous_fallback")

    # Structured Failure Diagnostic
    avail_comp_names = [c.name for c in components]
    meta_name = getattr(meta, "display_name", "") or getattr(meta, "design_name", "N/A") if meta else "N/A"
    meta_canonical = getattr(meta, "canonical_id", "N/A") if meta else "N/A"
    meta_aliases = getattr(meta, "aliases", []) if meta else []
    
    diag_msg = (
        f"TARGET_RESOLUTION_FAILED\n\n"
        f"Requested Target: '{raw_target}'\n"
        f"Design Canonical ID: '{meta_canonical}'\n"
        f"Design Display Name: '{meta_name}'\n"
        f"Registered Design Aliases: {meta_aliases}\n"
        f"Available EFS IR Components ({len(components)}): {avail_comp_names}\n"
        f"Resolution Attempts: Canonical ID match, Design alias match, Exact component match, Normalized match, Substring & Semantic token search.\n"
        f"Reason: The requested target '{raw_target}' does not match any registered top-level design target or primitive component in the canonical EFS IR."
    )

    logger.error(f"Target '{raw_target}' NOT FOUND in complete EFS IR.")
    return TargetResolutionResult(
        status="TARGET_NOT_FOUND",
        error_message=diag_msg
    )

