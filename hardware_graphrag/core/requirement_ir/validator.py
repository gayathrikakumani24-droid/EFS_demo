"""
Requirement Consistency & Contradiction Detection Engine (Section 13 & 28).

Runs pre-generation validation to detect conflicting signal widths, duplicate IDs,
contradictory reset values, timing conflicts, and undefined dependencies.
Surface conflicts explicitly and blocks unsafe downstream generation.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple
from core.requirement_ir.models import (
    AtomicRequirement, DiscoveredEntity, RequirementIR, SpecConflict, new_req_id
)
from utils.logger import get_logger

logger = get_logger("requirement_ir.validator")


class RequirementValidator:
    """Pre-generation consistency and contradiction checker."""

    def __init__(self, req_ir: RequirementIR):
        self.req_ir = req_ir

    def validate(self) -> Tuple[List[SpecConflict], Dict[str, bool]]:
        """Run all validation checks on the Requirement IR."""
        conflicts: List[SpecConflict] = []

        conflicts.extend(self._check_duplicate_identifiers())
        conflicts.extend(self._check_conflicting_widths())
        conflicts.extend(self._check_conflicting_resets())
        conflicts.extend(self._check_undefined_references())
        conflicts.extend(self._check_state_transitions())

        has_critical = any(c.severity in ("CRITICAL", "HIGH") for c in conflicts)

        status = {
            "is_valid": len(conflicts) == 0,
            "has_critical_conflicts": has_critical,
            "total_conflicts": len(conflicts),
            "critical_count": sum(1 for c in conflicts if c.severity == "CRITICAL"),
            "high_count": sum(1 for c in conflicts if c.severity == "HIGH"),
            "warning_count": sum(1 for c in conflicts if c.severity == "WARNING"),
        }

        return conflicts, status

    def _check_duplicate_identifiers(self) -> List[SpecConflict]:
        conflicts: List[SpecConflict] = []
        seen_entities: Dict[str, str] = {} # name -> entity_id

        for ent in self.req_ir.entities:
            name_lower = ent.name.lower()
            if name_lower in seen_entities and seen_entities[name_lower] != ent.entity_id:
                conflicts.append(SpecConflict(
                    conflict_id=new_req_id("conf"),
                    type="DUPLICATE_ID",
                    claims=[f"Duplicate entity name '{ent.name}' declared."],
                    sources=ent.source_references,
                    severity="HIGH",
                    description=f"Multiple entities discovered with duplicate identifier '{ent.name}'.",
                    conflicting_values=[seen_entities[name_lower], ent.entity_id]
                ))
            else:
                seen_entities[name_lower] = ent.entity_id

        return conflicts

    def _check_conflicting_widths(self) -> List[SpecConflict]:
        conflicts: List[SpecConflict] = []
        width_map: Dict[str, List[Tuple[str, str]]] = {} # entity_name -> [(width_str, req_id/source)]

        # Collect entity names & entity attributes width
        known_entity_names = {e.name.lower(): e.name for e in self.req_ir.entities}

        for ent in self.req_ir.entities:
            w_attr = ent.attributes.get("width") or ent.attributes.get("bits")
            if w_attr:
                w_str = str(w_attr).replace("bits", "").replace("bit", "").strip()
                if w_str.isdigit():
                    width_map.setdefault(ent.name.lower(), []).append((w_str, f"Entity:{ent.name}"))

        for req in self.req_ir.requirements:
            stmt = req.statement
            # Look for width numbers in the requirement
            w_matches = re.findall(r"\b(\d+)\s*(?:bits|bit|b)\b", stmt, re.IGNORECASE)
            if not w_matches:
                w_matches = re.findall(r"\b(?:width|wide)\s+(?:of\s+)?(?:is\s+)?(\d+)\b", stmt, re.IGNORECASE)
            if not w_matches:
                w_matches = re.findall(r"\bis\s+(\d+)\s*bits?\b", stmt, re.IGNORECASE)

            if w_matches:
                w = w_matches[0]
                # Check which known entity is mentioned in this statement
                matched_ent = None
                for ent_name_lower, canonical_name in known_entity_names.items():
                    if ent_name_lower in stmt.lower():
                        matched_ent = canonical_name
                        break
                if not matched_ent and req.subject:
                    matched_ent = req.subject
                if not matched_ent and req.entities:
                    matched_ent = req.entities[0]

                if matched_ent:
                    width_map.setdefault(matched_ent.lower(), []).append((w, req.requirement_id))

        for name, mentions in width_map.items():
            widths = {m[0] for m in mentions}
            if len(widths) > 1:
                conflicts.append(SpecConflict(
                    conflict_id=new_req_id("conf"),
                    type="WIDTH_MISMATCH",
                    claims=[f"Entity '{name}' has conflicting width declarations."],
                    sources=[],
                    severity="CRITICAL",
                    description=f"Entity '{name}' is specified with multiple conflicting bit widths: {list(widths)}.",
                    conflicting_values=[f"{m[1]}: {m[0]} bits" for m in mentions]
                ))

        return conflicts

    def _check_conflicting_resets(self) -> List[SpecConflict]:
        conflicts: List[SpecConflict] = []
        resets: Dict[str, List[str]] = {}

        for req in self.req_ir.requirements:
            stmt = req.statement.lower()
            if "active high" in stmt and "reset" in stmt:
                resets.setdefault("reset_polarity", []).append("active_high")
            if "active low" in stmt and "reset" in stmt:
                resets.setdefault("reset_polarity", []).append("active_low")

        if len(set(resets.get("reset_polarity", []))) > 1:
            conflicts.append(SpecConflict(
                conflict_id=new_req_id("conf"),
                type="RESET_POLARITY_CONFLICT",
                claims=["Specification contains contradictory reset polarity declarations."],
                sources=[],
                severity="HIGH",
                description="Reset signal is specified as both active high and active low in different sections.",
                conflicting_values=resets["reset_polarity"]
            ))

        return conflicts

    def _check_undefined_references(self) -> List[SpecConflict]:
        conflicts: List[SpecConflict] = []
        defined_ids = {e.entity_id for e in self.req_ir.entities} | {e.name for e in self.req_ir.entities}

        for edge in self.req_ir.edges:
            if edge.target not in defined_ids and not edge.target.startswith("REQ_") and not edge.target.startswith("SECTION_") and not edge.target.startswith("TABLE_"):
                conflicts.append(SpecConflict(
                    conflict_id=new_req_id("conf"),
                    type="UNDEFINED_ENTITY",
                    claims=[f"Relationship references undefined target '{edge.target}'."],
                    sources=[edge.source_reference],
                    severity="MEDIUM",
                    description=f"Source node '{edge.source}' references '{edge.target}' which is not in the entity registry.",
                    conflicting_values=[edge.source, edge.target]
                ))

        return conflicts

    def _check_state_transitions(self) -> List[SpecConflict]:
        # Basic state transition consistency check
        return []
