"""
Human-in-the-Loop (HITL) Plan Integration Module.

Manages architectural conflict presentation with source evidence and proposed alternatives.
Supports:
- Approval
- Rejection
- Alternative selection
- Manual corrections
- Re-generation after correction
Persists approved decisions into EFS IR resolutions so subsequent context compilation
and plan validation recognize and honor them without re-prompting.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

from core.efs_ir.models import EFSConflict, EFSIR, EFSResolution, new_efs_id
from core.agents.rtl_plan_validator import PlanDiagnostic, RTLPlanValidationReport
from utils.logger import get_logger

logger = get_logger("verification.hitl_plan_manager")


class HITLPlanManager:
    """Manages Human-in-the-Loop interactions for RTL generation planning."""

    @staticmethod
    def get_unresolved_architectural_conflicts(
        efs_ir: EFSIR,
        validation_report: Optional[RTLPlanValidationReport] = None,
    ) -> List[Dict[str, Any]]:
        """
        Presents all unresolved architectural conflicts with source evidence
        and generated concrete alternatives for human review.
        """
        conflicts_view: List[Dict[str, Any]] = []
        approved_cids = {
            getattr(r, "conflict_id", "").lower()
            for r in getattr(efs_ir, "resolutions", [])
            if getattr(r, "approval_status", "") == "APPROVED"
        }

        # 1. Conflicts from EFS IR
        for conf in getattr(efs_ir, "conflicts", []):
            cid = getattr(conf, "conflict_id", getattr(conf, "id", "CONF"))
            if cid.lower() in approved_cids:
                continue
            if getattr(conf, "resolution_status", "UNRESOLVED") in ("UNRESOLVED", "ACTIVE"):
                desc = getattr(conf, "description", "")
                evid = getattr(conf, "evidence", "") or desc
                cat = getattr(conf, "category", "GENERAL")

                alternatives = HITLPlanManager._synthesize_alternatives(cat, desc, evid)
                conflicts_view.append({
                    "conflict_id": cid,
                    "category": cat,
                    "affected_object": getattr(conf, "entities", [cid])[0] if getattr(conf, "entities", None) else cid,
                    "issue": desc,
                    "evidence": evid,
                    "alternatives": alternatives,
                    "status": "UNRESOLVED",
                    "hitl_required": True,
                })

        # 2. Diagnostics from Validation Report
        if validation_report:
            for diag in validation_report.diagnostics:
                if not diag.hitl_required:
                    continue
                cid = diag.conflict_id or f"DIAG_{abs(hash(diag.evidence)) % 100000}"
                if cid.lower() in approved_cids:
                    continue
                # Avoid duplicates
                if any(c["conflict_id"] == cid for c in conflicts_view):
                    continue

                alts = diag.alternatives or HITLPlanManager._synthesize_alternatives(diag.category, diag.suggested_resolution, diag.evidence)
                conflicts_view.append({
                    "conflict_id": cid,
                    "category": diag.category,
                    "affected_object": diag.affected_object,
                    "issue": diag.suggested_resolution or diag.evidence,
                    "evidence": diag.evidence,
                    "alternatives": alts,
                    "status": "UNRESOLVED",
                    "hitl_required": True,
                })

        return conflicts_view

    @staticmethod
    def _synthesize_alternatives(category: str, issue: str, evidence: str) -> List[str]:
        """Propose concrete, actionable alternatives for a conflict based on category and evidence."""
        issue_lower = (issue + " " + evidence).lower()
        if "reset" in issue_lower:
            return [
                "Option A: Standardize to active-low asynchronous reset (rst_n == 0)",
                "Option B: Standardize to active-high synchronous reset (rst == 1)",
            ]
        elif "width" in issue_lower:
            return [
                "Option A: Use specification width (e.g., 8-bit)",
                "Option B: Use wider bus width (e.g., 32-bit)",
            ]
        elif "opcode" in issue_lower:
            return [
                "Option A: Preserve first mnemonic encoding and assign next available binary opcode to second mnemonic",
                "Option B: Re-encode both instructions with explicit unique bitfields",
            ]
        elif "direction" in issue_lower:
            return [
                "Option A: Declare signal as input",
                "Option B: Declare signal as output",
            ]
        elif "fsm" in issue_lower:
            return [
                "Option A: Add explicit transition condition using input port handshake",
                "Option B: Make transition unconditional single-cycle dwell (1'b1)",
            ]
        else:
            return [
                "Option A: Accept proposed synthesis default",
                "Option B: Provide manual architectural clarification",
            ]

    @staticmethod
    def approve_resolution(
        efs_ir: EFSIR,
        conflict_id: str,
        approved_value: Any,
        explanation: str = "Approved by human engineer",
        target_entity: str = "",
        target_property: str = ""
    ) -> EFSResolution:
        """
        Record human approval for an architectural conflict.
        Updates conflict status to RESOLVED and records approved EFSResolution.
        """
        # Find conflict if present
        matched_conflict: Optional[EFSConflict] = None
        for c in getattr(efs_ir, "conflicts", []):
            cid = getattr(c, "conflict_id", getattr(c, "id", ""))
            if cid.lower() == conflict_id.lower():
                matched_conflict = c
                c.resolution_status = "RESOLVED"
                break

        res = EFSResolution(
            resolution_id=new_efs_id("res"),
            conflict_id=conflict_id,
            target_entity=target_entity or (matched_conflict.entities[0] if matched_conflict and getattr(matched_conflict, "entities", None) else conflict_id),
            property=target_property or getattr(matched_conflict, "category", "general"),
            resolved_value=approved_value,
            approval_status="APPROVED",
            resolution_source="human",
            rationale=explanation,
            reason=explanation,
            affected_requirements=[conflict_id, target_entity] if target_entity else [conflict_id],
        )

        efs_ir.resolutions.append(res)
        logger.info(f"Approved HITL resolution for conflict '{conflict_id}' -> {approved_value}")
        return res

    @staticmethod
    def reject_resolution(
        efs_ir: EFSIR,
        conflict_id: str,
        reason: str = "Rejected by human engineer"
    ) -> EFSResolution:
        """Record human rejection of a proposed resolution."""
        res = EFSResolution(
            resolution_id=new_efs_id("res"),
            conflict_id=conflict_id,
            target_entity=conflict_id,
            approval_status="REJECTED",
            resolution_source="human",
            rationale=reason,
            reason=reason,
            affected_requirements=[conflict_id]
        )
        efs_ir.resolutions.append(res)
        logger.info(f"Rejected resolution for conflict '{conflict_id}': {reason}")
        return res

    @staticmethod
    def select_alternative(
        efs_ir: EFSIR,
        conflict_id: str,
        alternative_index: int,
        alternatives: List[str],
        explanation: str = "Selected from proposed alternatives"
    ) -> EFSResolution:
        """Select one of the proposed alternatives and approve it."""
        if not (0 <= alternative_index < len(alternatives)):
            raise ValueError(f"Invalid alternative_index {alternative_index}; must be between 0 and {len(alternatives) - 1}")
        selected_text = alternatives[alternative_index]
        return HITLPlanManager.approve_resolution(
            efs_ir=efs_ir,
            conflict_id=conflict_id,
            approved_value=selected_text,
            explanation=f"{explanation}: {selected_text}"
        )

    @staticmethod
    def apply_manual_correction(
        efs_ir: EFSIR,
        conflict_id: str,
        corrected_value: Any,
        target_property: str = "",
        explanation: str = "Manual engineer correction applied"
    ) -> EFSResolution:
        """Apply a direct manual correction to resolve an architectural ambiguity."""
        return HITLPlanManager.approve_resolution(
            efs_ir=efs_ir,
            conflict_id=conflict_id,
            approved_value=corrected_value,
            explanation=explanation,
            target_property=target_property
        )
