"""
Canonical EFS IR Resolution Applier with Provenance Preservation.

Safely incorporates human-approved specification gap resolutions into the canonical
EFS IR model without mutating or overwriting immutable source-derived requirements.
Records provenance with origin 'HUMAN_APPROVED' and maintains bidirectional links
between:
Source Requirement -> EFS IR Entity -> Specification Gap -> AI Proposal -> Human Decision -> RTL Plan
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from core.efs_ir.models import (
    EFSFunctionalBehavior,
    EFSIR,
    EFSResolution,
    EFSSignal,
    EFSTraceability,
    EFSTransition,
    new_efs_id,
)
from core.efs_ir.specification_gap import (
    HITLDecisionRecord,
    SpecificationGap,
    SpecificationGapCategory,
    SpecificationGapProposal,
)
from utils.logger import get_logger

logger = get_logger("verification.gap_resolution_applier")


class GapResolutionApplier:
    """
    Applies human-approved specification gap resolutions to canonical EFS IR entities.
    Strictly preserves provenance, marking approved additions as 'HUMAN_APPROVED'.
    """

    @classmethod
    def apply_approved_decision(
        cls,
        efs_ir: EFSIR,
        gap: SpecificationGap,
        decision: HITLDecisionRecord,
        proposal: Optional[SpecificationGapProposal] = None
    ) -> EFSIR:
        """
        Incorporate approved decision into EFS IR.
        Ensures original source requirements remain intact in the base model.
        """
        entity = gap.affected_entity.strip()
        cat = gap.category
        resolved_text = decision.edited_resolution or (proposal.proposed_resolution if proposal else "")
        reviewer = decision.reviewer or "human_engineer"
        rationale = decision.decision_rationale or "Approved via HITL decision workflow"

        # 0. Idempotency and Conflict Detection (Phase 3.2.1)
        for existing_res in getattr(efs_ir, "resolutions", []):
            if getattr(existing_res, "approval_status", "") == "APPROVED":
                ex_cid = getattr(existing_res, "conflict_id", "")
                ex_did = (getattr(existing_res, "metadata", {}) or {}).get("decision_id")
                ex_val = str(getattr(existing_res, "resolved_value", "") or getattr(existing_res, "proposed_change", "")).strip()

                # Idempotency check: exactly same decision/conflict already applied
                if (ex_cid == gap.gap_id or (ex_did and ex_did == decision.decision_id)) and ex_val == resolved_text.strip():
                    logger.info(f"IDEMPOTENCY: Decision {decision.decision_id} for gap '{gap.gap_id}' is already applied. Skipping duplicate application.")
                    return efs_ir

                # Conflict detection: same target entity already approved with contradictory value
                ex_ent = str(getattr(existing_res, "target_entity", "") or "").lower()
                if ex_ent and ex_ent == entity.lower() and ex_cid != gap.gap_id:
                    if ex_val and ex_val != resolved_text.strip():
                        raise ValueError(
                            f"Contradictory approved decision detected for entity '{entity}': "
                            f"already approved as '{ex_val}' (Resolution {existing_res.resolution_id}), "
                            f"conflicting with new resolution '{resolved_text}'."
                        )

        # 1. Create traceability record linking back to gap, proposal, and decision
        # Strict Phase 3.2.1 Provenance: confidence=None (human authorization is not an automated extraction metric)
        trace = EFSTraceability(
            source_document_id=gap.gap_id,
            source_entity_id=entity,
            original_text=f"Human-approved decision by {reviewer}: {rationale} (Decision: {decision.decision_id}, Gap: {gap.gap_id})",
            extraction_method="human",
            confidence=None
        )

        # 2. Add or update behavioral/structural entity based on category
        if cat == SpecificationGapCategory.UNSPECIFIED_REGISTER_UPDATE:
            cls._apply_register_update(efs_ir, entity, resolved_text, proposal, trace)
        elif cat == SpecificationGapCategory.MISSING_SIGNAL_DRIVER:
            cls._apply_signal_driver(efs_ir, entity, resolved_text, proposal, trace)
        elif cat == SpecificationGapCategory.UNSPECIFIED_CONTROL_FLOW:
            cls._apply_control_flow(efs_ir, entity, resolved_text, proposal, trace)
        elif cat == SpecificationGapCategory.UNSPECIFIED_RESET_BEHAVIOR:
            cls._apply_reset_behavior(efs_ir, entity, resolved_text, proposal, trace)
        else:
            cls._apply_generic_behavior(efs_ir, entity, resolved_text, proposal, trace)

        # 3. Record canonical EFSResolution record in efs_ir.resolutions
        res = EFSResolution(
            resolution_id=new_efs_id("res"),
            conflict_id=gap.gap_id,
            target_entity=entity,
            property=cat.value if hasattr(cat, "value") else str(cat),
            resolution_type="HUMAN_APPROVED_GAP",
            proposed_change=resolved_text,
            resolved_value=resolved_text,
            approval_status="APPROVED",
            resolution_source="human",
            approved_by=reviewer,
            rationale=rationale,
            affected_requirements=[gap.gap_id, entity],
            metadata={
                "gap_id": gap.gap_id,
                "proposal_id": proposal.proposal_id if proposal else None,
                "decision_id": decision.decision_id,
                "provenance_origin": "HUMAN_APPROVED"
            }
        )
        efs_ir.resolutions.append(res)
        logger.info(f"GAP_RESOLVED_CANONICAL: Gap '{gap.gap_id}' ({entity}) resolved into EFS IR as 'HUMAN_APPROVED'.")

        return efs_ir

    @classmethod
    def _apply_register_update(
        cls,
        efs_ir: EFSIR,
        entity: str,
        resolved_text: str,
        proposal: Optional[SpecificationGapProposal],
        trace: EFSTraceability
    ) -> None:
        """Add functional behavior specifying register update."""
        # Parse enable condition and next value
        enable_cond = "1'b1"
        next_val = entity
        if "if (" in resolved_text:
            m = re.search(r"if\s*\(([^)]+)\)\s*(?:[A-Za-z0-9_]+\s*<=\s*([^;]+))?", resolved_text)
            if m:
                enable_cond = m.group(1).strip()
                if m.group(2):
                    next_val = m.group(2).strip()
        elif proposal and proposal.proposed_rtl_semantics:
            enable_cond = str(proposal.proposed_rtl_semantics.get("enable") or proposal.proposed_rtl_semantics.get("condition") or "1'b1")
            next_val = str(proposal.proposed_rtl_semantics.get("next_value") or entity)

        fb = EFSFunctionalBehavior(
            behavior_id=new_efs_id("beh"),
            name=f"update_{entity}",
            subject=entity,
            action=f"{entity} <= {next_val}",
            trigger=enable_cond,
            conditions=[f"when {enable_cond}"] if enable_cond != "1'b1" else [],
            description=f"Human-approved register update for '{entity}' when {enable_cond}: {entity} <= {next_val}",
            traceability=trace
        )
        fb.origin = "HUMAN_APPROVED"
        efs_ir.functional_behavior.append(fb)

    @classmethod
    def _apply_signal_driver(
        cls,
        efs_ir: EFSIR,
        entity: str,
        resolved_text: str,
        proposal: Optional[SpecificationGapProposal],
        trace: EFSTraceability
    ) -> None:
        """Add functional behavior specifying output driver equation."""
        expr = resolved_text
        if "assign " in resolved_text:
            m = re.search(r"assign\s+[A-Za-z0-9_]+\s*=\s*([^;]+);?", resolved_text)
            if m:
                expr = m.group(1).strip()
        elif proposal and proposal.proposed_rtl_semantics and not resolved_text:
            expr = str(proposal.proposed_rtl_semantics.get("expression") or resolved_text)

        fb = EFSFunctionalBehavior(
            behavior_id=new_efs_id("beh"),
            name=f"drive_{entity}",
            subject=entity,
            action=f"{entity} = {expr}",
            description=f"Human-approved driving logic for output port '{entity}': {expr}",
            traceability=trace
        )
        fb.origin = "HUMAN_APPROVED"
        efs_ir.functional_behavior.append(fb)

    @classmethod
    def _apply_control_flow(
        cls,
        efs_ir: EFSIR,
        entity: str,
        resolved_text: str,
        proposal: Optional[SpecificationGapProposal],
        trace: EFSTraceability
    ) -> None:
        """Apply approved transition condition to FSM."""
        cond_expr = resolved_text
        if proposal and proposal.proposed_rtl_semantics:
            cond_expr = str(proposal.proposed_rtl_semantics.get("guard") or resolved_text)
        elif "condition = " in resolved_text:
            cond_expr = resolved_text.split("condition = ")[-1].strip().strip("()")

        # If entity identifies transition e.g. "transition:IDLE->RUN"
        if "->" in entity:
            parts = entity.replace("transition:", "").split("->")
            src_st = parts[0].strip()
            dst_st = parts[1].strip()
            if efs_ir.fsms:
                for fsm in efs_ir.fsms:
                    tr = next((t for t in fsm.transitions if t.source_state.lower() == src_st.lower() and t.target_state.lower() == dst_st.lower()), None)
                    if tr:
                        tr.condition = cond_expr
                        return
                    else:
                        fsm.transitions.append(EFSTransition(
                            source_state=src_st,
                            target_state=dst_st,
                            condition=cond_expr
                        ))
                        return

        # Generic behavior fallback
        fb = EFSFunctionalBehavior(
            behavior_id=new_efs_id("beh"),
            name=f"transition_{entity}",
            subject=entity,
            action=f"transition_condition = {cond_expr}",
            description=f"Human-approved FSM transition condition for '{entity}': {cond_expr}",
            traceability=trace
        )
        efs_ir.functional_behavior.append(fb)

    @classmethod
    def _apply_reset_behavior(
        cls,
        efs_ir: EFSIR,
        entity: str,
        resolved_text: str,
        proposal: Optional[SpecificationGapProposal],
        trace: EFSTraceability
    ) -> None:
        """Apply approved reset polarity and signal name."""
        pol = "active_low" if any(k in resolved_text.lower() for k in ("active_low", "active-low", "rst_n")) else "active_high"
        sig_name = "rst_n" if pol == "active_low" else "rst"

        existing = next((s for s in efs_ir.signals if s.name.lower() in ("rst", "rst_n", "reset")), None)
        if existing:
            existing.name = sig_name
            existing.description = f"Human-approved system reset ({pol})"
        else:
            efs_ir.signals.append(EFSSignal(
                name=sig_name,
                direction="input",
                width="1",
                semantic_role="reset",
                description=f"Human-approved system reset ({pol})"
            ))

    @classmethod
    def _apply_generic_behavior(
        cls,
        efs_ir: EFSIR,
        entity: str,
        resolved_text: str,
        proposal: Optional[SpecificationGapProposal],
        trace: EFSTraceability
    ) -> None:
        """Generic behavioral update."""
        fb = EFSFunctionalBehavior(
            behavior_id=new_efs_id("beh"),
            name=f"resolved_{entity}",
            subject=entity,
            action=resolved_text,
            description=f"Human-approved requirement clarification for '{entity}': {resolved_text}",
            traceability=trace
        )
        efs_ir.functional_behavior.append(fb)
