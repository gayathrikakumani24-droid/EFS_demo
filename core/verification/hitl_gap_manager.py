"""
HITL Specification Gap Manager.

Coordinates the full human-in-the-loop lifecycle for hardware specification gaps:
- Detection & translation into structured SpecificationGap models
- Evidence-grounded proposal generation via GapProposalEngine
- Human decision execution: APPROVE, REJECT, EDIT, REQUEST_ALTERNATIVES, REQUEST_CLARIFICATION, DEFER
- Canonical EFS IR mutation via GapResolutionApplier (HUMAN_APPROVED provenance)
- Recompilation of GenerationContext and RTLGenerationPlan
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from core.efs_ir.models import EFSIR
from core.efs_ir.specification_gap import (
    GapResolutionStatus,
    GapSeverity,
    HITLAction,
    HITLDecisionRecord,
    SpecificationGap,
    SpecificationGapCategory,
    SpecificationGapProposal,
)
from core.verification.gap_proposal_engine import GapProposalEngine
from core.verification.gap_resolution_applier import GapResolutionApplier
from core.verification.hitl_plan_manager import HITLPlanManager
from utils.logger import get_logger

logger = get_logger("verification.hitl_gap_manager")


class HITLSpecificationGapManager:
    """
    Central manager for evidence-grounded human specification gap resolution.
    Reuses existing HITLPlanManager persistence while elevating to typed SpecificationGaps.
    """

    @classmethod
    def collect_specification_gaps(
        cls,
        efs_ir: Optional[EFSIR] = None,
        rtl_plan: Optional[Dict[str, Any]] = None,
        design_spec: Optional[str] = None
    ) -> List[SpecificationGap]:
        """
        Collect all genuine specification gaps from RTLGenerationPlan and EFSIR conflicts.
        Deduplicates by entity and category.
        """
        gaps: List[SpecificationGap] = []
        seen_keys = set()

        # 1. From RTL Generation Plan unresolved_logic
        if rtl_plan and isinstance(rtl_plan, dict):
            unres_list = rtl_plan.get("unresolved_logic", [])
            comp_id = rtl_plan.get("module_name", "")
            for u in unres_list:
                gap = SpecificationGap.from_unresolved_logic(u, component_id=comp_id, spec_text=design_spec or "")
                key = f"{gap.affected_entity.lower()}_{gap.category.value}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    gaps.append(gap)

        # 2. From EFS IR conflicts
        if efs_ir and hasattr(efs_ir, "conflicts"):
            for conf in efs_ir.conflicts:
                if getattr(conf, "resolution_status", "") != "RESOLVED" and getattr(conf, "status", "") not in ("RESOLVED", "APPROVED"):
                    gap = SpecificationGap.from_efs_conflict(conf)
                    key = f"{gap.affected_entity.lower()}_{gap.category.value}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        gaps.append(gap)

        # 3. Filter out any gaps that were already approved in efs_ir.resolutions
        if efs_ir and hasattr(efs_ir, "resolutions"):
            approved_targets = {
                (getattr(r, "target_entity", "").lower(), getattr(r, "conflict_id", "").lower())
                for r in efs_ir.resolutions
                if getattr(r, "approval_status", "") == "APPROVED"
            }
            gaps = [
                g for g in gaps
                if (g.affected_entity.lower(), g.gap_id.lower()) not in approved_targets
                and not any(g.affected_entity.lower() == tgt and tgt != "" for tgt, _ in approved_targets)
            ]

        return gaps

    @classmethod
    def generate_proposals_for_gaps(
        cls,
        gaps: List[SpecificationGap],
        efs_ir: Optional[EFSIR] = None,
        context_pack: Optional[Any] = None,
        design_spec: Optional[str] = None
    ) -> List[SpecificationGap]:
        """
        Invoke GapProposalEngine to synthesize evidence-grounded proposals for every gap.
        """
        for gap in gaps:
            if not gap.proposals or gap.resolution_status == GapResolutionStatus.UNRESOLVED:
                GapProposalEngine.analyze_and_propose(
                    gap=gap,
                    efs_ir=efs_ir,
                    context_pack=context_pack,
                    design_spec=design_spec
                )
        return gaps

    @classmethod
    def execute_human_action(
        cls,
        gap: SpecificationGap,
        action: HITLAction,
        proposal_id: Optional[str] = None,
        reviewer: str = "human_engineer",
        rationale: str = "",
        edited_resolution: Optional[str] = None,
        efs_ir: Optional[EFSIR] = None,
        design_spec: Optional[str] = None
    ) -> Tuple[SpecificationGap, Optional[EFSIR]]:
        """
        Execute an explicit human review action on a specification gap proposal.
        """
        prev_status = gap.resolution_status.value if hasattr(gap.resolution_status, "value") else str(gap.resolution_status)

        # Match proposal
        matched_proposal: Optional[SpecificationGapProposal] = None
        if proposal_id:
            matched_proposal = next((p for p in gap.proposals if p.proposal_id == proposal_id), None)
        elif gap.proposals:
            matched_proposal = gap.proposals[0]

        # 1. ACTION: APPROVE
        if action == HITLAction.APPROVE:
            if not matched_proposal and not edited_resolution:
                raise ValueError("Cannot approve gap without a valid proposal or edited resolution.")

            resulting_status = GapResolutionStatus.APPROVED
            gap.resolution_status = resulting_status
            gap.selected_proposal_id = matched_proposal.proposal_id if matched_proposal else None

            decision = HITLDecisionRecord(
                gap_id=gap.gap_id,
                proposal_id=matched_proposal.proposal_id if matched_proposal else None,
                action=HITLAction.APPROVE,
                reviewer=reviewer,
                decision_rationale=rationale or "Explicitly authorized by human engineer",
                edited_resolution=None,
                original_proposal=matched_proposal.to_dict() if matched_proposal else None,
                source_evidence_references=matched_proposal.evidence_references if matched_proposal else [],
                previous_gap_status=prev_status,
                resulting_gap_status=resulting_status.value
            )
            gap.decision = decision

            # Apply to canonical EFS IR if provided
            if efs_ir is not None:
                efs_ir = GapResolutionApplier.apply_approved_decision(efs_ir, gap, decision, matched_proposal)

            logger.info(f"HITL_ACTION: Approved gap '{gap.gap_id}' by {reviewer}")
            return gap, efs_ir

        # 2. ACTION: EDIT (Edit proposed behavior then approve)
        elif action == HITLAction.EDIT:
            if not edited_resolution or not edited_resolution.strip():
                raise ValueError("Edit action requires non-empty edited_resolution string.")

            resulting_status = GapResolutionStatus.APPROVED
            gap.resolution_status = resulting_status
            gap.selected_proposal_id = matched_proposal.proposal_id if matched_proposal else None

            decision = HITLDecisionRecord(
                gap_id=gap.gap_id,
                proposal_id=matched_proposal.proposal_id if matched_proposal else None,
                action=HITLAction.EDIT,
                reviewer=reviewer,
                decision_rationale=rationale or f"Edited and authorized by {reviewer}",
                edited_resolution=edited_resolution.strip(),
                original_proposal=matched_proposal.to_dict() if matched_proposal else None,
                source_evidence_references=matched_proposal.evidence_references if matched_proposal else [],
                previous_gap_status=prev_status,
                resulting_gap_status=resulting_status.value
            )
            gap.decision = decision

            if efs_ir is not None:
                efs_ir = GapResolutionApplier.apply_approved_decision(efs_ir, gap, decision, matched_proposal)

            logger.info(f"HITL_ACTION: Edited and approved gap '{gap.gap_id}' by {reviewer}")
            return gap, efs_ir

        # 3. ACTION: REJECT
        elif action == HITLAction.REJECT:
            resulting_status = GapResolutionStatus.REJECTED
            gap.resolution_status = resulting_status

            decision = HITLDecisionRecord(
                gap_id=gap.gap_id,
                proposal_id=matched_proposal.proposal_id if matched_proposal else None,
                action=HITLAction.REJECT,
                reviewer=reviewer,
                decision_rationale=rationale or "Proposal rejected by human engineer",
                previous_gap_status=prev_status,
                resulting_gap_status=resulting_status.value
            )
            gap.decision = decision

            # Record rejection in EFS IR resolutions so it is auditable
            if efs_ir is not None:
                HITLPlanManager.reject_resolution(efs_ir, gap.gap_id, reason=rationale or "Rejected by human engineer")

            logger.info(f"HITL_ACTION: Rejected gap '{gap.gap_id}' by {reviewer}")
            return gap, efs_ir

        # 4. ACTION: DEFER
        elif action == HITLAction.DEFER:
            resulting_status = GapResolutionStatus.DEFERRED
            gap.resolution_status = resulting_status

            decision = HITLDecisionRecord(
                gap_id=gap.gap_id,
                proposal_id=matched_proposal.proposal_id if matched_proposal else None,
                action=HITLAction.DEFER,
                reviewer=reviewer,
                decision_rationale=rationale or "Gap deferred for future specification release",
                previous_gap_status=prev_status,
                resulting_gap_status=resulting_status.value
            )
            gap.decision = decision
            logger.info(f"HITL_ACTION: Deferred gap '{gap.gap_id}' by {reviewer}")
            return gap, efs_ir

        # 5. ACTION: REQUEST_ALTERNATIVES
        elif action == HITLAction.REQUEST_ALTERNATIVES:
            gap.resolution_status = GapResolutionStatus.UNRESOLVED
            # Preserve previously seen / rejected proposal IDs
            rej_ids = gap.metadata.setdefault("rejected_proposal_ids", [])
            if matched_proposal and matched_proposal.proposal_id not in rej_ids:
                rej_ids.append(matched_proposal.proposal_id)
            for p in gap.proposals:
                if p.proposal_id not in rej_ids:
                    rej_ids.append(p.proposal_id)

            # Re-run proposal engine to synthesize distinct variants
            all_props = GapProposalEngine.analyze_and_propose(
                gap=gap,
                efs_ir=efs_ir,
                design_spec=design_spec
            )
            # Filter out previously rejected/seen proposal IDs
            distinct_props = [p for p in all_props if p.proposal_id not in rej_ids]
            if not distinct_props and all_props:
                # If engine generated variants, ensure proposal IDs are distinct
                for idx, p in enumerate(all_props):
                    p.proposal_id = f"{p.proposal_id}_alt{idx+1}"
                distinct_props = all_props
            gap.proposals = distinct_props

            logger.info(f"HITL_ACTION: Requested new alternatives for gap '{gap.gap_id}', returned {len(distinct_props)} proposals")
            return gap, efs_ir

        # 6. ACTION: REQUEST_CLARIFICATION
        elif action == HITLAction.REQUEST_CLARIFICATION:
            gap.resolution_status = GapResolutionStatus.UNRESOLVED
            clarification_text = str(edited_resolution or rationale or "").strip()
            if clarification_text:
                gap.metadata.setdefault("human_clarifications", []).append({
                    "reviewer": reviewer,
                    "clarification": clarification_text
                })
                # Append human clarification to gap source evidence
                gap.source_evidence = (gap.source_evidence + f" | Clarification by {reviewer}: {clarification_text}").strip(" |")
                # Re-synthesize proposals using the newly provided human evidence
                GapProposalEngine.analyze_and_propose(
                    gap=gap,
                    efs_ir=efs_ir,
                    design_spec=design_spec
                )
            logger.info(f"HITL_ACTION: Captured clarification for gap '{gap.gap_id}' by {reviewer}")
            return gap, efs_ir

        else:
            raise ValueError(f"Unsupported HITL action: {action}")

    @classmethod
    def recompile_after_approvals(
        cls,
        efs_ir: EFSIR,
        design_spec: str,
        target_component_name: str = "",
        target_lang: str = "Verilog"
    ) -> Dict[str, Any]:
        """
        Recompiles GenerationContext and regenerates the RTLGenerationPlan
        following human approvals.
        """
        from core.context_compiler import compile_context
        from core.context_compiler.behavioral_plan_compiler import BehavioralPlanCompiler
        from core.agents.rtl_agent import plan_rtl

        # 1. Obtain resolved view of EFS IR
        resolved_ir = efs_ir.get_resolved_view() if hasattr(efs_ir, "get_resolved_view") else efs_ir

        # 2. Re-plan RTL with resolved EFS IR
        new_plan = plan_rtl(
            design_spec=design_spec,
            target=target_lang,
            efs_ir=resolved_ir
        )

        return {
            "resolved_ir": resolved_ir,
            "plan": new_plan
        }
