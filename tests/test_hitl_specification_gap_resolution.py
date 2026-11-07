"""
Unit and Integration Tests for Phase 3.2: Evidence-Grounded Human-in-the-Loop Specification Gap Resolution.

Covers all 17 required scenarios:
1. Missing register update behavior.
2. Missing output driver.
3. Ambiguous reset semantics.
4. Conflicting specification requirements.
5. Missing FSM transition behavior.
6. Incomplete protocol/application-level behavior.
7. Multiple AI proposals.
8. No-safe-proposal response.
9. Human approval.
10. Human rejection.
11. Human editing followed by approval.
12. Deferred gaps.
13. Canonical EFS IR provenance preservation.
14. Regeneration after approved decisions.
15. Rejection of unauthorized AI-generated implementation decisions.
16. Prevention of unresolved behavior being marked as functional.
17. Regression tests for existing approval workflows.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.agents.constrained_generator import GenerationMode, generate_hybrid_rtl
from core.efs_ir.models import (
    EFSComponent,
    EFSConflict,
    EFSFSM,
    EFSFunctionalBehavior,
    EFSIR,
    EFSMetadata,
    EFSOpcode,
    EFSProtocolRule,
    EFSRegister,
    EFSSignal,
    EFSState,
    EFSTraceability,
    EFSTransition,
    new_efs_id,
)
from core.efs_ir.rtl_plan import PlanOrigin, PlanUnresolvedLogic, RTLGenerationPlan
from core.efs_ir.specification_gap import (
    EvidenceClassification,
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
from core.verification.hitl_gap_manager import HITLSpecificationGapManager
from core.verification.hitl_plan_manager import HITLPlanManager


def _compile_with_iverilog(verilog_code: str, is_sv: bool = False) -> subprocess.CompletedProcess:
    """Helper to compile verilog code with iverilog."""
    iverilog_bin = shutil.which("iverilog")
    if not iverilog_bin:
        raise unittest.SkipTest("iverilog binary not found in PATH")

    with tempfile.TemporaryDirectory() as tmpdir:
        src_file = os.path.join(tmpdir, "test_module.v")
        out_file = os.path.join(tmpdir, "test_module.out")
        with open(src_file, "w", encoding="utf-8") as f:
            f.write(verilog_code)

        std_flag = "-g2012" if is_sv else "-g2005"
        cmd = [iverilog_bin, std_flag, "-o", out_file, src_file]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return res


class TestHITLSpecificationGapResolution(unittest.TestCase):
    """Test suite for evidence-grounded HITL specification gap resolution."""

    def setUp(self):
        self.efs_ir = EFSIR(
            metadata=EFSMetadata(design_name="accel_unit", description="Hardware accelerator unit with matrix compute"),
            components=[EFSComponent(name="accel_unit", type="module")],
            signals=[
                EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
                EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset"),
                EFSSignal(name="valid_in", direction="input", width="1"),
                EFSSignal(name="data_in", direction="input", width="32"),
                EFSSignal(name="done", direction="output", width="1"),
                EFSSignal(name="data_out", direction="output", width="32"),
            ],
            registers=[
                EFSRegister(name="acc_reg", width=32, reset_value="0x0"),
                EFSRegister(name="cfg_reg", width=32, reset_value="0x0")
            ],
            fsms=[
                EFSFSM(
                    name="fsm_main",
                    initial_state="IDLE",
                    states=[EFSState(name="IDLE"), EFSState(name="COMPUTE"), EFSState(name="DONE")],
                    transitions=[
                        EFSTransition(source_state="IDLE", target_state="COMPUTE", condition="valid_in == 1'b1"),
                        EFSTransition(source_state="COMPUTE", target_state="DONE", condition="done_flag == 1'b1"),
                        EFSTransition(source_state="DONE", target_state="IDLE", condition="1'b1")
                    ]
                )
            ]
        )

    def test_1_missing_register_update_behavior(self):
        """Scenario 1: Missing register update behavior is detected and classified."""
        unres = PlanUnresolvedLogic(
            affected_component="accel_unit",
            affected_entity="acc_reg",
            description="Register 'acc_reg' is declared in the specification, but no update condition or next-value expression is defined.",
            missing_information="Update condition or load equation for register 'acc_reg'",
            severity="WARNING",
            blocks_generation=False
        )
        gap = SpecificationGap.from_unresolved_logic(unres, component_id="accel_unit")
        self.assertEqual(gap.category, SpecificationGapCategory.UNSPECIFIED_REGISTER_UPDATE)
        self.assertEqual(gap.affected_entity, "acc_reg")
        self.assertFalse(gap.blocks_generation)

        # Propose update
        proposals = GapProposalEngine.analyze_and_propose(gap, efs_ir=self.efs_ir)
        self.assertTrue(len(proposals) > 0)
        top_prop = proposals[0]
        self.assertTrue(top_prop.is_safe_proposal)
        self.assertEqual(top_prop.evidence_classification, EvidenceClassification.IMPLEMENTATION_CHOICE)
        self.assertIn("acc_reg", top_prop.relevant_registers)

    def test_2_missing_output_driver(self):
        """Scenario 2: Missing output driver is classified as blocking specification gap."""
        unres = PlanUnresolvedLogic(
            affected_component="accel_unit",
            affected_entity="data_out",
            description="Output port 'data_out' is declared in the interface, but no driving continuous expression or register assignment is specified.",
            missing_information="Driving equation for output port 'data_out'",
            severity="BLOCKING",
            blocks_generation=True
        )
        gap = SpecificationGap.from_unresolved_logic(unres, component_id="accel_unit")
        self.assertEqual(gap.category, SpecificationGapCategory.MISSING_SIGNAL_DRIVER)
        self.assertEqual(gap.severity, GapSeverity.BLOCKING)
        self.assertTrue(gap.blocks_generation)

    def test_3_ambiguous_reset_semantics(self):
        """Scenario 3: Ambiguous reset semantics produces distinct async-low and sync-high proposals."""
        conf = EFSConflict(
            conflict_id="CONF_RESET_01",
            category="RESET",
            type="RESET_POLARITY_CONFLICT",
            description="Specification mentions active-low reset in header but active-high reset in description.",
            blocking=True,
            entities=["rst_n", "rst"]
        )
        gap = SpecificationGap.from_efs_conflict(conf)
        self.assertEqual(gap.category, SpecificationGapCategory.UNSPECIFIED_RESET_BEHAVIOR)

        proposals = GapProposalEngine.analyze_and_propose(gap, efs_ir=self.efs_ir)
        self.assertEqual(len(proposals), 2)
        titles = [p.title for p in proposals]
        self.assertTrue(any("Active-Low" in t for t in titles))
        self.assertTrue(any("Active-High" in t for t in titles))

    def test_4_conflicting_specification_requirements(self):
        """Scenario 4: Conflicting specification requirements are captured and structured."""
        conf = EFSConflict(
            conflict_id="CONF_DIR_01",
            category="INTERFACE",
            type="CONFLICTING_DIRECTION",
            description="Signal 'ctrl_flag' declared as input in block diagram but output in register map.",
            blocking=True,
            entities=["ctrl_flag"]
        )
        gap = SpecificationGap.from_efs_conflict(conf)
        self.assertEqual(gap.category, SpecificationGapCategory.CONFLICTING_REQUIREMENTS)
        self.assertTrue(gap.blocks_generation)
        self.assertEqual(gap.severity, GapSeverity.BLOCKING)

    def test_5_missing_fsm_transition_behavior(self):
        """Scenario 5: Missing FSM transition condition asks for trigger signal."""
        unres = PlanUnresolvedLogic(
            affected_component="accel_unit",
            affected_entity="transition:IDLE->COMPUTE",
            description="FSM transition from 'IDLE' to 'COMPUTE' has an incomplete or unspecified transition condition.",
            missing_information="Boolean transition condition expression",
            severity="BLOCKING",
            blocks_generation=True
        )
        gap = SpecificationGap.from_unresolved_logic(unres, component_id="accel_unit")
        self.assertEqual(gap.category, SpecificationGapCategory.UNSPECIFIED_CONTROL_FLOW)

        proposals = GapProposalEngine.analyze_and_propose(gap, efs_ir=self.efs_ir)
        self.assertTrue(len(proposals) > 0)
        self.assertIn("valid_in", proposals[0].proposed_resolution)

    def test_6_incomplete_protocol_behavior(self):
        """Scenario 6: Incomplete protocol/application-level behavior separates handshake from trigger."""
        gap = SpecificationGap(
            gap_id="GAP_PROTO_01",
            component_id="accel_unit",
            affected_entity="axi_stream_tx",
            category=SpecificationGapCategory.INCOMPLETE_PROTOCOL_BEHAVIOR,
            description="Protocol specifies valid/ready handshake, but controller initiation trigger is undefined.",
            severity=GapSeverity.WARNING,
            blocks_generation=False
        )
        proposals = GapProposalEngine.analyze_and_propose(gap, efs_ir=self.efs_ir)
        self.assertEqual(len(proposals), 1)
        self.assertIn("Application Initiation Trigger", proposals[0].title)
        self.assertTrue(len(proposals[0].clarification_questions) > 0)

    def test_7_multiple_ai_proposals(self):
        """Scenario 7: Generates multiple alternatives with distinct trade-offs for completion signals."""
        gap = SpecificationGap(
            gap_id="GAP_DONE_01",
            component_id="accel_unit",
            affected_entity="done",
            category=SpecificationGapCategory.MISSING_SIGNAL_DRIVER,
            description="Output port 'done' has no driving logic defined.",
            severity=GapSeverity.BLOCKING
        )
        proposals = GapProposalEngine.analyze_and_propose(gap, efs_ir=self.efs_ir)
        self.assertGreaterEqual(len(proposals), 2)
        titles = [p.title for p in proposals]
        self.assertTrue(any("Level" in t for t in titles))
        self.assertTrue(any("Pulse" in t for t in titles))

    def test_8_no_safe_proposal_response(self):
        """Scenario 8: Returns NO_SAFE_PROPOSAL when evidence is completely missing."""
        empty_ir = EFSIR(metadata=EFSMetadata(design_name="empty", description=""))
        gap = SpecificationGap(
            gap_id="GAP_UNKNOWN_01",
            component_id="empty",
            affected_entity="secret_output",
            category=SpecificationGapCategory.MISSING_SIGNAL_DRIVER,
            description="Output port 'secret_output' has no driving equation.",
            severity=GapSeverity.BLOCKING
        )
        proposals = GapProposalEngine.analyze_and_propose(gap, efs_ir=empty_ir)
        self.assertEqual(len(proposals), 1)
        self.assertFalse(proposals[0].is_safe_proposal)
        self.assertEqual(proposals[0].proposed_resolution, "NO_SAFE_PROPOSAL")
        self.assertTrue(len(proposals[0].clarification_questions) > 0)

    def test_9_human_approval(self):
        """Scenario 9: Human executes APPROVE on proposal; canonical EFS IR updated with HUMAN_APPROVED provenance."""
        gap = SpecificationGap(
            gap_id="GAP_REG_01",
            component_id="accel_unit",
            affected_entity="acc_reg",
            category=SpecificationGapCategory.UNSPECIFIED_REGISTER_UPDATE,
            description="Unspecified register update",
            severity=GapSeverity.WARNING
        )
        proposals = GapProposalEngine.analyze_and_propose(gap, efs_ir=self.efs_ir)
        top_prop = proposals[0]

        updated_gap, updated_ir = HITLSpecificationGapManager.execute_human_action(
            gap=gap,
            action=HITLAction.APPROVE,
            proposal_id=top_prop.proposal_id,
            reviewer="lead_engineer_alice",
            rationale="Approved synchronous clock enable with data_in",
            efs_ir=self.efs_ir
        )
        self.assertEqual(updated_gap.resolution_status, GapResolutionStatus.APPROVED)
        self.assertIsNotNone(updated_gap.decision)
        self.assertEqual(updated_gap.decision.action, HITLAction.APPROVE)
        self.assertEqual(updated_gap.decision.reviewer, "lead_engineer_alice")

        # Verify canonical EFS IR updated with HUMAN_APPROVED provenance
        added_fb = next((fb for fb in updated_ir.functional_behavior if fb.subject == "acc_reg"), None)
        self.assertIsNotNone(added_fb)
        self.assertEqual(added_fb.traceability.extraction_method, "human")
        self.assertIn("lead_engineer_alice", added_fb.traceability.original_text)

    def test_10_human_rejection(self):
        """Scenario 10: Human executes REJECT; proposal is rejected and not applied to canonical EFS IR."""
        gap = SpecificationGap(
            gap_id="GAP_REJ_01",
            component_id="accel_unit",
            affected_entity="cfg_reg",
            category=SpecificationGapCategory.UNSPECIFIED_REGISTER_UPDATE,
            description="Unspecified configuration register update",
            severity=GapSeverity.WARNING
        )
        proposals = GapProposalEngine.analyze_and_propose(gap, efs_ir=self.efs_ir)
        top_prop = proposals[0]

        initial_fb_count = len(self.efs_ir.functional_behavior)
        updated_gap, updated_ir = HITLSpecificationGapManager.execute_human_action(
            gap=gap,
            action=HITLAction.REJECT,
            proposal_id=top_prop.proposal_id,
            reviewer="architect_bob",
            rationale="Rejected: cfg_reg is read-only in this hardware revision",
            efs_ir=self.efs_ir
        )
        self.assertEqual(updated_gap.resolution_status, GapResolutionStatus.REJECTED)
        self.assertEqual(len(updated_ir.functional_behavior), initial_fb_count)

        # Check rejection is logged in resolutions
        rej_res = next((r for r in updated_ir.resolutions if r.conflict_id == "GAP_REJ_01"), None)
        self.assertIsNotNone(rej_res)
        self.assertEqual(rej_res.approval_status, "REJECTED")

    def test_11_human_editing_followed_by_approval(self):
        """Scenario 11: Human modifies proposed behavior before approval (EDIT action)."""
        gap = SpecificationGap(
            gap_id="GAP_EDIT_01",
            component_id="accel_unit",
            affected_entity="data_out",
            category=SpecificationGapCategory.MISSING_SIGNAL_DRIVER,
            description="Missing driver for data_out",
            severity=GapSeverity.BLOCKING
        )
        GapProposalEngine.analyze_and_propose(gap, efs_ir=self.efs_ir)

        custom_resolution = "assign data_out = acc_reg ^ 32'hFFFFFFFF;"
        updated_gap, updated_ir = HITLSpecificationGapManager.execute_human_action(
            gap=gap,
            action=HITLAction.EDIT,
            reviewer="designer_charlie",
            rationale="Invert accumulator bits for active-low downstream parity",
            edited_resolution=custom_resolution,
            efs_ir=self.efs_ir
        )
        self.assertEqual(updated_gap.resolution_status, GapResolutionStatus.APPROVED)
        self.assertEqual(updated_gap.decision.action, HITLAction.EDIT)
        self.assertEqual(updated_gap.decision.edited_resolution, custom_resolution)

        # Verify custom equation in canonical EFS IR
        added_fb = next((fb for fb in updated_ir.functional_behavior if fb.subject == "data_out"), None)
        self.assertIsNotNone(added_fb)
        self.assertIn("acc_reg ^ 32'hFFFFFFFF", added_fb.action)
        self.assertEqual(added_fb.traceability.extraction_method, "human")

    def test_12_deferred_gaps(self):
        """Scenario 12: Human executes DEFER; gap remains unresolved and carried forward."""
        gap = SpecificationGap(
            gap_id="GAP_DEF_01",
            component_id="accel_unit",
            affected_entity="optional_debug_port",
            category=SpecificationGapCategory.MISSING_SIGNAL_DRIVER,
            description="Optional debug trace output port omitted from v1 spec.",
            severity=GapSeverity.WARNING
        )
        updated_gap, _ = HITLSpecificationGapManager.execute_human_action(
            gap=gap,
            action=HITLAction.DEFER,
            reviewer="program_manager_dan",
            rationale="Defer debug instrumentation to Rev B milestone"
        )
        self.assertEqual(updated_gap.resolution_status, GapResolutionStatus.DEFERRED)
        self.assertEqual(updated_gap.decision.action, HITLAction.DEFER)

    def test_13_canonical_efs_ir_provenance_preservation(self):
        """Scenario 13: Verifies original source requirements are preserved and distinct from HUMAN_APPROVED additions."""
        # Add base source requirement
        base_fb = EFSFunctionalBehavior(
            behavior_id="BEH_SRC_01",
            name="base_alu",
            subject="alu",
            action="alu_out = a + b",
            traceability=EFSTraceability(extraction_method="heuristic", source_entity_id="REQ_ALU_01")
        )
        self.efs_ir.functional_behavior.append(base_fb)

        gap = SpecificationGap(
            gap_id="GAP_ACC_01",
            component_id="accel_unit",
            affected_entity="acc_reg",
            category=SpecificationGapCategory.UNSPECIFIED_REGISTER_UPDATE,
            description="acc_reg update unspecified"
        )
        prop = SpecificationGapProposal(
            gap_id="GAP_ACC_01",
            proposed_resolution="if (valid_in) acc_reg <= data_in;",
            proposed_rtl_semantics={"enable": "valid_in", "next_value": "data_in"}
        )
        gap.proposals = [prop]

        decision = HITLDecisionRecord(
            gap_id="GAP_ACC_01",
            proposal_id=prop.proposal_id,
            action=HITLAction.APPROVE,
            reviewer="engineer_eve",
            decision_rationale="Approved by Eve"
        )
        GapResolutionApplier.apply_approved_decision(self.efs_ir, gap, decision, prop)

        # Verify base requirement remains unchanged
        original = next((fb for fb in self.efs_ir.functional_behavior if fb.behavior_id == "BEH_SRC_01"), None)
        self.assertIsNotNone(original)
        self.assertEqual(original.traceability.extraction_method, "heuristic")

        # Verify approved requirement is distinct
        approved = next((fb for fb in self.efs_ir.functional_behavior if fb.subject == "acc_reg"), None)
        self.assertIsNotNone(approved)
        self.assertEqual(approved.traceability.extraction_method, "human")

    def test_14_regeneration_after_approved_decisions(self):
        """Scenario 14: Recompilation after approval produces RTL plan containing approved logic."""
        gap = SpecificationGap(
            gap_id="GAP_OUT_01",
            component_id="accel_unit",
            affected_entity="data_out",
            category=SpecificationGapCategory.MISSING_SIGNAL_DRIVER,
            description="data_out driver missing"
        )
        prop = SpecificationGapProposal(
            gap_id="GAP_OUT_01",
            proposed_resolution="assign data_out = data_in + 32'd1;",
            proposed_rtl_semantics={"expression": "data_in + 32'd1"}
        )
        gap.proposals = [prop]

        decision = HITLDecisionRecord(
            gap_id="GAP_OUT_01",
            proposal_id=prop.proposal_id,
            action=HITLAction.APPROVE,
            reviewer="frank"
        )
        GapResolutionApplier.apply_approved_decision(self.efs_ir, gap, decision, prop)

        # Verify resolved view includes new behavior
        resolved_ir = self.efs_ir.get_resolved_view()
        fb_names = [fb.subject for fb in resolved_ir.functional_behavior]
        self.assertIn("data_out", fb_names)

    def test_15_rejection_of_unauthorized_ai_proposals(self):
        """Scenario 15: Unapproved AI proposals are never incorporated into canonical EFS IR or RTL plan."""
        gap = SpecificationGap(
            gap_id="GAP_UNAUTH_01",
            component_id="accel_unit",
            affected_entity="cfg_reg",
            category=SpecificationGapCategory.UNSPECIFIED_REGISTER_UPDATE,
            description="cfg_reg update undefined"
        )
        # Generate proposal but DO NOT approve it
        proposals = GapProposalEngine.analyze_and_propose(gap, efs_ir=self.efs_ir)
        self.assertTrue(len(proposals) > 0)

        # Check efs_ir: no functional behavior for cfg_reg should exist
        fb_targets = [fb.subject for fb in self.efs_ir.functional_behavior]
        self.assertNotIn("cfg_reg", fb_targets)

    def test_16_prevention_of_unresolved_behavior_marked_as_functional(self):
        """Scenario 16: Unresolved outputs with placeholders are not reported as functional implementations."""
        plan_dict = {
            "module_name": "partial_unit",
            "target_language": "Verilog",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "valid_out", "direction": "output", "width": "1"}
            ],
            "unresolved_logic": [
                {
                    "affected_entity": "valid_out",
                    "description": "Output port 'valid_out' has no driving logic.",
                    "severity": "BLOCKING"
                }
            ]
        }
        # Emitted RTL should preserve unresolved boundary and not claim full implementation
        code, report = generate_hybrid_rtl(
            design_spec="Partial module with undriven valid_out",
            target="Verilog",
            rtl_plan=plan_dict,
            mode=GenerationMode.DETERMINISTIC
        )
        self.assertIn("// [UNRESOLVED_LOGIC_BOUNDARIES]", code)
        self.assertIn("valid_out", code)

        # Verify with iverilog compiler
        res = _compile_with_iverilog(code)
        self.assertEqual(res.returncode, 0, f"iverilog compilation failed: {res.stderr}\nCode:\n{code}")

    def test_17_regression_existing_hitl_plan_manager(self):
        """Scenario 17: Preserves full backward compatibility with HITLPlanManager."""
        conf = EFSConflict(
            conflict_id="CONF_OPCODE_99",
            category="OPCODE",
            type="DUPLICATE_OPCODE",
            description="Duplicate opcode 0x01",
            entities=["LOAD", "STORE"]
        )
        self.efs_ir.conflicts.append(conf)

        res = HITLPlanManager.approve_resolution(
            efs_ir=self.efs_ir,
            conflict_id="CONF_OPCODE_99",
            approved_value="STORE.opcode = 0x02",
            explanation="Reassigned STORE opcode to 0x02"
        )
        self.assertEqual(res.approval_status, "APPROVED")
        self.assertEqual(conf.resolution_status, "RESOLVED")
        self.assertIn(res, self.efs_ir.resolutions)

    def test_18_proposal_safety_gate_no_safe_proposal(self):
        """Scenario 18: Proposal safety gate returns NO_SAFE_PROPOSAL when evidence lacks functional intent."""
        gap = SpecificationGap(
            gap_id="GAP_SAFE_01",
            component_id="accel_unit",
            affected_entity="aux_status",
            category=SpecificationGapCategory.MISSING_SIGNAL_DRIVER,
            description="Specification mentions aux_status port but defines no logic or conditions.",
            severity=GapSeverity.BLOCKING
        )
        # Empty IR with no matching inputs or FSM states for aux_status
        empty_ir = EFSIR(metadata=EFSMetadata(design_name="empty", description=""))
        proposals = GapProposalEngine.analyze_and_propose(gap, efs_ir=empty_ir)
        self.assertEqual(len(proposals), 1)
        self.assertFalse(proposals[0].is_safe_proposal)
        self.assertEqual(proposals[0].proposed_resolution, "NO_SAFE_PROPOSAL")
        self.assertTrue(len(proposals[0].clarification_questions) > 0)

    def test_19_evidence_classification_taxonomy(self):
        """Scenario 19: Proved classification across SOURCE_SUPPORTED, DERIVED, IMPLEMENTATION_CHOICE, UNSUPPORTED_ASSUMPTION."""
        # A: SOURCE_SUPPORTED (Spec directly defines reset semantics)
        gap_rst = SpecificationGap(
            gap_id="GAP_RST_01",
            component_id="accel_unit",
            affected_entity="rst_n",
            category=SpecificationGapCategory.UNSPECIFIED_RESET_BEHAVIOR,
            description="Reset polarity check",
            source_evidence="Reset is active-low asynchronous rst_n asserting all registers to zero."
        )
        props_rst = GapProposalEngine.analyze_and_propose(gap_rst, efs_ir=self.efs_ir, design_spec="Reset is active-low asynchronous rst_n asserting all registers to zero.")
        self.assertTrue(any(p.evidence_classification == EvidenceClassification.SOURCE_SUPPORTED for p in props_rst))

        # B: IMPLEMENTATION_CHOICE (Direct load or registered pipeline stage)
        gap_reg = SpecificationGap(
            gap_id="GAP_REG_02",
            component_id="accel_unit",
            affected_entity="acc_reg",
            category=SpecificationGapCategory.UNSPECIFIED_REGISTER_UPDATE,
            description="Register acc_reg update logic unspecified"
        )
        props_reg = GapProposalEngine.analyze_and_propose(gap_reg, efs_ir=self.efs_ir)
        self.assertTrue(any(p.evidence_classification == EvidenceClassification.IMPLEMENTATION_CHOICE for p in props_reg))

        # C: UNSUPPORTED_ASSUMPTION (NO_SAFE_PROPOSAL when evidence missing)
        gap_unk = SpecificationGap(
            gap_id="GAP_UNK_01",
            component_id="accel_unit",
            affected_entity="mystery_pin",
            category=SpecificationGapCategory.MISSING_SIGNAL_DRIVER,
            description="Mystery pin has no definition"
        )
        empty_ir = EFSIR(metadata=EFSMetadata(design_name="empty", description=""))
        props_unk = GapProposalEngine.analyze_and_propose(gap_unk, efs_ir=empty_ir)
        self.assertTrue(any(p.evidence_classification == EvidenceClassification.UNSUPPORTED_ASSUMPTION for p in props_unk))

    def test_20_provenance_immutability_layer_a_vs_layer_b(self):
        """Scenario 20: Layer A base requirements remain immutable while Layer B human decisions have confidence=None and origin='HUMAN_APPROVED'."""
        orig_req = EFSFunctionalBehavior(
            behavior_id="FB_DATA_IN",
            subject="data_in",
            description="Primary input port for data samples",
            traceability=EFSTraceability(
                source_document_id="SPEC_DOC_01",
                source_entity_id="SEC_1",
                original_text="Input data stream is 32-bit data_in",
                extraction_method="llm_extraction",
                confidence=0.95
            )
        )
        orig_req.origin = "SPECIFICATION"
        self.efs_ir.functional_behavior.append(orig_req)

        # Apply human approved decision for data_out
        gap = SpecificationGap(
            gap_id="GAP_PROV_01",
            component_id="accel_unit",
            affected_entity="data_out",
            category=SpecificationGapCategory.MISSING_SIGNAL_DRIVER,
            description="Output data_out driver"
        )
        prop = SpecificationGapProposal(
            proposal_id="PROP_PROV_01",
            gap_id="GAP_PROV_01",
            title="Drive data_out from acc_reg",
            proposed_resolution="assign data_out = acc_reg;",
            detailed_behavioral_explanation="Drive data_out directly from acc_reg"
        )
        gap.proposals = [prop]

        updated_gap, updated_ir = HITLSpecificationGapManager.execute_human_action(
            gap=gap,
            action=HITLAction.APPROVE,
            proposal_id="PROP_PROV_01",
            reviewer="lead_architect",
            efs_ir=self.efs_ir
        )

        # Base requirement (Layer A) remains completely untouched
        base_req = next(fb for fb in updated_ir.functional_behavior if fb.subject == "data_in")
        self.assertEqual(base_req.origin, "SPECIFICATION")
        self.assertEqual(base_req.traceability.confidence, 0.95)
        self.assertEqual(base_req.traceability.extraction_method, "llm_extraction")

        # Human approved decision (Layer B) has origin HUMAN_APPROVED and confidence=None
        approved_req = next(fb for fb in updated_ir.functional_behavior if fb.subject == "data_out")
        self.assertEqual(approved_req.origin, "HUMAN_APPROVED")
        self.assertIsNone(approved_req.traceability.confidence)
        self.assertEqual(approved_req.traceability.extraction_method, "human")

    def test_21_idempotent_gap_resolution_application(self):
        """Scenario 21: Re-applying the exact same resolution is safe and idempotent without duplicates."""
        gap = SpecificationGap(
            gap_id="GAP_IDEMP_01",
            component_id="accel_unit",
            affected_entity="acc_reg",
            category=SpecificationGapCategory.UNSPECIFIED_REGISTER_UPDATE,
            description="Register acc_reg update"
        )
        prop = SpecificationGapProposal(
            proposal_id="PROP_IDEMP_01",
            gap_id="GAP_IDEMP_01",
            title="Update acc_reg on valid_in",
            proposed_resolution="if (valid_in) acc_reg <= data_in;",
            detailed_behavioral_explanation="Standard load on valid_in"
        )
        gap.proposals = [prop]

        # First application
        _, ir_1 = HITLSpecificationGapManager.execute_human_action(
            gap=gap,
            action=HITLAction.APPROVE,
            proposal_id="PROP_IDEMP_01",
            reviewer="eng_1",
            efs_ir=self.efs_ir
        )
        count_fb_1 = len([fb for fb in ir_1.functional_behavior if fb.subject == "acc_reg"])
        count_res_1 = len([r for r in ir_1.resolutions if r.conflict_id == "GAP_IDEMP_01"])
        self.assertEqual(count_fb_1, 1)
        self.assertEqual(count_res_1, 1)

        # Second application of identical decision
        _, ir_2 = HITLSpecificationGapManager.execute_human_action(
            gap=gap,
            action=HITLAction.APPROVE,
            proposal_id="PROP_IDEMP_01",
            reviewer="eng_1",
            efs_ir=ir_1
        )
        count_fb_2 = len([fb for fb in ir_2.functional_behavior if fb.subject == "acc_reg"])
        count_res_2 = len([r for r in ir_2.resolutions if r.conflict_id == "GAP_IDEMP_01"])
        self.assertEqual(count_fb_2, 1)
        self.assertEqual(count_res_2, 1)

    def test_22_contradiction_detection(self):
        """Scenario 22: Applying contradictory resolutions for the same entity raises ValueError."""
        gap_1 = SpecificationGap(
            gap_id="GAP_CONTR_01",
            component_id="accel_unit",
            affected_entity="data_out",
            category=SpecificationGapCategory.MISSING_SIGNAL_DRIVER,
            description="First driver specification"
        )
        prop_1 = SpecificationGapProposal(
            proposal_id="PROP_CONTR_01",
            gap_id="GAP_CONTR_01",
            title="Drive data_out from data_in",
            proposed_resolution="assign data_out = data_in;",
            detailed_behavioral_explanation="Pass through"
        )
        gap_1.proposals = [prop_1]

        _, ir_step1 = HITLSpecificationGapManager.execute_human_action(
            gap=gap_1,
            action=HITLAction.APPROVE,
            proposal_id="PROP_CONTR_01",
            reviewer="eng_1",
            efs_ir=self.efs_ir
        )

        gap_2 = SpecificationGap(
            gap_id="GAP_CONTR_02",
            component_id="accel_unit",
            affected_entity="data_out",
            category=SpecificationGapCategory.MISSING_SIGNAL_DRIVER,
            description="Contradictory second driver specification"
        )
        prop_2 = SpecificationGapProposal(
            proposal_id="PROP_CONTR_02",
            gap_id="GAP_CONTR_02",
            title="Drive data_out from 32'd0",
            proposed_resolution="assign data_out = 32'd0;",
            detailed_behavioral_explanation="Constant zero"
        )
        gap_2.proposals = [prop_2]

        with self.assertRaises(ValueError) as ctx:
            HITLSpecificationGapManager.execute_human_action(
                gap=gap_2,
                action=HITLAction.APPROVE,
                proposal_id="PROP_CONTR_02",
                reviewer="eng_2",
                efs_ir=ir_step1
            )
        self.assertIn("Contradictory approved decision detected", str(ctx.exception))

    def test_23_complete_six_action_hitl_lifecycle(self):
        """Scenario 23: Complete 6-action HITL lifecycle (Approve, Reject, Edit, Defer, Request Alternatives, Request Clarification)."""
        # 1. Action: REJECT
        gap_rej = SpecificationGap(gap_id="G_REJ", component_id="accel_unit", affected_entity="acc_reg", category=SpecificationGapCategory.UNSPECIFIED_REGISTER_UPDATE, description="Reject test")
        prop_rej = SpecificationGapProposal(proposal_id="P_REJ", gap_id="G_REJ", title="Rej", proposed_resolution="acc_reg <= 0;", detailed_behavioral_explanation="Rej")
        gap_rej.proposals = [prop_rej]
        g_res_rej, _ = HITLSpecificationGapManager.execute_human_action(gap_rej, HITLAction.REJECT, proposal_id="P_REJ")
        self.assertEqual(g_res_rej.resolution_status, GapResolutionStatus.REJECTED)

        # 2. Action: DEFER
        gap_def = SpecificationGap(gap_id="G_DEF", component_id="accel_unit", affected_entity="acc_reg", category=SpecificationGapCategory.UNSPECIFIED_REGISTER_UPDATE, description="Defer test")
        prop_def = SpecificationGapProposal(proposal_id="P_DEF", gap_id="G_DEF", title="Def", proposed_resolution="acc_reg <= 0;", detailed_behavioral_explanation="Def")
        gap_def.proposals = [prop_def]
        g_res_def, _ = HITLSpecificationGapManager.execute_human_action(gap_def, HITLAction.DEFER, proposal_id="P_DEF")
        self.assertEqual(g_res_def.resolution_status, GapResolutionStatus.DEFERRED)

        # 3. Action: REQUEST_ALTERNATIVES
        gap_alt = SpecificationGap(gap_id="G_ALT", component_id="accel_unit", affected_entity="acc_reg", category=SpecificationGapCategory.UNSPECIFIED_REGISTER_UPDATE, description="Alt test")
        prop_alt = SpecificationGapProposal(proposal_id="P_ALT1", gap_id="G_ALT", title="Alt 1", proposed_resolution="if (valid_in) acc_reg <= data_in;", detailed_behavioral_explanation="Alt 1")
        gap_alt.proposals = [prop_alt]
        g_res_alt, _ = HITLSpecificationGapManager.execute_human_action(gap_alt, HITLAction.REQUEST_ALTERNATIVES, proposal_id="P_ALT1", efs_ir=self.efs_ir)
        self.assertEqual(g_res_alt.resolution_status, GapResolutionStatus.PROPOSED)
        self.assertIn("P_ALT1", g_res_alt.metadata.get("rejected_proposal_ids", []))

        # 4. Action: REQUEST_CLARIFICATION
        gap_clar = SpecificationGap(gap_id="G_CLAR", component_id="accel_unit", affected_entity="acc_reg", category=SpecificationGapCategory.UNSPECIFIED_REGISTER_UPDATE, description="Clar test")
        g_res_clar, _ = HITLSpecificationGapManager.execute_human_action(gap_clar, HITLAction.REQUEST_CLARIFICATION, rationale="Register updates only when load_enable is 1")
        self.assertEqual(g_res_clar.resolution_status, GapResolutionStatus.PROPOSED)
        self.assertTrue(len(g_res_clar.metadata.get("human_clarifications", [])) > 0)

        # 5. Action: EDIT
        gap_edit = SpecificationGap(gap_id="G_EDIT", component_id="accel_unit", affected_entity="data_out", category=SpecificationGapCategory.MISSING_SIGNAL_DRIVER, description="Edit test")
        prop_edit = SpecificationGapProposal(proposal_id="P_ED", gap_id="G_EDIT", title="Original", proposed_resolution="assign data_out = 0;", detailed_behavioral_explanation="Original")
        gap_edit.proposals = [prop_edit]
        g_res_edit, ir_edit = HITLSpecificationGapManager.execute_human_action(gap_edit, HITLAction.EDIT, proposal_id="P_ED", edited_resolution="assign data_out = data_in & 32'hFFFF;", efs_ir=self.efs_ir)
        self.assertEqual(g_res_edit.resolution_status, GapResolutionStatus.APPROVED)
        self.assertEqual(g_res_edit.decision.edited_resolution, "assign data_out = data_in & 32'hFFFF;")

        # 6. Action: APPROVE
        gap_app = SpecificationGap(gap_id="G_APP", component_id="accel_unit", affected_entity="cfg_reg", category=SpecificationGapCategory.UNSPECIFIED_REGISTER_UPDATE, description="Approve test")
        prop_app = SpecificationGapProposal(proposal_id="P_APP", gap_id="G_APP", title="App", proposed_resolution="cfg_reg <= data_in;", detailed_behavioral_explanation="App")
        gap_app.proposals = [prop_app]
        g_res_app, ir_app = HITLSpecificationGapManager.execute_human_action(gap_app, HITLAction.APPROVE, proposal_id="P_APP", efs_ir=ir_edit)
        self.assertEqual(g_res_app.resolution_status, GapResolutionStatus.APPROVED)

    def test_24_driver_accounting_and_elimination_of_silent_zero_tie_offs(self):
        """Scenario 24: Verifies driver accounting map and elimination of silent zero tie-offs for unresolved outputs."""
        plan_dict = {
            "module_name": "driver_audit_mod",
            "target_language": "Verilog",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "data_in", "direction": "input", "width": "32"},
                {"name": "driven_out", "direction": "output", "width": "32"},
                {"name": "unresolved_out", "direction": "output", "width": "32"},
            ],
            "combinational_logic": [
                {
                    "process_type": "ALWAYS_COMB",
                    "assignments": [
                        {
                            "target": "driven_out",
                            "expression_raw": "data_in",
                            "origin": "HUMAN_APPROVED"
                        }
                    ]
                }
            ],
            "unresolved_logic": [
                {
                    "affected_entity": "unresolved_out",
                    "description": "Output port 'unresolved_out' has no specified driving logic.",
                    "severity": "WARNING"
                }
            ]
        }
        code, report = generate_hybrid_rtl(
            design_spec="Module with driven_out and unresolved_out",
            target="Verilog",
            rtl_plan=plan_dict,
            mode=GenerationMode.DETERMINISTIC
        )

        # Silent tie off (assign unresolved_out = 32'd0;) MUST NOT BE PRESENT
        self.assertNotIn("assign unresolved_out = 32'd0;", code)
        self.assertNotIn("assign unresolved_out = 0;", code)
        # Diagnostic comment must be present
        self.assertIn("// [UNRESOLVED_OUTPUT] Output port 'unresolved_out'", code)

        # Driver accounting checks
        driver_acc = report.get("driver_accounting", {})
        self.assertIn("driven_out", driver_acc)
        self.assertEqual(driver_acc["driven_out"]["driver_type"], "CONTINUOUS_ASSIGN")
        self.assertEqual(driver_acc["driven_out"]["validation_status"], "DRIVEN")

        self.assertIn("unresolved_out", driver_acc)
        self.assertEqual(driver_acc["unresolved_out"]["driver_type"], "UNRESOLVED")
        self.assertEqual(driver_acc["unresolved_out"]["validation_status"], "UNDRIVEN")
        self.assertEqual(report.get("generation_status"), "PARTIALLY_IMPLEMENTED")

    def test_25_rtl_contract_validator_multiple_drivers(self):
        """Scenario 25: RTLContractValidator flags MULTIPLE_DRIVERS violation when output has multiple drivers."""
        from core.verification.rtl_contract_validator import RTLContractValidator, ContractViolationType

        plan_dict = {
            "module_name": "multi_driver_mod",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "a", "direction": "input", "width": "1"},
                {"name": "b", "direction": "input", "width": "1"},
                {"name": "out_sig", "direction": "output", "width": "1"},
            ]
        }

        # Code with duplicate continuous assign statements driving out_sig
        buggy_code = """
module multi_driver_mod (
    input wire clk,
    input wire a,
    input wire b,
    output wire out_sig
);
    assign out_sig = a;
    assign out_sig = b;
endmodule
"""
        validator = RTLContractValidator()
        report = validator.validate_contract(buggy_code, plan_dict)
        self.assertFalse(report.is_valid)
        violation_types = [v.violation_type for v in report.contract_violations]
        self.assertIn(ContractViolationType.MULTIPLE_DRIVERS, violation_types)

    def test_26_end_to_end_flow_with_iverilog_verification(self):
        """Scenario 26: Full end-to-end flow with iverilog verification for resolved vs unresolved module."""
        # Clean fully approved plan
        approved_plan = {
            "module_name": "e2e_unit",
            "target_language": "Verilog",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "rst_n", "direction": "input", "width": "1"},
                {"name": "din", "direction": "input", "width": "8"},
                {"name": "dout", "direction": "output", "width": "8"},
            ],
            "internal_registers": [
                {"name": "q_reg", "width": "8", "reset_val": "0"}
            ],
            "sequential_logic": [
                {
                    "clock_signal": "clk",
                    "clock_edge": "POSEDGE",
                    "reset_signal": "rst_n",
                    "reset_active_level": "LOW",
                    "reset_type": "ASYNC",
                    "reset_actions": {"q_reg": "8'd0"},
                    "register_updates": [
                        {
                            "target": "q_reg",
                            "next_value": "din",
                            "condition": "1'b1",
                            "origin": "HUMAN_APPROVED"
                        }
                    ]
                }
            ],
            "combinational_logic": [
                {
                    "assignments": [
                        {
                            "target": "dout",
                            "expression_raw": "q_reg",
                            "origin": "HUMAN_APPROVED"
                        }
                    ]
                }
            ]
        }
        code, report = generate_hybrid_rtl(
            design_spec="Clean registered unit with human-approved updates",
            target="Verilog",
            rtl_plan=approved_plan,
            mode=GenerationMode.DETERMINISTIC
        )
        self.assertEqual(report.get("generation_status"), "FULLY_IMPLEMENTED")
        res = _compile_with_iverilog(code)
        self.assertEqual(res.returncode, 0, f"iverilog failed to compile valid RTL:\n{code}\nError: {res.stderr}")


if __name__ == "__main__":
    unittest.main()
