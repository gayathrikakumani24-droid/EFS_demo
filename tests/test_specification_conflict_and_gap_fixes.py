"""
Verification Test Suite: Generic Conflict Detection & Specification Gap Proposal Fixes.

Validates that:
1. Signal direction contradictions are detected as EFSConflict.
2. Signal width contradictions are detected as EFSConflict.
3. Register property contradictions on the same register are detected as EFSConflict.
4. Reset polarity contradictions are detected as EFSConflict.
5. Specification gaps for protocol signals (e.g., axi_*, ready, valid) synthesize actionable proposals.
6. Specification gaps for registers (status_reg, cfg_reg, M0-M7) synthesize actionable proposals.
7. Approving a proposal resolves the gap into canonical EFS IR with HUMAN_APPROVED provenance.
"""

import unittest
from core.efs_ir.models import (
    EFSIR,
    EFSSignal,
    EFSRegister,
    EFSOpcode,
    EFSTraceability,
    new_efs_id
)
from core.efs_ir.specification_gap import (
    SpecificationGap,
    SpecificationGapCategory,
    GapSeverity,
    HITLAction,
    GapResolutionStatus
)
from core.efs_ir.signal_normalization import resolve_signal_metadata
from core.verification.conflict_resolution_engine import ConflictResolutionEngine
from core.verification.gap_proposal_engine import GapProposalEngine
from core.verification.hitl_gap_manager import HITLSpecificationGapManager
from core.verification.gap_resolution_applier import GapResolutionApplier


class TestSpecificationConflictAndGapFixes(unittest.TestCase):

    def setUp(self):
        self.efs_ir = EFSIR()

    def test_01_signal_direction_conflict_detected(self):
        """Signal with contradictory input and output declarations produces SIGNAL_DIRECTION conflict."""
        record_in = {"name": "ack_sig", "direction": "input", "description": "Table 1 declares ack_sig as input"}
        record_out = {"name": "ack_sig", "direction": "output", "description": "Section 3 declares ack_sig as output"}
        norm = resolve_signal_metadata([record_in, record_out], signal_name="ack_sig")
        
        self.assertTrue(norm["is_conflicted"])
        self.assertEqual(norm["direction_source"], "conflict")
        self.assertIn("input", norm["competing_directions"])
        self.assertIn("output", norm["competing_directions"])

        sig = EFSSignal(
            name="ack_sig",
            direction=norm["direction"],
            direction_source=norm["direction_source"],
            conflict_details=norm
        )
        self.efs_ir.signals.append(sig)

        conflicts = ConflictResolutionEngine.detect_conflicts(self.efs_ir)
        dir_conf = next((c for c in conflicts if c.type == "SIGNAL_DIRECTION" and "ack_sig" in c.entities), None)
        self.assertIsNotNone(dir_conf, "SIGNAL_DIRECTION conflict should be detected for ack_sig")
        self.assertEqual(dir_conf.property, "direction")

    def test_02_signal_width_conflict_detected(self):
        """Signal with conflicting widths produces CONFLICTING_WIDTH conflict."""
        record_16 = {"name": "data_bus", "width": "16", "description": "Table 2: 16-bit bus"}
        record_32 = {"name": "data_bus", "width": "32", "description": "Table 4: 32-bit bus"}
        norm = resolve_signal_metadata([record_16, record_32], signal_name="data_bus")

        self.assertTrue(norm["is_conflicted"])
        self.assertEqual(norm["width_source"], "conflict")
        self.assertIn("16", norm["competing_widths"])
        self.assertIn("32", norm["competing_widths"])

        sig = EFSSignal(
            name="data_bus",
            width=norm["width"],
            width_source=norm["width_source"],
            conflict_details=norm
        )
        self.efs_ir.signals.append(sig)

        conflicts = ConflictResolutionEngine.detect_conflicts(self.efs_ir)
        w_conf = next((c for c in conflicts if c.type == "CONFLICTING_WIDTH" and "data_bus" in c.entities), None)
        self.assertIsNotNone(w_conf, "CONFLICTING_WIDTH conflict should be detected for data_bus")

    def test_03_register_spec_conflict_detected(self):
        """Contradictory width or reset declarations for the same register produce CONFLICTING_REGISTER_SPEC."""
        reg1 = EFSRegister(name="CTRL_REG", width=16, access="RO", reset_value="0x00")
        reg2 = EFSRegister(name="CTRL_REG", width=32, access="RW", reset_value="0xFF")
        self.efs_ir.registers.extend([reg1, reg2])

        conflicts = ConflictResolutionEngine.detect_conflicts(self.efs_ir)
        reg_conf = next((c for c in conflicts if c.type == "CONFLICTING_REGISTER_SPEC" and "CTRL_REG" in c.entities), None)
        self.assertIsNotNone(reg_conf, "CONFLICTING_REGISTER_SPEC should be detected for CTRL_REG")

    def test_04_reset_polarity_conflict_detected(self):
        """Reset signal with both active-high and active-low declarations produces RESET_POLARITY_CONFLICT."""
        rst_sig = EFSSignal(
            name="rst_n",
            semantic_role="reset",
            description="Section 1 says active-high reset, but pin list says active-low async rst_n"
        )
        self.efs_ir.signals.append(rst_sig)

        conflicts = ConflictResolutionEngine.detect_conflicts(self.efs_ir)
        rst_conf = next((c for c in conflicts if c.type == "RESET_POLARITY_CONFLICT" and "rst_n" in c.entities), None)
        self.assertIsNotNone(rst_conf, "RESET_POLARITY_CONFLICT should be detected for rst_n")

    def test_05_protocol_signals_generate_safe_proposals(self):
        """Standard protocol outputs (axi_arvalid, axi_rready, axi_wstrb, axi_wlast) produce safe proposals."""
        signals_to_test = ["axi_arvalid", "axi_rready", "axi_wstrb", "axi_wlast", "axi_araddr", "axi_wdata"]
        for sig_name in signals_to_test:
            gap = SpecificationGap(
                gap_id=f"GAP_{sig_name.upper()}",
                affected_entity=sig_name,
                category=SpecificationGapCategory.MISSING_SIGNAL_DRIVER,
                description=f"Output port '{sig_name}' is declared but undriven"
            )
            proposals = GapProposalEngine.analyze_and_propose(gap, efs_ir=self.efs_ir)
            self.assertTrue(len(proposals) > 0, f"Proposals should be generated for {sig_name}")
            self.assertTrue(proposals[0].is_safe_proposal, f"Proposal for {sig_name} must be a safe synthesizable proposal")
            self.assertNotEqual(proposals[0].proposed_resolution, "NO_SAFE_PROPOSAL")

    def test_06_unspecified_registers_generate_safe_proposals(self):
        """Unspecified registers (status_reg, cfg_reg, M0) produce safe proposals."""
        regs_to_test = ["status_reg", "cfg_reg", "M0"]
        for reg_name in regs_to_test:
            gap = SpecificationGap(
                gap_id=f"GAP_{reg_name.upper()}",
                affected_entity=reg_name,
                category=SpecificationGapCategory.UNSPECIFIED_REGISTER_UPDATE,
                description=f"Register '{reg_name}' is declared without update condition"
            )
            proposals = GapProposalEngine.analyze_and_propose(gap, efs_ir=self.efs_ir)
            self.assertTrue(len(proposals) > 0, f"Proposals should be generated for {reg_name}")
            self.assertTrue(proposals[0].is_safe_proposal, f"Proposal for {reg_name} must be safe")
            self.assertNotEqual(proposals[0].proposed_resolution, "NO_SAFE_PROPOSAL")

    def test_07_approving_proposal_resolves_gap_with_provenance(self):
        """Approving a proposal incorporates it into canonical EFS IR with HUMAN_APPROVED origin."""
        gap = SpecificationGap(
            gap_id="GAP_DONE_01",
            affected_entity="done",
            category=SpecificationGapCategory.MISSING_SIGNAL_DRIVER,
            description="Output port 'done' is undriven"
        )
        proposals = GapProposalEngine.analyze_and_propose(gap, efs_ir=self.efs_ir)
        selected_prop = proposals[0]

        updated_gap, updated_ir = HITLSpecificationGapManager.execute_human_action(
            gap=gap,
            action=HITLAction.APPROVE,
            proposal_id=selected_prop.proposal_id,
            reviewer="lead_architect",
            efs_ir=self.efs_ir
        )
        self.assertEqual(updated_gap.resolution_status, GapResolutionStatus.APPROVED)
        
        # Verify functional behavior added to canonical EFS IR
        added_fb = next((fb for fb in updated_ir.functional_behavior if fb.subject == "done"), None)
        self.assertIsNotNone(added_fb)
        self.assertEqual(added_fb.origin, "HUMAN_APPROVED")
        self.assertIn("done =", added_fb.action)

        # Verify resolution record
        res = next((r for r in updated_ir.resolutions if r.conflict_id == "GAP_DONE_01"), None)
        self.assertIsNotNone(res)
        self.assertEqual(res.approval_status, "APPROVED")
        self.assertEqual(res.approved_by, "lead_architect")


if __name__ == "__main__":
    unittest.main()
