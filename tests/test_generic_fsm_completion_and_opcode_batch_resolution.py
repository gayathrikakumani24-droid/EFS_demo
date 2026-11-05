"""
Targeted Regression Suite for Generic FSM Completion Semantics and Opcode Batch Resolution.

Tests the 7 critical architectural behaviors:
1. FSM with internal completion event.
2. FSM with external completion output (circularity broken).
3. Multi-cycle operation with known latency.
4. Protocol-controlled completion.
5. Unresolved completion semantics.
6. Batch opcode collision resolution.
7. Intentional instruction alias preservation.
8. RTLPlanValidator enhanced diagnostics (FSM_CIRCULAR_DEPENDENCY, FSM_UNREACHABLE_STATE, etc.)
"""

import pytest
from core.efs_ir.models import EFSIR, EFSSignal, EFSOpcode, EFSInstruction, EFSFSM, EFSState, EFSTransition, EFSConflict, EFSResolution
from core.efs_ir.rtl_plan import PlanFSM, PlanFSMTransition, EvidenceClassification
from core.agents.fsm_inference_engine import FSMInferenceEngine, FSMHandlingCase
from core.agents.rtl_plan_validator import RTLPlanValidator, DiagnosticCode
from core.verification.conflict_resolution_engine import ConflictResolutionEngine


class TestGenericFSMCompletionAndOpcodeBatchResolution:

    # --------------------------------------------------------------------------
    # CASE 1: FSM with internal completion event
    # --------------------------------------------------------------------------
    def test_fsm_with_internal_completion_event(self):
        """Internal completion event triggers DONE transition, distinct from observable output done."""
        efs_ir = EFSIR()
        efs_ir.signals = [
            EFSSignal(name="clk", direction="input", width="1"),
            EFSSignal(name="rst_n", direction="input", width="1"),
            EFSSignal(name="start", direction="input", width="1"),
            EFSSignal(name="done", direction="output", width="1"),
            EFSSignal(name="compute_done", direction="internal", width="1"),
        ]
        explicit_fsm = {
            "states": [{"name": "IDLE"}, {"name": "EXECUTE"}, {"name": "DONE"}],
            "initial_state": "IDLE",
            "transitions": [
                {"from_state": "IDLE", "to_state": "EXECUTE", "condition": "start == 1'b1"},
                {"from_state": "EXECUTE", "to_state": "DONE", "condition": "compute_done == 1'b1"},
                {"from_state": "DONE", "to_state": "IDLE", "condition": "!start"},
            ]
        }
        res = FSMInferenceEngine.resolve_fsm(efs_ir=efs_ir, context_pack={"target_fsm": explicit_fsm})
        assert res.case == FSMHandlingCase.EXPLICIT_FSM
        assert res.fsm is not None
        
        # Check transitions
        tr_exec_done = next(t for t in res.fsm.transitions if t.from_state == "EXECUTE" and t.to_state == "DONE")
        assert tr_exec_done.condition_expression == "compute_done == 1'b1"
        assert tr_exec_done.trigger_type in ("internal_completion", "condition")
        
        # Validate plan passes without circularity
        plan_dict = {
            "module_name": "test_unit",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "rst_n", "direction": "input", "width": "1"},
                {"name": "start", "direction": "input", "width": "1"},
                {"name": "done", "direction": "output", "width": "1"},
            ],
            "fsm": res.fsm.to_dict(),
            "output_driving_strategy": {"done": "state == DONE"},
        }
        validator = RTLPlanValidator(efs_ir=efs_ir)
        report = validator.validate_plan(plan_dict)
        circ_diags = [d for d in report.diagnostics if d.diagnostic_type == DiagnosticCode.FSM_CIRCULAR_DEPENDENCY]
        assert len(circ_diags) == 0

    # --------------------------------------------------------------------------
    # CASE 2: FSM with external completion output (circularity broken)
    # --------------------------------------------------------------------------
    def test_fsm_with_external_completion_output_circularity_broken(self):
        """When specification states 'done' as completion condition, circularity is broken and never conditions on output port."""
        efs_ir = EFSIR()
        efs_ir.signals = [
            EFSSignal(name="clk", direction="input", width="1"),
            EFSSignal(name="rst_n", direction="input", width="1"),
            EFSSignal(name="start", direction="input", width="1"),
            EFSSignal(name="done", direction="output", width="1"),
        ]
        # Raw transition incorrectly used 'done' or 'done asserted'
        explicit_fsm = {
            "states": [{"name": "IDLE"}, {"name": "EXECUTE"}, {"name": "DONE"}],
            "initial_state": "IDLE",
            "transitions": [
                {"from_state": "IDLE", "to_state": "EXECUTE", "condition": "start == 1'b1"},
                {"from_state": "EXECUTE", "to_state": "DONE", "condition": "done asserted"},
                {"from_state": "DONE", "to_state": "IDLE", "condition": "done deasserted"},
            ]
        }
        res = FSMInferenceEngine.resolve_fsm(efs_ir=efs_ir, context_pack={"target_fsm": explicit_fsm})
        assert res.fsm is not None
        
        tr_exec_done = next(t for t in res.fsm.transitions if t.from_state == "EXECUTE" and t.to_state == "DONE")
        # Condition must NOT be 'done' or 'done asserted'
        assert "done" not in tr_exec_done.condition_expression
        assert tr_exec_done.condition_expression in ("1'b1", "timer_cnt >= 4'd0")
        assert tr_exec_done.trigger_type == "single_cycle_dwell"
        assert tr_exec_done.completion_event == "exec_done"

        # Return transition to IDLE must use !start (initiating input deassertion)
        tr_done_idle = next(t for t in res.fsm.transitions if t.from_state == "DONE" and t.to_state == "IDLE")
        assert "!start" in tr_done_idle.condition_expression or tr_done_idle.condition_expression == "1'b1"

    # --------------------------------------------------------------------------
    # CASE 3: Multi-cycle operation with known latency
    # --------------------------------------------------------------------------
    def test_multi_cycle_operation_with_known_latency(self):
        """Known cycle latency correctly derives timer dwell condition and trigger_type."""
        efs_ir = EFSIR()
        efs_ir.signals = [
            EFSSignal(name="clk", direction="input", width="1"),
            EFSSignal(name="rst_n", direction="input", width="1"),
            EFSSignal(name="start", direction="input", width="1"),
            EFSSignal(name="done", direction="output", width="1"),
        ]
        explicit_fsm = {
            "states": [
                {"name": "IDLE"},
                {"name": "EXECUTE", "duration_cycles": 4},
                {"name": "DONE"}
            ],
            "initial_state": "IDLE",
            "transitions": [
                {"from_state": "IDLE", "to_state": "EXECUTE", "condition": "start == 1'b1"},
                {"from_state": "EXECUTE", "to_state": "DONE", "condition": "operation completes"},
                {"from_state": "DONE", "to_state": "IDLE", "condition": "!start"},
            ]
        }
        res = FSMInferenceEngine.resolve_fsm(efs_ir=efs_ir, context_pack={"target_fsm": explicit_fsm})
        assert res.fsm is not None

        tr_exec_done = next(t for t in res.fsm.transitions if t.from_state == "EXECUTE" and t.to_state == "DONE")
        assert "timer_cnt" in tr_exec_done.condition_expression
        assert tr_exec_done.trigger_type == "timer_dwell"
        assert tr_exec_done.completion_event == "timer_expired"

    # --------------------------------------------------------------------------
    # CASE 4: Protocol-controlled completion
    # --------------------------------------------------------------------------
    def test_protocol_controlled_completion(self):
        """Protocol-controlled bus interface uses protocol handshake for completion trigger."""
        efs_ir = EFSIR()
        efs_ir.signals = [
            EFSSignal(name="clk", direction="input", width="1"),
            EFSSignal(name="rst_n", direction="input", width="1"),
            EFSSignal(name="start", direction="input", width="1"),
            EFSSignal(name="done", direction="output", width="1"),
            EFSSignal(name="axi_bvalid", direction="input", width="1"),
            EFSSignal(name="axi_bready", direction="output", width="1"),
        ]
        explicit_fsm = {
            "states": [{"name": "IDLE"}, {"name": "EXECUTE"}, {"name": "DONE"}],
            "initial_state": "IDLE",
            "transitions": [
                {"from_state": "IDLE", "to_state": "EXECUTE", "condition": "start == 1'b1"},
                {"from_state": "EXECUTE", "to_state": "DONE", "condition": "done asserted"},
                {"from_state": "DONE", "to_state": "IDLE", "condition": "!start"},
            ]
        }
        res = FSMInferenceEngine.resolve_fsm(efs_ir=efs_ir, context_pack={"target_fsm": explicit_fsm})
        assert res.fsm is not None

        tr_exec_done = next(t for t in res.fsm.transitions if t.from_state == "EXECUTE" and t.to_state == "DONE")
        assert tr_exec_done.trigger_type == "protocol_handshake"
        assert "axi_bvalid" in tr_exec_done.condition_expression
        assert "axi_bready" in tr_exec_done.condition_expression

    # --------------------------------------------------------------------------
    # CASE 5: Unresolved completion semantics
    # --------------------------------------------------------------------------
    def test_unresolved_completion_semantics_flagged(self):
        """Completely empty/ambiguous transitions without latency or control signals are flagged as incomplete."""
        efs_ir = EFSIR()
        efs_ir.signals = [
            EFSSignal(name="clk", direction="input", width="1"),
            EFSSignal(name="rst_n", direction="input", width="1"),
        ]
        explicit_fsm = {
            "states": [{"name": "S0"}, {"name": "S1"}],
            "initial_state": "S0",
            "transitions": [
                {"from_state": "S0", "to_state": "S1", "condition": "undefined"},
                {"from_state": "S1", "to_state": "S0", "condition": ""},
            ]
        }
        res = FSMInferenceEngine.resolve_fsm(efs_ir=efs_ir, context_pack={"target_fsm": explicit_fsm})
        assert res.case == FSMHandlingCase.INSUFFICIENT_INFORMATION
        assert res.requires_hitl_approval is True
        assert len(res.missing_requirements) >= 1

    # --------------------------------------------------------------------------
    # CASE 6: Collision-Aware Batch Opcode Allocation
    # --------------------------------------------------------------------------
    def test_batch_opcode_collision_resolution(self):
        """Batch resolution proposals across multiple duplicate opcode conflicts generate distinct encodings."""
        efs_ir = EFSIR()
        efs_ir.opcodes = [
            EFSOpcode(mnemonic="ADD", binary_encoding="000001"),
            EFSOpcode(mnemonic="SUB", binary_encoding="000010"),
            EFSOpcode(mnemonic="DUP1_A", binary_encoding="000011"),
            EFSOpcode(mnemonic="DUP1_B", binary_encoding="000011"),
            EFSOpcode(mnemonic="DUP2_A", binary_encoding="000100"),
            EFSOpcode(mnemonic="DUP2_B", binary_encoding="000100"),
            EFSOpcode(mnemonic="DUP3_A", binary_encoding="000101"),
            EFSOpcode(mnemonic="DUP3_B", binary_encoding="000101"),
        ]
        conflicts = ConflictResolutionEngine.detect_conflicts(efs_ir)
        assert len(conflicts) == 3

        batch_proposals = ConflictResolutionEngine.generate_batch_resolution_proposals(conflicts, efs_ir)
        assert len(batch_proposals) == 3

        # Collect first candidate value for each conflict
        cand_values = []
        for conf_id, props in batch_proposals.items():
            assert len(props) >= 1
            cand_values.append(props[0].proposed_value)

        # All candidate values must be strictly distinct
        assert len(cand_values) == len(set(cand_values)), f"Duplicate candidate encodings proposed in batch: {cand_values}"

        # Apply batch resolutions and revalidate
        resolved_ir = efs_ir
        for props in batch_proposals.values():
            resolved_ir = ConflictResolutionEngine.apply_approved_resolution(resolved_ir, props[0])
        has_blocking, issues = ConflictResolutionEngine.revalidate_resolved_model(resolved_ir)
        assert not has_blocking
        active_opcode_conflicts = [c for c in resolved_ir.conflicts if c.type == "DUPLICATE_OPCODE_ENCODING" and getattr(c, "status", "") == "ACTIVE"]
        assert len(active_opcode_conflicts) == 0

    # --------------------------------------------------------------------------
    # CASE 7: Intentional instruction alias preservation
    # --------------------------------------------------------------------------
    def test_intentional_instruction_alias_preservation(self):
        """Instructions explicitly defined as aliases or pseudo-instructions sharing opcodes are preserved without conflict."""
        efs_ir = EFSIR()
        efs_ir.opcodes = [
            EFSOpcode(mnemonic="NOP", binary_encoding="000000", is_alias=True, alias_of="ADDI"),
            EFSOpcode(mnemonic="ADDI", binary_encoding="000000"),
            EFSOpcode(mnemonic="MV", binary_encoding="000001", operation="Pseudo-instruction alias of ADD"),
            EFSOpcode(mnemonic="ADD", binary_encoding="000001"),
        ]
        conflicts = ConflictResolutionEngine.detect_conflicts(efs_ir)
        opcode_conflicts = [c for c in conflicts if c.type == "DUPLICATE_OPCODE_ENCODING"]
        assert len(opcode_conflicts) == 0, f"False positive opcode conflicts on intentional aliases: {opcode_conflicts}"

    # --------------------------------------------------------------------------
    # CASE 8: RTLPlanValidator Enhanced Diagnostics
    # --------------------------------------------------------------------------
    def test_rtl_plan_validator_diagnostics(self):
        """RTLPlanValidator properly catches FSM_CIRCULAR_DEPENDENCY, FSM_UNREACHABLE_STATE, FSM_INVALID_STATE, FSM_UNINTENDED_TERMINAL_STATE."""
        validator = RTLPlanValidator()

        # 8a: Circular Dependency
        plan_circular = {
            "module_name": "mod_circ",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "done", "direction": "output", "width": "1"},
            ],
            "fsm": {
                "states": ["IDLE", "EXECUTE", "DONE"],
                "initial_state": "IDLE",
                "terminal_states": ["DONE"],
                "transitions": [
                    {"from": "IDLE", "to": "EXECUTE", "condition": "1'b1"},
                    {"from": "EXECUTE", "to": "DONE", "condition": "done == 1'b1"},
                    {"from": "DONE", "to": "IDLE", "condition": "1'b1"},
                ]
            }
        }
        report = validator.validate_plan(plan_circular)
        types = [d.diagnostic_type for d in report.diagnostics]
        assert DiagnosticCode.FSM_CIRCULAR_DEPENDENCY in types

        # 8b: BFS Unreachable State
        plan_unreachable = {
            "module_name": "mod_unreach",
            "ports": [{"name": "clk", "direction": "input", "width": "1"}],
            "fsm": {
                "states": ["IDLE", "RUN", "ORPHAN_A", "ORPHAN_B"],
                "initial_state": "IDLE",
                "terminal_states": ["RUN"],
                "transitions": [
                    {"from": "IDLE", "to": "RUN", "condition": "1'b1"},
                    {"from": "RUN", "to": "IDLE", "condition": "1'b1"},
                    # Disconnected cycle: ORPHAN_A <-> ORPHAN_B (has incoming, but unreachable from IDLE)
                    {"from": "ORPHAN_A", "to": "ORPHAN_B", "condition": "1'b1"},
                    {"from": "ORPHAN_B", "to": "ORPHAN_A", "condition": "1'b1"},
                ]
            }
        }
        report = validator.validate_plan(plan_unreachable)
        types = [d.diagnostic_type for d in report.diagnostics]
        assert DiagnosticCode.FSM_UNREACHABLE_STATE in types
        unreach_objs = [d.affected_object for d in report.diagnostics if d.diagnostic_type == DiagnosticCode.FSM_UNREACHABLE_STATE]
        assert "ORPHAN_A" in unreach_objs and "ORPHAN_B" in unreach_objs

        # 8c: Invalid State References
        plan_invalid = {
            "module_name": "mod_inv",
            "ports": [{"name": "clk", "direction": "input", "width": "1"}],
            "fsm": {
                "states": ["IDLE", "RUN"],
                "initial_state": "IDLE",
                "terminal_states": ["RUN"],
                "transitions": [
                    {"from": "IDLE", "to": "NONEXISTENT", "condition": "1'b1"},
                ]
            }
        }
        report = validator.validate_plan(plan_invalid)
        types = [d.diagnostic_type for d in report.diagnostics]
        assert DiagnosticCode.FSM_INVALID_STATE in types

        # 8d: Unintended Terminal State (not in terminal_states)
        plan_deadend = {
            "module_name": "mod_dead",
            "ports": [{"name": "clk", "direction": "input", "width": "1"}],
            "fsm": {
                "states": ["IDLE", "STUCK"],
                "initial_state": "IDLE",
                "terminal_states": [],  # STUCK is not registered
                "transitions": [
                    {"from": "IDLE", "to": "STUCK", "condition": "1'b1"},
                ]
            }
        }
        report = validator.validate_plan(plan_deadend)
        types = [d.diagnostic_type for d in report.diagnostics]
        assert DiagnosticCode.FSM_UNINTENDED_TERMINAL_STATE in types

