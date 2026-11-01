"""
Automated Test Suite for FSM Transition-Condition Resolution, Verification, and RTL Correctness.

Validates the 10 requirements from Step 5 of the Task:
1. RED -> GREEN transition condition derivation.
2. GREEN -> YELLOW transition condition derivation.
3. YELLOW -> RED transition condition derivation.
4. Timed and unconditional transitions distinction.
5. Explicit Boolean conditions (relational and signal assertions).
6. Missing timing information handling (never silently replaces with 1'b1).
7. Counter-boundary and off-by-one errors (terminal count N-1 for duration N).
8. Reset and state-entry behavior (cycle counter cleared synchronously).
9. Conflict generation and automatic/AI resolution proposal calculation.
10. Generated RTL correctness and mutual exclusivity validation.
"""

import re
import pytest
from core.efs_ir.models import (
    EFSIR, EFSMetadata, EFSComponent, EFSSignal, EFSRegister,
    EFSFSM, EFSState, EFSTransition, EFSTimingRule, EFSConflict, EFSResolution
)
from core.efs_ir.builder import EFSIRBuilder
from core.efs_ir.repair_engine import EFSStructuralRepairEngine
from core.verification.conflict_resolution_engine import ConflictResolutionEngine
from core.agents.rtl_agent import plan_rtl, validate_rtl_plan


class TestFSMTransitionConditionsAndRTL:

    def test_1_2_3_red_green_yellow_transition_synthesis(self):
        """1-3. RED->GREEN (5 cyc), GREEN->YELLOW (10 cyc), YELLOW->RED (3 cyc) derivation."""
        ir = EFSIR()
        ir.metadata.design_name = "traffic_light_controller"
        ir.components = [EFSComponent(name="traffic_light_controller", type="top_level_design")]
        ir.signals = [
            EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
            EFSSignal(name="reset_n", direction="input", width="1", semantic_role="reset"),
            EFSSignal(name="red_light", direction="output", width="1"),
            EFSSignal(name="green_light", direction="output", width="1"),
            EFSSignal(name="yellow_light", direction="output", width="1"),
        ]

        # Requirement model with explicit timing requirements
        req_model = {
            "timing_requirements": [
                "The RED light remains ON for 5 clock cycles.",
                "The GREEN state remains ON for 10 clock cycles.",
                "The YELLOW state remains ON for 3 clock cycles."
            ],
            "fsm_info": {
                "name": "traffic_light_fsm",
                "states": [
                    {"name": "RED", "duration_cycles": 5},
                    {"name": "GREEN", "duration_cycles": 10},
                    {"name": "YELLOW", "duration_cycles": 3}
                ],
                "initial_state": "RED",
                "transitions": [
                    {"from": "RED", "to": "GREEN"},
                    {"from": "GREEN", "to": "YELLOW"},
                    {"from": "YELLOW", "to": "RED"}
                ]
            }
        }

        builder = EFSIRBuilder(req_model=req_model)
        built_ir = builder.build()

        # Execute structural repair
        repaired_ir, log, issues = EFSStructuralRepairEngine.repair_and_validate(built_ir, req_model=req_model)

        assert len(repaired_ir.fsms) == 1
        fsm = repaired_ir.fsms[0]
        tr_map = {(t.source_state, t.target_state): t for t in fsm.transitions}

        # Terminal count for RED (duration 5): 5 - 1 = 4 -> 4'd4
        tr_rg = tr_map[("RED", "GREEN")]
        assert "timer_cnt == 4'd4" in tr_rg.condition
        assert tr_rg.condition_type == "signal_assertion"
        assert tr_rg.duration_cycles == 5

        # Terminal count for GREEN (duration 10): 10 - 1 = 9 -> 4'd9
        tr_gy = tr_map[("GREEN", "YELLOW")]
        assert "timer_cnt == 4'd9" in tr_gy.condition
        assert tr_gy.condition_type == "signal_assertion"
        assert tr_gy.duration_cycles == 10

        # Terminal count for YELLOW (duration 3): 3 - 1 = 2 -> 4'd2
        tr_yr = tr_map[("YELLOW", "RED")]
        assert "timer_cnt == 4'd2" in tr_yr.condition
        assert tr_yr.condition_type == "signal_assertion"
        assert tr_yr.duration_cycles == 3

        # Cycle counter register must be synthesized
        timer_reg = next((r for r in repaired_ir.registers if r.name == "timer_cnt"), None)
        assert timer_reg is not None
        assert timer_reg.width >= 4

        # Conflict detector must NOT flag these explicit synthesized guards
        conflicts = ConflictResolutionEngine.detect_conflicts(repaired_ir)
        tc_conflicts = [c for c in conflicts if c.property == "transition_condition"]
        assert len(tc_conflicts) == 0

    def test_4_timed_vs_unconditional_transitions(self):
        """4. Distinguishes valid unconditional transitions (duration 1 or None) from timed transitions."""
        ir = EFSIR()
        ir.metadata.design_name = "pipeline_fsm"
        ir.components = [EFSComponent(name="pipeline_fsm", type="top_level")]
        ir.signals = [
            EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset")
        ]

        fsm = EFSFSM(
            name="pipeline_fsm",
            states=[
                EFSState(name="INIT"),
                EFSState(name="RUN", duration_cycles=4),
                EFSState(name="CLEANUP")
            ],
            transitions=[
                # Unconditional transition without duration requirement: valid 1'b1
                EFSTransition(source_state="INIT", target_state="RUN", condition="1'b1", condition_type="unconditional"),
                # Timed transition: must have timer guard
                EFSTransition(source_state="RUN", target_state="CLEANUP", condition="timer_cnt == 2'd3", condition_type="signal_assertion", duration_cycles=4),
                # Single-cycle transition
                EFSTransition(source_state="CLEANUP", target_state="INIT", condition="1'b1", condition_type="unconditional", duration_cycles=1)
            ]
        )
        ir.fsms.append(fsm)

        conflicts = ConflictResolutionEngine.detect_conflicts(ir)
        tc_conflicts = [c for c in conflicts if c.property == "transition_condition"]
        # Neither unconditional 1'b1 nor synthesized timer_cnt guard should trigger conflicts
        assert len(tc_conflicts) == 0

    def test_5_explicit_boolean_conditions(self):
        """5. Explicit Boolean conditions with relational operators or signals are accepted without conflict."""
        ir = EFSIR()
        ir.metadata.design_name = "handshake_fsm"
        ir.components = [EFSComponent(name="handshake_fsm", type="top_level")]
        ir.signals = [
            EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset"),
            EFSSignal(name="start", direction="input", width="1"),
            EFSSignal(name="done", direction="input", width="1")
        ]

        fsm = EFSFSM(
            name="handshake_fsm",
            states=[EFSState(name="IDLE"), EFSState(name="BUSY")],
            transitions=[
                EFSTransition(source_state="IDLE", target_state="BUSY", condition="start == 1'b1", condition_type="signal_assertion", associated_signal="start"),
                EFSTransition(source_state="BUSY", target_state="IDLE", condition="done == 1'b1", condition_type="boolean")
            ]
        )
        ir.fsms.append(fsm)

        conflicts = ConflictResolutionEngine.detect_conflicts(ir)
        tc_conflicts = [c for c in conflicts if c.property == "transition_condition"]
        assert len(tc_conflicts) == 0

    def test_6_missing_timing_information_never_silently_replaces_with_1b1(self):
        """6. Missing/unresolved condition is not replaced with 1'b1 and does NOT create a blocking conflict in conflict queue."""
        ir = EFSIR()
        ir.metadata.design_name = "unresolved_fsm"
        ir.components = [EFSComponent(name="unresolved_fsm", type="top_level")]
        ir.signals = [
            EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset")
        ]

        # Spec states that STATE_A has undefined duration and unknown condition
        req_model = {
            "fsm_info": {
                "name": "unresolved_fsm",
                "states": [
                    {"name": "STATE_A"},
                    {"name": "STATE_B"}
                ],
                "transitions": [
                    {"from": "STATE_A", "to": "STATE_B"}  # No condition provided
                ]
            }
        }

        builder = EFSIRBuilder(req_model=req_model)
        built_ir = builder.build()

        tr = built_ir.fsms[0].transitions[0]
        # Must NOT be 1'b1, must be preserved as unresolved/empty
        assert tr.condition != "1'b1"
        assert tr.condition == ""
        assert tr.condition_type == "unresolved"

        # Policy: missing/unspecified transition condition must NOT create a blocking conflict
        conflicts = ConflictResolutionEngine.detect_conflicts(built_ir)
        blocking_tc_conflicts = [c for c in conflicts if c.blocking and c.property == "transition_condition"]
        assert len(blocking_tc_conflicts) == 0

        # RTL-generation validation must safely report unresolved transitions rather than source conflicts
        plan = {
            "fsm": {
                "states": ["STATE_A", "STATE_B"],
                "transitions": [{"from": "STATE_A", "to": "STATE_B", "condition": ""}]
            }
        }
        val = validate_rtl_plan(plan, efs_ir=built_ir)
        assert val["valid"] is False
        fsm_issues = [i for i in val["issues"] if i.get("category") == "FSM"]
        assert len(fsm_issues) >= 1
        assert any("undefined" in i["issue"].lower() for i in fsm_issues)

    def test_7_counter_boundary_and_off_by_one_timing(self):
        """7. Counter boundary semantics: duration N occupies cycles 0 to N-1; terminal count is N-1."""
        ir = EFSIR()
        ir.metadata.design_name = "boundary_fsm"
        ir.components = [EFSComponent(name="boundary_fsm", type="top_level")]
        ir.signals = [
            EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset")
        ]

        durations = [1, 2, 7, 16]
        expected_term_vals = [0, 1, 6, 15]

        states = [EFSState(name=f"S_{d}", duration_cycles=d) for d in durations]
        transitions = [
            EFSTransition(source_state=f"S_{durations[i]}", target_state=f"S_{durations[(i+1)%len(durations)]}", condition="1'b1", condition_type="semantic")
            for i in range(len(durations))
        ]

        fsm = EFSFSM(name="boundary_fsm", states=states, transitions=transitions)
        ir.fsms.append(fsm)

        repaired_ir, _, _ = EFSStructuralRepairEngine.repair_and_validate(ir)

        for i, d in enumerate(durations):
            tr = next(t for t in repaired_ir.fsms[0].transitions if t.source_state == f"S_{d}")
            exp_term = expected_term_vals[i]
            assert f"timer_cnt ==" in tr.condition
            assert str(exp_term) in tr.condition
            assert tr.duration_cycles == d

    def test_8_reset_and_state_entry_behavior_constraints(self):
        """8. Architectural constraints explicitly capture reset clear and cycle counter behavior."""
        ir = EFSIR()
        ir.metadata.design_name = "reset_test"
        ir.components = [EFSComponent(name="reset_test", type="top_level")]
        ir.signals = [
            EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset")
        ]
        fsm = EFSFSM(
            name="reset_fsm",
            states=[EFSState(name="A", duration_cycles=5), EFSState(name="B", duration_cycles=5)],
            transitions=[
                EFSTransition(source_state="A", target_state="B", condition="1'b1", condition_type="semantic"),
                EFSTransition(source_state="B", target_state="A", condition="1'b1", condition_type="semantic")
            ]
        )
        ir.fsms.append(fsm)

        repaired_ir, _, _ = EFSStructuralRepairEngine.repair_and_validate(ir)

        # Check recorded constraints
        c_types = {c.type for c in repaired_ir.constraints}
        assert "timing_off_by_one" in c_types
        assert "reset_behavior" in c_types
        assert "counter_clear_timing" in c_types
        assert "output_decoding" in c_types

    def test_9_conflict_generation_and_ai_proposal_candidates(self):
        """9. AI proposal candidates are generated for transition_condition conflicts, avoiding 'AI proposal unavailable'."""
        ir = EFSIR()
        ir.metadata.design_name = "traffic_light"
        ir.components = [EFSComponent(name="traffic_light", type="top_level")]
        ir.signals = [
            EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset")
        ]
        fsm = EFSFSM(
            name="traffic_light_fsm",
            states=[
                EFSState(name="RED", duration_cycles=5),
                EFSState(name="GREEN", duration_cycles=10)
            ],
            transitions=[
                EFSTransition(source_state="RED", target_state="GREEN", condition="1'b1", condition_type="semantic")
            ]
        )
        ir.fsms.append(fsm)

        # Deliberately detect conflicts on un-synthesized IR
        conflicts = ConflictResolutionEngine.detect_conflicts(ir)
        tc_conflicts = [c for c in conflicts if c.property == "transition_condition"]
        assert len(tc_conflicts) == 1
        conf = tc_conflicts[0]

        # Calculate concrete candidates for AI proposal
        candidates = ConflictResolutionEngine.calculate_concrete_candidates(conf, ir, count=2)
        assert len(candidates) >= 1
        proposed_val, proposed_change, _ = candidates[0]

        # Candidate must propose explicit timer guard
        assert "timer_cnt == 4'd4" in proposed_val

        # Generate structured resolution proposals
        proposals = ConflictResolutionEngine.generate_resolution_proposals(conf, ir)
        assert len(proposals) >= 1
        top_proposal = proposals[0]
        assert top_proposal.proposed_value == "timer_cnt == 4'd4"
        assert "cycle-accurate RTL synthesis" in top_proposal.rationale

        # Applying resolution updates in-memory IR and overlay
        resolved_ir = ConflictResolutionEngine.apply_approved_resolution(ir, top_proposal)
        tr = resolved_ir.fsms[0].transitions[0]
        assert tr.condition == "timer_cnt == 4'd4"

    def test_10_rtl_plan_generation_and_mutual_exclusivity(self):
        """10. RTL plan validation accepts synthesized guards and enforces mutual exclusivity."""
        ir = EFSIR()
        ir.metadata.design_name = "traffic_light_controller"
        ir.components = [EFSComponent(name="traffic_light_controller", type="top_level")]
        ir.signals = [
            EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
            EFSSignal(name="reset_n", direction="input", width="1", semantic_role="reset"),
            EFSSignal(name="red_light", direction="output", width="1"),
            EFSSignal(name="green_light", direction="output", width="1"),
            EFSSignal(name="yellow_light", direction="output", width="1"),
        ]

        spec_text = (
            "Traffic light controller with states RED (5 cycles), GREEN (10 cycles), YELLOW (3 cycles). "
            "Only one traffic light should be ON at any given time."
        )

        fsm = EFSFSM(
            name="tlc_fsm",
            states=[
                EFSState(name="RED", duration_cycles=5),
                EFSState(name="GREEN", duration_cycles=10),
                EFSState(name="YELLOW", duration_cycles=3)
            ],
            transitions=[
                EFSTransition(source_state="RED", target_state="GREEN", condition="timer_cnt == 4'd4", condition_type="signal_assertion"),
                EFSTransition(source_state="GREEN", target_state="YELLOW", condition="timer_cnt == 4'd9", condition_type="signal_assertion"),
                EFSTransition(source_state="YELLOW", target_state="RED", condition="timer_cnt == 4'd2", condition_type="signal_assertion")
            ]
        )
        ir.fsms.append(fsm)

        # Stage 1: Plan RTL
        plan = plan_rtl(design_spec=spec_text, target="Verilog", efs_ir=ir)
        assert isinstance(plan, dict)
        assert "fsm" in plan
        assert len(plan["fsm"]["transitions"]) == 3

        # Validation must pass without blocking issues
        val = validate_rtl_plan(plan, efs_ir=ir)
        assert val["valid"] is True
        assert val["status"] == "COMPLETE"
        assert len(val["issues"]) == 0
