"""
Comprehensive Integration & Regression Test Suite for Generic FSM Inference and EFS IR Pipeline.

Validates the 10 requirements from Section 7 of the task:
1. An explicitly declared FSM is correctly represented in EFS IR.
2. An FSM implied by behavioral requirements is inferred without requiring the literal term 'FSM'.
3. State and transition entities have valid canonical references.
4. Timing, guards, reset behavior, and control signals are preserved when supported by the source.
5. A combinational design does not receive a fabricated FSM.
6. A sequential design without sufficient evidence produces an appropriate unresolved diagnostic.
7. An FSM mentioned in the design plan is not treated as successfully constructed unless valid FSM entities exist.
8. Inference runs before final completeness validation.
9. Repeated compilation is idempotent and does not duplicate entities.
10. Existing structural validation and RTL/PlantUML generation continue to work.
"""

import json
import os
import pytest
from core.efs_ir.models import (
    EFSIR, EFSMetadata, EFSComponent, EFSInterface, EFSSignal,
    EFSState, EFSTransition, EFSFSM, EFSTimingRule, EFSTraceability
)
from core.efs_ir.entity_registry import CanonicalEntityRegistry
from core.efs_ir.behavioral_analyzer import GenericBehavioralAnalyzer
from core.efs_ir.repair_engine import EFSStructuralRepairEngine
from core.efs_ir.builder import EFSIRBuilder
from core.verification.conflict_resolution_engine import ConflictResolutionEngine
from core.verification.inference_engine import ControlledInferenceEngine
from core.context_compiler.rtl_context import RTLContextCompiler
from core.context_compiler.diagram_context import DiagramContextCompiler


class TestGenericFSMPipeline:
    """Test suite covering generic behavioral inference, FSM construction, and validation."""

    def test_1_explicitly_declared_fsm_represented_in_ir(self):
        """1. An explicitly declared FSM is correctly represented in EFS IR."""
        ir = EFSIR()
        ir.components = [EFSComponent(name="SeqController", type="module")]
        ir.signals = [
            EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset"),
            EFSSignal(name="grant", direction="output", width="1")
        ]
        
        req_model = {
            "fsm_info": {
                "name": "Arbiter_FSM",
                "states": ["IDLE", "REQUEST", "GRANT"],
                "initial_state": "IDLE",
                "transitions": [
                    {"from": "IDLE", "to": "REQUEST", "condition": "req == 1'b1"},
                    {"from": "REQUEST", "to": "GRANT", "condition": "ack == 1'b1"},
                    {"from": "GRANT", "to": "IDLE", "condition": "done == 1'b1"}
                ]
            }
        }
        
        builder = EFSIRBuilder(req_model=req_model)
        built_ir = builder.build()
        
        assert len(built_ir.fsms) == 1
        fsm = built_ir.fsms[0]
        assert fsm.name == "Arbiter_FSM"
        assert len(fsm.states) == 3
        state_names = [s.name for s in fsm.states]
        assert state_names == ["IDLE", "REQUEST", "GRANT"]
        assert len(fsm.transitions) == 3
        assert fsm.initial_state == "IDLE"

    def test_2_fsm_implied_by_behavioral_requirements_inferred_without_keyword(self):
        """2. An FSM implied by behavioral requirements is inferred without requiring the literal term 'FSM'."""
        ir = EFSIR()
        ir.metadata.design_name = "phase_sequencer"
        ir.components = [EFSComponent(name="phase_sequencer", type="module")]
        ir.signals = [
            EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset"),
            EFSSignal(name="phase_a_active", direction="output", width="1"),
            EFSSignal(name="phase_b_active", direction="output", width="1")
        ]
        # Text avoids the literal word "FSM" or "state machine"
        spec_text = (
            "The phase sequencer cycles through two operational modes. "
            "After reset, the controller enters the PHASE_A mode for 8 clock cycles. "
            "It then switches to the PHASE_B mode for 16 clock cycles, then returns to PHASE_A."
        )
        
        repaired_ir, repair_log, _ = EFSStructuralRepairEngine.repair_and_validate(ir, spec_text=spec_text)
        
        assert len(repaired_ir.fsms) == 1
        inferred_fsm = repaired_ir.fsms[0]
        st_names = [s.name for s in inferred_fsm.states]
        assert "PHASE_A" in st_names
        assert "PHASE_B" in st_names
        assert any(t.source_state == "PHASE_A" and t.target_state == "PHASE_B" for t in inferred_fsm.transitions)

    def test_3_state_and_transition_entities_have_valid_canonical_references(self):
        """3. State and transition entities have valid canonical references registered in registry."""
        ir = EFSIR()
        ir.components = [EFSComponent(name="Engine", type="module")]
        ir.signals = [
            EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset")
        ]
        state_idle = EFSState(name="IDLE")
        state_work = EFSState(name="WORK")
        fsm = EFSFSM(
            name="Engine_FSM",
            states=[state_idle, state_work],
            initial_state="idle",
            transitions=[
                EFSTransition(source_state="idle", target_state="work", condition="go"),
                EFSTransition(source_state="work", target_state="idle", condition="done")
            ]
        )
        ir.fsms.append(fsm)
        
        registry = CanonicalEntityRegistry(ir)
        EFSStructuralRepairEngine._resolve_fsm_references(ir, registry, [])
        
        # Verify IDs were assigned and transitions canonicalized to exact state names
        assert state_idle.state_id.upper().startswith("ST_")
        assert state_work.state_id.upper().startswith("ST_")
        assert fsm.initial_state == "IDLE"
        assert fsm.transitions[0].source_state == "IDLE"
        assert fsm.transitions[0].target_state == "WORK"
        assert registry.canonical_id_for(state_idle.state_id, expected_type="state") == state_idle.state_id

    def test_4_timing_guards_reset_behavior_and_control_signals_preserved(self):
        """4. Timing, guards, reset behavior, and control signals are preserved when supported by source."""
        ir = EFSIR()
        ir.components = [EFSComponent(name="DwellController", type="module")]
        ir.signals = [
            EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset"),
            EFSSignal(name="alert_active", direction="output", width="1"),
            EFSSignal(name="standby_active", direction="output", width="1")
        ]
        ir.timing_rules = [
            EFSTimingRule(name="standby_dwell", cycle_relationship="The STANDBY state remains active for 12 clock cycles.")
        ]
        spec_text = (
            "When reset_n is asserted low, the module enters STANDBY mode. "
            "The STANDBY state remains active for 12 clock cycles, after which it switches to ALERT mode for 4 clock cycles."
        )
        
        repaired_ir, _, _ = EFSStructuralRepairEngine.repair_and_validate(ir, spec_text=spec_text)
        
        assert len(repaired_ir.fsms) == 1
        fsm = repaired_ir.fsms[0]
        assert fsm.initial_state == "STANDBY"
        assert fsm.reset_state == "STANDBY"
        assert "active-low" in fsm.reset_behavior.lower()
        
        # Check mutual exclusivity on control signals
        st_standby = next(s for s in fsm.states if s.name == "STANDBY")
        assert st_standby.outputs.get("standby_active") == "1'b1"
        assert st_standby.outputs.get("alert_active") == "1'b0"

    def test_5_combinational_design_does_not_receive_fabricated_fsm(self):
        """5. A combinational design does not receive a fabricated FSM."""
        ir = EFSIR()
        ir.metadata.design_name = "adder_32bit"
        ir.components = [EFSComponent(name="adder_32bit", type="combinational")]
        ir.signals = [
            EFSSignal(name="a", direction="input", width="32"),
            EFSSignal(name="b", direction="input", width="32"),
            EFSSignal(name="sum", direction="output", width="32"),
            EFSSignal(name="cout", direction="output", width="1")
        ]
        spec_text = "A 32-bit combinational adder that computes sum = a + b and carry out cout without clock or reset."
        
        # Analyzer must recognize it as non-sequential
        is_seq = GenericBehavioralAnalyzer.is_sequential_design(ir, spec_text=spec_text)
        assert is_seq is False
        
        repaired_ir, _, validation_issues = EFSStructuralRepairEngine.repair_and_validate(ir, spec_text=spec_text)
        assert len(repaired_ir.fsms) == 0
        
        # Completeness validation must not raise missing FSM or clock/reset
        conflicts = ConflictResolutionEngine.validate_hardware_completeness(repaired_ir)
        fsm_conflicts = [c for c in conflicts if "fsm" in c.title.lower() or "clock" in c.title.lower()]
        assert len(fsm_conflicts) == 0

    def test_6_sequential_design_without_sufficient_evidence_produces_unresolved_diagnostic(self):
        """6. A sequential design without sufficient evidence produces an appropriate unresolved diagnostic."""
        ir = EFSIR()
        ir.components = [EFSComponent(name="CustomSequencer", type="module")]
        ir.signals = [
            EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset")
        ]
        # Plan claims FSM, but source has NO evidence, NO transitions, NO control signals
        plan = {"fsm_architecture": {"states": ["STAGE_1", "STAGE_2"]}}
        
        conflicts = ConflictResolutionEngine.validate_hardware_completeness(ir, plan=plan)
        titles = [c.title for c in conflicts]
        assert "Missing FSM Definition" in titles
        
        missing_fsm_conf = next(c for c in conflicts if c.title == "Missing FSM Definition")
        assert missing_fsm_conf.requires_human_decision is True
        assert missing_fsm_conf.status == "UNRESOLVED"

    def test_7_fsm_mentioned_in_plan_not_treated_as_constructed_without_entities(self):
        """7. An FSM mentioned in design plan is not treated as successfully constructed unless valid FSM entities exist."""
        ir = EFSIR()
        ir.components = [EFSComponent(name="Core", type="module")]
        ir.signals = [
            EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset")
        ]
        # Incomplete FSM with 0 states
        empty_fsm = EFSFSM(name="Empty_FSM", states=[], transitions=[])
        ir.fsms.append(empty_fsm)
        plan = {"fsm_architecture": {"name": "Empty_FSM"}}
        
        conflicts = ConflictResolutionEngine.validate_hardware_completeness(ir, plan=plan)
        titles = [c.title for c in conflicts]
        assert any("Has No States" in t for t in titles)

    def test_8_inference_runs_before_final_completeness_validation(self):
        """8. Inference runs before final completeness validation."""
        ir = EFSIR()
        ir.metadata.design_name = "handshake_unit"
        ir.components = [EFSComponent(name="handshake_unit", type="module")]
        ir.signals = [
            EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset"),
            EFSSignal(name="send_en", direction="input", width="1"),
            EFSSignal(name="tx_active", direction="output", width="1"),
            EFSSignal(name="rx_active", direction="output", width="1")
        ]
        plan = {
            "fsm_architecture": {
                "states": [
                    {"name": "TX", "actions": "tx_active = 1'b1"},
                    {"name": "RX", "actions": "rx_active = 1'b1"}
                ],
                "transitions": [
                    {"from": "TX", "to": "RX", "condition": "done == 1'b1"},
                    {"from": "RX", "to": "TX", "condition": "ack == 1'b1"}
                ]
            }
        }
        
        # When validate_hardware_completeness is invoked directly, it runs inference first
        conflicts = ConflictResolutionEngine.validate_hardware_completeness(ir, plan=plan)
        assert len(ir.fsms) == 1
        assert "Missing FSM Definition" not in [c.title for c in conflicts]

    def test_9_repeated_compilation_is_idempotent(self):
        """9. Repeated compilation is idempotent and does not duplicate entities."""
        ir = EFSIR()
        ir.metadata.design_name = "timer_device"
        ir.components = [EFSComponent(name="timer_device", type="module")]
        ir.signals = [
            EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset"),
            EFSSignal(name="sig_a", direction="output", width="1"),
            EFSSignal(name="sig_b", direction="output", width="1")
        ]
        ir.timing_rules = [
            EFSTimingRule(name="a_rule", cycle_relationship="SIG_A active for 4 clock cycles"),
            EFSTimingRule(name="b_rule", cycle_relationship="SIG_B active for 8 clock cycles")
        ]
        
        # First repair
        ir1, logs1, _ = EFSStructuralRepairEngine.repair_and_validate(ir)
        fsm_count_1 = len(ir1.fsms)
        state_count_1 = len(ir1.fsms[0].states)
        trans_count_1 = len(ir1.fsms[0].transitions)
        
        # Second repair
        ir2, logs2, _ = EFSStructuralRepairEngine.repair_and_validate(ir1)
        fsm_count_2 = len(ir2.fsms)
        state_count_2 = len(ir2.fsms[0].states)
        trans_count_2 = len(ir2.fsms[0].transitions)
        
        assert fsm_count_1 == fsm_count_2 == 1
        assert state_count_1 == state_count_2
        assert trans_count_1 == trans_count_2
        assert len(logs2) == 0  # Zero delta on subsequent pass

    def test_10_rtl_and_plantuml_generation_consume_validated_efs_ir(self):
        """10. Existing structural validation and RTL/PlantUML generation continue to work."""
        active_ir_path = os.path.join("hardware_graphrag", "data", "efs_ir", "traffic_light_initial_ir.json")
        if not os.path.exists(active_ir_path):
            active_ir_path = os.path.join("hardware_graphrag", "data", "efs_ir", "active_design_ir.json")
        if not os.path.exists(active_ir_path):
            pytest.skip("Traffic light IR fixture not available.")
            
        with open(active_ir_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        ir = EFSIR.from_dict(data)
        
        # Run schema-driven repair engine
        repaired_ir, logs, val_issues = EFSStructuralRepairEngine.repair_and_validate(ir)
        assert len(repaired_ir.fsms) == 1
        fsm = repaired_ir.fsms[0]
        assert [s.name for s in fsm.states] == ["RED", "GREEN", "YELLOW"]
        assert len(fsm.transitions) == 3
        
        # Verify Completeness Validation passes without Missing FSM conflict
        completeness_conflicts = ConflictResolutionEngine.validate_hardware_completeness(repaired_ir)
        titles = [c.title for c in completeness_conflicts]
        assert "Missing FSM Definition" not in titles
        
        # Verify RTL Context Compiler consumes repaired IR
        rtl_bundle = RTLContextCompiler().compile(efs_ir=repaired_ir)
        assert "fsm" in rtl_bundle
        assert rtl_bundle["fsm"]["initial_state"] == "RED"
        assert rtl_bundle["fsm"]["states"] == ["RED", "GREEN", "YELLOW"]
        
        # Verify Diagram Context Compiler consumes repaired IR
        diag_bundle = DiagramContextCompiler().compile(efs_ir=repaired_ir)
        assert "fsm" in diag_bundle
        assert diag_bundle["fsm"]["initial_state"] == "RED"
        diag_states = [s["name"] if isinstance(s, dict) else s for s in diag_bundle["fsm"]["states"]]
        assert diag_states == ["RED", "GREEN", "YELLOW"]
