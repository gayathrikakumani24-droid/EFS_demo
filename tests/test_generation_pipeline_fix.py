"""
Tests for the fixed generation pipeline:
1. Complete hardware specification check before generation (interfaces, ports/signals, modules,
   relationships, transactions, registers, FSM states/transitions, clock/reset, memory interfaces, constraints).
2. Detection that a list of components alone is NOT a complete specification.
3. PUML and RTL generated from the same resolved EFS IR.
4. RTL ContextPack token pruning preserves interfaces, signals, registers, FSM, clock/reset, protocol rules.
5. Idempotent resolution application (same resolution never applied twice).
6. RTL validation failure reports error without restarting pipeline.
"""

import pytest
from core.efs_ir.models import (
    EFSIR, EFSComponent, EFSSignal, EFSInterface, EFSRegister,
    EFSFSM, EFSState, EFSTransition, EFSResolution, EFSConflict,
    EFSTimingRule, EFSConstraint
)
from core.verification.conflict_resolution_engine import ConflictResolutionEngine
from core.retrieval.context_pack import ContextPack, TaskIntent
from core.retrieval.token_budget import TokenBudgetManager as RetrievalTokenBudget
from core.context_compiler.budget import TokenBudgetController as CompilerTokenBudget


class TestCompleteSpecCheck:
    """Tests upfront hardware completeness verification."""

    def test_empty_ir_fails_completeness(self):
        conflicts = ConflictResolutionEngine.validate_hardware_completeness(None)
        assert len(conflicts) > 0
        assert any("No EFS IR Available" in c.title for c in conflicts)

    def test_component_list_alone_is_not_complete_spec(self):
        """A list of component names alone must NOT be treated as a complete specification."""
        ir = EFSIR()
        ir.components = [
            EFSComponent(name="ALU", type="module"),
            EFSComponent(name="RegisterFile", type="module")
        ]
        ir.signals = []
        ir.interfaces = []
        
        conflicts = ConflictResolutionEngine.validate_hardware_completeness(ir)
        assert len(conflicts) > 0
        titles = [c.title for c in conflicts]
        assert "Component List Incomplete Specification" in titles or "Missing Port/Signal Definitions" in titles

    def test_missing_clock_and_reset_reported(self):
        ir = EFSIR()
        ir.components = [EFSComponent(name="Core", type="module")]
        ir.signals = [
            EFSSignal(name="data_in", direction="input", width="32", semantic_role="data")
        ]
        conflicts = ConflictResolutionEngine.validate_hardware_completeness(ir)
        titles = [c.title for c in conflicts]
        assert "Missing Clock Signal" in titles
        assert "Missing Reset Signal" in titles

    def test_missing_inter_module_relationships(self):
        ir = EFSIR()
        ir.components = [
            EFSComponent(name="Master", type="module"),
            EFSComponent(name="Slave", type="module")
        ]
        ir.signals = [
            EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset")
        ]
        ir.interfaces = []  # No interface connecting Master and Slave
        conflicts = ConflictResolutionEngine.validate_hardware_completeness(ir)
        titles = [c.title for c in conflicts]
        assert "Missing Inter-Module Relationships" in titles

    def test_missing_memory_interface_when_memory_component_present(self):
        ir = EFSIR()
        ir.components = [
            EFSComponent(name="SRAM_Controller", type="memory")
        ]
        ir.signals = [
            EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset")
        ]
        conflicts = ConflictResolutionEngine.validate_hardware_completeness(ir)
        titles = [c.title for c in conflicts]
        assert "Missing Memory Interface" in titles

    def test_missing_fsm_when_plan_mentions_state_machine(self):
        ir = EFSIR()
        ir.components = [EFSComponent(name="FSM_Module", type="module")]
        ir.signals = [
            EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset")
        ]
        plan = {"fsm_architecture": {"states": ["IDLE", "RUN"]}}
        conflicts = ConflictResolutionEngine.validate_hardware_completeness(ir, plan=plan)
        titles = [c.title for c in conflicts]
        assert "Missing FSM Definition" in titles


class TestResolutionIdempotency:
    """Tests that resolution application is strictly idempotent."""

    def test_resolution_never_applied_twice(self):
        ir = EFSIR()
        ir.components = [EFSComponent(name="Top", type="module")]
        ir.conflicts = [
            EFSConflict(
                conflict_id="CONF_CLK",
                type="HARDWARE_COMPLETENESS_GAP",
                property="clock",
                blocking=True,
                status="UNRESOLVED",
                resolution_status="UNRESOLVED",
                entities=["Clock_Reset"]
            )
        ]

        resolution = EFSResolution(
            resolution_id="RES_CLK_001",
            conflict_id="CONF_CLK",
            resolution_type="ALLOW_AI",
            proposed_change="clock = clk: input 1",
            proposed_value="clk: input 1",
            affected_requirements=["Clock_Reset"]
        )

        # 1st application
        resolved_1 = ConflictResolutionEngine.apply_approved_resolution(ir, resolution)
        assert "RES_CLK_001" in getattr(resolved_1, "applied_resolution_ids", set())
        assert any(s.name == "clk" for s in resolved_1.signals)
        sig_count_1 = len(resolved_1.signals)
        res_count_1 = len(resolved_1.resolutions)

        # 2nd application with same resolution object
        resolved_2 = ConflictResolutionEngine.apply_approved_resolution(resolved_1, resolution)
        assert len(resolved_2.signals) == sig_count_1
        assert len(resolved_2.resolutions) == res_count_1

        # 3rd application with duplicate resolution ID
        duplicate_res = EFSResolution(
            resolution_id="RES_CLK_001",
            conflict_id="CONF_CLK",
            resolution_type="ALLOW_AI",
            proposed_change="clock = clk: input 1",
            proposed_value="clk: input 1",
            affected_requirements=["Clock_Reset"]
        )
        resolved_3 = ConflictResolutionEngine.apply_approved_resolution(resolved_2, duplicate_res)
        assert len(resolved_3.signals) == sig_count_1
        assert len(resolved_3.resolutions) == res_count_1


class TestTokenPruningPreservation:
    """Tests that RTL ContextPack required fields are preserved during token pruning."""

    def test_context_pack_pruning_preserves_p0_fields(self):
        pack = ContextPack(
            task=TaskIntent(task="RTL_GENERATION", target="Core"),
            interfaces=[{"name": "axi_if", "protocol": "axi4_lite"}],
            signals=[
                {"name": "clk", "direction": "input", "width": "1", "semantic_role": "clock"},
                {"name": "rst_n", "direction": "input", "width": "1", "semantic_role": "reset"},
                {"name": "data_in", "direction": "input", "width": "32", "semantic_role": "data"},
                {"name": "data_out", "direction": "output", "width": "32", "semantic_role": "data"},
                {"name": "aux_ctrl", "direction": "input", "width": "8", "semantic_role": "control"}
            ],
            registers=[{"name": "CTRL", "offset": "0x00", "width": 32}],
            fsm={"states": ["IDLE", "ACTIVE"], "transitions": [{"from": "IDLE", "to": "ACTIVE", "condition": "start"}]},
            protocol_rules=[{"name": "rule1", "text": "Signal must be stable"}],
            source_evidence=[{"text": "Very long narrative " * 200}]
        )

        controller = RetrievalTokenBudget(max_tokens=200)
        pruned = controller.enforce_budget(pack)

        # Interfaces, signals, registers, FSM, protocol_rules must all be preserved
        assert len(pruned.interfaces) == 1
        assert len(pruned.signals) == 5
        assert any(s["name"] == "clk" for s in pruned.signals)
        assert any(s["name"] == "data_in" for s in pruned.signals)
        assert len(pruned.registers) == 1
        assert pruned.fsm["states"] == ["IDLE", "ACTIVE"]
        assert len(pruned.protocol_rules) == 1

    def test_compiler_rtl_budget_preserves_behavioral_fields(self):
        context = {
            "module": {"name": "Core", "type": "module"},
            "ports": [{"name": "clk", "direction": "input", "width": "1"}],
            "signals": [{"name": "clk", "direction": "input", "width": "1"}],
            "clock_reset": [{"name": "clk", "type": "clock"}],
            "registers": [{"name": "CTRL", "offset": "0x00", "width": "32"}],
            "fsm": {"states": ["IDLE", "ACTIVE"], "transitions": []},
            "protocol_rules": [{"text": "Handshake requires valid before ready"}],
            "source_evidence": [{"text": "Extremely verbose narrative evidence text " * 500}]
        }

        pruned, exceeded, warning = CompilerTokenBudget.enforce_rtl_budget(context, max_tokens=100)
        assert "ports" in pruned
        assert "signals" in pruned
        assert "clock_reset" in pruned
        assert "registers" in pruned
        assert "fsm" in pruned
        assert "protocol_rules" in pruned
        assert len(pruned["ports"]) == 1
        assert len(pruned["registers"]) == 1


class TestSingleSourceAndPipelineFlow:
    """Tests single source resolved EFS IR for PUML & RTL, and non-restarting error handling."""

    def test_rtl_validation_failure_reports_error_without_restart(self):
        """If RTL validation fails, show error without restarting the pipeline."""
        from unittest.mock import patch

        ir = EFSIR()
        ir.components = [EFSComponent(name="Core", type="module")]
        ir.signals = [
            EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset"),
            EFSSignal(name="data_out", direction="output", width="32", semantic_role="data")
        ]
        ir.interfaces = [EFSInterface(name="bus_if", protocol="generic")]

        design_results = {
            "status": "AWAITING_HUMAN_DECISION",
            "efs_ir": ir,
            "conflicts": [],
            "requirement_model": {},
            "plan": {},
            "protocol_rules": [],
            "design_spec": "Spec for Core",
            "outputs": {},
            "diagrams": {}
        }

        # Mock RTL verification to report a failure
        mock_failure_report = {
            "compliance_score": 40,
            "failed_checks": [
                {"severity": "CRITICAL", "check_name": "Reset Logic", "violation": "Missing synchronous reset condition."}
            ],
            "passed_checks": [],
            "warnings": []
        }

        with patch("core.agents.verification_agent.verify_code", return_value=mock_failure_report):
            with patch("core.agents.rtl_agent.generate_code", return_value="module Core (input clk, input rst_n); endmodule"):
                with patch("core.agents.plantuml_agent.generate_diagrams", return_value={"architecture": "@startuml\n@enduml"}):
                    with patch("core.agents.spec_agent.generate_design_spec", return_value="# Mock Spec"):
                        from core.agents.orchestrator import resume_design_flow_with_resolutions
                        result = resume_design_flow_with_resolutions(
                            design_results,
                            approved_resolutions=[],
                            bypass_conflicts=True
                        )

                    # Verify that error is reported without restarting the pipeline
                    assert result["status"] == "RTL_VALIDATION_ERROR"
                    assert "Missing synchronous reset condition" in result.get("error_message", "")
                    assert result.get("requires_human_decision") is False
                    assert result.get("is_blocked") is False
                    assert "Verilog" in result.get("outputs", {})
                    assert "architecture" in result.get("diagrams", {})

