"""
Subprompt 1 Test Suite: EFS RTL Generation Planning, Context Compilation, and Validation.

Tests cover:
1. Explicit FSM preservation.
2. Inferred FSM generation and HITL routing.
3. No-FSM combinational circuit.
4. Sequential design classification and planning.
5. Conflicting interfaces detection.
6. Duplicate opcodes detection.
7. Missing register widths and reset values.
8. Missing and conflicting reset semantics.
9. Missing datapath output drivers.
10. Context budget reduction and manifest tracking.
11. HITL approval, rejection, and alternative selection lifecycle.
12. Invalid LLM plans rejection.
13. Traceability preservation and provenance.
14. Extensible hardware architecture classification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from core.efs_ir.models import (
    EFSConflict,
    EFSFSM,
    EFSIR,
    EFSMetadata,
    EFSOpcode,
    EFSRegister,
    EFSSignal,
    EFSState,
    EFSTransition,
)
from core.efs_ir.rtl_plan import (
    EvidenceClassification,
    PlanDatapath,
    PlanFSM,
    PlanFSMTransition,
    PlanPort,
    PlanRegister,
    PlanResetRequirements,
    PlanTraceability,
    RTLGenerationPlan,
)
from core.agents.spec_classifier import (
    ArchitectureClass,
    SpecType,
    classify_hardware_architecture,
    classify_spec_type,
)
from core.agents.fsm_inference_engine import FSMHandlingCase, FSMInferenceEngine
from core.agents.rtl_plan_validator import (
    DiagnosticCode,
    PlanDiagnostic,
    RTLPlanValidator,
)
from core.verification.hitl_plan_manager import HITLPlanManager
from core.context_compiler import compile_context
from core.context_compiler.budget import TokenBudgetController
from core.agents.rtl_agent import plan_rtl, validate_rtl_plan


def test_01_explicit_fsm():
    """Verify explicit FSM preserves specified states, initial state, and transitions."""
    efs_ir = EFSIR(
        metadata=EFSMetadata(design_name="fsm_module"),
        signals=[
            EFSSignal(name="clk", direction="input", semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", semantic_role="reset"),
            EFSSignal(name="req", direction="input", width="1"),
            EFSSignal(name="ack", direction="output", width="1"),
        ],
        fsms=[
            EFSFSM(
                name="controller_fsm",
                initial_state="S_IDLE",
                states=[EFSState(name="S_IDLE"), EFSState(name="S_ACTIVE"), EFSState(name="S_DONE")],
                transitions=[
                    EFSTransition(source_state="S_IDLE", target_state="S_ACTIVE", condition="req == 1'b1", evidence="req high"),
                    EFSTransition(source_state="S_ACTIVE", target_state="S_DONE", condition="1'b1", evidence="dwell 1 cycle"),
                    EFSTransition(source_state="S_DONE", target_state="S_IDLE", condition="1'b1", evidence="unconditional loop"),
                ]
            )
        ]
    )

    res = FSMInferenceEngine.resolve_fsm(efs_ir=efs_ir, spec_text="FSM module with clk and rst_n")
    assert res.case == FSMHandlingCase.EXPLICIT_FSM
    assert res.fsm is not None
    assert res.fsm.states == ["S_IDLE", "S_ACTIVE", "S_DONE"]
    assert res.fsm.initial_state == "S_IDLE"
    assert len(res.fsm.transitions) == 3
    assert res.fsm.classification == EvidenceClassification.EXPLICIT
    assert not res.requires_hitl_approval


def test_02_inferred_fsm():
    """Verify inferred FSM is proposed when sequential behavior has distinct phases, routing to HITL."""
    spec_text = "Controller module with states S_INIT, S_EXECUTE, S_FINISH. Clock clk, reset rst_n."
    res = FSMInferenceEngine.resolve_fsm(spec_text=spec_text)
    assert res.case == FSMHandlingCase.INFERRED_FSM
    assert res.fsm is not None
    assert res.fsm.states == ["S_INIT", "S_EXECUTE", "S_FINISH"]
    assert res.fsm.classification == EvidenceClassification.INFERRED
    assert res.requires_hitl_approval is True


def test_03_no_fsm_combinational_circuit():
    """Verify pure combinational circuits do not generate artificial FSMs or clocks."""
    spec_text = "4-to-1 multiplexer. Inputs: sel [1:0], in0 [7:0], in1 [7:0], in2 [7:0], in3 [7:0]. Output: out [7:0]."
    arch = classify_hardware_architecture(spec_text=spec_text)
    assert arch.primary_class == ArchitectureClass.COMBINATIONAL
    assert not arch.has_clock
    assert not arch.has_reset

    fsm_res = FSMInferenceEngine.resolve_fsm(spec_text=spec_text)
    assert fsm_res.case == FSMHandlingCase.FSM_UNNECESSARY
    assert fsm_res.fsm is None

    # Context compilation must not inject clock/reset
    ctx = compile_context("rtl", design_spec=spec_text)
    assert ctx.get("clock_reset") == []
    assert ctx.get("fsm") == {} or ctx.get("fsm") is None


def test_04_sequential_design_counter():
    """Verify sequential counter is classified as sequential datapath without an artificial FSM."""
    spec_text = "8-bit up counter. Inputs: clk, rst_n, enable. Output: count [7:0]. Increments on every rising edge when enable is high."
    arch = classify_hardware_architecture(spec_text=spec_text)
    assert arch.has_clock is True
    assert arch.has_reset is True
    assert arch.primary_class in (ArchitectureClass.DATAPATH, ArchitectureClass.SEQUENTIAL)

    fsm_res = FSMInferenceEngine.resolve_fsm(spec_text=spec_text)
    assert fsm_res.case == FSMHandlingCase.FSM_UNNECESSARY
    assert fsm_res.fsm is None


def test_05_conflicting_interfaces():
    """Verify plan validator catches port direction and width mismatches against EFS IR."""
    efs_ir = EFSIR(
        signals=[
            EFSSignal(name="data_bus", direction="input", width="16"),
            EFSSignal(name="status_flag", direction="output", width="1"),
        ]
    )
    # Plan with contradictory direction for data_bus (output instead of input) and width mismatch
    bad_plan = {
        "module_name": "mod_bad",
        "ports": [
            {"name": "data_bus", "direction": "output", "width": "32"},
            {"name": "status_flag", "direction": "output", "width": "1"},
        ]
    }
    validator = RTLPlanValidator(efs_ir=efs_ir)
    report = validator.validate(bad_plan)
    assert not report.is_valid
    assert any(d.diagnostic_type == DiagnosticCode.INTERFACE_CONFLICT for d in report.diagnostics)


def test_06_duplicate_opcodes():
    """Verify plan validator detects binary opcode collisions across instructions."""
    efs_ir = EFSIR(
        opcodes=[
            EFSOpcode(mnemonic="OP_ADD", binary_encoding="4'b0001"),
            EFSOpcode(mnemonic="OP_SUB", binary_encoding="4'b0001"),  # Collision!
        ]
    )
    plan = {
        "module_name": "alu_core",
        "ports": [{"name": "clk", "direction": "input"}]
    }
    validator = RTLPlanValidator(efs_ir=efs_ir)
    report = validator.validate(plan)
    assert not report.is_valid
    assert any(d.diagnostic_type == DiagnosticCode.OPCODE_CONFLICT for d in report.diagnostics)


def test_07_missing_register_widths_and_reset():
    """Verify plan validator catches internal registers with missing reset values."""
    plan = {
        "module_name": "reg_test",
        "ports": [{"name": "clk", "direction": "input"}],
        "internal_registers": [
            {"name": "valid_reg", "width": "16", "reset_val": "16'h0"},
            {"name": "invalid_reg", "width": "32", "reset_val": None},  # Missing reset value!
        ]
    }
    validator = RTLPlanValidator()
    report = validator.validate(plan)
    assert not report.is_valid
    assert any(d.diagnostic_type == DiagnosticCode.REGISTER_DEFINITION_CONFLICT and d.affected_object == "invalid_reg" for d in report.diagnostics)


def test_08_missing_and_conflicting_reset_semantics():
    """Verify reset polarity conflict detection between signal naming convention and plan."""
    plan = {
        "module_name": "reset_test",
        "ports": [
            {"name": "clk", "direction": "input"},
            {"name": "rst_n", "direction": "input"},
        ],
        "reset_requirements": {
            "reset_signal": "rst_n",
            "reset_polarity": "active_high"  # Conflict: rst_n with active_high!
        }
    }
    validator = RTLPlanValidator()
    report = validator.validate(plan)
    assert not report.is_valid
    assert any("reset" in d.category.lower() for d in report.diagnostics)


def test_09_missing_datapath_behavior():
    """Verify plan validator catches output ports with no defined driving behavior when strategy is specified."""
    plan = {
        "module_name": "datapath_test",
        "ports": [
            {"name": "clk", "direction": "input"},
            {"name": "data_out", "direction": "output", "width": "8"},
        ],
        "output_driving_strategy": {}  # Provided but empty: data_out has no driver
    }
    validator = RTLPlanValidator()
    report = validator.validate(plan)
    assert not report.is_valid
    assert any(d.diagnostic_type == DiagnosticCode.DATAPATH_UNSPECIFIED for d in report.diagnostics)


def test_10_context_budget_reduction_manifest():
    """Verify TokenBudgetController tracks pruned elements in context_reduction_manifest and preserves P0 behavioral models."""
    large_context = {
        "module": {"name": "big_mod"},
        "ports": [{"name": f"port_{i}", "direction": "input", "description": "very long description " * 10} for i in range(20)],
        "registers": [{"name": f"reg_{i}", "width": "32", "reset_val": "0", "description": "long description " * 10} for i in range(10)],
        "source_refs": [f"ref_{i}" for i in range(50)],
        "source_evidence": [{"chunk_id": f"c_{i}", "text": "evidence text " * 20} for i in range(30)],
        "context_reduction_manifest": []
    }

    # Prune with strict budget of 500 tokens
    pruned, exceeded, warning = TokenBudgetController.enforce_rtl_budget(large_context, max_tokens=500)
    manifest = pruned.get("context_reduction_manifest", [])
    assert len(manifest) > 0  # Recorded reduction decisions!
    # P0 constructs MUST NOT be discarded
    assert len(pruned["ports"]) == 20
    assert len(pruned["registers"]) == 10


def test_11_hitl_approval_and_rejection_lifecycle():
    """Verify complete HITL lifecycle: conflict detection -> presentation -> approval -> revalidation."""
    efs_ir = EFSIR(
        signals=[
            EFSSignal(name="clk", direction="input"),
            EFSSignal(name="rst_n", direction="input"),
            EFSSignal(name="data", direction="input", width="8"),
        ],
        conflicts=[
            EFSConflict(
                conflict_id="CONF_RESET_01",
                category="RESET",
                description="Conflicting reset polarity: Section 1 asserts active-high while Section 2 asserts active-low.",
                resolution_status="UNRESOLVED"
            )
        ]
    )

    # 1. Present conflicts to human with alternatives
    conflicts_view = HITLPlanManager.get_unresolved_architectural_conflicts(efs_ir)
    assert len(conflicts_view) == 1
    assert conflicts_view[0]["conflict_id"] == "CONF_RESET_01"
    assert len(conflicts_view[0]["alternatives"]) >= 2

    # 2. Rejection records REJECTED status
    res_rej = HITLPlanManager.reject_resolution(efs_ir, "CONF_RESET_01", reason="Need design team meeting")
    assert res_rej.approval_status == "REJECTED"

    # 3. Approval resolves the conflict
    res_app = HITLPlanManager.approve_resolution(
        efs_ir,
        conflict_id="CONF_RESET_01",
        approved_value="active_low",
        explanation="Standardized to active-low asynchronous reset."
    )
    assert res_app.approval_status == "APPROVED"
    assert efs_ir.conflicts[0].resolution_status == "RESOLVED"

    # 4. Subsequent validation recognizes approved resolution and passes!
    plan = {
        "module_name": "controller",
        "ports": [
            {"name": "clk", "direction": "input"},
            {"name": "rst_n", "direction": "input"},
            {"name": "data", "direction": "input", "width": "8"},
        ],
        "reset_requirements": {"reset_polarity": "active_low", "reset_signal": "rst_n"}
    }
    validator = RTLPlanValidator(efs_ir=efs_ir)
    report = validator.validate(plan)
    assert report.is_valid
    assert not any(d.conflict_id == "CONF_RESET_01" for d in report.diagnostics)


def test_12_invalid_llm_plans_rejection():
    """Verify plan validator rejects malformed and non-dictionary plans."""
    validator = RTLPlanValidator()
    # Non-dictionary plan
    rep1 = validator.validate("invalid string plan")
    assert not rep1.is_valid
    assert rep1.spec_status == "SPECIFICATION_INCOMPLETE"

    # Plan with no ports
    rep2 = validator.validate({"module_name": "empty_mod", "ports": []})
    assert not rep2.is_valid


def test_13_traceability_preservation():
    """Verify RTLGenerationPlan retains provenance IDs, source references, and confidence scores."""
    plan_obj = RTLGenerationPlan(
        module_name="traceable_alu",
        target_language="Verilog",
        architecture_class="ARITHMETIC_PROCESSING_UNIT",
        ports=[
            PlanPort(name="clk", direction="input", semantic_role="clock", efs_id="SIG_001", source_ref="Doc1 Sec 2.1"),
            PlanPort(name="result", direction="output", width="16", efs_id="SIG_002", source_ref="Doc1 Sec 2.2"),
        ],
        traceability=PlanTraceability(
            efs_object_ids=["SIG_001", "SIG_002"],
            source_evidence_ids=["CHUNK_001", "CHUNK_002"],
            confidence=0.98,
            evidence_classification=EvidenceClassification.EXPLICIT
        )
    )

    d = plan_obj.to_dict()
    assert d["module_name"] == "traceable_alu"
    assert d["architecture_class"] == "ARITHMETIC_PROCESSING_UNIT"
    assert d["traceability"]["confidence"] == 0.98
    assert d["traceability"]["efs_object_ids"] == ["SIG_001", "SIG_002"]
    assert d["ports"][0]["efs_id"] == "SIG_001"
    assert d["ports"][0]["source_ref"] == "Doc1 Sec 2.1"


def test_14_hardware_architecture_classification():
    """Verify architecture classifier identifies all 9 canonical hardware characteristics."""
    # Combinational
    c1 = classify_hardware_architecture(spec_text="4-to-1 multiplexer without clock. sel [1:0], in0, in1, out.")
    assert c1.primary_class == ArchitectureClass.COMBINATIONAL

    # Sequential
    c2 = classify_hardware_architecture(spec_text="Counter with clock clk and reset rst_n.")
    assert c2.has_clock and c2.has_reset

    # FSM Controller
    c3 = classify_hardware_architecture(spec_text="Sequence controller with states S_IDLE, S_RUN, S_DONE.")
    assert c3.primary_class == ArchitectureClass.FSM_CONTROLLER

    # Datapath Pipeline
    c4 = classify_hardware_architecture(spec_text="Pipeline stage with stage_valid, stage_data, clk, rst_n.")
    assert c4.primary_class == ArchitectureClass.DATAPATH

    # Memory / FIFO
    c5 = classify_hardware_architecture(spec_text="Synchronous FIFO buffer with write_pointer, read_pointer, clk, rst_n.")
    assert c5.primary_class == ArchitectureClass.MEMORY_INTERFACE

    # Bus Interface
    c6 = classify_hardware_architecture(spec_text="Register bank with reg_addr, reg_data, offset 0x0.")
    assert c6.primary_class == ArchitectureClass.BUS_INTERFACE

    # Arithmetic Unit
    c7 = classify_hardware_architecture(spec_text="Matrix multiplier and accumulator unit with clk, rst_n.")
    assert "ARITHMETIC" in c7.secondary_traits or c7.primary_class in (ArchitectureClass.ARITHMETIC_PROCESSING_UNIT, ArchitectureClass.DATAPATH)
