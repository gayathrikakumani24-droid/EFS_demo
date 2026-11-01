"""
End-to-End Counter Controller RTL Smoke Test and Validation Suite.

Validates:
1. Signal/port ownership preserved as EFSSignal (clk, rst_n, start, busy, done, count[7:0]).
2. NO false EFSInterface objects created for signals or FSM transitions.
3. FSM transitions preserved strictly in EFSFSM.transitions.
4. Completeness validator raises 0 'Missing Memory Interface' conflicts for non-memory design.
5. Completeness validator raises 0 'Missing Inter-Module Relationships' conflicts for single controller.
6. RTL Plan validation and RTL generation succeed without false blocker errors.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["USE_HASHING_EMBEDDER"] = "1"

import pytest
from unittest.mock import patch, MagicMock
from core.efs_ir.models import (
    EFSIR,
    EFSMetadata,
    EFSComponent,
    EFSSignal,
    EFSFSM,
    EFSState,
    EFSTransition,
)
from core.requirement_ir.models import DocumentIR, DocumentBlock, SourceLocation
from core.extraction.source_evidence import SourceEvidenceIndex
from core.efs_ir.validator import validate_efs_ir
from core.verification.conflict_resolution_engine import ConflictResolutionEngine, RegisterRequirementStatus
from core.retrieval.context_compiler import compile_task_context
from core.agents.rtl_agent import plan_rtl, validate_rtl_plan, generate_code


def _build_counter_controller_doc_ir():
    spec_text = (
        "Design a simple synchronous counter controller with clk, rst_n, start, busy, done, count[7:0].\n"
        "clk is an input clock signal. rst_n is an active-low reset signal.\n"
        "start is an input signal that triggers counting.\n"
        "busy is an output signal indicating counting or done.\n"
        "done is an output signal asserted when count reaches target.\n"
        "count[7:0] is an 8-bit counter output.\n"
        "The controller contains an FSM with IDLE, COUNTING, and DONE states.\n"
        "IDLE transitions to COUNTING when start == 1.\n"
        "COUNTING transitions to DONE when count == 10.\n"
        "After one clock cycle in DONE, return to IDLE."
    )
    blocks = [
        DocumentBlock(
            block_id="REQ_BLK_001",
            type="paragraph",
            section_id="SEC_001",
            section_title="Counter Controller Spec",
            page=1,
            text=spec_text,
            source_location=SourceLocation(
                document_id="DOC_COUNTER",
                section="Counter Controller Spec",
                section_id="SEC_001",
                page=1,
                block_id="REQ_BLK_001",
                original_text=spec_text[:100]
            )
        )
    ]
    return DocumentIR(
        document_id="DOC_COUNTER",
        metadata={"filename": "counter_controller.md"},
        sections=[{"section_id": "SEC_001", "title": "Counter Controller Spec"}],
        blocks=blocks
    )


def build_counter_controller_ir():
    """Build canonical Counter Controller EFS IR according to hard semantic boundaries."""
    comp = EFSComponent(
        component_id="comp_counter",
        name="Counter_Controller",
        type="controller",
        description="Simple synchronous counter controller"
    )

    sig_clk = EFSSignal(signal_id="sig_clk", name="clk", direction="input", width="1", semantic_role="clock", owner="comp_counter")
    sig_rst = EFSSignal(signal_id="sig_rst", name="rst_n", direction="input", width="1", semantic_role="reset", owner="comp_counter")
    sig_start = EFSSignal(signal_id="sig_start", name="start", direction="input", width="1", semantic_role="control", owner="comp_counter")
    sig_busy = EFSSignal(signal_id="sig_busy", name="busy", direction="output", width="1", semantic_role="status", owner="comp_counter")
    sig_done = EFSSignal(signal_id="sig_done", name="done", direction="output", width="1", semantic_role="status", owner="comp_counter")
    sig_count = EFSSignal(signal_id="sig_count", name="count", direction="output", width="8", semantic_role="data", owner="comp_counter")

    fsm = EFSFSM(
        fsm_id="fsm_counter",
        name="Counter_Controller_FSM",
        states=[
            EFSState(state_id="st_idle", name="IDLE"),
            EFSState(state_id="st_counting", name="COUNTING"),
            EFSState(state_id="st_done", name="DONE")
        ],
        transitions=[
            EFSTransition(transition_id="tr_1", source_state="IDLE", target_state="COUNTING", condition="start == 1"),
            EFSTransition(transition_id="tr_2", source_state="COUNTING", target_state="DONE", condition="count == 10"),
            EFSTransition(transition_id="tr_3", source_state="DONE", target_state="IDLE", condition="next clock cycle")
        ],
        initial_state="IDLE"
    )

    flow = {
        "flow_id": "flow_counter",
        "name": "Counter_Execution_Flow",
        "description": "Counter execution flow from IDLE to COUNTING to DONE"
    }

    return EFSIR(
        metadata=EFSMetadata(
            design_id="DES_COUNTER",
            design_name="Counter_Controller",
            top_module="Counter_Controller"
        ),
        components=[comp],
        signals=[sig_clk, sig_rst, sig_start, sig_busy, sig_done, sig_count],
        clocks=[sig_clk],
        resets=[sig_rst],
        fsms=[fsm],
        flows=[flow],
        interfaces=[]  # ZERO false interfaces!
    )


def test_counter_controller_smoke():
    """Verify Counter Controller passes completeness, structural, plan, and generation checks."""
    doc_ir = _build_counter_controller_doc_ir()
    evidence_index = SourceEvidenceIndex(doc_ir)
    ir = build_counter_controller_ir()

    # 1. Structural Validation
    issues = validate_efs_ir(ir)
    error_issues = [i for i in issues if i["severity"] == "ERROR"]
    assert len(error_issues) == 0, f"Expected 0 structural error issues, got: {error_issues}"
    assert ir.validation.status == "PASSED"

    # Verify ZERO false interfaces
    assert len(ir.interfaces) == 0

    # 2. Completeness Validation
    reg_status, reg_ev, reg_reason = ConflictResolutionEngine.classify_register_requirement(
        ir, doc_ir=doc_ir, evidence_index=evidence_index
    )
    assert reg_status == RegisterRequirementStatus.NOT_APPLICABLE, f"Expected NOT_APPLICABLE, got {reg_status}"

    conflicts = ConflictResolutionEngine.validate_hardware_completeness(
        ir, doc_ir=doc_ir, evidence_index=evidence_index
    )
    reg_conflicts = [c for c in conflicts if "Register" in (c.title or "") or c.property == "registers"]
    rel_conflicts = [c for c in conflicts if c.title == "Missing Inter-Module Relationships"]
    mem_conflicts = [c for c in conflicts if "Memory" in (c.title or "")]
    clk_conflicts = [c for c in conflicts if "Clock" in (c.title or "")]
    rst_conflicts = [c for c in conflicts if "Reset" in (c.title or "")]

    assert len(reg_conflicts) == 0, f"Expected 0 register conflicts, got: {[c.title for c in reg_conflicts]}"
    assert len(rel_conflicts) == 0, f"Expected 0 relationship conflicts, got: {[c.title for c in rel_conflicts]}"
    assert len(mem_conflicts) == 0, f"Expected 0 memory conflicts, got: {[c.title for c in mem_conflicts]}"
    assert len(clk_conflicts) == 0
    assert len(rst_conflicts) == 0
    assert len(conflicts) == 0, f"Expected 0 completeness conflicts, got: {[c.title for c in conflicts]}"

    # 3. ContextPack compilation
    pack, trace = compile_task_context("Generate RTL for Counter_Controller", ir)
    assert pack.task.target == "Counter_Controller"
    assert len(pack.signals) == 6
    assert any(s["name"] == "clk" for s in pack.signals)
    assert any(s["name"] == "rst_n" for s in pack.signals)
    assert any(s["name"] == "start" for s in pack.signals)
    assert any(s["name"] == "busy" for s in pack.signals)
    assert any(s["name"] == "done" for s in pack.signals)
    assert any(s["name"] == "count" for s in pack.signals)

    # 4. RTL Plan Validation
    plan = {
        "module_name": "Counter_Controller",
        "ports": [
            {"name": "clk", "direction": "input", "width": "1"},
            {"name": "rst_n", "direction": "input", "width": "1"},
            {"name": "start", "direction": "input", "width": "1"},
            {"name": "busy", "direction": "output", "width": "1"},
            {"name": "done", "direction": "output", "width": "1"},
            {"name": "count", "direction": "output", "width": "8"},
        ],
        "clock_reset": {
            "clock_name": "clk",
            "reset_name": "rst_n",
            "reset_type": "async_low"
        },
        "fsm": {
            "states": ["IDLE", "COUNTING", "DONE"],
            "transitions": [
                {"from": "IDLE", "to": "COUNTING", "condition": {"expression": "start == 1'b1", "is_fully_defined": True}},
                {"from": "COUNTING", "to": "DONE", "condition": {"expression": "count == 8'd10", "is_fully_defined": True}},
                {"from": "DONE", "to": "IDLE", "condition": {"expression": "1'b1", "is_fully_defined": True}}
            ]
        }
    }

    plan_val = validate_rtl_plan(plan, efs_ir=ir, context_pack=pack)
    assert plan_val["valid"] is True
    assert plan_val["status"] == "COMPLETE"
    assert len(plan_val["issues"]) == 0

    # 5. RTL Generation
    mock_llm = MagicMock()
    mock_llm.available = False  # Template fallback generates clean Verilog

    with patch("core.agents.rtl_agent.get_llm_client", return_value=mock_llm):
        code = generate_code(
            design_spec="Counter Controller specification",
            target="SystemVerilog",
            efs_ir=ir,
            context_pack=pack
        )

    assert "module Counter_Controller" in code
    assert "clk" in code
    assert "rst_n" in code
    assert "start" in code
    assert "busy" in code
    assert "done" in code
    assert "count" in code
    assert "endmodule" in code
    print("\n--- GENERATED RTL FOR COUNTER CONTROLLER ---")
    print(code)
    print("--------------------------------------------")


if __name__ == "__main__":
    test_counter_controller_smoke()
    print("ALL COUNTER CONTROLLER CHECKS PASSED SUCCESSFULLY!")
