"""
Comprehensive Behavioral RTL Generation & Generality Test Suite.

Verifies specification-agnostic functional Verilog generation across multiple
canonical hardware architecture classes:
1. Combinational ALU / Data Selector
2. Sequential Counter / Timer
3. FSM-based Protocol Controller
4. Instruction Execution Unit (Opcode decoding, Datapath, Register File)
5. End-to-end Spec Conflict Resolution, Resume Flow, and Physical Verification (iverilog/vvp)
"""

import os
import re
import pytest
from core.efs_ir.models import (
    EFSIR,
    EFSMetadata,
    EFSComponent,
    EFSSignal,
    EFSRequirement,
    EFSOpcode,
    EFSRegister,
    EFSFSM,
    EFSState,
    EFSTransition,
)
from core.agents.spec_classifier import classify_hardware_architecture, ArchitectureClass, SpecType
from core.agents.fsm_inference_engine import FSMInferenceEngine, FSMHandlingCase
from core.agents.rtl_agent import plan_rtl, generate_code
from core.verification.candidate_verifier import (
    verify_candidate_rtl,
    compile_candidate_hdl,
    simulate_candidate_testbench,
)


def test_combinational_alu_generality():
    """Verify that pure combinational specs generate correct multiplexed ALU logic without clocks or FSM."""
    spec = """
    Module generic_arithmetic_selector.
    Inputs:
      - a: 8-bit input data
      - b: 8-bit input data
      - sel: 2-bit operation selector
    Outputs:
      - result: 8-bit output result
    Operation:
      When sel == 2'b00, result = a + b.
      When sel == 2'b01, result = a - b.
      When sel == 2'b10, result = a & b.
      When sel == 2'b11, result = a | b.
    """
    signals = [
        EFSSignal(name="a", direction="input", width="8", semantic_role="data"),
        EFSSignal(name="b", direction="input", width="8", semantic_role="data"),
        EFSSignal(name="sel", direction="input", width="2", semantic_role="control"),
        EFSSignal(name="result", direction="output", width="8", semantic_role="data"),
    ]
    ir = EFSIR(
        metadata=EFSMetadata(design_name="generic_arithmetic_selector", top_module="generic_arithmetic_selector"),
        components=[EFSComponent(name="generic_arithmetic_selector")],
        signals=signals,
        requirements=[EFSRequirement(requirement_id="REQ_COMB_1", description=spec)]
    )

    arch_res = classify_hardware_architecture(efs_ir=ir, spec_text=spec)
    assert arch_res.primary_class in (ArchitectureClass.COMBINATIONAL, ArchitectureClass.ARITHMETIC_PROCESSING_UNIT)
    assert not arch_res.has_clock
    assert not arch_res.requires_fsm

    plan = plan_rtl(spec, "Verilog", efs_ir=ir)
    assert plan.get("fsm") is None

    code, rep = generate_code(spec, "Verilog", efs_ir=ir, return_report=True, user_query="Generate generic_arithmetic_selector")
    assert "always @(*)" in code or "assign result" in code
    assert "posedge clk" not in code
    assert rep.get("passed", False) is True

    # Compile with iverilog
    c_res = compile_candidate_hdl(code, language="Verilog")
    assert c_res.passed is True


def test_sequential_counter_generality():
    """Verify that sequential counter designs maintain state without artificial FSMs."""
    spec = """
    Module generic_programmable_counter.
    Inputs:
      - clk: system clock
      - rst_n: active-low asynchronous reset
      - enable: count enable
      - load: synchronous load control
      - load_val: 16-bit load data
    Outputs:
      - count_out: 16-bit current count
      - overflow: pulse when counter rolls over
    Behavior:
      On reset, count_out clears to 0.
      If load is asserted, count_out loads load_val.
      Else if enable is asserted, count_out increments by 1.
      Otherwise, count_out holds its value.
    """
    signals = [
        EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
        EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset"),
        EFSSignal(name="enable", direction="input", width="1", semantic_role="enable"),
        EFSSignal(name="load", direction="input", width="1", semantic_role="control"),
        EFSSignal(name="load_val", direction="input", width="16", semantic_role="data"),
        EFSSignal(name="count_out", direction="output", width="16", semantic_role="data"),
        EFSSignal(name="overflow", direction="output", width="1", semantic_role="status"),
    ]
    ir = EFSIR(
        metadata=EFSMetadata(design_name="generic_programmable_counter", top_module="generic_programmable_counter"),
        components=[EFSComponent(name="generic_programmable_counter")],
        signals=signals,
        requirements=[EFSRequirement(requirement_id="REQ_CNT_1", description=spec)]
    )

    arch_res = classify_hardware_architecture(efs_ir=ir, spec_text=spec)
    assert arch_res.has_clock is True
    assert arch_res.has_reset is True

    fsm_res = FSMInferenceEngine.resolve_fsm(efs_ir=ir, spec_text=spec)
    assert fsm_res.case == FSMHandlingCase.FSM_UNNECESSARY

    code, rep = generate_code(spec, "Verilog", efs_ir=ir, return_report=True, user_query="Generate generic_programmable_counter")
    assert "always @ (posedge clk or negedge rst_n)" in code
    assert rep.get("passed", False) is True

    # Check compilation
    c_res = compile_candidate_hdl(code, language="Verilog")
    assert c_res.passed is True


def test_fsm_handshake_controller_generality():
    """Verify that multi-state controllers generate compliant FSM transition logic and output driving."""
    spec = """
    Module generic_bus_arbiter.
    Inputs:
      - clk: system clock
      - rst_n: active-low reset
      - req: client request signal
    Outputs:
      - grant: client grant signal
      - busy: controller busy status
    FSM States:
      - IDLE: Waiting for req. If req is asserted, transition to EVAL.
      - EVAL: Evaluate request. Transition unconditionally to GRANT.
      - GRANT: Assert grant. If req is deasserted, transition to IDLE.
    """
    fsm = EFSFSM(
        name="arbiter_fsm",
        initial_state="IDLE",
        states=[
            EFSState(name="IDLE"),
            EFSState(name="EVAL"),
            EFSState(name="GRANT"),
        ],
        transitions=[
            EFSTransition(source_state="IDLE", target_state="EVAL", condition={"expression": "req"}),
            EFSTransition(source_state="EVAL", target_state="GRANT", condition={"expression": "1'b1"}),
            EFSTransition(source_state="GRANT", target_state="IDLE", condition={"expression": "!req"}),
        ]
    )
    signals = [
        EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
        EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset"),
        EFSSignal(name="req", direction="input", width="1", semantic_role="control"),
        EFSSignal(name="grant", direction="output", width="1", semantic_role="control"),
        EFSSignal(name="busy", direction="output", width="1", semantic_role="status"),
    ]
    ir = EFSIR(
        metadata=EFSMetadata(design_name="generic_bus_arbiter", top_module="generic_bus_arbiter"),
        components=[EFSComponent(name="generic_bus_arbiter")],
        signals=signals,
        fsms=[fsm],
        requirements=[EFSRequirement(requirement_id="REQ_ARB_1", description=spec)]
    )

    code, rep = generate_code(spec, "Verilog", efs_ir=ir, return_report=True, user_query="Generate generic_bus_arbiter")
    assert "current_state <= next_state;" in code
    assert "case (current_state)" in code
    assert "IDLE:" in code
    assert "EVAL:" in code
    assert "GRANT:" in code
    assert rep.get("passed", False) is True

    # Compile and simulate
    c_res = compile_candidate_hdl(code, language="Verilog")
    assert c_res.passed is True


def test_instruction_execution_unit_generality():
    """Verify that instruction execution architectures decode opcodes and execute functional datapath operations."""
    spec = """
    Module generic_processor_core.
    Registers:
      - R0: 32-bit general purpose register
      - R1: 32-bit general purpose register
    Opcodes:
      - ADD: 4'b0001 - R0 <= R0 + R1
      - SUB: 4'b0010 - R0 <= R0 - R1
      - AND: 4'b0011 - R0 <= R0 & R1
      - ZERO: 4'b0100 - R0 <= 32'd0
    """
    signals = [
        EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
        EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset"),
        EFSSignal(name="instruction", direction="input", width="32", semantic_role="instruction"),
        EFSSignal(name="data_out", direction="output", width="32", semantic_role="data"),
    ]
    registers = [
        EFSRegister(name="R0", width="32", reset_value="0"),
        EFSRegister(name="R1", width="32", reset_value="0"),
    ]
    opcodes = [
        EFSOpcode(mnemonic="ADD", binary_encoding="0001", opcode_id="OPC_ADD"),
        EFSOpcode(mnemonic="SUB", binary_encoding="0010", opcode_id="OPC_SUB"),
        EFSOpcode(mnemonic="AND", binary_encoding="0011", opcode_id="OPC_AND"),
        EFSOpcode(mnemonic="ZERO", binary_encoding="0100", opcode_id="OPC_ZERO"),
    ]
    ir = EFSIR(
        metadata=EFSMetadata(design_name="generic_processor_core", top_module="generic_processor_core"),
        components=[EFSComponent(name="generic_processor_core")],
        signals=signals,
        registers=registers,
        opcodes=opcodes,
        requirements=[EFSRequirement(requirement_id="REQ_CORE_1", description=spec)]
    )

    code, rep = generate_code(spec, "Verilog", efs_ir=ir, return_report=True, user_query="Generate generic_processor_core")
    assert "localparam [3:0] ADD = 4'b0001;" in code
    assert "case (instruction[3:0])" in code
    assert "ADD: R0 <= R0 + R1;" in code
    assert "SUB: R0 <= R0 - R1;" in code
    assert "AND: R0 <= R0 & R1;" in code
    assert "ZERO: R0 <= 32'd0;" in code
    assert "default: R0 <= R0;" in code

    # No bogus unconditional assignments
    assert "R0 <= instruction;" not in code

    # Verification passes
    assert rep.get("passed", False) is True
    assert rep.get("score", 0) >= 95.0

    c_res = compile_candidate_hdl(code, language="Verilog")
    assert c_res.passed is True


def test_full_pipeline_matrix_compute_unit_resolution_and_synthesis():
    """
    End-to-end integration test executing the complete pipeline on matrix_compute_unit:
    Ingestion -> Conflict Detection -> AI Proposals -> Resolution -> Resuming -> Physical Verilog Verification.
    """
    from core.agents.orchestrator import run_design_flow, resume_design_flow_with_resolutions

    # Load actual specification document if present, or canonical specification text
    doc_path = r"C:\Users\Acer\.gemini\antigravity\brain\354eb1b4-323f-49aa-941e-0c1b56321975\.user_uploaded\media_1793678436813_842dda95.docx"
    spec_text = ""
    if os.path.exists(doc_path):
        import docx
        doc = docx.Document(doc_path)
        spec_text = "\n".join([p.text for p in doc.paragraphs])
        for t in doc.tables:
            for row in t.rows:
                spec_text += "\n" + "\t".join([c.text for c in row.cells])
    else:
        pytest.skip("matrix_compute_unit.docx not found at expected path")

    # 1. Run initial flow
    initial_res = run_design_flow(
        spec_text,
        target_outputs=["Verilog"],
        user_query="Generate Verilog RTL for matrix compute unit",
        bypass_conflicts=False,
        generation_mode="HYBRID",
        k_candidates=1,
        allow_partial=True
    )

    # Must detect duplicate opcode conflicts and pause safely for human decision
    assert initial_res.get("status") == "AWAITING_HUMAN_DECISION"
    assert len(initial_res.get("conflicts", [])) >= 4

    # 2. Approve AI proposals
    approved_resolutions = []
    proposals_map = initial_res.get("proposals", {})
    for cid, proposals in proposals_map.items():
        if proposals:
            p = proposals[0]
            p.approval_status = "APPROVED"
            approved_resolutions.append(p)

    # 3. Resume design flow with approved resolutions
    resumed_res = resume_design_flow_with_resolutions(
        initial_res,
        approved_resolutions,
        target_outputs=["Verilog"],
        user_query="Generate Verilog RTL for matrix compute unit",
        bypass_conflicts=False,
        generation_mode="HYBRID",
        k_candidates=1,
        allow_partial=True
    )

    assert resumed_res.get("status") == "SUCCESS"
    assert resumed_res.get("pipeline_state") == "GENERATION_COMPLETE"
    assert "Verilog" in resumed_res.get("outputs", {})

    verilog_out = resumed_res["outputs"]["Verilog"]
    code = verilog_out.get("final_code", "")
    assert verilog_out.get("compliance_score", 0) >= 95.0
    assert verilog_out.get("readiness_status") == "RTL_VERIFIED"

    # Verilog code quality assertions:
    # 1. Real opcode decoding
    assert "case (instruction[" in code
    # 2. No bogus output driving (axi_arvalid = M0)
    assert "assign axi_arvalid = M0;" not in code
    assert "assign axi_awvalid = M0;" not in code
    assert "assign axi_wvalid = M0;" not in code
    assert "assign axi_wdata = M0;" not in code
    # 3. Registers hold state on default
    assert "default: M0 <= M0;" in code
    # 4. FSM transitions present
    assert "IDLE: begin" in code
    assert "EXECUTE: begin" in code
    assert "DONE: begin" in code

    # Physical verification: iverilog compilation and vvp simulation pass
    rep = verilog_out.get("verification_report", {})
    assert rep.get("compile", {}).get("status") == "PASS"
    assert rep.get("simulation", {}).get("status") == "PASS"
