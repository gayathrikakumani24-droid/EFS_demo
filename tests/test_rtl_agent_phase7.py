"""
Tests for Phase 7: Verification That Means Something.

Validates:
1. Tool-driven linting hierarchy (Verilator -> Icarus -> Yosys -> Static Emulation)
   and strict error rules (LATCH, MULTIDRIVEN, COMBDLY, BLKSEQ, UNOPTFLAT).
2. Built-in static checks: exact port equality with plan, balanced constructs,
   no placeholders/delays/initial, single driver, blocking vs non-blocking correctness,
   default assignments in combinational blocks.
3. Plan-conformance check: extracts FSM states and transitions from code and verifies
   no missing states/transitions and no extra phantom states/transitions.
4. Auto-generated testbench: clock, reset dwell sequence, directed transition stimulus,
   dwell-time checks, SVA assertions for IR requirements, and simulation verification.
5. Candidate selection: K candidates evaluation, first-pass selection, bounded repair
   with actual tool error feedback, and best candidate selection with clearly flagged defects.
"""

import pytest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from core.efs_ir.models import EFSIR, EFSRequirement, EFSSignal, EFSFSM, EFSState, EFSTransition
from core.verification.candidate_verifier import (
    run_candidate_tool_lint,
    parse_verilator_lint_output,
    run_static_tool_emulation,
    check_exact_port_equality,
    check_balanced_constructs,
    check_placeholders_and_synthesizability,
    check_single_driver_and_blocking_rules,
    run_builtin_static_checks,
    extract_fsm_from_code,
    verify_fsm_plan_conformance,
    generate_auto_testbench,
    simulate_candidate_testbench,
    evaluate_candidate,
    select_best_candidate,
    CandidateEvaluationReport,
    ToolLintResult,
    StaticCheckResult,
    FSMConformanceResult,
    SimulationResult,
)
from core.agents.rtl_agent import (
    generate_code,
    generate_candidate_k_rtl,
    validate_generated_code,
)
from core.verification.hdl_validator import HDLLanguage, ValidationSeverity


# ==============================================================================
# 1. Tool-Driven Linting & Strict Error Rules (LATCH, MULTIDRIVEN, etc.)
# ==============================================================================

def test_01_verilator_warning_types_treated_as_errors():
    """Verify LATCH, MULTIDRIVEN, COMBDLY, BLKSEQ, UNOPTFLAT are treated as critical errors."""
    mock_verilator_out = """
%Warning-LATCH: module.sv:25:5: Inferred latch for signal 'next_state'
%Warning-MULTIDRIVEN: module.sv:40:10: Signal has multiple drivers: 'data_out'
%Warning-COMBDLY: module.sv:55:12: Delayed assignment in non-clocked always block
%Warning-BLKSEQ: module.sv:70:8: Blocking assignment (=) in sequential always block
%Warning-UNOPTFLAT: module.sv:85:3: Circular logic loop detected between 'a' and 'b'
%Error: module.sv:99:1: Syntax error
"""
    errors = parse_verilator_lint_output(mock_verilator_out)
    rule_ids = [e.rule_id for e in errors]
    severities = [e.severity for e in errors]

    assert "RULE_VERILATOR_LATCH" in rule_ids
    assert "RULE_VERILATOR_MULTIDRIVEN" in rule_ids
    assert "RULE_VERILATOR_COMBDLY" in rule_ids
    assert "RULE_VERILATOR_BLKSEQ" in rule_ids
    assert "RULE_VERILATOR_UNOPTFLAT" in rule_ids
    assert "RULE_VERILATOR_SYNTAX_ERROR" in rule_ids

    # All of these must have CRITICAL severity
    assert all(s == ValidationSeverity.CRITICAL for s in severities)


def test_01_static_tool_emulation_catches_all_strict_patterns():
    """Verify deterministic static tool emulation detects strict Verilator rules when tool is absent."""
    # 1. Inferred latch (case missing default)
    code_latch = """
module latch_mod (input wire clk, input wire [1:0] sel, output reg [7:0] out);
  always @(*) begin
    case (sel)
      2'b00: out = 8'h00;
      2'b01: out = 8'h01;
    endcase
  end
endmodule
"""
    res_latch = run_static_tool_emulation(code_latch, HDLLanguage.VERILOG)
    assert res_latch.status == "FAIL"
    assert any(e.rule_id == "RULE_VERILATOR_LATCH" for e in res_latch.errors)

    # 2. Multi-driven signal
    code_multidrive = """
module multidrive_mod (input wire clk, input wire d1, input wire d2, output reg q);
  always @(posedge clk) begin
    q <= d1;
  end
  always @(posedge clk) begin
    q <= d2;
  end
endmodule
"""
    res_multi = run_static_tool_emulation(code_multidrive, HDLLanguage.VERILOG)
    assert res_multi.status == "FAIL"
    assert any(e.rule_id == "RULE_VERILATOR_MULTIDRIVEN" for e in res_multi.errors)

    # 3. Blocking assignment in sequential clocked block (BLKSEQ)
    code_blkseq = """
module blkseq_mod (input wire clk, input wire d, output reg q);
  always @(posedge clk) begin
    q = d;
  end
endmodule
"""
    res_blk = run_static_tool_emulation(code_blkseq, HDLLanguage.VERILOG)
    assert res_blk.status == "FAIL"
    assert any(e.rule_id == "RULE_VERILATOR_BLKSEQ" for e in res_blk.errors)

    # 4. Combinational delay (COMBDLY)
    code_combdly = """
module combdly_mod (input wire a, output reg b);
  always @(*) begin
    #5 b = ~a;
  end
endmodule
"""
    res_comb = run_static_tool_emulation(code_combdly, HDLLanguage.VERILOG)
    assert res_comb.status == "FAIL"
    assert any(e.rule_id == "RULE_VERILATOR_COMBDLY" for e in res_comb.errors)

    # 5. Circular combinational loop (UNOPTFLAT)
    code_unopt = """
module unopt_mod (input wire clk);
  assign a = b;
  assign b = a;
endmodule
"""
    res_unopt = run_static_tool_emulation(code_unopt, HDLLanguage.VERILOG)
    assert res_unopt.status == "FAIL"
    assert any(e.rule_id == "RULE_VERILATOR_UNOPTFLAT" for e in res_unopt.errors)


# ==============================================================================
# 2. Built-in Static Checks
# ==============================================================================

def test_02_exact_port_equality_with_plan():
    """Verify exact port equality (missing ports, extra ports, mismatched direction/width)."""
    rtl_plan = {
        "module_name": "fsm_ctrl",
        "ports": [
            {"name": "clk", "direction": "input", "width": "1"},
            {"name": "rst_n", "direction": "input", "width": "1"},
            {"name": "start", "direction": "input", "width": "1"},
            {"name": "data_in", "direction": "input", "width": "32"},
            {"name": "done", "direction": "output", "width": "1"},
        ]
    }

    # Code with:
    # 1. Missing port: 'start'
    # 2. Extra port: 'unplanned_signal'
    # 3. Direction mismatch: 'done' declared as input
    # 4. Width mismatch: 'data_in' declared as 8-bit instead of 32-bit
    bad_code = """
module fsm_ctrl (
  input wire clk,
  input wire rst_n,
  input wire [7:0] data_in,
  input wire done,
  output wire unplanned_signal
);
  assign unplanned_signal = 1'b0;
endmodule
"""
    valid, errors, details = check_exact_port_equality(bad_code, rtl_plan)
    assert not valid
    rule_ids = [e.rule_id for e in errors]
    assert "RULE_PORT_MISSING" in rule_ids
    assert "RULE_PORT_EXTRA" in rule_ids
    assert "RULE_PORT_DIRECTION_MISMATCH" in rule_ids
    assert "RULE_PORT_WIDTH_MISMATCH" in rule_ids


def test_02_balanced_constructs_checks():
    """Verify balanced constructs: begin/end, case/endcase, brackets."""
    unbalanced_code = """
module unbal_mod (input wire clk, input wire [1:0] sel);
  always @(posedge clk) begin
    case (sel)
      2'b00: begin
        $display("Zero");
      // missing end for begin
      default: $display("Default");
    endcase
  // missing end for always
endmodule
"""
    valid, errors = check_balanced_constructs(unbalanced_code)
    assert not valid
    assert any(e.rule_id == "RULE_UNBALANCED_BEGIN_END" for e in errors)


def test_02_placeholders_and_synthesizability_checks():
    """Verify rejection of TODO/placeholders, #delays, and initial blocks in synthesizable logic."""
    code_with_todo = """
module todo_mod (input wire clk, output reg out);
  always @(posedge clk) begin
    // TODO: implement logic here
    out <= 1'b1;
  end
endmodule
"""
    v_todo, err_todo = check_placeholders_and_synthesizability(code_with_todo)
    assert not v_todo
    assert any(e.rule_id == "RULE_PLACEHOLDER_IN_CODE" for e in err_todo)

    code_with_initial = """
module init_mod (input wire clk, output reg out);
  initial begin
    out = 1'b0;
  end
  always @(posedge clk) out <= ~out;
endmodule
"""
    v_init, err_init = check_placeholders_and_synthesizability(code_with_initial)
    assert not v_init
    assert any(e.rule_id == "RULE_INITIAL_BLOCK_IN_SYNTHESIZABLE_CODE" for e in err_init)


# ==============================================================================
# 3. Plan-Conformance Check: FSM States and Transitions
# ==============================================================================

def test_03_fsm_extraction_and_strict_conformance():
    """Verify planned states/transitions exist and no phantom states/transitions exist."""
    rtl_plan = {
        "module_name": "traffic_fsm",
        "fsm": {
            "states": ["IDLE", "ACTIVE", "FINISH"],
            "transitions": [
                {"from_state": "IDLE", "to_state": "ACTIVE", "condition": "start"},
                {"from_state": "ACTIVE", "to_state": "FINISH", "condition": "count_done"},
                {"from_state": "FINISH", "to_state": "IDLE", "condition": "ack"},
            ]
        }
    }

    # Clean code that conforms 100% to the plan
    good_code = """
module traffic_fsm (input wire clk, input wire rst_n, input wire start, input wire count_done, input wire ack);
  localparam [1:0] IDLE = 2'd0, ACTIVE = 2'd1, FINISH = 2'd2;
  reg [1:0] state, next_state;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) state <= IDLE;
    else state <= next_state;
  end

  always @(*) begin
    next_state = state;
    case (state)
      IDLE: begin
        if (start) next_state = ACTIVE;
      end
      ACTIVE: begin
        if (count_done) next_state = FINISH;
      end
      FINISH: begin
        if (ack) next_state = IDLE;
      end
      default: next_state = IDLE;
    endcase
  end
endmodule
"""
    res_good = verify_fsm_plan_conformance(good_code, rtl_plan)
    assert res_good.is_conformant
    assert len(res_good.errors) == 0
    assert set(res_good.extracted_states) == {"IDLE", "ACTIVE", "FINISH"}

    # Bad code:
    # 1. Missing state 'FINISH'
    # 2. Extra phantom state 'PHANTOM_STATE'
    # 3. Missing transition 'ACTIVE -> FINISH'
    # 4. Extra phantom transition 'ACTIVE -> PHANTOM_STATE'
    bad_code = """
module traffic_fsm (input wire clk, input wire rst_n, input wire start);
  localparam [1:0] IDLE = 2'd0, ACTIVE = 2'd1, PHANTOM_STATE = 2'd2;
  reg [1:0] state, next_state;

  always @(*) begin
    next_state = state;
    case (state)
      IDLE: begin
        if (start) next_state = ACTIVE;
      end
      ACTIVE: begin
        next_state = PHANTOM_STATE;
      end
      default: next_state = IDLE;
    endcase
  end
endmodule
"""
    res_bad = verify_fsm_plan_conformance(bad_code, rtl_plan)
    assert not res_bad.is_conformant
    err_rules = [e.rule_id for e in res_bad.errors]
    assert "RULE_FSM_MISSING_STATE" in err_rules
    assert "RULE_FSM_EXTRA_STATE" in err_rules
    assert "RULE_FSM_MISSING_TRANSITION" in err_rules
    assert "RULE_FSM_EXTRA_TRANSITION" in err_rules


# ==============================================================================
# 4. Auto-Generated Testbench & Simulation (iverilog/vvp + SVA)
# ==============================================================================

def test_04_auto_testbench_generation_structure():
    """Verify testbench generation contains reset dwell, directed transitions, and SVA assertions."""
    rtl_plan = {
        "module_name": "arbiter",
        "clock_reset": {
            "clock_name": "clk",
            "reset_name": "rst_n",
            "reset_polarity": "active_low",
        },
        "ports": [
            {"name": "clk", "direction": "input", "width": "1"},
            {"name": "rst_n", "direction": "input", "width": "1"},
            {"name": "req_a", "direction": "input", "width": "1"},
            {"name": "gnt_a", "direction": "output", "width": "1"},
        ],
        "fsm": {
            "states": ["IDLE", "GRANT_A"],
            "transitions": [
                {"from_state": "IDLE", "to_state": "GRANT_A", "condition": "req_a == 1"},
                {"from_state": "GRANT_A", "to_state": "IDLE", "condition": "req_a == 0"},
            ]
        }
    }

    req = EFSRequirement(
        requirement_id="REQ_ARB_01",
        description="Grant shall be asserted only when request is active",
    )
    efs_ir = EFSIR(
        requirements=[req],
    )

    tb = generate_auto_testbench(rtl_plan, target_language="SystemVerilog", efs_ir=efs_ir)

    # 1. Module header & DUT instantiation
    assert "module tb_arbiter;" in tb
    assert "arbiter dut (" in tb

    # 2. Clock & Reset dwell sequence
    assert "clk = 0;" in tb
    assert "forever #5 clk = ~clk;" in tb
    assert "rst_n = 0;" in tb
    assert "repeat (2) @(posedge clk);" in tb
    assert "rst_n = 1;" in tb

    # 3. Directed transitions & dwell checks
    assert "Transition 'IDLE' -> 'GRANT_A'" in tb
    assert "req_a = 1;" in tb
    assert "dwell_cycles" in tb

    # 4. SVA assertion for requirement
    assert "REQ_ARB_01" in tb
    assert "assert property" in tb

    # 5. Simulation verdict
    assert "ALL_CHECKS_PASSED" in tb


def test_04_simulation_verifies_candidate_correctness():
    """Verify simulation runner passes valid candidate and rejects failing candidate."""
    rtl_plan = {
        "module_name": "simple_gate",
        "clock_reset": {"clock_name": "clk", "reset_name": "rst_n", "reset_polarity": "active_low"},
        "ports": [
            {"name": "clk", "direction": "input", "width": "1"},
            {"name": "rst_n", "direction": "input", "width": "1"},
            {"name": "in_data", "direction": "input", "width": "1"},
            {"name": "out_data", "direction": "output", "width": "1"},
        ],
        "fsm": {
            "states": ["IDLE", "ACTIVE"],
            "transitions": [
                {"from_state": "IDLE", "to_state": "ACTIVE", "condition": "in_data == 1"}
            ]
        }
    }
    tb = generate_auto_testbench(rtl_plan, target_language="Verilog")

    # Valid candidate
    dut_valid = """
module simple_gate (input wire clk, input wire rst_n, input wire in_data, output reg out_data);
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) out_data <= 1'b0;
    else out_data <= in_data;
  end
endmodule
"""
    res_val = simulate_candidate_testbench(dut_valid, tb, target_language="Verilog")
    assert res_val.status == "PASS"

    # Mismatched candidate (missing port)
    dut_bad = """
module simple_gate (input wire clk, input wire rst_n, output reg out_data);
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) out_data <= 1'b0;
    else out_data <= 1'b1;
  end
endmodule
"""
    res_bad = simulate_candidate_testbench(dut_bad, tb, target_language="Verilog")
    assert res_bad.status == "FAIL"


# ==============================================================================
# 5. Candidate Selection (K Candidates, First Pass, Best Candidate Fallback)
# ==============================================================================

def test_05_candidate_selection_picks_first_valid():
    """Verify candidate selector returns first candidate that passes all verification checks."""
    eval_fail1 = CandidateEvaluationReport(
        candidate_index=1,
        is_valid=False,
        score=25.0,
        tool_lint=ToolLintResult(status="FAIL", tool_name="verilator"),
        static_checks=StaticCheckResult(is_valid=False),
        fsm_conformance=FSMConformanceResult(is_conformant=False),
        simulation=SimulationResult(status="FAIL", tool="none"),
        all_errors=["[RULE_VERILATOR_LATCH] Inferred latch"],
    )

    eval_pass2 = CandidateEvaluationReport(
        candidate_index=2,
        is_valid=True,
        score=0.0,
        tool_lint=ToolLintResult(status="PASS", tool_name="verilator"),
        static_checks=StaticCheckResult(is_valid=True),
        fsm_conformance=FSMConformanceResult(is_conformant=True),
        simulation=SimulationResult(status="PASS", tool="none"),
        all_errors=[],
    )

    eval_pass3 = CandidateEvaluationReport(
        candidate_index=3,
        is_valid=True,
        score=0.0,
        tool_lint=ToolLintResult(status="PASS", tool_name="verilator"),
        static_checks=StaticCheckResult(is_valid=True),
        fsm_conformance=FSMConformanceResult(is_conformant=True),
        simulation=SimulationResult(status="PASS", tool="none"),
        all_errors=[],
    )

    best = select_best_candidate([eval_fail1, eval_pass2, eval_pass3])
    assert best.candidate_index == 2
    assert best.is_valid


def test_05_candidate_selection_falls_back_to_best_when_all_fail():
    """Verify when all candidates fail, selector returns the candidate with lowest defect score."""
    eval_bad1 = CandidateEvaluationReport(
        candidate_index=1,
        is_valid=False,
        score=45.0,
        tool_lint=ToolLintResult(status="FAIL", tool_name="verilator"),
        static_checks=StaticCheckResult(is_valid=False),
        fsm_conformance=FSMConformanceResult(is_conformant=False),
        simulation=SimulationResult(status="FAIL", tool="none"),
        all_errors=["3 critical errors"],
        flagged_code="// Candidate 1 flagged",
    )

    eval_bad2 = CandidateEvaluationReport(
        candidate_index=2,
        is_valid=False,
        score=10.0,  # Best score among defectives
        tool_lint=ToolLintResult(status="PASS", tool_name="verilator"),
        static_checks=StaticCheckResult(is_valid=False),
        fsm_conformance=FSMConformanceResult(is_conformant=True),
        simulation=SimulationResult(status="PASS", tool="none"),
        all_errors=["1 minor port order mismatch"],
        flagged_code="// Candidate 2 flagged",
    )

    best = select_best_candidate([eval_bad1, eval_bad2])
    assert best.candidate_index == 2
    assert not best.is_valid
    assert best.flagged_code == "// Candidate 2 flagged"


def test_05_generate_candidate_k_rtl_integration():
    """Verify generate_candidate_k_rtl executes generation and returns code and report."""
    spec = """
# Module: counter_fsm
## Interface
| Port | Width | Direction | Role | Description |
| :--- | :--- | :--- | :--- | :--- |
| clk | 1 | input | clock | Master clock |
| rst_n | 1 | input | reset | Active-low reset |
| start | 1 | input | control | Start signal |
| count | 8 | output | data | Output counter |
| done | 1 | output | status | Completion flag |

## State Machine
- States: IDLE, COUNT, DONE
- Transitions:
  - IDLE -> COUNT on `start == 1`
  - COUNT -> DONE on `count == 8'd10`
  - DONE -> IDLE on `start == 0`
"""
    code, report = generate_candidate_k_rtl(spec, target="SystemVerilog", k_candidates=2)
    assert code is not None
    assert "module counter_fsm" in code
    assert "endmodule" in code
    assert isinstance(report, CandidateEvaluationReport)
