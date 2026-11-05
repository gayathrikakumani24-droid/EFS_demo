"""
Unit and regression test suite for Phase 2:
- Remove guesswork from get_heuristic_code_template (ALUs, alu_result, phantom triggers, muxes, hardcoded AXI, substring matches).
- Ensure SPECIFICATION_INCOMPLETE lists concrete gaps (code, location, what is needed) when specs are under-specified or retries exhaust.
- Ensure fallback strictly reads validated plan transitions instead of raw prose from efs_ir.fsms[0].transitions.
"""

import re
from unittest.mock import MagicMock, patch

import pytest
from core.agents.rtl_agent import (
    format_specification_incomplete_result,
    generate_code,
    get_heuristic_code_template,
    plan_rtl,
    validate_generated_code,
    validate_rtl_plan,
)
from core.efs_ir.models import EFSFSM, EFSIR, EFSMetadata, EFSSignal, EFSState, EFSTransition


def test_01_acceptance_cmd_and_length_no_alu_no_phantom_trigger():
    """
    Acceptance Criteria Test:
    A test spec with an input named `cmd` and one named `length` produces:
    - NO ALU logic (no case statement performing add/sub/mul/and)
    - NO `alu_result` register declaration or reset
    - NO phantom trigger (`length` must not match 'en' substring to trigger state transitions)
    """
    spec = """# Hardware Design Specification: packet_parser
states: IDLE, PROCESS, FINISH
transitions:
IDLE -> PROCESS: 1'b1
PROCESS -> FINISH: 1'b1
FINISH -> IDLE: 1'b1

## Interfaces
| Signal Name | Direction | Width | Description |
| clk | input | 1 | Clock signal |
| rst_n | input | 1 | Active-low reset |
| cmd | input | 8 | Command input |
| length | input | 16 | Packet length input |
| out_ready | output | 1 | Ready output |

## Output Strategy
out_ready driven by state == IDLE
"""
    plan = {
        "module_name": "packet_parser",
        "specification_status": {"status": "COMPLETE", "blocking_issues": []},
        "ports": [
            {"name": "clk", "direction": "input", "width": "1", "semantic_role": "clock"},
            {"name": "rst_n", "direction": "input", "width": "1", "semantic_role": "reset"},
            {"name": "cmd", "direction": "input", "width": "8"},
            {"name": "length", "direction": "input", "width": "16"},
            {"name": "out_ready", "direction": "output", "width": "1"},
        ],
        "clock_reset": {"clock_name": "clk", "reset_name": "rst_n", "reset_polarity": "low"},
        "fsm": {
            "states": ["IDLE", "PROCESS", "FINISH"],
            "initial_state": "IDLE",
            "transitions": [
                {"from": "IDLE", "to": "PROCESS", "condition": {"expression": "1'b1", "is_fully_defined": True}},
                {"from": "PROCESS", "to": "FINISH", "condition": {"expression": "1'b1", "is_fully_defined": True}},
                {"from": "FINISH", "to": "IDLE", "condition": {"expression": "1'b1", "is_fully_defined": True}},
            ]
        },
        "output_driving_strategy": {
            "out_ready": "driven by state == IDLE"
        }
    }

    sv_code = get_heuristic_code_template("SystemVerilog", spec, rtl_plan=plan)

    # Must be valid SystemVerilog module
    assert "module packet_parser" in sv_code
    assert "endmodule" in sv_code

    # 1. No alu_result register
    assert "alu_result" not in sv_code

    # 2. No ALU operations (+, -, *, case on cmd)
    assert "case (cmd" not in sv_code
    assert "case (instr" not in sv_code
    assert "+ 32'd1" not in sv_code

    # 3. No phantom trigger from 'length' matching 'en'
    assert "if (length)" not in sv_code
    assert "if (length == 1" not in sv_code
    assert "length" not in sv_code.split("case (state)")[1]  # 'length' should not appear in FSM state logic


def test_02_acceptance_underspecified_spec_yields_gap_list_no_rtl():
    """
    Acceptance Criteria Test:
    An under-specified spec yields a structured gap list (SPECIFICATION_INCOMPLETE) and NO RTL.
    """
    spec = """# Hardware Design Specification: underspec_module
states: S_INIT, S_WORK, S_DONE
transitions:
S_INIT -> S_WORK: "start condition met"
S_WORK -> S_DONE: "work finishes"
S_DONE -> S_INIT: "ack received"

## Interfaces
| Signal Name | Direction | Width | Description |
| clk | input | 1 | Clock signal |
| rst_n | input | 1 | Reset signal |
| data_out | output | 16 | Output data |
"""
    result = get_heuristic_code_template("SystemVerilog", spec)

    # Must return SPECIFICATION_INCOMPLETE
    assert result.startswith("SPECIFICATION_INCOMPLETE")
    assert "Missing/ambiguous requirements:" in result
    assert "Required clarification:" in result

    # Must contain structured gap details (code, location, what is needed)
    assert "[Code:" in result
    assert "Location:" in result
    assert "Needed:" in result

    # Must contain NO fabricated RTL
    assert not re.search(r"^\s*module\s+[a-zA-Z0-9_]+", result, re.MULTILINE)
    assert "endmodule" not in result


def test_03_fallback_reads_validated_plan_transitions_not_raw_efs_ir_prose():
    """
    Requirement 3:
    The fallback must read the validated plan's transitions, not raw efs_ir.fsms[0].transitions
    where raw conditions are prose like 'when start is asserted'.
    """
    # Raw EFS IR with human-language prose condition
    raw_ir = EFSIR(
        metadata=EFSMetadata(design_name="fsm_trans_test"),
        signals=[
            EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset"),
            EFSSignal(name="start_pulse", direction="input", width="1"),
            EFSSignal(name="done_pulse", direction="output", width="1"),
        ],
        fsms=[
            EFSFSM(
                fsm_id="FSM_01",
                name="main_fsm",
                states=[
                    EFSState(name="IDLE"),
                    EFSState(name="ACTIVE"),
                ],
                transitions=[
                    # Prose condition in raw EFS IR that would break Verilog if emitted directly:
                    EFSTransition(
                        source_state="IDLE",
                        target_state="ACTIVE",
                        condition="when start_pulse is asserted high by host",
                    ),
                    EFSTransition(
                        source_state="ACTIVE",
                        target_state="IDLE",
                        condition="unconditional return on next clock cycle",
                    )
                ]
            )
        ]
    )

    # Validated plan with normalized, synthesizable boolean expressions
    validated_plan = {
        "module_name": "fsm_trans_test",
        "specification_status": {"status": "COMPLETE", "blocking_issues": []},
        "ports": [
            {"name": "clk", "direction": "input", "width": "1", "semantic_role": "clock"},
            {"name": "rst_n", "direction": "input", "width": "1", "semantic_role": "reset"},
            {"name": "start_pulse", "direction": "input", "width": "1"},
            {"name": "done_pulse", "direction": "output", "width": "1"},
        ],
        "clock_reset": {"clock_name": "clk", "reset_name": "rst_n", "reset_polarity": "low"},
        "fsm": {
            "states": ["IDLE", "ACTIVE"],
            "initial_state": "IDLE",
            "transitions": [
                {"from": "IDLE", "to": "ACTIVE", "condition": {"expression": "start_pulse == 1'b1", "is_fully_defined": True}, "source_ref": "FSM_01"},
                {"from": "ACTIVE", "to": "IDLE", "condition": {"expression": "1'b1", "is_fully_defined": True}, "source_ref": "FSM_01"},
            ]
        },
        "output_driving_strategy": {
            "done_pulse": "driven by state == ACTIVE"
        }
    }

    code = get_heuristic_code_template("SystemVerilog", "", efs_ir=raw_ir, rtl_plan=validated_plan)

    # The emitted Verilog must NOT contain the prose from raw efs_ir
    assert "when start_pulse is asserted" not in code
    assert "unconditional return" not in code

    # It MUST contain the validated boolean expression from validated_plan
    assert "start_pulse == 1'b1" in code
    assert "module fsm_trans_test" in code
    assert "endmodule" in code


def test_04_repair_retries_exhausted_returns_specification_incomplete():
    """
    Requirement 2:
    When generation fails after retries, return a SPECIFICATION_INCOMPLETE result
    that lists the concrete gaps (code, location, what is needed), never fabricated RTL.
    """
    spec = """
    # Hardware Design Specification: broken_alu
    ## Interfaces
    | Signal Name | Direction | Width | Description |
    | clk | input | 1 | Clock signal |
    | rst_n | input | 1 | Reset signal |
    | result | output | 32 | Result data |
    """

    mock_llm = MagicMock()
    mock_llm.available = True
    # LLM repeatedly produces invalid syntax / empty blocks that fail validation
    mock_llm.complete.return_value = """
    module broken_alu (input clk, input rst_n, output [31:0] result);
        always_comb begin
            // Placeholder: logic not implemented
        end
    endmodule
    """

    with patch("core.agents.rtl_agent.get_llm_client", return_value=mock_llm):
        res = generate_code(spec, "SystemVerilog")

        # Must return SPECIFICATION_INCOMPLETE, NOT fall back to fabricated heuristic RTL
        assert res.startswith("SPECIFICATION_INCOMPLETE")
        assert "Missing/ambiguous requirements:" in res
        assert "Required clarification:" in res

        # Must contain concrete gaps
        assert "[Code:" in res
        assert "Location:" in res
        assert "Needed:" in res

        # Must contain NO RTL
        assert "module broken_alu" not in res
        assert "endmodule" not in res


def test_05_no_mux_invented_from_sel_input():
    """
    Requirement 1:
    No mux fabricated from input named 'sel' / 'select'.
    """
    plan = {
        "module_name": "no_mux_invention",
        "specification_status": {"status": "COMPLETE", "blocking_issues": []},
        "ports": [
            {"name": "sel", "direction": "input", "width": "1"},
            {"name": "data_a", "direction": "input", "width": "8"},
            {"name": "data_b", "direction": "input", "width": "8"},
            {"name": "data_out", "direction": "output", "width": "8"},
        ],
        "output_driving_strategy": {
            "data_out": "8'h00"
        }
    }
    code = get_heuristic_code_template("SystemVerilog", "", rtl_plan=plan)
    assert "module no_mux_invention" in code
    # Must NOT invent 'sel ? data_b : data_a'
    assert "sel ?" not in code
    assert "? data_b" not in code


def test_06_no_axi_wstrb_addr_invented():
    """
    Requirement 1:
    No 'wstrb = 4'hF' or 'addr = 0' fabricated for ports named wstrb/addr.
    """
    plan = {
        "module_name": "no_axi_invention",
        "specification_status": {"status": "COMPLETE", "blocking_issues": []},
        "ports": [
            {"name": "wstrb", "direction": "output", "width": "4"},
            {"name": "addr", "direction": "output", "width": "32"},
        ],
        "output_driving_strategy": {
            "wstrb": "4'h0",
            "addr": "32'h0",
        }
    }
    code = get_heuristic_code_template("SystemVerilog", "", rtl_plan=plan)
    assert "module no_axi_invention" in code
    # Must NOT invent hardcoded 4'hF
    assert "4'hF" not in code


def test_07_format_specification_incomplete_result_structure():
    """
    Verify format_specification_incomplete_result produces structured output
    containing code, location, and what is needed for various issue types.
    """
    issues = [
        {
            "code": "RULE_UNDRIVEN_OUTPUT",
            "category": "INTERFACE",
            "location": "output port 'tx_valid'",
            "what_is_needed": "Specify driving logic expression for tx_valid.",
            "issue": "Output port 'tx_valid' is undriven.",
            "classification": "PLAN_MISSING",
            "evidence": "Plan interface"
        },
        {
            "code": "RULE_UNDEFINED_TRANSITION",
            "category": "FSM",
            "location": "transition RUN -> DONE",
            "what_is_needed": "Provide boolean trigger condition for transition.",
            "issue": "Transition condition is undefined.",
            "classification": "MISSING",
            "evidence": "FSM transition matrix"
        }
    ]

    out = format_specification_incomplete_result(issues, module_name="uart_tx")

    assert out.startswith("SPECIFICATION_INCOMPLETE")
    assert "Missing/ambiguous requirements:" in out
    assert "Required clarification:" in out

    # Check for code, location, and needed fields
    assert "[Code: RULE_UNDRIVEN_OUTPUT]" in out
    assert "Location: output port 'tx_valid'" in out
    assert "Needed: Specify driving logic expression for tx_valid." in out

    assert "[Code: RULE_UNDEFINED_TRANSITION]" in out
    assert "Location: transition RUN -> DONE" in out
    assert "Needed: Provide boolean trigger condition for transition." in out
