"""Test Suite for Phase 4: Real Parsing of Conditions / Actions & Priority Masking.

Requirements:
1. Verilog expression parser (tokenizer + precedence parser):
   - identifiers, sized/unsized literals, ! ~ - reductions, arithmetic, shifts, comparisons, & ^ | && ||, ?:, bit/part select, concatenation.
   - verify every identifier is a declared port, register, parameter, or state.
   - reject keywords and unknown names.
2. Every transition condition/action in the plan must parse.
   - natural-language conditions get a bounded re-formalisation step: send LLM condition text + declared signals, ask for expression, re-parse.
   - after N failures, raise a gap (GAP_UNPARSABLE_CONDITION). Prose never reaches the code.
3. Do not auto-interpret 'X asserted/high' for active-low names (_n, _b, _l):
   - raise GAP_ACTIVE_LOW_ASSERTION_AMBIGUOUS instead.
4. Mask transition actions by priority:
   - an action on transition k must fire only when its source state matches, its condition is true, and no earlier transition from that state fires.
5. Comparison vs assignment confusion:
   - 'a <= b' state assignment in condition or 'a = b' single '=' handled by tests.
"""

import pytest
from core.agents.expression_parser import (
    VerilogExpressionParser,
    check_active_low_assertion,
    check_comparison_assignment_confusion,
    compute_priority_masked_conditions,
)
from core.agents.rtl_agent import (
    plan_rtl,
    validate_rtl_plan,
    generate_code,
    get_heuristic_code_template,
)


def test_phase4_expression_parser_literals_and_identifiers():
    """Verify tokenizer and parser handle sized/unsized literals and declared identifiers."""
    symbols = {"clk", "rst_n", "start", "ready", "count", "data_in", "data_out", "IDLE", "RUN"}

    # Sized literals
    valid, _, refs, err = VerilogExpressionParser.parse_condition("start == 1'b1", symbols)
    assert valid and refs == ["start"]

    valid, _, refs, err = VerilogExpressionParser.parse_condition("count == 8'hFF", symbols)
    assert valid and refs == ["count"]

    valid, _, refs, err = VerilogExpressionParser.parse_condition("data_in == 32'd1000", symbols)
    assert valid and refs == ["data_in"]

    valid, _, refs, err = VerilogExpressionParser.parse_condition("data_in == 4'b00_10", symbols)
    assert valid and refs == ["data_in"]

    # Unsized literals
    valid, _, refs, err = VerilogExpressionParser.parse_condition("count == '0", symbols)
    assert valid and refs == ["count"]

    valid, _, refs, err = VerilogExpressionParser.parse_condition("count == 10", symbols)
    assert valid and refs == ["count"]


def test_phase4_expression_parser_operators_and_precedence():
    """Verify reductions, arithmetic, shifts, comparisons, logic, ternary, part-select, concat."""
    symbols = {"a", "b", "c", "d", "sel", "val", "data"}

    # Unary & reductions
    valid, _, refs, err = VerilogExpressionParser.parse_condition("&a && |b && ^c && !d && ~val", symbols)
    assert valid and set(refs) == {"a", "b", "c", "d", "val"}

    # Arithmetic and shifts
    valid, _, refs, err = VerilogExpressionParser.parse_condition("(a + b * c) > (d << 2)", symbols)
    assert valid and set(refs) == {"a", "b", "c", "d"}

    # Comparisons
    valid, _, refs, err = VerilogExpressionParser.parse_condition("a <= b && c >= d && a != b && c == d", symbols)
    assert valid and set(refs) == {"a", "b", "c", "d"}

    # Ternary operator
    valid, _, refs, err = VerilogExpressionParser.parse_condition("(sel ? a : b) == c", symbols)
    assert valid and set(refs) == {"sel", "a", "b", "c"}

    # Bit-select and part-select
    valid, _, refs, err = VerilogExpressionParser.parse_condition("data[7:0] == a[3 +: 8]", symbols)
    assert valid and set(refs) == {"data", "a"}

    # Concatenation
    valid, _, refs, err = VerilogExpressionParser.parse_condition("{a, b} == {2{c}}", symbols)
    assert valid and set(refs) == {"a", "b", "c"}


def test_phase4_reject_keywords_and_unknown_identifiers():
    """Verify parser rejects Verilog reserved keywords and undeclared identifiers."""
    symbols = {"clk", "rst_n", "valid"}

    # Keyword rejection
    valid, _, _, err = VerilogExpressionParser.parse_condition("wire == 1", symbols)
    assert not valid
    assert "Verilog keyword 'wire'" in err

    valid, _, _, err = VerilogExpressionParser.parse_condition("module == 1", symbols)
    assert not valid
    assert "Verilog keyword 'module'" in err

    # Unknown identifier rejection
    valid, _, _, err = VerilogExpressionParser.parse_condition("unregistered_signal == 1", symbols)
    assert not valid
    assert "Unknown identifier 'unregistered_signal'" in err


def test_phase4_action_parsing_and_writable_check():
    """Verify action parsing for non-blocking and blocking assignments and input port rejection."""
    symbols = {"clk", "rst_n", "start", "ready", "count"}
    writable = {"ready", "count"}  # start is input, not writable

    # Valid non-blocking and blocking actions
    valid, nodes, refs, err = VerilogExpressionParser.parse_action("ready <= 1'b1; count <= count + 1;", symbols, writable)
    assert valid
    assert len(nodes) == 2
    assert "ready" in refs and "count" in refs

    # Rejection of action assigning to input port
    valid, nodes, refs, err = VerilogExpressionParser.parse_action("start <= 1'b0;", symbols, writable)
    assert not valid
    assert "input port and cannot be assigned" in err

    # Rejection of unknown identifier in action
    valid, nodes, refs, err = VerilogExpressionParser.parse_action("unknown_reg <= 1'b1;", symbols, writable)
    assert not valid
    assert "Unknown identifier 'unknown_reg'" in err


def test_phase4_comparison_assignment_confusion():
    """Verify comparison vs assignment confusion detection (e.g. 'a <= b' state assignment in condition)."""
    states = {"IDLE", "RUN", "DONE"}

    # State assignment mistakenly used as condition: `state <= RUN`
    err = check_comparison_assignment_confusion("state <= RUN", states)
    assert err is not None
    assert "Comparison/assignment confusion" in err
    assert "non-blocking state assignment" in err

    # Single '=' in condition: `count = 10`
    err = check_comparison_assignment_confusion("count = 10", states)
    assert err is not None
    assert "Comparison/assignment confusion: single '='" in err

    # Valid comparison `count <= 10` is NOT confused with state assignment
    err = check_comparison_assignment_confusion("count <= 10", states)
    assert err is None


def test_phase4_active_low_assertion_raises_gap():
    """Verify 'X asserted/high' for active-low names (_n, _b, _l) raises GAP_ACTIVE_LOW_ASSERTION_AMBIGUOUS."""
    signals = [
        {"name": "clk", "direction": "input"},
        {"name": "rst_n", "direction": "input"},
        {"name": "intr_b", "direction": "input"},
        {"name": "cs_l", "direction": "input"},
        {"name": "start", "direction": "input"}
    ]

    # rst_n asserted
    gap = check_active_low_assertion("rst_n asserted", signals)
    assert gap is not None
    assert gap["code"] == "GAP_ACTIVE_LOW_ASSERTION_AMBIGUOUS"
    assert "rst_n" in gap["issue"]

    # intr_b high
    gap = check_active_low_assertion("intr_b is high", signals)
    assert gap is not None
    assert gap["code"] == "GAP_ACTIVE_LOW_ASSERTION_AMBIGUOUS"
    assert "intr_b" in gap["issue"]

    # Normal active-high signal does not trigger active-low gap
    gap = check_active_low_assertion("start asserted", signals)
    assert gap is None


def test_phase4_prose_reformalisation_and_unparsable_gap(monkeypatch):
    """Verify natural-language conditions undergo bounded re-formalisation and raise GAP_UNPARSABLE_CONDITION if exhausted."""
    spec_text = """
    # Hardware Design Specification: test_fsm
    states: IDLE, DONE
    transitions:
    IDLE -> DONE: when data reception completes
    ## Interface
    - clk: input, 1 bit, Clock
    - rst_n: input, 1 bit, Reset
    - rx_done: input, 1 bit, Receive done
    """

    # 1. LLM fails or is unavailable: raises GAP_UNPARSABLE_CONDITION; prose never reaches code
    class FailingLLM:
        available = False
        def complete_json(self, *args, **kwargs):
            return None
        def complete_text(self, *args, **kwargs):
            return None

    monkeypatch.setattr("core.agents.rtl_agent.get_llm_client", lambda: FailingLLM())

    plan = {
        "module_name": "test_fsm",
        "ports": [
            {"name": "clk", "direction": "input", "width": "1"},
            {"name": "rst_n", "direction": "input", "width": "1"},
            {"name": "rx_done", "direction": "input", "width": "1"}
        ],
        "fsm": {
            "states": ["IDLE", "DONE"],
            "transitions": [
                {
                    "from": "IDLE",
                    "to": "DONE",
                    "source_ref": "when data reception completes",
                    "condition": {"expression": "when data reception completes"}
                }
            ]
        }
    }
    val = validate_rtl_plan(plan, design_spec=spec_text)
    assert val["valid"] is False
    assert any(i.get("code") == "GAP_UNPARSABLE_CONDITION" for i in val["issues"])

    # 2. LLM succeeds with formal expression: condition parses and passes
    class SucceedingLLM:
        available = True
        def complete_json(self, *args, **kwargs):
            return {"verilog_expression": "rx_done == 1'b1"}
        def complete_text(self, *args, **kwargs):
            return "module test_fsm;\nendmodule"

    monkeypatch.setattr("core.agents.rtl_agent.get_llm_client", lambda: SucceedingLLM())

    plan2 = {
        "module_name": "test_fsm",
        "ports": [
            {"name": "clk", "direction": "input", "width": "1"},
            {"name": "rst_n", "direction": "input", "width": "1"},
            {"name": "rx_done", "direction": "input", "width": "1"}
        ],
        "fsm": {
            "states": ["IDLE", "DONE"],
            "transitions": [
                {
                    "from": "IDLE",
                    "to": "DONE",
                    "source_ref": "when data reception completes",
                    "condition": {"expression": "when data reception completes"}
                }
            ]
        }
    }
    val_succeed = validate_rtl_plan(plan2, design_spec=spec_text)
    assert val_succeed["valid"] is True
    assert val_succeed["status"] == "COMPLETE"
    assert plan2["fsm"]["transitions"][0]["condition"]["expression"] == "rx_done == 1'b1"


def test_phase4_mask_transition_actions_by_priority():
    """Verify transition actions are masked by priority so action k fires only when earlier transitions do not."""
    transitions_by_state = {
        "IDLE": [
            {"from": "IDLE", "to": "STATE_A", "condition": {"expression": "req_a == 1"}, "action": "grant_a <= 1'b1", "source_ref": "req_a == 1"},
            {"from": "IDLE", "to": "STATE_B", "condition": {"expression": "req_b == 1"}, "action": "grant_b <= 1'b1", "source_ref": "req_b == 1"},
            {"from": "IDLE", "to": "IDLE", "condition": {"expression": "1'b1"}, "action": "idle_cnt <= idle_cnt + 1", "source_ref": "1'b1"}
        ],
        "STATE_A": [
            {"from": "STATE_A", "to": "IDLE", "condition": {"expression": "1'b1"}, "source_ref": "1'b1"}
        ],
        "STATE_B": [
            {"from": "STATE_B", "to": "IDLE", "condition": {"expression": "1'b1"}, "source_ref": "1'b1"}
        ]
    }

    masked = compute_priority_masked_conditions(transitions_by_state)
    idle_trans = masked["IDLE"]

    # Transition 0: condition is req_a == 1
    assert idle_trans[0]["priority_mask"] == "(req_a == 1)"

    # Transition 1: must be masked by !(req_a == 1)
    assert "!(req_a == 1)" in idle_trans[1]["priority_mask"]
    assert "(req_b == 1)" in idle_trans[1]["priority_mask"]

    # Transition 2: must be masked by both prior conditions
    assert "!(req_a == 1)" in idle_trans[2]["priority_mask"]
    assert "!(req_b == 1)" in idle_trans[2]["priority_mask"]

    # Check heuristic code generation: actions are emitted within priority-masked if-else structure
    all_transitions = idle_trans + masked["STATE_A"] + masked["STATE_B"]
    spec_text = """
    # Hardware Design Specification: arbiter
    states: IDLE, STATE_A, STATE_B
    transitions:
    req_a == 1
    req_b == 1
    1'b1
    idle counter
    """
    plan = {
        "module_name": "arbiter",
        "ports": [
            {"name": "clk", "direction": "input", "width": "1"},
            {"name": "rst_n", "direction": "input", "width": "1"},
            {"name": "req_a", "direction": "input", "width": "1"},
            {"name": "req_b", "direction": "input", "width": "1"},
            {"name": "grant_a", "direction": "output", "width": "1"},
            {"name": "grant_b", "direction": "output", "width": "1"}
        ],
        "registers": [
            {"name": "idle_cnt", "width": "8", "source_ref": "idle counter"}
        ],
        "fsm": {
            "states": ["IDLE", "STATE_A", "STATE_B"],
            "transitions": all_transitions
        }
    }
    code = get_heuristic_code_template("SystemVerilog", design_spec=spec_text, rtl_plan=plan)
    assert "if (req_a == 1) begin" in code
    assert "grant_a <= 1'b1;" in code
    assert "else if (req_b == 1) begin" in code
    assert "grant_b <= 1'b1;" in code
