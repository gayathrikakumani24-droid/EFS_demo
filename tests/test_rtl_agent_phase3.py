"""Test Suite for Phase 3: Strict, Traceable Plan & Polarity Grounding.

Tests:
1. RTL plan schema validation rejects/repairs schema errors (missing keys, bad types) with PLAN_SCHEMA_VIOLATION.
2. Source ref validation rejects missing source_ref with PLAN_MISSING_SOURCE_REF.
3. Source ref validation rejects non-existent quote with PLAN_INVALID_SOURCE_REF.
4. Structured issue suppression suppresses only through tested rules keyed by issue code.
5. Reset polarity detection correctly identifies conventional names, descriptions, or raises GAP_RESET_POLARITY_UNDETERMINED.
"""

import pytest
from core.agents.rtl_agent import (
    validate_plan_schema,
    validate_plan_source_refs,
    detect_reset_polarity,
    check_issue_suppression,
    validate_rtl_plan,
    plan_rtl,
    generate_code
)
from core.efs_ir.models import EFSIR, EFSSignal, EFSRegister, EFSFSM, EFSState, EFSTransition


def test_phase3_plan_schema_validation():
    """Verify schema validation rejects malformed plans with code PLAN_SCHEMA_VIOLATION."""
    # 1. Non-dict plan
    valid, schema_obj, issues = validate_plan_schema("not a dict")
    assert not valid
    assert len(issues) > 0
    assert issues[0]["code"] == "PLAN_SCHEMA_VIOLATION"

    # 2. Dict missing required 'ports'
    malformed_plan = {
        "module_name": "test_mod"
        # missing ports
    }
    valid, schema_obj, issues = validate_plan_schema(malformed_plan)
    assert not valid
    assert any(i.get("code") == "PLAN_SCHEMA_VIOLATION" for i in issues)

    # 3. Valid minimal plan
    valid_plan = {
        "module_name": "test_mod",
        "ports": [
            {"name": "clk", "direction": "input", "width": "1"},
            {"name": "rst_n", "direction": "input", "width": "1"}
        ]
    }
    valid, schema_obj, issues = validate_plan_schema(valid_plan)
    assert valid
    assert len(issues) == 0
    assert schema_obj is not None
    assert schema_obj.module_name == "test_mod"


def test_phase3_source_ref_missing_and_invalid():
    """Verify that transitions, registers, and output rules without verifiable source_ref are rejected."""
    spec_text = """
    # Hardware Design Specification: counter_ctrl
    states: IDLE, RUN
    transitions:
    IDLE -> RUN: start == 1
    ## Interface
    - clk: input, 1 bit, Clock
    - rst_n: input, 1 bit, Active low reset
    - start: input, 1 bit, Start trigger
    - count_out: output, 8 bits, Counter output
    ## Registers
    - step_reg: 8 bits, Step size for counting
    """

    # Plan with missing source_ref on transition
    plan_missing_ref = {
        "module_name": "counter_ctrl",
        "ports": [
            {"name": "clk", "direction": "input", "width": "1"},
            {"name": "rst_n", "direction": "input", "width": "1"},
            {"name": "start", "direction": "input", "width": "1"},
            {"name": "count_out", "direction": "output", "width": "8"}
        ],
        "internal_registers": [
            {"name": "step_reg", "width": "8", "reset_value": "8'h1", "source_ref": "step size for counting"}
        ],
        "fsm": {
            "states": ["IDLE", "RUN"],
            "transitions": [
                {"from": "IDLE", "to": "RUN", "condition": {"expression": "start == 1"}}
                # No source_ref!
            ]
        }
    }
    issues = validate_plan_source_refs(plan_missing_ref, design_spec=spec_text)
    assert any(i.get("code") == "PLAN_MISSING_SOURCE_REF" for i in issues)

    # Plan with invented source_ref (quote not in spec text)
    plan_invalid_ref = {
        "module_name": "counter_ctrl",
        "ports": [
            {"name": "clk", "direction": "input", "width": "1"},
            {"name": "rst_n", "direction": "input", "width": "1"},
            {"name": "start", "direction": "input", "width": "1"},
            {"name": "count_out", "direction": "output", "width": "8"}
        ],
        "internal_registers": [
            {"name": "step_reg", "width": "8", "reset_value": "8'h1", "source_ref": "invented quote that does not exist"}
        ],
        "fsm": {
            "states": ["IDLE", "RUN"],
            "transitions": [
                {"from": "IDLE", "to": "RUN", "source_ref": "start == 1", "condition": {"expression": "start == 1"}}
            ]
        }
    }
    issues = validate_plan_source_refs(plan_invalid_ref, design_spec=spec_text)
    assert any(i.get("code") == "PLAN_INVALID_SOURCE_REF" for i in issues)

    # Plan with valid source_ref from spec text
    plan_valid_ref = {
        "module_name": "counter_ctrl",
        "ports": [
            {"name": "clk", "direction": "input", "width": "1"},
            {"name": "rst_n", "direction": "input", "width": "1"},
            {"name": "start", "direction": "input", "width": "1"},
            {"name": "count_out", "direction": "output", "width": "8"}
        ],
        "internal_registers": [
            {"name": "step_reg", "width": "8", "reset_value": "8'h1", "source_ref": "step size for counting"}
        ],
        "fsm": {
            "states": ["IDLE", "RUN"],
            "transitions": [
                {"from": "IDLE", "to": "RUN", "source_ref": "start == 1", "condition": {"expression": "start == 1"}}
            ]
        }
    }
    issues = validate_plan_source_refs(plan_valid_ref, design_spec=spec_text)
    assert len(issues) == 0


def test_phase3_source_ref_against_efs_ir():
    """Verify source_ref validation checks against EFS IR element IDs and evidence."""
    efs_ir = EFSIR(
        signals=[
            EFSSignal(name="req", direction="input", width="1", signal_id="SIG_REQ_001", description="Request line")
        ],
        registers=[
            EFSRegister(name="cfg_reg", width="32", register_id="REG_CFG_001", description="Configuration Register")
        ],
        fsms=[
            EFSFSM(
                name="main_fsm",
                fsm_id="FSM_MAIN_001",
                states=[EFSState(name="IDLE", state_id="STATE_IDLE_001")],
                transitions=[
                    EFSTransition(source_state="IDLE", target_state="BUSY", transition_id="TRANS_001", evidence="req asserted")
                ]
            )
        ]
    )

    plan = {
        "module_name": "test_mod",
        "ports": [{"name": "req", "direction": "input", "width": "1"}],
        "internal_registers": [
            {"name": "cfg_reg", "width": "32", "reset_value": "32'h0", "source_ref": "REG_CFG_001"}
        ],
        "fsm": {
            "states": ["IDLE", "BUSY"],
            "transitions": [
                {"from": "IDLE", "to": "BUSY", "source_ref": "TRANS_001", "condition": {"expression": "req"}}
            ]
        }
    }
    issues = validate_plan_source_refs(plan, design_spec="", efs_ir=efs_ir)
    assert len(issues) == 0


def test_phase3_structured_issue_suppression():
    """Verify issue suppression works strictly on registered codes and rules."""
    ctx = {
        "has_fsm_in_spec": True,
        "has_timer_reg": True,
        "has_normalized_port_assertion": True,
        "has_authoritative_direction": True
    }

    # 1. Registered dwell code
    dwell_issue = {
        "code": "ISSUE_DWELL_CYCLE_UNCONDITIONAL",
        "issue": "Unconditional transition to next clock cycle"
    }
    suppressed, reason = check_issue_suppression(dwell_issue, ctx)
    assert suppressed is True
    assert "dwell" in reason

    # 2. Registered timer duration code
    timer_issue = {
        "code": "ISSUE_TIMER_DURATION_SYNTHESIZED",
        "issue": "FSM timing wait is synthesized"
    }
    suppressed, reason = check_issue_suppression(timer_issue, ctx)
    assert suppressed is True
    assert "timer" in reason

    # 3. Unregistered code must NOT be suppressed
    unknown_issue = {
        "code": "ISSUE_ARBITRARY_BLOCKING_ERROR",
        "issue": "Critical spec missing"
    }
    suppressed, reason = check_issue_suppression(unknown_issue, ctx)
    assert suppressed is False
    assert reason is None


def test_phase3_reset_polarity_detection():
    """Verify reset polarity detection distinguishes active-low/high and raises a gap for undetermined."""
    # 1. Exact conventional active-low names
    for name in ["rst_n", "reset_n", "aresetn", "sys_rst_n", "rst_b", "rst_l"]:
        pol, gap = detect_reset_polarity(name)
        assert pol == "active_low"
        assert gap is None

    # 2. Exact conventional active-high names
    for name in ["rst", "reset", "sys_rst", "por", "areset", "rst_p"]:
        pol, gap = detect_reset_polarity(name)
        assert pol == "active_high"
        assert gap is None

    # 3. Unconventional name with explicit description
    pol, gap = detect_reset_polarity("reset_ctrl", description="active low system reset")
    assert pol == "active_low"
    assert gap is None

    pol, gap = detect_reset_polarity("reset_ctrl", description="active high power-on reset")
    assert pol == "active_high"
    assert gap is None

    # 4. Ambiguous name with no description raises gap GAP_RESET_POLARITY_UNDETERMINED
    pol, gap = detect_reset_polarity("rst_in")
    assert pol is None
    assert gap is not None
    assert gap["code"] == "GAP_RESET_POLARITY_UNDETERMINED"
    assert gap["is_blocking"] is True

    # 5. Non-reset signal containing letter 'n' should raise gap if tested as reset
    pol, gap = detect_reset_polarity("pin_ctrl")
    assert pol is None
    assert gap is not None
    assert gap["code"] == "GAP_RESET_POLARITY_UNDETERMINED"


def test_phase3_validate_rtl_plan_rejects_missing_source_ref():
    """Verify validate_rtl_plan blocks plan when source_ref is omitted."""
    spec_text = """
    # Hardware Design Specification: sample_ctrl
    states: IDLE, RUN
    transitions:
    IDLE -> RUN: start == 1
    ## Interface
    - clk: input, 1 bit, Clock
    - rst_n: input, 1 bit, Active low reset
    - start: input, 1 bit, Start trigger
    - ready: output, 1 bit, Ready status
    """
    plan = {
        "module_name": "sample_ctrl",
        "ports": [
            {"name": "clk", "direction": "input", "width": "1"},
            {"name": "rst_n", "direction": "input", "width": "1"},
            {"name": "start", "direction": "input", "width": "1"},
            {"name": "ready", "direction": "output", "width": "1"}
        ],
        "fsm": {
            "states": ["IDLE", "RUN"],
            "transitions": [
                # Missing source_ref
                {"from": "IDLE", "to": "RUN", "condition": {"expression": "start == 1"}}
            ]
        }
    }
    val = validate_rtl_plan(plan, design_spec=spec_text)
    assert val["valid"] is False
    assert val["status"] == "INCOMPLETE"
    assert any(i.get("code") == "PLAN_MISSING_SOURCE_REF" for i in val["issues"])
