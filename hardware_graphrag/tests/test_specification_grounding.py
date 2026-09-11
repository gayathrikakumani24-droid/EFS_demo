"""Specification Grounding Test Suite.

Tests specification completeness validation, Stage 2 blocking, structured RTL plan
grounding, and anti-hallucination invariants in core/agents/rtl_agent.py.
"""

import pytest
from typing import Optional, Dict, Any
from core.efs_ir.models import EFSIR, EFSSignal, EFSComponent, EFSFSM, EFSState, EFSTransition, EFSRegister
from core.agents.rtl_agent import plan_rtl, validate_rtl_plan, generate_code, get_heuristic_code_template


@pytest.fixture(autouse=True)
def mock_llm_default(monkeypatch):
    class MockLLM:
        available = False
        def complete_text(self, *args, **kwargs):
            return None
        def complete_json(self, *args, **kwargs):
            return None

    monkeypatch.setattr("core.agents.rtl_agent.get_llm_client", lambda: MockLLM())


def test_grounding_1_incomplete_fsm():
    """TEST 1 — INCOMPLETE FSM

    Vague FSM transition conditions must block Stage 2 RTL generation and return SPECIFICATION_INCOMPLETE.
    """
    spec_text = """
    # Hardware Design Specification: fsm_controller
    states: IDLE, RUN, DONE

    transitions:
    IDLE -> RUN: "control start condition met"
    RUN -> DONE: "internal counter/operation completes"
    DONE -> IDLE: "transaction acknowledged or reset"

    ## Interface
    - clk: input, 1 bit, System clock
    - rst_n: input, 1 bit, Active low reset
    - valid: input, 1 bit, Data valid
    - ready: output, 1 bit, Device ready
    """
    
    rtl_plan = plan_rtl(spec_text, "SystemVerilog")
    validation = validate_rtl_plan(rtl_plan)
    
    assert validation["valid"] is False
    assert validation["status"] == "INCOMPLETE"
    assert len(validation["issues"]) > 0
    assert any(i["category"] == "FSM" for i in validation["issues"])
    
    code = generate_code(spec_text, "SystemVerilog")
    assert code.startswith("SPECIFICATION_INCOMPLETE")
    assert "Missing/ambiguous requirements:" in code
    assert "control start condition met" in code or "transition condition is undefined" in code


def test_grounding_2_complete_fsm(monkeypatch):
    """TEST 2 — COMPLETE FSM

    Explicit signal conditions for all transitions must pass completeness validation.
    """
    spec_text = """
    # Hardware Design Specification: complete_fsm_controller
    states: IDLE, RUN, DONE

    transitions:
    IDLE -> RUN: start == 1
    RUN -> DONE: counter == 8'd255
    DONE -> IDLE: valid && ready

    ## Interface
    - clk: input, 1 bit, System clock
    - rst_n: input, 1 bit, Active low reset
    - start: input, 1 bit, Operation start pulse
    - counter: input, 8 bits, Internal cycle counter
    - valid: input, 1 bit, Data valid
    - ready: output, 1 bit, Ready signal
    """
    
    class MockLLM:
        available = True
        def complete_text(self, *args, **kwargs):
            return "module complete_fsm_controller(\n  input logic clk,\n  input logic rst_n\n);\nendmodule"
        def complete_json(self, *args, **kwargs):
            return None

    monkeypatch.setattr("core.agents.rtl_agent.get_llm_client", lambda: MockLLM())

    rtl_plan = plan_rtl(spec_text, "SystemVerilog")
    validation = validate_rtl_plan(rtl_plan)
    
    assert validation["valid"] is True
    assert validation["status"] == "COMPLETE"
    assert len(validation["issues"]) == 0
    
    code = generate_code(spec_text, "SystemVerilog")
    assert not code.startswith("SPECIFICATION_INCOMPLETE")
    assert "module complete_fsm_controller" in code


def test_grounding_3_unspecified_register_bit():
    """TEST 3 — UNSPECIFIED REGISTER BIT

    Referencing ctrl_reg[0] without explicitly defining bit field semantics must trigger INCOMPLETE.
    """
    spec_text = """
    # Hardware Design Specification: reg_bit_controller
    states: IDLE, RUN

    transitions:
    IDLE -> RUN: ctrl_reg[0] == 1

    ## Interface
    - clk: input, 1 bit, Clock
    - rst_n: input, 1 bit, Reset

    ## Registers
    - ctrl_reg: 32 bits, Control register
    """
    
    rtl_plan = plan_rtl(spec_text, "SystemVerilog")
    validation = validate_rtl_plan(rtl_plan)
    
    assert validation["valid"] is False
    assert validation["status"] == "INCOMPLETE"
    assert any(i["category"] == "REGISTER" for i in validation["issues"])
    
    code = generate_code(spec_text, "SystemVerilog")
    assert code.startswith("SPECIFICATION_INCOMPLETE")
    assert "ctrl_reg" in code


def test_grounding_4_valid_ready_must_not_become_start():
    """TEST 4 — VALID/READY MUST NOT BECOME START

    The presence of valid and ready signals must NOT cause the validator to infer valid && ready == start
    when the spec says "control start condition met".
    """
    spec_text = """
    # Hardware Design Specification: handshake_start_module
    states: IDLE, RUN

    transitions:
    IDLE -> RUN: "control start condition met"

    ## Interface
    - clk: input, 1 bit, Clock
    - rst_n: input, 1 bit, Reset
    - valid: input, 1 bit, Transfer valid
    - ready: output, 1 bit, Transfer ready
    """
    
    protocol_rules = [
        {"citation": "Handshake Spec Section 3.1", "text": "Transfer occurs when VALID && READY"}
    ]
    
    rtl_plan = plan_rtl(spec_text, "SystemVerilog")
    validation = validate_rtl_plan(rtl_plan)
    
    assert validation["valid"] is False
    assert validation["status"] == "INCOMPLETE"
    
    code = generate_code(spec_text, "SystemVerilog", protocol_rules=protocol_rules)
    assert code.startswith("SPECIFICATION_INCOMPLETE")


def test_grounding_5_contradictory_interface():
    """TEST 5 — CONTRADICTORY INTERFACE

    Signal direction mismatches between EFS IR and protocol contract must trigger CONTRADICTORY and block RTL.
    """
    spec_text = """
    # Hardware Design Specification: contradictory_module
    states: IDLE, RUN
    transitions:
    IDLE -> RUN: start == 1

    ## Interface
    - clk: input, 1 bit, Clock
    - rst_n: input, 1 bit, Reset
    - start: input, 1 bit, Start signal
    - ready: output, 1 bit, Ready output
    """
    
    ir = EFSIR()
    ir.signals.append(EFSSignal(name="clk", width="1", direction="input", semantic_role="clock"))
    ir.signals.append(EFSSignal(name="rst_n", width="1", direction="input", semantic_role="reset"))
    ir.signals.append(EFSSignal(name="start", width="1", direction="input", semantic_role="data"))
    ir.signals.append(EFSSignal(name="ready", width="1", direction="input", semantic_role="handshake"))
    
    rtl_plan = plan_rtl(spec_text, "SystemVerilog", efs_ir=ir)
    validation = validate_rtl_plan(rtl_plan)
    
    assert validation["valid"] is False
    assert validation["status"] == "CONTRADICTORY"
    assert any("direction conflict" in i["issue"].lower() for i in validation["issues"])
    
    code = generate_code(spec_text, "SystemVerilog", efs_ir=ir)
    assert code.startswith("SPECIFICATION_INCOMPLETE") or "CONTRADICTORY" in code or "direction conflict" in code


def test_grounding_6_llm_failure_on_valid_spec():
    """TEST 6 — LLM FAILURE ON VALID SPEC

    A complete, valid specification with simulated LLM failure MUST execute heuristic fallback,
    because completeness validation already passed.
    """
    spec_text = """
    # Hardware Design Specification: valid_fallback_module
    states: IDLE, RUN, DONE

    transitions:
    IDLE -> RUN: start == 1
    RUN -> DONE: counter == 8'd255
    DONE -> IDLE: valid && ready

    ## Interface
    - clk: input, 1 bit, System clock
    - rst_n: input, 1 bit, Active low reset
    - start: input, 1 bit, Operation start pulse
    - counter: input, 8 bits, Cycle counter
    - valid: input, 1 bit, Data valid
    - ready: output, 1 bit, Ready signal
    """
    
    code = generate_code(spec_text, "SystemVerilog")
    
    assert not code.startswith("SPECIFICATION_INCOMPLETE")
    assert "module valid_fallback_module" in code or "module target_module" in code or "module " in code
    assert "always_ff" in code or "always" in code
