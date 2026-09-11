"""
Regression Test Suite for Requirement-Driven Design Pipeline.

Tests complete specification validation, anti-hardcoding FSM validation,
negative incomplete specification detection, datapath validation, reset polarity,
and unified RTL/PlantUML design model grounding.
"""

import pytest
from typing import Dict, Any
from core.efs_ir.models import EFSIR, EFSComponent, EFSInterface, EFSSignal, EFSRegister, EFSMetadata
from core.agents.rtl_agent import plan_rtl, validate_rtl_plan, generate_code
from core.agents.plantuml_agent import generate_diagrams


@pytest.fixture(autouse=True)
def mock_llm_default(monkeypatch):
    class MockLLM:
        available = False
        def complete_text(self, *args, **kwargs):
            return None
        def complete_json(self, *args, **kwargs):
            return None

    monkeypatch.setattr("core.agents.rtl_agent.get_llm_client", lambda: MockLLM())
    monkeypatch.setattr("core.agents.plantuml_agent.get_llm_client", lambda: MockLLM())


def test_1_complete_specification():
    """TEST 1 — Complete Register-Controlled Data Engine Specification.
    
    Must pass requirement-driven validation cleanly without demanding hardcoded FSMs.
    """
    spec_text = """
    # Hardware Design Specification: data_engine_controller
    
    ## Interface
    | Name | Width | Direction | Description |
    | aclk | 1 | input | System Clock |
    | aresetn | 1 | input | Active-low Reset |
    | s_axi_awvalid | 1 | input | AXI Write Address Valid |
    | s_axi_awready | 1 | output | AXI Write Address Ready |
    | s_axi_wdata | 32 | input | AXI Write Data |
    | s_axi_wvalid | 1 | input | AXI Write Data Valid |

    ## Registers
    | Name | Offset | Access | Description |
    | CONTROL | 0x00 | RW | Bit[0]: ENABLE, Bit[1]: START |
    | STATUS | 0x04 | RO | Bit[0]: BUSY, Bit[1]: DONE, Bit[2]: ERROR |
    | DATA | 0x08 | RW | Bit[31:0]: Input Data Payload |

    ## Processing Logic
    RESULT = DATA + 1
    """

    plan = plan_rtl(spec_text, "Verilog")
    val = validate_rtl_plan(plan)

    assert val["valid"] is True
    assert val["status"] == "COMPLETE"
    assert len(val["issues"]) == 0


def test_2_intentionally_incomplete_specification():
    """TEST 2 — Negative Test: Explicit vague FSM transition condition.
    
    Must detect genuine missing/ambiguous requirement.
    """
    spec_text = """
    # Hardware Design Specification: vague_fsm_module
    states: IDLE, RUN, DONE

    transitions:
    IDLE -> RUN: "control start condition met"
    RUN -> DONE: "operation completes"

    ## Interface
    - clk: input, 1 bit, System Clock
    - rst_n: input, 1 bit, Active-low Reset
    """

    plan = plan_rtl(spec_text, "Verilog")
    val = validate_rtl_plan(plan)

    assert val["valid"] is False
    assert val["status"] == "INCOMPLETE"
    assert len(val["issues"]) > 0
    assert any(i["category"] == "FSM" and i["classification"] == "AMBIGUOUS" for i in val["issues"])


def test_3_anti_hardcoding_custom_fsm_names():
    """TEST 3 — Custom FSM state names (IDLE, LOAD_DATA, CALCULATE, WRITE_RESULT).
    
    Validator must NOT demand LOAD/EXECUTE/FINISH/WAIT_ACK or fail on custom states.
    """
    spec_text = """
    # Hardware Design Specification: custom_fsm_engine
    states: IDLE, LOAD_DATA, CALCULATE, WRITE_RESULT

    transitions:
    IDLE -> LOAD_DATA: start == 1
    LOAD_DATA -> CALCULATE: load_done == 1
    CALCULATE -> WRITE_RESULT: calc_done == 1
    WRITE_RESULT -> IDLE: ack == 1

    ## Interface
    - clk: input, 1 bit, Clock
    - rst_n: input, 1 bit, Reset
    - start: input, 1 bit, Start signal
    - load_done: input, 1 bit, Load done signal
    - calc_done: input, 1 bit, Calc done signal
    - ack: input, 1 bit, Ack signal
    """

    plan = plan_rtl(spec_text, "SystemVerilog")
    val = validate_rtl_plan(plan)

    assert val["valid"] is True
    assert val["status"] == "COMPLETE"
    assert plan["fsm"]["states"] == ["IDLE", "LOAD_DATA", "CALCULATE", "WRITE_RESULT"]


def test_4_behavioral_fsm_without_explicit_states():
    """TEST 4 — Behavioral specification without explicit state names.
    
    Validator must recognize behavioral requirement and treat internal FSM as IMPLEMENTATION_CHOICE.
    """
    spec_text = """
    # Hardware Design Specification: behavioral_adder
    
    ## Functional Requirement
    When start is high, the module computes RESULT = DATA + 1 and asserts done when finished.

    ## Interface
    - clk: input, 1 bit, System Clock
    - rst_n: input, 1 bit, Reset
    - start: input, 1 bit, Start trigger
    - DATA: input, 32 bits, Input Data
    - RESULT: output, 32 bits, Output Data
    - done: output, 1 bit, Done indicator
    """

    plan = plan_rtl(spec_text, "Verilog")
    val = validate_rtl_plan(plan)

    assert val["valid"] is True
    assert val["status"] == "COMPLETE"


def test_5_axi_channel_fsms_and_derived_signals():
    """TEST 5 — AXI Channel FSMs and reg_write_en treated as derived implementation choices.
    
    Validator must NOT flag reg_write_en or 5 AXI channel FSMs as missing user requirements.
    """
    spec_text = """
    # Hardware Design Specification: axi_slave_controller
    
    ## Interface
    - aclk: input, 1 bit, System Clock
    - aresetn: input, 1 bit, Reset
    - s_axi_awvalid: input, 1 bit, AWVALID
    - s_axi_awready: output, 1 bit, AWREADY
    - s_axi_wdata: input, 32 bits, WDATA
    - s_axi_wvalid: input, 1 bit, WVALID

    ## Registers
    - CONTROL: 32 bits, Offset 0x00
    """

    plan = plan_rtl(spec_text, "Verilog")
    val = validate_rtl_plan(plan)

    assert val["valid"] is True
    assert val["status"] == "COMPLETE"
    # Ensure no blocking issue claims reg_write_en is missing
    assert not any("reg_write_en" in i.get("issue", "") for i in val["issues"])


def test_6_unified_rtl_and_plantuml_design_model():
    """TEST 6 — Unified RTL and PlantUML generation grounded in same design model."""
    spec_text = """
    # Hardware Design Specification: unified_core
    
    ## Interface
    - clk: input, 1 bit, Clock
    - rst_n: input, 1 bit, Reset
    - in_data: input, 32 bits, Input
    - out_data: output, 32 bits, Output
    """

    plan = plan_rtl(spec_text, "Verilog")
    val = validate_rtl_plan(plan)
    assert val["valid"] is True

    puml_diagrams = generate_diagrams(spec_text, plan)
    assert "architecture" in puml_diagrams
    assert "fsm" in puml_diagrams
    assert "sequence" in puml_diagrams
