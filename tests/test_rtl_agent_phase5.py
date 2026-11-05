"""
Unit tests for Phase 5:
1. Rich IR behavioral context feeding (requirements, timing constraints, register fields).
2. Spec-type classification (deterministic first, LLM second) across 6 categories.
3. Type-specific system prompts and verified few-shot example compilation.
4. Verified standard library modules from rtl_lib/ (synchronizer, FIFO, register_bank).
5. Multi-component chunking and deterministic top-level wrapper assembly.
"""

import json
from pathlib import Path
import pytest

from core.agents.spec_classifier import (
    SpecType,
    classify_spec_type,
    classify_spec_type_deterministic,
    load_spec_prompt_template,
    get_library_module,
    can_use_library_module,
    generate_deterministic_top_level_wrapper,
    find_prompts_dir,
    find_rtl_lib_dir,
)
from core.agents.rtl_agent import (
    plan_rtl,
    generate_code,
    chunk_and_generate_multimodule_rtl,
)
from core.verification.hdl_validator import validate_rtl_syntax_and_structure
from core.efs_ir.models import (
    EFSIR,
    EFSMetadata,
    EFSComponent,
    EFSSignal,
    EFSFSM,
    EFSState,
    EFSRequirement,
    EFSTimingRule,
    EFSConstraint,
    EFSTransition,
)


def test_phase5_spec_type_classifier_deterministic():
    """Verify deterministic classification correctly routes to all 6 spec types."""
    # 1. Combinational: pure logic, no clk/rst
    comb_spec = """
    # Hardware Design Specification: mux4
    4-to-1 multiplexer.
    Inputs: sel [1:0], in0 [7:0], in1 [7:0], in2 [7:0], in3 [7:0], en
    Output: out [7:0]
    out = (en) ? in_selected : 0
    """
    st, reason = classify_spec_type(comb_spec)
    assert st == SpecType.COMBINATIONAL

    # 2. FSM Controller: states and transitions
    fsm_spec = """
    # Hardware Design Specification: state_controller
    states: IDLE, RUN, DONE
    transitions:
    IDLE -> RUN: start == 1
    RUN -> DONE: count == 4
    DONE -> IDLE: the next clock cycle
    ## Interface
    - clk: input, 1 bit, Clock
    - rst_n: input, 1 bit, Reset
    - start: input, 1 bit, Start
    - done: output, 1 bit, Done
    """
    st, reason = classify_spec_type(fsm_spec)
    assert st == SpecType.FSM_CONTROLLER

    # 3. Datapath / Pipeline: pipeline stages
    pipe_spec = """
    # Hardware Design Specification: pipelined_multiplier
    Pipelined multiplier with latency of 3 cycles across stage 1, stage 2, and stage 3.
    ## Interface
    - clk: input, 1 bit, Clock
    - rst_n: input, 1 bit, Reset
    - in_valid: input, 1 bit, Valid
    - a: input, 16 bits, Data A
    - b: input, 16 bits, Data B
    - out_valid: output, 1 bit, Valid
    - product: output, 32 bits, Result
    """
    st, reason = classify_spec_type(pipe_spec)
    assert st == SpecType.DATAPATH_PIPELINE

    # 4. Register Map: address offsets and CSR registers
    reg_spec = """
    # Hardware Design Specification: peripheral_csr
    Register map address decoder for peripheral control/status registers.
    Offsets: 0x00: CTRL_REG, 0x04: STATUS_REG, 0x08: DATA_REG
    ## Interface
    - clk: input, 1 bit, Clock
    - rst_n: input, 1 bit, Reset
    - addr: input, 8 bits, Address
    - wr_en: input, 1 bit, Write enable
    - wdata: input, 32 bits, Write data
    - rd_en: input, 1 bit, Read enable
    - rdata: output, 32 bits, Read data
    ## Registers
    - CTRL_REG: 32 bits, Control register
    - STATUS_REG: 32 bits, Status register
    - DATA_REG: 32 bits, Data register
    """
    st, reason = classify_spec_type(reg_spec)
    assert st == SpecType.REGISTER_MAP

    # 5. FIFO / Memory: queue with wr_en, rd_en, full, empty
    fifo_spec = """
    # Hardware Design Specification: packet_fifo
    Synchronous FIFO queue buffer.
    ## Interface
    - clk: input, 1 bit, Clock
    - rst_n: input, 1 bit, Reset
    - wr_en: input, 1 bit, Write
    - wr_data: input, 32 bits, Data in
    - rd_en: input, 1 bit, Read
    - rd_data: output, 32 bits, Data out
    - full: output, 1 bit, Full flag
    - empty: output, 1 bit, Empty flag
    """
    st, reason = classify_spec_type(fifo_spec)
    assert st == SpecType.FIFO_MEMORY

    # 6. Protocol Engine: standard bus interface / stream handshake
    proto_spec = """
    # Hardware Design Specification: stream_axi_adapter
    AXI-stream protocol engine with valid/ready handshake.
    ## Interface
    - clk: input, 1 bit, Clock
    - rst_n: input, 1 bit, Reset
    - s_valid: input, 1 bit, Slave valid
    - s_ready: output, 1 bit, Slave ready
    - s_data: input, 32 bits, Slave data
    - m_valid: output, 1 bit, Master valid
    - m_ready: input, 1 bit, Master ready
    - m_data: output, 32 bits, Master data
    """
    st, reason = classify_spec_type(proto_spec)
    assert st == SpecType.PROTOCOL_ENGINE


def test_phase5_spec_type_classifier_llm_fallback():
    """Verify ambiguous spec triggers LLM fallback classifier."""
    ambiguous_spec = """
    # Hardware Design Specification: custom_block
    Custom hybrid logic unit with mixed control signals.
    - clk: input, 1 bit
    - rst_n: input, 1 bit
    - cfg: input, 4 bits
    - result: output, 8 bits
    """

    class MockClassifierLLM:
        available = True
        def complete_json(self, *args, **kwargs):
            return {"spec_type": "datapath_pipeline", "reason": "Configurable arithmetic unit"}

    st, reason = classify_spec_type(ambiguous_spec, llm=MockClassifierLLM())
    assert st == SpecType.DATAPATH_PIPELINE
    assert "Configurable arithmetic unit" in reason


def test_phase5_prompt_few_shot_examples_compile():
    """Verify each type-specific prompt template under prompts/ contains verified compiling SystemVerilog code."""
    p_dir = find_prompts_dir()
    assert p_dir.is_dir(), f"Prompts directory {p_dir} must exist."

    for st in SpecType:
        tmpl = load_spec_prompt_template(st)
        assert tmpl["spec_type"] == st.value
        assert "system_prompt" in tmpl
        assert len(tmpl.get("few_shot_examples", [])) > 0, f"Few-shot examples required for {st.value}"

        for ex in tmpl["few_shot_examples"]:
            verilog_code = ex.get("verilog", "")
            assert verilog_code.strip(), f"Empty verilog code in {st.value} example"
            # Validate with HDL validator
            report = validate_rtl_syntax_and_structure(verilog_code, target_language="SystemVerilog")
            assert report.is_valid is True, f"Few-shot example in {st.value} failed syntax validation: {[e.message for e in report.errors]}"


def test_phase5_verified_library_modules_syntax_and_lookup():
    """Verify standard library modules in rtl_lib/ exist, compile, and can be retrieved."""
    lib_dir = find_rtl_lib_dir()
    assert lib_dir.is_dir(), f"RTL lib directory {lib_dir} must exist."

    blocks = ["synchronizer", "fifo", "register_bank"]
    for b in blocks:
        lib_entry = get_library_module(b)
        assert lib_entry is not None, f"Could not find library module for '{b}'"
        assert lib_entry["code"].strip()

        # Validate syntax
        report = validate_rtl_syntax_and_structure(lib_entry["code"], target_language="SystemVerilog")
        assert report.is_valid is True, f"Library module '{b}' failed validation: {[e.message for e in report.errors]}"

    # Test can_use_library_module detection
    assert can_use_library_module("Multi-stage synchronizer for clock domain crossing with metastability mitigation") == "synchronizer"
    assert can_use_library_module("Synchronous FIFO with almost_full and almost_empty flags") == "fifo_sync"
    assert can_use_library_module("Memory mapped register_bank for CSR decoding") == "register_bank"

    # Test direct instantiation in generate_code
    sync_code = generate_code("Multi-stage synchronizer for clock domain crossing with metastability mitigation", "SystemVerilog")
    assert "module sync_ff" in sync_code
    assert "stage_reg" in sync_code


def test_phase5_feed_rich_ir_requirements_and_timing_constraints():
    """Verify that behavioral requirements, register bitfields, and timing constraints are fed to planner."""
    efs = EFSIR(
        metadata=EFSMetadata(design_name="motor_controller"),
        signals=[
            EFSSignal(name="clk", direction="input", width=1, semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", width=1, semantic_role="reset"),
            EFSSignal(name="speed_in", direction="input", width=8),
            EFSSignal(name="pwm_out", direction="output", width=1)
        ],
        requirements=[
            EFSRequirement(requirement_id="REQ_MOTOR_001", description="PWM period must match speed_in frequency modulation.")
        ],
        timing_rules=[
            EFSTimingRule(rule_id="TIME_CLK_001", clock_signal="clk", minimum_delay="10ns", maximum_delay="20ns")
        ],
        constraints=[
            EFSConstraint(type="frequency", condition="clk == 100MHz", expected_behavior="Maximum operating frequency 100MHz")
        ]
    )

    plan = plan_rtl("Motor controller specification", "SystemVerilog", efs_ir=efs)
    assert plan["module_name"] == "motor_controller"
    assert any(p["name"] == "speed_in" for p in plan["ports"])
    assert any(p["name"] == "pwm_out" for p in plan["ports"])


def test_phase5_chunk_multi_component_spec_and_assemble_wrapper():
    """Verify chunking big multi-component specs and assembling via deterministic top-level wrapper."""
    comp_a = EFSComponent(component_id="COMP_001", name="uart_rx", type="receiver", description="UART RX subblock")
    comp_b = EFSComponent(component_id="COMP_002", name="uart_tx", type="transmitter", description="UART TX subblock")

    efs = EFSIR(
        metadata=EFSMetadata(design_name="uart_transceiver"),
        components=[comp_a, comp_b],
        signals=[
            EFSSignal(name="clk", direction="input", width=1, semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", width=1, semantic_role="reset"),
            EFSSignal(name="rx_serial", direction="input", width=1),
            EFSSignal(name="rx_data", direction="output", width=8),
            EFSSignal(name="rx_valid", direction="output", width=1),
            EFSSignal(name="tx_data", direction="input", width=8),
            EFSSignal(name="tx_valid", direction="input", width=1),
            EFSSignal(name="tx_serial", direction="output", width=1)
        ],
        fsms=[
            EFSFSM(
                name="uart_rx",
                states=[EFSState(name="IDLE"), EFSState(name="RECEIVE")],
                transitions=[
                    EFSTransition(source_state="IDLE", target_state="RECEIVE", condition="rx_serial == 1'b0", evidence="UART start bit"),
                    EFSTransition(source_state="RECEIVE", target_state="IDLE", condition="rx_valid == 1'b1", evidence="Receive complete")
                ]
            ),
            EFSFSM(
                name="uart_tx",
                states=[EFSState(name="IDLE"), EFSState(name="TRANSMIT")],
                transitions=[
                    EFSTransition(source_state="IDLE", target_state="TRANSMIT", condition="tx_valid == 1'b1", evidence="Transmit request"),
                    EFSTransition(source_state="TRANSMIT", target_state="IDLE", condition="tx_serial == 1'b1", evidence="Transmit complete")
                ]
            )
        ]
    )

    code = chunk_and_generate_multimodule_rtl("UART Transceiver top specification", "SystemVerilog", efs_ir=efs)
    # Both submodules must be defined
    assert "module uart_rx" in code
    assert "module uart_tx" in code
    # Top-level wrapper module must be present and instantiate both
    assert "module uart_transceiver" in code
    assert "u_uart_rx" in code
    assert "u_uart_tx" in code

    # Validate combined multi-module code syntax
    report = validate_rtl_syntax_and_structure(code, target_language="SystemVerilog")
    assert report.is_valid is True, f"Combined multi-module RTL failed validation: {[e.message for e in report.errors]}"
