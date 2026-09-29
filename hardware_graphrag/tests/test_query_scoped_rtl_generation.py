"""
Tests for query-scoped RTL generation from Neo4j knowledge graph and EFS IR.

Validates that only relevant content according to the user query (target component,
its connected interfaces, signals, registers, FSM, and protocol rules) is extracted
into the ContextPack and passed to LLM during Stage 1 RTL planning and Stage 2 RTL generation.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import pytest
from unittest.mock import MagicMock, patch

from core.efs_ir.models import (
    EFSIR,
    EFSMetadata,
    EFSComponent,
    EFSInterface,
    EFSSignal,
    EFSRegister,
    EFSRegisterField,
    EFSFSM,
    EFSState,
    EFSTransition,
    EFSConflict,
)
from core.retrieval.context_compiler import compile_task_context
from core.retrieval.dependency_resolver import resolve_dependencies
from core.agents.rtl_agent import plan_rtl, validate_rtl_plan, generate_code


def build_multi_component_ir():
    """Create a realistic multi-component SoC EFS IR with DMA controller and FIFO."""
    comp_dma = EFSComponent(
        component_id="comp_dma",
        name="DMA_Main_Controller",
        type="controller",
        description="Main DMA controller handling bus transfers",
        clock_domain="clk_sys",
        reset_domain="rst_sys_n",
        interfaces=["axi_master", "dma_cfg"],
    )
    comp_fifo = EFSComponent(
        component_id="comp_fifo",
        name="Data_FIFO",
        type="fifo",
        description="Internal FIFO buffer",
        clock_domain="clk_sys",
        reset_domain="rst_sys_n",
        interfaces=["fifo_data_in", "fifo_data_out"],
    )

    # DMA interfaces and signals
    if_axi = EFSInterface(
        interface_id="if_axi",
        name="axi_master",
        protocol="AXI4-Lite",
        role="master",
        signals=["axi_awaddr", "axi_awvalid", "axi_awready"],
    )
    if_fifo = EFSInterface(
        interface_id="if_fifo",
        name="fifo_data_in",
        protocol="Native",
        role="slave",
        signals=["fifo_din", "fifo_wr_en", "fifo_full"],
    )

    # Signals
    clk_sig = EFSSignal(signal_id="sig_clk", name="clk_sys", direction="input", width=1, semantic_role="clock")
    rst_sig = EFSSignal(signal_id="sig_rst", name="rst_sys_n", direction="input", width=1, semantic_role="reset")
    axi_awaddr = EFSSignal(signal_id="sig_awaddr", name="axi_awaddr", direction="output", width=32, semantic_role="address")
    axi_awvalid = EFSSignal(signal_id="sig_awvalid", name="axi_awvalid", direction="output", width=1, semantic_role="control")
    axi_awready = EFSSignal(signal_id="sig_awready", name="axi_awready", direction="input", width=1, semantic_role="status")
    
    # FIFO-only signals (should NOT leak into DMA controller prompt)
    fifo_din = EFSSignal(signal_id="sig_fifo_din", name="fifo_din", direction="input", width=32, semantic_role="data")
    fifo_wr_en = EFSSignal(signal_id="sig_fifo_wr", name="fifo_wr_en", direction="input", width=1, semantic_role="control")
    fifo_full = EFSSignal(signal_id="sig_fifo_full", name="fifo_full", direction="output", width=1, semantic_role="status")

    # Registers
    reg_dma_ctrl = EFSRegister(
        register_id="reg_dma_ctrl",
        name="DMA_CTRL",
        offset="0x00",
        width=32,
        owner="DMA_Main_Controller",
        description="DMA control register",
        fields=[
            EFSRegisterField(name="START", msb=0, lsb=0, description="Start transfer"),
            EFSRegisterField(name="BURST_MODE", msb=1, lsb=1, description="Burst mode"),
        ]
    )
    reg_fifo_status = EFSRegister(
        register_id="reg_fifo_stat",
        name="FIFO_STATUS",
        offset="0x10",
        width=32,
        owner="Data_FIFO",
        description="FIFO status register (FIFO internal)",
        fields=[
            EFSRegisterField(name="ALMOST_FULL", msb=0, lsb=0, description="FIFO almost full"),
        ]
    )

    # FSM for DMA
    fsm_dma = EFSFSM(
        fsm_id="fsm_dma",
        name="DMA_FSM",
        initial_state="IDLE",
        states=[
            EFSState(name="IDLE"),
            EFSState(name="SETUP"),
            EFSState(name="BURST"),
            EFSState(name="DONE"),
        ],
        transitions=[
            EFSTransition(source_state="IDLE", target_state="SETUP", condition="DMA_CTRL[0] == 1'b1"),
            EFSTransition(source_state="SETUP", target_state="BURST", condition="axi_awready == 1'b1"),
            EFSTransition(source_state="BURST", target_state="DONE", condition="burst_count == 4'h8"),
            EFSTransition(source_state="DONE", target_state="IDLE", condition="1'b1"),
        ]
    )

    return EFSIR(
        metadata=EFSMetadata(design_name="SoC_Top"),
        components=[comp_dma, comp_fifo],
        interfaces=[if_axi, if_fifo],
        signals=[clk_sig, rst_sig, axi_awaddr, axi_awvalid, axi_awready, fifo_din, fifo_wr_en, fifo_full],
        registers=[reg_dma_ctrl, reg_fifo_status],
        fsms=[fsm_dma],
        conflicts=[]
    )


@pytest.fixture
def multi_component_ir():
    return build_multi_component_ir()


def test_dependency_resolver_scopes_strictly_to_target_component(multi_component_ir):
    """Test that dependency_resolver only extracts signals and registers belonging to target component."""
    target_comp = multi_component_ir.components[0]  # DMA_Main_Controller
    resolved = resolve_dependencies(target_comp, multi_component_ir, max_depth=1)

    signal_names = [s.name for s in resolved.signals]
    register_names = [r.name for r in resolved.registers]

    # Clocks, resets, and AXI signals should be included
    assert "clk_sys" in signal_names
    assert "rst_sys_n" in signal_names
    assert "axi_awaddr" in signal_names
    assert "axi_awvalid" in signal_names
    assert "axi_awready" in signal_names

    # FIFO internal signals MUST NOT be included
    assert "fifo_din" not in signal_names
    assert "fifo_wr_en" not in signal_names
    assert "fifo_full" not in signal_names

    # DMA control register must be included; FIFO internal register must NOT
    assert "DMA_CTRL" in register_names
    assert "FIFO_STATUS" not in register_names


def test_compile_task_context_scopes_to_query_target(multi_component_ir):
    """Test that compile_task_context produces a ContextPack scoped strictly to user query target."""
    query = "Generate synthesizable Verilog RTL for DMA_Main_Controller with AXI4-Lite master interface"
    pack, res = compile_task_context(query, multi_component_ir)

    assert res.is_success()
    assert res.target_component.name == "DMA_Main_Controller"

    pack_dict = pack.to_dict()
    sig_names = [s["name"] for s in pack_dict.get("target_signals", [])]
    reg_names = [r["name"] for r in pack_dict.get("target_registers", [])]

    # DMA signals should be present
    assert "axi_awaddr" in sig_names
    assert "axi_awvalid" in sig_names
    assert "clk_sys" in sig_names

    # FIFO signals should be absent
    assert "fifo_din" not in sig_names
    assert "fifo_full" not in sig_names

    # Registers
    assert "DMA_CTRL" in reg_names
    assert "FIFO_STATUS" not in reg_names


def test_plan_rtl_uses_scoped_context_in_prompt(multi_component_ir):
    """Test that plan_rtl injects only scoped ContextPack hardware constructs to LLM prompt."""
    query = "Generate RTL for DMA_Main_Controller"
    pack, _ = compile_task_context(query, multi_component_ir)

    mock_llm = MagicMock()
    mock_llm.available = True
    # Return mock plan
    mock_llm.complete_json.return_value = {
        "module_name": "DMA_Main_Controller",
        "specification_status": {"status": "COMPLETE", "blocking_issues": []},
        "ports": [
            {"name": "clk_sys", "direction": "input", "width": "1"},
            {"name": "rst_sys_n", "direction": "input", "width": "1"},
            {"name": "axi_awaddr", "direction": "output", "width": "32"},
            {"name": "axi_awvalid", "direction": "output", "width": "1"},
            {"name": "axi_awready", "direction": "input", "width": "1"},
        ],
        "parameters": [],
        "internal_registers": [{"name": "DMA_CTRL", "width": "32", "reset_val": "32'h0", "purpose": "DMA control"}],
        "fsm": {"states": ["IDLE", "SETUP", "BURST", "DONE"], "initial_state": "IDLE", "transitions": []},
        "datapath_strategy": "Direct FSM transfer",
        "reset_polarity": "active_low"
    }

    with patch("core.agents.rtl_agent.get_llm_client", return_value=mock_llm):
        plan = plan_rtl("dummy spec", "DMA_Main_Controller", efs_ir=multi_component_ir, context_pack=pack)

    # Check that LLM was called with scoped context
    assert mock_llm.complete_json.called
    call_args = mock_llm.complete_json.call_args[0]
    prompt_sent = call_args[1]

    # Prompt MUST contain DMA_Main_Controller and its signals
    assert "DMA_Main_Controller" in prompt_sent
    assert "axi_awaddr" in prompt_sent
    assert "DMA_CTRL" in prompt_sent

    # Prompt MUST NOT contain FIFO internal signals or FIFO internal register
    assert "fifo_din" not in prompt_sent
    assert "FIFO_STATUS" not in prompt_sent


def test_validate_rtl_plan_with_context_pack_does_not_falsely_block_on_unrelated_signals(multi_component_ir):
    """Test that validate_rtl_plan validates only target module ports when context_pack is supplied."""
    query = "Generate RTL for DMA_Main_Controller"
    pack, _ = compile_task_context(query, multi_component_ir)

    # RTL plan containing only DMA_Main_Controller ports
    rtl_plan = {
        "module_name": "DMA_Main_Controller",
        "specification_status": {"status": "COMPLETE", "blocking_issues": []},
        "ports": [
            {"name": "clk_sys", "direction": "input", "width": "1"},
            {"name": "rst_sys_n", "direction": "input", "width": "1"},
            {"name": "axi_awaddr", "direction": "output", "width": "32"},
            {"name": "axi_awvalid", "direction": "output", "width": "1"},
            {"name": "axi_awready", "direction": "input", "width": "1"},
        ],
        "fsm": {"states": ["IDLE", "SETUP", "BURST", "DONE"], "transitions": [
            {"from": "IDLE", "to": "SETUP", "condition": {"expression": "DMA_CTRL[0] == 1'b1", "is_fully_defined": True}}
        ]}
    }

    # When validated WITH context_pack:
    val_scoped = validate_rtl_plan(rtl_plan, efs_ir=multi_component_ir, context_pack=pack)
    assert val_scoped["valid"] is True
    assert val_scoped["status"] == "COMPLETE"
    assert len(val_scoped["issues"]) == 0

    # When validated WITHOUT context_pack (global full efs_ir check):
    # It would notice missing FIFO signals from DMA's plan ports
    val_global = validate_rtl_plan(rtl_plan, efs_ir=multi_component_ir, context_pack=None)
    missing_issues = [i for i in val_global["issues"] if "fifo_din" in i.get("issue", "")]
    assert len(missing_issues) > 0  # Confirms global check catches it, proving scoped check correctly isolated it!


def test_generate_code_end_to_end_with_context_pack(multi_component_ir):
    """Test end-to-end generate_code with context_pack passes through Stage 1 & Stage 2."""
    query = "Generate RTL for DMA_Main_Controller"
    pack, _ = compile_task_context(query, multi_component_ir)

    mock_llm = MagicMock()
    mock_llm.available = False  # Use deterministic template generation

    with patch("core.agents.rtl_agent.get_llm_client", return_value=mock_llm):
        code = generate_code(
            design_spec="DMA Controller specification",
            target="SystemVerilog",
            efs_ir=multi_component_ir,
            context_pack=pack
        )

    assert "module DMA_Main_Controller" in code
    assert "input  logic       clk_sys" in code or "input logic" in code
    assert "axi_awaddr" in code
    assert "fifo_din" not in code  # Scoped!
    assert "endmodule" in code


if __name__ == "__main__":
    ir = build_multi_component_ir()
    print("Testing test_dependency_resolver_scopes_strictly_to_target_component...")
    test_dependency_resolver_scopes_strictly_to_target_component(ir)
    print("PASSED")

    print("Testing test_compile_task_context_scopes_to_query_target...")
    test_compile_task_context_scopes_to_query_target(ir)
    print("PASSED")

    print("Testing test_plan_rtl_uses_scoped_context_in_prompt...")
    test_plan_rtl_uses_scoped_context_in_prompt(ir)
    print("PASSED")

    print("Testing test_validate_rtl_plan_with_context_pack_does_not_falsely_block_on_unrelated_signals...")
    test_validate_rtl_plan_with_context_pack_does_not_falsely_block_on_unrelated_signals(ir)
    print("PASSED")

    print("Testing test_generate_code_end_to_end_with_context_pack...")
    test_generate_code_end_to_end_with_context_pack(ir)
    print("PASSED")

    print("\nALL 5 TESTS PASSED SUCCESSFULLY!")

