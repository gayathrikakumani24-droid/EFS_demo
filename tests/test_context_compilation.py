"""
Unit and Integration tests for Query-Driven, Dependency-Aware Context Compilation Layer.
"""

import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from core.efs_ir.models import EFSIR, EFSComponent, EFSInterface, EFSSignal, EFSRegister, EFSMetadata
from core.retrieval.query_analyzer import analyze_query
from core.retrieval.target_resolver import resolve_target
from core.retrieval.dependency_resolver import resolve_dependencies
from core.retrieval.protocol_resolver import resolve_protocol_context
from core.retrieval.context_compiler import compile_task_context


@pytest.fixture
def sample_matrix_efs_ir():
    """Create a sample multi-module system EFS IR representing a Matrix Compute system."""
    ir = EFSIR()
    ir.metadata = EFSMetadata(design_name="MatrixComputeSystem", protocol="AXI4")

    # Components
    adder = EFSComponent(component_id="COMP_ADDER", name="MatrixAdder", type="adder", description="32x32 Matrix Adder unit.")
    subtractor = EFSComponent(component_id="COMP_SUB", name="MatrixSubtractor", type="subtractor", description="32x32 Matrix Subtractor unit.")
    multiplier = EFSComponent(component_id="COMP_MULT", name="MatrixMultiplier", type="multiplier", description="Matrix Multiplier core.")
    regfile = EFSComponent(component_id="COMP_REGFILE", name="RegisterFile", type="regfile", description="Configuration registers.")
    dma = EFSComponent(component_id="COMP_DMA", name="DMAController", type="dma", description="DMA engine for memory transfers.")
    
    ir.components = [adder, subtractor, multiplier, regfile, dma]

    # Signals
    clk = EFSSignal(signal_id="SIG_CLK", name="clk", direction="input", semantic_role="clock", owner="COMP_ADDER")
    rst = EFSSignal(signal_id="SIG_RST", name="rst_n", direction="input", semantic_role="reset", owner="COMP_ADDER")
    op_a = EFSSignal(signal_id="SIG_OPA", name="operand_a", width="64", direction="input", owner="COMP_ADDER")
    op_b = EFSSignal(signal_id="SIG_OPB", name="operand_b", width="64", direction="input", owner="COMP_ADDER")
    res = EFSSignal(signal_id="SIG_RES", name="result", width="64", direction="output", owner="COMP_ADDER")
    done = EFSSignal(signal_id="SIG_DONE", name="done", width="1", direction="output", owner="COMP_ADDER")

    # AXI Signals for Adder
    awvalid = EFSSignal(signal_id="SIG_AWVALID", name="s_axi_awvalid", width="1", direction="input", owner="COMP_ADDER")
    awready = EFSSignal(signal_id="SIG_AWREADY", name="s_axi_awready", width="1", direction="output", owner="COMP_ADDER")
    wdata = EFSSignal(signal_id="SIG_WDATA", name="s_axi_wdata", width="32", direction="input", owner="COMP_ADDER")
    wvalid = EFSSignal(signal_id="SIG_WVALID", name="s_axi_wvalid", width="1", direction="input", owner="COMP_ADDER")

    ir.signals = [clk, rst, op_a, op_b, res, done, awvalid, awready, wdata, wvalid]

    # Interface
    axi_if = EFSInterface(
        interface_id="IF_AXI",
        name="s_axi",
        protocol="AXI4",
        role="slave",
        source_component="COMP_ADDER",
        signals=["SIG_AWVALID", "SIG_AWREADY", "SIG_WDATA", "SIG_WVALID"]
    )
    ir.interfaces = [axi_if]

    # Register
    ctrl_reg = EFSRegister(
        register_id="REG_CTRL",
        name="ADDER_CTRL",
        offset="0x00",
        width=32,
        owner="COMP_ADDER"
    )
    ir.registers = [ctrl_reg]

    return ir


def test_query_analyzer_rtl():
    intent = analyze_query("Generate synthesizable Verilog RTL for Matrix Adder.")
    assert intent.task == "RTL_GENERATION"
    assert intent.target == "MatrixAdder"
    assert intent.language == "Verilog"


def test_query_analyzer_sva():
    intent = analyze_query("Generate SVA assertions for the AXI interface of Matrix Adder.")
    assert intent.task == "SVA_GENERATION"
    assert intent.target == "MatrixAdder"
    assert intent.interface == "AXI"
    assert intent.language == "SystemVerilog"


def test_query_analyzer_uvm():
    intent = analyze_query("Generate UVM testbench for Matrix Multiplier.")
    assert intent.task == "UVM_GENERATION"
    assert intent.target == "MatrixMultiplier"
    assert intent.language == "SystemVerilog"


def test_target_resolver(sample_matrix_efs_ir):
    res = resolve_target("MatrixAdder", sample_matrix_efs_ir)
    assert res.is_success()
    assert res.target_component.name == "MatrixAdder"
    assert res.match_method == "exact"


def test_target_resolver_not_found(sample_matrix_efs_ir):
    res = resolve_target("NonExistentModule", sample_matrix_efs_ir)
    assert res.status == "TARGET_NOT_FOUND"


def test_dependency_resolver(sample_matrix_efs_ir):
    adder = sample_matrix_efs_ir.components[0]
    deps = resolve_dependencies(adder, sample_matrix_efs_ir, max_depth=1)
    assert deps.target_component.name == "MatrixAdder"
    assert len(deps.signals) >= 6


def test_protocol_resolver():
    all_rules = [
        {"text": "AWVALID must remain asserted until AWREADY is asserted.", "citation": "AXI Spec 3.1"},
        {"text": "ARVALID must remain asserted until ARREADY is asserted.", "citation": "AXI Spec 3.2"},
        {"text": "PCIe TLP header format rule", "citation": "PCIe Spec"}
    ]
    # Create dummy deps for write-only interface
    class DummyDeps:
        signals = [EFSSignal(name="s_axi_awvalid"), EFSSignal(name="s_axi_wdata")]
        interfaces = [EFSInterface(name="s_axi")]
    
    rules, _, _ = resolve_protocol_context(DummyDeps(), all_rules, requested_interface="AXI")
    assert len(rules) >= 1
    assert "AWVALID" in rules[0]["text"]


def test_context_compiler(sample_matrix_efs_ir):
    pack, res = compile_task_context(
        "Generate synthesizable Verilog RTL for MatrixAdder.",
        sample_matrix_efs_ir,
        all_protocol_rules=[{"text": "AXI write handshake rule", "citation": "AXI"}]
    )
    assert res.is_success()
    assert pack.task.target == "MatrixAdder"
    assert pack.selected_efs_objects <= pack.total_full_efs_objects
    assert pack.estimated_tokens > 0
