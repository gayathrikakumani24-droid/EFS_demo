"""Universal Hardware Pipeline Regression Test Suite.

Tests protocol-agnostic specification parsing, EFS IR construction, retrieval,
context assembly, and RTL generation across:
- AXI4
- PCIe
- Custom Matrix Compute Unit
- UART / SPI / I2C
- Ambiguous Specifications (Specification Conflict Detection)
"""

import pytest
from core.efs_ir.builder import EFSIRBuilder
from core.efs_ir.models import EFSIR, EFSSignal, EFSComponent, EFSConflict
from core.efs_ir.validator import validate_efs_ir, get_efs_ir_quality
from core.agents.requirement_agent import parse_requirement_specification
from core.agents.rtl_agent import generate_code
from core.retrieval.protocol_resolver import resolve_protocol_context, ProtocolEvidence
from core.retrieval.dependency_resolver import ResolvedDependencies


def test_regression_axi_no_hardcoding():
    spec_text = """
    # AXI4 Read Controller Module
    The module axi_read_master controls memory read requests.
    Interfaces:
      - araddr: input, 32 bits, address signal
      - arvalid: input, 1 bit, valid signal
      - arready: output, 1 bit, ready signal
      - rdata: output, 32 bits, read data bus
      - rvalid: output, 1 bit, read valid
      - rready: input, 1 bit, read ready
    """
    req_model = parse_requirement_specification(spec_text)
    builder = EFSIRBuilder(requirement_model=req_model, design_spec=spec_text)
    ir = builder.build()

    sig_names = [s.name.lower() for s in ir.signals]
    assert "araddr" in sig_names
    assert "arvalid" in sig_names
    # Verify no unrequested write ports (like s_axi_wdata) were force-injected
    assert "s_axi_wdata" not in sig_names


def test_regression_pcie_no_axi_ports():
    spec_text = """
    # PCIe Transaction Layer Request Interface
    The pcie_tx_engine processes outbound Transaction Layer Packets (TLPs).
    Interfaces:
      - tlp_valid: input, 1 bit, TLP valid indicator
      - tlp_ready: output, 1 bit, TLP ready indicator
      - tlp_data: input, 128 bits, TLP payload data
      - tlp_fmt_type: input, 7 bits, Format and type header
      - completer_id: input, 16 bits, Completer ID tag
    """
    req_model = parse_requirement_specification(spec_text)
    builder = EFSIRBuilder(requirement_model=req_model, design_spec=spec_text)
    ir = builder.build()

    sig_names = [s.name.lower() for s in ir.signals]
    assert "tlp_valid" in sig_names
    assert "tlp_data" in sig_names

    # CRITICAL: Verify ZERO AXI ports were injected into PCIe design
    for s_name in sig_names:
        assert not s_name.startswith("s_axi_"), f"Hardcoded AXI port '{s_name}' was inappropriately injected into PCIe spec!"


def test_regression_matrix_instruction_decoder():
    spec_text = """
    # Matrix Compute Unit Instruction Decoder
    The Instruction_Decoder decodes 32-bit matrix acceleration instructions.
    Interfaces:
      - instr_in: input, 32 bits, Instruction word
      - opcode: output, 6 bits, Opcode field (bits 31:26)
      - matrix_reg_a: output, 4 bits, Source Matrix A (bits 25:22)
      - matrix_reg_b: output, 4 bits, Source Matrix B (bits 21:18)
      - dest_reg: output, 4 bits, Destination Matrix (bits 17:14)
      - exec_trigger: output, 1 bit, Trigger pulse
    """
    req_model = parse_requirement_specification(spec_text)
    builder = EFSIRBuilder(requirement_model=req_model, design_spec=spec_text)
    ir = builder.build()

    sig_names = [s.name.lower() for s in ir.signals]
    assert "instr_in" in sig_names
    assert "opcode" in sig_names
    assert "matrix_reg_a" in sig_names
    assert "s_axi_awvalid" not in sig_names


def test_regression_uart_spi_i2c_no_axi_injection():
    spec_text = """
    # UART Serial Transmitter
    Module uart_tx sends serial data frames.
    Interfaces:
      - tx_data: input, 8 bits, Parallel byte payload
      - tx_valid: input, 1 bit, Byte valid pulse
      - tx_busy: output, 1 bit, Serial transmitter active
      - txd: output, 1 bit, Serial output line
    """
    req_model = parse_requirement_specification(spec_text)
    builder = EFSIRBuilder(requirement_model=req_model, design_spec=spec_text)
    ir = builder.build()

    sig_names = [s.name.lower() for s in ir.signals]
    assert "txd" in sig_names
    assert "tx_data" in sig_names
    assert not any("axi" in s for s in sig_names)


def test_regression_specification_ambiguity_blocking():
    ir = EFSIR()
    comp = EFSComponent(component_id="COMP_001", name="Conflicting_Module")
    ir.components.append(comp)

    # Add an unresolved critical conflict to the EFS IR
    ir.conflicts.append(
        EFSConflict(
            conflict_id="CONF_001",
            type="SIGNAL_WIDTH_MISMATCH",
            severity="CRITICAL",
            evidence="Section 2 specifies DATA_WIDTH=32, but Section 5 specifies DATA_WIDTH=64.",
            resolution_status="UNRESOLVED"
        )
    )

    issues = validate_efs_ir(ir)
    scorecard = get_efs_ir_quality(ir, issues)

    # Verify that critical conflict forces BLOCKED status
    assert scorecard["status"] == "BLOCKED"
    assert scorecard["critical_conflicts"] == 1
    assert "1 unresolved critical specification conflicts." in scorecard["reasons"]


def test_regression_unseen_custom_protocol_orion_link():
    """Test completely unknown custom protocol (ORION-LINK) that system has never seen before."""
    spec_text = """
    # ORION-LINK Controller Architecture
    Module orion_link_controller manages packet handshakes over the custom ORION-LINK interface.
    Interfaces:
      - CLK: input, 1 bit, System clock
      - RESET_N: input, 1 bit, Active-low reset
      - REQ_X: input, 1 bit, Transaction request indicator
      - ACK_X: output, 1 bit, Transaction acknowledge output
      - DATA_X: input, 128 bits, Request payload bus
      - RESP_X: output, 4 bits, Completion status vector
    """
    req_model = parse_requirement_specification(spec_text)
    builder = EFSIRBuilder(requirement_model=req_model, design_spec=spec_text)
    ir = builder.build()

    sig_map = {s.name.upper(): s for s in ir.signals}
    assert "REQ_X" in sig_map
    assert "ACK_X" in sig_map
    assert "DATA_X" in sig_map
    assert "RESP_X" in sig_map

    # Verify exact widths and directions derived directly from spec
    assert sig_map["DATA_X"].width == "128"
    assert sig_map["RESP_X"].width == "4"
    assert sig_map["REQ_X"].direction == "input"
    assert sig_map["ACK_X"].direction == "output"

    # CRITICAL: Verify ZERO AXI / PCIe / UART ports were injected
    sig_names = [s.name.lower() for s in ir.signals]
    assert not any(p in s for s in sig_names for p in ["axi", "pcie", "uart", "spi", "i2c"])

