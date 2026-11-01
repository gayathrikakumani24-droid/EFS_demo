"""Generic Regression Test Suite for Signal Metadata Normalization and Conflict Resolution.

Verifies:
1. Arbitrary input, output, and bidirectional signals.
2. Arbitrary vector widths and bit ranges.
3. Equivalent representations of the same direction and width.
4. Duplicate metadata records with consistent values.
5. Conflicting values with different source authority (evidence-based precedence).
6. Conflicting equally authoritative values (blocking conflict preserved).
7. Missing or unknown width and direction.
8. Unspecified interface roles.
9. Different protocols and previously unseen signal names.
10. Valid plans reaching Stage 2.
11. Unresolved plans remaining blocked.
12. Regression cases reproducing the three reported errors without hardcoding.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath("hardware_graphrag"))
sys.path.insert(0, os.path.abspath("."))

from core.efs_ir.models import EFSIR, EFSMetadata, EFSSignal, EFSConflict
from core.efs_ir.builder import EFSIRBuilder
from core.efs_ir.signal_normalization import (
    normalize_direction,
    parse_bit_range,
    parse_hdl_port_declaration,
    parse_markdown_bullet_port,
    resolve_signal_metadata,
    compare_signal_metadata,
    EvidenceAuthority,
)
from core.agents.rtl_agent import (
    plan_rtl,
    validate_rtl_plan,
    generate_code,
    parse_markdown_spec,
)


# =========================================================================
# 1. Arbitrary Direction Normalization (Protocol-Agnostic)
# =========================================================================
@pytest.mark.parametrize("raw_dir, expected", [
    ("in", "input"),
    ("input", "input"),
    ("in_port", "input"),
    ("ip", "input"),
    ("INPUT", "input"),
    ("In", "input"),
    ("out", "output"),
    ("output", "output"),
    ("out_port", "output"),
    ("op", "output"),
    ("OUTPUT", "output"),
    ("Out", "output"),
    ("inout", "inout"),
    ("in_out", "inout"),
    ("bidir", "inout"),
    ("bidirectional", "inout"),
    ("io", "inout"),
    ("INOUT", "inout"),
    ("None", None),
    ("", None),
    ("tbd", None),
    ("unspecified", None),
])
def test_direction_normalization(raw_dir, expected):
    assert normalize_direction(raw_dir) == expected


def test_direction_normalization_does_not_derive_from_name_alone():
    # Verifies that naming alone without direction keyword does NOT invent a direction
    assert normalize_direction("clk") is None
    assert normalize_direction("tx_data_out") == "output"  # contains 'out'
    assert normalize_direction("arbitrary_signal_xyz") is None


# =========================================================================
# 2. Arbitrary Vector Widths and Bit Ranges
# =========================================================================
@pytest.mark.parametrize("raw_val, expected_w, expected_range", [
    (1, 1, ""),
    (32, 32, "[31:0]"),
    (512, 512, "[511:0]"),
    ("1", 1, ""),
    ("32", 32, "[31:0]"),
    ("32 bits", 32, "[31:0]"),
    ("64 bit", 64, "[63:0]"),
    ("512", 512, "[511:0]"),
    ("[31:0]", 32, "[31:0]"),
    ("[0:31]", 32, "[0:31]"),
    ("[511:0]", 512, "[511:0]"),
    ("[7:0]", 8, "[7:0]"),
    ("[DATA_WIDTH-1:0]", None, "[DATA_WIDTH-1:0]"),
    ("ADDR_WIDTH", None, "[ADDR_WIDTH-1:0]"),
    (None, None, None),
    ("", None, None),
])
def test_bit_range_parsing(raw_val, expected_w, expected_range):
    w, r = parse_bit_range(raw_val)
    assert w == expected_w
    assert r == expected_range


# =========================================================================
# 3. Explicit HDL Port Declaration Parsing (Arbitrary Signals)
# =========================================================================
@pytest.mark.parametrize("line, exp_name, exp_dir, exp_w", [
    ("output wire [31:0] custom_bus_addr,", "custom_bus_addr", "output", 32),
    ("[105] input wire [511:0] stream_rx_payload,", "stream_rx_payload", "input", 512),
    ("- inout [7:0] i2c_sda_io;", "i2c_sda_io", "inout", 8),
    ("input logic clk_core", "clk_core", "input", 1),
    ("output reg ready_strobe;", "ready_strobe", "output", 1),
    ("12: output wire [63:0] dsp_accum_out,", "dsp_accum_out", "output", 64),
    ("input wire [DATA_WIDTH-1:0] pcie_tdata", "pcie_tdata", "input", None),
])
def test_hdl_port_declaration_parsing(line, exp_name, exp_dir, exp_w):
    parsed = parse_hdl_port_declaration(line)
    assert parsed is not None
    assert parsed["name"] == exp_name
    assert parsed["direction"] == exp_dir
    assert parsed["width"] == exp_w


# =========================================================================
# 4. Equivalent Representations Semantic Comparison
# =========================================================================
def test_semantic_comparison_equivalent_representations():
    port_a = {"name": "sig_x", "direction": "in", "width": "32"}
    port_b = {"name": "sig_x", "direction": "input", "width": "[31:0]"}
    is_compat, reason = compare_signal_metadata(port_a, port_b)
    assert is_compat
    assert reason is None

    port_c = {"name": "sig_y", "direction": "out", "width": "512 bits"}
    port_d = {"name": "sig_y", "direction": "output", "width": "[511:0]"}
    is_compat, reason = compare_signal_metadata(port_c, port_d)
    assert is_compat


def test_semantic_comparison_genuine_conflicts():
    # Genuine direction conflict
    port_a = {"name": "sig_z", "direction": "input", "width": "32"}
    port_b = {"name": "sig_z", "direction": "output", "width": "32"}
    is_compat, reason = compare_signal_metadata(port_a, port_b)
    assert not is_compat
    assert "Direction conflict" in reason

    # Genuine width conflict
    port_c = {"name": "sig_w", "direction": "input", "width": "32"}
    port_d = {"name": "sig_w", "direction": "input", "width": "64"}
    is_compat, reason = compare_signal_metadata(port_c, port_d)
    assert not is_compat
    assert "Width conflict" in reason


# =========================================================================
# 5. Duplicate Metadata Records With Consistent Values
# =========================================================================
def test_duplicate_records_merged_cleanly():
    candidates = [
        {"name": "sensor_val", "direction": "input", "width": "16", "description": "ADC channel 0 reading"},
        {"name": "sensor_val", "direction": "in", "width": "[15:0]", "description": "input wire [15:0] sensor_val,"},
    ]
    resolved = resolve_signal_metadata(candidates, signal_name="sensor_val")
    assert not resolved["is_conflicted"]
    assert resolved["direction"] == "input"
    assert resolved["width"] == "16"
    assert resolved["direction_authority"] >= EvidenceAuthority.EXPLICIT_SPEC


# =========================================================================
# 6. Precedence Policy: Higher Authority Overrides Inferred/Default
# =========================================================================
def test_precedence_policy_explicit_overrides_inferred():
    # Inferred role says 'input' (e.g. from an assumed slave contract),
    # but explicit HDL in description specifies 'output wire [31:0] initiator_addr,'
    record = {
        "name": "initiator_addr",
        "description": "output wire [31:0] initiator_addr,",
        "attributes": {"direction": "output", "width": "32"}
    }
    resolved = resolve_signal_metadata(
        [record],
        signal_name="initiator_addr",
        role_inferred_dir="input"  # Inferred with lower authority
    )
    assert not resolved["is_conflicted"]
    assert resolved["direction"] == "output"
    assert resolved["width"] == "32"
    assert resolved["direction_source"] == "explicit_hdl"


# =========================================================================
# 7. Conflicting Equally Authoritative Values (Unresolved Conflict)
# =========================================================================
def test_conflicting_equally_authoritative_values_flagged():
    # Two explicit specification sources with equal authority disagree on direction
    rec_a = {"name": "bidir_or_not", "attributes": {"direction": "input", "width": "32"}, "description": "Port spec A"}
    rec_b = {"name": "bidir_or_not", "attributes": {"direction": "output", "width": "32"}, "description": "Port spec B"}

    resolved = resolve_signal_metadata([rec_a, rec_b], signal_name="bidir_or_not")
    assert resolved["is_conflicted"]
    assert len(resolved["conflicts"]) > 0
    assert "Conflicting direction" in resolved["conflicts"][0]["issue"]


# =========================================================================
# 8. Unspecified / Missing Metadata Preserved
# =========================================================================
def test_unspecified_missing_metadata_preserves_provenance():
    # Signal with no explicit width or direction
    rec = {"name": "mystery_signal", "description": "Internal test hook"}
    resolved = resolve_signal_metadata([rec], signal_name="mystery_signal")
    assert not resolved["is_conflicted"]
    assert resolved["direction_source"] == "unspecified_default"
    assert resolved["width_source"] == "unspecified_default"
    assert resolved["direction_authority"] == EvidenceAuthority.UNKNOWN.value


# =========================================================================
# 9. Previously Unseen Protocols and Signals Handled Generically
# =========================================================================
def test_previously_unseen_protocol_signals_in_builder():
    spec_text = """# Hardware Design Specification: NeuralProcessingCore
Module NeuralProcessingCore ports:
- spi_sclk: input, 1 bit, Serial peripheral clock
- spi_mosi: input, 1 bit, Master out slave in
- spi_miso: output, 1 bit, Master in slave out
- dsp_matrix_din: input, 128 bits, High-bandwidth parallel feature input
- dsp_matrix_dout: output, 128 bits, Transformed feature output
- pcie_rst_n: input, 1 bit, Active-low PCI Express link reset
"""
    req_model = {
        "title": "NeuralProcessingCore",
        "signals": [
            {"name": "spi_sclk", "description": "input wire spi_sclk,", "attributes": {"direction": "input", "width": "1"}},
            {"name": "spi_mosi", "description": "input wire spi_mosi,", "attributes": {"direction": "input", "width": "1"}},
            {"name": "spi_miso", "description": "output wire spi_miso,", "attributes": {"direction": "output", "width": "1"}},
            {"name": "dsp_matrix_din", "description": "input wire [127:0] dsp_matrix_din,", "attributes": {"direction": "input", "width": "128"}},
            {"name": "dsp_matrix_dout", "description": "output wire [127:0] dsp_matrix_dout,", "attributes": {"direction": "output", "width": "128"}},
            {"name": "pcie_rst_n", "description": "input wire pcie_rst_n,", "attributes": {"direction": "input", "width": "1"}},
        ]
    }
    builder = EFSIRBuilder(req_model=req_model, design_spec=spec_text)
    efs_ir = builder.build()

    sig_map = {s.name: s for s in efs_ir.signals}
    assert sig_map["spi_miso"].direction == "output"
    assert sig_map["dsp_matrix_din"].direction == "input"
    assert sig_map["dsp_matrix_din"].width == "128"
    assert sig_map["dsp_matrix_dout"].direction == "output"
    assert sig_map["dsp_matrix_dout"].width == "128"
    assert sig_map["dsp_matrix_dout"].direction_source in ("explicit_hdl", "explicit_spec")


# =========================================================================
# 10. Regression Tests for Reported Problem Cases
# =========================================================================
def test_regression_case_01_axi_araddr_resolved_as_output_32():
    """
    Case 1: axi_araddr listed as output wire [31:0] in spec.
    Previously failed because assumed slave role forced direction=input, width=1.
    """
    spec_text = """
    output wire [31:0] axi_araddr,
    """
    req_model = {
        "title": "MatrixComputeUnit",
        "signals": [
            {
                "name": "axi_araddr",
                "description": "[242] output wire [31:0] axi_araddr,",
                "attributes": {"direction": "output", "width": "32"}
            }
        ]
    }
    builder = EFSIRBuilder(req_model=req_model, design_spec="Compute unit initiating memory reads.")
    efs_ir = builder.build()

    araddr_sig = next(s for s in efs_ir.signals if s.name == "axi_araddr")
    assert araddr_sig.direction == "output"
    assert araddr_sig.width == "32"
    assert araddr_sig.direction_source in ("explicit_hdl", "explicit_spec")


def test_regression_case_02_axi_rdata_resolved_as_input_512():
    """
    Case 2: axi_rdata listed as input wire [511:0] in spec.
    Previously failed because assumed slave role forced direction=output, width=1.
    """
    spec_text = """
    input wire [511:0] axi_rdata,
    """
    req_model = {
        "title": "MatrixComputeUnit",
        "signals": [
            {
                "name": "axi_rdata",
                "description": "[244] input wire [511:0] axi_rdata,",
                "attributes": {"direction": "input", "width": "512"}
            }
        ]
    }
    builder = EFSIRBuilder(req_model=req_model, design_spec="Compute unit receiving read data.")
    efs_ir = builder.build()

    rdata_sig = next(s for s in efs_ir.signals if s.name == "axi_rdata")
    assert rdata_sig.direction == "input"
    assert rdata_sig.width == "512"
    assert rdata_sig.direction_source in ("explicit_hdl", "explicit_spec")


def test_regression_case_03_axi_arready_resolved_as_input():
    """
    Case 3: axi_arready listed as input wire in spec.
    Previously failed because assumed slave role forced direction=output.
    """
    spec_text = """
    input wire axi_arready,
    """
    req_model = {
        "title": "MatrixComputeUnit",
        "signals": [
            {
                "name": "axi_arready",
                "description": "[243] input wire axi_arready,",
                "attributes": {"direction": "input", "width": "1"}
            }
        ]
    }
    builder = EFSIRBuilder(req_model=req_model, design_spec="Compute unit receiving handshake.")
    efs_ir = builder.build()

    arready_sig = next(s for s in efs_ir.signals if s.name == "axi_arready")
    assert arready_sig.direction == "input"
    assert arready_sig.width == "1"


# =========================================================================
# 11. End-to-End: Valid Plan Reaches Stage 2
# =========================================================================
def test_valid_plan_with_normalized_signals_reaches_stage_2():
    """Verify that consistent, normalized interface metadata allows plan to proceed to Stage 2."""
    spec_text = """# Hardware Design Specification: MemoryStreamingController
Module MemoryStreamingController ports:
- clk: input, 1 bit
- rst_n: input, 1 bit
- axi_araddr: output, 32 bits
- axi_rdata: input, 512 bits
- axi_arready: input, 1 bit
- axi_arvalid: output, 1 bit
"""
    req_model = {
        "title": "MemoryStreamingController",
        "signals": [
            {"name": "clk", "direction": "input", "width": "1"},
            {"name": "rst_n", "direction": "input", "width": "1"},
            {"name": "axi_araddr", "description": "output wire [31:0] axi_araddr,", "attributes": {"direction": "output", "width": "32"}},
            {"name": "axi_rdata", "description": "input wire [511:0] axi_rdata,", "attributes": {"direction": "input", "width": "512"}},
            {"name": "axi_arready", "description": "input wire axi_arready,", "attributes": {"direction": "input", "width": "1"}},
            {"name": "axi_arvalid", "description": "output wire axi_arvalid,", "attributes": {"direction": "output", "width": "1"}},
        ]
    }
    builder = EFSIRBuilder(req_model=req_model, design_spec=spec_text)
    efs_ir = builder.build()

    plan = plan_rtl(spec_text, "Verilog", efs_ir=efs_ir)
    val = validate_rtl_plan(plan, efs_ir=efs_ir)

    # Must validate cleanly without SPECIFICATION_INCOMPLETE blocking issues
    assert val["valid"] is True
    assert len(val["issues"]) == 0

    # Ensure Stage 2 is reached and synthesizable code is generated (testing with mock LLM client)
    mock_llm = MagicMock()
    mock_llm.available = True
    mock_llm.complete.return_value = """
    module MemoryStreamingController (
        input  logic clk,
        input  logic rst_n,
        output logic [31:0] axi_araddr,
        input  logic [511:0] axi_rdata,
        input  logic axi_arready,
        output logic axi_arvalid
    );
        always_comb begin
            axi_araddr = 32'h0;
            axi_arvalid = 1'b0;
        end
    endmodule
    """
    mock_llm.last_usage = {"prompt_tokens": 100, "completion_tokens": 50}

    with patch("core.agents.rtl_agent.get_llm_client", return_value=mock_llm):
        code = generate_code(spec_text, "Verilog", efs_ir=efs_ir)
        assert "SPECIFICATION_INCOMPLETE" not in code
        assert "module MemoryStreamingController" in code
        assert "endmodule" in code
        assert "axi_araddr" in code
        assert "axi_rdata" in code

    # Also verify deterministic fallback execution reaches Stage 2 when LLM is unavailable
    mock_llm_unavailable = MagicMock()
    mock_llm_unavailable.available = False
    with patch("core.agents.rtl_agent.get_llm_client", return_value=mock_llm_unavailable):
        code_fallback = generate_code(spec_text, "Verilog", efs_ir=efs_ir)
        assert "SPECIFICATION_INCOMPLETE" not in code_fallback
        assert "module " in code_fallback
        assert "endmodule" in code_fallback


# =========================================================================
# 12. End-to-End: Genuine Unresolved Conflicts Remain Blocked
# =========================================================================
def test_genuine_unresolved_conflict_remains_blocked():
    """Verify that a genuine unresolved conflict correctly blocks Stage 2."""
    spec_text = """# Hardware Design Specification: ConflictingModule
Module ConflictingModule ports:
- clk: input, 1 bit
- rst_n: input, 1 bit
"""
    # EFSIR with an unresolved critical conflict on data_bus direction
    efs_ir = EFSIR(
        metadata=EFSMetadata(design_name="ConflictingModule"),
        signals=[
            EFSSignal(name="clk", direction="input", width="1"),
            EFSSignal(name="rst_n", direction="input", width="1"),
            EFSSignal(name="data_bus", direction="input", width="32"),
        ],
        conflicts=[
            EFSConflict(
                conflict_id="CONF_PORT_01",
                type="SPEC_CONFLICT",
                category="INTERFACE",
                entities=["data_bus"],
                property="direction",
                description="Contradictory interface specification: Table 1 specifies input, Table 2 specifies output.",
                severity="CRITICAL",
                blocking=True,
                status="UNRESOLVED"
            )
        ]
    )
    result = generate_code(spec_text, "Verilog", efs_ir=efs_ir)
    assert "SPECIFICATION_INCOMPLETE" in result
    assert "CONF_PORT_01" in result or "Contradictory interface specification" in result


# =========================================================================
# 13. Regression: Port Comments Do Not Trigger Procedural Placeholder Error
# =========================================================================
def test_port_description_with_reset_comment_not_flagged_as_placeholder():
    """Verify that HDL port descriptions containing '// active low reset' are not flagged as procedural placeholders."""
    from core.agents.rtl_agent import validate_rtl_plan

    plan = {
        "module_name": "TestModule",
        "ports": [
            {"name": "clk", "direction": "input", "width": "1", "description": "Clock input"},
            {"name": "rst_n", "direction": "input", "width": "1", "description": "input wire rst_n, // Active low reset"},
            {"name": "data_out", "direction": "output", "width": "32", "description": "Data output"}
        ],
        "always_ff_body": "always_ff @(posedge clk or negedge rst_n) begin if (!rst_n) data_out <= 32'd0; else data_out <= data_out + 1'b1; end",
        "always_comb_body": "always_comb begin end",
        "specification_status": {"status": "COMPLETE", "blocking_issues": []}
    }

    val = validate_rtl_plan(plan)
    # Ensure no PLAN_INVALID error is generated for '// active low reset' in ports
    placeholder_issues = [
        iss for iss in val.get("issues", [])
        if "placeholder code comment" in iss.get("issue", "").lower()
        and ("ports" in iss.get("issue", "").lower() or "specification_status" in iss.get("issue", "").lower())
    ]
    assert len(placeholder_issues) == 0, f"False positive placeholder issues found: {placeholder_issues}"


def test_signal_from_dict_self_heals_contradictory_hdl_description():
    """Verify that deserializing an EFSSignal with stale direction resolves against authoritative HDL description."""
    sig_raw = {
        "signal_id": "SIG_TEST_01",
        "name": "axi_araddr",
        "direction": "input",
        "width": "1",
        "description": "output wire [31:0] axi_araddr,"
    }
    sig = EFSSignal.from_dict(sig_raw)
    assert sig.direction == "output"
    assert sig.width == "32"

