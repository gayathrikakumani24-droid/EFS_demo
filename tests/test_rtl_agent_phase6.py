"""
Unit tests for Phase 6: Constrained Generation:
1. Deterministic Module Header Generation and Strict Plan Header Matching.
2. Structured Sections Prompting, Assembly, and Independent Validation/Repair.
3. NEEDS_REVIEW Marker Detection and Surfacing for Non-Traceable Plan Logic.
"""

import json
import pytest

from core.agents.constrained_generator import (
    emit_deterministic_module_header,
    extract_header_from_code,
    verify_module_header_matches_plan,
    enforce_deterministic_module_header,
    assemble_structured_sections,
    parse_structured_sections_from_json,
    validate_single_section,
    extract_needs_review_markers,
    surface_needs_review_header,
    format_port_width,
)
from core.agents.rtl_agent import generate_code, plan_rtl
from core.verification.hdl_validator import validate_rtl_syntax_and_structure
from core.efs_ir.models import (
    EFSIR,
    EFSMetadata,
    EFSSignal,
    EFSFSM,
    EFSState,
    EFSTransition,
)


def test_phase6_format_port_width():
    """Verify width formatting logic for scalar, bit vectors, and parameterized ranges."""
    assert format_port_width(1) == ""
    assert format_port_width("1") == ""
    assert format_port_width("0") == ""
    assert format_port_width(8) == "[7:0]"
    assert format_port_width("32") == "[31:0]"
    assert format_port_width("31:0") == "[31:0]"
    assert format_port_width("[15:0]") == "[15:0]"
    assert format_port_width("WIDTH") == "[WIDTH-1:0]"


def test_phase6_deterministic_module_header_emission_and_verification():
    """Verify deterministic module header emission matches plan parameters, ports, widths, and order."""
    plan = {
        "module_name": "spi_master_controller",
        "parameters": [
            {"name": "DATA_WIDTH", "type": "int", "default_value": "8"},
            {"name": "CLK_DIV", "type": "int", "default_value": "4"}
        ],
        "ports": [
            {"name": "clk", "direction": "input", "width": "1"},
            {"name": "rst_n", "direction": "input", "width": "1"},
            {"name": "start", "direction": "input", "width": "1"},
            {"name": "tx_data", "direction": "input", "width": "8"},
            {"name": "sclk", "direction": "output", "width": "1"},
            {"name": "mosi", "direction": "output", "width": "1"},
            {"name": "miso", "direction": "input", "width": "1"},
            {"name": "rx_data", "direction": "output", "width": "8"},
            {"name": "busy", "direction": "output", "width": "1"},
            {"name": "done", "direction": "output", "width": "1"}
        ]
    }

    # 1. Emit SystemVerilog Header
    sv_header = emit_deterministic_module_header(plan, target_language="SystemVerilog")
    assert "module spi_master_controller #(" in sv_header
    assert "parameter int DATA_WIDTH = 8" in sv_header
    assert "parameter int CLK_DIV = 4" in sv_header
    assert "input logic [7:0] tx_data" in sv_header
    assert "output logic [7:0] rx_data" in sv_header
    assert "output logic done" in sv_header
    assert sv_header.endswith(");")

    # 2. Emit Verilog Header
    v_header = emit_deterministic_module_header(plan, target_language="Verilog")
    assert "module spi_master_controller #(" in v_header
    assert "parameter DATA_WIDTH = 8" in v_header
    assert "input wire [7:0] tx_data" in v_header
    assert "output reg [7:0] rx_data" in v_header

    # 3. Verify matching header passes
    is_match, errs = verify_module_header_matches_plan(sv_header, sv_header, plan)
    assert is_match is True
    assert len(errs) == 0

    # 4. Verify modified header (dropped port or wrong direction) fails verification
    bad_header = """
    module spi_master_controller #(
        parameter int DATA_WIDTH = 8
    ) (
        input  logic clk,
        input  logic rst_n,
        output logic tx_data  // Direction inverted!
    );
    """
    is_match, errs = verify_module_header_matches_plan(bad_header, sv_header, plan)
    assert is_match is False
    assert any("direction mismatch" in e.lower() for e in errs)
    assert any("missing from generated module header" in e.lower() for e in errs)

    # 5. Enforce deterministic header replaces bad header
    body_only = """
        logic [7:0] shift_reg;
        assign busy = 1'b0;
    endmodule
    """
    enforced = enforce_deterministic_module_header(body_only, sv_header, plan)
    assert enforced.startswith("module spi_master_controller")
    assert "input logic [7:0] tx_data" in enforced
    assert "logic [7:0] shift_reg;" in enforced
    assert enforced.strip().endswith("endmodule")


def test_phase6_structured_sections_assembly_and_independent_validation():
    """Verify assembling structured sections into synthesizable module and independent section validation."""
    header = """module alu_core (
    input  logic        clk,
    input  logic        rst_n,
    input  logic [1:0]  opcode,
    input  logic [15:0] a,
    input  logic [15:0] b,
    output logic [15:0] result,
    output logic        zero
);"""

    sections = {
        "declarations": "logic [15:0] alu_out_comb;\n    logic [15:0] alu_reg;",
        "next_state_logic": """always_comb begin
        case (opcode)
            2'b00: alu_out_comb = a + b;
            2'b01: alu_out_comb = a - b;
            2'b10: alu_out_comb = a & b;
            default: alu_out_comb = a ^ b;
        endcase
    end""",
        "sequential_logic": """always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            alu_reg <= 16'd0;
        end else begin
            alu_reg <= alu_out_comb;
        end
    end""",
        "output_logic": """assign result = alu_reg;
    assign zero = (alu_reg == 16'd0);"""
    }

    # 1. Validate each section independently
    plan = {"module_name": "alu_core"}
    for sec_name, sec_code in sections.items():
        is_val, errs = validate_single_section(sec_name, sec_code, plan, target_language="SystemVerilog")
        assert is_val is True, f"Section {sec_name} failed validation: {errs}"

    # 2. Test independent validation catches defects in faulty section
    bad_seq_logic = """
    // Missing clocked always trigger
    alu_reg <= alu_out_comb;
    """
    is_val, errs = validate_single_section("sequential_logic", bad_seq_logic, plan, target_language="SystemVerilog")
    assert is_val is False
    assert any("clocked always" in e for e in errs)

    bad_comb_logic = """
    // Combinational block using posedge clk
    always_comb @(posedge clk) begin
        alu_out_comb = a + b;
    end
    """
    is_val, errs = validate_single_section("next_state_logic", bad_comb_logic, plan, target_language="SystemVerilog")
    assert is_val is False
    assert any("clock" in e.lower() for e in errs)

    # 3. Assemble structured sections into full module
    assembled = assemble_structured_sections(header, sections, target_language="SystemVerilog")
    assert "Section 1: Internal Signal & Register Declarations" in assembled
    assert "Section 2: Next-State Combinational Logic" in assembled
    assert "Section 3: Clocked Sequential State" in assembled
    assert "Section 4: Output Decode Logic" in assembled
    assert assembled.strip().endswith("endmodule")

    # 4. Validate full assembled code syntax
    report = validate_rtl_syntax_and_structure(assembled, target_language="SystemVerilog")
    assert report.is_valid is True, f"Assembled module syntax errors: {[e.message for e in report.errors]}"


def test_phase6_needs_review_markers_extraction_and_surfacing():
    """Verify extraction of // NEEDS_REVIEW: <reason> comments and surfacing in banner."""
    code_with_markers = """module dma_engine (
    input  logic clk,
    input  logic rst_n,
    output logic done
);

    logic [3:0] burst_counter; // NEEDS_REVIEW: Burst counter width of 4 bits assumed since burst length unspecified
    logic watchdog_timeout;    // NEEDS_REVIEW: Unspecified safety timeout timer introduced to prevent bus deadlock

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            burst_counter <= 4'd0;
        end else begin
            burst_counter <= burst_counter + 4'd1;
        end
    end

    assign done = (burst_counter == 4'd15);

endmodule
"""
    markers = extract_needs_review_markers(code_with_markers)
    assert len(markers) == 2
    assert "Burst counter width of 4 bits assumed" in markers[0]["reason"]
    assert "Unspecified safety timeout timer introduced" in markers[1]["reason"]
    assert markers[0]["line_number"] == 7
    assert markers[1]["line_number"] == 8

    # Surface header banner
    surfaced_code = surface_needs_review_header(code_with_markers, markers)
    assert "ATTENTION: 2 UNTRACEABLE CONSTRUCT(S) REQUIRE REVIEW" in surfaced_code
    assert "NEEDS_REVIEW - Burst counter width" in surfaced_code
    assert "NEEDS_REVIEW - Unspecified safety timeout" in surfaced_code
    assert surfaced_code.endswith("endmodule\n")


def test_phase6_end_to_end_constrained_generation_flow():
    """Verify generate_code incorporates deterministic header, structured assembly, and review surfacing."""
    spec_text = """
    # Hardware Design Specification: packet_checker
    Packet integrity check block.
    ## Interface
    - clk: input, 1 bit, Clock
    - rst_n: input, 1 bit, Reset
    - data_in: input, 8 bits, Input stream
    - valid_in: input, 1 bit, Valid
    - err_out: output, 1 bit, Parity error flag
    - ready_out: output, 1 bit, Ready flag
    ## Output Rules
    - err_out: driven by parity error detection logic
    - ready_out: driven by 1'b1
    """

    class MockConstrainedLLM:
        available = True
        last_usage = {"prompt_tokens": 500, "completion_tokens": 300}

        def complete_json(self, system_prompt, prompt, model=None):
            return {
                "module_name": "packet_checker",
                "specification_status": {"status": "COMPLETE", "blocking_issues": []},
                "ports": [
                    {"name": "clk", "direction": "input", "width": "1", "semantic_role": "clock"},
                    {"name": "rst_n", "direction": "input", "width": "1", "semantic_role": "reset"},
                    {"name": "data_in", "direction": "input", "width": "8", "semantic_role": "data"},
                    {"name": "valid_in", "direction": "input", "width": "1", "semantic_role": "control"},
                    {"name": "err_out", "direction": "output", "width": "1", "semantic_role": "status"},
                    {"name": "ready_out", "direction": "output", "width": "1", "semantic_role": "status"},
                ],
                "parameters": [],
                "internal_registers": [],
                "fsm": None,
                "output_rules": {
                    "err_out": {"driving_expression": "^data_in", "source_ref": "err_out: driven by parity error detection logic"},
                    "ready_out": {"driving_expression": "1'b1", "source_ref": "ready_out: driven by 1'b1"}
                }
            }

        def complete(self, system_prompt, prompt, model=None):
            # Return body logic with a NEEDS_REVIEW marker
            return """
    logic [7:0] latched_data;
    logic parity_calc; // NEEDS_REVIEW: Odd parity calculation chosen as default since spec parity format unspecified

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            latched_data <= 8'd0;
            err_out <= 1'b0;
        end else if (valid_in) begin
            latched_data <= data_in;
            err_out <= ^data_in;
        end
    end

    assign ready_out = 1'b1;
endmodule
"""

    import core.agents.rtl_agent as rtl_mod
    orig_get_llm = rtl_mod.get_llm_client
    rtl_mod.get_llm_client = lambda: MockConstrainedLLM()

    try:
        code = generate_code(spec_text, "SystemVerilog")
        # 1. Deterministic header must be present
        assert "module packet_checker (" in code
        assert "input logic [7:0] data_in" in code
        assert "output logic err_out" in code
        assert "output logic ready_out" in code

        # 2. NEEDS_REVIEW banner must be surfaced at top
        assert "ATTENTION: 1 UNTRACEABLE CONSTRUCT(S) REQUIRE REVIEW" in code
        assert "Odd parity calculation chosen as default" in code

        # 3. Overall syntax must validate cleanly
        report = validate_rtl_syntax_and_structure(code, target_language="SystemVerilog")
        assert report.is_valid is True, f"Constrained generation syntax failed: {[e.message for e in report.errors]}"
    finally:
        rtl_mod.get_llm_client = orig_get_llm
