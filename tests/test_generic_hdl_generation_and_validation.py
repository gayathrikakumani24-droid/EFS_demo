"""
Test Suite: Generic Verilog and SystemVerilog RTL Generation, Validation, Repair, and Traceability.

Verifies:
1. Language selection and resolution (Verilog vs SystemVerilog).
2. Pure combinational designs (no clocks/resets fabricated).
3. Verilog generation (strict Verilog-2001/2005 constructs, NO logic, NO always_ff/always_comb, NO typedef).
4. SystemVerilog generation (proper logic, always_ff, always_comb, typedef enum).
5. Language-aware validation detecting SV constructs in Verilog mode.
6. Language-aware repair preserving the target language.
7. Requirement-to-RTL Traceability report generation and unmapped element detection.
8. Non-hardcoding: arbitrary ports, arbitrary state names, arbitrary datapath widths.
"""

import pytest
from core.efs_ir.models import (
    EFSIR,
    EFSMetadata,
    EFSComponent,
    EFSInterface,
    EFSSignal,
    EFSRegister,
    EFSFSM,
    EFSState,
    EFSTransition,
    EFSTraceability,
)
from core.verification.hdl_validator import (
    validate_rtl_syntax_and_structure,
    resolve_hdl_language,
    HDLLanguage,
)
from core.verification.traceability import (
    generate_traceability_report,
    TraceabilityReport,
)
from core.agents.rtl_agent import (
    generate_code,
    get_heuristic_code_template,
    get_system_prompt,
    validate_generated_code,
)
from core.verification.static_verifier import verify_statically
from core.agents.repair_agent import get_heuristic_repair


def test_language_resolution():
    """Test resolution of various target language strings."""
    assert resolve_hdl_language("Verilog") == HDLLanguage.VERILOG
    assert resolve_hdl_language("verilog-2001") == HDLLanguage.VERILOG
    assert resolve_hdl_language("Verilog-2005") == HDLLanguage.VERILOG
    assert resolve_hdl_language("v") == HDLLanguage.VERILOG

    assert resolve_hdl_language("SystemVerilog") == HDLLanguage.SYSTEMVERILOG
    assert resolve_hdl_language("systemverilog-2012") == HDLLanguage.SYSTEMVERILOG
    assert resolve_hdl_language("sv") == HDLLanguage.SYSTEMVERILOG

    with pytest.raises(ValueError):
        resolve_hdl_language("UnsupportedHDL")


def test_pure_combinational_alu_verilog_no_clocks():
    """Verify that a pure combinational design does NOT generate clocks, resets, or sequential blocks."""
    efs_ir = EFSIR(
        metadata=EFSMetadata(top_module="combinational_alu"),
        components=[EFSComponent(component_id="comp_alu", name="combinational_alu")],
        signals=[
            EFSSignal(signal_id="s1", name="op_a", direction="input", width=8, semantic_role="data"),
            EFSSignal(signal_id="s2", name="op_b", direction="input", width=8, semantic_role="data"),
            EFSSignal(signal_id="s3", name="alu_sel", direction="input", width=2, semantic_role="control"),
            EFSSignal(signal_id="s4", name="alu_res", direction="output", width=8, semantic_role="data"),
        ]
    )

    design_spec = """
    # Combinational ALU Specification
    Module: combinational_alu
    | Port | Direction | Width |
    | op_a | input | 8 |
    | op_b | input | 8 |
    | alu_sel | input | 2 |
    | alu_res | output | 8 |
    """

    code = get_heuristic_code_template(design_spec, "Verilog", efs_ir=efs_ir)

    # Must be valid Verilog
    assert "module combinational_alu" in code
    assert "endmodule" in code
    assert "input  wire [7:0] op_a" in code
    assert "input  wire [7:0] op_b" in code
    assert "input  wire [1:0] alu_sel" in code
    assert "output reg  [7:0] alu_res" in code

    # Must NOT fabricate clock or reset
    assert "clk" not in code.lower()
    assert "rst" not in code.lower()
    assert "posedge" not in code
    assert "negedge" not in code

    # Must NOT contain SystemVerilog-only keywords
    assert "logic" not in code
    assert "always_comb" not in code
    assert "always_ff" not in code

    # Must validate cleanly in Verilog mode
    report = validate_rtl_syntax_and_structure(code, "Verilog", efs_ir=efs_ir)
    assert report.valid, f"Validation errors: {[e.message for e in report.errors]}"


def test_pure_combinational_alu_systemverilog():
    """Verify pure combinational design generates clean SystemVerilog with logic and always_comb."""
    efs_ir = EFSIR(
        metadata=EFSMetadata(top_module="combinational_alu"),
        components=[EFSComponent(component_id="comp_alu", name="combinational_alu")],
        signals=[
            EFSSignal(signal_id="s1", name="op_a", direction="input", width=8, semantic_role="data"),
            EFSSignal(signal_id="s2", name="op_b", direction="input", width=8, semantic_role="data"),
            EFSSignal(signal_id="s3", name="alu_sel", direction="input", width=2, semantic_role="control"),
            EFSSignal(signal_id="s4", name="alu_res", direction="output", width=8, semantic_role="data"),
        ]
    )

    design_spec = """
    # Combinational ALU Specification
    Module: combinational_alu
    | Port | Direction | Width |
    | op_a | input | 8 |
    | op_b | input | 8 |
    | alu_sel | input | 2 |
    | alu_res | output | 8 |
    """

    code = get_heuristic_code_template(design_spec, "SystemVerilog", efs_ir=efs_ir)

    assert "module combinational_alu" in code
    assert "input  logic [7:0] op_a" in code
    assert "output logic [7:0] alu_res" in code
    assert "always_comb" in code

    # Must validate cleanly in SystemVerilog mode
    report = validate_rtl_syntax_and_structure(code, "SystemVerilog", efs_ir=efs_ir)
    assert report.valid, f"Validation errors: {[e.message for e in report.errors]}"


def test_fsm_sequential_design_verilog_no_hardcoded_states():
    """Verify sequential FSM uses EFS IR state names and strict Verilog-2001 localparam."""
    efs_ir = EFSIR(
        metadata=EFSMetadata(top_module="packet_processor"),
        components=[EFSComponent(component_id="comp_pkt", name="packet_processor")],
        signals=[
            EFSSignal(signal_id="clk_sig", name="sys_clk", direction="input", width=1, semantic_role="clock"),
            EFSSignal(signal_id="rst_sig", name="sys_rst_n", direction="input", width=1, semantic_role="reset"),
            EFSSignal(signal_id="sop", name="sop_in", direction="input", width=1, semantic_role="control"),
            EFSSignal(signal_id="eop", name="eop_in", direction="input", width=1, semantic_role="control"),
            EFSSignal(signal_id="pkt_valid", name="pkt_valid_out", direction="output", width=1, semantic_role="control"),
        ],
        fsms=[
            EFSFSM(
                fsm_id="fsm1",
                name="packet_fsm",
                states=[
                    EFSState(state_id="st1", name="ST_HEADER_PARSE", encoding=0),
                    EFSState(state_id="st2", name="ST_PAYLOAD_EXTRACT", encoding=1),
                    EFSState(state_id="st3", name="ST_CHECKSUM_VERIFY", encoding=2),
                ],
                transitions=[
                    EFSTransition(source_state="ST_HEADER_PARSE", target_state="ST_PAYLOAD_EXTRACT", condition="sop_in"),
                    EFSTransition(source_state="ST_PAYLOAD_EXTRACT", target_state="ST_CHECKSUM_VERIFY", condition="eop_in"),
                ]
            )
        ]
    )

    design_spec = """
    # Packet Processor Specification
    Module: packet_processor
    """

    code = get_heuristic_code_template(design_spec, "Verilog", efs_ir=efs_ir)

    # Assert exact arbitrary state names are present
    assert "ST_HEADER_PARSE" in code
    assert "ST_PAYLOAD_EXTRACT" in code
    assert "ST_CHECKSUM_VERIFY" in code
    # Assert NO hardcoded state names
    assert "IDLE" not in code
    assert "RUN" not in code
    assert "DONE" not in code

    # Assert strict Verilog constructs
    assert "localparam" in code
    assert "always @(posedge sys_clk or negedge sys_rst_n)" in code
    assert "reg" in code
    assert "logic" not in code
    assert "always_ff" not in code
    assert "always_comb" not in code
    assert "typedef" not in code

    report = validate_rtl_syntax_and_structure(code, "Verilog", efs_ir=efs_ir)
    assert report.valid, f"Validation errors: {[e.message for e in report.errors]}"


def test_fsm_sequential_design_systemverilog():
    """Verify sequential FSM in SystemVerilog uses typedef enum logic and always_ff."""
    efs_ir = EFSIR(
        metadata=EFSMetadata(top_module="packet_processor"),
        components=[EFSComponent(component_id="comp_pkt", name="packet_processor")],
        signals=[
            EFSSignal(signal_id="clk_sig", name="sys_clk", direction="input", width=1, semantic_role="clock"),
            EFSSignal(signal_id="rst_sig", name="sys_rst_n", direction="input", width=1, semantic_role="reset"),
            EFSSignal(signal_id="sop", name="sop_in", direction="input", width=1, semantic_role="control"),
            EFSSignal(signal_id="pkt_valid", name="pkt_valid_out", direction="output", width=1, semantic_role="control"),
        ],
        fsms=[
            EFSFSM(
                fsm_id="fsm1",
                name="packet_fsm",
                states=[
                    EFSState(state_id="st1", name="ST_HEADER_PARSE", encoding=0),
                    EFSState(state_id="st2", name="ST_PAYLOAD_EXTRACT", encoding=1),
                ]
            )
        ]
    )

    code = get_heuristic_code_template("Module: packet_processor", "SystemVerilog", efs_ir=efs_ir)

    assert "typedef enum logic" in code
    assert "ST_HEADER_PARSE" in code
    assert "ST_PAYLOAD_EXTRACT" in code
    assert "always_ff @(posedge sys_clk or negedge sys_rst_n)" in code
    assert "always_comb" in code

    report = validate_rtl_syntax_and_structure(code, "SystemVerilog", efs_ir=efs_ir)
    assert report.valid, f"Validation errors: {[e.message for e in report.errors]}"


def test_validator_detects_systemverilog_keywords_in_verilog_mode():
    """Verify validator flags 'logic', 'always_ff', and 'typedef' when validating Verilog."""
    invalid_verilog_code = """
    module bad_verilog (
        input logic clk,
        input logic rst_n,
        output logic [7:0] data_out
    );
        typedef enum logic [1:0] { S0, S1 } state_t;
        state_t state;

        always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) state <= S0;
            else state <= S1;
        end

        always_comb begin
            data_out = 8'h00;
        end
    endmodule
    """

    report = validate_rtl_syntax_and_structure(invalid_verilog_code, "Verilog")
    assert not report.valid
    error_cats = [e.category for e in report.errors]
    assert any("SV_KEYWORD" in c or "FORBIDDEN" in c for c in error_cats)

    # The exact line and keyword should be captured
    sv_errors = [e for e in report.errors if "SV_KEYWORD" in e.category or "FORBIDDEN" in e.category]
    messages = " ".join(e.message for e in sv_errors)
    assert "logic" in messages
    assert "always_ff" in messages
    assert "always_comb" in messages
    assert "typedef" in messages


def test_validator_detects_incomplete_module_and_unbalanced_blocks():
    """Verify validator catches truncated code and unclosed begin/end blocks."""
    truncated_code = """
    module truncated_module (
        input wire clk,
        output reg [7:0] out
    );
        always @(posedge clk) begin
            out <= 8'hAA;
    """

    report = validate_rtl_syntax_and_structure(truncated_code, "Verilog")
    assert not report.valid
    error_cats = [e.category for e in report.errors]
    assert any("ENDMODULE" in c or "BEGIN_END" in c for c in error_cats)


def test_traceability_report_generation():
    """Verify requirement-to-RTL traceability maps requirements and catches unmapped elements."""
    efs_ir = EFSIR(
        metadata=EFSMetadata(top_module="secure_crypto_core"),
        components=[EFSComponent(component_id="comp_1", name="secure_crypto_core")],
        signals=[
            EFSSignal(signal_id="sig_clk", name="core_clk", direction="input", width=1, semantic_role="clock",
                      traceability=EFSTraceability(doc_id="sec_spec.md", section="3.1", original_text="System clock core_clk")),
            EFSSignal(signal_id="sig_rst", name="core_rst_n", direction="input", width=1, semantic_role="reset",
                      traceability=EFSTraceability(doc_id="sec_spec.md", section="3.1", original_text="Active-low reset core_rst_n")),
            EFSSignal(signal_id="sig_key", name="aes_key", direction="input", width=128, semantic_role="data",
                      traceability=EFSTraceability(doc_id="sec_spec.md", section="3.2", original_text="128-bit AES encryption key")),
            EFSSignal(signal_id="sig_out", name="cipher_data", direction="output", width=128, semantic_role="data",
                      traceability=EFSTraceability(doc_id="sec_spec.md", section="3.3", original_text="Encrypted ciphertext output")),
        ],
        registers=[
            EFSRegister(register_id="reg_ctrl", name="control_reg", offset="0x04", width=32,
                        traceability=EFSTraceability(doc_id="sec_spec.md", section="4.1", original_text="Control register at 0x04"))
        ]
    )

    # Generated RTL has all signals and one unmapped/fabricated signal 'secret_backdoor_wire'
    rtl_code = """
    module secure_crypto_core (
        input wire core_clk,
        input wire core_rst_n,
        input wire [127:0] aes_key,
        output reg [127:0] cipher_data
    );
        reg [31:0] control_reg;
        wire [31:0] secret_backdoor_wire;

        always @(posedge core_clk or negedge core_rst_n) begin
            if (!core_rst_n) begin
                cipher_data <= 128'd0;
                control_reg <= 32'd0;
            end else begin
                cipher_data <= aes_key;
            end
        end
    endmodule
    """

    trace_report = generate_traceability_report(
        code=rtl_code,
        efs_ir=efs_ir,
        design_spec="Module: secure_crypto_core",
        target_language="Verilog"
    )

    assert trace_report.is_specification_compliant
    assert trace_report.coverage_percentage == 100.0
    assert len(trace_report.items) == 5  # 4 signals + 1 register

    # Verify detection of unmapped element
    unmapped_ids = [u.identifier for u in trace_report.unmapped_elements]
    assert "secret_backdoor_wire" in unmapped_ids

    # Markdown table must format correctly
    md = trace_report.to_markdown()
    assert "| Requirement | Type | EFS IR Entity | Status | RTL Mapping | Specification Citation |" in md
    assert "core_clk" in md
    assert "control_reg" in md


def test_language_aware_repair():
    """Verify repair heuristic sanitizes SystemVerilog constructs in Verilog mode."""
    broken_verilog = """
    module simple_counter (
        input wire clk,
        input wire rst_n
    );
    endmodule
    """

    compliance_report = {
        "failed_checks": [
            {
                "check_name": "Protocol Port Signal Presence",
                "severity": "CRITICAL",
                "violation": "Required signal 'count_out' is missing",
                "suggested_correction": "output logic [7:0] count_out;",
                "rtl_snippet": ");"
            }
        ]
    }

    # Repair in Verilog mode should replace 'logic' with 'wire'
    repair_result = get_heuristic_repair(
        broken_verilog,
        compliance_report,
        target_language="Verilog"
    )

    repaired_code = repair_result["repaired_code"]
    assert "logic" not in repaired_code
    assert "wire" in repaired_code or "output" in repaired_code
