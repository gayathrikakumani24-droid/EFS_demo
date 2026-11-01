"""
Comprehensive Test Suite for Semantic and Functional Correctness in EFS RTL Generation.

Validates that:
1. Declarations use legal identifiers without natural language fragments or group headings.
2. Registers have sized synthesizable reset values (never 'None' or 'null').
3. Datapath operations and ALU arithmetic are fully implemented and connected to FSM execution.
4. FSM wait states correctly evaluate external transaction handshakes (rvalid, bvalid, etc.).
5. Status signals are semantically correct (done != ERROR, error != M0).
6. Interface signals are driven by protocol states, never arbitrary data registers.
7. HDL validator correctly catches illegal identifiers, undefined values, status inversions,
   and arbitrary register mappings across generic designs.
"""

import pytest
import re
from core.efs_ir.models import EFSIR, EFSMetadata, EFSSignal, EFSRegister, EFSFSM, EFSState, EFSTransition
from core.agents.rtl_agent import plan_rtl, get_heuristic_code_template, sanitize_hardware_registers, sanitize_module_name
from core.verification.hdl_validator import validate_rtl_syntax_and_structure, HDLLanguage


def test_sanitize_hardware_registers_and_module_name():
    """Verify generic sanitization of hardware registers and module names."""
    # Module name sanitization
    long_desc = "builds upon the original SME architecture while adding configurability for matrix sizes"
    mod_name = sanitize_module_name(long_desc)
    assert re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", mod_name)
    assert len(mod_name) <= 40
    assert " " not in mod_name

    # Register sanitization with redundant range headers and categories
    raw_regs = [
        {"name": "M0", "width": "32", "reset_val": "32'h0"},
        {"name": "M1", "width": "32", "reset_val": None},
        {"name": "M2", "width": "32", "reset_val": "0"},
        {"name": "Matrix Registers M0-M7", "width": "32", "reset_val": None},
        {"name": "Matrix Registers", "width": "32", "reset_val": None},
        {"name": "Configuration Registers", "width": "32", "reset_val": None},
        {"name": "Status Registers", "width": "32", "reset_val": "32'h0"},
    ]
    sanitized = sanitize_hardware_registers(raw_regs)
    san_names = [r["name"] for r in sanitized]

    # Individual registers retained
    assert "M0" in san_names
    assert "M1" in san_names
    assert "M2" in san_names

    # Redundant range headers removed
    assert "Matrix Registers M0-M7" not in san_names
    assert "Matrix Registers" not in san_names

    # Category names converted to legal identifiers
    assert "cfg_reg" in san_names
    assert "status_reg" in san_names

    # All names are legal HDL identifiers
    for r in sanitized:
        assert re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", r["name"])
        # Reset value is non-empty sized constant, never None or null
        assert r["reset_val"] is not None
        assert "None" not in str(r["reset_val"])
        assert "null" not in str(r["reset_val"])
        assert "'" in str(r["reset_val"]) or str(r["reset_val"]).isdigit()


def test_matrix_compute_unit_semantic_and_functional_generation():
    """
    Verify complete, specification-driven RTL generation for multi-core matrix compute unit.
    Must have legal identifiers, valid resets, FSM wait states, ALU execution, and correct outputs.
    """
    efs_ir = EFSIR(
        metadata=EFSMetadata(
            design_id="DSN_MATRIX_COMPUTE",
            design_name="matrix_compute_unit",
            top_module="matrix_compute_unit",
            description="High-performance matrix compute engine with AXI4 memory interface"
        ),
        signals=[
            EFSSignal(name="clk", direction="input", width=1, semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", width=1, semantic_role="reset"),
            EFSSignal(name="start", direction="input", width=1, semantic_role="control"),
            EFSSignal(name="instruction", direction="input", width=32, semantic_role="data"),
            EFSSignal(name="axi_araddr", direction="output", width=32, semantic_role="address"),
            EFSSignal(name="axi_arvalid", direction="output", width=1, semantic_role="control"),
            EFSSignal(name="axi_rready", direction="output", width=1, semantic_role="control"),
            EFSSignal(name="axi_rdata", direction="input", width=32, semantic_role="data"),
            EFSSignal(name="axi_rvalid", direction="input", width=1, semantic_role="control"),
            EFSSignal(name="axi_awaddr", direction="output", width=32, semantic_role="address"),
            EFSSignal(name="axi_awvalid", direction="output", width=1, semantic_role="control"),
            EFSSignal(name="axi_wdata", direction="output", width=32, semantic_role="data"),
            EFSSignal(name="axi_wstrb", direction="output", width=4, semantic_role="control"),
            EFSSignal(name="axi_wvalid", direction="output", width=1, semantic_role="control"),
            EFSSignal(name="axi_bready", direction="output", width=1, semantic_role="control"),
            EFSSignal(name="axi_bvalid", direction="input", width=1, semantic_role="control"),
            EFSSignal(name="done", direction="output", width=1, semantic_role="status"),
            EFSSignal(name="busy", direction="output", width=1, semantic_role="status"),
            EFSSignal(name="error", direction="output", width=1, semantic_role="status"),
            EFSSignal(name="ready", direction="output", width=1, semantic_role="status"),
        ],
        registers=[
            EFSRegister(name="M0", width=32, reset_value="32'h0", description="Matrix register 0"),
            EFSRegister(name="M1", width=32, reset_value=None, description="Matrix register 1"),
            EFSRegister(name="M2", width=32, reset_value=None, description="Matrix register 2"),
            EFSRegister(name="M3", width=32, reset_value=None, description="Matrix register 3"),
            EFSRegister(name="Matrix Registers M0-M7", width=32, reset_value=None, description="Header"),
            EFSRegister(name="Configuration Registers", width=32, reset_value=None, description="Header"),
            EFSRegister(name="Status Registers", width=32, reset_value=None, description="Header"),
        ],
        fsms=[
            EFSFSM(
                name="main_fsm",
                states=[
                    EFSState(name="IDLE"),
                    EFSState(name="FETCH"),
                    EFSState(name="DECODE"),
                    EFSState(name="EXECUTE"),
                    EFSState(name="WRITEBACK"),
                    EFSState(name="DONE"),
                    EFSState(name="ERROR")
                ],
                transitions=[
                    EFSTransition(from_state="IDLE", to_state="FETCH", condition="start"),
                    EFSTransition(from_state="FETCH", to_state="DECODE", condition="axi_rvalid"),
                    EFSTransition(from_state="DECODE", to_state="EXECUTE", condition="1'b1"),
                    EFSTransition(from_state="EXECUTE", to_state="WRITEBACK", condition="1'b1"),
                    EFSTransition(from_state="WRITEBACK", to_state="DONE", condition="axi_bvalid"),
                    EFSTransition(from_state="DONE", to_state="IDLE", condition="1'b1"),
                    EFSTransition(from_state="ERROR", to_state="IDLE", condition="start"),
                ]
            )
        ]
    )

    # 1. Generate SystemVerilog RTL
    code_sv = get_heuristic_code_template("SystemVerilog", efs_ir=efs_ir)

    # Check that illegal declarations with spaces do NOT exist
    assert "Matrix Registers M0-M7" not in code_sv
    assert "Configuration Registers" not in code_sv
    assert "Status Registers" not in code_sv

    # Check that registers have legal identifiers and sized resets
    assert "logic [31:0] M0;" in code_sv or "reg [31:0] M0;" in code_sv
    assert "logic [31:0] M1;" in code_sv or "reg [31:0] M1;" in code_sv
    assert "M0 <= 32'h0;" in code_sv
    assert "M1 <= 32'h0;" in code_sv
    assert "None" not in code_sv
    assert "null" not in code_sv

    # Check that status outputs are semantically correct
    assert "done = (state == DONE);" in code_sv
    assert "(state == ERROR)" not in code_sv.split("done =")[1].split(";")[0]
    assert "error = (state == ERROR);" in code_sv
    assert "error = M0;" not in code_sv

    # Check that interface handshake outputs are driven by protocol states, not M0
    assert "axi_arvalid = (state == FETCH);" in code_sv
    assert "axi_rready = (state == FETCH);" in code_sv
    assert "axi_awvalid = (state == WRITEBACK);" in code_sv
    assert "axi_wvalid = (state == WRITEBACK);" in code_sv
    assert "axi_bready = (state == WRITEBACK);" in code_sv
    assert "axi_wstrb = (state == WRITEBACK) ? 4'hF : 4'h0;" in code_sv
    assert "axi_arvalid = M0;" not in code_sv
    assert "axi_araddr = M0;" not in code_sv

    # Check datapath implementation: ALU operations in EXECUTE state
    assert "EXECUTE" in code_sv
    assert "alu_result <=" in code_sv
    assert "case (instruction" in code_sv or "alu_result <= M0 +" in code_sv

    # Validate SystemVerilog code with HDL validator
    val_report_sv = validate_rtl_syntax_and_structure(code_sv, target_language="SystemVerilog")
    critical_errors_sv = [e for e in val_report_sv.errors if e.severity.value in ("CRITICAL", "MAJOR")]
    assert len(critical_errors_sv) == 0, f"SystemVerilog errors: {[e.message for e in critical_errors_sv]}"
    assert val_report_sv.is_valid

    # 2. Generate Verilog RTL
    code_v = get_heuristic_code_template("Verilog", efs_ir=efs_ir)

    # Check Verilog syntax
    assert "reg [31:0] M0;" in code_v
    assert "done = (state == DONE);" in code_v
    assert "error = (state == ERROR);" in code_v
    assert "logic" not in code_v

    val_report_v = validate_rtl_syntax_and_structure(code_v, target_language="Verilog")
    critical_errors_v = [e for e in val_report_v.errors if e.severity.value in ("CRITICAL", "MAJOR")]
    assert len(critical_errors_v) == 0, f"Verilog errors: {[e.message for e in critical_errors_v]}"
    assert val_report_v.is_valid


def test_validator_catches_illegal_identifiers():
    """Verify that HDL validator rejects declarations with spaces or invalid identifier syntax."""
    bad_code = """
module bad_module (
    input  wire clk,
    input  wire rst_n,
    output reg  done
);
    reg [31:0] Matrix Registers M0-M7;
    reg [31:0] Configuration Registers;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 1'b0;
        end else begin
            done <= 1'b1;
        end
    end
endmodule
"""
    report = validate_rtl_syntax_and_structure(bad_code, target_language="Verilog")
    rule_ids = {e.rule_id for e in report.errors}
    assert "RULE_ILLEGAL_IDENTIFIER" in rule_ids


def test_validator_catches_undefined_values():
    """Verify that HDL validator rejects assignments of Python None or null."""
    bad_code = """
module bad_module (
    input  wire clk,
    input  wire rst_n,
    output reg [31:0] data_out
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= None;
        end else begin
            data_out <= 32'hA;
        end
    end
endmodule
"""
    report = validate_rtl_syntax_and_structure(bad_code, target_language="Verilog")
    rule_ids = {e.rule_id for e in report.errors}
    assert "RULE_UNDEFINED_VALUE" in rule_ids


def test_validator_catches_semantic_status_inversion():
    """Verify that HDL validator rejects done = (state == ERROR) and error = M0."""
    bad_code_done = """
module bad_fsm (
    input  logic clk,
    input  logic rst_n,
    output logic done,
    output logic error
);
    typedef enum logic [2:0] { IDLE, RUN, DONE, ERROR } state_t;
    state_t state, next_state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= IDLE;
        else state <= next_state;
    end

    always_comb begin
        next_state = state;
        done = (state == ERROR);
        error = 1'b0;
    end
endmodule
"""
    report_done = validate_rtl_syntax_and_structure(bad_code_done, target_language="SystemVerilog")
    rule_ids_done = {e.rule_id for e in report_done.errors}
    assert "RULE_SEMANTIC_STATUS_INVERSION" in rule_ids_done

    bad_code_error = """
module bad_error (
    input  wire clk,
    input  wire rst_n,
    output reg  error
);
    reg [31:0] M0;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            error <= 1'b0;
            M0 <= 32'h0;
        end else begin
            error <= M0;
            M0 <= M0 + 1;
        end
    end
endmodule
"""
    report_error = validate_rtl_syntax_and_structure(bad_code_error, target_language="Verilog")
    rule_ids_error = {e.rule_id for e in report_error.errors}
    assert "RULE_SEMANTIC_STATUS_INVERSION" in rule_ids_error


def test_validator_catches_unjustified_register_mapping():
    """Verify that HDL validator rejects interface handshake signals driven by arbitrary data registers."""
    bad_code = """
module bad_axi (
    input  logic clk,
    input  logic rst_n,
    output logic axi_arvalid,
    output logic [31:0] axi_araddr
);
    logic [31:0] M0;
    always_comb begin
        axi_arvalid = M0;
        axi_araddr = M0;
    end
endmodule
"""
    report = validate_rtl_syntax_and_structure(bad_code, target_language="SystemVerilog")
    rule_ids = {e.rule_id for e in report.errors}
    assert "RULE_UNJUSTIFIED_REGISTER_MAPPING" in rule_ids


def test_validator_catches_arbitrary_unconditional_assignments():
    """Verify that HDL validator flags ungated round-robin input assignments to multiple registers."""
    bad_code = """
module bad_seq (
    input  wire clk,
    input  wire rst_n,
    input  wire [31:0] in0,
    input  wire [31:0] in1,
    input  wire [31:0] in2,
    output reg  [31:0] out0
);
    reg [31:0] M0;
    reg [31:0] M1;
    reg [31:0] M2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            M0 <= 32'h0;
            M1 <= 32'h0;
            M2 <= 32'h0;
        end else begin
            M0 <= in0;
            M1 <= in1;
            M2 <= in2;
        end
    end

    always @(*) begin
        out0 = M0;
    end
endmodule
"""
    report = validate_rtl_syntax_and_structure(bad_code, target_language="Verilog")
    rule_ids = {e.rule_id for e in report.errors}
    assert "RULE_ARBITRARY_UNCONDITIONAL_ASSIGNMENT" in rule_ids


def test_multiple_diverse_architectures_generalization():
    """
    Test generalization across 3 distinct hardware architectures:
    1. Streaming FIFO packet processor
    2. Memory controller with read/write wait states
    3. Multi-function ALU with opcode decoding
    """
    # Arch 1: Streaming packet processor
    spec_packet = """
# Specification: packet_streamer
Module: packet_streamer
Interfaces:
  - clk: input, width 1, semantic_role: clock
  - rst_n: input, width 1, semantic_role: reset
  - in_valid: input, width 1, semantic_role: control
  - in_data: input, width 32, semantic_role: data
  - in_ready: output, width 1, semantic_role: control
  - out_valid: output, width 1, semantic_role: control
  - out_data: output, width 32, semantic_role: data
  - out_ready: input, width 1, semantic_role: control
States:
  - IDLE: initial
  - RECEIVE: wait for in_valid
  - PROCESS: compute packet
  - TRANSMIT: wait for out_ready
"""
    code_pkg = get_heuristic_code_template("packet_streamer", "SystemVerilog", design_spec=spec_packet)
    val_pkg = validate_rtl_syntax_and_structure(code_pkg, target_language="SystemVerilog")
    assert val_pkg.is_valid, f"Packet streamer errors: {[e.message for e in val_pkg.errors]}"

    # Arch 2: Memory controller with read/write wait states
    spec_mem = """
# Specification: mem_bus_controller
Module: mem_bus_controller
Interfaces:
  - clk: input, width 1, semantic_role: clock
  - rst_n: input, width 1, semantic_role: reset
  - req: input, width 1, semantic_role: control
  - rvalid: input, width 1, semantic_role: control
  - bvalid: input, width 1, semantic_role: control
  - rdata: input, width 32, semantic_role: data
  - arvalid: output, width 1, semantic_role: control
  - awvalid: output, width 1, semantic_role: control
  - done: output, width 1, semantic_role: status
  - busy: output, width 1, semantic_role: status
States:
  - IDLE: initial
  - FETCH: wait for rvalid
  - WRITEBACK: wait for bvalid
  - DONE: complete
"""
    code_mem = get_heuristic_code_template("mem_bus_controller", "Verilog", design_spec=spec_mem)
    val_mem = validate_rtl_syntax_and_structure(code_mem, target_language="Verilog")
    assert val_mem.is_valid, f"Memory controller errors: {[e.message for e in val_mem.errors]}"

    # Arch 3: Pure combinational ALU
    spec_alu = """
# Specification: combinational_alu
Module: combinational_alu
Interfaces:
  - a: input, width 16, semantic_role: data
  - b: input, width 16, semantic_role: data
  - sel: input, width 1, semantic_role: control
  - result: output, width 16, semantic_role: data
"""
    code_alu = get_heuristic_code_template("combinational_alu", "Verilog", design_spec=spec_alu)
    val_alu = validate_rtl_syntax_and_structure(code_alu, target_language="Verilog")
    assert val_alu.is_valid, f"Combinational ALU errors: {[e.message for e in val_alu.errors]}"
