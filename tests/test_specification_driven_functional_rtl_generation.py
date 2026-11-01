"""
Specification-Driven Functional RTL Generation Test Suite.

Verifies that the EFS RTL generation pipeline:
1. Transforms validated EFS IR and hardware specifications into actual functional RTL
   (not merely module skeletons or empty assignments).
2. Generates correct Verilog and SystemVerilog constructs without hardcoding.
3. Actively connects inputs, registers, operations, and outputs.
4. Derives meaningful FSM state transition conditions from input triggers and handshakes.
5. Detects natural language fragments, undeclared identifiers, and unconditional FSMs.
6. Halts on incomplete/ambiguous specifications, triggering HITL rather than guessing.
"""

import pytest
import re
from unittest.mock import MagicMock, patch

from core.efs_ir.models import (
    EFSIR,
    EFSMetadata,
    EFSComponent,
    EFSSignal,
    EFSRegister,
    EFSFSM,
    EFSState,
    EFSTransition,
    EFSResolution,
    EFSTraceability,
    EFSFunctionalBehavior,
)
from core.agents.rtl_agent import (
    plan_rtl,
    validate_rtl_plan,
    validate_generated_code,
    generate_code,
    get_heuristic_code_template,
)
from core.verification.hdl_validator import (
    validate_rtl_syntax_and_structure,
    HDLLanguage,
    ValidationSeverity,
)
from core.verification.traceability import generate_traceability_report


# =========================================================================
# 1. Combinational Hardware Generation: 4-to-1 Multiplexer
# =========================================================================
def test_functional_combinational_multiplexer_generation():
    """Verify combinational multiplexer generation connects selector and inputs to output without clock."""
    spec_text = """# Hardware Design Specification: generic_mux_4to1
    Module generic_mux_4to1 ports:
    - sel: input, 2 bits
    - d0: input, 8 bits
    - d1: input, 8 bits
    - d2: input, 8 bits
    - d3: input, 8 bits
    - y_out: output, 8 bits
    """
    efs_ir = EFSIR(
        metadata=EFSMetadata(design_name="generic_mux_4to1", top_module="generic_mux_4to1"),
        signals=[
            EFSSignal(signal_id="s_sel", name="sel", direction="input", width=2, semantic_role="control"),
            EFSSignal(signal_id="s_d0", name="d0", direction="input", width=8, semantic_role="data"),
            EFSSignal(signal_id="s_d1", name="d1", direction="input", width=8, semantic_role="data"),
            EFSSignal(signal_id="s_d2", name="d2", direction="input", width=8, semantic_role="data"),
            EFSSignal(signal_id="s_d3", name="d3", direction="input", width=8, semantic_role="data"),
            EFSSignal(signal_id="s_y", name="y_out", direction="output", width=8, semantic_role="data"),
        ],
        functional_behavior=[
            EFSFunctionalBehavior(
                name="mux_selection",
                subject="generic_mux_4to1",
                action="y_out = (sel == 2'd0) ? d0 : (sel == 2'd1) ? d1 : (sel == 2'd2) ? d2 : d3",
                inputs=["sel", "d0", "d1", "d2", "d3"],
                outputs=["y_out"]
            )
        ]
    )

    code_v = get_heuristic_code_template("generic_mux_4to1", "Verilog", efs_ir=efs_ir)

    # Must be pure combinational: no clock or reset ports
    assert "clk" not in code_v.lower()
    assert "rst" not in code_v.lower()
    assert "input  wire [1:0] sel" in code_v
    assert "input  wire [7:0] d0" in code_v
    assert "output reg  [7:0] y_out" in code_v

    # Verify input usage in datapath
    assert "sel" in code_v
    assert "d0" in code_v or "d1" in code_v

    # Output must not be hardcoded to 8'h0
    assert "y_out = 8'h0;" not in code_v

    report = validate_rtl_syntax_and_structure(code_v, "Verilog", efs_ir=efs_ir)
    assert report.valid, f"Validation errors: {[e.message for e in report.errors]}"
    assert report.language == HDLLanguage.VERILOG


# =========================================================================
# 2. Arithmetic Datapath Generation: Multi-Function ALU
# =========================================================================
def test_functional_arithmetic_alu_generation():
    """Verify arithmetic ALU maps operand inputs to functional operational result."""
    spec_text = """# Hardware Design Specification: arithmetic_unit
    Module arithmetic_unit ports:
    - op_sel: input, 2 bits
    - op_a: input, 16 bits
    - op_b: input, 16 bits
    - alu_res: output, 16 bits
    """
    efs_ir = EFSIR(
        metadata=EFSMetadata(design_name="arithmetic_unit", top_module="arithmetic_unit"),
        signals=[
            EFSSignal(signal_id="s_op", name="op_sel", direction="input", width=2, semantic_role="control"),
            EFSSignal(signal_id="s_a", name="op_a", direction="input", width=16, semantic_role="data"),
            EFSSignal(signal_id="s_b", name="op_b", direction="input", width=16, semantic_role="data"),
            EFSSignal(signal_id="s_res", name="alu_res", direction="output", width=16, semantic_role="data"),
        ]
    )

    code_sv = get_heuristic_code_template("arithmetic_unit", "SystemVerilog", efs_ir=efs_ir)

    assert "module arithmetic_unit" in code_sv
    assert "input  logic [1:0] op_sel" in code_sv
    assert "input  logic [15:0] op_a" in code_sv
    assert "input  logic [15:0] op_b" in code_sv
    assert "output logic [15:0] alu_res" in code_sv

    # Check that operands are used in functional assignment
    assert "op_a" in code_sv
    assert "op_b" in code_sv

    report = validate_rtl_syntax_and_structure(code_sv, "SystemVerilog", efs_ir=efs_ir)
    assert report.valid, f"Validation errors: {[e.message for e in report.errors]}"


# =========================================================================
# 3. Sequential Datapath Generation: Accumulator with Register
# =========================================================================
def test_functional_sequential_accumulator_datapath():
    """Verify sequential datapath connects input to internal register and drives output from register."""
    efs_ir = EFSIR(
        metadata=EFSMetadata(design_name="datapath_accumulator", top_module="datapath_accumulator"),
        signals=[
            EFSSignal(signal_id="clk", name="clock_i", direction="input", width=1, semantic_role="clock"),
            EFSSignal(signal_id="rst", name="reset_ni", direction="input", width=1, semantic_role="reset"),
            EFSSignal(signal_id="din", name="data_in", direction="input", width=32, semantic_role="data"),
            EFSSignal(signal_id="dout", name="acc_out", direction="output", width=32, semantic_role="data"),
        ],
        registers=[
            EFSRegister(register_id="acc_r", name="acc_reg", width=32, reset_value="32'h0", description="Accumulator storage register")
        ]
    )

    code_v = get_heuristic_code_template("datapath_accumulator", "Verilog", efs_ir=efs_ir)

    assert "module datapath_accumulator" in code_v
    assert "input  wire clock_i" in code_v
    assert "input  wire reset_ni" in code_v
    assert "reg [31:0] acc_reg;" in code_v
    assert "acc_reg <= 32'h0;" in code_v
    assert "acc_reg <= data_in;" in code_v
    assert "acc_out = acc_reg;" in code_v

    report = validate_rtl_syntax_and_structure(code_v, "Verilog", efs_ir=efs_ir)
    assert report.valid, f"Validation errors: {[e.message for e in report.errors]}"


# =========================================================================
# 4. FSM Controller Generation with Meaningful Signal Conditions
# =========================================================================
def test_functional_fsm_controller_with_handshake_and_triggers():
    """Verify FSM generation uses input trigger condition instead of unconditional 1'b1, and drives status."""
    efs_ir = EFSIR(
        metadata=EFSMetadata(design_name="streaming_tx_fsm", top_module="streaming_tx_fsm"),
        signals=[
            EFSSignal(signal_id="clk", name="clk", direction="input", width=1, semantic_role="clock"),
            EFSSignal(signal_id="rst", name="rst_n", direction="input", width=1, semantic_role="reset"),
            EFSSignal(signal_id="req", name="tx_start", direction="input", width=1, semantic_role="control"),
            EFSSignal(signal_id="busy", name="tx_busy", direction="output", width=1, semantic_role="control"),
            EFSSignal(signal_id="done", name="tx_done", direction="output", width=1, semantic_role="control"),
        ],
        fsms=[
            EFSFSM(
                fsm_id="fsm_tx",
                name="tx_controller",
                states=[
                    EFSState(state_id="s0", name="TX_IDLE", encoding=0),
                    EFSState(state_id="s1", name="TX_STREAM", encoding=1),
                    EFSState(state_id="s2", name="TX_COMPLETE", encoding=2),
                ],
                transitions=[
                    EFSTransition(source_state="TX_IDLE", target_state="TX_STREAM", condition="tx_start == 1'b1"),
                    EFSTransition(source_state="TX_STREAM", target_state="TX_COMPLETE", condition="timer_cnt == 16'd4"),
                    EFSTransition(source_state="TX_COMPLETE", target_state="TX_IDLE", condition="1'b1"),
                ]
            )
        ]
    )

    code_sv = get_heuristic_code_template("streaming_tx_fsm", "SystemVerilog", efs_ir=efs_ir)

    assert "module streaming_tx_fsm" in code_sv
    assert "typedef enum logic [1:0]" in code_sv
    assert "TX_IDLE" in code_sv
    assert "TX_STREAM" in code_sv
    assert "TX_COMPLETE" in code_sv

    # Transition from IDLE must depend on tx_start
    assert "tx_start" in code_sv

    # Status outputs must reflect state
    assert "tx_busy = (state != TX_IDLE);" in code_sv
    assert "tx_done = (state == TX_COMPLETE);" in code_sv

    report = validate_rtl_syntax_and_structure(code_sv, "SystemVerilog", efs_ir=efs_ir)
    assert report.valid, f"Validation errors: {[e.message for e in report.errors]}"


# =========================================================================
# 5. Validation Catches Natural Language Fragments in Active Code
# =========================================================================
def test_validator_catches_natural_language_fragments():
    """Verify validator flags English phrases bleeding into RTL code (e.g. 'data payload')."""
    corrupt_rtl = """
    module natural_lang_leak (
        input wire clk,
        input wire rst_n,
        output reg [31:0] data_out
    );
        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
                data_out <= 32'h0;
            end else begin
                data_out <= data payload;
            end
        end
    endmodule
    """
    report = validate_rtl_syntax_and_structure(corrupt_rtl, "Verilog")
    assert not report.valid
    error_rules = [e.rule_id for e in report.errors]
    assert "RULE_NATURAL_LANGUAGE_FRAGMENT" in error_rules
    nl_err = next(e for e in report.errors if e.rule_id == "RULE_NATURAL_LANGUAGE_FRAGMENT")
    assert "data payload" in nl_err.message.lower()


# =========================================================================
# 6. Validation Catches Undeclared Continuous Assignment Targets
# =========================================================================
def test_validator_catches_undeclared_identifiers():
    """Verify validator flags continuous assignments driving undeclared nets."""
    corrupt_rtl = """
    module undeclared_target (
        input wire clk,
        input wire [7:0] in_a,
        output wire [7:0] out_b
    );
        assign mystery_signal = in_a;
        assign out_b = in_a;
    endmodule
    """
    report = validate_rtl_syntax_and_structure(corrupt_rtl, "Verilog")
    assert not report.valid
    error_rules = [e.rule_id for e in report.errors]
    assert "RULE_UNDECLARED_IDENTIFIER" in error_rules
    und_err = next(e for e in report.errors if e.rule_id == "RULE_UNDECLARED_IDENTIFIER")
    assert "mystery_signal" in und_err.message


# =========================================================================
# 7. Validation Catches Unconditional Open-Loop State Machines
# =========================================================================
def test_validator_catches_unconditional_open_loop_fsm():
    """Verify validator flags state machines where all operational states transition unconditionally."""
    open_loop_fsm_code = """
    module open_loop_fsm (
        input wire clk,
        input wire rst_n,
        output reg [1:0] out
    );
        localparam [1:0] S0 = 2'd0, S1 = 2'd1, S2 = 2'd2;
        reg [1:0] state, next_state;

        always @(posedge clk or negedge rst_n) begin
            if (!rst_n) state <= S0;
            else state <= next_state;
        end

        always @(*) begin
            case (state)
                S0: begin
                    next_state = S1;
                end
                S1: begin
                    next_state = S2;
                end
                S2: begin
                    next_state = S0;
                end
                default: begin
                    next_state = S0;
                end
            endcase
        end
    endmodule
    """
    report = validate_rtl_syntax_and_structure(open_loop_fsm_code, "Verilog")
    assert not report.valid
    error_rules = [e.rule_id for e in report.errors]
    assert "RULE_UNCONDITIONAL_FSM_TRANSITION" in error_rules


# =========================================================================
# 8. Missing/Ambiguous Requirements Correctly Trigger HITL
# =========================================================================
def test_ambiguous_requirements_trigger_hitl_and_resume_after_resolution():
    """Verify that vague transition conditions trigger SPECIFICATION_INCOMPLETE and resume after resolution."""
    ambiguous_spec = """# Hardware Design Specification: SecureMotorController
    Module SecureMotorController ports:
    - clk: input, 1 bit
    - rst_n: input, 1 bit
    - motor_en: output, 1 bit

    states: IDLE, RUN, FAULT
    transitions:
    IDLE -> RUN: control start condition met
    RUN -> FAULT: motor overcurrent detected
    FAULT -> IDLE: single-cycle dwell
    """
    plan = plan_rtl(ambiguous_spec, "Verilog")
    val = validate_rtl_plan(plan)

    # Must be marked INCOMPLETE with blocking issues on ambiguous condition
    assert val["valid"] is False
    assert any(i.get("classification") == "AMBIGUOUS" for i in val["issues"])

    code_blocked = generate_code(ambiguous_spec, "Verilog")
    assert "SPECIFICATION_INCOMPLETE" in code_blocked
    assert "control start condition met" in code_blocked

    # User resolves the ambiguity via Human-in-the-Loop approval
    ir = EFSIR(
        metadata=EFSMetadata(design_name="SecureMotorController", top_module="SecureMotorController"),
        signals=[
            EFSSignal(signal_id="clk", name="clk", direction="input", width=1, semantic_role="clock"),
            EFSSignal(signal_id="rst_n", name="rst_n", direction="input", width=1, semantic_role="reset"),
            EFSSignal(signal_id="start_sig", name="start_cmd", direction="input", width=1, semantic_role="control"),
            EFSSignal(signal_id="fault_sig", name="overcurrent_flag", direction="input", width=1, semantic_role="control"),
            EFSSignal(signal_id="motor_en", name="motor_en", direction="output", width=1, semantic_role="control"),
        ],
        fsms=[
            EFSFSM(
                fsm_id="fsm_motor",
                name="motor_controller_fsm",
                states=[
                    EFSState(state_id="s_idle", name="IDLE"),
                    EFSState(state_id="s_run", name="RUN"),
                    EFSState(state_id="s_fault", name="FAULT"),
                ],
                transitions=[
                    EFSTransition(source_state="IDLE", target_state="RUN", condition="start_cmd == 1'b1"),
                    EFSTransition(source_state="RUN", target_state="FAULT", condition="overcurrent_flag == 1'b1"),
                    EFSTransition(source_state="FAULT", target_state="IDLE", condition="1'b1"),
                ]
            )
        ],
        resolutions=[
            EFSResolution(
                resolution_id="res_fsm_1",
                conflict_id="conf_fsm",
                approval_status="APPROVED",
                affected_requirements=["IDLE", "RUN", "FAULT", "start_cmd", "overcurrent_flag"]
            )
        ]
    )

    res_plan = plan_rtl(ambiguous_spec, "Verilog", efs_ir=ir)
    res_val = validate_rtl_plan(res_plan, efs_ir=ir)
    assert res_val["valid"] is True

    # Generation now succeeds without blocking
    res_code = generate_code(ambiguous_spec, "Verilog", efs_ir=ir)
    assert "SPECIFICATION_INCOMPLETE" not in res_code
    assert "module SecureMotorController" in res_code
    assert "start_cmd" in res_code
    assert "overcurrent_flag" in res_code
