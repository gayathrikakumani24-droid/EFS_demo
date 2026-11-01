"""
Test Suite for Semantic Preservation and Dependency-Based Relevance Selection.

Verifies:
1. Target: Instruction Decoder -> Opcode definitions, instruction fields, formats, and operations MUST be included.
2. Transaction determining FSM state transition -> Transaction semantics MUST be included.
3. Command controlling datapath operation -> Command and datapath operation MUST be included.
4. Protocol transaction determining handshake -> Handshake semantics MUST be included.
5. Preserves parameters, memory regions, timeouts, ordering constraints, out-of-order behavior, and alternate branches.
6. MINIMUM SUFFICIENT CONTEXT: preserves relevant semantics without bloating context with unrelated modules.
"""

import pytest
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
    EFSTransaction,
    EFSInstruction,
    EFSOpcode,
    EFSInstructionFormat,
    EFSInstructionField,
    EFSParameter,
    EFSMemoryRegion,
    EFSFunctionalBehavior,
    EFSError,
    EFSFlow,
    EFSFlowStep,
    EFSConstraint,
    EFSTimingRule
)
from core.context_compiler import compile_context, SemanticDependencySelector, ContextResolver


@pytest.fixture
def instruction_decoder_ir():
    """Create an EFS IR containing an Instruction Decoder, ALU, and bus interface."""
    decoder = EFSComponent(
        component_id="COMP_DEC",
        name="RISCV_Instruction_Decoder",
        type="instruction_decoder",
        description="32-bit RISC-V instruction decoder and control unit",
        clock_domain="clk",
        reset_domain="rst_n",
        interfaces=["IF_INSTR_IN", "IF_ALU_CTRL"]
    )
    alu = EFSComponent(
        component_id="COMP_ALU",
        name="Arithmetic_Logic_Unit",
        type="alu",
        description="32-bit integer arithmetic logic unit",
        clock_domain="clk",
        reset_domain="rst_n",
        interfaces=["IF_ALU_CTRL"]
    )

    # Signals
    clk = EFSSignal(signal_id="SIG_CLK", name="clk", direction="input", semantic_role="clock", owner="COMP_DEC")
    rst = EFSSignal(signal_id="SIG_RST", name="rst_n", direction="input", semantic_role="reset", owner="COMP_DEC")
    instr_in = EFSSignal(signal_id="SIG_INSTR", name="instr_in", direction="input", width="32", semantic_role="data", owner="COMP_DEC")
    opcode_sig = EFSSignal(signal_id="SIG_OPC", name="opcode", direction="output", width="7", semantic_role="control", owner="COMP_DEC")
    alu_cmd = EFSSignal(signal_id="SIG_ALU_CMD", name="alu_cmd", direction="output", width="4", semantic_role="control", owner="COMP_DEC")

    # Parameters
    p_data_w = EFSParameter(parameter_id="PARAM_01", name="DATA_WIDTH", value="32", owner="COMP_DEC", description="Datapath width")
    p_timeout = EFSParameter(parameter_id="PARAM_02", name="DECODE_TIMEOUT", value="4", owner="COMP_DEC", description="Max decode cycles")

    # Instruction Format
    r_type_fmt = EFSInstructionFormat(
        format_id="FMT_R",
        name="R_TYPE",
        total_width=32,
        fields=[
            EFSInstructionField(name="opcode", width=7, msb=6, lsb=0, description="Opcode field"),
            EFSInstructionField(name="rd", width=5, msb=11, lsb=7, description="Destination register"),
            EFSInstructionField(name="funct3", width=3, msb=14, lsb=12, description="Function 3"),
            EFSInstructionField(name="rs1", width=5, msb=19, lsb=15, description="Source register 1"),
            EFSInstructionField(name="rs2", width=5, msb=24, lsb=20, description="Source register 2"),
            EFSInstructionField(name="funct7", width=7, msb=31, lsb=25, description="Function 7")
        ]
    )

    # Opcodes
    opc_add = EFSOpcode(
        opcode_id="OPC_ADD",
        mnemonic="ADD",
        binary_encoding="0110011",
        operation="rd = rs1 + rs2",
        operands=["rd", "rs1", "rs2"]
    )
    opc_sub = EFSOpcode(
        opcode_id="OPC_SUB",
        mnemonic="SUB",
        binary_encoding="0110011",
        operation="rd = rs1 - rs2",
        operands=["rd", "rs1", "rs2"]
    )

    # Instruction
    instr_add = EFSInstruction(
        instruction_id="INS_ADD",
        name="ADD",
        mnemonic="ADD",
        opcode="0110011",
        format="R_TYPE",
        operands=["rd", "rs1", "rs2"],
        description="Addition without overflow"
    )

    # FSM with alternate branches and conditions
    fsm_dec = EFSFSM(
        fsm_id="FSM_DEC",
        name="Decoder_FSM",
        initial_state="FETCH_ST",
        states=[
            EFSState(state_id="ST_0", name="FETCH_ST"),
            EFSState(state_id="ST_1", name="DECODE_ST"),
            EFSState(state_id="ST_2", name="EXEC_R_ST"),
            EFSState(state_id="ST_3", name="EXEC_I_ST")
        ],
        transitions=[
            EFSTransition(
                source_state="FETCH_ST",
                target_state="DECODE_ST",
                condition="valid_instr == 1'b1",
                action="latch_instruction"
            ),
            # Alternate branches branching from DECODE_ST based on opcode
            EFSTransition(
                source_state="DECODE_ST",
                target_state="EXEC_R_ST",
                condition="opcode == 7'b0110011",
                action="trigger_r_type_decode",
                priority=1
            ),
            EFSTransition(
                source_state="DECODE_ST",
                target_state="EXEC_I_ST",
                condition="opcode == 7'b0010011",
                action="trigger_i_type_decode",
                priority=2
            )
        ]
    )

    # Functional behavior (Command controlling datapath)
    cmd_add = EFSFunctionalBehavior(
        behavior_id="BEH_CMD_ADD",
        name="cmd_add_exec",
        subject="RISCV_Instruction_Decoder",
        trigger="opcode == 7'b0110011 && funct3 == 3'b000",
        action="alu_cmd = 4'b0001",
        expected_result="ALU performs addition"
    )

    return EFSIR(
        metadata=EFSMetadata(design_name="RISCV_Instruction_Decoder", top_module="RISCV_Instruction_Decoder"),
        components=[decoder, alu],
        signals=[clk, rst, instr_in, opcode_sig, alu_cmd],
        parameters=[p_data_w, p_timeout],
        instruction_formats=[r_type_fmt],
        opcodes=[opc_add, opc_sub],
        instructions=[instr_add],
        fsms=[fsm_dec],
        functional_behavior=[cmd_add]
    )


def test_target_instruction_decoder_preserves_opcodes_and_formats(instruction_decoder_ir):
    """Test that targeting an Instruction Decoder preserves opcodes, formats, fields, and operations."""
    compiled = compile_context(
        artifact_type="rtl",
        efs_ir=instruction_decoder_ir,
        target_component="RISCV_Instruction_Decoder"
    )

    # 1. Opcode definitions MUST be included
    assert "opcodes" in compiled
    opcodes = compiled["opcodes"]
    assert len(opcodes) >= 2
    mnemonics = [op["mnemonic"] for op in opcodes]
    assert "ADD" in mnemonics
    assert "SUB" in mnemonics
    add_op = next(op for op in opcodes if op["mnemonic"] == "ADD")
    assert add_op["binary_encoding"] == "0110011"
    assert "rd = rs1 + rs2" in add_op["operation"]

    # 2. Instruction formats and fields MUST be included
    assert "instruction_formats" in compiled
    fmt = compiled["instruction_formats"][0]
    assert fmt["format_name"] == "R_TYPE"
    assert fmt["total_width"] == 32
    fld_names = [f["name"] for f in fmt["fields"]]
    assert "opcode" in fld_names
    assert "rd" in fld_names
    assert "funct3" in fld_names

    # 3. Parameters MUST be included
    assert "parameters" in compiled
    param_names = [p["name"] for p in compiled["parameters"]]
    assert "DATA_WIDTH" in param_names
    assert "DECODE_TIMEOUT" in param_names


def test_command_controlling_datapath_is_preserved(instruction_decoder_ir):
    """Test that a command controlling a datapath operation is preserved."""
    compiled = compile_context(
        artifact_type="rtl",
        efs_ir=instruction_decoder_ir,
        target_component="RISCV_Instruction_Decoder"
    )

    assert "commands" in compiled
    cmd = compiled["commands"][0]
    assert "cmd_add" in cmd["name"]
    assert "alu_cmd = 4'b0001" in cmd["action"]


def test_fsm_alternate_branches_and_actions_preserved(instruction_decoder_ir):
    """Test that FSM alternate branches and transition actions are preserved."""
    compiled = compile_context(
        artifact_type="rtl",
        efs_ir=instruction_decoder_ir,
        target_component="RISCV_Instruction_Decoder"
    )

    fsm = compiled["fsm"]
    assert fsm["has_alternate_branches"] is True
    transitions = fsm["transitions"]
    assert len(transitions) == 3
    
    # Check that action and priority are preserved
    r_branch = next(t for t in transitions if t["to"] == "EXEC_R_ST")
    assert "trigger_r_type_decode" in r_branch.get("action", "")
    assert r_branch.get("priority") == 1
    assert r_branch.get("is_alternate_branch") is True


def test_transaction_determining_fsm_transition_is_preserved():
    """Test that a transaction determining an FSM state transition is included."""
    comp = EFSComponent(
        component_id="COMP_DMA",
        name="DMA_Master",
        type="controller",
        interfaces=["IF_AXI"]
    )
    sig_clk = EFSSignal(signal_id="S_CLK", name="clk", semantic_role="clock", owner="COMP_DMA")
    sig_rst = EFSSignal(signal_id="S_RST", name="rst_n", semantic_role="reset", owner="COMP_DMA")

    # Transaction that determines the state transition
    burst_tx = EFSTransaction(
        transaction_id="TX_BURST_RD",
        name="AXI_BURST_READ",
        type="BURST_READ",
        protocol="AXI4",
        initiator="COMP_DMA",
        ordered_steps=[
            "Assert ARVALID with ARADDR",
            "Wait for ARREADY handshake",
            "Receive RDATA bursts while RVALID is high",
            "Terminate on RLAST"
        ],
        preconditions=["Channel is idle", "Address is 4KB aligned"],
        completion_condition="RLAST && RVALID && RREADY",
        ordering="out-of-order",
        response="RRESP == 2'b00 (OKAY)",
        error_behavior="Assert interrupt on DECERR or SLVERR"
    )

    # FSM where transition condition references AXI_BURST_READ
    fsm = EFSFSM(
        fsm_id="FSM_DMA",
        name="DMA_Engine_FSM",
        initial_state="IDLE",
        states=[EFSState(name="IDLE"), EFSState(name="READ_BURST_ACTIVE")],
        transitions=[
            EFSTransition(
                source_state="IDLE",
                target_state="READ_BURST_ACTIVE",
                condition="start_tx && tx_type == AXI_BURST_READ",
                action="issue_arvalid"
            )
        ]
    )

    efs_ir = EFSIR(
        metadata=EFSMetadata(design_name="DMA_Master", top_module="DMA_Master"),
        components=[comp],
        signals=[sig_clk, sig_rst],
        transactions=[burst_tx],
        fsms=[fsm]
    )

    compiled = compile_context(
        artifact_type="rtl",
        efs_ir=efs_ir,
        target_component="DMA_Master"
    )

    # Transaction MUST be included because FSM transition depends on it
    assert "relevant_transactions" in compiled
    txs = compiled["relevant_transactions"]
    assert len(txs) == 1
    t = txs[0]
    assert t["name"] == "AXI_BURST_READ"
    assert t["ordering"] == "out-of-order"
    assert "Assert ARVALID with ARADDR" in t["ordered_steps"]
    assert "RLAST && RVALID && RREADY" in t["completion_condition"]
    assert "DECERR" in t["error_behavior"]


def test_handshake_and_timeout_constraints_preserved():
    """Test that protocol handshakes, stability rules, and timeouts are preserved."""
    comp = EFSComponent(component_id="COMP_BUS", name="Bus_Controller", type="controller")
    clk = EFSSignal(signal_id="S_CLK", name="clk", semantic_role="clock", owner="COMP_BUS")
    rst = EFSSignal(signal_id="S_RST", name="rst_n", semantic_role="reset", owner="COMP_BUS")
    valid = EFSSignal(signal_id="S_VAL", name="req_valid", direction="output", semantic_role="handshake", owner="COMP_BUS")
    ready = EFSSignal(signal_id="S_RDY", name="req_ready", direction="input", semantic_role="handshake", owner="COMP_BUS")

    handshake_const = EFSConstraint(
        constraint_id="CONST_HS",
        type="handshake",
        source_objects=["req_valid"],
        target_objects=["req_ready"],
        condition="req_valid == 1'b1 && req_ready == 1'b0",
        expected_behavior="req_valid and payload signals must remain stable until req_ready is asserted",
        severity="CRITICAL"
    )

    timeout_rule = EFSTimingRule(
        rule_id="TIME_01",
        type="timeout",
        minimum_delay="1 cycle",
        maximum_delay="16 cycles",
        ordering="strict"
    )

    efs_ir = EFSIR(
        metadata=EFSMetadata(design_name="Bus_Controller", top_module="Bus_Controller"),
        components=[comp],
        signals=[clk, rst, valid, ready],
        constraints=[handshake_const],
        timing_rules=[timeout_rule]
    )

    compiled = compile_context(
        artifact_type="rtl",
        efs_ir=efs_ir,
        target_component="Bus_Controller"
    )

    assert "handshakes_and_constraints" in compiled
    hs = compiled["handshakes_and_constraints"][0]
    assert "remain stable until req_ready" in hs["expected_behavior"]

    assert "ordering_and_timeouts" in compiled
    tm = compiled["ordering_and_timeouts"][0]
    assert tm["max_delay"] == "16 cycles"


def test_verification_contexts_preserve_semantic_constructs(instruction_decoder_ir):
    """Test that TB, UVM, and SVA compilers preserve semantic dependencies (opcodes, instructions, assertions)."""
    # 1. Testbench
    tb_ctx = compile_context("tb", efs_ir=instruction_decoder_ir, target_component="RISCV_Instruction_Decoder")
    assert "opcodes" in tb_ctx
    assert "parameters" in tb_ctx
    assert len(tb_ctx["opcodes"]) >= 2

    # 2. UVM
    uvm_ctx = compile_context("uvm", efs_ir=instruction_decoder_ir, target_component="RISCV_Instruction_Decoder")
    assert "opcodes" in uvm_ctx
    assert "instruction_formats" in uvm_ctx
    assert uvm_ctx["instruction_formats"][0]["format_name"] == "R_TYPE"

    # 3. SVA
    sva_ctx = compile_context("sva", efs_ir=instruction_decoder_ir, target_component="RISCV_Instruction_Decoder")
    assert "opcodes" in sva_ctx
    props = sva_ctx["properties_to_assert"]
    assert any("Instruction opcode valid decoding check" in p for p in props)
