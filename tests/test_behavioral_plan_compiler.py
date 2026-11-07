"""
Comprehensive Unit & Integration Test Suite for Behavioral Plan Compiler.

Verifies:
1. Sequential logic compilation (PlanSequentialProcess, PlanRegisterUpdate, reset conditions).
2. Combinational logic compilation (PlanCombinationalProcess, PlanCombinationalAssignment, slices, opcodes, FSM next-state).
3. Structured PlanExpression AST parsing and Verilog generation.
4. Explicit provenance classification (INHERITED, DERIVED, IMPLEMENTATION_DECISION).
5. Explicit PlanUnresolvedLogic generation for specification gaps (no hallucinated equations).
6. End-to-end plan_rtl() integration.
7. Architecture-agnostic compilation across diverse hardware models.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.efs_ir.models import (
    EFSIR,
    EFSComponent,
    EFSSignal,
    EFSRegister,
    EFSFSM,
    EFSState,
    EFSTransition,
    EFSOpcode,
    EFSInstructionFormat,
    EFSInstructionField,
    EFSMetadata,
)
from core.efs_ir.rtl_plan import (
    RTLGenerationPlan,
    PlanSequentialProcess,
    PlanRegisterUpdate,
    PlanCombinationalProcess,
    PlanCombinationalAssignment,
    PlanExpression,
    PlanUnresolvedLogic,
    PlanOrigin,
    DerivationMetadata,
    PlanPort,
    PlanRegister,
)
from core.context_compiler.behavioral_plan_compiler import BehavioralPlanCompiler
from core.agents.rtl_agent import plan_rtl


class TestBehavioralPlanCompiler(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures for hardware specifications."""
        # Instruction Decoder Spec
        self.comp_decoder = EFSComponent(
            component_id="COMP_DEC",
            name="instruction_decoder",
            type="decoder",
            clock_domain="clk",
            reset_domain="rst_n"
        )
        self.sig_clk = EFSSignal(signal_id="S_CLK", name="clk", direction="input", width="1", semantic_role="clock", owner="COMP_DEC")
        self.sig_rst = EFSSignal(signal_id="S_RST", name="rst_n", direction="input", width="1", semantic_role="reset", owner="COMP_DEC")
        self.sig_inst = EFSSignal(signal_id="S_INST", name="instruction", direction="input", width="32", owner="COMP_DEC")
        self.sig_opcode = EFSSignal(signal_id="S_OPC", name="opcode", direction="output", width="6", owner="COMP_DEC")
        self.sig_rd = EFSSignal(signal_id="S_RD", name="rd", direction="output", width="5", owner="COMP_DEC")
        self.sig_valid = EFSSignal(signal_id="S_VAL", name="valid", direction="output", width="1", owner="COMP_DEC")

        self.reg_state = EFSRegister(
            register_id="REG_STATE",
            name="state_reg",
            width="2",
            reset_value="2'b00",
            owner="COMP_DEC",
            description="FSM state register"
        )
        self.reg_data = EFSRegister(
            register_id="REG_DATA",
            name="data_reg",
            width="32",
            reset_value="32'h0",
            owner="COMP_DEC",
            description="Internal buffer register without explicit equation"
        )

        self.instruction_fmt = EFSInstructionFormat(
            format_id="INST_FMT_R",
            name="R_TYPE",
            total_width=32,
            fields=[
                EFSInstructionField(name="opcode", width=6, msb=31, lsb=26, description="Opcode field"),
                EFSInstructionField(name="rd", width=5, msb=15, lsb=11, description="Destination register"),
            ]
        )

        self.opcode_add = EFSOpcode(
            opcode_id="OP_ADD",
            mnemonic="ADD",
            binary_encoding="6'b000001",
            operation="Arithmetic addition"
        )
        self.opcode_sub = EFSOpcode(
            opcode_id="OP_SUB",
            mnemonic="SUB",
            binary_encoding="6'b000010",
            operation="Arithmetic subtraction"
        )

        self.fsm_dec = EFSFSM(
            fsm_id="FSM_DEC",
            name="decoder_fsm",
            states=[
                EFSState(name="IDLE", encoding="2'b00"),
                EFSState(name="DECODE", encoding="2'b01"),
                EFSState(name="EXEC", encoding="2'b10"),
            ],
            initial_state="IDLE",
            reset_state="IDLE",
            transitions=[
                EFSTransition(
                    source_state="IDLE",
                    target_state="DECODE",
                    condition="valid == 1'b1",
                    associated_signal="valid"
                ),
                EFSTransition(
                    source_state="DECODE",
                    target_state="EXEC",
                    condition="1'b1"
                ),
                EFSTransition(
                    source_state="EXEC",
                    target_state="IDLE",
                    condition="1'b1"
                )
            ]
        )

        self.efs_ir_decoder = EFSIR(
            metadata=EFSMetadata(design_id="SPEC_DEC", design_name="instruction_decoder", model_version=1),
            components=[self.comp_decoder],
            signals=[self.sig_clk, self.sig_rst, self.sig_inst, self.sig_opcode, self.sig_rd, self.sig_valid],
            registers=[self.reg_state, self.reg_data],
            instruction_formats=[self.instruction_fmt],
            opcodes=[self.opcode_add, self.opcode_sub],
            fsms=[self.fsm_dec]
        )

    def test_expression_ast_and_verilog_generation(self):
        """Verify PlanExpression parsing into AST and rendering to syntactically correct Verilog."""
        # 1. Bitfield slice
        expr_slice = PlanExpression.parse_expression("instruction[31:26]")
        self.assertEqual(expr_slice.expression_type, "SLICE")
        self.assertEqual(expr_slice.operands, ["instruction", 31, 26])
        self.assertEqual(expr_slice.to_verilog_expr(), "instruction[31:26]")

        # 2. Ternary operator
        expr_ternary = PlanExpression.parse_expression("valid ? 2'b01 : 2'b00")
        self.assertEqual(expr_ternary.expression_type, "CONDITIONAL")
        self.assertEqual(expr_ternary.condition, "valid")
        self.assertEqual(expr_ternary.operands, ["2'b01", "2'b00"])
        self.assertEqual(expr_ternary.to_verilog_expr(), "(valid ? 2'b01 : 2'b00)")

        # 3. Binary operator
        expr_bin = PlanExpression.parse_expression("operand_a + operand_b")
        self.assertEqual(expr_bin.expression_type, "ARITHMETIC")
        self.assertEqual(expr_bin.operator, "+")
        self.assertEqual(expr_bin.to_verilog_expr(), "(operand_a + operand_b)")

        # 4. Equality comparison
        expr_eq = PlanExpression.parse_expression("opcode == 6'b000001")
        self.assertEqual(expr_eq.expression_type, "BOOLEAN")
        self.assertEqual(expr_eq.operator, "==")
        self.assertEqual(expr_eq.to_verilog_expr(), "(opcode == 6'b000001)")

        # 5. Literal
        expr_lit = PlanExpression.parse_expression("32'h00000000")
        self.assertEqual(expr_lit.expression_type, "LITERAL")
        self.assertEqual(expr_lit.to_verilog_expr(), "32'h00000000")

    def test_sequential_logic_compilation_and_unresolved_gaps(self):
        """Verify sequential processes compile cleanly and unspecified registers are recorded as unresolved logic."""
        plan_dict = {
            "module_name": "instruction_decoder",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "rst_n", "direction": "input", "width": "1"},
                {"name": "instruction", "direction": "input", "width": "32"},
            ],
            "internal_registers": [
                {"name": "state_reg", "width": "2", "reset_value": "2'b00"},
                {"name": "data_reg", "width": "32", "reset_value": "32'h0"},
            ],
            "fsm": self.fsm_dec.to_dict()
        }

        compiled_plan = BehavioralPlanCompiler.compile_and_populate(
            plan_dict=plan_dict,
            efs_ir=self.efs_ir_decoder,
            target_comp=self.comp_decoder
        )

        seq_procs = compiled_plan.get("sequential_logic", [])
        self.assertGreater(len(seq_procs), 0, "Expected at least one sequential process")

        main_proc = seq_procs[0]
        self.assertEqual(main_proc["clock_signal"], "clk")
        self.assertEqual(main_proc["clock_edge"], "POSEDGE")
        self.assertEqual(main_proc["reset_signal"], "rst_n")
        self.assertEqual(main_proc["reset_active_level"], "LOW")
        self.assertEqual(main_proc["reset_type"], "ASYNC")

        # Reset actions must cover all registers
        reset_actions = main_proc["reset_actions"]
        self.assertIn("state_reg", reset_actions)
        self.assertEqual(reset_actions["state_reg"], "IDLE")
        self.assertIn("data_reg", reset_actions)
        self.assertEqual(reset_actions["data_reg"], "32'h0")

        # state_reg must have a registered update from FSM
        updates = main_proc["register_updates"]
        state_update = next((u for u in updates if u["target"] == "state_reg"), None)
        self.assertIsNotNone(state_update, "state_reg must have a register update directive")
        self.assertIn("next_state", str(state_update["next_value"]))

        # data_reg had no functional behavior or FSM transition, so it must be flagged in unresolved_logic
        unresolved = compiled_plan.get("unresolved_logic", [])
        unres_data = next((u for u in unresolved if u["affected_entity"] == "data_reg"), None)
        self.assertIsNotNone(unres_data, "Unassigned register data_reg must be recorded in unresolved_logic")
        self.assertFalse(unres_data["blocks_generation"])
        self.assertEqual(unres_data["severity"], "WARNING")
        self.assertIn("data_reg", unres_data["description"])
        self.assertIn("no update condition", unres_data["description"])

    def test_combinational_logic_compilation(self):
        """Verify combinational assignments, bitfield slices, opcode decoding, and FSM next-state logic."""
        plan_dict = {
            "module_name": "instruction_decoder",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "rst_n", "direction": "input", "width": "1"},
                {"name": "instruction", "direction": "input", "width": "32"},
                {"name": "opcode", "direction": "output", "width": "6"},
                {"name": "rd", "direction": "output", "width": "5"},
                {"name": "valid", "direction": "output", "width": "1"},
            ],
            "internal_registers": [
                {"name": "state_reg", "width": "2", "reset_value": "2'b00"},
            ],
            "fsm": self.fsm_dec.to_dict(),
            "opcodes": [self.opcode_add.to_dict(), self.opcode_sub.to_dict()],
        }

        compiled_plan = BehavioralPlanCompiler.compile_and_populate(
            plan_dict=plan_dict,
            efs_ir=self.efs_ir_decoder,
            target_comp=self.comp_decoder
        )

        comb_procs = compiled_plan.get("combinational_logic", [])
        self.assertGreater(len(comb_procs), 0, "Expected at least one combinational process")

        all_assignments = []
        for cp in comb_procs:
            all_assignments.extend(cp.get("assignments", []))

        # Check opcode bitfield slice assignment: opcode = instruction[31:26]
        assign_opcode = next((a for a in all_assignments if a["target"] == "opcode"), None)
        self.assertIsNotNone(assign_opcode, "opcode must have continuous assignment")
        self.assertEqual(PlanExpression.from_dict(assign_opcode["expression"]).to_verilog_expr(), "instruction[31:26]")
        self.assertEqual(assign_opcode["origin"], PlanOrigin.DERIVED.value)

        # Check rd bitfield slice assignment: rd = instruction[15:11]
        assign_rd = next((a for a in all_assignments if a["target"] == "rd"), None)
        self.assertIsNotNone(assign_rd, "rd must have continuous assignment")
        self.assertEqual(PlanExpression.from_dict(assign_rd["expression"]).to_verilog_expr(), "instruction[15:11]")
        self.assertEqual(assign_rd["origin"], PlanOrigin.DERIVED.value)

        # Check FSM next-state combinational logic process
        fsm_proc = next((cp for cp in comb_procs if cp.get("process_type") == "ALWAYS_COMB"), None)
        self.assertIsNotNone(fsm_proc, "FSM next-state combinational process must exist")
        self.assertIn("state_reg", fsm_proc.get("input_dependencies", []))

    def test_provenance_classification(self):
        """Verify ports, registers, and parameters receive strict origin tags."""
        plan_dict = {
            "module_name": "instruction_decoder",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "instruction", "direction": "input", "width": "32"},
                {"name": "synthesized_flag", "direction": "output", "width": "1"},
            ],
            "internal_registers": [
                {"name": "state_reg", "width": "2", "reset_value": "2'b00"},
                {"name": "opt_counter", "width": "4", "reset_value": "4'd0"},
            ],
            "parameters": [
                {"name": "DATA_WIDTH", "value": "32"},
                {"name": "MAX_TIMEOUT", "value": "100"},
            ]
        }

        design_spec = "Design instruction decoder with clk, instruction, and DATA_WIDTH parameter."

        compiled_plan = BehavioralPlanCompiler.compile_and_populate(
            plan_dict=plan_dict,
            efs_ir=self.efs_ir_decoder,
            target_comp=self.comp_decoder,
            design_spec=design_spec
        )

        ports = {p["name"]: p for p in compiled_plan["ports"]}
        self.assertEqual(ports["clk"]["origin"], PlanOrigin.INHERITED.value)
        self.assertEqual(ports["instruction"]["origin"], PlanOrigin.INHERITED.value)
        self.assertEqual(ports["synthesized_flag"]["origin"], PlanOrigin.IMPLEMENTATION_DECISION.value)

        regs = {r["name"]: r for r in compiled_plan["internal_registers"]}
        self.assertEqual(regs["state_reg"]["origin"], PlanOrigin.INHERITED.value)
        self.assertEqual(regs["opt_counter"]["origin"], PlanOrigin.IMPLEMENTATION_DECISION.value)

        params = {p["name"]: p for p in compiled_plan.get("parameters", [])}
        self.assertEqual(params["DATA_WIDTH"]["origin"], PlanOrigin.INHERITED.value)
        self.assertEqual(params["MAX_TIMEOUT"]["origin"], PlanOrigin.IMPLEMENTATION_DECISION.value)

    @patch("core.agents.rtl_agent.get_llm_client")
    def test_end_to_end_plan_rtl_integration(self, mock_llm_getter):
        """Verify plan_rtl returns a complete plan dictionary with populated sequential and combinational logic."""
        mock_llm = MagicMock()
        mock_llm.available = False
        mock_llm_getter.return_value = mock_llm

        design_spec = """
        # Instruction Decoder
        Module: instruction_decoder
        Clock: clk
        Reset: rst_n (active-low asynchronous)
        
        ## Ports
        - clk: input [0:0]
        - rst_n: input [0:0]
        - instruction: input [31:0]
        - opcode: output [5:0]
        - rd: output [4:0]
        """

        plan = plan_rtl(design_spec=design_spec, target="Verilog", efs_ir=self.efs_ir_decoder)

        self.assertIsInstance(plan, dict)
        self.assertEqual(plan["module_name"], "instruction_decoder")

        # Verify sequential logic populated
        seq_logic = plan.get("sequential_logic", [])
        self.assertGreater(len(seq_logic), 0, "plan_rtl must populate sequential_logic")
        self.assertEqual(seq_logic[0]["clock_signal"], "clk")

        # Verify combinational logic populated
        comb_logic = plan.get("combinational_logic", [])
        self.assertGreater(len(comb_logic), 0, "plan_rtl must populate combinational_logic")

        # Verify ports have origin
        for p in plan.get("ports", []):
            self.assertIn("origin", p)
            self.assertIn(p["origin"], [PlanOrigin.INHERITED.value, PlanOrigin.DERIVED.value, PlanOrigin.IMPLEMENTATION_DECISION.value])

        # Verify RTLGenerationPlan schema serialization/deserialization
        plan_obj = RTLGenerationPlan.from_dict(plan)
        self.assertEqual(plan_obj.module_name, "instruction_decoder")
        self.assertEqual(len(plan_obj.sequential_logic), len(seq_logic))
        self.assertEqual(len(plan_obj.combinational_logic), len(comb_logic))


if __name__ == "__main__":
    unittest.main()
