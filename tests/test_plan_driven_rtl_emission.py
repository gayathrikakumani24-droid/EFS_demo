"""
Unit tests for Phase 3: Plan-Driven Hybrid RTL Emission.

Verifies that the final emitted Verilog/SystemVerilog RTL is faithfully and
deterministically generated from the structured RTLGenerationPlan:
- Sequential logic (clocks, resets, register updates)
- Combinational logic (continuous assignments, sliced bitfields, data selections)
- Opcode decoding
- FSM state registers and next-state transition decoders
- Protocol contracts and formal verification assertions
- Unresolved specification logic boundaries
- Strict module header and port direction locking
- Verification of syntax and elaboration via real Icarus Verilog compiler (iverilog)
"""

import os
import sys
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.agents.constrained_generator import (
    GenerationMode,
    assemble_rtl_module,
    emit_deterministic_behavioral_logic,
    emit_deterministic_declarations,
    emit_deterministic_module_header,
    generate_hybrid_rtl,
)
from core.efs_ir.models import EFSComponent, EFSIR, EFSMetadata, EFSSignal
from core.efs_ir.rtl_plan import (
    DerivationMetadata,
    PlanCombinationalAssignment,
    PlanCombinationalProcess,
    PlanExpression,
    PlanFSM,
    PlanFSMTransition,
    PlanOrigin,
    PlanProtocolContract,
    PlanProtocolRule,
    PlanRegisterUpdate,
    PlanSequentialProcess,
    PlanUnresolvedLogic,
    RTLGenerationPlan,
)


def _compile_with_iverilog(verilog_code: str, is_sv: bool = False) -> subprocess.CompletedProcess:
    """Helper to compile verilog code with iverilog."""
    iverilog_bin = shutil.which("iverilog")
    if not iverilog_bin:
        raise unittest.SkipTest("iverilog binary not found in PATH")

    with tempfile.TemporaryDirectory() as tmpdir:
        src_file = os.path.join(tmpdir, "test_module.v")
        out_file = os.path.join(tmpdir, "test_module.out")
        with open(src_file, "w", encoding="utf-8") as f:
            f.write(verilog_code)

        std_flag = "-g2012" if is_sv else "-g2005"
        cmd = [iverilog_bin, std_flag, "-o", out_file, src_file]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return res


class TestPlanDrivenRTLEmission(unittest.TestCase):
    """Test suite validating plan-driven RTL generation across all hardware archetypes."""

    def test_1_synchronous_counter_emission(self):
        """Test 1: Synchronous counter with async active-low reset and structured sequential logic."""
        plan_dict = {
            "module_name": "sync_counter",
            "target_language": "Verilog",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "rst_n", "direction": "input", "width": "1"},
                {"name": "count", "direction": "output", "width": "8"},
            ],
            "internal_registers": [
                {"name": "count", "width": "8", "reset_val": "8'd0"}
            ],
            "reset_requirements": {
                "clock_name": "clk",
                "reset_name": "rst_n",
                "polarity": "ACTIVE_LOW",
                "is_asynchronous": True,
            },
            "sequential_logic": [
                PlanSequentialProcess(
                    process_id="SEQ_COUNT",
                    clock_signal="clk",
                    clock_edge="POSEDGE",
                    reset_signal="rst_n",
                    reset_type="ASYNC",
                    reset_active_level="LOW",
                    reset_actions={"count": "8'd0"},
                    register_updates=[
                        PlanRegisterUpdate(
                            target="count",
                            condition="1'b1",
                            next_value=PlanExpression(
                                expression_type="ARITHMETIC",
                                operator="+",
                                operands=["count", "1'b1"]
                            )
                        )
                    ]
                )
            ]
        }

        code, report = generate_hybrid_rtl(
            design_spec="8-bit synchronous counter",
            target="Verilog",
            rtl_plan=plan_dict,
            mode=GenerationMode.DETERMINISTIC
        )

        # Assert deterministic module header declared 'count' as output reg (since driven in always block)
        self.assertIn("output reg  [7:0] count", code)
        # Assert sensitivity list handles async active-low reset
        self.assertIn("always @ (posedge clk or negedge rst_n) begin", code)
        self.assertIn("if (!rst_n) begin", code)
        self.assertIn("count <= 8'd0;", code)
        self.assertIn("count <= (count + 1'b1);", code)

        # Real Icarus Verilog compilation check
        res = _compile_with_iverilog(code)
        self.assertEqual(res.returncode, 0, f"iverilog failed: {res.stderr}\nCode:\n{code}")

    def test_2_up_down_counter_with_enable(self):
        """Test 2: Up/down counter with synchronous reset, enable, and conditional ternary next_value."""
        plan_dict = {
            "module_name": "up_down_counter",
            "target_language": "Verilog",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "reset", "direction": "input", "width": "1"},
                {"name": "en", "direction": "input", "width": "1"},
                {"name": "up_down", "direction": "input", "width": "1"},
                {"name": "q", "direction": "output", "width": "8"},
            ],
            "internal_registers": [
                {"name": "q", "width": "8", "reset_val": "8'd0"}
            ],
            "reset_requirements": {
                "clock_name": "clk",
                "reset_name": "reset",
                "polarity": "ACTIVE_HIGH",
                "is_asynchronous": False,
            },
            "sequential_logic": [
                PlanSequentialProcess(
                    process_id="SEQ_Q",
                    clock_signal="clk",
                    clock_edge="POSEDGE",
                    reset_signal="reset",
                    reset_type="SYNC",
                    reset_active_level="HIGH",
                    reset_actions={"q": "8'd0"},
                    register_updates=[
                        PlanRegisterUpdate(
                            target="q",
                            enable="en",
                            condition="1'b1",
                            next_value=PlanExpression(
                                expression_type="CONDITIONAL",
                                operator="?",
                                condition="up_down",
                                operands=[
                                    PlanExpression(expression_type="ARITHMETIC", operator="+", operands=["q", "8'd1"]),
                                    PlanExpression(expression_type="ARITHMETIC", operator="-", operands=["q", "8'd1"])
                                ]
                            )
                        )
                    ]
                )
            ]
        }

        code, _ = generate_hybrid_rtl(
            design_spec="Up/down counter with enable",
            target="Verilog",
            rtl_plan=plan_dict,
            mode=GenerationMode.DETERMINISTIC
        )

        self.assertIn("always @ (posedge clk) begin", code)
        self.assertIn("if (reset) begin", code)
        self.assertIn("q <= 8'd0;", code)
        self.assertIn("if (en) begin", code)
        self.assertIn("q <= (up_down ? (q + 8'd1) : (q - 8'd1));", code)

        res = _compile_with_iverilog(code)
        self.assertEqual(res.returncode, 0, f"iverilog failed: {res.stderr}\nCode:\n{code}")

    def test_3_instruction_decoder_combinational_slices(self):
        """Test 3: Instruction decoder with sliced bitfield continuous assignments and opcode decode."""
        plan_dict = {
            "module_name": "instruction_decoder",
            "target_language": "Verilog",
            "ports": [
                {"name": "instruction", "direction": "input", "width": "32"},
                {"name": "opcode", "direction": "output", "width": "7"},
                {"name": "dest_reg", "direction": "output", "width": "5"},
                {"name": "src1_reg", "direction": "output", "width": "5"},
                {"name": "src2_reg", "direction": "output", "width": "5"},
                {"name": "valid_instruction", "direction": "output", "width": "1"},
            ],
            "combinational_logic": [
                PlanCombinationalProcess(
                    process_id="COMB_FIELDS",
                    process_type="ALWAYS_COMB",
                    assignments=[
                        PlanCombinationalAssignment(
                            target="opcode",
                            expression=PlanExpression(
                                expression_type="SLICE",
                                operator="SLICE",
                                operands=["instruction", 6, 0]
                            )
                        ),
                        PlanCombinationalAssignment(
                            target="dest_reg",
                            expression=PlanExpression(
                                expression_type="SLICE",
                                operator="SLICE",
                                operands=["instruction", 11, 7]
                            )
                        ),
                        PlanCombinationalAssignment(
                            target="src1_reg",
                            expression=PlanExpression(
                                expression_type="SLICE",
                                operator="SLICE",
                                operands=["instruction", 19, 15]
                            )
                        ),
                        PlanCombinationalAssignment(
                            target="src2_reg",
                            expression=PlanExpression(
                                expression_type="SLICE",
                                operator="SLICE",
                                operands=["instruction", 24, 20]
                            )
                        ),
                    ]
                )
            ],
            "opcodes": [
                {"name": "OP_ADD", "mnemonic": "ADD", "encoding": "7'b0110011"},
                {"name": "OP_SUB", "mnemonic": "SUB", "encoding": "7'b0110011"},
                {"name": "OP_LOAD", "mnemonic": "LOAD", "encoding": "7'b0000011"},
            ]
        }

        code, _ = generate_hybrid_rtl(
            design_spec="Instruction decoder",
            target="Verilog",
            rtl_plan=plan_dict,
            mode=GenerationMode.DETERMINISTIC
        )

        # In Verilog-2001, outputs driven by continuous assign must be output wire
        self.assertIn("output wire [4:0] dest_reg", code)
        self.assertIn("output wire [6:0] opcode", code)
        self.assertIn("assign opcode = instruction[6:0];", code)
        self.assertIn("assign dest_reg = instruction[11:7];", code)
        self.assertIn("assign src1_reg = instruction[19:15];", code)
        self.assertIn("assign src2_reg = instruction[24:20];", code)
        self.assertIn("case (opcode)", code)
        self.assertIn("valid_instruction = 1'b1;", code)

        res = _compile_with_iverilog(code)
        self.assertEqual(res.returncode, 0, f"iverilog failed: {res.stderr}\nCode:\n{code}")

    def test_4_fsm_controller(self):
        """Test 4: FSM-based controller with 3 states, transition conditions, and Moore status flags."""
        plan_dict = {
            "module_name": "traffic_fsm",
            "target_language": "Verilog",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "rst_n", "direction": "input", "width": "1"},
                {"name": "sensor", "direction": "input", "width": "1"},
                {"name": "timer_done", "direction": "input", "width": "1"},
                {"name": "green", "direction": "output", "width": "1"},
                {"name": "yellow", "direction": "output", "width": "1"},
                {"name": "red", "direction": "output", "width": "1"},
            ],
            "reset_requirements": {
                "clock_name": "clk",
                "reset_name": "rst_n",
                "polarity": "ACTIVE_LOW",
                "is_asynchronous": True,
            },
            "fsm": {
                "states": [
                    {"name": "S_RED"},
                    {"name": "S_GREEN"},
                    {"name": "S_YELLOW"},
                ],
                "initial_state": "S_RED",
                "transitions": [
                    {"from": "S_RED", "to": "S_GREEN", "condition": "sensor"},
                    {"from": "S_GREEN", "to": "S_YELLOW", "condition": "timer_done"},
                    {"from": "S_YELLOW", "to": "S_RED", "condition": "timer_done"},
                ]
            },
            "combinational_logic": [
                PlanCombinationalProcess(
                    process_id="COMB_LIGHTS",
                    process_type="ALWAYS_COMB",
                    assignments=[
                        PlanCombinationalAssignment(
                            target="red",
                            expression=PlanExpression(expression_type="BOOLEAN", operator="==", operands=["current_state", "S_RED"])
                        ),
                        PlanCombinationalAssignment(
                            target="green",
                            expression=PlanExpression(expression_type="BOOLEAN", operator="==", operands=["current_state", "S_GREEN"])
                        ),
                        PlanCombinationalAssignment(
                            target="yellow",
                            expression=PlanExpression(expression_type="BOOLEAN", operator="==", operands=["current_state", "S_YELLOW"])
                        ),
                    ]
                )
            ]
        }

        code, _ = generate_hybrid_rtl(
            design_spec="Traffic light controller FSM",
            target="Verilog",
            rtl_plan=plan_dict,
            mode=GenerationMode.DETERMINISTIC
        )

        self.assertIn("localparam [1:0] S_RED = 2'd0;", code)
        self.assertIn("localparam [1:0] S_GREEN = 2'd1;", code)
        self.assertIn("localparam [1:0] S_YELLOW = 2'd2;", code)
        self.assertIn("reg [1:0] current_state, next_state;", code)
        self.assertIn("assign red = (current_state == S_RED);", code)
        self.assertIn("assign green = (current_state == S_GREEN);", code)
        self.assertIn("assign yellow = (current_state == S_YELLOW);", code)
        self.assertIn("case (current_state)", code)
        self.assertIn("if (sensor) next_state = S_GREEN;", code)

        res = _compile_with_iverilog(code)
        self.assertEqual(res.returncode, 0, f"iverilog failed: {res.stderr}\nCode:\n{code}")

    def test_5_datapath_alu_multiplexer(self):
        """Test 5: Datapath ALU with multiplexed operations inside combinational block."""
        plan_dict = {
            "module_name": "alu_unit",
            "target_language": "Verilog",
            "ports": [
                {"name": "alu_op", "direction": "input", "width": "2"},
                {"name": "a", "direction": "input", "width": "8"},
                {"name": "b", "direction": "input", "width": "8"},
                {"name": "result", "direction": "output", "width": "8"},
            ],
            "combinational_logic": [
                PlanCombinationalProcess(
                    process_id="COMB_ALU",
                    process_type="ALWAYS_COMB",
                    data_selections=[
                        {
                            "destination": "result",
                            "selector_signal": "alu_op",
                            "cases": {
                                "2'b00": "(a + b)",
                                "2'b01": "(a - b)",
                                "2'b10": "(a & b)",
                                "2'b11": "(a | b)",
                            },
                            "default_assignment": "8'd0"
                        }
                    ]
                )
            ]
        }

        code, _ = generate_hybrid_rtl(
            design_spec="ALU datapath unit",
            target="Verilog",
            rtl_plan=plan_dict,
            mode=GenerationMode.DETERMINISTIC
        )

        self.assertIn("always @(*) begin", code)
        self.assertIn("case (alu_op)", code)
        self.assertIn("2'b00: result = (a + b);", code)
        self.assertIn("default: result = 8'd0;", code)

        res = _compile_with_iverilog(code)
        self.assertEqual(res.returncode, 0, f"iverilog failed: {res.stderr}\nCode:\n{code}")

    def test_6_parameterized_hardware(self):
        """Test 6: Parameterized hardware preserves #(parameter WIDTH = 16) and locks module interface."""
        plan_dict = {
            "module_name": "param_reg",
            "target_language": "Verilog",
            "parameters": [
                {"name": "DATA_WIDTH", "default": "16", "description": "Bus data width"}
            ],
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "rst_n", "direction": "input", "width": "1"},
                {"name": "d", "direction": "input", "width": "DATA_WIDTH"},
                {"name": "q", "direction": "output", "width": "DATA_WIDTH"},
            ],
            "internal_registers": [
                {"name": "q", "width": "DATA_WIDTH", "reset_val": "0"}
            ],
            "sequential_logic": [
                PlanSequentialProcess(
                    process_id="SEQ_PARAM",
                    clock_signal="clk",
                    clock_edge="POSEDGE",
                    reset_signal="rst_n",
                    reset_type="ASYNC",
                    reset_active_level="LOW",
                    reset_actions={"q": "{DATA_WIDTH{1'b0}}"},
                    register_updates=[
                        PlanRegisterUpdate(
                            target="q",
                            next_value=PlanExpression(expression_type="IDENTIFIER", operands=["d"])
                        )
                    ]
                )
            ]
        }

        code, _ = generate_hybrid_rtl(
            design_spec="Parameterized register",
            target="Verilog",
            rtl_plan=plan_dict,
            mode=GenerationMode.DETERMINISTIC
        )

        self.assertIn("parameter DATA_WIDTH = 16", code)
        self.assertIn("[DATA_WIDTH-1:0] d", code)
        self.assertIn("[DATA_WIDTH-1:0] q", code)
        self.assertIn("q <= d;", code)

        res = _compile_with_iverilog(code)
        self.assertEqual(res.returncode, 0, f"iverilog failed: {res.stderr}\nCode:\n{code}")

    def test_7_protocol_independent_component(self):
        """Test 7: Protocol-independent component generates clean RTL without spurious protocol artifacts."""
        plan_dict = {
            "module_name": "math_adder",
            "target_language": "Verilog",
            "ports": [
                {"name": "in_a", "direction": "input", "width": "4"},
                {"name": "in_b", "direction": "input", "width": "4"},
                {"name": "sum", "direction": "output", "width": "5"},
            ],
            "combinational_logic": [
                PlanCombinationalProcess(
                    process_id="COMB_ADD",
                    assignments=[
                        PlanCombinationalAssignment(
                            target="sum",
                            expression=PlanExpression(
                                expression_type="ARITHMETIC",
                                operator="+",
                                operands=["in_a", "in_b"]
                            )
                        )
                    ]
                )
            ]
        }

        code, _ = generate_hybrid_rtl(
            design_spec="4-bit adder with 5-bit output",
            target="Verilog",
            rtl_plan=plan_dict,
            mode=GenerationMode.DETERMINISTIC
        )

        self.assertIn("assign sum = (in_a + in_b);", code)
        self.assertNotIn("// [PROTOCOL_CONTRACT]", code)

        res = _compile_with_iverilog(code)
        self.assertEqual(res.returncode, 0, f"iverilog failed: {res.stderr}\nCode:\n{code}")

    def test_8_streaming_handshake_protocol_contract(self):
        """Test 8: Streaming handshake protocol contract is preserved and emitted in RTL."""
        contract = PlanProtocolContract(
            protocol_name="STREAM",
            protocol_version="1.0",
            rules=[
                PlanProtocolRule(
                    rule_id="RULE_STREAM_HS",
                    rule_type="HANDSHAKE",
                    description="Transfer occurs only when valid and ready are asserted concurrently.",
                    applicable_interfaces=["data_in"],
                    applicable_signals=["valid", "ready"],
                    handshake_condition="valid && ready",
                    acceptance_condition="ready == 1'b1",
                    is_synthesizable_behavior=True
                )
            ]
        )

        plan_dict = {
            "module_name": "stream_fifo",
            "target_language": "Verilog",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "rst_n", "direction": "input", "width": "1"},
                {"name": "valid", "direction": "input", "width": "1"},
                {"name": "ready", "direction": "output", "width": "1"},
                {"name": "data_in", "direction": "input", "width": "8"},
                {"name": "data_out", "direction": "output", "width": "8"},
            ],
            "protocol_contract": contract.to_dict(),
            "combinational_logic": [
                PlanCombinationalProcess(
                    process_id="COMB_READY",
                    assignments=[
                        PlanCombinationalAssignment(
                            target="ready",
                            expression=PlanExpression(expression_type="LITERAL", operands=["1'b1"])
                        ),
                        PlanCombinationalAssignment(
                            target="data_out",
                            expression=PlanExpression(expression_type="IDENTIFIER", operands=["data_in"])
                        )
                    ]
                )
            ]
        }

        code, _ = generate_hybrid_rtl(
            design_spec="Streaming FIFO receiver",
            target="Verilog",
            rtl_plan=plan_dict,
            mode=GenerationMode.DETERMINISTIC
        )

        self.assertIn("// [PROTOCOL_CONTRACT] Protocol: STREAM 1.0", code)
        self.assertIn("// Rule RULE_STREAM_HS [HANDSHAKE]", code)
        self.assertIn("//   Handshake: valid && ready", code)
        self.assertIn("assign ready = 1'b1;", code)
        self.assertIn("assign data_out = data_in;", code)

        res = _compile_with_iverilog(code)
        self.assertEqual(res.returncode, 0, f"iverilog failed: {res.stderr}\nCode:\n{code}")

    def test_9_verification_only_assertions_preservation(self):
        """Test 9: Verification-only assertions are emitted as formal verification directives."""
        contract = PlanProtocolContract(
            protocol_name="AXI4_STREAM",
            protocol_version="1.0",
            verification_assertions=[
                {
                    "rule_id": "RULE_AXI_VAL_STABLE",
                    "assertion_type": "SVA",
                    "property_statement": "valid && !ready |=> $stable(data)",
                    "applicable_signals": ["valid", "ready", "data"]
                }
            ]
        )

        plan_dict = {
            "module_name": "axi_receiver",
            "target_language": "Verilog",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "valid", "direction": "input", "width": "1"},
                {"name": "ready", "direction": "input", "width": "1"},
                {"name": "data", "direction": "input", "width": "32"},
                {"name": "done", "direction": "output", "width": "1"},
            ],
            "protocol_contract": contract.to_dict(),
            "combinational_logic": [
                PlanCombinationalProcess(
                    process_id="COMB_DONE",
                    assignments=[
                        PlanCombinationalAssignment(
                            target="done",
                            expression=PlanExpression(expression_type="BOOLEAN", operator="&&", operands=["valid", "ready"])
                        )
                    ]
                )
            ]
        }

        code, _ = generate_hybrid_rtl(
            design_spec="AXI stream receiver with assertions",
            target="Verilog",
            rtl_plan=plan_dict,
            mode=GenerationMode.DETERMINISTIC
        )

        self.assertIn("// [VERIFICATION_ASSERTION] Rule: RULE_AXI_VAL_STABLE Type: SVA", code)
        self.assertIn("// Property: valid && !ready |=> $stable(data)", code)
        self.assertIn("assign done = (valid && ready);", code)

        res = _compile_with_iverilog(code)
        self.assertEqual(res.returncode, 0, f"iverilog failed: {res.stderr}\nCode:\n{code}")

    def test_10_unresolved_logic_boundaries(self):
        """Test 10: Unresolved logic requirements are explicitly bounded without speculative fabrication."""
        unres = PlanUnresolvedLogic(
            affected_entity="error_response",
            description="Specification does not define behavior for parity error.",
            missing_information="Parity error interrupt vs silent drop undefined in spec.",
            severity="WARNING"
        )

        plan_dict = {
            "module_name": "packet_filter",
            "target_language": "Verilog",
            "ports": [
                {"name": "pkt_in", "direction": "input", "width": "8"},
                {"name": "pkt_out", "direction": "output", "width": "8"},
            ],
            "unresolved_logic": [unres.to_dict()],
            "combinational_logic": [
                PlanCombinationalProcess(
                    process_id="COMB_PKT",
                    assignments=[
                        PlanCombinationalAssignment(
                            target="pkt_out",
                            expression=PlanExpression(expression_type="IDENTIFIER", operands=["pkt_in"])
                        )
                    ]
                )
            ]
        }

        code, _ = generate_hybrid_rtl(
            design_spec="Packet filter with incomplete error specification",
            target="Verilog",
            rtl_plan=plan_dict,
            mode=GenerationMode.DETERMINISTIC
        )

        self.assertIn("// [UNRESOLVED_LOGIC_BOUNDARIES]", code)
        self.assertIn("// - [WARNING] error_response: Specification does not define behavior for parity error.", code)
        self.assertIn("//   Missing: Parity error interrupt vs silent drop undefined in spec.", code)

        res = _compile_with_iverilog(code)
        self.assertEqual(res.returncode, 0, f"iverilog failed: {res.stderr}\nCode:\n{code}")

    def test_11_full_pipeline_generate_code_integration(self):
        """Test 11: End-to-end integration through rtl_agent.generate_code() produces plan-driven RTL."""
        from core.agents.rtl_agent import generate_code

        # Build EFS IR for a streaming compute block
        ir = EFSIR(
            metadata=EFSMetadata(design_name="stream_acc", top_module="stream_acc"),
            components=[
                EFSComponent(
                    component_id="comp_acc",
                    name="stream_acc",
                    type="compute"
                )
            ],
            signals=[
                EFSSignal(signal_id="s1", name="clk", direction="input", width="1", owner="comp_acc"),
                EFSSignal(signal_id="s2", name="rst_n", direction="input", width="1", owner="comp_acc"),
                EFSSignal(signal_id="s3", name="en", direction="input", width="1", owner="comp_acc"),
                EFSSignal(signal_id="s4", name="in_data", direction="input", width="8", owner="comp_acc"),
                EFSSignal(signal_id="s5", name="acc_out", direction="output", width="8", owner="comp_acc"),
            ]
        )

        dummy_cp = MagicMock(
            functional_requirements=[],
            source_evidence=[],
            target_component={"name": "stream_acc"},
            signals=ir.signals,
            target_signals=ir.signals,
            registers=[],
            fsm=None
        )
        with patch("core.agents.rtl_agent.get_llm_client") as mock_getter:
            mock_llm = MagicMock()
            mock_llm.available = False
            mock_getter.return_value = mock_llm

            code, pipe_res = generate_code(
                design_spec="Stream accumulator accumulating in_data when en is active.",
                target="Verilog",
                efs_ir=ir,
                context_pack=dummy_cp,
                generation_mode="DETERMINISTIC",
                return_report=True
            )

        self.assertTrue(isinstance(code, str) and len(code) > 20)
        self.assertIn("module stream_acc", code)
        self.assertIn("input  wire", code)
        self.assertIn("output", code)
        self.assertIn("endmodule", code)

        res = _compile_with_iverilog(code)
        self.assertEqual(res.returncode, 0, f"iverilog failed: {res.stderr}\nCode:\n{code}")


if __name__ == "__main__":
    unittest.main()
