"""
Tests for RTL Generation Plan: Provenance, Sequential, and Combinational Logic.

Verifies:
1. PlanOrigin classification (INHERITED, DERIVED, IMPLEMENTATION_DECISION).
2. DerivationMetadata tracking input entity references.
3. Structured sequential_logic representation (PlanSequentialProcess, PlanRegisterUpdate).
4. Structured combinational_logic representation (PlanCombinationalProcess, PlanExpression).
5. Unresolved logic representation for specification gaps.
6. Clean serialization and deserialization roundtrip.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.efs_ir.rtl_plan import (
    RTLGenerationPlan,
    PlanPort,
    PlanParameter,
    PlanRegister,
    PlanInternalSignal,
    PlanOrigin,
    DerivationMetadata,
    PlanExpression,
    PlanSequentialProcess,
    PlanRegisterUpdate,
    PlanCombinationalProcess,
    PlanCombinationalAssignment,
    PlanUnresolvedLogic,
    PlanStatus,
    PlanSpecificationStatus
)


class TestRTLPlanSequentialCombinational(unittest.TestCase):

    def test_plan_origin_and_derivation_metadata(self):
        """Verify ports and registers can be tagged with provenance and derivation info."""
        port = PlanPort(
            name="clk",
            direction="input",
            width="1",
            origin=PlanOrigin.INHERITED
        )
        self.assertEqual(port.origin, PlanOrigin.INHERITED)

        derived_signal = PlanInternalSignal(
            name="alu_carry",
            width="1",
            origin=PlanOrigin.DERIVED,
            derivation=DerivationMetadata(
                description="Derived from operand overflow detection rule",
                input_entities=["operand_a", "operand_b"],
                rule="CARRY_LOOKAHEAD"
            )
        )
        self.assertEqual(derived_signal.origin, PlanOrigin.DERIVED)
        self.assertEqual(derived_signal.derivation.rule, "CARRY_LOOKAHEAD")
        self.assertIn("operand_a", derived_signal.derivation.input_entities)

    def test_structured_sequential_logic_process(self):
        """Verify sequential logic process structure with register update semantics."""
        seq_proc = PlanSequentialProcess(
            process_id="SEQ_COUNTER_01",
            process_type="ALWAYS_FF",
            clock_signal="clk",
            clock_edge="POSEDGE",
            reset_signal="rst_n",
            reset_type="ASYNC",
            reset_active_level="LOW",
            reset_priority="HIGH",
            reset_actions={"count_reg": "32'd0"},
            register_updates=[
                PlanRegisterUpdate(
                    target="count_reg",
                    condition="enable == 1'b1",
                    enable="enable",
                    next_value=PlanExpression(
                        expression_id="expr_inc",
                        expression_type="ARITHMETIC",
                        operator="+",
                        operands=["count_reg", "32'd1"]
                    ),
                    hold_behavior="PRESERVE"
                )
            ]
        )

        d = seq_proc.to_dict()
        self.assertEqual(d["process_id"], "SEQ_COUNTER_01")
        self.assertEqual(d["clock_signal"], "clk")
        self.assertEqual(len(d["register_updates"]), 1)
        self.assertEqual(d["register_updates"][0]["target"], "count_reg")
        self.assertEqual(d["register_updates"][0]["next_value"]["operator"], "+")

        # Roundtrip deserialization
        restored = PlanSequentialProcess.from_dict(d)
        self.assertEqual(restored.process_id, "SEQ_COUNTER_01")
        self.assertEqual(restored.clock_signal, "clk")
        self.assertEqual(restored.register_updates[0].target, "count_reg")
        self.assertEqual(restored.register_updates[0].next_value.operator, "+")

    def test_structured_combinational_logic_process(self):
        """Verify combinational logic process with assignments and expressions."""
        comb_proc = PlanCombinationalProcess(
            process_id="COMB_ALU_MUX",
            process_type="ALWAYS_COMB",
            input_dependencies=["opcode", "op_a", "op_b"],
            output_targets=["result"],
            assignments=[
                PlanCombinationalAssignment(
                    target="result",
                    expression=PlanExpression(
                        expression_id="expr_mux",
                        expression_type="MUX",
                        operator="MUX",
                        operands=["op_a + op_b", "op_a - op_b", "op_a & op_b"],
                        condition="opcode"
                    ),
                    default_assignment="32'd0"
                )
            ]
        )

        d = comb_proc.to_dict()
        self.assertEqual(d["process_id"], "COMB_ALU_MUX")
        self.assertEqual(d["output_targets"], ["result"])
        self.assertEqual(len(d["assignments"]), 1)

        restored = PlanCombinationalProcess.from_dict(d)
        self.assertEqual(restored.process_id, "COMB_ALU_MUX")
        self.assertEqual(restored.assignments[0].target, "result")
        self.assertEqual(restored.assignments[0].default_assignment, "32'd0")

    def test_unresolved_logic_representation(self):
        """Verify that specification gaps can be explicitly marked as unresolved."""
        unresolved = PlanUnresolvedLogic(
            affected_entity="timer_threshold",
            description="Timer timeout threshold duration not specified in specification",
            missing_information="Duration in clock cycles or milliseconds",
            severity="BLOCKING"
        )
        d = unresolved.to_dict()
        self.assertEqual(d["affected_entity"], "timer_threshold")
        self.assertEqual(d["severity"], "BLOCKING")

    def test_full_plan_serialization_and_roundtrip(self):
        """Verify complete RTLGenerationPlan serialization including sequential & combinational sections."""
        plan = RTLGenerationPlan(
            module_name="pipeline_alu",
            target_language="SystemVerilog",
            architecture_class="DATAPATH",
            specification_status=PlanSpecificationStatus(status=PlanStatus.COMPLETE),
            ports=[
                PlanPort(name="clk", direction="input", width="1", origin=PlanOrigin.INHERITED),
                PlanPort(name="rst_n", direction="input", width="1", origin=PlanOrigin.INHERITED),
                PlanPort(name="data_out", direction="output", width="32", origin=PlanOrigin.INHERITED),
            ],
            internal_registers=[
                PlanRegister(name="pipeline_reg", width="32", origin=PlanOrigin.IMPLEMENTATION_DECISION)
            ],
            sequential_logic=[
                PlanSequentialProcess(
                    process_id="SEQ_STAGE1",
                    clock_signal="clk",
                    reset_signal="rst_n",
                    register_updates=[
                        PlanRegisterUpdate(target="pipeline_reg", hold_behavior="PRESERVE")
                    ]
                )
            ],
            combinational_logic=[
                PlanCombinationalProcess(
                    process_id="COMB_OUT",
                    assignments=[
                        PlanCombinationalAssignment(target="data_out", default_assignment="32'd0")
                    ]
                )
            ]
        )

        plan_dict = plan.to_dict()
        self.assertEqual(plan_dict["module_name"], "pipeline_alu")
        self.assertEqual(len(plan_dict["sequential_logic"]), 1)
        self.assertEqual(len(plan_dict["combinational_logic"]), 1)

        restored_plan = RTLGenerationPlan.from_dict(plan_dict)
        self.assertEqual(restored_plan.module_name, "pipeline_alu")
        self.assertEqual(len(restored_plan.sequential_logic), 1)
        self.assertEqual(restored_plan.sequential_logic[0].process_id, "SEQ_STAGE1")
        self.assertEqual(len(restored_plan.combinational_logic), 1)
        self.assertEqual(restored_plan.combinational_logic[0].process_id, "COMB_OUT")


if __name__ == "__main__":
    unittest.main()

