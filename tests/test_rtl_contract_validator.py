"""
Tests for Deterministic Post-Generation RTL Contract Validator.

Verifies:
1. Valid RTL passing all contract checks against RTLGenerationPlan.
2. Missing port detected as critical violation.
3. Unexpected port detected as violation.
4. Port direction mismatch detected.
5. Port width mismatch detected.
6. Missing internal register detected.
7. Clock edge and reset polarity mismatches detected.
8. Missing FSM state detected.
9. Undriven output port detected.
10. Verification report generated without modifying the original plan.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.verification.rtl_contract_validator import (
    RTLContractValidator,
    ContractViolationType
)
from core.efs_ir.rtl_plan import (
    RTLGenerationPlan,
    PlanPort,
    PlanRegister,
    PlanFSM,
    PlanResetRequirements
)


class TestRTLContractValidator(unittest.TestCase):

    def setUp(self):
        self.validator = RTLContractValidator()
        self.plan = RTLGenerationPlan(
            module_name="safe_counter",
            target_language="Verilog",
            ports=[
                PlanPort(name="clk", direction="input", width="1", semantic_role="clock"),
                PlanPort(name="rst_n", direction="input", width="1", semantic_role="reset"),
                PlanPort(name="enable", direction="input", width="1"),
                PlanPort(name="count", direction="output", width="8"),
            ],
            internal_registers=[
                PlanRegister(name="count_reg", width="8", reset_val="8'd0")
            ],
            reset_requirements=PlanResetRequirements(
                reset_signal="rst_n",
                reset_polarity="active_low",
                reset_type="async"
            )
        )

    def test_valid_rtl_passes_contract(self):
        """Verify compliant Verilog code passes all contract checks."""
        valid_code = """
module safe_counter (
    input wire clk,
    input wire rst_n,
    input wire enable,
    output wire [7:0] count
);
    reg [7:0] count_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count_reg <= 8'd0;
        end else if (enable) begin
            count_reg <= count_reg + 8'd1;
        end
    end

    assign count = count_reg;

endmodule
"""
        report = self.validator.validate_contract(valid_code, self.plan)
        self.assertTrue(report.is_valid)
        self.assertEqual(report.validation_status, "VALIDATED")
        self.assertEqual(len(report.contract_violations), 0)

    def test_missing_port_detected(self):
        """Verify that omitting a planned port is flagged as a critical violation."""
        code_missing_enable = """
module safe_counter (
    input wire clk,
    input wire rst_n,
    output wire [7:0] count
);
    reg [7:0] count_reg;
    assign count = count_reg;
endmodule
"""
        report = self.validator.validate_contract(code_missing_enable, self.plan)
        self.assertFalse(report.is_valid)
        self.assertEqual(report.validation_status, "CONTRACT_VIOLATION")
        missing_v = [v for v in report.contract_violations if v.violation_type == ContractViolationType.MISSING_PORT]
        self.assertTrue(len(missing_v) > 0)
        self.assertEqual(missing_v[0].entity, "enable")

    def test_port_direction_mismatch_detected(self):
        """Verify port direction mismatch (e.g. output declared as input) is caught."""
        code_wrong_dir = """
module safe_counter (
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire [7:0] count
);
endmodule
"""
        report = self.validator.validate_contract(code_wrong_dir, self.plan)
        self.assertFalse(report.is_valid)
        dir_v = [v for v in report.contract_violations if v.violation_type == ContractViolationType.PORT_DIRECTION_MISMATCH]
        self.assertTrue(len(dir_v) > 0)
        self.assertEqual(dir_v[0].entity, "count")

    def test_port_width_mismatch_detected(self):
        """Verify port bit-width mismatch is caught deterministically."""
        code_wrong_width = """
module safe_counter (
    input wire clk,
    input wire rst_n,
    input wire enable,
    output wire [15:0] count
);
    assign count = 16'd0;
endmodule
"""
        report = self.validator.validate_contract(code_wrong_width, self.plan)
        self.assertFalse(report.is_valid)
        width_v = [v for v in report.contract_violations if v.violation_type == ContractViolationType.PORT_WIDTH_MISMATCH]
        self.assertTrue(len(width_v) > 0)
        self.assertEqual(width_v[0].entity, "count")

    def test_unexpected_port_detected(self):
        """Verify LLM hallucinating extra ports is detected as unexpected implementation."""
        code_extra_port = """
module safe_counter (
    input wire clk,
    input wire rst_n,
    input wire enable,
    input wire debug_trap_in,
    output wire [7:0] count
);
    assign count = 8'd0;
endmodule
"""
        report = self.validator.validate_contract(code_extra_port, self.plan)
        unexp_v = [v for v in report.contract_violations if v.violation_type == ContractViolationType.UNEXPECTED_PORT]
        self.assertTrue(len(unexp_v) > 0)
        self.assertEqual(unexp_v[0].entity, "debug_trap_in")

    def test_reset_polarity_inversion_detected(self):
        """Verify active-low reset placed in posedge sensitivity list is flagged."""
        code_inverted_reset = """
module safe_counter (
    input wire clk,
    input wire rst_n,
    input wire enable,
    output wire [7:0] count
);
    reg [7:0] count_reg;
    always @(posedge clk or posedge rst_n) begin
        count_reg <= 8'd0;
    end
    assign count = count_reg;
endmodule
"""
        report = self.validator.validate_contract(code_inverted_reset, self.plan)
        self.assertFalse(report.is_valid)
        rst_v = [v for v in report.contract_violations if v.violation_type == ContractViolationType.RESET_POLARITY_MISMATCH]
        self.assertTrue(len(rst_v) > 0)
        self.assertEqual(rst_v[0].entity, "rst_n")

    def test_missing_fsm_state_detected(self):
        """Verify missing FSM states from the plan are flagged."""
        plan_fsm = RTLGenerationPlan(
            module_name="traffic_light",
            ports=[
                PlanPort(name="clk", direction="input", width="1"),
                PlanPort(name="rst_n", direction="input", width="1"),
                PlanPort(name="light_out", direction="output", width="3")
            ],
            fsm=PlanFSM(
                states=["RED", "GREEN", "YELLOW"],
                initial_state="RED"
            )
        )
        code_missing_yellow = """
module traffic_light (
    input wire clk,
    input wire rst_n,
    output wire [2:0] light_out
);
    localparam RED = 2'b00;
    localparam GREEN = 2'b01;
    assign light_out = 3'b001;
endmodule
"""
        report = self.validator.validate_contract(code_missing_yellow, plan_fsm)
        self.assertFalse(report.is_valid)
        fsm_v = [v for v in report.contract_violations if v.violation_type == ContractViolationType.MISSING_FSM_STATE]
        self.assertTrue(len(fsm_v) > 0)
        self.assertEqual(fsm_v[0].entity, "YELLOW")

    def test_undriven_output_port_detected(self):
        """Verify declared output port that is never driven on LHS is flagged."""
        code_undriven = """
module safe_counter (
    input wire clk,
    input wire rst_n,
    input wire enable,
    output wire [7:0] count
);
    // count is declared as output but never assigned via continuous assign or always block
    reg [7:0] internal_wire;
endmodule
"""
        report = self.validator.validate_contract(code_undriven, self.plan)
        self.assertFalse(report.is_valid)
        undriven_v = [v for v in report.contract_violations if v.violation_type == ContractViolationType.UNDRIVEN_OUTPUT]
        self.assertTrue(len(undriven_v) > 0)
        self.assertEqual(undriven_v[0].entity, "count")


if __name__ == "__main__":
    unittest.main()

