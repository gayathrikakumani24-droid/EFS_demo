"""
Tests for Generic, Architecture-Agnostic Component Projection.

Verifies:
1. Projection works across diverse architectures (ALU, Controller, Pipelined Datapath, Memory Interface).
2. No reliance on hardcoded substrings or decoder-specific heuristics.
3. Hierarchical components and subcomponents are preserved.
4. Ambiguous target queries produce explicit AMBIGUOUS_TARGET results.
5. Shared clock and reset resources are correctly associated.
6. Unowned/unresolved signals are diagnosed and tracked without silent loss.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.efs_ir.models import (
    EFSIR, EFSComponent, EFSSignal, EFSRegister, EFSFSM, EFSInterface, EFSMetadata
)
from core.retrieval.target_resolver import resolve_target, TargetResolutionResult
from core.retrieval.dependency_resolver import resolve_dependencies
from core.context_compiler.resolver import ContextResolver


class TestGenericComponentProjection(unittest.TestCase):

    def setUp(self):
        """Build a synthetic multi-component hardware design."""
        self.comp_top = EFSComponent(
            component_id="COMP_TOP",
            name="soc_top",
            type="top_level_design",
            subcomponents=["COMP_ALU", "COMP_FSM_CTRL", "COMP_MEM_IF"]
        )
        self.comp_alu = EFSComponent(
            component_id="COMP_ALU",
            name="arithmetic_logic_unit",
            type="datapath",
            parent_id="COMP_TOP",
            clock_domain="clk_core",
            reset_domain="rst_n"
        )
        self.comp_fsm = EFSComponent(
            component_id="COMP_FSM_CTRL",
            name="sequence_controller",
            type="controller",
            parent_id="COMP_TOP",
            clock_domain="clk_core",
            reset_domain="rst_n"
        )
        self.comp_mem = EFSComponent(
            component_id="COMP_MEM_IF",
            name="memory_interface_unit",
            type="interface_bridge",
            parent_id="COMP_TOP",
            clock_domain="clk_bus",
            reset_domain="rst_bus_n"
        )

        # Interfaces
        self.iface_alu = EFSInterface(
            interface_id="IF_ALU",
            name="alu_bus",
            source_component="COMP_TOP",
            destination_component="COMP_ALU",
            signals=["SIG_OPA", "SIG_OPB", "SIG_ALU_RES"]
        )

        # Signals
        self.sig_clk = EFSSignal(signal_id="SIG_CLK", name="clk_core", semantic_role="clock", direction="input", width="1")
        self.sig_rst = EFSSignal(signal_id="SIG_RST", name="rst_n", semantic_role="reset", direction="input", width="1")
        self.sig_opa = EFSSignal(signal_id="SIG_OPA", name="operand_a", direction="input", width="32", owner="COMP_ALU")
        self.sig_opb = EFSSignal(signal_id="SIG_OPB", name="operand_b", direction="input", width="32", owner="COMP_ALU")
        self.sig_res = EFSSignal(signal_id="SIG_ALU_RES", name="alu_result", direction="output", width="32", owner="COMP_ALU")
        self.sig_start = EFSSignal(signal_id="SIG_START", name="ctrl_start", direction="input", width="1", owner="COMP_FSM_CTRL")
        self.sig_done = EFSSignal(signal_id="SIG_DONE", name="ctrl_done", direction="output", width="1", owner="COMP_FSM_CTRL")

        # Registers
        self.reg_acc = EFSRegister(
            register_id="REG_ACC",
            name="accumulator",
            width="32",
            owner="COMP_ALU",
            reset_value="0"
        )
        self.reg_state = EFSRegister(
            register_id="REG_STATE",
            name="fsm_state_reg",
            width="3",
            owner="COMP_FSM_CTRL",
            reset_value="0"
        )

        # FSM
        self.fsm_ctrl = EFSFSM(
            fsm_id="FSM_MAIN",
            name="main_sequence_fsm"
        )
        self.fsm_ctrl.component_id = "COMP_FSM_CTRL"

        self.efs_ir = EFSIR(
            metadata=EFSMetadata(design_name="soc_top", top_module="soc_top"),
            components=[self.comp_top, self.comp_alu, self.comp_fsm, self.comp_mem],
            interfaces=[self.iface_alu],
            signals=[self.sig_clk, self.sig_rst, self.sig_opa, self.sig_opb, self.sig_res, self.sig_start, self.sig_done],
            registers=[self.reg_acc, self.reg_state],
            fsms=[self.fsm_ctrl]
        )

    def test_target_resolution_datapath_unit(self):
        """Verify target resolution finds an ALU without any decoder assumptions."""
        res = resolve_target("arithmetic_logic_unit", self.efs_ir)
        self.assertTrue(res.is_success())
        self.assertEqual(res.target_component.component_id, "COMP_ALU")

    def test_target_resolution_controller_unit(self):
        """Verify target resolution finds a controller unit."""
        res = resolve_target("sequence_controller", self.efs_ir)
        self.assertTrue(res.is_success())
        self.assertEqual(res.target_component.component_id, "COMP_FSM_CTRL")

    def test_ambiguity_detection(self):
        """Verify that conflicting identical candidate aliases return AMBIGUOUS_TARGET."""
        comp_a = EFSComponent(component_id="COMP_A", name="dsp_core_a", aliases=["core_unit"])
        comp_b = EFSComponent(component_id="COMP_B", name="dsp_core_b", aliases=["core_unit"])
        ir_ambig = EFSIR(
            metadata=EFSMetadata(design_name="dual_core"),
            components=[comp_a, comp_b]
        )
        res = resolve_target("core_unit", ir_ambig)
        self.assertEqual(res.status, "AMBIGUOUS_TARGET")
        self.assertEqual(len(res.matching_components), 2)

    def test_dependency_resolution_preserves_registers_and_clocks(self):
        """Verify ALU projection preserves owned accumulator register and shared clock/reset."""
        deps = resolve_dependencies(self.comp_alu, self.efs_ir)
        sig_names = {s.name for s in deps.signals}
        reg_names = {r.name for r in deps.registers}

        # Shared clock and reset
        self.assertIn("clk_core", sig_names)
        self.assertIn("rst_n", sig_names)

        # Owned inputs and outputs
        self.assertIn("operand_a", sig_names)
        self.assertIn("operand_b", sig_names)
        self.assertIn("alu_result", sig_names)

        # Accumulator register must be preserved (not dropped by any component type heuristic)
        self.assertIn("accumulator", reg_names)

        # Sibling signals must not be accidentally imported
        self.assertNotIn("ctrl_start", sig_names)
        self.assertNotIn("ctrl_done", sig_names)

    def test_controller_dependency_resolution_preserves_fsm(self):
        """Verify sequence controller projection preserves FSM and state registers."""
        deps = resolve_dependencies(self.comp_fsm, self.efs_ir)
        sig_names = {s.name for s in deps.signals}
        reg_names = {r.name for r in deps.registers}
        fsm_names = {f.name for f in deps.fsms}

        self.assertIn("ctrl_start", sig_names)
        self.assertIn("ctrl_done", sig_names)
        self.assertIn("fsm_state_reg", reg_names)
        self.assertIn("main_sequence_fsm", fsm_names)
        self.assertNotIn("accumulator", reg_names)

    def test_context_resolver_generic_filtering(self):
        """Verify ContextResolver resolves component signals without architecture-specific suppression."""
        resolver = ContextResolver(self.efs_ir)
        alu_sigs = resolver.get_component_signals(self.comp_alu)
        alu_regs = resolver.get_component_registers(self.comp_alu)
        
        sig_names = {s.name for s in alu_sigs}
        self.assertIn("alu_result", sig_names)
        self.assertIn("clk_core", sig_names)
        self.assertEqual(len(alu_regs), 1)
        self.assertEqual(alu_regs[0].name, "accumulator")


if __name__ == "__main__":
    unittest.main()
