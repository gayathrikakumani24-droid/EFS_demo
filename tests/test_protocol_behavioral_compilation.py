"""
Comprehensive Unit & Integration Test Suite for Protocol-Aware Behavioral Compilation.

Verifies:
1. Applicable protocol rules survive context compilation and component projection.
2. Protocol contracts (PlanProtocolContract & PlanProtocolRule) are populated in RTLGenerationPlan.
3. Handshake rules with sufficient evidence generate appropriate behavioral representation (transfer enable, reset action).
4. Verification-only constraints remain protocol contracts and do not fabricate spurious RTL assignments.
5. Unresolved protocol behavior (e.g., missing paired handshake port) is explicitly reported in PlanUnresolvedLogic.
6. Irrelevant protocol rules are excluded from unrelated components (strict component isolation).
7. Source traceability is preserved from protocol evidence to RTL Generation Plan directives.
8. Multiple protocol-independent test fixtures (Streaming channel, Request/Grant bus, Non-protocol decoder).
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
    EFSInterface,
    EFSProtocolRule,
    EFSMetadata,
)
from core.efs_ir.rtl_plan import (
    RTLGenerationPlan,
    PlanProtocolContract,
    PlanProtocolRule,
    PlanOrigin,
)
from core.context_compiler.behavioral_plan_compiler import (
    BehavioralPlanCompiler,
    normalize_protocol_rule,
    NormalizedProtocolRule,
)
from core.context_compiler import compile_context
from core.agents.rtl_agent import plan_rtl


class TestProtocolBehavioralCompilation(unittest.TestCase):

    def setUp(self):
        """Set up diverse, protocol-independent test fixtures."""
        # -------------------------------------------------------------
        # FIXTURE 1: Streaming Channel Receiver (Valid/Ready Handshake)
        # -------------------------------------------------------------
        self.comp_stream = EFSComponent(
            component_id="COMP_STREAM_RX",
            name="stream_receiver",
            type="interface_receiver",
            interfaces=["IF_STREAM_IN"],
            clock_domain="clk",
            reset_domain="rst_n"
        )
        self.iface_stream = EFSInterface(
            interface_id="IF_STREAM_IN",
            name="stream_in",
            protocol="GENERIC_STREAM",
            source_component="EXTERNAL",
            destination_component="COMP_STREAM_RX",
            signals=["SIG_S_CLK", "SIG_S_RST", "SIG_S_VALID", "SIG_S_READY", "SIG_S_DATA"]
        )
        self.sig_s_clk = EFSSignal(signal_id="SIG_S_CLK", name="clk", direction="input", width="1", semantic_role="clock", owner="COMP_STREAM_RX")
        self.sig_s_rst = EFSSignal(signal_id="SIG_S_RST", name="rst_n", direction="input", width="1", semantic_role="reset", owner="COMP_STREAM_RX")
        self.sig_s_valid = EFSSignal(signal_id="SIG_S_VALID", name="s_valid", direction="input", width="1", interface="IF_STREAM_IN", owner="COMP_STREAM_RX")
        self.sig_s_ready = EFSSignal(signal_id="SIG_S_READY", name="s_ready", direction="input", width="1", interface="IF_STREAM_IN", owner="COMP_STREAM_RX")
        self.sig_s_data = EFSSignal(signal_id="SIG_S_DATA", name="s_data", direction="input", width="32", interface="IF_STREAM_IN", owner="COMP_STREAM_RX")
        self.sig_s_accept = EFSSignal(signal_id="SIG_S_ACCEPT", name="transfer_active", direction="output", width="1", owner="COMP_STREAM_RX")

        self.reg_buffer = EFSRegister(
            register_id="REG_BUF",
            name="payload_buffer",
            width="32",
            reset_value="32'h0",
            owner="COMP_STREAM_RX",
            description="Payload buffer capturing incoming stream words on handshake"
        )

        # Synthesizable handshake rule
        self.prule_handshake = EFSProtocolRule(
            rule_id="PRULE_STREAM_HS_01",
            name="Stream Handshake Transfer",
            type="handshake",
            description="Data payload transferred when s_valid and s_ready are both asserted on rising clk edge.",
            details={
                "interface": "stream_in",
                "protocol": "GENERIC_STREAM",
                "signals": ["s_valid", "s_ready", "s_data"],
                "condition": "s_valid && s_ready",
                "is_synthesizable_behavior": True
            }
        )

        # Reset rule
        self.prule_reset = EFSProtocolRule(
            rule_id="PRULE_STREAM_RST_02",
            name="Stream Reset Initialization",
            type="reset",
            description="Control signals must be deasserted to 0 upon rst_n assertion.",
            details={
                "interface": "stream_in",
                "protocol": "GENERIC_STREAM",
                "signals": ["transfer_active"],
                "reset_behavior": "deasserted_to_zero",
                "is_synthesizable_behavior": True
            }
        )

        # Verification-only assertion constraint
        self.prule_stability = EFSProtocolRule(
            rule_id="PRULE_STREAM_STABLE_03",
            name="Stream Payload Stability Assertion",
            type="stability",
            description="Once s_valid is asserted, s_data must never change until handshake completes. (Property check)",
            details={
                "interface": "stream_in",
                "protocol": "GENERIC_STREAM",
                "signals": ["s_valid", "s_data", "s_ready"],
                "verification_only": True
            }
        )

        # -------------------------------------------------------------
        # FIXTURE 2: Request-Grant Bus Arbiter (Req/Ack Handshake)
        # -------------------------------------------------------------
        self.comp_bus = EFSComponent(
            component_id="COMP_BUS_ARB",
            name="bus_arbiter",
            type="arbiter",
            interfaces=["IF_BUS_REQ"],
            clock_domain="clk",
            reset_domain="rst_n"
        )
        self.iface_bus = EFSInterface(
            interface_id="IF_BUS_REQ",
            name="bus_req_if",
            protocol="SIMPLE_BUS",
            source_component="EXTERNAL",
            destination_component="COMP_BUS_ARB",
            signals=["SIG_B_REQ", "SIG_B_ACK"]
        )
        self.sig_b_req = EFSSignal(signal_id="SIG_B_REQ", name="bus_req", direction="input", width="1", interface="IF_BUS_REQ", owner="COMP_BUS_ARB")
        self.sig_b_ack = EFSSignal(signal_id="SIG_B_ACK", name="bus_ack", direction="output", width="1", interface="IF_BUS_REQ", owner="COMP_BUS_ARB")

        self.prule_bus_ack = EFSProtocolRule(
            rule_id="PRULE_BUS_GRANT_01",
            name="Bus Grant Handshake",
            type="handshake",
            description="Bus acknowledge bus_ack asserted when bus_req is received and arbiter is grant-ready.",
            details={
                "interface": "bus_req_if",
                "protocol": "SIMPLE_BUS",
                "signals": ["bus_req", "bus_ack"],
                "condition": "bus_req && bus_ack",
                "is_synthesizable_behavior": True
            }
        )

        # -------------------------------------------------------------
        # FIXTURE 3: Pure Computation Unit (No Protocol Interfaces)
        # -------------------------------------------------------------
        self.comp_calc = EFSComponent(
            component_id="COMP_CALC",
            name="math_multiplier",
            type="computational_core",
            interfaces=[],
            clock_domain="clk",
            reset_domain="rst_n"
        )
        self.sig_c_op1 = EFSSignal(signal_id="SIG_C_OP1", name="operand_x", direction="input", width="16", owner="COMP_CALC")
        self.sig_c_op2 = EFSSignal(signal_id="SIG_C_OP2", name="operand_y", direction="input", width="16", owner="COMP_CALC")
        self.sig_c_res = EFSSignal(signal_id="SIG_C_RES", name="mult_result", direction="output", width="32", owner="COMP_CALC")

        # Multi-component EFS IR
        self.efs_ir = EFSIR(
            metadata=EFSMetadata(design_id="DSN_MULTI_PROTO", design_name="heterogeneous_soc", model_version=1),
            components=[self.comp_stream, self.comp_bus, self.comp_calc],
            interfaces=[self.iface_stream, self.iface_bus],
            signals=[
                self.sig_s_clk, self.sig_s_rst, self.sig_s_valid, self.sig_s_ready, self.sig_s_data, self.sig_s_accept,
                self.sig_b_req, self.sig_b_ack,
                self.sig_c_op1, self.sig_c_op2, self.sig_c_res
            ],
            registers=[self.reg_buffer],
            protocol_rules=[self.prule_handshake, self.prule_reset, self.prule_stability, self.prule_bus_ack]
        )

    def test_protocol_rules_survive_context_compilation_and_projection(self):
        """Verify protocol rules survive RTLContextCompiler projection for relevant component."""
        compiled = compile_context(
            "rtl",
            efs_ir=self.efs_ir,
            target_component="stream_receiver"
        )

        self.assertIn("protocol_rules", compiled)
        proto_rules = compiled["protocol_rules"]
        self.assertGreater(len(proto_rules), 0, "stream_receiver must retain protocol rules")

        # Check rule IDs or texts
        rule_texts = [r.get("text", "") for r in proto_rules]
        self.assertTrue(any("s_valid" in t and "s_ready" in t for t in rule_texts), "Handshake rule must survive projection")

    def test_protocol_contract_population_in_plan(self):
        """Verify BehavioralPlanCompiler populates PlanProtocolContract with structured PlanProtocolRule entries."""
        plan_dict = {
            "module_name": "stream_receiver",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1", "semantic_role": "clock"},
                {"name": "rst_n", "direction": "input", "width": "1", "semantic_role": "reset"},
                {"name": "s_valid", "direction": "input", "width": "1", "interface": "stream_in"},
                {"name": "s_ready", "direction": "input", "width": "1", "interface": "stream_in"},
                {"name": "s_data", "direction": "input", "width": "32", "interface": "stream_in"},
                {"name": "transfer_active", "direction": "output", "width": "1"},
            ],
            "internal_registers": [
                {"name": "payload_buffer", "width": "32", "reset_value": "32'h0"},
            ]
        }

        compiled_plan = BehavioralPlanCompiler.compile_and_populate(
            plan_dict=plan_dict,
            efs_ir=self.efs_ir,
            target_comp=self.comp_stream
        )

        self.assertIn("protocol_contract", compiled_plan)
        proto_contract = compiled_plan["protocol_contract"]
        self.assertEqual(proto_contract["protocol_name"], "GENERIC_STREAM")
        self.assertGreater(len(proto_contract["rules"]), 0)

        # Deserialize into PlanProtocolContract
        contract_obj = PlanProtocolContract.from_dict(proto_contract)
        self.assertIsInstance(contract_obj, PlanProtocolContract)
        self.assertGreater(len(contract_obj.rules), 0)

        # Find handshake rule
        hs_rule = next((r for r in contract_obj.rules if r.rule_id == "PRULE_STREAM_HS_01"), None)
        self.assertIsNotNone(hs_rule)
        self.assertEqual(hs_rule.rule_type, "HANDSHAKE")
        self.assertTrue(hs_rule.is_synthesizable_behavior)
        self.assertIn("s_valid", hs_rule.applicable_signals)
        self.assertIn("s_ready", hs_rule.applicable_signals)
        self.assertEqual(hs_rule.handshake_condition, "s_valid && s_ready")

    def test_handshake_rule_compiles_into_register_update_enable(self):
        """Verify handshake condition is compiled into register update enable directive."""
        plan_dict = {
            "module_name": "stream_receiver",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "rst_n", "direction": "input", "width": "1"},
                {"name": "s_valid", "direction": "input", "width": "1", "interface": "stream_in"},
                {"name": "s_ready", "direction": "input", "width": "1", "interface": "stream_in"},
                {"name": "s_data", "direction": "input", "width": "32", "interface": "stream_in"},
                {"name": "transfer_active", "direction": "output", "width": "1"},
            ],
            "internal_registers": [
                {"name": "payload_buffer", "width": "32", "reset_value": "32'h0"},
            ]
        }

        compiled_plan = BehavioralPlanCompiler.compile_and_populate(
            plan_dict=plan_dict,
            efs_ir=self.efs_ir,
            target_comp=self.comp_stream
        )

        seq_procs = compiled_plan.get("sequential_logic", [])
        self.assertGreater(len(seq_procs), 0)

        main_proc = seq_procs[0]
        updates = main_proc.get("register_updates", [])
        buf_update = next((u for u in updates if u["target"] == "payload_buffer"), None)
        self.assertIsNotNone(buf_update, "payload_buffer must have register update")
        self.assertEqual(buf_update["enable"], "s_valid && s_ready")
        self.assertEqual(buf_update["origin"], PlanOrigin.DERIVED.value)

        # Combinational transaction acceptance flag
        comb_procs = compiled_plan.get("combinational_logic", [])
        all_assigns = []
        for cp in comb_procs:
            all_assigns.extend(cp.get("assignments", []))

        accept_assign = next((a for a in all_assigns if a["target"] == "transfer_active"), None)
        self.assertIsNotNone(accept_assign, "transfer_active must have combinational assignment")
        self.assertIn("s_valid && s_ready", accept_assign["expression_raw"])
        self.assertEqual(accept_assign["origin"], PlanOrigin.DERIVED.value)

    def test_verification_only_constraint_preserved_without_rtl_fabrication(self):
        """Verify assertion / property check rules are preserved as verification contracts, not RTL assignments."""
        plan_dict = {
            "module_name": "stream_receiver",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "rst_n", "direction": "input", "width": "1"},
                {"name": "s_valid", "direction": "input", "width": "1", "interface": "stream_in"},
                {"name": "s_ready", "direction": "input", "width": "1", "interface": "stream_in"},
                {"name": "s_data", "direction": "input", "width": "32", "interface": "stream_in"},
                {"name": "transfer_active", "direction": "output", "width": "1"},
            ],
            "internal_registers": [
                {"name": "payload_buffer", "width": "32", "reset_value": "32'h0"},
            ]
        }

        compiled_plan = BehavioralPlanCompiler.compile_and_populate(
            plan_dict=plan_dict,
            efs_ir=self.efs_ir,
            target_comp=self.comp_stream
        )

        proto_contract = compiled_plan["protocol_contract"]
        contract_obj = PlanProtocolContract.from_dict(proto_contract)

        # Stability rule must have is_synthesizable_behavior = False
        stability_rule = next((r for r in contract_obj.rules if r.rule_id == "PRULE_STREAM_STABLE_03"), None)
        self.assertIsNotNone(stability_rule)
        self.assertFalse(stability_rule.is_synthesizable_behavior)

        # Must be listed in verification_assertions
        assertions = contract_obj.verification_assertions
        self.assertTrue(any(a["rule_id"] == "PRULE_STREAM_STABLE_03" for a in assertions))

        # Must NOT generate an RTL assignment or register update targeting s_data
        comb_assigns = []
        for cp in compiled_plan.get("combinational_logic", []):
            comb_assigns.extend(cp.get("assignments", []))
        self.assertFalse(any(a["target"] == "s_data" for a in comb_assigns), "Verification constraint must not fabricate assignment to s_data")

    def test_unresolved_protocol_behavior_reported(self):
        """Verify incomplete protocol interfaces (missing paired handshake signal) are recorded in unresolved_logic."""
        # Module with s_valid, but missing s_ready
        plan_dict = {
            "module_name": "broken_receiver",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "rst_n", "direction": "input", "width": "1"},
                {"name": "s_valid", "direction": "input", "width": "1", "interface": "stream_in"},
                {"name": "s_data", "direction": "input", "width": "32", "interface": "stream_in"},
            ],
            "internal_registers": []
        }

        compiled_plan = BehavioralPlanCompiler.compile_and_populate(
            plan_dict=plan_dict,
            efs_ir=self.efs_ir,
            target_comp=self.comp_stream
        )

        unresolved = compiled_plan.get("unresolved_logic", [])
        proto_unres = next((u for u in unresolved if "PRULE_STREAM_HS_01" in str(u.get("affected_entity", ""))), None)
        self.assertIsNotNone(proto_unres, "Missing paired handshake signal s_ready must be recorded in unresolved_logic")
        self.assertIn("s_ready", str(proto_unres.get("description", "")))

    def test_strict_component_isolation_of_protocol_rules(self):
        """Verify irrelevant protocol rules targeting bus or stream are excluded from unrelated math_multiplier."""
        plan_dict = {
            "module_name": "math_multiplier",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "rst_n", "direction": "input", "width": "1"},
                {"name": "operand_x", "direction": "input", "width": "16"},
                {"name": "operand_y", "direction": "input", "width": "16"},
                {"name": "mult_result", "direction": "output", "width": "32"},
            ],
            "internal_registers": []
        }

        compiled_plan = BehavioralPlanCompiler.compile_and_populate(
            plan_dict=plan_dict,
            efs_ir=self.efs_ir,
            target_comp=self.comp_calc
        )

        # math_multiplier has NO protocol interfaces and disjoint signals
        # Protocol contract should be None or have 0 rules
        proto_contract = compiled_plan.get("protocol_contract")
        if proto_contract:
            self.assertEqual(len(proto_contract.get("rules", [])), 0, "Unrelated component must not inherit stream or bus protocol rules")

    def test_end_to_end_plan_rtl_with_protocol_contract(self):
        """Verify end-to-end plan_rtl creates and serializes RTLGenerationPlan with protocol_contract."""
        spec_text = """
        # Stream Receiver Specification
        Module: stream_receiver
        Clock: clk
        Reset: rst_n
        
        ## Ports
        - clk: input [0:0]
        - rst_n: input [0:0]
        - s_valid: input [0:0]
        - s_ready: input [0:0]
        - s_data: input [31:0]
        - transfer_active: output [0:0]
        """

        with patch("core.agents.rtl_agent.get_llm_client") as mock_getter:
            mock_llm = MagicMock()
            mock_llm.available = False
            mock_getter.return_value = mock_llm

            plan = plan_rtl(design_spec=spec_text, target="stream_receiver", efs_ir=self.efs_ir)

            self.assertIsInstance(plan, dict)
            self.assertEqual(plan["module_name"], "stream_receiver")
            self.assertIn("protocol_contract", plan)

            # Roundtrip to RTLGenerationPlan object
            plan_obj = RTLGenerationPlan.from_dict(plan)
            self.assertIsNotNone(plan_obj.protocol_contract)
            self.assertEqual(plan_obj.protocol_contract.protocol_name, "GENERIC_STREAM")
            self.assertGreater(len(plan_obj.protocol_contract.rules), 0)

    # -------------------------------------------------------------
    # PHASE 2.1 FOCUSED TESTS: NORMALIZATION & MULTI-SOURCE INTEGRATION
    # -------------------------------------------------------------

    def test_normalization_nested_details_dictionaries(self):
        """Verify normalize_protocol_rule merges nested details into canonical NormalizedProtocolRule."""
        raw_rule = {
            "rule_id": "PRULE_NESTED_01",
            "name": "Nested Spec Rule",
            "type": "handshake",
            "details": {
                "interface": "stream_in",
                "protocol": "STREAM_V1",
                "signals": ["s_valid", "s_ready"],
                "condition": "s_valid && s_ready",
                "verification_only": False,
                "citation": "Section 3.1"
            }
        }
        norm = normalize_protocol_rule(raw_rule)
        self.assertIsNotNone(norm)
        self.assertEqual(norm.rule_id, "PRULE_NESTED_01")
        self.assertEqual(norm.interface, "stream_in")
        self.assertEqual(norm.protocol, "STREAM_V1")
        self.assertEqual(norm.signals, ["s_valid", "s_ready"])
        self.assertEqual(norm.handshake_condition, "s_valid && s_ready")
        self.assertTrue(norm.is_synthesizable_behavior)
        self.assertFalse(norm.verification_only)
        self.assertEqual(norm.source_evidence, "Section 3.1")

    def test_normalization_top_level_protocol_attributes(self):
        """Verify normalize_protocol_rule correctly handles flat dictionary rules without details dict."""
        raw_rule = {
            "rule_id": "PRULE_FLAT_01",
            "name": "Flat Protocol Rule",
            "type": "handshake",
            "description": "Valid transfer when req and ack are both 1",
            "interface": "bus_if",
            "protocol": "SIMPLE_BUS",
            "signals": ["req", "ack"],
            "handshake_condition": "req && ack",
            "is_synthesizable_behavior": True,
            "source_evidence": "BusSpec_p42"
        }
        norm = normalize_protocol_rule(raw_rule)
        self.assertIsNotNone(norm)
        self.assertEqual(norm.rule_id, "PRULE_FLAT_01")
        self.assertEqual(norm.interface, "bus_if")
        self.assertEqual(norm.protocol, "SIMPLE_BUS")
        self.assertEqual(norm.signals, ["req", "ack"])
        self.assertEqual(norm.handshake_condition, "req && ack")
        self.assertTrue(norm.is_synthesizable_behavior)

    def test_normalization_model_object_protocol_rules(self):
        """Verify normalize_protocol_rule processes EFSProtocolRule dataclass instances without mutation."""
        prule = EFSProtocolRule(
            rule_id="PRULE_MODEL_01",
            name="Model Rule",
            type="handshake",
            description="Payload handshake transfer rule",
            details={
                "interface": "stream_in",
                "protocol": "GENERIC_STREAM",
                "signals": ["s_valid", "s_ready"]
            }
        )
        norm = normalize_protocol_rule(prule)
        self.assertIsNotNone(norm)
        self.assertEqual(norm.rule_id, "PRULE_MODEL_01")
        self.assertEqual(norm.interface, "stream_in")
        self.assertEqual(norm.protocol, "GENERIC_STREAM")
        self.assertEqual(norm.signals, ["s_valid", "s_ready"])

        # Verify source object was not mutated
        self.assertEqual(prule.rule_id, "PRULE_MODEL_01")
        self.assertIn("interface", prule.details)

    def test_normalization_signals_as_objects_and_dicts(self):
        """Verify normalization extracts signal names from strings, dicts, and EFSSignal objects."""
        raw_rule = {
            "rule_id": "PRULE_SIGS_OBJ",
            "description": "Rule referencing signals in multiple representations",
            "signals": [
                EFSSignal(signal_id="SIG_1", name="s_valid", direction="input"),
                {"name": "s_ready", "id": "SIG_2"},
                "s_data"
            ]
        }
        norm = normalize_protocol_rule(raw_rule)
        self.assertIsNotNone(norm)
        self.assertEqual(norm.signals, ["s_valid", "s_ready", "s_data"])

    def test_rules_supplied_through_each_of_three_sources(self):
        """Verify rules supplied through compiled_rtl, context_pack, and efs_ir are all captured."""
        plan_dict = {
            "module_name": "stream_receiver",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "rst_n", "direction": "input", "width": "1"},
                {"name": "s_valid", "direction": "input", "width": "1", "interface": "stream_in"},
                {"name": "s_ready", "direction": "input", "width": "1", "interface": "stream_in"},
                {"name": "s_data", "direction": "input", "width": "32", "interface": "stream_in"},
            ],
            "internal_registers": []
        }

        # Source 1: compiled_rtl
        compiled_rtl = {
            "protocol_rules": [
                {
                    "rule_id": "RULE_FROM_COMPILED_RTL",
                    "type": "handshake",
                    "interface": "stream_in",
                    "signals": ["s_valid", "s_ready"],
                    "condition": "s_valid && s_ready"
                }
            ]
        }

        # Source 2: context_pack
        context_pack = {
            "protocol_rules": [
                {
                    "rule_id": "RULE_FROM_CONTEXT_PACK",
                    "type": "stability",
                    "interface": "stream_in",
                    "signals": ["s_data"],
                    "description": "Payload must never change while valid is asserted",
                    "verification_only": True
                }
            ]
        }

        # Source 3: efs_ir
        efs_ir_src = EFSIR(
            metadata=EFSMetadata(design_id="DSN_3SRC", design_name="src_test", model_version=1),
            components=[self.comp_stream],
            interfaces=[self.iface_stream],
            signals=[self.sig_s_valid, self.sig_s_ready, self.sig_s_data],
            protocol_rules=[
                EFSProtocolRule(
                    rule_id="RULE_FROM_EFS_IR",
                    type="reset",
                    description="Signals reset upon rst_n assertion",
                    details={"interface": "stream_in", "signals": ["s_valid"]}
                )
            ]
        )

        compiled_plan = BehavioralPlanCompiler.compile_and_populate(
            plan_dict=plan_dict,
            efs_ir=efs_ir_src,
            target_comp=self.comp_stream,
            compiled_rtl=compiled_rtl,
            context_pack=context_pack
        )

        self.assertIn("protocol_contract", compiled_plan)
        contract = PlanProtocolContract.from_dict(compiled_plan["protocol_contract"])
        rule_ids = {r.rule_id for r in contract.rules}
        self.assertIn("RULE_FROM_COMPILED_RTL", rule_ids)
        self.assertIn("RULE_FROM_CONTEXT_PACK", rule_ids)
        self.assertIn("RULE_FROM_EFS_IR", rule_ids)

    def test_rules_with_missing_optional_interface_information(self):
        """Verify rules without explicit interface binding are evaluated using signal intersection."""
        plan_dict = {
            "module_name": "stream_receiver",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "rst_n", "direction": "input", "width": "1"},
                {"name": "s_valid", "direction": "input", "width": "1"},
                {"name": "s_ready", "direction": "input", "width": "1"},
                {"name": "s_data", "direction": "input", "width": "32"},
            ],
            "internal_registers": []
        }

        compiled_rtl = {
            "protocol_rules": [
                {
                    "rule_id": "RULE_NO_IFACE",
                    "type": "handshake",
                    # No interface specified!
                    "signals": ["s_valid", "s_ready"],
                    "condition": "s_valid && s_ready",
                    "description": "Generic handshake transfer"
                }
            ]
        }

        compiled_plan = BehavioralPlanCompiler.compile_and_populate(
            plan_dict=plan_dict,
            efs_ir=None,
            target_comp=self.comp_stream,
            compiled_rtl=compiled_rtl
        )

        self.assertIn("protocol_contract", compiled_plan)
        contract = PlanProtocolContract.from_dict(compiled_plan["protocol_contract"])
        self.assertTrue(any(r.rule_id == "RULE_NO_IFACE" for r in contract.rules))

    def test_duplicate_protocol_rules_handled_deterministically(self):
        """Verify duplicate rules across sources are deduplicated while preserving richer fields."""
        rule_partial = {
            "rule_id": "PRULE_DUP_01",
            "type": "handshake",
            "interface": "stream_in",
            "signals": ["s_valid", "s_ready"]
        }
        rule_rich = {
            "rule_id": "PRULE_DUP_01",
            "type": "handshake",
            "interface": "stream_in",
            "signals": ["s_valid", "s_ready"],
            "condition": "s_valid && s_ready",
            "citation": "IEEE 802.3 Clause 4",
            "protocol": "GENERIC_STREAM"
        }

        plan_dict = {
            "module_name": "stream_receiver",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "rst_n", "direction": "input", "width": "1"},
                {"name": "s_valid", "direction": "input", "width": "1", "interface": "stream_in"},
                {"name": "s_ready", "direction": "input", "width": "1", "interface": "stream_in"},
            ],
            "internal_registers": []
        }

        compiled_plan = BehavioralPlanCompiler.compile_and_populate(
            plan_dict=plan_dict,
            efs_ir=None,
            target_comp=self.comp_stream,
            compiled_rtl={"protocol_rules": [rule_partial]},
            context_pack={"protocol_rules": [rule_rich]}
        )

        contract = PlanProtocolContract.from_dict(compiled_plan["protocol_contract"])
        matching_rules = [r for r in contract.rules if r.rule_id == "PRULE_DUP_01"]
        self.assertEqual(len(matching_rules), 1, "Duplicate rule must be deduplicated into exactly one entry")
        merged = matching_rules[0]
        self.assertEqual(merged.handshake_condition, "s_valid && s_ready")
        self.assertEqual(merged.source_evidence, "IEEE 802.3 Clause 4")

    def test_source_traceability_preserved(self):
        """Verify source evidence and references are retained on PlanProtocolRule."""
        plan_dict = {
            "module_name": "stream_receiver",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "rst_n", "direction": "input", "width": "1"},
                {"name": "s_valid", "direction": "input", "width": "1", "interface": "stream_in"},
                {"name": "s_ready", "direction": "input", "width": "1", "interface": "stream_in"},
            ],
            "internal_registers": []
        }

        compiled_rtl = {
            "protocol_rules": [
                {
                    "rule_id": "PRULE_TRACE_01",
                    "type": "handshake",
                    "interface": "stream_in",
                    "signals": ["s_valid", "s_ready"],
                    "condition": "s_valid && s_ready",
                    "citation": "Spec_v2.0_Section_5.4",
                    "chunk_id": "chunk_doc_42"
                }
            ]
        }

        compiled_plan = BehavioralPlanCompiler.compile_and_populate(
            plan_dict=plan_dict,
            efs_ir=None,
            target_comp=self.comp_stream,
            compiled_rtl=compiled_rtl
        )

        contract = PlanProtocolContract.from_dict(compiled_plan["protocol_contract"])
        rule = next(r for r in contract.rules if r.rule_id == "PRULE_TRACE_01")
        self.assertEqual(rule.source_evidence, "Spec_v2.0_Section_5.4")
        self.assertIn("chunk_doc_42", rule.source_references)
        self.assertIn("Spec_v2.0_Section_5.4", contract.source_evidence_ids)

    def test_empty_and_malformed_rule_handling(self):
        """Verify empty, malformed, or None rules do not crash compilation and are cleanly skipped."""
        self.assertIsNone(normalize_protocol_rule(None))
        self.assertIsNone(normalize_protocol_rule({}))
        self.assertIsNone(normalize_protocol_rule("not a rule object"))
        self.assertIsNone(normalize_protocol_rule({"empty": True}))

        plan_dict = {
            "module_name": "stream_receiver",
            "ports": [
                {"name": "clk", "direction": "input", "width": "1"},
                {"name": "rst_n", "direction": "input", "width": "1"},
                {"name": "s_valid", "direction": "input", "width": "1", "interface": "stream_in"},
                {"name": "s_ready", "direction": "input", "width": "1", "interface": "stream_in"},
            ],
            "internal_registers": []
        }

        # Provide list containing malformed items
        compiled_plan = BehavioralPlanCompiler.compile_and_populate(
            plan_dict=plan_dict,
            efs_ir=None,
            target_comp=self.comp_stream,
            compiled_rtl={"protocol_rules": [None, {}, "corrupted_rule", {"rule_id": "PRULE_VALID_01", "interface": "stream_in", "signals": ["s_valid", "s_ready"], "type": "handshake"}]}
        )

        contract = PlanProtocolContract.from_dict(compiled_plan["protocol_contract"])
        self.assertEqual(len(contract.rules), 1)
        self.assertEqual(contract.rules[0].rule_id, "PRULE_VALID_01")


if __name__ == "__main__":
    unittest.main()
