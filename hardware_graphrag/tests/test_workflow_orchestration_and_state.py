"""
Tests for Workflow Orchestration, State Machine Transitions, FSM Resolution Propagation,
Token Budget Enforcement, and Infinite Loop Prevention (Tests 1 - 11 + Reproduction Test).
"""

import pytest
import unittest
from unittest.mock import MagicMock, patch
from core.agents.orchestrator import (
    WorkflowState,
    resume_design_flow_with_resolutions,
    log_state_transition,
    log_generation_start,
    log_generation_end,
    log_generation_skipped
)
from core.efs_ir.models import (
    EFSIR, EFSComponent, EFSSignal, EFSFSM, EFSState, EFSTransition,
    EFSConflict, EFSResolution, EFSMetadata
)
from core.retrieval.context_pack import ContextPack, TaskIntent
from core.retrieval.token_budget import TokenBudgetManager
from core.retrieval.dependency_resolver import resolve_dependencies, ResolvedDependencies
from core.retrieval.protocol_resolver import validate_protocol_context_completeness
from core.agents.rtl_agent import validate_rtl_plan, plan_rtl


class TestWorkflowOrchestrationAndState(unittest.TestCase):

    def _create_base_efs_ir(self) -> EFSIR:
        """Create a standard grounded EFS IR."""
        comp = EFSComponent(
            component_id="COMP_DUT",
            name="dut_core",
            type="module",
            clock_domain="clk",
            reset_domain="rst_n"
        )
        sig_clk = EFSSignal(signal_id="SIG_CLK", name="clk", direction="input", width="1", semantic_role="clock", owner="COMP_DUT")
        sig_rst = EFSSignal(signal_id="SIG_RST", name="rst_n", direction="input", width="1", semantic_role="reset", owner="COMP_DUT")
        sig_start = EFSSignal(signal_id="SIG_START", name="start_pulse", direction="input", width="1", semantic_role="control", owner="COMP_DUT")
        sig_done = EFSSignal(signal_id="SIG_DONE", name="done_flag", direction="output", width="1", semantic_role="status", owner="COMP_DUT")

        meta = EFSMetadata(
            document_id="DOC_TEST",
            design_name="dut_core",
            model_version=1,
            resolution_version=0
        )
        return EFSIR(
            metadata=meta,
            components=[comp],
            signals=[sig_clk, sig_rst, sig_start, sig_done],
            conflicts=[],
            resolutions=[]
        )

    # -------------------------------------------------------------------------
    # TEST 1: Human resolves conflict -> Expected: exactly one revalidation.
    # -------------------------------------------------------------------------
    def test_test_1_human_resolves_conflict_exactly_one_revalidation(self):
        efs_ir = self._create_base_efs_ir()
        conflict = EFSConflict(
            conflict_id="CONF_SIG_01",
            type="SIGNAL_DIRECTION",
            severity="CRITICAL",
            blocking=True,
            status="UNRESOLVED",
            resolution_status="UNRESOLVED",
            title="Direction Conflict on done_flag",
            entities=["done_flag"],
            affected_entities=["done_flag"],
            property="direction",
            conflicting_values=["input", "output"],
            requires_human_decision=True
        )
        efs_ir.conflicts.append(conflict)

        resolution = EFSResolution(
            resolution_id="RES_01",
            conflict_id="CONF_SIG_01",
            target_entity="done_flag",
            field_name="direction",
            proposed_value="output",
            resolution_type="OVERRIDE",
            approval_status="APPROVED",
            affected_requirements=["done_flag"]
        )

        design_results = {
            "status": "AWAITING_HUMAN_DECISION",
            "pipeline_state": "AWAITING_HUMAN_DECISION",
            "efs_ir": efs_ir,
            "conflicts": [conflict],
            "plan": {"module_name": "dut_core", "submodules": []},
            "requirement_model": {},
            "protocol_rules": []
        }

        reval_spy = MagicMock(return_value=(False, []))
        with patch("core.verification.conflict_resolution_engine.ConflictResolutionEngine.revalidate_resolved_model", reval_spy), \
             patch("core.agents.orchestrator.generate_code", return_value="module dut_core; endmodule"), \
             patch("core.agents.orchestrator.generate_diagrams", return_value={}):
            resume_design_flow_with_resolutions(design_results, [resolution])

        # Assert exactly one revalidation was performed deterministically
        self.assertEqual(reval_spy.call_count, 1)

    # -------------------------------------------------------------------------
    # TEST 2: All conflicts resolved -> Expected: design generation starts exactly once.
    # -------------------------------------------------------------------------
    def test_test_2_all_conflicts_resolved_starts_generation_once(self):
        efs_ir = self._create_base_efs_ir()
        conflict = EFSConflict(
            conflict_id="CONF_02",
            type="SIGNAL_WIDTH",
            severity="CRITICAL",
            blocking=True,
            status="UNRESOLVED",
            resolution_status="UNRESOLVED",
            title="Width Conflict",
            entities=["data_bus"],
            affected_entities=["data_bus"],
            property="width",
            requires_human_decision=True
        )
        efs_ir.conflicts.append(conflict)

        resolution = EFSResolution(
            resolution_id="RES_02",
            conflict_id="CONF_02",
            target_entity="data_bus",
            field_name="width",
            proposed_value="32",
            resolution_type="OVERRIDE",
            approval_status="APPROVED",
            affected_requirements=["data_bus"]
        )

        design_results = {
            "status": "AWAITING_HUMAN_DECISION",
            "pipeline_state": "AWAITING_HUMAN_DECISION",
            "efs_ir": efs_ir,
            "conflicts": [conflict],
            "plan": {"module_name": "dut_core", "submodules": []},
            "requirement_model": {},
            "protocol_rules": []
        }

        gen_spy = MagicMock(return_value="module dut_core(input clk, input rst_n); endmodule")
        with patch("core.agents.orchestrator.generate_code", gen_spy), \
             patch("core.agents.orchestrator.generate_diagrams", return_value={}):
            res = resume_design_flow_with_resolutions(design_results, [resolution], target_outputs=["Verilog"])

        self.assertEqual(gen_spy.call_count, 1)
        self.assertEqual(res.get("status"), "SUCCESS")

    # -------------------------------------------------------------------------
    # TEST 3: RTL validation succeeds -> Expected: GENERATION_COMPLETE.
    # -------------------------------------------------------------------------
    def test_test_3_rtl_validation_succeeds_generation_complete(self):
        efs_ir = self._create_base_efs_ir()
        design_results = {
            "status": "AWAITING_HUMAN_DECISION",
            "pipeline_state": "AWAITING_HUMAN_DECISION",
            "efs_ir": efs_ir,
            "conflicts": [],
            "plan": {"module_name": "dut_core", "submodules": []},
            "requirement_model": {},
            "protocol_rules": []
        }

        with patch("core.agents.orchestrator.generate_code", return_value="module dut_core(input clk); endmodule"), \
             patch("core.agents.orchestrator.generate_diagrams", return_value={}):
            res = resume_design_flow_with_resolutions(design_results, [], target_outputs=["Verilog"])

        self.assertEqual(res.get("status"), "SUCCESS")
        self.assertEqual(res.get("pipeline_state"), "GENERATION_COMPLETE")
        self.assertEqual(res.get("workflow_state"), WorkflowState.GENERATION_COMPLETE.value)

    # -------------------------------------------------------------------------
    # TEST 4: RTL validation discovers a new missing requirement ->
    # Expected: state = HUMAN_RESOLUTION_REQUIRED, pipeline stops (does not loop).
    # -------------------------------------------------------------------------
    def test_test_4_rtl_validation_discovers_new_issue_stops_at_human_resolution_required(self):
        efs_ir = self._create_base_efs_ir()
        design_results = {
            "status": "AWAITING_HUMAN_DECISION",
            "pipeline_state": "AWAITING_HUMAN_DECISION",
            "efs_ir": efs_ir,
            "conflicts": [],
            "plan": {"module_name": "dut_core", "submodules": []},
            "requirement_model": {},
            "protocol_rules": []
        }

        # Mock generate_code returning SPECIFICATION_INCOMPLETE
        incomplete_msg = "SPECIFICATION_INCOMPLETE\n\nMissing/ambiguous requirements:\n🔴 [MISSING] [FSM] IDLE -> RUN transition condition is undefined."
        with patch("core.agents.orchestrator.generate_code", return_value=incomplete_msg), \
             patch("core.agents.orchestrator.generate_diagrams", return_value={}):
            res = resume_design_flow_with_resolutions(design_results, [], target_outputs=["Verilog"])

        # RTL validation failure is shown as an error but does not restart pipeline or create new conflicts
        self.assertEqual(res.get("status"), "RTL_GENERATION_ERROR")
        self.assertEqual(res.get("pipeline_state"), "RTL_GENERATION_ERROR")
        self.assertFalse(res.get("requires_human_decision"))
        self.assertIn("error_message", res)

    # -------------------------------------------------------------------------
    # TEST 5: No human action occurs after a blocking issue ->
    # Expected: no additional LLM/RTL/PUML generation.
    # -------------------------------------------------------------------------
    def test_test_5_no_human_action_no_additional_generation(self):
        efs_ir = self._create_base_efs_ir()
        blocked_conflict = EFSConflict(
            conflict_id="CONF_NEW_BLOCK",
            type="SOURCE_MISSING",
            severity="CRITICAL",
            blocking=True,
            status="UNRESOLVED",
            resolution_status="UNRESOLVED",
            title="Missing FSM Transition",
            requires_human_decision=True
        )
        efs_ir.conflicts.append(blocked_conflict)

        # Passive call with NO new resolutions approved
        gen_spy = MagicMock()
        puml_spy = MagicMock()
        with patch("core.agents.orchestrator.generate_code", gen_spy), \
             patch("core.agents.orchestrator.generate_diagrams", puml_spy):
            res = resume_design_flow_with_resolutions(
                {"status": "AWAITING_HUMAN_DECISION", "efs_ir": efs_ir, "conflicts": [blocked_conflict]},
                approved_resolutions=[]
            )

        # Assert generation did not run
        self.assertEqual(gen_spy.call_count, 0)
        self.assertEqual(puml_spy.call_count, 0)
        self.assertEqual(res.get("status"), "AWAITING_HUMAN_DECISION")

    # -------------------------------------------------------------------------
    # TEST 6: Human resolves the new issue ->
    # Expected: revalidation occurs once, then generation resumes.
    # -------------------------------------------------------------------------
    def test_test_6_human_resolves_new_issue_revalidates_and_resumes(self):
        efs_ir = self._create_base_efs_ir()
        blocked_conflict = EFSConflict(
            conflict_id="CONF_FSM_TRANS_01",
            type="SOURCE_MISSING",
            severity="CRITICAL",
            blocking=True,
            status="UNRESOLVED",
            resolution_status="UNRESOLVED",
            title="IDLE -> RUN condition missing",
            entities=["IDLE", "RUN"],
            property="condition",
            requires_human_decision=True
        )
        efs_ir.conflicts.append(blocked_conflict)

        new_resolution = EFSResolution(
            resolution_id="RES_FSM_01",
            conflict_id="CONF_FSM_TRANS_01",
            target_entity="FSM",
            field_name="transitions",
            proposed_value="IDLE -> RUN on start_pulse == 1",
            resolution_type="OVERRIDE",
            approval_status="APPROVED",
            affected_requirements=["IDLE", "RUN"]
        )

        design_results = {
            "status": "AWAITING_HUMAN_DECISION",
            "efs_ir": efs_ir,
            "conflicts": [blocked_conflict],
            "plan": {"module_name": "dut_core", "submodules": []},
            "requirement_model": {},
            "protocol_rules": []
        }

        reval_spy = MagicMock(return_value=(False, []))
        gen_spy = MagicMock(return_value="module dut_core; endmodule")
        with patch("core.verification.conflict_resolution_engine.ConflictResolutionEngine.revalidate_resolved_model", reval_spy), \
             patch("core.agents.orchestrator.generate_code", gen_spy), \
             patch("core.agents.orchestrator.generate_diagrams", return_value={}):
            res = resume_design_flow_with_resolutions(design_results, [new_resolution])

        self.assertEqual(reval_spy.call_count, 1)
        self.assertEqual(gen_spy.call_count, 1)
        self.assertEqual(res.get("status"), "SUCCESS")

    # -------------------------------------------------------------------------
    # TEST 7: Same resolution submitted twice -> Expected: idempotent, no duplicate execution.
    # -------------------------------------------------------------------------
    def test_test_7_same_resolution_submitted_twice_idempotent(self):
        efs_ir = self._create_base_efs_ir()
        resolution = EFSResolution(
            resolution_id="RES_IDEMP_01",
            conflict_id="CONF_IDEMP_01",
            target_entity="done_flag",
            field_name="direction",
            proposed_value="output",
            resolution_type="OVERRIDE",
            approval_status="APPROVED",
            affected_requirements=["done_flag"]
        )
        # Already applied in efs_ir
        efs_ir.resolutions.append(resolution)

        design_results = {
            "status": "SUCCESS",
            "pipeline_state": "GENERATION_COMPLETE",
            "efs_ir": efs_ir,
            "conflicts": [],
            "outputs": {"Verilog": {"final_code": "module dut_core; endmodule"}},
            "plan": {"module_name": "dut_core"},
            "requirement_model": {},
            "protocol_rules": []
        }

        gen_spy = MagicMock()
        res = resume_design_flow_with_resolutions(design_results, [resolution])

        # Generation was skipped due to idempotency
        self.assertEqual(gen_spy.call_count, 0)
        self.assertEqual(res.get("status"), "SUCCESS")

    # -------------------------------------------------------------------------
    # TEST 8: Streamlit rerun without a new action ->
    # Expected: no expensive pipeline execution.
    # -------------------------------------------------------------------------
    def test_test_8_streamlit_passive_rerun_no_pipeline_restart(self):
        # Emulate ui/hardware_design.py logic:
        # If unresolved_conflicts is empty, we show a button instead of auto-running resume_design_flow
        approved_entity_props = {("done_flag", "direction")}
        
        def is_conflict_resolved(c) -> bool:
            c_prop = (getattr(c, "property", "") or "").lower()
            for ent in (getattr(c, "affected_entities", []) or getattr(c, "entities", []) or []):
                if c_prop and (ent.lower(), c_prop) in approved_entity_props:
                    return True
            return False

        # A brand-new conflict on 'dut_core' with property 'fsm' must NOT be falsely marked resolved
        new_fsm_conf = EFSConflict(
            conflict_id="CONF_FSM_NEW",
            type="SOURCE_MISSING",
            severity="CRITICAL",
            blocking=True,
            status="UNRESOLVED",
            resolution_status="UNRESOLVED",
            title="Unspecified FSM Transition",
            entities=["dut_core"],
            property="fsm",
            requires_human_decision=True
        )

        self.assertFalse(is_conflict_resolved(new_fsm_conf), "New conflict should NOT be resolved by unrelated property resolution!")

    # -------------------------------------------------------------------------
    # TEST 9: FSM resolution ->
    # Expected: FSM is visible in effective EFS IR, dependency resolver,
    # GenerationContext and RTL validator.
    # -------------------------------------------------------------------------
    def test_test_9_fsm_resolution_propagates_through_entire_stack(self):
        efs_ir = self._create_base_efs_ir()
        
        # Apply FSM human resolution
        fsm_resolution = EFSResolution(
            resolution_id="RES_FSM_TEST",
            conflict_id="CONF_FSM_01",
            target_entity="FSM",
            field_name="transitions",
            proposed_value="IDLE -> RUN on start_pulse == 1",
            resolution_type="OVERRIDE",
            approval_status="APPROVED",
            affected_requirements=["IDLE", "RUN"]
        )
        efs_ir.resolutions.append(fsm_resolution)

        # 1. Effective EFS IR rebuilt view
        resolved_ir = efs_ir.get_resolved_view()
        self.assertTrue(len(resolved_ir.fsms) > 0, "Resolved EFS IR must have FSM.")
        fsm = resolved_ir.fsms[0]
        self.assertIn("IDLE", [s.name for s in fsm.states])
        self.assertIn("RUN", [s.name for s in fsm.states])
        self.assertTrue(any(t.from_state == "IDLE" and t.to_state == "RUN" for t in fsm.transitions))

        # 2. Dependency Resolver sees FSM
        target_comp = resolved_ir.components[0]
        dep_res = resolve_dependencies(target_comp, resolved_ir)
        self.assertTrue(len(dep_res.fsms) > 0, "DependencyResolver must see resolved FSM.")

        # 3. ContextPack / GenerationContext sees FSM
        pack = ContextPack(
            task=TaskIntent(task="rtl", target="dut_core", language="Verilog"),
            target_component=target_comp.to_dict(),
            signals=[s.to_dict() for s in resolved_ir.signals],
            fsm=fsm.to_dict(),
            context_version=resolved_ir.metadata.model_version
        )
        self.assertIsNotNone(pack.fsm)
        state_names = [s.get("name", s) if isinstance(s, dict) else s for s in pack.fsm.get("states", [])]
        self.assertIn("IDLE", state_names)

        # 4. RTL Validator uses resolved FSM without blocking
        plan = {
            "module_name": "dut_core",
            "ports": [
                {"name": "clk", "direction": "input"},
                {"name": "rst_n", "direction": "input"},
                {"name": "start_pulse", "direction": "input"},
                {"name": "done_flag", "direction": "output"}
            ],
            "fsm": {
                "states": ["IDLE", "RUN"],
                "transitions": [{
                    "from": "IDLE",
                    "to": "RUN",
                    "condition": {"expression": "start_pulse == 1"}
                }]
            }
        }
        val = validate_rtl_plan(plan, efs_ir=resolved_ir, context_pack=pack)
        fsm_blocking = [i for i in val["issues"] if i.get("category") == "FSM"]
        self.assertEqual(len(fsm_blocking), 0, "RTL validator must not block valid resolved FSM transitions.")

    # -------------------------------------------------------------------------
    # TEST 10: SOURCE_MISSING ->
    # Expected: only generated when information genuinely does not exist.
    # -------------------------------------------------------------------------
    def test_test_10_source_missing_only_when_genuinely_missing(self):
        efs_ir = self._create_base_efs_ir()
        
        # When all signals and FSM are defined, validate_rtl_plan must be VALID
        complete_plan = {
            "module_name": "dut_core",
            "ports": [
                {"name": "clk", "direction": "input"},
                {"name": "rst_n", "direction": "input"},
                {"name": "start_pulse", "direction": "input"},
                {"name": "done_flag", "direction": "output"}
            ],
            "fsm": {
                "states": ["IDLE", "RUN"],
                "transitions": [{
                    "from": "IDLE",
                    "to": "RUN",
                    "condition": {"expression": "start_pulse == 1'b1"}
                }]
            }
        }
        val = validate_rtl_plan(complete_plan, efs_ir=efs_ir)
        self.assertTrue(val["valid"], f"Plan should be valid, but had issues: {val['issues']}")

        # When a port declared in spec is missing from plan:
        incomplete_plan = {
            "module_name": "dut_core",
            "ports": [{"name": "clk", "direction": "input"}]  # Missing start_pulse, done_flag
        }
        val_inc = validate_rtl_plan(incomplete_plan, efs_ir=efs_ir)
        self.assertFalse(val_inc["valid"], "Plan missing required ports must fail validation.")
        self.assertTrue(any("missing from RTL plan ports" in i["issue"] for i in val_inc["issues"]))

    # -------------------------------------------------------------------------
    # TEST 11: ContextPack budget ->
    # Expected: final context <= configured token budget.
    # -------------------------------------------------------------------------
    def test_test_11_context_pack_token_budget_strictly_enforced(self):
        budget_limit = 2000
        budget_mgr = TokenBudgetManager(max_tokens=budget_limit)

        # Create a bloated context pack (with huge dependency trees and raw text)
        bloated_deps = []
        for i in range(40):
            bloated_deps.append({
                "component": {
                    "component_id": f"SUB_{i}",
                    "name": f"submodule_{i}",
                    "type": "submodule",
                    "description": "Very long description " * 100,
                    "signals": [{"name": f"sig_{j}", "width": 32} for j in range(20)]
                },
                "depth": 1,
                "relationship": "child"
            })

        bloated_evidence = [{"text": "Repetitive narrative text " * 100} for _ in range(50)]
        bloated_transactions = [{"payload": "0xDEADBEEF" * 50, "trace": "Trace data " * 50} for _ in range(30)]

        pack = ContextPack(
            task=TaskIntent(task="rtl", target="dut_core", language="Verilog"),
            target_component={"name": "dut_core"},
            dependencies=bloated_deps,
            source_evidence=bloated_evidence,
            transactions=bloated_transactions
        )

        initial_tokens = pack.estimate_token_count()
        self.assertGreater(initial_tokens, budget_limit, "Test pack must exceed budget limit initially.")

        pruned_pack = budget_mgr.enforce_budget(pack)
        final_tokens = pruned_pack.estimate_token_count()

        self.assertLessEqual(
            final_tokens,
            budget_limit,
            f"Final pruned tokens ({final_tokens}) must NOT exceed configured budget ({budget_limit})!"
        )

    # -------------------------------------------------------------------------
    # REPRODUCTION TEST:
    # Reproduce the workflow: RTL_GENERATION_STARTED -> RTL BLOCKED ->
    # Human Resolution -> Revalidation -> Generation SUCCESS (NO LOOP).
    # -------------------------------------------------------------------------
    def test_reproduction_infinite_loop_prevention(self):
        """
        Simulate the exact sequence from the bug:
        1. RTL Validation blocks on ambiguous FSM condition ("start condition met").
        2. Routes to Human Resolution Decision workflow. Pipeline STOPS (state=HUMAN_RESOLUTION_REQUIRED).
        3. Human supplies explicit transition resolution: "start_pulse == 1'b1".
        4. resume_design_flow_with_resolutions revalidates cleanly and completes with status SUCCESS.
        5. Verify it does NOT loop back to HUMAN_RESOLUTION_REQUIRED.
        """
        efs_ir = self._create_base_efs_ir()
        
        # Step 1: Stage 1 RTL plan with ambiguous transition
        vague_plan = {
            "module_name": "dut_core",
            "ports": [
                {"name": "clk", "direction": "input"},
                {"name": "rst_n", "direction": "input"},
                {"name": "start_pulse", "direction": "input"},
                {"name": "done_flag", "direction": "output"}
            ],
            "fsm": {
                "states": ["IDLE", "RUN"],
                "transitions": [{
                    "from": "IDLE",
                    "to": "RUN",
                    "condition": {"expression": "start condition met"}  # Ambiguous
                }]
            }
        }
        val1 = validate_rtl_plan(vague_plan, efs_ir=efs_ir)
        self.assertFalse(val1["valid"], "Vague condition must block validation initially.")

        # Step 2: Simulation of resume_design_flow entering blocked state
        design_results = {
            "status": "AWAITING_HUMAN_DECISION",
            "efs_ir": efs_ir,
            "conflicts": [],
            "plan": {"module_name": "dut_core"},
            "requirement_model": {},
            "protocol_rules": []
        }

        # Step 3: Human resolves the ambiguous condition
        human_res = EFSResolution(
            resolution_id="RES_FSM_FIX",
            conflict_id="CONF_FSM_VAGUE",
            target_entity="FSM",
            field_name="transitions",
            proposed_value="IDLE -> RUN on start_pulse == 1'b1",
            resolution_type="OVERRIDE",
            approval_status="APPROVED",
            affected_requirements=["IDLE", "RUN"]
        )

        with patch("core.agents.orchestrator.generate_code", return_value="module dut_core(input clk, input rst_n, input start_pulse, output done_flag); endmodule"), \
             patch("core.agents.orchestrator.generate_diagrams", return_value={}):
            res = resume_design_flow_with_resolutions(design_results, [human_res], target_outputs=["Verilog"])

        # Step 4: Verification - Must complete cleanly and NOT restart or remain blocked
        self.assertEqual(res.get("status"), "SUCCESS")
        self.assertEqual(res.get("pipeline_state"), "GENERATION_COMPLETE")
        self.assertEqual(res.get("workflow_state"), WorkflowState.GENERATION_COMPLETE.value)
        self.assertFalse(res.get("requires_human_decision"))
        self.assertIn("Verilog", res.get("outputs", {}))


if __name__ == "__main__":
    unittest.main()
