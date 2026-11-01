"""
Regression tests for conflict generation and classification in the EFS pipeline.
Verifies proper separation between:
1. SPEC_CONFLICT (genuine specification contradiction)
2. MISSING_SPECIFICATION (missing requirement / unelaborated FSM)
3. TARGET_RESOLUTION_FAILED / PLANNER_ERROR (internal diagnostic, not a spec conflict)
4. Property semantic validation and prevention of false AI proposals (e.g. INTERFACE.offset = 0x04)
5. Deduplication across pipeline stages

All tests are synthetic, protocol-agnostic, and design-agnostic.
"""

import pytest
from core.efs_ir.models import EFSIR, EFSConflict, EFSSignal, EFSRegister, DiagnosticType
from core.retrieval.target_resolver import resolve_target, TargetResolutionResult
from core.retrieval.context_compiler import compile_task_context
from core.retrieval.context_pack import ContextPack
from core.verification.conflict_resolution_engine import ConflictResolutionEngine


class TestConflictClassificationAndDiagnostics:
    """Regression test suite for EFS pipeline conflict classification and diagnostic integrity."""

    def test_1_resolver_failure_not_spec_conflict(self):
        """
        Test 1 — Resolver failure:
        Input: TARGET_RESOLUTION_FAILED diagnostic.
        Expected: diagnostic_type == TARGET_RESOLUTION_FAILED, not SPEC_CONFLICT.
        """
        # Given a synthetic EFS IR with an unrelated component
        efs_ir = EFSIR()
        res = resolve_target("non_existent_accelerator_unit", efs_ir)
        assert not res.is_success()
        assert res.status == "TARGET_NOT_FOUND"
        assert res.diagnostic_details is not None
        assert res.diagnostic_details["type"] == "TARGET_RESOLUTION_FAILED"

        # ContextPack must hold diagnostic in diagnostics, NOT in conflicts
        pack, res_result = compile_task_context("synthesize non_existent_accelerator_unit", efs_ir)
        assert len(pack.conflicts) == 0, "Resolver failures must never be placed into pack.conflicts"
        assert len(pack.diagnostics) >= 1
        assert pack.diagnostics[0]["type"] == "TARGET_RESOLUTION_FAILED"

        # Parsing the failure text must NOT produce a SPEC_CONFLICT
        confs = ConflictResolutionEngine.parse_specification_issues_to_conflicts(res.error_message, efs_ir)
        for c in confs:
            assert c.diagnostic_type != DiagnosticType.SPEC_CONFLICT.value, (
                f"Resolver diagnostic was classified as SPEC_CONFLICT: {c}"
            )
            assert c.diagnostic_type == DiagnosticType.TARGET_RESOLUTION_FAILED.value
            assert c.blocking is False, "Diagnostics must not block HITL as spec conflicts"
            assert c.requires_human_decision is False

    def test_2_closest_candidates_not_spec_conflict(self):
        """
        Test 2 — Closest candidates:
        Input containing closest_candidates = [...].
        Expected: not SPEC_CONFLICT.
        """
        diag_snippet = (
            "SPECIFICATION_INCOMPLETE\n\n"
            "Missing/ambiguous requirements:\n"
            "    Closest Candidates: ['UnitAlpha', 'UnitBeta', 'UnitGamma']\n"
            "    Available EFS IR Components (3): ['UnitAlpha', 'UnitBeta', 'UnitGamma']\n\n"
            "Required clarification:\n"
            "    Specify target."
        )
        efs_ir = EFSIR()
        confs = ConflictResolutionEngine.parse_specification_issues_to_conflicts(diag_snippet, efs_ir)
        for c in confs:
            assert c.diagnostic_type != DiagnosticType.SPEC_CONFLICT.value
            assert "closest candidates" not in [v.lower() for v in c.conflicting_values]

    def test_3_canonical_id_and_aliases_not_spec_conflict(self):
        """
        Test 3 — Canonical ID / aliases:
        Input containing canonical_design_id, aliases, display_name.
        Expected: not SPEC_CONFLICT.
        """
        diag_snippet = (
            "SPECIFICATION_INCOMPLETE\n\n"
            "Missing/ambiguous requirements:\n"
            "    Design Canonical ID: 'EFS_synth_core'\n"
            "    Design Display Name: 'Synthetic Core Unit'\n"
            "    Registered Design Aliases: ['synth_core', 'core_v1']\n\n"
            "Required clarification:\n"
            "    Provide definition."
        )
        efs_ir = EFSIR()
        confs = ConflictResolutionEngine.parse_specification_issues_to_conflicts(diag_snippet, efs_ir)
        for c in confs:
            assert c.diagnostic_type != DiagnosticType.SPEC_CONFLICT.value
            assert c.property != "interface"

    def test_4_invalid_property_value_and_proposal_prevention(self):
        """
        Test 4 — Invalid property value:
        Attempt to create offset = 'Available EFS IR Components [...]'.
        Expected: validation failure, no semantic conflict, and no INTERFACE.offset = 0x04 proposal.
        """
        efs_ir = EFSIR()

        # 1. Semantic property validation directly rejects arbitrary diagnostic strings
        invalid_val = "Available EFS IR Components (48): ['COMP_A', 'COMP_B', 'COMP_C']"
        is_valid, msg = ConflictResolutionEngine.validate_property_value("INTERFACE", "offset", invalid_val, efs_ir)
        assert not is_valid, "Long diagnostic component list must be rejected as an offset"
        assert "not an addressable" in msg or "internal resolver" in msg or "diagnostic" in msg

        is_valid_reg, msg_reg = ConflictResolutionEngine.validate_property_value("STATUS_REG", "offset", invalid_val, efs_ir)
        assert not is_valid_reg, "Component dump must not pass as register offset"

        # 2. Valid offset passes for addressable entity
        is_valid_hex, _ = ConflictResolutionEngine.validate_property_value("STATUS_REG", "offset", "0x10", efs_ir)
        assert is_valid_hex

        # 3. AI Proposal generator MUST NEVER propose INTERFACE.offset = 0x04
        fake_conflict = EFSConflict(
            conflict_id="CONF_TEST_OFFSET",
            type="PLAN_CONFLICT",
            entities=["INTERFACE"],
            property="offset",
            conflicting_values=[invalid_val],
            diagnostic_type=DiagnosticType.SPEC_CONFLICT.value
        )
        proposals = ConflictResolutionEngine.generate_resolution_proposals(fake_conflict, efs_ir)
        for p in proposals:
            assert "INTERFACE.offset" not in p.proposed_change, (
                f"Generated illegal proposal on INTERFACE: {p.proposed_change}"
            )
        assert len(proposals) == 0, "No proposals should be generated for invalid entity/property combination"

    def test_5_missing_fsm_handling(self):
        """
        Test 5 — Missing FSM:
        Input: FSM definition absent.
        Expected: MISSING_SPECIFICATION (PLAN_MISSING/SOURCE_MISSING), NOT SPEC_CONFLICT(interface).
        """
        fsm_missing_snippet = (
            "SPECIFICATION_INCOMPLETE\n\n"
            "Missing/ambiguous requirements:\n"
            "    🔴 [PLAN_MISSING] [FSM] No finite state machine states, initial state, or transition conditions are specified. | Source: Compiled generation context\n\n"
            "Required clarification:\n"
            "    Provide explicit FSM state machine definitions."
        )
        efs_ir = EFSIR()
        confs = ConflictResolutionEngine.parse_specification_issues_to_conflicts(fsm_missing_snippet, efs_ir)
        assert len(confs) == 1
        c = confs[0]

        # Must be classified as MISSING_SPECIFICATION
        assert c.diagnostic_type == DiagnosticType.MISSING_SPECIFICATION.value
        assert c.diagnostic_type != DiagnosticType.SPEC_CONFLICT.value
        assert c.entities == ["FSM"]
        assert c.property != "interface"
        assert c.property == "fsm_definition"
        assert c.conflicting_values == [], "Missing specifications must not have conflicting values"
        assert "No finite state machine" in c.description, f"First letters were clipped: '{c.description}'"

    def test_6_genuine_specification_conflict_preserved(self):
        """
        Test 6 — Genuine conflict:
        Synthetic, document-independent example:
        Entity: synth_component_A, Property: instruction_width, Source 1: 32, Source 2: 64.
        Expected: SPEC_CONFLICT.
        """
        spec_conflict_snippet = (
            "SPECIFICATION_INCOMPLETE\n\n"
            "Missing/ambiguous requirements:\n"
            "    🟠 [SOURCE_CONFLICT] [SIGNAL] Signal width conflict on 'synth_data_bus': specification section 2 specifies '32', but table 4 specifies '64'. | Source: Synthetic Spec\n\n"
            "Required clarification:\n"
            "    Choose between width 32 and 64."
        )
        efs_ir = EFSIR()
        confs = ConflictResolutionEngine.parse_specification_issues_to_conflicts(spec_conflict_snippet, efs_ir)
        assert len(confs) == 1
        c = confs[0]

        assert c.diagnostic_type == DiagnosticType.SPEC_CONFLICT.value
        assert c.entities == ["synth_data_bus"]
        assert c.property == "width"
        assert "32" in c.conflicting_values and "64" in c.conflicting_values
        assert c.blocking is True
        assert c.requires_human_decision is True

    def test_7_genuine_interface_direction_conflict_preserved(self):
        """
        Test 7 — Genuine interface direction conflict:
        Generic example:
        Entity: synth_port_ready, Property: direction, Source 1: input, Source 2: output.
        Expected: SPEC_CONFLICT.
        """
        spec_direction_snippet = (
            "SPECIFICATION_INCOMPLETE\n\n"
            "Missing/ambiguous requirements:\n"
            "    🟠 [SOURCE_CONFLICT] [INTERFACE] Signal direction conflict on 'synth_tx_ready': RTL plan specifies 'input', but canonical EFS IR specifies 'output'. | Source: Synthetic Spec\n\n"
            "Required clarification:\n"
            "    Resolve port direction."
        )
        efs_ir = EFSIR()
        confs = ConflictResolutionEngine.parse_specification_issues_to_conflicts(spec_direction_snippet, efs_ir)
        assert len(confs) == 1
        c = confs[0]

        assert c.diagnostic_type == DiagnosticType.SPEC_CONFLICT.value
        assert c.entities == ["synth_tx_ready"]
        assert c.property == "direction"
        assert "input" in c.conflicting_values and "output" in c.conflicting_values
        assert c.blocking is True

    def test_8_deduplication_of_repeated_issues(self):
        """
        Test 8 — Duplicate conflict handling:
        Semantically identical issues received multiple times must not produce duplicate records.
        """
        repeated_snippet = (
            "SPECIFICATION_INCOMPLETE\n\n"
            "Missing/ambiguous requirements:\n"
            "    🔴 [PLAN_MISSING] [FSM] Transition 'STATE_A' -> 'STATE_B' condition is undefined. | Source: Synthetic Spec\n"
            "    🔴 [PLAN_MISSING] [FSM] Transition 'STATE_A' -> 'STATE_B' condition is undefined. | Source: Synthetic Spec duplicate pass\n\n"
            "Required clarification:\n"
            "    Define transition condition."
        )
        efs_ir = EFSIR()
        confs = ConflictResolutionEngine.parse_specification_issues_to_conflicts(repeated_snippet, efs_ir)
        assert len(confs) == 1, f"Expected 1 deduplicated conflict, got {len(confs)}"

    def test_9_conflicting_reset_definitions_classified_and_resolved(self):
        """
        Test 9 — Conflicting Reset:
        Input: Conflicting reset definitions exist: Reset is active_high while rst_n is active_low.
        Expected: Classified as SPEC_CONFLICT with property='reset', produces concrete candidates
        (active_low rst_n prioritized for AXI and active_high Reset), and cleanly mutates resolved_ir.
        """
        diag_snippet = (
            "SPECIFICATION_INCOMPLETE\n\n"
            "Missing/ambiguous requirements:\n"
            "    🔴 [PLAN_CONFLICT] [RESET] Conflicting reset definitions exist: Reset is active_high while rst_n is active_low. | Source: Hardware Spec\n\n"
            "Required clarification:\n"
            "    Single reset signal name, polarity, and synchronization requirements."
        )
        efs_ir = EFSIR()
        efs_ir.signals = [
            EFSSignal(name="Reset", direction="input", width=1, description="Asynchronous system reset, active-high"),
            EFSSignal(name="rst_n", direction="input", width=1, description="Active-low reset"),
            EFSSignal(name="data_in", direction="input", width=32)
        ]
        confs = ConflictResolutionEngine.parse_specification_issues_to_conflicts(diag_snippet, efs_ir)
        assert len(confs) == 1
        c = confs[0]

        assert c.diagnostic_type == DiagnosticType.SPEC_CONFLICT.value
        assert c.property == "reset"
        assert set(c.entities) == {"Reset", "rst_n"}
        assert c.blocking is True

        proposals = ConflictResolutionEngine.generate_resolution_proposals(c, efs_ir)
        assert len(proposals) >= 2
        # Proposal 1 is rst_n (active_low), Proposal 2 is Reset (active_high)
        p1 = proposals[0]
        assert "rst_n" in p1.proposed_change
        assert "active_low" in p1.proposed_change

        # Apply proposal 1
        resolved_ir = ConflictResolutionEngine.apply_approved_resolution(efs_ir, p1)
        sig_names = [s.name for s in resolved_ir.signals]
        assert "rst_n" in sig_names
        assert "Reset" not in sig_names
        assert "data_in" in sig_names

    def test_10_conflicting_clock_definitions_classified_and_resolved(self):
        """
        Test 10 — Conflicting Clock:
        Input: Conflicting clock definitions exist: Clock vs clk.
        Expected: Classified as SPEC_CONFLICT with property='clock', produces concrete candidates
        and mutates resolved_ir by keeping selected clock and removing superseded clock.
        """
        diag_snippet = (
            "SPECIFICATION_INCOMPLETE\n\n"
            "Missing/ambiguous requirements:\n"
            "    🔴 [PLAN_CONFLICT] [CLOCK] Conflicting clock definitions exist: Multiple clock signals defined: Clock, clk. | Source: Hardware Spec\n\n"
            "Required clarification:\n"
            "    Select primary system clock."
        )
        efs_ir = EFSIR()
        efs_ir.signals = [
            EFSSignal(name="Clock", direction="input", width=1),
            EFSSignal(name="clk", direction="input", width=1),
            EFSSignal(name="valid", direction="input", width=1)
        ]
        confs = ConflictResolutionEngine.parse_specification_issues_to_conflicts(diag_snippet, efs_ir)
        assert len(confs) == 1
        c = confs[0]

        assert c.diagnostic_type == DiagnosticType.SPEC_CONFLICT.value
        assert c.property == "clock"
        assert c.blocking is True

        proposals = ConflictResolutionEngine.generate_resolution_proposals(c, efs_ir)
        assert len(proposals) >= 2
        p_clk = next(p for p in proposals if "clk" in p.proposed_change and "Clock" not in p.proposed_change.split("=")[-1].strip())
        resolved_ir = ConflictResolutionEngine.apply_approved_resolution(efs_ir, p_clk)
        sig_names = [s.name for s in resolved_ir.signals]
        assert "clk" in sig_names
        assert "Clock" not in sig_names
        assert "valid" in sig_names

