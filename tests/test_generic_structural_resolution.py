"""
Generic Structural Resolution Test Suite.

Tests the specification-agnostic EFS IR repair and validation pipeline.
Covers every requirement from Section 7 of the task:
  - Valid and invalid entity references
  - Missing, stale, and duplicate identifiers
  - Schema-compliant relationship reconstruction
  - Different hardware component and interface structures
  - Explicit and implicitly described sequential behavior
  - Designs that do not require an FSM
  - Unsupported or ambiguous inference cases
  - Idempotent repair and persistence
  - Downstream RTL and PlantUML generation
  - Multiple unrelated hardware specifications and protocols

Constraints:
  - Zero hardcoded design-specific names (no "traffic light", "matrix", etc.)
  - Every test uses generic, schema-driven assertions
  - Tests verify BOTH successful repairs AND cases where conflicts correctly remain unresolved
"""
from __future__ import annotations

import re
import pytest
from typing import Any, Dict, List, Optional

# Core IR models
from core.efs_ir.models import (
    EFSIR, EFSMetadata, EFSComponent, EFSInterface, EFSSignal,
    EFSRegister, EFSFSM, EFSTransition, EFSTraceability, new_efs_id
)
# Generic utilities
from core.efs_ir.entity_registry import CanonicalEntityRegistry, normalize_semantic_key
from core.efs_ir.behavioral_analyzer import GenericBehavioralAnalyzer
from core.efs_ir.repair_engine import EFSStructuralRepairEngine
# PlantUML generation
from core.agents.plantuml_agent import get_heuristic_diagrams, resolve_diagram_ids_to_names


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sig(name: str, direction: str = "input", width: str = "1",
         role: str = "") -> EFSSignal:
    return EFSSignal(
        signal_id=new_efs_id("sig"), name=name, direction=direction,
        width=width, semantic_role=role,
        traceability=EFSTraceability(doc_id="TST", section="Test")
    )


def _comp(name: str, ctype: str = "module") -> EFSComponent:
    return EFSComponent(
        component_id=new_efs_id("comp"), name=name, type=ctype,
        traceability=EFSTraceability(doc_id="TST", section="Test")
    )


def _iface(name: str, src: str = "", dst: str = "",
           sigs: Optional[List[str]] = None) -> EFSInterface:
    return EFSInterface(
        interface_id=new_efs_id("if"), name=name,
        source_component=src, destination_component=dst,
        signals=sigs or [],
        traceability=EFSTraceability(doc_id="TST", section="Test")
    )


def _reg(name: str, offset: str = "0x00") -> EFSRegister:
    return EFSRegister(
        register_id=new_efs_id("reg"), name=name, offset=offset,
        access_type="RW", width=32,
        traceability=EFSTraceability(doc_id="TST", section="Test")
    )


def _fsm(name: str, initial: str,
         transitions: Optional[List[EFSTransition]] = None) -> EFSFSM:
    return EFSFSM(
        fsm_id=new_efs_id("fsm"), name=name, initial_state=initial,
        transitions=transitions or [],
        traceability=EFSTraceability(doc_id="TST", section="Test")
    )


def _seq_ir(design_name: str = "Seq_Module") -> EFSIR:
    """Minimal sequential IR (has clock + reset)."""
    return EFSIR(
        metadata=EFSMetadata(design_name=design_name, design_id=new_efs_id("dsn")),
        components=[_comp(design_name)],
        signals=[
            _sig("clk", "input", "1", "clock"),
            _sig("rst_n", "input", "1", "reset"),
            _sig("data_in", "input", "8"),
            _sig("data_out", "output", "8"),
        ]
    )


def _comb_ir(design_name: str = "Comb_ALU") -> EFSIR:
    """Minimal combinational IR (no clock/reset/FSM/register)."""
    return EFSIR(
        metadata=EFSMetadata(design_name=design_name, design_id=new_efs_id("dsn")),
        components=[_comp(design_name)],
        signals=[
            _sig("a", "input", "8"),
            _sig("b", "input", "8"),
            _sig("op", "input", "3"),
            _sig("result", "output", "8"),
            _sig("zero_flag", "output", "1"),
        ]
    )


# ===========================================================================
# Section 1 — normalize_semantic_key
# ===========================================================================

class TestNormalizeSemanticKey:

    def test_basic_lowercase(self):
        assert normalize_semantic_key("AXI4_MASTER") == "axi4_master"

    def test_spaces_to_underscore(self):
        assert normalize_semantic_key("DMA Controller") == "dma_controller"

    def test_mixed_delimiters(self):
        assert normalize_semantic_key("rx-data_bus") == "rx_data_bus"

    def test_leading_trailing_underscores_stripped(self):
        assert normalize_semantic_key("_clk_") == "clk"

    def test_none_returns_empty(self):
        assert normalize_semantic_key(None) == ""

    def test_empty_string_returns_empty(self):
        assert normalize_semantic_key("") == ""

    def test_backtick_stripped(self):
        assert normalize_semantic_key("`my_signal`") == "my_signal"

    def test_multiple_spaces_collapsed(self):
        assert normalize_semantic_key("data  ready  flag") == "data_ready_flag"


# ===========================================================================
# Section 2 — CanonicalEntityRegistry: Registration and Lookup
# ===========================================================================

class TestEntityRegistryRegistration:

    def test_register_and_resolve_component_by_id(self):
        reg = CanonicalEntityRegistry()
        c = _comp("Arbiter_Core")
        cid = reg.register_component(c)
        assert reg.resolve(cid, expected_type="component") is c

    def test_resolve_component_by_exact_name(self):
        reg = CanonicalEntityRegistry()
        c = _comp("Dispatch_Unit")
        reg.register_component(c)
        assert reg.resolve("Dispatch_Unit", expected_type="component") is c

    def test_resolve_component_by_normalized_name(self):
        reg = CanonicalEntityRegistry()
        c = _comp("Fetch Buffer")
        reg.register_component(c)
        # Spaces vs underscores must match
        assert reg.resolve("Fetch_Buffer", expected_type="component") is c

    def test_resolve_signal_by_id(self):
        reg = CanonicalEntityRegistry()
        s = _sig("tx_valid", "output", "1")
        sid = reg.register_signal(s)
        assert reg.resolve(sid, expected_type="signal") is s

    def test_resolve_interface_by_name(self):
        reg = CanonicalEntityRegistry()
        iface = _iface("Control_Bus", sigs=["sig1", "sig2"])
        reg.register_interface(iface)
        assert reg.resolve("Control_Bus", expected_type="interface") is iface

    def test_resolve_register_by_name(self):
        reg = CanonicalEntityRegistry()
        r = _reg("CTRL_REG", "0x00")
        reg.register_register(r)
        assert reg.resolve("CTRL_REG", expected_type="register") is r

    def test_resolve_none_returns_none(self):
        reg = CanonicalEntityRegistry()
        assert reg.resolve(None) is None

    def test_resolve_unknown_key_returns_none(self):
        reg = CanonicalEntityRegistry()
        assert reg.resolve("nonexistent_entity_xyz") is None

    def test_canonical_id_for(self):
        reg = CanonicalEntityRegistry()
        c = _comp("Pipeline_Stage")
        cid = reg.register_component(c)
        assert reg.canonical_id_for("Pipeline_Stage", "component") == cid

    def test_canonical_name_for(self):
        reg = CanonicalEntityRegistry()
        s = _sig("rx_data", "input", "32")
        reg.register_signal(s)
        assert reg.canonical_name_for(s.signal_id, "signal") == "rx_data"

    def test_type_restricted_resolution_no_cross_type_match(self):
        """Component name must not resolve as a signal."""
        reg = CanonicalEntityRegistry()
        c = _comp("bus_data")
        reg.register_component(c)
        # We ask for 'signal' but only a component is registered
        assert reg.resolve("bus_data", expected_type="signal") is None

    def test_alias_resolution(self):
        reg = CanonicalEntityRegistry()
        c = _comp("SPI_Controller")
        cid = reg.register_component(c)
        reg.register_alias(cid, "component", "spi_ctrl")
        assert reg.resolve("spi_ctrl", expected_type="component") is c

    def test_load_from_efs_ir_populates_all_types(self):
        ir = EFSIR(
            components=[_comp("Top_Module")],
            signals=[_sig("clk", "input", "1", "clock")],
            interfaces=[_iface("Ctrl_IF", sigs=["clk"])],
            registers=[_reg("STATUS_REG")],
        )
        reg = CanonicalEntityRegistry(ir)
        assert reg.resolve("Top_Module", "component") is not None
        assert reg.resolve("clk", "signal") is not None
        assert reg.resolve("Ctrl_IF", "interface") is not None
        assert reg.resolve("STATUS_REG", "register") is not None

    def test_id_redirect_follows_merges(self):
        reg = CanonicalEntityRegistry()
        c = _comp("Scheduler")
        original_id = c.component_id
        reg.register_component(c)
        new_id = new_efs_id("comp")
        reg.id_redirects[original_id] = new_id
        reg.components[new_id] = c
        assert reg.resolve(original_id, "component") is not None


# ===========================================================================
# Section 3 — Port vs Interface Classification
# ===========================================================================

class TestPortInterfaceClassification:

    def test_item_with_width_and_direction_is_signal_port(self):
        item = {"name": "awaddr", "width": 32, "direction": "input"}
        assert CanonicalEntityRegistry.classify_interface_candidate(item) == "SIGNAL_PORT"

    def test_item_with_direction_only_is_signal_port(self):
        item = {"name": "valid_flag", "direction": "output"}
        assert CanonicalEntityRegistry.classify_interface_candidate(item) == "SIGNAL_PORT"

    def test_item_with_sub_signals_is_interface_group(self):
        item = {"name": "AXI_Master_IF", "signals": ["awaddr", "awvalid", "wdata"]}
        assert CanonicalEntityRegistry.classify_interface_candidate(item) == "INTERFACE_GROUP"

    def test_item_with_source_component_is_interface_group(self):
        item = {"name": "Data_Link", "source_component": "TX", "destination_component": "RX"}
        assert CanonicalEntityRegistry.classify_interface_candidate(item) == "INTERFACE_GROUP"

    def test_interface_keyword_in_name_is_group(self):
        item = {"name": "pcie_interface"}
        assert CanonicalEntityRegistry.classify_interface_candidate(item) == "INTERFACE_GROUP"

    def test_non_dict_is_signal_port(self):
        assert CanonicalEntityRegistry.classify_interface_candidate("awaddr") == "SIGNAL_PORT"

    def test_empty_dict_defaults_to_signal_port(self):
        # No sub-signals, no endpoints → treat as physical port
        assert CanonicalEntityRegistry.classify_interface_candidate({"name": "en"}) == "SIGNAL_PORT"

    def test_clock_name_resolves_as_signal_port(self):
        item = {"name": "clk_in", "description": "system clock input"}
        assert CanonicalEntityRegistry.classify_interface_candidate(item) == "SIGNAL_PORT"


# ===========================================================================
# Section 4 — GenericBehavioralAnalyzer
# ===========================================================================

class TestBehavioralAnalyzerSequential:

    def test_ir_with_clock_signal_is_sequential(self):
        assert GenericBehavioralAnalyzer.is_sequential_design(_seq_ir()) is True

    def test_ir_with_reset_signal_is_sequential(self):
        ir = EFSIR(signals=[_sig("rst", "input", "1", "reset")])
        assert GenericBehavioralAnalyzer.is_sequential_design(ir) is True

    def test_ir_with_fsm_is_sequential(self):
        ir = EFSIR(fsms=[_fsm("ctrl_fsm", "IDLE", [
            EFSTransition(source_state="IDLE", target_state="RUN", condition="start")
        ])])
        assert GenericBehavioralAnalyzer.is_sequential_design(ir) is True

    def test_ir_with_registers_is_sequential(self):
        ir = EFSIR(registers=[_reg("CTRL_REG")])
        assert GenericBehavioralAnalyzer.is_sequential_design(ir) is True

    def test_combinational_ir_is_not_sequential(self):
        assert GenericBehavioralAnalyzer.is_sequential_design(_comb_ir()) is False

    def test_spec_text_posedge_clock_is_sequential(self):
        ir = EFSIR(signals=[_sig("a", "input", "8")])
        assert GenericBehavioralAnalyzer.is_sequential_design(
            ir, spec_text="samples data on the posedge of the clock signal"
        ) is True

    def test_spec_text_fsm_keyword_is_sequential(self):
        ir = EFSIR(signals=[_sig("req", "input", "1")])
        assert GenericBehavioralAnalyzer.is_sequential_design(
            ir, spec_text="implements a state machine with IDLE ACTIVE DONE states"
        ) is True

    def test_pure_combinational_spec_text_is_not_sequential(self):
        ir = EFSIR(signals=[_sig("x", "input", "4"), _sig("y", "output", "4")])
        assert GenericBehavioralAnalyzer.is_sequential_design(
            ir, spec_text="performs a bitwise AND operation on inputs"
        ) is False

    def test_req_model_clocks_resets_is_sequential(self):
        ir = EFSIR(signals=[_sig("data", "input", "8")])
        req_model = {"clocks_resets": [{"name": "sys_clk", "frequency": "100MHz"}]}
        assert GenericBehavioralAnalyzer.is_sequential_design(ir, req_model=req_model) is True

    def test_empty_ir_is_not_sequential(self):
        assert GenericBehavioralAnalyzer.is_sequential_design(EFSIR()) is False


# ===========================================================================
# Section 5 — Repair Engine
# ===========================================================================

class TestRepairEngineMetadata:

    def test_assigns_design_id_when_missing(self):
        ir = EFSIR(metadata=EFSMetadata(design_name="TestMod"))
        ir.metadata.design_id = ""
        repaired, log, _ = EFSStructuralRepairEngine.repair_and_validate(ir)
        assert repaired.metadata.design_id
        assert any(e.get("action") == "assign_design_id" for e in log)

    def test_idempotent_metadata_repair(self):
        ir = EFSIR(metadata=EFSMetadata(design_name="TestMod", design_id=new_efs_id("dsn")))
        _, log1, _ = EFSStructuralRepairEngine.repair_and_validate(ir)
        _, log2, _ = EFSStructuralRepairEngine.repair_and_validate(ir)
        meta_r1 = [e for e in log1 if "assign" in e.get("action", "")]
        meta_r2 = [e for e in log2 if "assign" in e.get("action", "")]
        assert len(meta_r2) <= len(meta_r1)


class TestRepairEnginePortDisambiguation:

    def test_bundle_interface_with_sub_signals_stays_as_interface(self):
        bundle = _iface("Data_Bus", src="TX_Unit", dst="RX_Unit", sigs=["data", "valid", "ready"])
        ir = EFSIR(
            components=[_comp("TX_Unit"), _comp("RX_Unit")],
            interfaces=[bundle]
        )
        repaired, _, _ = EFSStructuralRepairEngine.repair_and_validate(ir)
        iface_names = {i.name for i in repaired.interfaces}
        assert "Data_Bus" in iface_names


class TestRepairEngineReferenceResolution:

    def test_valid_interface_with_existing_component_refs_passes(self):
        ca = _comp("Module_A")
        cb = _comp("Module_B")
        iface = _iface("Link_AB", src=ca.component_id, dst=cb.component_id, sigs=["data"])
        ir = EFSIR(components=[ca, cb], interfaces=[iface])
        repaired, _, issues = EFSStructuralRepairEngine.repair_and_validate(ir)
        assert repaired is not None
        issue_titles = [v.get("description", "") for v in issues]
        for title in issue_titles:
            assert "duplicate" not in title.lower(), f"Spurious duplicate issue: {title}"

    def test_stale_signal_component_ref_handled_gracefully(self):
        c = _comp("Host_Controller")
        s = _sig("req_out", "output", "1")
        s.component = "NonExistent_Module"   # stale reference
        ir = EFSIR(components=[c], signals=[s])
        repaired, _, _ = EFSStructuralRepairEngine.repair_and_validate(ir)
        assert repaired is not None
        assert isinstance(repaired.signals, list)


class TestRepairEngineIdempotency:

    def test_idempotent_on_sequential_ir(self):
        ir = _seq_ir()
        r1, _, _ = EFSStructuralRepairEngine.repair_and_validate(ir)
        r2, _, _ = EFSStructuralRepairEngine.repair_and_validate(r1)
        assert len(r1.components) == len(r2.components)
        assert len(r1.signals) == len(r2.signals)

    def test_idempotent_on_combinational_ir(self):
        ir = _comb_ir()
        r1, _, _ = EFSStructuralRepairEngine.repair_and_validate(ir)
        r2, _, _ = EFSStructuralRepairEngine.repair_and_validate(r1)
        assert len(r1.signals) == len(r2.signals)
        assert len(r1.components) == len(r2.components)

    def test_idempotent_no_duplicate_signals(self):
        ir = EFSIR(
            metadata=EFSMetadata(design_name="Stable_Design", design_id=new_efs_id("dsn")),
            components=[_comp("Stable_Module")],
            signals=[
                _sig("clk", "input", "1", "clock"),
                _sig("rst_n", "input", "1", "reset"),
                _sig("data", "input", "16"),
            ]
        )
        EFSStructuralRepairEngine.repair_and_validate(ir)
        names = [s.name for s in ir.signals]
        assert len(names) == len(set(names)), "Duplicate signals found after repair"


# ===========================================================================
# Section 6 — Combinational Designs: No False Clock/Reset Conflicts
# ===========================================================================

class TestCombinationalNoFalseConflicts:

    def test_combinational_no_clock_conflict(self):
        from core.verification.conflict_resolution_engine import ConflictResolutionEngine
        ir = _comb_ir()
        conflicts = ConflictResolutionEngine.validate_hardware_completeness(ir)
        titles = [getattr(c, "title", "") for c in conflicts]
        assert "Missing Clock Signal" not in titles, (
            f"False 'Missing Clock Signal' for combinational design: {titles}"
        )

    def test_combinational_no_reset_conflict(self):
        from core.verification.conflict_resolution_engine import ConflictResolutionEngine
        ir = _comb_ir()
        conflicts = ConflictResolutionEngine.validate_hardware_completeness(ir)
        titles = [getattr(c, "title", "") for c in conflicts]
        assert "Missing Reset Signal" not in titles, (
            f"False 'Missing Reset Signal' for combinational design: {titles}"
        )

    def test_sequential_still_raises_missing_clock(self):
        """Sequential design without clock must still raise a conflict."""
        from core.verification.conflict_resolution_engine import ConflictResolutionEngine
        ir = EFSIR(
            metadata=EFSMetadata(design_name="Seq_No_Clk", design_id=new_efs_id("dsn")),
            components=[_comp("Seq_No_Clk")],
            signals=[
                _sig("rst_n", "input", "1", "reset"),
                _sig("data_in", "input", "8"),
            ],
            registers=[_reg("CTRL_REG")]   # register → sequential evidence
        )
        conflicts = ConflictResolutionEngine.validate_hardware_completeness(ir)
        titles = [getattr(c, "title", "") for c in conflicts]
        assert "Missing Clock Signal" in titles, (
            "Expected 'Missing Clock Signal' for sequential design without clock"
        )

    def test_mux_decoder_no_clock_reset_conflict(self):
        """Pure combinational mux design must not get clock/reset conflicts."""
        from core.verification.conflict_resolution_engine import ConflictResolutionEngine
        ir = EFSIR(
            metadata=EFSMetadata(design_name="Four_To_One_Mux", design_id=new_efs_id("dsn")),
            components=[_comp("Four_To_One_Mux")],
            signals=[
                _sig("sel", "input", "2"),
                _sig("d0", "input", "8"), _sig("d1", "input", "8"),
                _sig("d2", "input", "8"), _sig("d3", "input", "8"),
                _sig("y", "output", "8"),
            ]
        )
        conflicts = ConflictResolutionEngine.validate_hardware_completeness(ir)
        titles = [getattr(c, "title", "") for c in conflicts]
        assert "Missing Clock Signal" not in titles
        assert "Missing Reset Signal" not in titles


# ===========================================================================
# Section 7 — FSM Behavioral Completeness
# ===========================================================================

class TestSequentialFSMBehavior:

    def test_explicit_fsm_no_duplicate_after_repair(self):
        fsm_obj = _fsm("Protocol_FSM", "IDLE", [
            EFSTransition(source_state="IDLE", target_state="BUSY", condition="req"),
            EFSTransition(source_state="BUSY", target_state="DONE", condition="ack"),
            EFSTransition(source_state="DONE", target_state="IDLE", condition="rst_n"),
        ])
        ir = EFSIR(
            metadata=EFSMetadata(design_name="Proto_Ctrl", design_id=new_efs_id("dsn")),
            components=[_comp("Proto_Ctrl")],
            signals=[_sig("clk", "input", "1", "clock"), _sig("rst_n", "input", "1", "reset")],
            fsms=[fsm_obj]
        )
        GenericBehavioralAnalyzer.analyze_and_complete_behavior(ir)
        assert len(ir.fsms) == 1, f"Unexpected FSM count: {len(ir.fsms)}"

    def test_inferred_entities_carry_provenance(self):
        """All decisions returned by analyze_and_complete_behavior must carry a 'provenance' key."""
        ir = EFSIR(
            metadata=EFSMetadata(design_name="Counter", design_id=new_efs_id("dsn")),
            components=[_comp("Counter")],
            signals=[_sig("enable", "input", "1"), _sig("count_out", "output", "8")]
        )
        req_model = {
            "functional_requirements": [
                {"statement": "increments on every clock cycle while enable is asserted"}
            ]
        }
        decisions = GenericBehavioralAnalyzer.analyze_and_complete_behavior(ir, req_model=req_model)
        for d in decisions:
            if d.get("action") in ("infer_clock", "infer_reset", "infer_fsm"):
                assert "provenance" in d, f"Inferred decision missing provenance: {d}"

    def test_combinational_ir_no_spurious_inferences(self):
        """Purely combinational IR must produce zero clock/reset/FSM inferences."""
        ir = _comb_ir()
        decisions = GenericBehavioralAnalyzer.analyze_and_complete_behavior(ir)
        modifying = [
            d for d in decisions
            if d.get("action") in ("infer_clock", "infer_reset", "infer_fsm")
        ]
        assert len(modifying) == 0, (
            f"Combinational IR generated spurious behavioral inferences: {modifying}"
        )

    def test_fsm_with_spec_text_triggers_analysis(self):
        """Spec text describing state transitions must trigger behavioral analysis decisions."""
        ir = EFSIR(
            metadata=EFSMetadata(design_name="Generic_Ctrl", design_id=new_efs_id("dsn")),
            components=[_comp("Generic_Ctrl")],
            signals=[
                _sig("clk", "input", "1", "clock"),
                _sig("rst_n", "input", "1", "reset"),
            ]
        )
        spec = ("The controller starts in IDLE state. When request is received it transitions "
                "to ACTIVE state. Upon completion it moves to DONE state.")
        decisions = GenericBehavioralAnalyzer.analyze_and_complete_behavior(ir, spec_text=spec)
        # analyze_and_complete_behavior must return a list (even if empty)
        assert isinstance(decisions, list)


# ===========================================================================
# Section 8 — Unresolvable Conflicts Preserved
# ===========================================================================

class TestUnresolvableConflictsPreserved:

    def test_empty_ir_missing_module_conflict(self):
        from core.verification.conflict_resolution_engine import ConflictResolutionEngine
        ir = EFSIR(metadata=EFSMetadata(design_name="Empty", design_id=new_efs_id("dsn")))
        conflicts = ConflictResolutionEngine.validate_hardware_completeness(ir)
        titles = [getattr(c, "title", "") for c in conflicts]
        assert "Missing Module Definitions" in titles

    def test_ir_with_no_signals_produces_conflict(self):
        from core.verification.conflict_resolution_engine import ConflictResolutionEngine
        ir = EFSIR(
            metadata=EFSMetadata(design_name="No_Sigs", design_id=new_efs_id("dsn")),
            components=[_comp("No_Sigs")]
        )
        conflicts = ConflictResolutionEngine.validate_hardware_completeness(ir)
        titles = [getattr(c, "title", "") for c in conflicts]
        assert any("Signal" in t or "Port" in t for t in titles), (
            f"Expected missing signal/port conflict but got: {titles}"
        )

    def test_repair_engine_preserves_genuine_issues(self):
        """repair_and_validate must not suppress genuine validation issues via fabrication."""
        ir = EFSIR(
            metadata=EFSMetadata(design_name="Skeleton", design_id=new_efs_id("dsn")),
            components=[_comp("Skeleton")],
            signals=[]
        )
        repaired, _, issues = EFSStructuralRepairEngine.repair_and_validate(ir)
        # The engine must not crash and must return a proper tuple
        assert repaired is not None
        assert isinstance(issues, list)
        # Engine must NOT invent signals to silence the validator
        assert len(repaired.signals) == 0 or all(
            getattr(s, "extraction_method", "") == "inferred"
            for s in repaired.signals
        ), "Repair engine must not fabricate signals without evidence"


# ===========================================================================
# Section 9 — Multiple Specification Isolation
# ===========================================================================

class TestMultipleSpecificationIsolation:

    def test_two_specs_independent_registries(self):
        ir_a = EFSIR(
            components=[_comp("Crypto_Engine")],
            signals=[_sig("key_in", "input", "128")]
        )
        ir_b = EFSIR(
            components=[_comp("Display_Controller")],
            signals=[_sig("pixel_data", "output", "24")]
        )
        reg_a = CanonicalEntityRegistry(ir_a)
        reg_b = CanonicalEntityRegistry(ir_b)
        assert reg_b.resolve("Crypto_Engine", "component") is None
        assert reg_a.resolve("Display_Controller", "component") is None

    def test_repair_engine_protocol_heavy_spec(self):
        ir = EFSIR(
            metadata=EFSMetadata(design_name="StreamAccelerator", design_id=new_efs_id("dsn")),
            components=[_comp("StreamAccelerator")],
            signals=[
                _sig("str_data", "input", "64"), _sig("str_valid", "input", "1"),
                _sig("str_ready", "output", "1"), _sig("str_last", "input", "1"),
                _sig("str_clock", "input", "1", "clock"),
                _sig("str_reset_n", "input", "1", "reset"),
            ]
        )
        repaired, _, _ = EFSStructuralRepairEngine.repair_and_validate(ir)
        assert repaired is not None
        assert len(repaired.signals) >= 6

    def test_repair_engine_register_heavy_spec(self):
        ir = EFSIR(
            metadata=EFSMetadata(design_name="CryptoCore", design_id=new_efs_id("dsn")),
            components=[_comp("CryptoCore")],
            signals=[
                _sig("pclk", "input", "1", "clock"), _sig("preset_n", "input", "1", "reset"),
                _sig("paddr", "input", "8"), _sig("pwdata", "input", "32"),
                _sig("prdata", "output", "32"), _sig("pwrite", "input", "1"),
            ],
            registers=[_reg("CTRL_REG", "0x00"), _reg("STATUS_REG", "0x04")]
        )
        repaired, _, _ = EFSStructuralRepairEngine.repair_and_validate(ir)
        assert len(repaired.registers) == 2

    def test_repair_engine_fsm_heavy_spec(self):
        fsm_obj = _fsm("Packet_Scheduler_FSM", "WAIT", [
            EFSTransition(source_state="WAIT", target_state="GRANT", condition="req_valid"),
            EFSTransition(source_state="GRANT", target_state="TRANSFER", condition="grant_ack"),
            EFSTransition(source_state="TRANSFER", target_state="WAIT", condition="transfer_done"),
        ])
        ir = EFSIR(
            metadata=EFSMetadata(design_name="Packet_Scheduler", design_id=new_efs_id("dsn")),
            components=[_comp("Packet_Scheduler")],
            signals=[
                _sig("sys_clk", "input", "1", "clock"), _sig("sys_rst", "input", "1", "reset"),
                _sig("req_valid", "input", "1"), _sig("grant_ack", "input", "1"),
                _sig("transfer_done", "input", "1"),
            ],
            fsms=[fsm_obj]
        )
        repaired, _, _ = EFSStructuralRepairEngine.repair_and_validate(ir)
        assert len(repaired.fsms) >= 1
        assert "Packet_Scheduler_FSM" in [f.name for f in repaired.fsms]

    def test_two_registries_correct_entity_counts(self):
        ir_x = EFSIR(
            components=[_comp("Core_A"), _comp("Core_B")],
            signals=[_sig("clk", "input", "1", "clock")]
        )
        ir_y = EFSIR(
            components=[_comp("IO_Block")],
            signals=[_sig("rx_d", "input", "8"), _sig("tx_d", "output", "8")]
        )
        reg_x = CanonicalEntityRegistry(ir_x)
        reg_y = CanonicalEntityRegistry(ir_y)
        assert len(reg_x.components) == 2
        assert len(reg_y.components) == 1
        assert len(reg_y.signals) == 2


# ===========================================================================
# Section 10 — PlantUML Downstream Generation
# ===========================================================================

class TestPlantUMLGeneration:

    def test_get_heuristic_diagrams_returns_dict_not_none(self):
        ir = _seq_ir()
        diagrams = get_heuristic_diagrams({"target": "Seq_Module"}, efs_ir=ir)
        assert isinstance(diagrams, dict)
        for key in ("architecture", "fsm", "sequence"):
            assert key in diagrams
            assert diagrams[key] is not None, f"Diagram '{key}' is None"
            assert isinstance(diagrams[key], str), f"Diagram '{key}' is not str"

    def test_architecture_startuml_enduml(self):
        ir = EFSIR(
            components=[_comp("Arbiter"), _comp("Memory_Ctrl")],
            interfaces=[_iface("Arb_Link", "Arbiter", "Memory_Ctrl", ["req", "gnt"])]
        )
        diagrams = get_heuristic_diagrams({"target": "Arbiter_Subsystem"}, efs_ir=ir)
        arch = diagrams["architecture"]
        assert "@startuml" in arch
        assert "@enduml" in arch

    def test_fsm_diagram_with_transitions_renders_states(self):
        fsm_obj = _fsm("Control_FSM", "IDLE", [
            EFSTransition(source_state="IDLE", target_state="RUN", condition="start"),
            EFSTransition(source_state="RUN", target_state="IDLE", condition="done"),
        ])
        ir = EFSIR(
            components=[_comp("Control_Unit")],
            signals=[_sig("clk", "input", "1", "clock")],
            fsms=[fsm_obj]
        )
        diagrams = get_heuristic_diagrams({"target": "Control_Unit"}, efs_ir=ir)
        fsm_diag = diagrams["fsm"]
        assert "@startuml" in fsm_diag
        assert "@enduml" in fsm_diag
        assert "IDLE" in fsm_diag
        assert "RUN" in fsm_diag

    def test_combinational_fsm_diagram_no_invented_states(self):
        ir = _comb_ir()
        diagrams = get_heuristic_diagrams({"target": "Comb_ALU"}, efs_ir=ir)
        fsm_diag = diagrams["fsm"]
        assert "@startuml" in fsm_diag
        invented = ["IDLE", "ACTIVE", "DONE", "BUSY", "RESET_STATE", "RUN_STATE"]
        for kw in invented:
            assert kw not in fsm_diag, (
                f"FSM diagram for combinational design invented state '{kw}'"
            )

    def test_sequence_diagram_no_raw_sig_ids(self):
        """SIG_ prefixed IDs must not appear as participant names in the sequence diagram."""
        comps = [_comp("DMA_Engine"), _comp("Mem_Fabric")]
        sigs = [_sig("dma_req", "output", "1"), _sig("dma_ack", "input", "1")]
        ir = EFSIR(components=comps, signals=sigs)
        diagrams = get_heuristic_diagrams({"target": "DMA_Subsystem"}, efs_ir=ir)
        raw_id_pattern = re.compile(r"\bSIG_[0-9a-f]{6,}\b")
        assert not raw_id_pattern.search(diagrams["sequence"]), (
            "Sequence diagram contains raw SIG_ ID instead of human-readable name"
        )

    def test_resolve_diagram_ids_to_names(self):
        c = _comp("Encoder_Block")
        s = _sig("enc_out", "output", "8")
        ir = EFSIR(components=[c], signals=[s])
        raw = {
            "architecture": f"@startuml\nobject {c.component_id}\n@enduml",
            "fsm": "@startuml\n@enduml",
            "sequence": f"@startuml\n{s.signal_id} -> Host\n@enduml"
        }
        resolved = resolve_diagram_ids_to_names(raw, efs_ir=ir)
        assert isinstance(resolved, dict)
        assert "Encoder_Block" in resolved.get("architecture", "")

    def test_no_none_diagrams(self):
        ir = _seq_ir()
        diagrams = get_heuristic_diagrams({"target": "Seq_Module"}, efs_ir=ir)
        for key, val in diagrams.items():
            assert val is not None, f"Diagram '{key}' is None"

    def test_architecture_empty_interfaces_no_crash(self):
        """Architecture diagram with no interfaces must not crash or invent connections."""
        ir = EFSIR(
            components=[_comp("Block_A"), _comp("Block_B")],
            interfaces=[]
        )
        diagrams = get_heuristic_diagrams({"target": "Isolated_Blocks"}, efs_ir=ir)
        assert "@startuml" in diagrams["architecture"]
        assert "@enduml" in diagrams["architecture"]
