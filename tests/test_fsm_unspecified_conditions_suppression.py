"""
Regression and verification test suite for FSM transition conflict suppression in EFS IR.

Verifies:
1. Inferred transitions and transitions with absent/empty conditions do NOT create blocking conflicts.
2. Contradictory transition conditions explicitly stated in specification are still reported as blocking conflicts.
3. Absent conditions are NOT automatically replaced with 1'b1 (preserved as empty / unresolved).
4. Explicitly defined conditions and their provenance remain unchanged.
5. Genuine duplicate opcode conflicts (including Matrix unit opcode collisions) are still detected.
6. Unresolved transitions are safely handled by RTL-generation validation (validate_rtl_plan) rather than source conflict queue.
7. Behavior works for arbitrary FSM state names and multiple hardware specifications.
8. Genuine structural errors (invalid state references, conflicting state definitions) are detected.
"""

import pytest
from core.efs_ir.models import (
    EFSIR,
    EFSComponent,
    EFSSignal,
    EFSFSM,
    EFSState,
    EFSTransition,
    EFSOpcode,
    EFSTraceability,
    EFSConflict,
    DiagnosticType
)
from core.efs_ir.builder import EFSIRBuilder
from core.verification.conflict_resolution_engine import ConflictResolutionEngine
from core.agents.rtl_agent import validate_rtl_plan


def test_1_inferred_transition_no_condition_not_blocking():
    """1. Inferred transition with no condition does not create a blocking conflict."""
    ir = EFSIR()
    ir.metadata.design_name = "pipeline_processor"
    ir.components = [EFSComponent(name="pipeline_processor", type="top_level")]
    ir.signals = [
        EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
        EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset")
    ]

    # Pipeline FSM: IDLE -> FETCH -> DECODE -> DISPATCH -> WAIT -> DONE -> ERROR -> IDLE
    # All transitions have empty conditions and are marked 'unresolved'
    states = ["IDLE", "FETCH", "DECODE", "DISPATCH", "WAIT", "DONE", "ERROR"]
    fsm = EFSFSM(
        name="pipeline_fsm",
        states=[EFSState(name=s) for s in states],
        initial_state="IDLE",
        transitions=[
            EFSTransition(source_state="IDLE", target_state="FETCH", condition="", condition_type="unresolved"),
            EFSTransition(source_state="FETCH", target_state="DECODE", condition="", condition_type="unresolved"),
            EFSTransition(source_state="DECODE", target_state="DISPATCH", condition="", condition_type="unresolved"),
            EFSTransition(source_state="DISPATCH", target_state="WAIT", condition="", condition_type="unresolved"),
            EFSTransition(source_state="WAIT", target_state="DONE", condition="", condition_type="unresolved"),
            EFSTransition(source_state="DONE", target_state="ERROR", condition="", condition_type="unresolved"),
            EFSTransition(source_state="ERROR", target_state="IDLE", condition="", condition_type="unresolved"),
        ]
    )
    ir.fsms.append(fsm)

    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    blocking_conflicts = [c for c in conflicts if getattr(c, "blocking", True)]
    # Must NOT have any blocking conflicts solely for unspecified conditions
    assert len(blocking_conflicts) == 0, f"Expected 0 blocking conflicts, got: {[c.title for c in blocking_conflicts]}"

    # Verify no 'FSM Transition Undefined' blocking conflict in conflict queue
    undefined_confs = [c for c in conflicts if "Transition Undefined" in getattr(c, "title", "")]
    assert len(undefined_confs) == 0


def test_2_contradictory_transition_conditions_reported():
    """2. Explicitly specified transitions with contradictory conditions are still reported."""
    ir = EFSIR()
    ir.metadata.design_name = "arbiter"
    ir.components = [EFSComponent(name="arbiter", type="top_level")]
    ir.signals = [
        EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
        EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset"),
        EFSSignal(name="req", direction="input", width="1")
    ]

    # Two transitions between same states with contradictory conditions
    fsm = EFSFSM(
        name="arbiter_fsm",
        states=[EFSState(name="IDLE"), EFSState(name="GRANT")],
        transitions=[
            EFSTransition(source_state="IDLE", target_state="GRANT", condition="req == 1'b1", condition_type="boolean"),
            EFSTransition(source_state="IDLE", target_state="GRANT", condition="req == 1'b0", condition_type="boolean")
        ]
    )
    ir.fsms.append(fsm)

    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    contradictory_confs = [
        c for c in conflicts
        if c.property == "transition_condition" and ("conflicting" in c.type.lower() or "conflicting" in c.title.lower())
    ]
    assert len(contradictory_confs) >= 1
    conf = contradictory_confs[0]
    assert conf.blocking is True
    assert conf.severity == "CRITICAL"
    assert "req == 1'b1" in conf.conflicting_values
    assert "req == 1'b0" in conf.conflicting_values


def test_3_absent_condition_not_replaced_with_1b1():
    """3. An absent condition is not automatically replaced with 1'b1."""
    req_model = {
        "fsm_info": {
            "name": "custom_controller",
            "states": [{"name": "ST_INIT"}, {"name": "ST_RUN"}],
            "transitions": [{"from": "ST_INIT", "to": "ST_RUN"}]  # Missing condition
        }
    }
    builder = EFSIRBuilder(req_model=req_model)
    built_ir = builder.build()

    tr = built_ir.fsms[0].transitions[0]
    # Condition must NOT be fabricated as 1'b1 or true
    assert tr.condition != "1'b1"
    assert tr.condition != "1"
    assert tr.condition != "true"
    assert tr.condition == ""
    assert tr.condition_type == "unresolved"


def test_4_explicit_defined_conditions_remain_unchanged():
    """4. Explicitly defined conditions and their provenance remain unchanged."""
    ir = EFSIR()
    ir.metadata.design_name = "comm_unit"
    ir.components = [EFSComponent(name="comm_unit", type="top_level")]
    ir.signals = [
        EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
        EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset"),
        EFSSignal(name="pkt_valid", direction="input", width="1")
    ]

    custom_trace = EFSTraceability(
        doc_id="DOC_SPEC_01",
        chunk_id="CHUNK_42",
        source_text="Advance to RX_PACKET when pkt_valid == 1'b1",
        extraction_method="manual_spec",
        confidence=1.0
    )
    tr = EFSTransition(
        source_state="RX_IDLE",
        target_state="RX_PACKET",
        condition="pkt_valid == 1'b1",
        condition_type="boolean",
        traceability=custom_trace
    )
    fsm = EFSFSM(
        name="comm_fsm",
        states=[EFSState(name="RX_IDLE"), EFSState(name="RX_PACKET")],
        transitions=[tr]
    )
    ir.fsms.append(fsm)

    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    tc_conflicts = [c for c in conflicts if c.property == "transition_condition"]
    assert len(tc_conflicts) == 0

    # Ensure explicit condition and provenance are strictly intact
    assert fsm.transitions[0].condition == "pkt_valid == 1'b1"
    assert fsm.transitions[0].traceability.chunk_id == "CHUNK_42"
    assert fsm.transitions[0].traceability.source_text == "Advance to RX_PACKET when pkt_valid == 1'b1"


def test_5_genuine_duplicate_opcode_conflicts_still_detected():
    """5. Genuine duplicate opcode conflicts are still detected."""
    ir = EFSIR()
    ir.metadata.design_name = "matrix_unit"
    ir.components = [EFSComponent(name="matrix_unit", type="top_level")]
    ir.signals = [
        EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
        EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset")
    ]

    # Matrix compute unit duplicate opcodes: MINV and MSTORE share 110000
    ir.opcodes = [
        EFSOpcode(mnemonic="MINV", binary_encoding="110000"),
        EFSOpcode(mnemonic="MSTORE", binary_encoding="110000"),
        EFSOpcode(mnemonic="MADD", binary_encoding="000001")
    ]

    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    opc_confs = [c for c in conflicts if c.type in ("DUPLICATE_OPCODE_ENCODING", "DUPLICATE_OPCODE")]
    assert len(opc_confs) >= 1, "Duplicate opcode collision MINV/MSTORE must be detected"
    conf = opc_confs[0]
    assert conf.blocking is True
    assert conf.severity == "CRITICAL"
    assert "MINV" in conf.entities and "MSTORE" in conf.entities


def test_6_unresolved_transitions_handled_by_rtl_validation():
    """6. Unresolved transitions are safely handled by RTL-generation validation (validate_rtl_plan)."""
    ir = EFSIR()
    ir.metadata.design_name = "pipeline"
    ir.components = [EFSComponent(name="pipeline", type="top_level")]
    ir.signals = [
        EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
        EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset")
    ]
    fsm = EFSFSM(
        name="pipeline_fsm",
        states=[EFSState(name="FETCH"), EFSState(name="EXECUTE")],
        transitions=[
            EFSTransition(source_state="FETCH", target_state="EXECUTE", condition="", condition_type="unresolved")
        ]
    )
    ir.fsms.append(fsm)

    # 1. Source conflict detector does not block on missing condition
    source_conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    blocking_src = [c for c in source_conflicts if c.blocking and c.property == "transition_condition"]
    assert len(blocking_src) == 0

    # 2. RTL plan validation catches unresolved condition during synthesis planning
    incomplete_rtl_plan = {
        "module_name": "pipeline",
        "fsm": {
            "states": ["FETCH", "EXECUTE"],
            "transitions": [
                {"from": "FETCH", "to": "EXECUTE", "condition": ""}
            ]
        }
    }
    val = validate_rtl_plan(incomplete_rtl_plan, efs_ir=ir)
    assert val["valid"] is False
    assert val["status"] == "INCOMPLETE"
    assert any(i.get("classification") == "PLAN_MISSING" for i in val["issues"])
    assert any("undefined" in i.get("issue", "").lower() for i in val["issues"])


def test_7_arbitrary_fsm_names_and_multiple_specs():
    """7. Generic across arbitrary state names and diverse hardware specifications."""
    arbitrary_state_sequences = [
        # Sequence A: Crypto accelerator
        ["KEY_EXP", "SUB_BYTES", "SHIFT_ROWS", "MIX_COLS", "ADD_ROUND_KEY"],
        # Sequence B: Network packet processing
        ["PREAMBLE", "HEADER_PARSE", "PAYLOAD_CRC", "EGRESS_PUSH"],
        # Sequence C: Memory controller
        ["ACTIVATE", "READ_LATENCY", "BURST_DATA", "PRECHARGE"]
    ]

    for seq in arbitrary_state_sequences:
        ir = EFSIR()
        ir.metadata.design_name = f"design_{seq[0].lower()}"
        ir.components = [EFSComponent(name=f"comp_{seq[0].lower()}", type="top_level")]
        ir.signals = [
            EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
            EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset")
        ]

        transitions = [
            EFSTransition(source_state=seq[i], target_state=seq[i + 1], condition="", condition_type="unresolved")
            for i in range(len(seq) - 1)
        ]
        fsm = EFSFSM(
            name=f"fsm_{seq[0].lower()}",
            states=[EFSState(name=s) for s in seq],
            transitions=transitions
        )
        ir.fsms.append(fsm)

        conflicts = ConflictResolutionEngine.detect_conflicts(ir)
        blocking_confs = [c for c in conflicts if c.blocking and c.property == "transition_condition"]
        assert len(blocking_confs) == 0, f"Expected 0 blocking conflicts for {seq}, got {len(blocking_confs)}"


def test_8_structural_errors_invalid_state_and_conflicting_definitions():
    """8. Genuine structural errors (invalid state references and conflicting state definitions) are reported."""
    ir = EFSIR()
    ir.metadata.design_name = "structural_test"
    ir.components = [EFSComponent(name="structural_test", type="top_level")]
    ir.signals = [
        EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
        EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset")
    ]

    # Transition references undefined state 'NON_EXISTENT_STATE'
    fsm = EFSFSM(
        name="test_fsm",
        states=[
            EFSState(name="STATE_A", encoding="2'b00"),
            EFSState(name="STATE_A", encoding="2'b11"),  # Conflicting duplicate state definition
            EFSState(name="STATE_B")
        ],
        transitions=[
            EFSTransition(source_state="STATE_A", target_state="NON_EXISTENT_STATE", condition="start == 1'b1", condition_type="boolean")
        ]
    )
    ir.fsms.append(fsm)

    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    invalid_state_confs = [c for c in conflicts if c.type == "INVALID_STATE_REFERENCE"]
    assert len(invalid_state_confs) >= 1
    assert "NON_EXISTENT_STATE" in invalid_state_confs[0].entities

    conflicting_state_confs = [c for c in conflicts if c.type == "CONFLICTING_STATE_DEFINITION"]
    assert len(conflicting_state_confs) >= 1
    assert "STATE_A" in conflicting_state_confs[0].entities
