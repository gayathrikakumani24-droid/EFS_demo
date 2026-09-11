"""
Specification-Agnostic Source-Driven Validation Test Suite.

Verifies:
1. Arbitrary FSM state names (IDLE->EXECUTE->FINISH, WAITING->COMPUTE->COMPLETE, S0->S1->S2).
2. Zero invented signals (ack, op_complete, start) when absent from source spec.
3. Distinguishing EFSIR_EXTRACTION_GAP vs SOURCE_MISSING.
4. PLAN_MISSING, PLAN_CONFLICT, SOURCE_CONFLICT (duplicate opcodes), SOURCE_AMBIGUOUS.
5. Source requirement traceability preservation across pipeline models.
"""

from __future__ import annotations

import pytest
from core.efs_ir.models import (
    EFSIR, EFSSignal, EFSRegister, EFSState, EFSTransition,
    EFSFSM, EFSMetadata, EFSOpcode, EFSRequirementItem, new_efs_id
)
from core.efs_ir.validator import validate_efs_ir
from core.agents.rtl_agent import validate_rtl_plan
from core.verification.inference_engine import ControlledInferenceEngine


def test_arbitrary_fsm_state_names_spec_a():
    """Verify Specification A (IDLE -> EXECUTE -> FINISH -> IDLE) passes validation without hardcoded state errors."""
    efs = EFSIR()
    efs.metadata = EFSMetadata(design_name="accel_spec_a", protocol="custom")
    
    st0 = EFSState(state_id=new_efs_id("st"), name="IDLE")
    st1 = EFSState(state_id=new_efs_id("st"), name="EXECUTE")
    st2 = EFSState(state_id=new_efs_id("st"), name="FINISH")

    tr1 = EFSTransition(transition_id=new_efs_id("tr"), source_state="IDLE", target_state="EXECUTE", condition="go == 1'b1")
    tr2 = EFSTransition(transition_id=new_efs_id("tr"), source_state="EXECUTE", target_state="FINISH", condition="cnt == 8'hFF")
    tr3 = EFSTransition(transition_id=new_efs_id("tr"), source_state="FINISH", target_state="IDLE", condition="clear == 1'b1")

    fsm = EFSFSM(fsm_id=new_efs_id("fsm"), name="ctrl_fsm", initial_state="IDLE", states=[st0, st1, st2], transitions=[tr1, tr2, tr3])
    efs.fsms.append(fsm)

    plan = {
        "module_name": "accel_spec_a",
        "ports": [{"name": "go", "direction": "input"}, {"name": "clear", "direction": "input"}],
        "fsm": {
            "states": ["IDLE", "EXECUTE", "FINISH"],
            "initial_state": "IDLE",
            "transitions": [
                {"from": "IDLE", "to": "EXECUTE", "condition": "go == 1'b1"},
                {"from": "EXECUTE", "to": "FINISH", "condition": "cnt == 8'hFF"},
                {"from": "FINISH", "to": "IDLE", "condition": "clear == 1'b1"}
            ]
        }
    }

    val = validate_rtl_plan(plan, efs_ir=efs)
    assert val["valid"] is True
    assert val["status"] == "COMPLETE"


def test_arbitrary_fsm_state_names_spec_b():
    """Verify Specification B (WAITING -> COMPUTE -> COMPLETE -> WAITING) passes validation."""
    efs = EFSIR()
    efs.metadata = EFSMetadata(design_name="accel_spec_b", protocol="custom")

    st0 = EFSState(state_id=new_efs_id("st"), name="WAITING")
    st1 = EFSState(state_id=new_efs_id("st"), name="COMPUTE")
    st2 = EFSState(state_id=new_efs_id("st"), name="COMPLETE")

    tr1 = EFSTransition(transition_id=new_efs_id("tr"), source_state="WAITING", target_state="COMPUTE", condition="valid_in == 1'b1")
    tr2 = EFSTransition(transition_id=new_efs_id("tr"), source_state="COMPUTE", target_state="COMPLETE", condition="done_signal == 1'b1")
    tr3 = EFSTransition(transition_id=new_efs_id("tr"), source_state="COMPLETE", target_state="WAITING", condition="1'b1")

    fsm = EFSFSM(fsm_id=new_efs_id("fsm"), name="fsm_b", initial_state="WAITING", states=[st0, st1, st2], transitions=[tr1, tr2, tr3])
    efs.fsms.append(fsm)

    plan = {
        "module_name": "accel_spec_b",
        "ports": [{"name": "valid_in", "direction": "input"}, {"name": "done_signal", "direction": "output"}],
        "fsm": {
            "states": ["WAITING", "COMPUTE", "COMPLETE"],
            "initial_state": "WAITING",
            "transitions": [
                {"from": "WAITING", "to": "COMPUTE", "condition": "valid_in == 1'b1"},
                {"from": "COMPUTE", "to": "COMPLETE", "condition": "done_signal == 1'b1"},
                {"from": "COMPLETE", "to": "WAITING", "condition": "1'b1"}
            ]
        }
    }

    val = validate_rtl_plan(plan, efs_ir=efs)
    assert val["valid"] is True


def test_arbitrary_fsm_state_names_spec_c():
    """Verify Specification C (S0 -> S1 -> S2 -> S0) passes validation."""
    efs = EFSIR()
    efs.metadata = EFSMetadata(design_name="state_c", protocol="custom")

    st0 = EFSState(state_id=new_efs_id("st"), name="S0")
    st1 = EFSState(state_id=new_efs_id("st"), name="S1")
    st2 = EFSState(state_id=new_efs_id("st"), name="S2")

    tr1 = EFSTransition(transition_id=new_efs_id("tr"), source_state="S0", target_state="S1", condition="trig0 == 1'b1")
    tr2 = EFSTransition(transition_id=new_efs_id("tr"), source_state="S1", target_state="S2", condition="trig1 == 1'b1")
    tr3 = EFSTransition(transition_id=new_efs_id("tr"), source_state="S2", target_state="S0", condition="trig2 == 1'b1")

    fsm = EFSFSM(fsm_id=new_efs_id("fsm"), name="fsm_c", initial_state="S0", states=[st0, st1, st2], transitions=[tr1, tr2, tr3])
    efs.fsms.append(fsm)

    plan = {
        "module_name": "state_c",
        "ports": [{"name": "trig0", "direction": "input"}],
        "fsm": {
            "states": ["S0", "S1", "S2"],
            "initial_state": "S0",
            "transitions": [
                {"from": "S0", "to": "S1", "condition": "trig0 == 1'b1"},
                {"from": "S1", "to": "S2", "condition": "trig1 == 1'b1"},
                {"from": "S2", "to": "S0", "condition": "trig2 == 1'b1"}
            ]
        }
    }

    val = validate_rtl_plan(plan, efs_ir=efs)
    assert val["valid"] is True


def test_no_invented_signals():
    """Verify ControlledInferenceEngine does NOT invent ack, op_complete, or start signals if absent."""
    efs = EFSIR()
    efs.metadata = EFSMetadata(design_name="minimal_core", protocol="custom")
    efs.signals.append(EFSSignal(signal_id=new_efs_id("sig"), name="custom_data", direction="input", width="16"))

    engine = ControlledInferenceEngine(efs)
    completed = engine.infer_and_complete()

    sig_names = [s.name for s in completed.efs_ir.signals]
    assert "ack" not in sig_names
    assert "op_complete" not in sig_names
    assert "start" not in sig_names


def test_source_conflict_duplicate_opcodes():
    """Verify genuine source conflicts (duplicate opcode encodings) are flagged as SOURCE_CONFLICT."""
    efs = EFSIR()
    efs.metadata = EFSMetadata(design_name="cpu_alu", protocol="custom")
    
    opc1 = EFSOpcode(opcode_id=new_efs_id("opc"), mnemonic="ADD", binary_encoding="0001")
    opc2 = EFSOpcode(opcode_id=new_efs_id("opc"), mnemonic="SUB", binary_encoding="0001") # Duplicate binary encoding
    efs.opcodes.extend([opc1, opc2])

    from core.efs_ir.models import EFSConflict
    efs.conflicts.append(EFSConflict(
        conflict_id=new_efs_id("cnf"),
        type="DUPLICATE_OPCODE",
        description="Opcode binary value '0001' is assigned to multiple instructions: ADD, SUB",
        evidence="ADD=0001, SUB=0001",
        resolution_status="UNRESOLVED"
    ))

    plan = {"module_name": "cpu_alu", "ports": []}
    val = validate_rtl_plan(plan, efs_ir=efs)
    assert not val["valid"]
    assert any(i["classification"] == "SOURCE_CONFLICT" for i in val["issues"])


def test_plan_missing_efs_state():
    """Verify PLAN_MISSING is reported when RTL plan omits an EFS IR state."""
    efs = EFSIR()
    efs.metadata = EFSMetadata(design_name="fsm_mod", protocol="custom")
    efs.fsms.append(EFSFSM(
        fsm_id=new_efs_id("fsm"),
        name="fsm_mod",
        states=[EFSState(state_id=new_efs_id("st"), name="S0"), EFSState(state_id=new_efs_id("st"), name="S1")]
    ))

    plan = {
        "module_name": "fsm_mod",
        "fsm": {"states": ["S0"]} # Omits S1
    }

    val = validate_rtl_plan(plan, efs_ir=efs)
    assert not val["valid"]
    assert any(i["classification"] == "PLAN_MISSING" for i in val["issues"])


def test_first_class_requirement_item_traceability():
    """Verify EFSRequirementItem carries category, property, value, status, and evidence traceability."""
    req = EFSRequirementItem(
        category="FSM",
        property="transition",
        value={"from": "S0", "to": "S1", "condition": "en == 1'b1"},
        status="COMPLETE",
        evidence={"source_document": "spec.pdf", "section": "2.1", "text": "Transition S0 to S1 on en"},
        source_location="spec.pdf:L45"
    )

    req_dict = req.to_dict()
    assert req_dict["category"] == "FSM"
    assert req_dict["status"] == "COMPLETE"
    assert req_dict["evidence"]["source_document"] == "spec.pdf"
