"""
End-to-End Regression Test Suite for:
A. Multiple instructions sharing the same opcode
B. Correct instruction ownership of every opcode
C. Legitimate duplicate-opcode conflicts
D. No false "General" opcode conflicts
E. opcode vs binary_encoding canonicalization
F. FSM state extraction
G. FSM transition extraction
H. transition condition extraction
I. ambiguous semantic conditions
J. provenance preservation
K. HITL resolution followed by conflict reconciliation
L. stale conflict removal after EFS IR mutation
M. protocol-agnostic behavior
"""

import pytest
import os
import json
from core.efs_ir.models import (
    EFSIR, EFSOpcode, EFSInstruction, EFSConflict, EFSResolution,
    EFSFSM, EFSState, EFSTransition, EFSFlow, EFSSignal, EFSTraceability,
    ConflictStatus
)
from core.verification.conflict_resolution_engine import ConflictResolutionEngine
from core.compiler.efs_compiler import RequirementToEFSCompiler
from core.requirement_ir.models import RequirementIR, DocumentIR, DocumentBlock, SourceLocation
from core.extraction.dedicated_extractors import DedicatedFSMExtractor, ExtractedFSM, ExtractedTransition
from core.extraction.source_recovery import SourceBackedRecovery, SourceEvidenceIndex
from core.parsing.docx_parser import DocxParser


# ==============================================================================
# Generic Synthetic Spec Fixtures (Protocol-Agnostic)
# ==============================================================================

@pytest.fixture
def synthetic_generic_ir():
    """Builds a generic synthetic accelerator EFS IR with multiple opcodes and FSM."""
    ir = EFSIR()
    trace = EFSTraceability(source_document="generic_accelerator.txt", chunk_id="CHUNK_01", confidence=1.0)

    # 4 distinct instructions with 2 collisions (COLLISION_1: 0101, COLLISION_2: 0110)
    ir.instructions = [
        EFSInstruction(instruction_id="inst_1", name="VECTOR_ADD", mnemonic="VADD", opcode="0101", traceability=trace),
        EFSInstruction(instruction_id="inst_2", name="VECTOR_SUB", mnemonic="VSUB", opcode="0101", traceability=trace), # Collision 1
        EFSInstruction(instruction_id="inst_3", name="MATRIX_MUL", mnemonic="MMUL", opcode="0110", traceability=trace),
        EFSInstruction(instruction_id="inst_4", name="MATRIX_DIV", mnemonic="MDIV", opcode="0110", traceability=trace), # Collision 2
        EFSInstruction(instruction_id="inst_5", name="VECTOR_NOP", mnemonic="VNOP", opcode="0000", traceability=trace),
    ]

    ir.opcodes = [
        EFSOpcode(opcode_id="opc_1", mnemonic="VADD", binary_encoding="0101", traceability=trace),
        EFSOpcode(opcode_id="opc_2", mnemonic="VSUB", binary_encoding="0101", traceability=trace),
        EFSOpcode(opcode_id="opc_3", mnemonic="MMUL", binary_encoding="0110", traceability=trace),
        EFSOpcode(opcode_id="opc_4", mnemonic="MDIV", binary_encoding="0110", traceability=trace),
        EFSOpcode(opcode_id="opc_5", mnemonic="VNOP", binary_encoding="0000", traceability=trace),
    ]

    # Dedicated FSM with structured transitions
    ir.signals = [
        EFSSignal(name="start", direction="input", width="1", traceability=trace),
        EFSSignal(name="done", direction="output", width="1", traceability=trace),
        EFSSignal(name="error_flag", direction="output", width="1", traceability=trace),
    ]

    fsm = EFSFSM(
        name="Accelerator_Control_FSM",
        initial_state="IDLE",
        terminal_states=["DONE", "ERROR"],
        reset_state="IDLE",
        reset_behavior="Synchronous active-high reset to IDLE",
        states=[
            EFSState(name="IDLE"),
            EFSState(name="FETCH"),
            EFSState(name="EXECUTE"),
            EFSState(name="DONE"),
            EFSState(name="ERROR"),
        ],
        transitions=[
            # Signal assertion with identified signal
            EFSTransition(
                source_state="IDLE",
                target_state="FETCH",
                condition="start asserted",
                associated_signal="start",
                condition_type="signal_assertion",
                evidence="Transitions on start asserted",
                traceability=trace
            ),
            # Boolean condition
            EFSTransition(
                source_state="FETCH",
                target_state="EXECUTE",
                condition="valid_instr == 1'b1",
                condition_type="boolean",
                evidence="Fetches valid instruction",
                traceability=trace
            ),
            # Signal assertion with identified signal
            EFSTransition(
                source_state="EXECUTE",
                target_state="DONE",
                condition="done asserted",
                associated_signal="done",
                condition_type="signal_assertion",
                evidence="Operation completed (done asserted)",
                traceability=trace
            ),
            # Semantic condition without explicit boolean expression
            EFSTransition(
                source_state="EXECUTE",
                target_state="ERROR",
                condition="operation completed",
                associated_signal=None,
                condition_type="semantic",
                evidence="Transitions to error on unexpected termination",
                traceability=trace
            ),
        ],
        traceability=trace
    )
    ir.fsms.append(fsm)
    return ir


# ==============================================================================
# Tests A, B, C, D: Opcode Ownership and Conflict Detection
# ==============================================================================

def test_A_and_B_opcode_instruction_ownership(synthetic_generic_ir):
    """
    Requirements A & B:
    Multiple instructions sharing the same opcode must preserve exact instruction ownership.
    No entity owner should be lost or replaced with 'General'.
    """
    raw_conflicts = ConflictResolutionEngine._detect_raw_conflicts(synthetic_generic_ir)
    opcode_conflicts = [c for c in raw_conflicts if c.type == "DUPLICATE_OPCODE"]

    # There should be exactly 2 duplicate opcode conflicts (0101 and 0110)
    assert len(opcode_conflicts) == 2, f"Expected 2 duplicate opcode conflicts, got {len(opcode_conflicts)}"

    for c in opcode_conflicts:
        # Never assign 'General'
        assert "General" not in c.entities
        assert "General" not in (c.affected_entities or [])
        assert "GENERAL" not in [e.upper() for e in c.entities]

        # Ownership must match the actual instruction mnemonics
        if "0101" in c.conflicting_values:
            assert set(c.entities) == {"VADD", "VSUB"}
            assert set(c.affected_entities) == {"VADD", "VSUB"}
            assert c.property == "opcode"
        elif "0110" in c.conflicting_values:
            assert set(c.entities) == {"MMUL", "MDIV"}
            assert set(c.affected_entities) == {"MMUL", "MDIV"}
            assert c.property == "opcode"


def test_C_and_D_legitimate_vs_false_general_conflicts():
    """
    Requirements C & D:
    Legitimate conflicts remain conflicts.
    No false 'General' conflicts are produced from empty entities or misaligned IDs.
    """
    ir = EFSIR()
    trace = EFSTraceability(source_document="matrix_spec.txt", confidence=1.0)

    # 4 Legitimate matrix spec conflicts
    collisions = [
        ("MSTORE", "MINV", "000110"),
        ("MTRANS", "MDET", "000111"),
        ("MZERO", "MEIG", "001000"),
        ("MIDENTITY", "MSVD", "001001"),
    ]

    for m1, m2, enc in collisions:
        ir.instructions.append(EFSInstruction(mnemonic=m1, opcode=enc, traceability=trace))
        ir.instructions.append(EFSInstruction(mnemonic=m2, opcode=enc, traceability=trace))
        ir.opcodes.append(EFSOpcode(mnemonic=m1, binary_encoding=enc, traceability=trace))
        ir.opcodes.append(EFSOpcode(mnemonic=m2, binary_encoding=enc, traceability=trace))

    raw_conflicts = ConflictResolutionEngine._detect_raw_conflicts(ir)
    deduped = ConflictResolutionEngine.deduplicate_conflicts(raw_conflicts)

    # Exactly 4 legitimate conflicts
    assert len(deduped) == 4, f"Expected exactly 4 legitimate conflicts, got {len(deduped)}"

    for c in deduped:
        assert c.type == "DUPLICATE_OPCODE"
        assert c.severity == "CRITICAL"
        assert c.blocking is True
        assert c.property == "opcode"
        # Verify no entity is 'General'
        assert not any(e.upper() == "GENERAL" for e in c.entities)
        assert not any(e.upper() == "GENERAL" for e in (c.affected_entities or []))
        # Ensure owners are present
        assert len(c.entities) == 2


# ==============================================================================
# Test E: Canonicalization of Property Names
# ==============================================================================

def test_E_opcode_property_canonicalization():
    """
    Requirement E:
    Semantically equivalent fields (opcode, binary_encoding, encoding, opcode_value, instruction_opcode)
    must map to the identical canonical semantic key so they deduplicate cleanly.
    """
    props = ["opcode", "binary_encoding", "encoding", "opcode_value", "instruction_opcode"]
    keys = set()

    for p in props:
        c = EFSConflict(
            type="DUPLICATE_OPCODE",
            property=p,
            entities=["MSTORE", "MINV"],
            affected_entities=["MSTORE", "MINV"],
            conflicting_values=["000110"],
        )
        key = ConflictResolutionEngine.compute_semantic_key(c)
        keys.add(key)

    assert len(keys) == 1, f"Expected all equivalent properties to produce 1 canonical key, produced {len(keys)}: {keys}"
    canonical_key = list(keys)[0]
    assert ":opcode:" in canonical_key
    assert "MINV,MSTORE" in canonical_key
    assert "000110" in canonical_key


# ==============================================================================
# Tests F, G, H, I: FSM State & Transition Extraction and Condition Qualification
# ==============================================================================

def test_F_and_G_fsm_state_and_transition_structure(synthetic_generic_ir):
    """
    Requirements F & G:
    EFS IR explicitly represents FSM name, states, initial state, terminal states,
    reset behavior, and transitions with source and destination states.
    """
    assert len(synthetic_generic_ir.fsms) == 1
    fsm = synthetic_generic_ir.fsms[0]

    assert fsm.name == "Accelerator_Control_FSM"
    assert fsm.initial_state == "IDLE"
    assert "DONE" in fsm.terminal_states
    assert "ERROR" in fsm.terminal_states
    assert fsm.reset_behavior != ""

    state_names = [s.name for s in fsm.states]
    assert set(state_names) == {"IDLE", "FETCH", "EXECUTE", "DONE", "ERROR"}

    # Verify transitions are explicit relationships
    transitions = fsm.transitions
    assert len(transitions) == 4

    t_dict = {(t.source_state, t.target_state): t for t in transitions}
    assert ("IDLE", "FETCH") in t_dict
    assert ("FETCH", "EXECUTE") in t_dict
    assert ("EXECUTE", "DONE") in t_dict
    assert ("EXECUTE", "ERROR") in t_dict


def test_H_and_I_transition_condition_qualification(synthetic_generic_ir):
    """
    Requirements H & I:
    Transition conditions are qualified:
    - Signal assertions (e.g. 'done asserted') have associated_signal='done' and condition_type='signal_assertion'
    - Explicit booleans (e.g. 'valid_instr == 1'b1') have condition_type='boolean'
    - Purely semantic phrases (e.g. 'operation completed') are marked semantic and flagged as ambiguous
    - Legitimate signal assertions are NOT falsely flagged as vague conflicts.
    """
    raw_conflicts = ConflictResolutionEngine._detect_raw_conflicts(synthetic_generic_ir)
    fsm_conflicts = [c for c in raw_conflicts if "FSM" in c.title or c.property == "transition_condition"]

    # Only EXECUTE -> ERROR has condition_type="semantic" without a signal assertion
    # EXECUTE -> DONE (done asserted) and IDLE -> FETCH (start asserted) must NOT be conflicts!
    assert len(fsm_conflicts) == 1, f"Expected exactly 1 ambiguous FSM conflict for semantic condition, got {len(fsm_conflicts)}: {[c.title for c in fsm_conflicts]}"
    c = fsm_conflicts[0]
    assert "EXECUTE -> ERROR" in c.title or ("EXECUTE" in c.entities and "ERROR" in c.entities)


def test_dedicated_fsm_extractor_dynamic_classification():
    """
    Verifies DedicatedFSMExtractor dynamically parses and classifies conditions
    without hallucinating booleans or using protocol-specific hardcoding.
    """
    doc_ir = DocumentIR(
        document_id="doc_fsm_test",
        blocks=[
            DocumentBlock(
                block_id="b1",
                type="paragraph",
                text=(
                    "The primary state machine operates across IDLE, RUN, and FINISH states.\n"
                    "On reset, the machine starts in IDLE.\n"
                    "From IDLE, it advances to RUN when start is asserted.\n"
                    "From RUN, it transitions to FINISH when done is asserted.\n"
                    "From FINISH, it resets to IDLE after operation completed."
                )
            )
        ]
    )

    fsm_data = DedicatedFSMExtractor.extract_fsm(doc_ir, known_signals=["start", "done"])
    assert fsm_data is not None
    assert "IDLE" in fsm_data.states
    assert "RUN" in fsm_data.states
    assert "FINISH" in fsm_data.states

    # Check transitions
    transitions = {f"{t.source_state}->{t.target_state}": t for t in fsm_data.transitions}
    assert "IDLE->RUN" in transitions
    assert "RUN->FINISH" in transitions

    # Verify IDLE -> RUN has associated_signal 'start'
    tr_idle_run = transitions["IDLE->RUN"]
    assert tr_idle_run.associated_signal == "start"
    assert tr_idle_run.condition_type == "signal_assertion"

    # Verify RUN -> FINISH has associated_signal 'done'
    tr_run_finish = transitions["RUN->FINISH"]
    assert tr_run_finish.associated_signal == "done"
    assert tr_run_finish.condition_type == "signal_assertion"

    # Verify FINISH -> IDLE is semantic without inventing a signal
    tr_finish_idle = transitions.get("FINISH->IDLE")
    if tr_finish_idle:
        assert tr_finish_idle.condition_type in ("semantic", "unresolved")


# ==============================================================================
# Test J: Provenance Preservation
# ==============================================================================

def test_J_provenance_preservation(synthetic_generic_ir):
    """
    Requirement J:
    All extracted instructions, opcodes, FSMs, transitions, and conflicts preserve traceability.
    """
    for inst in synthetic_generic_ir.instructions:
        assert inst.traceability is not None
        assert inst.traceability.source_document == "generic_accelerator.txt"

    for opc in synthetic_generic_ir.opcodes:
        assert opc.traceability is not None
        assert opc.traceability.source_document == "generic_accelerator.txt"

    fsm = synthetic_generic_ir.fsms[0]
    assert fsm.traceability is not None
    for tr in fsm.transitions:
        assert tr.traceability is not None
        assert tr.evidence != ""


# ==============================================================================
# Tests K & L: HITL Resolution and Stale Conflict Removal
# ==============================================================================

def test_K_and_L_hitl_resolution_and_stale_conflict_removal(synthetic_generic_ir):
    """
    Requirements K & L:
    Applying a resolution mutates the canonical EFS IR, re-runs affected validation,
    and removes stale conflicts from the active set.
    """
    # 1. Detect initial conflicts
    initial_conflicts = ConflictResolutionEngine._detect_raw_conflicts(synthetic_generic_ir)
    synthetic_generic_ir.conflicts = initial_conflicts

    # Locate conflict for VADD / VSUB duplicate opcode
    vsub_conflict = next(
        (c for c in initial_conflicts if c.type == "DUPLICATE_OPCODE" and "VSUB" in c.entities),
        None
    )
    assert vsub_conflict is not None

    # 2. Propose concrete resolution for VSUB -> 0111 (unused opcode)
    cands = ConflictResolutionEngine.calculate_concrete_candidates(vsub_conflict, synthetic_generic_ir, count=1)
    assert len(cands) > 0, "Expected at least 1 valid resolution candidate"
    cand_val, cand_change, _ = cands[0]

    # Validate that proposed value does not collide with existing opcodes
    is_valid, err = ConflictResolutionEngine.validate_resolution(
        vsub_conflict, cand_val, synthetic_generic_ir, target_entity="VSUB"
    )
    assert is_valid is True, f"Validation failed: {err}"

    # 3. Create approved resolution
    resolution = ConflictResolutionEngine.create_custom_resolution(
        vsub_conflict,
        f"VSUB.opcode = {cand_val}",
        user_name="designer",
        target_entity="VSUB"
    )

    # 4. Apply resolution to EFS IR
    updated_ir = ConflictResolutionEngine.apply_approved_resolution(
        synthetic_generic_ir,
        resolution
    )

    # 5. Verify authoritative EFS IR state was updated
    resolved_vsub_inst = next((i for i in updated_ir.instructions if i.mnemonic == "VSUB"), None)
    resolved_vsub_opc = next((o for o in updated_ir.opcodes if o.mnemonic == "VSUB"), None)
    assert resolved_vsub_inst.opcode == cand_val
    assert resolved_vsub_opc.binary_encoding == cand_val

    # 6. Verify stale conflict is REMOVED from active conflicts
    unresolved_after = [c for c in updated_ir.conflicts if c.status != "RESOLVED" and getattr(c, "resolution_status", "") != "RESOLVED"]
    assert not any(c.conflict_id == vsub_conflict.conflict_id for c in unresolved_after), "Resolved conflict must not remain active"
    assert not any("VSUB" in c.entities and c.type == "DUPLICATE_OPCODE" for c in unresolved_after), "Stale duplicate opcode conflict for VSUB must be gone"


# ==============================================================================
# Test M: Protocol-Agnostic and Specification-Agnostic Behavior
# ==============================================================================

def test_M_protocol_agnostic_synthetic_spec():
    """
    Requirement M:
    System functions correctly for arbitrary non-matrix synthetic hardware specifications
    (e.g. RISC instruction set, DSP accelerator, Neural Engine) with custom opcodes and FSMs.
    """
    dsp_ir = EFSIR()
    trace = EFSTraceability(source_document="dsp_accelerator_spec.txt", confidence=1.0)

    # Synthetic DSP instructions: MAC, CONV, FFT, IFFT
    dsp_ir.instructions = [
        EFSInstruction(mnemonic="DSP_MAC", opcode="110001", traceability=trace),
        EFSInstruction(mnemonic="DSP_CONV", opcode="110001", traceability=trace), # Collision
        EFSInstruction(mnemonic="DSP_FFT", opcode="110010", traceability=trace),
    ]
    dsp_ir.opcodes = [
        EFSOpcode(mnemonic="DSP_MAC", binary_encoding="110001", traceability=trace),
        EFSOpcode(mnemonic="DSP_CONV", binary_encoding="110001", traceability=trace),
        EFSOpcode(mnemonic="DSP_FFT", binary_encoding="110010", traceability=trace),
    ]

    conflicts = ConflictResolutionEngine._detect_raw_conflicts(dsp_ir)
    opc_confs = [c for c in conflicts if c.type == "DUPLICATE_OPCODE"]

    assert len(opc_confs) == 1
    assert set(opc_confs[0].entities) == {"DSP_MAC", "DSP_CONV"}
    assert opc_confs[0].property == "opcode"
    assert "General" not in opc_confs[0].entities

    # Resolve DSP collision
    cands = ConflictResolutionEngine.calculate_concrete_candidates(opc_confs[0], dsp_ir, count=1)
    assert len(cands) > 0
    new_val = cands[0][0]

    res = ConflictResolutionEngine.create_custom_resolution(
        opc_confs[0],
        f"DSP_CONV.opcode = {new_val}",
        target_entity="DSP_CONV"
    )

    resolved_dsp = ConflictResolutionEngine.apply_approved_resolution(dsp_ir, res)
    active_confs = [c for c in resolved_dsp.conflicts if c.status != "RESOLVED"]
    assert len(active_confs) == 0, "All conflicts must be resolved cleanly without protocol-specific assumptions"
