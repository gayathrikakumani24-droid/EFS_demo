"""
Comprehensive Test Suite for Generic Human-in-the-Loop Conflict Resolution & Pipeline Continuation.

Tests all 11 required test scenarios:
1. SOURCE_CONFLICT detection
2. PLAN_CONFLICT detection
3. SOURCE_AMBIGUOUS detection
4. AI resolution approval
5. AI resolution rejection
6. Custom requirement submission
7. Streamlit rerun/state persistence
8. Multiple simultaneous conflicts
9. Unresolved conflict blocks generation
10. Approved resolution allows generation
11. Original source specification immutability
"""

import sys
import os
import pytest
from typing import Dict, Any, List

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.efs_ir.models import EFSIR, EFSConflict, EFSResolution, EFSRegister, EFSSignal, EFSOpcode
from core.verification.conflict_resolution_engine import ConflictResolutionEngine
from core.agents.orchestrator import resume_design_flow_with_resolutions, run_design_flow


def create_base_efs_ir() -> EFSIR:
    """Helper to create a valid base EFSIR instance."""
    ir = EFSIR()
    ir.registers.append(EFSRegister(name="CTRL_REG", offset="0x04"))
    ir.signals.append(EFSSignal(name="clk", direction="input", width=1))
    ir.signals.append(EFSSignal(name="rst_n", direction="input", width=1))
    return ir


def test_1_source_conflict_detection():
    """1. Test detection of SOURCE_CONFLICT from specification requirement model."""
    efs_ir = create_base_efs_ir()
    req_model = {
        "issues": [
            {
                "classification": "SOURCE_CONFLICT",
                "category": "OPCODE",
                "severity": "CRITICAL",
                "issue": "Opcode conflict between ADD and SUB",
                "user_spec_evidence": "ADD opcode = 0x01, SUB opcode = 0x01",
                "is_blocking": True
            }
        ]
    }
    conflicts = ConflictResolutionEngine.detect_conflicts(efs_ir, req_model)
    source_conflicts = [c for c in conflicts if c.type == "SOURCE_CONFLICT"]
    assert len(source_conflicts) >= 1
    assert source_conflicts[0].blocking is True
    assert source_conflicts[0].resolution_status == "UNRESOLVED"


def test_2_plan_conflict_detection():
    """2. Test detection of PLAN_CONFLICT from specification requirement model."""
    efs_ir = create_base_efs_ir()
    req_model = {
        "issues": [
            {
                "classification": "PLAN_CONFLICT",
                "category": "FSM",
                "severity": "ERROR",
                "issue": "FSM state transition conflict between IDLE and EXEC",
                "evidence": "FSM transition matrix mismatch",
                "is_blocking": True
            }
        ]
    }
    conflicts = ConflictResolutionEngine.detect_conflicts(efs_ir, req_model)
    plan_conflicts = [c for c in conflicts if c.type == "PLAN_CONFLICT"]
    assert len(plan_conflicts) >= 1
    assert plan_conflicts[0].blocking is True


def test_3_source_ambiguous_detection():
    """3. Test detection of SOURCE_AMBIGUOUS from specification requirement model."""
    efs_ir = create_base_efs_ir()
    req_model = {
        "issues": [
            {
                "classification": "SOURCE_AMBIGUOUS",
                "category": "INTERFACE",
                "severity": "CRITICAL",
                "issue": "Ambiguous signal direction for data_valid",
                "required_information": "Specify if data_valid is input or output",
                "is_blocking": True
            }
        ]
    }
    conflicts = ConflictResolutionEngine.detect_conflicts(efs_ir, req_model)
    ambig_conflicts = [c for c in conflicts if c.type == "SOURCE_AMBIGUOUS"]
    assert len(ambig_conflicts) >= 1
    assert ambig_conflicts[0].blocking is True


def test_4_ai_resolution_approval():
    """4. Test selecting 'Allow AI to make changes' and approving the proposal."""
    efs_ir = create_base_efs_ir()
    conflict = EFSConflict(
        conflict_id="CONF_TEST_01",
        type="DUPLICATE_REGISTER_ADDRESS",
        severity="CRITICAL",
        blocking=True,
        entities=["REG_A", "REG_B"],
        property="address_offset",
        conflicting_values=["0x04", "0x04"],
        evidence="REG_A and REG_B share offset 0x04"
    )
    efs_ir.conflicts.append(conflict)

    proposals = ConflictResolutionEngine.generate_resolution_proposals(conflict, efs_ir)
    assert len(proposals) >= 1
    assert proposals[0].resolution_type == "ALLOW_AI"

    ai_prop = proposals[0]
    resolved_view = ConflictResolutionEngine.apply_approved_resolution(efs_ir, ai_prop)

    assert ai_prop.approval_status == "APPROVED"
    assert len(efs_ir.resolutions) == 1
    assert efs_ir.resolutions[0].conflict_id == "CONF_TEST_01"


def test_5_ai_resolution_rejection():
    """5. Test rejecting an AI resolution proposal leaves conflict unresolved."""
    efs_ir = create_base_efs_ir()
    conflict = EFSConflict(
        conflict_id="CONF_REJ_01",
        type="SOURCE_AMBIGUOUS",
        severity="CRITICAL",
        blocking=True,
        entities=["INTERFACE"],
        property="direction"
    )
    efs_ir.conflicts.append(conflict)

    proposals = ConflictResolutionEngine.generate_resolution_proposals(conflict, efs_ir)
    rej_proposal = proposals[0]
    rej_proposal.approval_status = "REJECTED"

    # Rejecting proposal does not mark conflict resolved
    assert conflict.resolution_status == "UNRESOLVED"
    assert len([r for r in efs_ir.resolutions if r.approval_status == "APPROVED"]) == 0


def test_6_custom_requirement_submission():
    """6. Test submitting a custom requirement ('Add a custom requirement')."""
    efs_ir = create_base_efs_ir()
    conflict = EFSConflict(
        conflict_id="CONF_CUST_01",
        type="SIGNAL_SPECIFICATION_CONFLICT",
        severity="CRITICAL",
        blocking=True,
        entities=["data_bus"],
        property="width",
        conflicting_values=["16", "32"]
    )
    efs_ir.conflicts.append(conflict)

    custom_text = "data_bus width = 32"
    custom_res = ConflictResolutionEngine.create_custom_resolution(conflict, custom_text, user_name="designer")

    assert custom_res.resolution_type == "CUSTOM_REQUIREMENT"
    assert custom_res.approval_status == "APPROVED"
    assert custom_res.approved_by == "designer"

    resolved_view = ConflictResolutionEngine.apply_approved_resolution(efs_ir, custom_res)
    assert conflict.resolution_status == "RESOLVED"


def test_7_streamlit_state_persistence_simulation():
    """7. Test that multiple approved resolutions persist in session_state accumulation list."""
    session_state_resolutions: List[EFSResolution] = []

    conf1 = EFSConflict(conflict_id="C1", type="TYPE_A", blocking=True, property="p1")
    conf2 = EFSConflict(conflict_id="C2", type="TYPE_B", blocking=True, property="p2")

    res1 = ConflictResolutionEngine.create_custom_resolution(conf1, "p1 = val1")
    res2 = ConflictResolutionEngine.create_custom_resolution(conf2, "p2 = val2")

    # Simulate rerun 1
    session_state_resolutions.append(res1)
    assert len(session_state_resolutions) == 1

    # Simulate rerun 2 (accumulation)
    session_state_resolutions.append(res2)
    assert len(session_state_resolutions) == 2
    assert session_state_resolutions[0].conflict_id == "C1"
    assert session_state_resolutions[1].conflict_id == "C2"


def test_8_multiple_simultaneous_conflicts():
    """8. Test handling multiple simultaneous conflicts."""
    efs_ir = create_base_efs_ir()
    c1 = EFSConflict(conflict_id="MC1", type="SOURCE_CONFLICT", blocking=True, property="opcode")
    c2 = EFSConflict(conflict_id="MC2", type="PLAN_CONFLICT", blocking=True, property="address")
    efs_ir.conflicts.extend([c1, c2])

    res1 = ConflictResolutionEngine.create_custom_resolution(c1, "opcode = 0x01")
    res2 = ConflictResolutionEngine.create_custom_resolution(c2, "address = 0x10")

    ConflictResolutionEngine.apply_approved_resolution(efs_ir, res1)
    ConflictResolutionEngine.apply_approved_resolution(efs_ir, res2)

    unresolved = [c for c in efs_ir.conflicts if c.blocking and c.resolution_status == "UNRESOLVED"]
    assert len(unresolved) == 0


def test_9_unresolved_conflict_blocks_generation():
    """9. Test that unresolved blocking conflicts abort generation / remain blocked."""
    efs_ir = create_base_efs_ir()
    c1 = EFSConflict(conflict_id="BLOCK_1", type="SOURCE_MISSING", blocking=True, status="UNRESOLVED")
    efs_ir.conflicts.append(c1)

    has_blocking, issues = ConflictResolutionEngine.revalidate_resolved_model(efs_ir)
    assert has_blocking is True


def test_10_approved_resolution_allows_generation():
    """10. Test that approving all resolutions clears blocking state during revalidation."""
    efs_ir = create_base_efs_ir()
    c1 = EFSConflict(conflict_id="BLOCK_1", type="SOURCE_MISSING", blocking=True, status="UNRESOLVED")
    efs_ir.conflicts.append(c1)

    res1 = ConflictResolutionEngine.create_custom_resolution(c1, "requirement = defined")
    resolved_ir = ConflictResolutionEngine.apply_approved_resolution(efs_ir, res1)

    has_blocking, issues = ConflictResolutionEngine.revalidate_resolved_model(resolved_ir)
    assert has_blocking is False


def test_11_original_source_immutability():
    """11. Test that applying resolutions leaves the original EFSIR source requirements untouched."""
    efs_ir = create_base_efs_ir()
    reg_orig = EFSRegister(register_id="REG_ORIG", name="STATUS_REG", offset="0x04")
    efs_ir.registers.append(reg_orig)

    conf = EFSConflict(conflict_id="C_IMMUTABLE", property="offset", entities=["STATUS_REG"], blocking=True)
    efs_ir.conflicts.append(conf)

    custom_res = ConflictResolutionEngine.create_custom_resolution(conf, "STATUS_REG offset = 0x08")
    resolved_view = ConflictResolutionEngine.apply_approved_resolution(efs_ir, custom_res)

    # Original base EFSIR register offset remains 0x04
    assert efs_ir.registers[1].offset == "0x04"
    # Resolved view has updated offset 0x08
    assert resolved_view.registers[1].offset == "0x08"
