"""
Tests ensuring that resolved conflicts are immediately filtered out and no longer visible on screen.
Requirement: "if a conflict is resolved, it shouldn't be visible anymore on screen"
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.efs_ir.models import EFSIR, EFSConflict, EFSResolution, EFSRegister, EFSSignal, EFSOpcode
from core.verification.conflict_resolution_engine import ConflictResolutionEngine


def test_resolved_conflict_not_in_unresolved_conflicts():
    """
    Test that once an AI proposal or custom resolution is applied and approved,
    the conflict is marked RESOLVED and is filtered out from active unresolved display.
    """
    ir = EFSIR()
    ir.signals = [
        EFSSignal(name="clk", width="1", direction="input", semantic_role="clock"),
        EFSSignal(name="rst_n", width="1", direction="input", semantic_role="reset")
    ]
    ir.registers = [
        EFSRegister(name="REG_STATUS", offset="0x04", width=32),
        EFSRegister(name="REG_CONTROL", offset="0x04", width=32)  # Conflict: duplicate offset
    ]

    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    assert len(conflicts) == 1
    conf = conflicts[0]
    assert conf.resolution_status == "UNRESOLVED"
    assert conf.blocking is True

    # 1. Option 1: AI Proposal Accepted
    proposals = ConflictResolutionEngine.generate_resolution_proposals(conf, ir)
    assert len(proposals) >= 1
    proposal = proposals[0]
    proposal.approval_status = "APPROVED"

    # Apply resolution to IR
    resolved_ir = ConflictResolutionEngine.apply_approved_resolution(ir, proposal)

    # Re-detect or check conflicts on resolved IR
    resolved_conflicts = ConflictResolutionEngine.detect_conflicts(resolved_ir)
    
    # Active/unresolved conflicts filter (matching ui/hardware_design.py and ui/efs_ir_review.py)
    approved_cids = {
        r.conflict_id for r in resolved_ir.resolutions if getattr(r, "approval_status", "") == "APPROVED"
    }
    unresolved_display = [
        c for c in resolved_conflicts
        if getattr(c, "resolution_status", "UNRESOLVED") == "UNRESOLVED"
        and getattr(c, "status", "") not in ("APPROVED", "RESOLVED")
        and getattr(c, "blocking", True)
        and getattr(c, "conflict_id", None) not in approved_cids
    ]

    # The resolved conflict MUST NOT appear on screen!
    assert len(unresolved_display) == 0, f"Expected 0 visible conflicts, but found: {unresolved_display}"


def test_custom_solution_resolves_and_hides_conflict():
    """
    Test that Option 2 (Custom Requirement) resolves the conflict and removes it from screen.
    """
    ir = EFSIR()
    ir.opcodes = [
        EFSOpcode(mnemonic="LOAD", binary_encoding="1010"),
        EFSOpcode(mnemonic="STORE", binary_encoding="1010")  # Collision!
    ]

    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    assert len(conflicts) == 1
    conf = conflicts[0]

    # Option 2: Human designer custom resolution
    custom_res = ConflictResolutionEngine.create_custom_resolution(
        conf,
        "STORE.binary_encoding = 1011",
        user_name="human_designer",
        target_entity="STORE"
    )
    assert custom_res.approval_status == "APPROVED"

    resolved_ir = ConflictResolutionEngine.apply_approved_resolution(ir, custom_res)
    resolved_conflicts = ConflictResolutionEngine.detect_conflicts(resolved_ir)

    approved_cids = {
        r.conflict_id for r in resolved_ir.resolutions if getattr(r, "approval_status", "") == "APPROVED"
    }
    unresolved_display = [
        c for c in resolved_conflicts
        if getattr(c, "resolution_status", "UNRESOLVED") == "UNRESOLVED"
        and getattr(c, "status", "") not in ("APPROVED", "RESOLVED")
        and getattr(c, "blocking", True)
        and getattr(c, "conflict_id", None) not in approved_cids
    ]

    # The custom-resolved conflict MUST NOT appear on screen!
    assert len(unresolved_display) == 0


def test_partial_resolution_only_shows_remaining():
    """
    When multiple conflicts exist and only one is resolved, only the remaining unresolved
    conflict is visible on screen. The resolved one disappears.
    """
    ir = EFSIR()
    ir.registers = [
        EFSRegister(name="REG_A", offset="0x08", width=32),
        EFSRegister(name="REG_B", offset="0x08", width=32)
    ]
    ir.opcodes = [
        EFSOpcode(mnemonic="OP_X", binary_encoding="0101"),
        EFSOpcode(mnemonic="OP_Y", binary_encoding="0101")
    ]

    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    assert len(conflicts) == 2

    # Resolve only the opcode conflict
    op_conf = next(c for c in conflicts if "OPCODE" in c.type)
    custom_res = ConflictResolutionEngine.create_custom_resolution(
        op_conf,
        "OP_Y.binary_encoding = 0110",
        user_name="human_designer",
        target_entity="OP_Y"
    )

    resolved_ir = ConflictResolutionEngine.apply_approved_resolution(ir, custom_res)
    resolved_conflicts = ConflictResolutionEngine.detect_conflicts(resolved_ir)

    approved_cids = {
        r.conflict_id for r in resolved_ir.resolutions if getattr(r, "approval_status", "") == "APPROVED"
    }
    unresolved_display = [
        c for c in resolved_conflicts
        if getattr(c, "resolution_status", "UNRESOLVED") == "UNRESOLVED"
        and getattr(c, "status", "") not in ("APPROVED", "RESOLVED")
        and getattr(c, "blocking", True)
        and getattr(c, "conflict_id", None) not in approved_cids
    ]

    # Exactly 1 conflict remaining on screen (the register address collision). The opcode conflict is GONE!
    assert len(unresolved_display) == 1
    assert unresolved_display[0].type == "DUPLICATE_REGISTER_ADDRESS"
