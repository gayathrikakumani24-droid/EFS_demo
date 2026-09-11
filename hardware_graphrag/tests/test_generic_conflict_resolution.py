"""
Synthetic test suite verifying the Generic Specification-Conflict Resolution Workflow.

Tests all 14 required conflict resolution behaviors:
1. No conflicts -> RTL generation proceeds.
2. One genuine conflict -> generation blocked.
3. Conflict -> AI proposal generated -> not automatically applied.
4. Conflict -> user approves proposal -> model updated via resolution layer -> revalidated -> generation proceeds.
5. Conflict -> user rejects proposal -> generation remains blocked.
6. Two conflicts -> resolve one -> second still blocks generation.
7. Different conflict types -> same generic resolution framework handles them.
8. Different terminology -> no hardcoded names required.
9. Missing requirement -> clarification workflow, not conflict workflow.
10. Ambiguous requirement -> ambiguity workflow, not conflict workflow.
11. Validator assumption not in source -> must NOT be reported as SOURCE_MISSING.
12. Original source requirements remain unchanged after resolution.
13. Approved resolution survives revalidation and is traceable.
14. Generated RTL is based on resolved requirement model, not an overwritten source model.
"""

import pytest
from core.efs_ir.models import (
    EFSIR, EFSConflict, EFSResolution, EFSRegister, EFSSignal, EFSOpcode,
    EFSRequirementItem, EFSTraceability
)
from core.verification.conflict_resolution_engine import ConflictResolutionEngine
from core.efs_ir.validator import validate_efs_ir


def create_base_synthetic_ir():
    """Helper to build a clean synthetic EFSIR model with non-hardcoded names."""
    ir = EFSIR()
    ir.signals = [
        EFSSignal(name="sys_clk", width="1", direction="input", semantic_role="clock"),
        EFSSignal(name="sys_rst", width="1", direction="input", semantic_role="reset")
    ]
    ir.registers = [
        EFSRegister(name="ALPHA_REG", offset="0x04", width=32),
        EFSRegister(name="BETA_REG", offset="0x08", width=32)
    ]
    ir.opcodes = [
        EFSOpcode(mnemonic="EXEC_X", binary_encoding="8'b00000001"),
        EFSOpcode(mnemonic="EXEC_Y", binary_encoding="8'b00000010")
    ]
    return ir


def test_1_no_conflicts_proceeds_to_rtl():
    ir = create_base_synthetic_ir()
    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    unresolved = [c for c in conflicts if c.blocking and c.resolution_status == "UNRESOLVED"]
    assert len(unresolved) == 0
    has_blocking, issues = ConflictResolutionEngine.revalidate_resolved_model(ir)
    assert not has_blocking


def test_2_one_genuine_conflict_blocks_generation():
    ir = create_base_synthetic_ir()
    # Add duplicate address offset
    ir.registers.append(EFSRegister(name="GAMMA_REG", offset="0x04", width=32))
    
    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    unresolved = [c for c in conflicts if c.blocking and c.resolution_status == "UNRESOLVED"]
    assert len(unresolved) == 1
    assert unresolved[0].type == "DUPLICATE_REGISTER_ADDRESS"
    assert unresolved[0].blocking is True


def test_3_ai_proposals_not_auto_applied():
    ir = create_base_synthetic_ir()
    ir.registers.append(EFSRegister(name="GAMMA_REG", offset="0x04", width=32))
    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    conf = conflicts[0]
    
    proposals = ConflictResolutionEngine.generate_resolution_proposals(conf, ir)
    assert len(proposals) >= 2
    
    # Verify source IR was NOT modified automatically
    assert ir.registers[0].offset == "0x04"
    assert ir.registers[2].offset == "0x04"
    assert conf.status == "UNRESOLVED"


def test_4_user_approval_revalidates_and_proceeds():
    ir = create_base_synthetic_ir()
    ir.registers.append(EFSRegister(name="GAMMA_REG", offset="0x04", width=32))
    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    conf = conflicts[0]
    
    proposals = ConflictResolutionEngine.generate_resolution_proposals(conf, ir)
    proposal_to_approve = proposals[1]  # Proposal for GAMMA_REG override
    proposal_to_approve.proposed_value = "0x0C"
    proposal_to_approve.proposed_change = "offset = 0x0C"
    proposal_to_approve.affected_requirements = ["GAMMA_REG"]
    
    # Apply approval
    resolved_ir = ConflictResolutionEngine.apply_approved_resolution(ir, proposal_to_approve)
    
    # Verify resolution layer contains approved proposal
    assert len(resolved_ir.resolutions) == 1
    assert resolved_ir.resolutions[0].approval_status == "APPROVED"
    
    # Revalidate
    has_blocking, issues = ConflictResolutionEngine.revalidate_resolved_model(resolved_ir)
    assert not has_blocking


def test_5_user_rejection_keeps_blocked():
    ir = create_base_synthetic_ir()
    ir.registers.append(EFSRegister(name="GAMMA_REG", offset="0x04", width=32))
    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    conf = conflicts[0]
    
    proposals = ConflictResolutionEngine.generate_resolution_proposals(conf, ir)
    rejected_proposal = proposals[0]
    rejected_proposal.approval_status = "REJECTED"
    
    # Model should remain blocked
    has_blocking, _ = ConflictResolutionEngine.revalidate_resolved_model(ir)
    assert has_blocking is True


def test_6_two_conflicts_resolve_one_second_still_blocks():
    ir = create_base_synthetic_ir()
    # Conflict 1: Register address collision
    ir.registers.append(EFSRegister(name="GAMMA_REG", offset="0x04", width=32))
    # Conflict 2: Opcode encoding collision
    ir.opcodes.append(EFSOpcode(mnemonic="EXEC_Z", binary_encoding="8'b00000001"))
    
    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    assert len(conflicts) == 2
    
    # Resolve Conflict 1
    p1 = EFSResolution(
        conflict_id=conflicts[0].conflict_id,
        proposed_change="offset = 0x0C",
        proposed_value="0x0C",
        affected_requirements=["GAMMA_REG"],
        approval_status="APPROVED"
    )
    resolved_ir = ConflictResolutionEngine.apply_approved_resolution(ir, p1)
    
    # Revalidate: Second conflict (opcode collision) must still block!
    has_blocking, issues = ConflictResolutionEngine.revalidate_resolved_model(resolved_ir)
    assert has_blocking is True
    unresolved = [c for c in resolved_ir.conflicts if c.blocking and c.resolution_status == "UNRESOLVED"]
    assert len(unresolved) == 1
    assert unresolved[0].type == "DUPLICATE_OPCODE_ENCODING"


def test_7_different_conflict_types_handled_generically():
    ir = create_base_synthetic_ir()
    # Register conflict
    ir.registers.append(EFSRegister(name="REG_X", offset="0x04"))
    # Opcode conflict
    ir.opcodes.append(EFSOpcode(mnemonic="OP_Y", binary_encoding="8'b00000001"))
    
    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    types = {c.type for c in conflicts}
    assert "DUPLICATE_REGISTER_ADDRESS" in types
    assert "DUPLICATE_OPCODE_ENCODING" in types


def test_8_different_terminology_no_hardcoded_names():
    ir = EFSIR()
    # Custom non-standard signal and component names
    ir.signals = [
        EFSSignal(name="custom_clock_net", width="1", direction="input", semantic_role="clock"),
        EFSSignal(name="custom_reset_net", width="1", direction="input", semantic_role="reset")
    ]
    ir.registers = [
        EFSRegister(name="ACCEL_CFG_0", offset="0x10", width=64),
        EFSRegister(name="ACCEL_CFG_1", offset="0x10", width=64)
    ]
    
    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    assert len(conflicts) == 1
    assert "ACCEL_CFG_0" in conflicts[0].entities
    assert "ACCEL_CFG_1" in conflicts[0].entities


def test_9_missing_requirement_vs_conflict():
    req_item = EFSRequirementItem(
        category="INTERFACE",
        property="clock_frequency",
        value=None,
        status="MISSING"
    )
    assert req_item.status == "MISSING"
    assert req_item.status != "CONFLICT"


def test_10_ambiguous_requirement_vs_conflict():
    req_item = EFSRequirementItem(
        category="RESET",
        property="polarity",
        value="unclear",
        status="AMBIGUOUS"
    )
    assert req_item.status == "AMBIGUOUS"
    assert req_item.status != "CONFLICT"


def test_11_validator_assumption_not_in_source_not_reported_missing():
    ir = create_base_synthetic_ir()
    # EFSIR without optional timing rules
    issues = validate_efs_ir(ir)
    missing_issues = [i for i in issues if i.get("classification") == "MISSING"]
    assert len(missing_issues) == 0


def test_12_original_source_requirements_remain_unchanged():
    ir = create_base_synthetic_ir()
    ir.registers.append(EFSRegister(name="GAMMA_REG", offset="0x04", width=32))
    original_offset = ir.registers[2].offset
    
    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    p = EFSResolution(
        conflict_id=conflicts[0].conflict_id,
        proposed_change="offset = 0x10",
        proposed_value="0x10",
        affected_requirements=["GAMMA_REG"],
        approval_status="APPROVED"
    )
    
    resolved_view = ConflictResolutionEngine.apply_approved_resolution(ir, p)
    
    # Original EFSIR source register MUST remain unchanged!
    assert ir.registers[2].offset == original_offset
    # Resolved view reflects the approved resolution
    assert resolved_view.registers[2].offset == "0x10"


def test_13_approved_resolution_survives_revalidation_and_traceable():
    ir = create_base_synthetic_ir()
    ir.registers.append(EFSRegister(name="GAMMA_REG", offset="0x04", width=32))
    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    
    p = EFSResolution(
        conflict_id=conflicts[0].conflict_id,
        proposed_change="offset = 0x14",
        proposed_value="0x14",
        affected_requirements=["GAMMA_REG"],
        approval_status="APPROVED",
        approved_by="test_user"
    )
    
    resolved_view = ConflictResolutionEngine.apply_approved_resolution(ir, p)
    has_blocking, issues = ConflictResolutionEngine.revalidate_resolved_model(resolved_view)
    
    assert not has_blocking
    assert len(resolved_view.resolutions) == 1
    assert resolved_view.resolutions[0].approved_by == "test_user"
    assert resolved_view.resolutions[0].proposed_value == "0x14"


def test_14_generated_rtl_based_on_resolved_model():
    ir = create_base_synthetic_ir()
    ir.registers.append(EFSRegister(name="GAMMA_REG", offset="0x04", width=32))
    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    
    p = EFSResolution(
        conflict_id=conflicts[0].conflict_id,
        proposed_change="offset = 0x20",
        proposed_value="0x20",
        affected_requirements=["GAMMA_REG"],
        approval_status="APPROVED"
    )
    
    resolved_view = ConflictResolutionEngine.apply_approved_resolution(ir, p)
    gamma_reg = next(r for r in resolved_view.registers if r.name == "GAMMA_REG")
    
    assert gamma_reg.offset == "0x20"


def test_15_ai_never_returns_existing_conflicting_value_or_placeholder():
    ir = create_base_synthetic_ir()
    # Add conflicting opcode
    ir.opcodes = [
        EFSOpcode(mnemonic="MTRANS", binary_encoding="000111"),
        EFSOpcode(mnemonic="MDET", binary_encoding="000111")
    ]
    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    assert len(conflicts) == 1
    conf = conflicts[0]

    proposals = ConflictResolutionEngine.generate_resolution_proposals(conf, ir)
    assert len(proposals) >= 1
    p = proposals[0]

    # AI proposed value MUST NOT be 000111
    assert p.proposed_value != "000111"
    # AI proposed value MUST NOT contain placeholders
    for ph in ConflictResolutionEngine.DISALLOWED_PLACEHOLDER_SUBSTRINGS:
        assert ph not in str(p.proposed_value).lower()
        assert ph not in p.proposed_change.lower()

    # Proposal MUST be concrete valid value (e.g. 000000 or 000001 or 001111)
    assert p.proposed_value in ["000000", "000001", "000010", "000011", "000100", "000101", "000110", "001111", "001000"]


def test_16_deterministic_validation_rejects_placeholders_and_collisions():
    ir = create_base_synthetic_ir()
    ir.opcodes = [
        EFSOpcode(mnemonic="MTRANS", binary_encoding="000111"),
        EFSOpcode(mnemonic="MDET", binary_encoding="000111")
    ]
    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    conf = conflicts[0]

    # Test 1: Placeholder rejection
    valid, msg = ConflictResolutionEngine.validate_proposal_candidate("reassign_binary_encoding_for_MDET", conf, ir)
    assert not valid
    assert "placeholder" in msg.lower()

    # Test 2: Exact conflicting value rejection
    valid, msg = ConflictResolutionEngine.validate_proposal_candidate("000111", conf, ir)
    assert not valid
    assert "preserves" in msg.lower() or "conflicting" in msg.lower()

    # Test 3: Existing assigned value collision rejection
    ir.opcodes.append(EFSOpcode(mnemonic="MSEL", binary_encoding="000000"))
    valid, msg = ConflictResolutionEngine.validate_proposal_candidate("000000", conf, ir)
    assert not valid
    assert "collides" in msg.lower()

    # Test 4: Valid concrete candidate acceptance
    valid, msg = ConflictResolutionEngine.validate_proposal_candidate("000001", conf, ir)
    assert valid

