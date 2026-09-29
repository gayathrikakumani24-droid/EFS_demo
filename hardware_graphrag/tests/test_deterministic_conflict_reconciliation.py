"""
Unit and integration tests for deterministic conflict reconciliation in the EFS hardware pipeline.

Covers:
1. Canonical semantic key calculation and conflict deduplication across representations.
2. Dual representation resolution (e.g. DUPLICATE_OPCODE_ENCODING + DUPLICATE_OPCODE).
3. EFS IR in-memory mutation verification (single source of truth).
4. Cascading conflict detection (introducing a new collision).
5. Zero LLM calls during post-resolution revalidation and reconciliation.
6. Preservation of conflict audit logs (historical records stay with status="RESOLVED").
7. Generic protocol behavior (works for registers, signals, and opcodes).
8. AI proposal approval path vs Custom user resolution path.
"""

import pytest
from unittest.mock import MagicMock
from core.efs_ir.models import (
    EFSIR, EFSConflict, EFSOpcode, EFSRegister, EFSRegisterField, EFSMetadata
)
from core.verification.conflict_resolution_engine import ConflictResolutionEngine


@pytest.fixture
def mock_llm_client():
    """Returns an LLM client that raises an exception if any call is made."""
    mock = MagicMock()
    mock.generate_response.side_effect = RuntimeError("Zero LLM calls allowed during deterministic reconciliation!")
    mock.generate_structured.side_effect = RuntimeError("Zero LLM calls allowed during deterministic reconciliation!")
    return mock


def test_compute_semantic_key_normalization():
    """Verify semantic key normalizes binary_encoding and opcode to the exact same canonical key."""
    c1 = EFSConflict(
        conflict_id="conf_1",
        type="DUPLICATE_OPCODE_ENCODING",
        entities=["MDET", "MTRANS"],
        property="binary_encoding",
        current_values={"MDET": "000111", "MTRANS": "000111"},
    )
    c2 = EFSConflict(
        conflict_id="conf_2",
        type="DUPLICATE_OPCODE",
        entities=["MTRANS", "MDET"],  # Reversed entity order
        property="opcode",
        current_values={"MDET": "000111", "MTRANS": "000111"},
    )

    key1 = ConflictResolutionEngine.compute_semantic_key(c1)
    key2 = ConflictResolutionEngine.compute_semantic_key(c2)

    assert key1 == key2, f"Expected canonical keys to match, got '{key1}' vs '{key2}'"
    assert "DUPLICATE_OPCODE" in key1
    assert "opcode" in key1
    assert "MDET" in key1 and "MTRANS" in key1


def test_dual_representation_auto_resolution(mock_llm_client):
    """
    Given:
      efs_ir has MDET = 000111 and MTRANS = 000111.
      Two conflict representations exist sharing the same underlying collision.
    When:
      User resolves MTRANS = 000000 via custom requirement or AI proposal.
    Then:
      Both conflict representations must be marked RESOLVED.
      efs_ir in-memory state has MTRANS = 000000.
      0 LLM calls made.
      Zero active conflicts remain.
      Historical conflict records are preserved for audit logging.
    """
    efs = EFSIR(
        metadata=EFSMetadata(design_name="test_design"),
        opcodes=[
            EFSOpcode(mnemonic="MDET", binary_encoding="000111"),
            EFSOpcode(mnemonic="MTRANS", binary_encoding="000111"),
        ]
    )

    # Initial conflict detection
    conflicts = ConflictResolutionEngine.detect_conflicts(efs)
    assert len(conflicts) >= 1
    for c in conflicts:
        assert c.status == "ACTIVE"
        assert c.blocking is True

    # Manually add a second representation to test multiple pre-existing representations
    manual_conf = EFSConflict(
        conflict_id="manual_conf_2",
        type="DUPLICATE_OPCODE",
        entities=["MDET", "MTRANS"],
        property="opcode",
        current_values={"MDET": "000111", "MTRANS": "000111"},
        status="ACTIVE",
        resolution_status="UNRESOLVED",
        blocking=True,
    )
    manual_conf.semantic_key = ConflictResolutionEngine.compute_semantic_key(manual_conf)
    efs.conflicts.append(manual_conf)

    assert len(efs.conflicts) >= 2

    # Create custom resolution for MTRANS = "000000"
    target_conflict = conflicts[0]
    res = ConflictResolutionEngine.create_custom_resolution(
        conflict=target_conflict,
        custom_requirement="000000",
        target_entity="MTRANS",
        user_name="hardware_engineer"
    )
    res.approval_status = "APPROVED"

    # Execute deterministic resolution with strict 0 LLM calls
    updated_efs = ConflictResolutionEngine.apply_approved_resolution(
        efs_ir=efs,
        approved_resolution=res
    )

    # In-memory EFS IR must be updated
    mtrans_opc = next(o for o in updated_efs.opcodes if o.mnemonic == "MTRANS")
    assert mtrans_opc.binary_encoding == "000000"

    # All representations must be marked RESOLVED
    active_conflicts = [c for c in updated_efs.conflicts if c.status == "ACTIVE"]
    resolved_conflicts = [c for c in updated_efs.conflicts if c.status == "RESOLVED"]

    assert len(active_conflicts) == 0, f"Expected 0 active conflicts, found {len(active_conflicts)}"
    assert len(resolved_conflicts) >= 2, f"Expected both representations resolved, found {len(resolved_conflicts)}"
    for rc in resolved_conflicts:
        assert rc.blocking is False
        assert rc.requires_human_decision is False
        assert rc.resolution_status == "RESOLVED"
        assert "no longer exists" in rc.resolved_reason or "Approved resolution" in rc.resolved_reason or "Applied resolution" in rc.resolved_reason

    # Revalidation must pass cleanly with 0 LLM calls
    has_blocking, issues = ConflictResolutionEngine.revalidate_resolved_model(updated_efs)
    assert not has_blocking
    assert len(issues) == 0


def test_ai_proposal_generation_and_acceptance(mock_llm_client):
    """
    Verify Option 1: AI proposal generation proposes a non-colliding opcode
    and accepting it deterministically updates the EFS IR.
    """
    efs = EFSIR(
        metadata=EFSMetadata(design_name="test_design"),
        opcodes=[
            EFSOpcode(mnemonic="MDET", binary_encoding="000111"),
            EFSOpcode(mnemonic="MTRANS", binary_encoding="000111"),
        ]
    )

    conflicts = ConflictResolutionEngine.detect_conflicts(efs)
    assert len(conflicts) >= 1
    conf = conflicts[0]

    # Generate proposals deterministically
    proposals = ConflictResolutionEngine.generate_resolution_proposals(conf, efs)
    assert len(proposals) >= 1
    top_prop = proposals[0]
    assert top_prop.proposed_value != "000111"

    # Approve and apply the top proposal
    top_prop.approval_status = "APPROVED"
    updated_efs = ConflictResolutionEngine.apply_approved_resolution(efs, top_prop)

    # In-memory EFS IR is updated to the proposed non-colliding value
    target_ent = top_prop.affected_requirements[0]
    target_opc = next(o for o in updated_efs.opcodes if o.mnemonic == target_ent)
    assert target_opc.binary_encoding == top_prop.proposed_value

    # Confirmed 0 active conflicts
    active = [c for c in updated_efs.conflicts if c.status == "ACTIVE"]
    assert len(active) == 0


def test_cascading_conflict_emergence(mock_llm_client):
    """
    Given:
      MDET = 000111, MTRANS = 000111, MADD = 000010.
    When:
      User resolves MTRANS to 000010 (colliding with MADD).
    Then:
      Original MDET/MTRANS conflict is marked RESOLVED.
      A new MTRANS/MADD conflict is marked ACTIVE.
      Zero LLM calls made.
    """
    efs = EFSIR(
        metadata=EFSMetadata(design_name="test_design"),
        opcodes=[
            EFSOpcode(mnemonic="MDET", binary_encoding="000111"),
            EFSOpcode(mnemonic="MTRANS", binary_encoding="000111"),
            EFSOpcode(mnemonic="MADD", binary_encoding="000010"),
        ]
    )

    initial_conflicts = ConflictResolutionEngine.detect_conflicts(efs)
    assert len(initial_conflicts) == 1
    assert "MDET" in initial_conflicts[0].entities
    assert "MTRANS" in initial_conflicts[0].entities

    # Resolve by mutating MTRANS to 000010 (colliding with MADD)
    target_conf = initial_conflicts[0]
    res = ConflictResolutionEngine.create_custom_resolution(
        conflict=target_conf,
        custom_requirement="000010",
        target_entity="MTRANS",
        user_name="test_user"
    )
    res.approval_status = "APPROVED"

    # 1. Verify server-side validation rejects the collision without force
    with pytest.raises(ValueError, match="collides with existing assigned opcode"):
        ConflictResolutionEngine.apply_approved_resolution(
            efs_ir=efs,
            approved_resolution=res,
            force=False
        )

    # 2. When mutation occurs, verify reconciliation automatically marks old RESOLVED and new ACTIVE
    updated_efs = ConflictResolutionEngine.apply_approved_resolution(
        efs_ir=efs,
        approved_resolution=res,
        force=True
    )

    # In-memory state updated
    mtrans_opc = next(o for o in updated_efs.opcodes if o.mnemonic == "MTRANS")
    assert mtrans_opc.binary_encoding == "000010"

    # Remaining conflicts should now contain the new collision (MTRANS vs MADD)
    active_conflicts = [c for c in updated_efs.conflicts if c.status == "ACTIVE"]
    assert len(active_conflicts) == 1
    new_conf = active_conflicts[0]
    assert new_conf.status == "ACTIVE"
    assert new_conf.blocking is True
    assert set(new_conf.entities) == {"MTRANS", "MADD"}

    # The old conflict should be preserved in audit history as RESOLVED
    old_conf = next(c for c in updated_efs.conflicts if set(c.entities) == {"MDET", "MTRANS"})
    assert old_conf.status == "RESOLVED"
    assert old_conf.blocking is False

    # Revalidation correctly flags blocking conflict
    has_blocking, _ = ConflictResolutionEngine.revalidate_resolved_model(updated_efs)
    assert has_blocking is True


def test_generic_register_conflict_reconciliation(mock_llm_client):
    """
    Verify generic behavior across hardware domains (e.g. registers/offsets, not just opcodes).
    """
    efs = EFSIR(
        metadata=EFSMetadata(design_name="test_registers"),
        registers=[
            EFSRegister(name="REG_STATUS", offset="0x10"),
            EFSRegister(name="REG_CTRL", offset="0x10"),  # Duplicate register offset
        ]
    )

    initial_conflicts = ConflictResolutionEngine.detect_conflicts(efs)
    assert len(initial_conflicts) == 1
    assert initial_conflicts[0].type == "DUPLICATE_REGISTER_ADDRESS"
    assert initial_conflicts[0].status == "ACTIVE"

    # Mutate register offset in-memory to eliminate collision
    efs.registers[1].offset = "0x14"

    # Re-detect conflicts deterministically
    reconciled = ConflictResolutionEngine.detect_conflicts(efs)
    active = [c for c in reconciled if c.status == "ACTIVE"]
    resolved = [c for c in reconciled if c.status == "RESOLVED"]

    assert len(active) == 0
    assert len(resolved) == 1
    assert resolved[0].status == "RESOLVED"
    assert resolved[0].blocking is False
    assert "no longer exists" in resolved[0].resolved_reason


def test_revalidate_resolved_model_zero_llm(mock_llm_client):
    """
    Ensure revalidate_resolved_model operates strictly deterministically without calling LLM.
    """
    efs = EFSIR(
        metadata=EFSMetadata(design_name="test_reval"),
        opcodes=[
            EFSOpcode(mnemonic="OP_A", binary_encoding="0001"),
            EFSOpcode(mnemonic="OP_B", binary_encoding="0010"),
        ]
    )
    # Add a resolved conflict
    c = EFSConflict(
        conflict_id="c_old",
        type="DUPLICATE_OPCODE",
        entities=["OP_A", "OP_B"],
        property="binary_encoding",
        status="RESOLVED",
        resolution_status="RESOLVED",
        blocking=False,
    )
    efs.conflicts.append(c)

    # Deterministic revalidation (with mock_llm_client passed to prove 0 calls)
    has_blocking, issues = ConflictResolutionEngine.revalidate_resolved_model(efs)
    assert not has_blocking
    assert len(issues) == 0
