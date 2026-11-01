"""
Comprehensive Regression & Safety Test Suite for False Entity and False Conflict Prevention.

Verifies:
1. Structural extraction prevents non-hardware tokens (e.g., 'contains', '1', 'Source', '2', 'Dimension', 'Addresses')
   from becoming registers.
2. No false DUPLICATE_REGISTER_ADDRESS conflicts are generated from malformed tables or missing offsets.
3. Missing offsets remain explicitly None (uncertainty preserved) and never default to '0x0'.
4. Explicit zero offsets ('0x0', '0x00') are strictly preserved as explicit hardware values.
5. True hardware register address conflicts (REG_X @ 0x10 and REG_Y @ 0x10) are detected accurately.
6. Natural language registers without addresses are preserved with offset=None without triggering false conflicts.
7. Intentional aliases/mirrored registers do not trigger false conflicts.
8. Reusable validation rules operate protocol-agnostically without hardcoded blacklists.
"""

import os
import sys
import pytest

# Ensure repository root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.efs_ir.models import (
    EFSIR, EFSRegister, EFSRegisterField, EFSSignal, EFSMetadata, EFSConflict,
    strip_nulls
)
from core.efs_ir.schema import validate_efs_ir_schema
from core.efs_ir.validator import validate_efs_ir
from core.efs_ir.validation_rules import (
    is_valid_register_candidate,
    has_sufficient_register_evidence,
    normalize_hardware_offset,
    is_explicit_address,
    is_missing_value,
    is_explicit_value,
)
from core.efs_ir.builder import EFSIRBuilder
from core.compiler.efs_compiler import RequirementToEFSCompiler
from core.verification.conflict_resolution_engine import ConflictResolutionEngine
from core.requirement_ir.models import (
    RequirementIR, DocumentIR, DocumentBlock, SourceLocation, DiscoveredEntity,
    AtomicRequirement
)
from core.extraction.requirement_extractor import GenericRequirementExtractor


# ==============================================================================
# 1. Validation Rules Unit Tests (Protocol-Agnostic, No Word Blacklists)
# ==============================================================================

def test_reusable_validation_rules_register_candidates():
    """Verify is_valid_register_candidate rejects non-hardware tokens without evidence."""
    # Pure numbers must never be registers
    assert is_valid_register_candidate("1") is False
    assert is_valid_register_candidate("2") is False
    assert is_valid_register_candidate("42") is False

    # English grammatical words / table labels without hardware evidence
    assert is_valid_register_candidate("contains") is False
    assert is_valid_register_candidate("Source") is False
    assert is_valid_register_candidate("Dimension") is False
    assert is_valid_register_candidate("Addresses") is False
    assert is_valid_register_candidate("Table") is False
    assert is_valid_register_candidate("Description") is False

    # Valid register names
    assert is_valid_register_candidate("CTRL") is True
    assert is_valid_register_candidate("STATUS") is True
    assert is_valid_register_candidate("CONFIG") is True
    assert is_valid_register_candidate("DMA_CTRL") is True
    assert is_valid_register_candidate("FIFO_STATUS") is True
    assert is_valid_register_candidate("INTR_ENABLE") is True
    assert is_valid_register_candidate("REG_X") is True
    assert is_valid_register_candidate("control_register") is True


def test_reusable_validation_rules_hardware_evidence_override():
    """If a generic word is supported by explicit hardware evidence (e.g. hex offset and access), evaluate appropriately."""
    attrs_with_evidence = {"offset": "0x14", "access": "RW"}
    # Word 'source' with explicit hardware offset and RW access could be a SOURCE_ADDR or SOURCE register
    assert is_valid_register_candidate("SOURCE_ADDR", attrs_with_evidence) is True

    # But bare digit '1' or '2' is never a register even if someone mistakenly put offset
    assert is_valid_register_candidate("1", attrs_with_evidence) is False


def test_normalize_hardware_offset_explicit_vs_missing():
    """Verify distinguish explicit zero from missing values."""
    # Explicit zeros must be preserved
    assert normalize_hardware_offset("0x0") == "0x0"
    assert normalize_hardware_offset("0x00") == "0x00"
    assert normalize_hardware_offset("0") == "0x0"
    assert normalize_hardware_offset(0) == "0x0"

    # Hex values
    assert normalize_hardware_offset("0x10") == "0x10"
    assert normalize_hardware_offset("0x04") == "0x04"
    assert normalize_hardware_offset(16) == "0x10"

    # Missing / Unassigned values must resolve to None
    assert normalize_hardware_offset(None) is None
    assert normalize_hardware_offset("") is None
    assert normalize_hardware_offset("   ") is None
    assert normalize_hardware_offset("TBD") is None
    assert normalize_hardware_offset("None") is None
    assert normalize_hardware_offset("null") is None
    assert normalize_hardware_offset("unassigned") is None
    assert normalize_hardware_offset("-") is None
    assert normalize_hardware_offset("--") is None
    assert normalize_hardware_offset("N/A") is None

    # Garbage non-hex text must resolve to None
    assert normalize_hardware_offset("contains") is None
    assert normalize_hardware_offset("Source") is None
    assert normalize_hardware_offset("Dimension") is None


def test_is_explicit_address():
    """Test is_explicit_address correctly identifies real addresses vs missing."""
    assert is_explicit_address("0x0") is True
    assert is_explicit_address("0x10") is True
    assert is_explicit_address(0) is True
    assert is_explicit_address(16) is True

    assert is_explicit_address(None) is False
    assert is_explicit_address("") is False
    assert is_explicit_address("TBD") is False
    assert is_explicit_address("unassigned") is False
    assert is_explicit_address("contains") is False


# ==============================================================================
# 2. Regression Test: Problematic Table Extraction & False Conflict Elimination
# ==============================================================================

def test_regression_malformed_table_no_fake_registers_no_conflict():
    """
    Regression Test for the exact failure observed:
    Table with tokens: contains, 1, Source, 2, Dimension, Addresses
    Must NOT produce fake registers with offset 0x0, and must NOT trigger DUPLICATE_REGISTER_ADDRESS.
    """
    doc_ir = DocumentIR(document_id="DOC_TEST_REGRESS")
    # Simulate table block with markdown text
    table_text = "| Source | Dimension | Addresses |\n| :--- | :--- | :--- |\n| contains | 1 | data |\n| Source | 2 | index |\n| Dimension | Addresses | computed |"
    table_block = DocumentBlock(
        block_id="BLK_TBL_1",
        type="table",
        section_title="Address Computation",
        text=table_text
    )
    doc_ir.blocks.append(table_block)

    # Run generic extraction
    extractor = GenericRequirementExtractor(doc_ir)
    extractor.llm._client = None
    req_ir = extractor.extract_requirement_ir()

    # Verify no fake registers were discovered from these non-register tokens
    discovered_regs = [e for e in req_ir.entities if e.type == "register"]
    discovered_reg_names = [e.name for e in discovered_regs]
    assert "contains" not in discovered_reg_names
    assert "1" not in discovered_reg_names
    assert "Source" not in discovered_reg_names
    assert "2" not in discovered_reg_names
    assert "Dimension" not in discovered_reg_names
    assert "Addresses" not in discovered_reg_names

    # Compile to canonical EFS IR
    compiler = RequirementToEFSCompiler(req_ir)
    efs_ir = compiler.compile()

    # Verify no fake registers in canonical EFS IR
    efs_reg_names = [r.name for r in efs_ir.registers]
    assert "contains" not in efs_reg_names
    assert "1" not in efs_reg_names
    assert "Source" not in efs_reg_names
    assert "2" not in efs_reg_names
    assert "Dimension" not in efs_reg_names
    assert "Addresses" not in efs_reg_names

    # Run conflict detection
    conflicts = ConflictResolutionEngine.detect_conflicts(efs_ir)
    dup_addr_conflicts = [c for c in conflicts if c.type == "DUPLICATE_REGISTER_ADDRESS"]

    # Invariant: NO DUPLICATE_REGISTER_ADDRESS conflict
    assert len(dup_addr_conflicts) == 0, f"False duplicate address conflict was generated: {dup_addr_conflicts}"


# ==============================================================================
# 3. Positive Register Test: Valid Registers with Explicit Addresses
# ==============================================================================

def test_positive_valid_registers_explicit_addresses():
    """
    Test valid register table:
    Register | Offset | Width | Access
    CTRL     | 0x00   | 32    | RW
    STATUS   | 0x04   | 32    | RO
    CONFIG   | 0x08   | 32    | RW
    Expected: 3 valid register entities, CTRL offset 0x0 explicit, zero conflicts.
    """
    doc_ir = DocumentIR(document_id="DOC_VALID_REGS")
    table_text = "| Register | Offset | Width | Access |\n| :--- | :--- | :--- | :--- |\n| CTRL | 0x00 | 32 | RW |\n| STATUS | 0x04 | 32 | RO |\n| CONFIG | 0x08 | 32 | RW |"
    table_block = DocumentBlock(
        block_id="BLK_REG_MAP",
        type="table",
        section_title="Register Map",
        text=table_text
    )
    doc_ir.blocks.append(table_block)

    extractor = GenericRequirementExtractor(doc_ir)
    extractor.llm._client = None
    req_ir = extractor.extract_requirement_ir()

    compiler = RequirementToEFSCompiler(req_ir)
    efs_ir = compiler.compile()

    # Must find all 3 registers
    assert len(efs_ir.registers) == 3
    ctrl = next(r for r in efs_ir.registers if r.name == "CTRL")
    status = next(r for r in efs_ir.registers if r.name == "STATUS")
    config = next(r for r in efs_ir.registers if r.name == "CONFIG")

    # Invariant 4: Explicit address 0x0 remains legitimate address and explicit
    assert ctrl.offset in ("0x0", "0x00")
    assert ctrl.offset_source == "explicit"
    assert status.offset in ("0x4", "0x04")
    assert config.offset in ("0x8", "0x08")

    # No duplicate address conflicts
    conflicts = ConflictResolutionEngine.detect_conflicts(efs_ir)
    dup_conflicts = [c for c in conflicts if c.type == "DUPLICATE_REGISTER_ADDRESS"]
    assert len(dup_conflicts) == 0


# ==============================================================================
# 4. True Conflict Test: Genuine Duplicate Address Detection
# ==============================================================================

def test_true_conflict_detected_for_same_explicit_address():
    """
    Test true conflict:
    REG_X -> 0x10
    REG_Y -> 0x10
    Expected: DUPLICATE_REGISTER_ADDRESS generated because both are valid entities with same explicit address.
    """
    efs_ir = EFSIR()
    efs_ir.metadata = EFSMetadata(design_name="top_mod")
    efs_ir.registers.append(EFSRegister(name="REG_X", offset="0x10", description="Register X"))
    efs_ir.registers.append(EFSRegister(name="REG_Y", offset="0x10", description="Register Y"))

    conflicts = ConflictResolutionEngine.detect_conflicts(efs_ir)
    dup_conflicts = [c for c in conflicts if c.type == "DUPLICATE_REGISTER_ADDRESS"]

    assert len(dup_conflicts) == 1
    assert "REG_X" in dup_conflicts[0].entities
    assert "REG_Y" in dup_conflicts[0].entities
    assert dup_conflicts[0].blocking is True


# ==============================================================================
# 5. Missing Values Test: Preserved as None and Never Collide
# ==============================================================================

def test_missing_offsets_remain_none_and_never_collide():
    """
    Test registers with missing offsets:
    REG_A (offset=None)
    REG_B (offset=None)
    Expected:
    - Both exist with offset=None, offset_source='missing'.
    - Invariant 3: An unknown address must never participate in duplicate-address detection.
    - Zero duplicate address conflicts.
    """
    efs_ir = EFSIR()
    efs_ir.metadata = EFSMetadata(design_name="core_mod")
    efs_ir.registers.append(EFSRegister(name="REG_A", offset=None, description="Control register A"))
    efs_ir.registers.append(EFSRegister(name="REG_B", offset=None, description="Control register B"))

    assert efs_ir.registers[0].offset is None
    assert efs_ir.registers[0].offset_source == "missing"
    assert efs_ir.registers[1].offset is None
    assert efs_ir.registers[1].offset_source == "missing"

    conflicts = ConflictResolutionEngine.detect_conflicts(efs_ir)
    dup_conflicts = [c for c in conflicts if c.type == "DUPLICATE_REGISTER_ADDRESS"]
    assert len(dup_conflicts) == 0


def test_missing_offset_does_not_collide_with_explicit_zero():
    """
    Test that a register with missing offset (None) does NOT collide with a register at explicit 0x0.
    REG_ZERO -> 0x00
    REG_UNKNOWN -> None
    Expected: Zero duplicate address conflicts.
    """
    efs_ir = EFSIR()
    efs_ir.registers.append(EFSRegister(name="REG_ZERO", offset="0x00", description="Primary control at zero"))
    efs_ir.registers.append(EFSRegister(name="REG_UNKNOWN", offset=None, description="General control unknown offset"))

    assert efs_ir.registers[0].offset in ("0x0", "0x00")
    assert efs_ir.registers[0].offset_source == "explicit"
    assert efs_ir.registers[1].offset is None
    assert efs_ir.registers[1].offset_source == "missing"

    conflicts = ConflictResolutionEngine.detect_conflicts(efs_ir)
    dup_conflicts = [c for c in conflicts if c.type == "DUPLICATE_REGISTER_ADDRESS"]
    assert len(dup_conflicts) == 0


# ==============================================================================
# 6. Intentional Aliasing / Mirrored Registers
# ==============================================================================

def test_intentional_alias_registers_do_not_generate_false_conflict():
    """
    When registers share an address because one is an alias or mirror view:
    REG_ALIAS is described as 'Alias view of REG_ORIG'
    Expected: No duplicate address conflict asserted.
    """
    efs_ir = EFSIR()
    efs_ir.registers.append(EFSRegister(name="REG_ORIG", offset="0x10", description="Original control register"))
    efs_ir.registers.append(EFSRegister(name="REG_ALIAS", offset="0x10", description="Alias view for backward compatibility"))

    conflicts = ConflictResolutionEngine.detect_conflicts(efs_ir)
    dup_conflicts = [c for c in conflicts if c.type == "DUPLICATE_REGISTER_ADDRESS"]
    assert len(dup_conflicts) == 0


# ==============================================================================
# 7. EFS IR Builder & Schema Serialization Roundtrip
# ==============================================================================

def test_efs_ir_builder_preserves_uncertainty_and_serializes_cleanly():
    """
    Verify EFSIRBuilder builds registers with None offset, serializes with strip_nulls preserving offset: null,
    and passes JSON schema validation.
    """
    req_model = {
        "registers": [
            {"name": "CTRL_REG", "offset": "0x00", "access": "RW", "description": "Control register"},
            {"name": "STATUS_REG", "offset": None, "access": "RO", "description": "Status register"}
        ]
    }
    builder = EFSIRBuilder(requirement_model=req_model)
    ir = builder.build()

    assert len(ir.registers) == 2
    ctrl = next(r for r in ir.registers if r.name == "CTRL_REG")
    status = next(r for r in ir.registers if r.name == "STATUS_REG")

    assert ctrl.offset == "0x00"
    assert ctrl.offset_source == "explicit"
    assert status.offset is None
    assert status.offset_source == "missing"

    # Serialization test
    d = ir.to_dict()
    regs_json = d["registers"]
    ctrl_json = next(r for r in regs_json if r["name"] == "CTRL_REG")
    status_json = next(r for r in regs_json if r["name"] == "STATUS_REG")

    assert ctrl_json["offset"] == "0x00"
    assert status_json["offset"] is None

    # Schema validation
    is_valid, errors = validate_efs_ir_schema(d)
    assert is_valid is True, f"Schema validation failed: {errors}"
