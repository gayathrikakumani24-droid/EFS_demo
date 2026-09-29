"""
Generic Regression Tests for Opcode & Instruction Deduplication, Mention Canonicalization,
and Conflict Resolution Engine (Ensuring zero false conflicts on missing/unknown data).
"""

import pytest
from core.requirement_ir.models import (
    DocumentIR, DocumentBlock, SourceLocation, DiscoveredEntity, RequirementIR
)
from core.extraction.requirement_extractor import GenericRequirementExtractor
from core.efs_ir.builder import EFSIRBuilder
from core.efs_ir.models import EFSIR, EFSOpcode, EFSInstruction, EFSTraceability
from core.verification.conflict_resolution_engine import ConflictResolutionEngine, is_explicit_encoding


def test_is_explicit_encoding():
    """Verify that missing/placeholder values are not recognized as explicit encodings."""
    assert not is_explicit_encoding("")
    assert not is_explicit_encoding(None)
    assert not is_explicit_encoding("UNKNOWN")
    assert not is_explicit_encoding("unknown")
    assert not is_explicit_encoding("TBD")
    assert not is_explicit_encoding("none")
    assert not is_explicit_encoding("N/A")
    assert not is_explicit_encoding("unresolved")
    assert not is_explicit_encoding("-")

    # Explicit values must be recognized
    assert is_explicit_encoding("0x01")
    assert is_explicit_encoding("0001")
    assert is_explicit_encoding("6'b000001")
    assert is_explicit_encoding("12")


def test_repeated_mentions_deduplicated_to_single_entity():
    """
    Verify that an instruction mentioned across multiple chunks/sections
    produces a single canonical DiscoveredEntity with accumulated source references.
    """
    blocks = [
        DocumentBlock(
            block_id="BLK_001",
            type="text",
            section_title="Instruction Set Architecture",
            page=1,
            text="- ACCEL_EXEC [31:0]: opcode 0x10, Execute hardware operation.",
            source_location=SourceLocation(document_id="doc_test", block_id="BLK_001", page=1)
        ),
        DocumentBlock(
            block_id="BLK_002",
            type="text",
            section_title="Detailed Operation Flow",
            page=5,
            text="- ACCEL_EXEC: Execute hardware operation with tile configuration.",
            source_location=SourceLocation(document_id="doc_test", block_id="BLK_002", page=5)
        ),
        DocumentBlock(
            block_id="BLK_003",
            type="text",
            section_title="Summary of Commands",
            page=10,
            text="- ACCEL_EXEC [31:0]: opcode 0x10, Complete execution block.",
            source_location=SourceLocation(document_id="doc_test", block_id="BLK_003", page=10)
        ),
    ]
    doc_ir = DocumentIR(document_id="doc_test", blocks=blocks)
    extractor = GenericRequirementExtractor(doc_ir)
    req_ir = extractor.extract_requirement_ir()

    # Find the extracted instruction entity
    accel_insts = [e for e in req_ir.entities if e.name.upper() == "ACCEL_EXEC" and e.type.lower() == "instruction"]
    assert len(accel_insts) == 1, f"Expected exactly 1 canonical instruction entity, got {len(accel_insts)}"
    
    inst_ent = accel_insts[0]
    # Verify multiple source references were accumulated
    assert len(inst_ent.source_references) >= 2, f"Expected multiple source references, got {len(inst_ent.source_references)}"
    ref_blocks = {s.block_id for s in inst_ent.source_references}
    assert "BLK_001" in ref_blocks


def test_missing_encoding_does_not_create_duplicate_conflict():
    """
    Verify that multiple instructions with absent/unknown encoding do NOT
    trigger a DUPLICATE_OPCODE_ENCODING or DUPLICATE_OPCODE conflict.
    """
    efs_ir = EFSIR()
    # 5 instructions with no explicit encoding (empty or "")
    for name in ("CUSTOM_ADD", "CUSTOM_SUB", "CUSTOM_MUL", "CUSTOM_DIV", "CUSTOM_MOD"):
        efs_ir.opcodes.append(EFSOpcode(
            opcode_id=f"opc_{name.lower()}",
            mnemonic=name,
            binary_encoding="",
            traceability=EFSTraceability(doc_id="doc_test", chunk_id="SPEC_EXTRACTION")
        ))
        efs_ir.instructions.append(EFSInstruction(
            instruction_id=f"inst_{name.lower()}",
            name=name,
            mnemonic=name,
            opcode="",
            traceability=EFSTraceability(doc_id="doc_test", chunk_id="SPEC_EXTRACTION")
        ))

    engine = ConflictResolutionEngine()
    conflicts = engine.detect_conflicts(efs_ir)
    opcode_conflicts = [c for c in conflicts if c.type in ("DUPLICATE_OPCODE_ENCODING", "DUPLICATE_OPCODE")]
    assert len(opcode_conflicts) == 0, f"Expected 0 opcode conflicts for missing encodings, got {len(opcode_conflicts)}"


def test_repeated_evidence_same_entity_same_value_no_conflict():
    """
    Verify that multiple occurrences of the same instruction with the same encoding
    are treated as repeated evidence, NOT as a conflict.
    """
    efs_ir = EFSIR()
    # Same instruction mentioned 3 times with identical encoding '0x05'
    for i in range(3):
        efs_ir.opcodes.append(EFSOpcode(
            opcode_id=f"opc_eval_{i}",
            mnemonic="EVAL_FN",
            binary_encoding="0x05",
            traceability=EFSTraceability(doc_id="doc_test", chunk_id=f"BLK_{i}")
        ))

    engine = ConflictResolutionEngine()
    conflicts = engine.detect_conflicts(efs_ir)
    opcode_conflicts = [c for c in conflicts if c.type == "DUPLICATE_OPCODE_ENCODING"]
    assert len(opcode_conflicts) == 0, "Repeated mentions of same instruction with same value should not conflict"


def test_genuine_duplicate_encoding_conflict_detected():
    """
    Verify that two DIFFERENT instructions sharing the SAME explicit encoding
    are correctly detected as a DUPLICATE_OPCODE_ENCODING conflict.
    """
    efs_ir = EFSIR()
    efs_ir.opcodes.append(EFSOpcode(
        opcode_id="opc_1",
        mnemonic="VECTOR_ADD",
        binary_encoding="0x08",
        traceability=EFSTraceability(doc_id="doc_test", chunk_id="BLK_1")
    ))
    efs_ir.opcodes.append(EFSOpcode(
        opcode_id="opc_2",
        mnemonic="VECTOR_SUB",
        binary_encoding="0x08",
        traceability=EFSTraceability(doc_id="doc_test", chunk_id="BLK_2")
    ))

    engine = ConflictResolutionEngine()
    conflicts = engine.detect_conflicts(efs_ir)
    opcode_conflicts = [c for c in conflicts if c.type == "DUPLICATE_OPCODE_ENCODING"]
    assert len(opcode_conflicts) == 1
    assert "VECTOR_ADD" in opcode_conflicts[0].entities
    assert "VECTOR_SUB" in opcode_conflicts[0].entities
    assert opcode_conflicts[0].property == "binary_encoding"


def test_contradictory_opcode_same_instruction_detected():
    """
    Verify that the SAME instruction assigned contradictory explicit encodings
    is detected as a CONFLICTING_OPCODE_VALUE conflict.
    """
    efs_ir = EFSIR()
    efs_ir.opcodes.append(EFSOpcode(
        opcode_id="opc_1",
        mnemonic="DMA_START",
        binary_encoding="0x20",
        traceability=EFSTraceability(doc_id="doc_test", chunk_id="BLK_1")
    ))
    efs_ir.opcodes.append(EFSOpcode(
        opcode_id="opc_2",
        mnemonic="DMA_START",
        binary_encoding="0x25",
        traceability=EFSTraceability(doc_id="doc_test", chunk_id="BLK_2")
    ))

    engine = ConflictResolutionEngine()
    conflicts = engine.detect_conflicts(efs_ir)
    contradiction_conflicts = [c for c in conflicts if c.type == "CONFLICTING_OPCODE_VALUE"]
    assert len(contradiction_conflicts) == 1
    assert contradiction_conflicts[0].entities == ["DMA_START"]
    assert set(contradiction_conflicts[0].conflicting_values) == {"0x20", "0x25"}


def test_efs_ir_builder_does_not_synthesize_opc_or_unknown():
    """
    Verify that EFSIRBuilder maps actual entity names and encodings,
    and NEVER defaults to 'OPC' or 'UNKNOWN'.
    """
    req_model = {
        "opcodes": [
            {
                "name": "CONV2D_OPCODE",
                "type": "opcode",
                "attributes": {
                    "instruction": "CONV2D",
                    "opcode_hex": "0x3A"
                }
            },
            {
                "name": "POOLING_OPCODE",
                "type": "opcode",
                "attributes": {
                    "instruction": "POOLING"
                    # missing encoding
                }
            }
        ],
        "instructions": [
            {
                "name": "CONV2D",
                "mnemonic": "CONV2D",
                "opcode": "0x3A",
                "description": "2D Convolution operation"
            },
            {
                "name": "POOLING",
                "mnemonic": "POOLING",
                "description": "Spatial pooling operation"
            }
        ]
    }
    builder = EFSIRBuilder(req_model)
    efs_ir = builder.build()

    mnemonics = [o.mnemonic for o in efs_ir.opcodes]
    assert "CONV2D" in mnemonics
    assert "POOLING" in mnemonics
    assert "OPC" not in mnemonics

    conv_opc = next(o for o in efs_ir.opcodes if o.mnemonic == "CONV2D")
    assert conv_opc.binary_encoding == "0x3A"

    pool_opc = next(o for o in efs_ir.opcodes if o.mnemonic == "POOLING")
    # Must be empty/unresolved, NEVER "UNKNOWN"
    assert pool_opc.binary_encoding == ""
