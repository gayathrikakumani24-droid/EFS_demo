"""
Unit tests for Document Isolation, Hash Validation, and Source Integrity Gate.
"""

from __future__ import annotations

import pytest
from core.requirement_ir.models import DocumentIR, RequirementIR, DocumentBlock, SourceLocation
from core.compiler.efs_compiler import RequirementToEFSCompiler
from core.pipeline_diagnostics import generate_pipeline_validation_report
from utils.models import ParsedDocument


def test_document_identity_propagation():
    """Verify document_id, document_hash, and ingestion_id propagate across IR stages."""
    doc_hash = "abc123sha256hash"
    ing_id = "ING_test001"
    doc_id = "DOC_test_subsystem"

    doc_ir = DocumentIR(
        document_id=doc_id,
        document_hash=doc_hash,
        ingestion_id=ing_id,
        source_type="docx",
        metadata={"filename": "matrix_compute_unit.docx"}
    )
    doc_ir.blocks.append(DocumentBlock(
        block_id="BLK_001",
        type="paragraph",
        text="Matrix Compute Accelerator Specification",
        source_location=SourceLocation(document_id=doc_id, document_hash=doc_hash, block_id="BLK_001")
    ))

    req_ir = RequirementIR(
        doc_ir=doc_ir,
        document_id=doc_id,
        document_hash=doc_hash,
        ingestion_id=ing_id
    )

    compiler = RequirementToEFSCompiler(req_ir)
    efs_ir = compiler.compile()

    assert efs_ir.metadata.document_id == doc_id
    assert efs_ir.metadata.document_hash == doc_hash
    assert efs_ir.metadata.ingestion_id == ing_id
    assert efs_ir.metadata.source_documents == ["matrix_compute_unit.docx"]


def test_source_validation_gate_mismatch_fails():
    """Verify compiler raises DOCUMENT_ID_MISMATCH when RequirementIR and DocumentIR document_ids mismatch."""
    doc_ir = DocumentIR(document_id="DOC_ORIGINAL", document_hash="hash1")
    req_ir = RequirementIR(doc_ir=doc_ir, document_id="DOC_FOREIGN", document_hash="hash2")

    compiler = RequirementToEFSCompiler(req_ir)
    with pytest.raises(ValueError, match="DOCUMENT_ID_MISMATCH"):
        compiler.compile()


def test_validation_report_detects_contamination():
    """Verify pipeline validation report detects foreign document contamination cleanly."""
    doc_ir = DocumentIR(document_id="DOC_SPEC_A", document_hash="hashA", metadata={"filename": "spec_a.docx"})
    req_ir = RequirementIR(doc_ir=doc_ir, document_id="DOC_SPEC_A")

    compiler = RequirementToEFSCompiler(req_ir)
    efs_ir = compiler.compile()
    efs_ir.metadata.source_documents = ["spec_a.docx", "amba_axi_protocol_spec.docx"]

    report = generate_pipeline_validation_report(req_ir, efs_ir)
    assert report["validation"]["status"] == "INVALID"
    assert report["source_validation"]["foreign_documents_detected"] is True
    assert "amba_axi_protocol_spec.docx" in report["contamination"]["foreign_sources"]
