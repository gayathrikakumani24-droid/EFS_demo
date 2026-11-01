"""
Comprehensive Regression Test Suite for Evidence-Backed Extraction,
Source-Backed Recovery, Completeness Validation, and Anti-Hallucination Guard.

Tests that:
1. Every extracted hardware property retains source evidence (document, section, text, lines).
2. Extraction is separated from inference; missing properties in IR trigger source search before HITL.
3. Source-backed recovery recovers explicit, partial, and inferred properties from the source document.
4. Dedicated relationship, interface, register, FSM, and constraint extractors operate protocol-agnostically.
5. Anti-hallucination guard strictly rejects AI proposals that lack source evidence (e.g. invented 0x00/0x04, fake ACK, fake 10ns).
6. Extraction quality metrics are computed accurately.
7. Uses matrix_compute_unit.docx as a regression fixture without hardcoding.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from typing import Any, Dict

from core.parsing import get_parser
from core.requirement_ir.models import DocumentIR, DocumentBlock, SourceLocation
from core.efs_ir.models import EFSIR, EFSConflict, EFSSignal, EFSRegister, EFSInterface, EFSComponent, DiagnosticType
from core.extraction.source_evidence import SourceEvidence, EvidenceBackedProperty, SourceEvidenceIndex
from core.extraction.dedicated_extractors import (
    DedicatedRelationshipExtractor,
    DedicatedInterfaceExtractor,
    DedicatedRegisterExtractor,
    DedicatedFSMExtractor,
    DedicatedConstraintExtractor
)
from core.extraction.source_recovery import SourceBackedRecovery
from core.verification.conflict_resolution_engine import ConflictResolutionEngine
from core.pipeline_diagnostics import calculate_extraction_quality_metrics


possible_paths = [
    os.path.join(os.path.dirname(__file__), "..", "matrix_compute_unit.docx"),
    os.path.join(os.path.dirname(__file__), "..", "data", "uploads", "matrix_compute_unit.docx"),
]
FIXTURE_PATH = next((p for p in possible_paths if os.path.exists(p)), possible_paths[0])


def _get_fixture_doc_ir() -> DocumentIR:
    """Helper to parse the matrix_compute_unit fixture into a DocumentIR."""
    assert os.path.exists(FIXTURE_PATH), f"Fixture not found at {FIXTURE_PATH}"
    parser = get_parser(FIXTURE_PATH, "matrix_compute_unit.docx")
    parsed_doc = parser.parse()
    blocks = [
        DocumentBlock(
            block_id=f"BLK_{i+1:03d}",
            type="text",
            section_id=getattr(s, "section_id", f"SEC_{i+1:03d}") or f"SEC_{i+1:03d}",
            section_title=getattr(s, "title", "") or f"Section {i+1}",
            page=getattr(s, "page_start", 1) or 1,
            text=s.text,
            source_location=SourceLocation(
                document_id="matrix_compute_unit",
                document_hash="dummy_hash",
                section=getattr(s, "title", "") or f"Section {i+1}",
                section_id=getattr(s, "section_id", f"SEC_{i+1:03d}") or f"SEC_{i+1:03d}",
                page=getattr(s, "page_start", 1) or 1,
                block_id=f"BLK_{i+1:03d}",
                original_text=s.text[:200]
            )
        ) for i, s in enumerate(parsed_doc.sections) if getattr(s, "text", None) and s.text.strip()
    ]
    return DocumentIR(
        document_id="matrix_compute_unit",
        metadata={"filename": "matrix_compute_unit.docx"},
        sections=[s.to_dict() if hasattr(s, "to_dict") else s for s in parsed_doc.sections],
        blocks=blocks
    )


# ==============================================================================
# 1. Evidence-Backed Extraction Tests
# ==============================================================================

def test_evidence_backed_property_structure():
    """Verify EvidenceBackedProperty retains provenance and supports explicit, partial, inferred statuses."""
    ev = SourceEvidence(
        document="matrix_compute_unit.docx",
        section="Clock & Reset",
        text="Clock : System clock input",
        line_start=15,
        line_end=15,
        block_id="BLK_015"
    )
    prop = EvidenceBackedProperty(
        name="clock",
        value="clk",
        status="explicit",
        evidence=[ev],
        confidence=1.0
    )
    d = prop.to_dict()
    assert d["value"] == "clk"
    assert d["status"] == "explicit"
    assert len(d["evidence"]) == 1
    assert d["evidence"][0]["document"] == "matrix_compute_unit.docx"
    assert d["evidence"][0]["section"] == "Clock & Reset"
    assert d["evidence"][0]["line_start"] == 15
    assert d["confidence"] == 1.0


def test_source_evidence_index_search():
    """Verify SourceEvidenceIndex indexes blocks and retrieves matching evidence with line numbers."""
    doc_ir = _get_fixture_doc_ir()
    index = SourceEvidenceIndex(doc_ir)
    
    # Search for clock evidence
    clk_evs = index.search(["clock", "system clock"])
    assert len(clk_evs) > 0
    first_ev = clk_evs[0]
    assert first_ev.document == "matrix_compute_unit.docx"
    assert first_ev.line_start is not None
    assert "clock" in first_ev.text.lower()


# ==============================================================================
# 2. Dedicated Extractors Tests
# ==============================================================================

def test_dedicated_relationship_extractor():
    """Verify relationship extractor detects relational verbs with provenance."""
    doc_ir = _get_fixture_doc_ir()
    index = SourceEvidenceIndex(doc_ir)
    rels = DedicatedRelationshipExtractor.extract_relationships(doc_ir, evidence_index=index)
    
    assert len(rels) > 0
    verbs = {r.relation for r in rels}
    # Should detect verbs like controls, contains, connects_to, interfaces_with
    assert any(v in verbs for v in ("controls", "connects_to", "interfaces_with", "communicates_with", "distributes_to", "reads_from", "writes_to"))
    
    # Verify provenance
    for r in rels:
        assert r.evidence is not None
        assert r.evidence.document == "matrix_compute_unit.docx"
        assert r.evidence.text != ""


def test_dedicated_interface_extractor():
    """Verify interface extractor extracts explicit interfaces without hallucinating missing ones."""
    doc_ir = _get_fixture_doc_ir()
    index = SourceEvidenceIndex(doc_ir)
    ifaces = DedicatedInterfaceExtractor.extract_interfaces(doc_ir, evidence_index=index)
    
    assert len(ifaces) > 0
    for iface in ifaces:
        assert iface.name != ""
        assert iface.source_evidence is not None
        assert iface.source_evidence.text != ""


def test_dedicated_register_extractor_distinguishes_category_from_explicit_offsets():
    """
    Verify register extractor distinguishes explicit registers vs categories with address=None.
    MUST NOT invent fake addresses like 0x00, 0x04.
    """
    doc_ir = _get_fixture_doc_ir()
    index = SourceEvidenceIndex(doc_ir)
    regs = DedicatedRegisterExtractor.extract_registers(doc_ir, evidence_index=index)
    
    assert len(regs) > 0
    # In matrix_compute_unit, Matrix Registers M0-M7 are specified without hex addresses
    matrix_regs = [r for r in regs if "matrix" in r.name.lower() or "m0" in r.name.lower()]
    assert len(matrix_regs) > 0
    for mr in matrix_regs:
        assert mr.status in ("explicit", "partial")
        # Must NOT have hallucinated 0x00, 0x04, etc.
        assert mr.address is None or mr.offset is None or mr.offset == ""


def test_dedicated_constraint_extractor_extracts_source_constraints():
    """Verify constraint extractor recovers frequency (1Ghz) and throughput from source without guessing defaults."""
    doc_ir = _get_fixture_doc_ir()
    index = SourceEvidenceIndex(doc_ir)
    constraints = DedicatedConstraintExtractor.extract_constraints(doc_ir, evidence_index=index)
    
    assert len(constraints) > 0
    freq_constraints = [c for c in constraints if c.category == "frequency" or "1ghz" in str(c.value).lower()]
    assert len(freq_constraints) > 0
    assert "1ghz" in freq_constraints[0].value.lower() or "1 ghz" in freq_constraints[0].value.lower()
    assert freq_constraints[0].evidence.text != ""


def test_dedicated_fsm_extractor_does_not_invent_ack():
    """Verify FSM extractor derives FSM from control signals and does NOT invent fake ACK transition if absent."""
    doc_ir = _get_fixture_doc_ir()
    index = SourceEvidenceIndex(doc_ir)
    fsm_data = DedicatedFSMExtractor.extract_fsm(doc_ir, evidence_index=index)
    
    assert fsm_data is not None
    assert len(fsm_data.states) >= 2
    # Verify transitions do not include hallucinated 'on ack' if ACK is not in source
    for t in fsm_data.transitions:
        cond = t.get("condition", "").lower()
        if "ack" in cond:
            # Only allow if source explicitly mentions ack
            assert "ack" in fsm_data.evidence.text.lower()


# ==============================================================================
# 3. Source-Backed Recovery Tests
# ==============================================================================

def test_source_backed_recovery_populates_missing_efs_properties():
    """Verify SourceBackedRecovery searches source evidence and recovers clock, reset, registers, relationships, and constraints."""
    doc_ir = _get_fixture_doc_ir()
    empty_efs = EFSIR()
    
    report = SourceBackedRecovery.recover_efs_ir(empty_efs, doc_ir)
    
    assert report.explicit_recovered > 0
    # Clocks and resets should be recovered from Section 15
    assert len(empty_efs.signals) > 0
    clk_sigs = [s for s in empty_efs.signals if "clk" in s.name.lower()]
    rst_sigs = [s for s in empty_efs.signals if "rst" in s.name.lower()]
    assert len(clk_sigs) > 0
    assert len(rst_sigs) > 0
    
    # Registers should be recovered from Section 16
    assert len(empty_efs.registers) > 0
    
    # Relationships/interconnects should be recovered
    assert len(empty_efs.interfaces) > 0
    
    # Performance/constraints should be recovered from Section 18
    assert len(empty_efs.performance_requirements) > 0 or len(empty_efs.timing_rules) > 0


# ==============================================================================
# 4. Completeness Validator & Anti-Hallucination Guard Tests
# ==============================================================================

def test_completeness_validator_does_not_flag_source_backed_properties():
    """
    Verify completeness validator searches source evidence before flagging gaps.
    For matrix_compute_unit, clock, reset, registers, relationships, and constraints
    are present in the source, so they must NOT be reported as missing conflicts.
    """
    doc_ir = _get_fixture_doc_ir()
    empty_efs = EFSIR()
    
    # Run validate_hardware_completeness with doc_ir
    conflicts = ConflictResolutionEngine.validate_hardware_completeness(
        empty_efs,
        plan={"architecture": "matrix compute unit"},
        doc_ir=doc_ir
    )
    
    # Should not report Missing Clock Signal or Missing Reset Signal or Missing Hardware Constraints
    conflict_props = [c.property for c in conflicts]
    assert "clock" not in conflict_props
    assert "reset" not in conflict_props
    assert "constraints" not in conflict_props


def test_anti_hallucination_guard_rejects_unsupported_ai_proposals():
    """
    Verify anti-hallucination guard rejects AI proposals that lack source evidence
    or introduce invented default hardware values (e.g. 0x00, 0x04, fake ACK, fake 10ns).
    """
    efs_ir = EFSIR()
    conflict = EFSConflict(
        type="HARDWARE_COMPLETENESS_GAP",
        diagnostic_type=DiagnosticType.MISSING_SPECIFICATION.value,
        property="registers",
        affected_entities=["Registers"],
        evidence="Missing from EFS IR"
    )
    
    # 1. Invented default register map
    val_fake_reg = "CTRL@0x00, STATUS@0x04, DATA@0x08"
    valid, reason = ConflictResolutionEngine.validate_resolution(conflict, val_fake_reg, efs_ir)
    assert not valid
    assert "unsupported_ai_inference" in reason
    
    # 2. Invented default FSM transition with ACK
    conflict_fsm = EFSConflict(
        type="HARDWARE_COMPLETENESS_GAP",
        diagnostic_type=DiagnosticType.MISSING_SPECIFICATION.value,
        property="fsm",
        affected_entities=["FSM"],
        evidence="Missing from EFS IR"
    )
    val_fake_fsm = "IDLE -> RUN on start, RUN -> DONE on complete, DONE -> IDLE on ack"
    valid, reason = ConflictResolutionEngine.validate_resolution(conflict_fsm, val_fake_fsm, efs_ir)
    assert not valid
    assert "unsupported_ai_inference" in reason
    
    # 3. Invented default clock period
    conflict_clk = EFSConflict(
        type="HARDWARE_COMPLETENESS_GAP",
        diagnostic_type=DiagnosticType.MISSING_SPECIFICATION.value,
        property="constraints",
        affected_entities=["Constraints"],
        evidence="Missing from EFS IR"
    )
    val_fake_clk = "clk period = 10ns"
    valid, reason = ConflictResolutionEngine.validate_resolution(conflict_clk, val_fake_clk, efs_ir)
    assert not valid
    assert "unsupported_ai_inference" in reason


def test_anti_hallucination_guard_accepts_source_grounded_proposals():
    """Verify anti-hallucination guard accepts proposals that ARE supported by source evidence."""
    efs_ir = EFSIR()
    conflict = EFSConflict(
        type="HARDWARE_COMPLETENESS_GAP",
        diagnostic_type=DiagnosticType.MISSING_SPECIFICATION.value,
        property="constraints",
        affected_entities=["Constraints"],
        evidence="Maximum clock frequency: 1Ghz (Section 18)",
        source_references=[{
            "entity": "Constraints",
            "source": "Document Source",
            "value": "Maximum clock frequency: 1Ghz"
        }]
    )
    val_grounded = "1Ghz"
    valid, reason = ConflictResolutionEngine.validate_resolution(conflict, val_grounded, efs_ir)
    assert valid
    assert "passed with source evidence" in reason.lower()


# ==============================================================================
# 5. Extraction Quality Metrics Tests
# ==============================================================================

def test_extraction_quality_metrics_calculation():
    """Verify calculate_extraction_quality_metrics computes recalls, coverage, and property status counts."""
    doc_ir = _get_fixture_doc_ir()
    efs_ir = EFSIR()
    SourceBackedRecovery.recover_efs_ir(efs_ir, doc_ir)
    
    metrics = calculate_extraction_quality_metrics(efs_ir, doc_ir=doc_ir)
    
    assert "entity_recall" in metrics
    assert "relationship_recall" in metrics
    assert "interface_recall" in metrics
    assert "register_recall" in metrics
    assert "constraint_recall" in metrics
    assert "fsm_recall" in metrics
    assert "evidence_coverage" in metrics
    assert "explicit_properties_found" in metrics
    assert "partial_properties_found" in metrics
    assert "inferred_properties_found" in metrics
    assert "unresolved_properties" in metrics
    assert "unsupported_ai_proposals" in metrics
    
    assert metrics["evidence_coverage"] > 0.0
    assert metrics["explicit_properties_found"] > 0


if __name__ == "__main__":
    tests = [
        test_evidence_backed_property_structure,
        test_source_evidence_index_search,
        test_dedicated_relationship_extractor,
        test_dedicated_interface_extractor,
        test_dedicated_register_extractor_distinguishes_category_from_explicit_offsets,
        test_dedicated_constraint_extractor_extracts_source_constraints,
        test_dedicated_fsm_extractor_does_not_invent_ack,
        test_source_backed_recovery_populates_missing_efs_properties,
        test_completeness_validator_does_not_flag_source_backed_properties,
        test_anti_hallucination_guard_rejects_unsupported_ai_proposals,
        test_anti_hallucination_guard_accepts_source_grounded_proposals,
        test_extraction_quality_metrics_calculation,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASSED: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAILED: {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\nSummary: {passed} passed, {failed} failed.")
    if failed > 0:
        exit(1)
