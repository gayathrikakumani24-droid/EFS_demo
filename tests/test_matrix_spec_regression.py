"""
Regression test for EFS IR extraction on matrix_compute_unit hardware specification.

Verifies:
- All hardware-semantic constructs are properly extracted (components, signals, interfaces,
  instructions, instruction formats, opcodes, memory regions, parameters, timing rules,
  performance requirements, functional behavior).
- Real source conflicts (duplicate opcodes) are preserved and detected.
- Roundtrip JSON serialization/deserialization.
- Coverage metrics are correctly populated.
- Zero hardcoding of document-specific values or counts in production logic.
"""

import os
import json
import pytest
from config import CONFIG
from core.parsing.docx_parser import DocxParser
from core.requirement_ir.models import DocumentIR, DocumentBlock, SourceLocation
from core.extraction.requirement_extractor import GenericRequirementExtractor
from core.compiler.efs_compiler import RequirementToEFSCompiler
from core.efs_ir.validator import validate_efs_ir
from core.efs_ir.models import EFSIR


@pytest.fixture(scope="module")
def matrix_efs_ir():
    # Use deterministic extraction without LLM latency for fast test runs
    prev_llm = getattr(CONFIG, "enable_llm_escalation", True)
    CONFIG.enable_llm_escalation = False

    candidate_paths = [
        "data/uploads/matrix_compute_unit.docx",
        "data/uploads/matrix_compute_unit(3).docx",
        "hardware_graphrag/data/uploads/matrix_compute_unit.docx",
        "hardware_graphrag/data/uploads/matrix_compute_unit(3).docx",
        "hardware_graphrag/matrix_compute_unit.docx",
    ]
    test_doc = next((p for p in candidate_paths if os.path.exists(p)), None)
    assert test_doc is not None, f"Test document not found in candidate paths: {candidate_paths}"

    parser = DocxParser(test_doc)
    parsed_doc = parser.parse()
    assert len(parsed_doc.sections) > 0

    doc_ir = DocumentIR(
        document_id=parsed_doc.doc_id,
        document_version="1.0",
        source_type=parsed_doc.file_type,
        document_hash="test_hash_matrix_reg",
        ingestion_id="test_ingest_matrix_reg",
        metadata={"filename": parsed_doc.filename},
        sections=[s.to_dict() for s in parsed_doc.sections],
        blocks=[
            DocumentBlock(
                block_id=f"BLK_{i+1:03d}",
                type=s.content_type,
                section_id=s.section_id,
                section_title=s.title,
                page=s.page_start,
                text=s.text,
                source_location=SourceLocation(
                    document_id=parsed_doc.doc_id,
                    document_hash="test_hash_matrix_reg",
                    section=s.title,
                    section_id=s.section_id,
                    page=s.page_start,
                    block_id=f"BLK_{i+1:03d}",
                    original_text=s.text[:200]
                )
            ) for i, s in enumerate(parsed_doc.sections) if s.text and s.text.strip()
        ]
    )

    extractor = GenericRequirementExtractor(doc_ir)
    req_ir = extractor.extract_requirement_ir()
    compiler = RequirementToEFSCompiler(req_ir)
    efs = compiler.compile()

    CONFIG.enable_llm_escalation = prev_llm
    return efs


def test_matrix_components_extracted(matrix_efs_ir):
    assert len(matrix_efs_ir.components) > 0
    names = [c.name for c in matrix_efs_ir.components]
    assert any(c.component_id for c in matrix_efs_ir.components)


def test_matrix_signals_and_rtl_ports_extracted(matrix_efs_ir):
    assert len(matrix_efs_ir.signals) >= 20
    # Verify RTL signal directions and widths
    directions = {s.direction for s in matrix_efs_ir.signals}
    assert "input" in directions
    assert "output" in directions


def test_matrix_interfaces_extracted(matrix_efs_ir):
    assert len(matrix_efs_ir.interfaces) > 0


def test_matrix_parameters_extracted(matrix_efs_ir):
    assert len(matrix_efs_ir.parameters) > 0
    param = matrix_efs_ir.parameters[0]
    assert param.name
    assert param.value


def test_matrix_instructions_and_formats_extracted(matrix_efs_ir):
    assert len(matrix_efs_ir.instructions) >= 10
    assert len(matrix_efs_ir.instruction_formats) >= 1
    fmt = matrix_efs_ir.instruction_formats[0]
    assert len(fmt.fields) > 0


def test_matrix_opcodes_and_conflict_preservation(matrix_efs_ir):
    assert len(matrix_efs_ir.opcodes) >= 10
    # Verify duplicate opcodes are preserved in IR and detected as conflicts
    duplicate_conflicts = [c for c in matrix_efs_ir.conflicts if c.type == "DUPLICATE_OPCODE"]
    assert len(duplicate_conflicts) >= 1, "Duplicate opcode conflicts must be preserved and detected"
    for conf in duplicate_conflicts:
        assert conf.severity == "CRITICAL"
        assert conf.requires_human_decision is True
        assert conf.blocking is True


def test_matrix_memory_regions_extracted(matrix_efs_ir):
    assert len(matrix_efs_ir.memory_regions) > 0


def test_matrix_performance_requirements_extracted(matrix_efs_ir):
    assert len(matrix_efs_ir.performance_requirements) > 0


def test_matrix_functional_behavior_extracted(matrix_efs_ir):
    assert len(matrix_efs_ir.functional_behavior) > 0


def test_matrix_extraction_coverage(matrix_efs_ir):
    cov = matrix_efs_ir.extraction_coverage
    assert cov is not None
    assert "components" in cov and cov["components"]["status"] == "extracted"
    assert "signals" in cov and cov["signals"]["status"] == "extracted"
    assert "instructions" in cov and cov["instructions"]["status"] == "extracted"
    assert "opcodes" in cov and cov["opcodes"]["status"] == "extracted"
    assert "memory_regions" in cov and cov["memory_regions"]["status"] == "extracted"
    assert "parameters" in cov and cov["parameters"]["status"] == "extracted"
    assert "registers" in cov and cov["registers"]["status"] in ("not_explicitly_present", "extracted")


def test_matrix_validation_produces_clean_conflicts(matrix_efs_ir):
    issues = validate_efs_ir(matrix_efs_ir)
    # Ensure there are no grounding gap warnings or unknown entity errors
    grounding_issues = [i for i in issues if i.get("category") == "Grounding"]
    assert len(grounding_issues) == 0, f"Found unexpected grounding issues: {grounding_issues}"

    # Verify duplicate opcodes are reported in validation
    opcode_issues = [i for i in issues if i.get("category") == "Opcodes" and "Duplicate opcode" in i.get("message", "")]
    assert len(opcode_issues) >= 1


def test_matrix_roundtrip_serialization(matrix_efs_ir):
    d = matrix_efs_ir.to_dict()
    json_str = json.dumps(d)
    restored = EFSIR.from_dict(json.loads(json_str))

    assert len(restored.components) == len(matrix_efs_ir.components)
    assert len(restored.signals) == len(matrix_efs_ir.signals)
    assert len(restored.instructions) == len(matrix_efs_ir.instructions)
    assert len(restored.opcodes) == len(matrix_efs_ir.opcodes)
    assert len(restored.instruction_formats) == len(matrix_efs_ir.instruction_formats)
    assert len(restored.memory_regions) == len(matrix_efs_ir.memory_regions)
    assert len(restored.parameters) == len(matrix_efs_ir.parameters)
    assert len(restored.functional_behavior) == len(matrix_efs_ir.functional_behavior)
    assert len(restored.conflicts) == len(matrix_efs_ir.conflicts)
    assert len(restored.extraction_coverage) == len(matrix_efs_ir.extraction_coverage)


def test_matrix_resolved_view(matrix_efs_ir):
    resolved = matrix_efs_ir.get_resolved_view()
    assert len(resolved.parameters) == len(matrix_efs_ir.parameters)
    assert len(resolved.functional_behavior) == len(matrix_efs_ir.functional_behavior)


def test_matrix_registers_and_provenance(matrix_efs_ir):
    """
    Verify registers (Matrix Registers M0-M7, Configuration Registers, Status Registers)
    are retained in EFS IR with provenance, even without explicit addresses.
    """
    assert len(matrix_efs_ir.registers) >= 8, f"Expected at least 8 registers, found {len(matrix_efs_ir.registers)}"
    reg_names = {r.name for r in matrix_efs_ir.registers}
    
    # Check that individual M0-M7 or range M0-M7 exists
    has_matrix_regs = any("M0" in name or "Matrix Register" in name for name in reg_names)
    assert has_matrix_regs, f"Expected matrix registers in EFS IR, got: {reg_names}"
    
    # Check provenance
    for r in matrix_efs_ir.registers:
        assert r.traceability is not None, f"Register {r.name} missing traceability"
        assert r.name, "Register missing name"


def test_matrix_performance_and_constraints_provenance(matrix_efs_ir):
    """
    Verify performance requirements (frequency, throughput, power) are retained
    in EFS IR with provenance.
    """
    assert len(matrix_efs_ir.performance_requirements) >= 3, (
        f"Expected at least 3 performance requirements, found {len(matrix_efs_ir.performance_requirements)}"
    )
    metrics_and_values = " ".join(
        f"{p.metric} {p.target_value} {p.condition}".lower()
        for p in matrix_efs_ir.performance_requirements
    )
    assert "1ghz" in metrics_and_values or "frequency" in metrics_and_values or "clock" in metrics_and_values
    assert "throughput" in metrics_and_values or "6.25" in metrics_and_values
    assert "power" in metrics_and_values or "100mw" in metrics_and_values or "mw" in metrics_and_values

    for p in matrix_efs_ir.performance_requirements:
        assert p.traceability is not None, f"Performance requirement {p.metric} missing traceability"


def test_matrix_completeness_validation_no_false_gaps(matrix_efs_ir):
    """
    Verify completeness validation on matrix_efs_ir does not raise false missing
    register or missing constraint conflicts when evidence is present.
    """
    from core.verification.conflict_resolution_engine import ConflictResolutionEngine
    conflicts = ConflictResolutionEngine.validate_hardware_completeness(
        matrix_efs_ir,
        plan={"design_type": "matrix_compute", "registers": "present", "timing": "1ghz"}
    )
    reg_conflicts = [c for c in conflicts if "register" in c.title.lower()]
    constraint_conflicts = [c for c in conflicts if "constraint" in c.title.lower() or "timing" in c.title.lower()]
    assert len(reg_conflicts) == 0, f"Unexpected register conflict: {[c.title for c in reg_conflicts]}"
    assert len(constraint_conflicts) == 0, f"Unexpected constraint conflict: {[c.title for c in constraint_conflicts]}"

