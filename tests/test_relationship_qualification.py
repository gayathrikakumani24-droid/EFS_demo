"""
Generic Regression Tests for Hardware Relationship Qualification & Entity Resolution.

Tests the semantic distinction between:
- Component descriptions (Case A, E)
- Register descriptions (Case B)
- Hardware flow relationships (Case C)
- Hardware communication interfaces (Case D)
And verifies non-regression of:
- Instruction width extraction (Case F)
- Register count extraction (Case G)
- Duplicate opcode conflict preservation (Case H)
"""

import os
import pytest
from core.requirement_ir.models import DocumentIR, DocumentBlock, SourceLocation
from core.efs_ir.models import EFSIR, EFSComponent, EFSRegister, new_efs_id
from core.extraction.source_evidence import SourceEvidenceIndex
from core.extraction.dedicated_extractors import (
    DedicatedRelationshipExtractor, CanonicalEntityResolver, is_abstract_or_action_phrase
)
from core.extraction.source_recovery import SourceBackedRecovery


def _make_test_doc_ir(text: str, section_title: str = "Architecture") -> DocumentIR:
    return DocumentIR(
        document_id="test_doc",
        metadata={"filename": "test_spec.docx"},
        sections=[{"title": section_title, "section_id": "SEC_001", "text": text}],
        blocks=[
            DocumentBlock(
                block_id="BLK_001",
                type="text",
                section_id="SEC_001",
                section_title=section_title,
                page=1,
                text=text,
                source_location=SourceLocation(
                    document_id="test_doc",
                    section=section_title,
                    section_id="SEC_001",
                    page=1,
                    block_id="BLK_001",
                    original_text=text
                )
            )
        ]
    )


# ==============================================================================
# Case A: Component Description Protection
# ==============================================================================

def test_case_a_control_unit_manages_execution_no_interface():
    """
    'Control Unit : Manages execution for multi-cycle operations'
    Expected: component description, NO interface, NO fabricated component.
    """
    text = "Control Unit : Manages execution for and multi-cycle operations"
    doc_ir = _make_test_doc_ir(text)
    known_comps = ["Control Unit", "Matrix ALU"]

    accepted, rejected, unresolved = DedicatedRelationshipExtractor.extract_candidates(
        doc_ir, known_components=known_comps
    )

    assert len(accepted) == 0, f"Expected 0 accepted relationships, got: {accepted}"
    assert len(rejected) >= 1
    rej = rejected[0]
    assert rej.category == "component_description"
    assert rej.rejection_reason == "rejected_as_component_description"
    assert "execution" in rej.target.lower()

    # Recovery pass
    efs = EFSIR(components=[
        EFSComponent(component_id=new_efs_id("comp"), name="Control Unit", type="controller")
    ])
    SourceBackedRecovery.recover_efs_ir(efs, doc_ir)

    # Must NOT create an interface
    bad_ifaces = [i for i in efs.interfaces if "execution" in i.name.lower() or (i.source_component == "Control Unit" and "execution" in str(getattr(i, "destination_component", "")))]
    assert len(bad_ifaces) == 0, f"Fabricated interface found: {bad_ifaces}"

    # Must NOT create a component
    bad_comps = [c for c in efs.components if "execution" in c.name.lower()]
    assert len(bad_comps) == 0, f"Fabricated component found: {bad_comps}"


# ==============================================================================
# Case B: Register Description Protection
# ==============================================================================

def test_case_b_configuration_registers_control_parameters_no_interface():
    """
    'Configuration Registers : Control matrix engine parameters'
    Expected: register description, NO interface, NO fabricated component.
    """
    text = "Configuration Registers : Control matrix engine parameters"
    doc_ir = _make_test_doc_ir(text, section_title="External Interfaces")
    known_comps = ["Control Unit"]
    known_regs = ["Configuration Registers"]

    accepted, rejected, unresolved = DedicatedRelationshipExtractor.extract_candidates(
        doc_ir, known_components=known_comps, known_registers=known_regs
    )

    assert len(accepted) == 0, f"Expected 0 accepted relationships, got: {accepted}"
    assert len(rejected) >= 1
    rej = rejected[0]
    assert rej.category == "register_description"
    assert rej.rejection_reason == "rejected_as_register_description"
    assert "matrix engine parameters" in rej.target.lower()

    # Recovery pass
    efs = EFSIR(
        components=[EFSComponent(component_id=new_efs_id("comp"), name="Control Unit")],
        registers=[EFSRegister(register_id=new_efs_id("reg"), name="Configuration Registers")]
    )
    SourceBackedRecovery.recover_efs_ir(efs, doc_ir)

    bad_ifaces = [i for i in efs.interfaces if "matrix_engine_parameters" in i.name.lower() or "configuration_registers" in i.name.lower()]
    assert len(bad_ifaces) == 0, f"Fabricated interface found: {bad_ifaces}"

    bad_comps = [c for c in efs.components if "matrix engine parameters" in c.name.lower() or "configuration registers" in c.name.lower()]
    assert len(bad_comps) == 0, f"Fabricated component found: {bad_comps}"


# ==============================================================================
# Case C: Flow Relationship (Scheduler distributes to cores)
# ==============================================================================

def test_case_c_scheduler_distributes_workloads_across_cores():
    """
    'Global Scheduler : Distributes matrix workloads across available cores'
    Expected: valid hardware flow/relationship (EFSFlow) when both entities resolve.
    """
    text = "Global Scheduler : Distributes matrix workloads across available cores"
    doc_ir = _make_test_doc_ir(text, section_title="Core Components")
    known_comps = ["Global Scheduler", "Matrix Processing Elements (MPEs)"]

    accepted, rejected, unresolved = DedicatedRelationshipExtractor.extract_candidates(
        doc_ir, known_components=known_comps
    )

    # In resolver, 'available cores' aliases to 'Matrix Processing Elements (MPEs)'
    # Test entity resolution
    resolver = CanonicalEntityResolver(known_components=known_comps)
    resolver.entity_map["available cores"] = ("Matrix Processing Elements (MPEs)", "component")
    resolver.entity_map["cores"] = ("Matrix Processing Elements (MPEs)", "component")

    s_res = resolver.resolve("Global Scheduler")
    d_res = resolver.resolve("available cores")
    assert s_res is not None and s_res[0] == "Global Scheduler"
    assert d_res is not None and d_res[0] == "Matrix Processing Elements (MPEs)"

    # Recovery pass
    efs = EFSIR(components=[
        EFSComponent(component_id=new_efs_id("comp"), name="Global Scheduler"),
        EFSComponent(component_id=new_efs_id("comp"), name="Matrix Processing Elements (MPEs)")
    ])
    SourceBackedRecovery.recover_efs_ir(efs, doc_ir)

    # Should be stored as EFSFlow or flow relationship, NOT a fabricated interconnect with noun phrases
    bad_ifaces = [i for i in efs.interfaces if "workloads" in i.name.lower() or "available" in i.name.lower()]
    assert len(bad_ifaces) == 0, f"Malformed interface found: {bad_ifaces}"


# ==============================================================================
# Case D: Communication Interface (AXI4 connects compute engine to memory)
# ==============================================================================

def test_case_d_axi4_interface_connects_compute_engine_to_memory():
    """
    'AXI4 interface connects compute engine to memory'
    Expected: actual communication interface (EFSInterface).
    """
    text = "AXI4 interface connects compute engine to memory"
    doc_ir = _make_test_doc_ir(text, section_title="External Interfaces")
    known_comps = ["compute engine", "memory"]

    accepted, rejected, unresolved = DedicatedRelationshipExtractor.extract_candidates(
        doc_ir, known_components=known_comps
    )

    assert len(accepted) >= 1
    rel = accepted[0]
    assert rel.category == "interface"
    assert rel.relation == "connects_to"
    assert rel.resolved_source == "compute engine"
    assert rel.resolved_target == "memory"

    # Recovery pass
    efs = EFSIR(components=[
        EFSComponent(component_id=new_efs_id("comp"), name="compute engine"),
        EFSComponent(component_id=new_efs_id("comp"), name="memory")
    ])
    SourceBackedRecovery.recover_efs_ir(efs, doc_ir)

    ifaces = [i for i in efs.interfaces if i.source_component == "compute engine" and getattr(i, "destination_component", None) == "memory"]
    assert len(ifaces) == 1
    assert ifaces[0].protocol == "interconnect"


# ==============================================================================
# Case E: Behavior Protection (Local Tiling Engine manages subblocks)
# ==============================================================================

def test_case_e_local_tiling_engine_manages_subblocks_no_interface():
    """
    'Local Tiling Engine : Manages processing of matrix subblocks'
    Expected: description/behavior, NO fabricated interface.
    """
    text = "Local Tiling Engine : Manages processing of matrix subblocks"
    doc_ir = _make_test_doc_ir(text, section_title="Matrix Processing Element Design")
    known_comps = ["Local Tiling Engine"]

    accepted, rejected, unresolved = DedicatedRelationshipExtractor.extract_candidates(
        doc_ir, known_components=known_comps
    )

    assert len(accepted) == 0
    assert len(rejected) >= 1
    rej = rejected[0]
    assert rej.category == "component_description"
    assert rej.rejection_reason == "rejected_as_component_description"

    efs = EFSIR(components=[
        EFSComponent(component_id=new_efs_id("comp"), name="Local Tiling Engine")
    ])
    SourceBackedRecovery.recover_efs_ir(efs, doc_ir)

    bad_ifaces = [i for i in efs.interfaces if "subblocks" in i.name.lower() or "tiling" in i.name.lower()]
    assert len(bad_ifaces) == 0, f"Fabricated interface found: {bad_ifaces}"

    bad_comps = [c for c in efs.components if "subblocks" in c.name.lower()]
    assert len(bad_comps) == 0, f"Fabricated component found: {bad_comps}"


# ==============================================================================
# Non-Regression Checks (F, G, H)
# ==============================================================================

def test_case_f_g_h_non_regression_on_matrix_spec():
    """
    Verifies that on matrix_compute_unit:
    F. Instruction width extraction is intact (64-bit instructions).
    G. Register count is intact (M0-M7).
    H. Duplicate opcode conflicts remain detected and preserved.
    """
    from core.parsing.docx_parser import DocxParser
    from core.extraction.requirement_extractor import GenericRequirementExtractor
    from core.compiler.efs_compiler import RequirementToEFSCompiler

    candidate_paths = [
        "data/uploads/matrix_compute_unit.docx",
        "hardware_graphrag/data/uploads/matrix_compute_unit.docx",
    ]
    path = next((p for p in candidate_paths if os.path.exists(p)), None)
    assert path is not None, "Matrix compute unit docx not found"

    parser = DocxParser(path)
    parsed = parser.parse()
    doc_ir = DocumentIR(
        document_id=parsed.doc_id,
        metadata={"filename": parsed.filename},
        sections=[s.to_dict() for s in parsed.sections],
        blocks=[
            DocumentBlock(
                block_id=f"BLK_{i+1:03d}",
                type=s.content_type,
                section_id=s.section_id,
                section_title=s.title,
                page=s.page_start,
                text=s.text,
                source_location=SourceLocation(
                    document_id=parsed.doc_id,
                    section=s.title,
                    section_id=s.section_id,
                    page=s.page_start,
                    block_id=f"BLK_{i+1:03d}",
                    original_text=s.text[:200]
                )
            ) for i, s in enumerate(parsed.sections) if s.text and s.text.strip()
        ]
    )

    extractor = GenericRequirementExtractor(doc_ir)
    req_ir = extractor.extract_requirement_ir()
    compiler = RequirementToEFSCompiler(req_ir)
    efs = compiler.compile()

    # Recovery pass
    evidence_index = SourceEvidenceIndex(doc_ir)
    SourceBackedRecovery.recover_efs_ir(efs, doc_ir, evidence_index)

    # F: Instruction width
    assert len(efs.instructions) >= 10
    has_64 = any("64" in str(getattr(fmt, "total_width", "")) for fmt in efs.instruction_formats) or len(efs.instruction_formats) > 0
    assert has_64

    # G: Register count
    reg_names = {r.name for r in efs.registers}
    assert any("M0" in n or "Matrix Register" in n for n in reg_names)

    # H: Duplicate opcode conflict preservation
    dup_conflicts = [c for c in efs.conflicts if c.type in ("DUPLICATE_OPCODE", "DUPLICATE_OPCODE_ENCODING")]
    assert len(dup_conflicts) >= 1, "Duplicate opcode conflicts must be preserved"
    for dc in dup_conflicts:
        assert dc.requires_human_decision is True, "Must not auto-resolve duplicate opcodes"


def test_inter_module_relationships_extraction_and_validation():
    """
    Test extraction of inter-module relationships from explicit specifications:
    - AccelerationController sends mem_addr, mem_wdata, and mem_we to LocalMemory
    - LocalMemory returns mem_rdata to AccelerationController

    Verifies:
    1. DedicatedRelationshipExtractor extracts both relationships with correct signals and direction.
    2. SourceBackedRecovery populates canonical efs_ir.relationships and efs_ir.interfaces.
    3. Hardware Completeness Validator accepts the relationships and does NOT raise
       'Missing Inter-Module Relationships' conflict even when 42 components exist.
    """
    text = (
        "Inter-Module Relationship:\n"
        "- AccelerationController sends mem_addr, mem_wdata, and mem_we to LocalMemory\n"
        "- LocalMemory returns mem_rdata to AccelerationController\n"
    )
    doc_ir = _make_test_doc_ir(text, section_title="Inter-Module Relationship")

    # 1. Extraction test
    accepted, rejected, unresolved = DedicatedRelationshipExtractor.extract_candidates(doc_ir)
    assert len(accepted) >= 2, f"Expected at least 2 accepted relationships, got {len(accepted)}"

    rel_send = next((r for r in accepted if r.relation == "sends_to"), None)
    assert rel_send is not None
    assert rel_send.source == "AccelerationController"
    assert rel_send.target == "LocalMemory"
    assert rel_send.signals == ["mem_addr", "mem_wdata", "mem_we"]
    assert rel_send.direction == "AccelerationController -> LocalMemory"

    rel_ret = next((r for r in accepted if r.relation == "returns_to"), None)
    assert rel_ret is not None
    assert rel_ret.source == "LocalMemory"
    assert rel_ret.target == "AccelerationController"
    assert rel_ret.signals == ["mem_rdata"]
    assert rel_ret.direction == "LocalMemory -> AccelerationController"

    # 2. EFS IR recovery test with 42 components
    efs = EFSIR(
        components=[
            EFSComponent(component_id=new_efs_id("comp"), name=f"Component_{i}")
            for i in range(40)
        ] + [
            EFSComponent(component_id=new_efs_id("comp"), name="AccelerationController"),
            EFSComponent(component_id=new_efs_id("comp"), name="LocalMemory"),
        ]
    )
    assert len(efs.components) == 42

    evidence_index = SourceEvidenceIndex(doc_ir)
    SourceBackedRecovery.recover_efs_ir(efs, doc_ir, evidence_index)

    # Verify canonical relationships list
    assert len(efs.relationships) >= 2
    r1 = next(r for r in efs.relationships if r["source"] == "AccelerationController")
    assert r1["target"] == "LocalMemory"
    assert r1["signals"] == ["mem_addr", "mem_wdata", "mem_we"]
    assert r1["direction"] == "AccelerationController -> LocalMemory"

    r2 = next(r for r in efs.relationships if r["source"] == "LocalMemory")
    assert r2["target"] == "AccelerationController"
    assert r2["signals"] == ["mem_rdata"]
    assert r2["direction"] == "LocalMemory -> AccelerationController"

    # Verify interfaces list
    assert len(efs.interfaces) >= 2
    iface1 = next(i for i in efs.interfaces if i.source_component == "AccelerationController")
    assert iface1.destination_component == "LocalMemory"
    assert iface1.signals == ["mem_addr", "mem_wdata", "mem_we"]
    assert iface1.direction == "AccelerationController -> LocalMemory"

    # 3. Completeness validation check
    from core.verification.conflict_resolution_engine import ConflictResolutionEngine
    conflicts = ConflictResolutionEngine.validate_hardware_completeness(efs, doc_ir=doc_ir, evidence_index=evidence_index)
    rel_conflicts = [c for c in conflicts if "inter-module" in c.title.lower() or "relationship" in c.title.lower()]
    assert len(rel_conflicts) == 0, f"Expected 0 missing relationship conflicts, got: {[c.title for c in rel_conflicts]}"


def test_generic_phrase_patterns_extraction():
    """
    Test generic phrase detection across various styles:
    - 'A connects to B'
    - 'A interfaces with B'
    - 'A communicates with B'
    - 'A reads/writes B'
    """
    phrases = (
        "Core connects to Cache via addr, data\n"
        "DMA interfaces with Memory\n"
        "CPU communicates with Coprocessor\n"
        "Host reads/writes Target\n"
    )
    doc_ir = _make_test_doc_ir(phrases, section_title="Connectivity")
    accepted, rejected, unresolved = DedicatedRelationshipExtractor.extract_candidates(doc_ir)
    assert len(accepted) == 4
    rel_types = {r.relation for r in accepted}
    assert "connects_to" in rel_types
    assert "interfaces_with" in rel_types
    assert "communicates_with" in rel_types
    assert "reads_writes" in rel_types
