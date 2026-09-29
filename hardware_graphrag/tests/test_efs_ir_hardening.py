"""
Unit tests for EFS IR Hardening.

Covers:
- Section 28: Duplicate Register Address Preservation & Conflict Flagging
- Section 29: Clean Specification & Validation Status
- Section 30: Signal References (Clocks, Resets, Dangling ID Detection)
- Section 31: Traceability, Confidence, and Source Documents Tracking
- Section 32: Schema Validation & Resolution Structure
"""

import pytest
from core.efs_ir.models import (
    EFSIR,
    EFSMetadata,
    EFSSourceDocument,
    EFSSignal,
    EFSRegister,
    EFSRegisterField,
    EFSConflict,
    EFSResolution,
    EFSResolutionCandidate,
    EFSTraceability,
    EFSExtractionQuality,
    EFSValidationResult,
    EFSValidationSummary,
)
from core.efs_ir.validator import validate_efs_ir
from core.efs_ir.schema import validate_efs_ir_schema
from core.verification.conflict_resolution_engine import ConflictResolutionEngine


def test_section_28_duplicate_register_address():
    """
    Section 28:
    When two registers have the same offset (0x10):
    - Both registers must retain offset 0x10 (no silent auto-fixing).
    - DUPLICATE_REGISTER_ADDRESS conflict is recorded with status='UNRESOLVED'.
    - requires_human_decision is True.
    - resolution_candidates are provided.
    - Validation flags status='FAILED'.
    """
    meta = EFSMetadata(
        design_id="DES_TEST_DUP",
        design_name="DupTest",
        canonical_id="CANON_DUP",
        source_documents=[EFSSourceDocument(document_id="DOC_001", document_name="spec.md")]
    )
    sig_clk = EFSSignal(signal_id="SIG_CLK", name="clk", width="1", direction="input", semantic_role="clock")
    reg_x = EFSRegister(
        register_id="REG_X",
        name="REG_X",
        address="0x10",
        offset="0x10",
        width=32,
        access_type="RW",
        access="RW"
    )
    reg_y = EFSRegister(
        register_id="REG_Y",
        name="REG_Y",
        address="0x10",
        offset="0x10",
        width=32,
        access_type="RW",
        access="RW"
    )
    
    conflict = EFSConflict(
        conflict_id="CONF_DUP_001",
        type="DUPLICATE_REGISTER_ADDRESS",
        severity="CRITICAL",
        target_entities=["REG_X", "REG_Y"],
        properties=["offset", "address"],
        description="Duplicate address offset 0x10 assigned to REG_X and REG_Y",
        requires_human_decision=True,
        resolution_status="UNRESOLVED",
        resolution_candidates=[
            EFSResolutionCandidate(
                candidate_id="CAND_01",
                action="ASSIGN_UNIQUE_OFFSET",
                target_entity="REG_Y",
                property="offset",
                current_value="0x10",
                proposed_value="0x14",
                reason="Auto-align to next 32-bit boundary"
            )
        ]
    )
    
    ir = EFSIR(
        metadata=meta,
        signals=[sig_clk],
        clocks=[sig_clk],
        registers=[reg_x, reg_y],
        conflicts=[conflict]
    )
    
    # 1. Verify preservation of original offsets
    assert ir.registers[0].offset == "0x10"
    assert ir.registers[1].offset == "0x10"
    
    # 2. Verify conflict properties
    assert ir.conflicts[0].type == "DUPLICATE_REGISTER_ADDRESS"
    assert ir.conflicts[0].resolution_status == "UNRESOLVED"
    assert ir.conflicts[0].requires_human_decision is True
    assert len(ir.conflicts[0].resolution_candidates) == 1
    assert ir.conflicts[0].resolution_candidates[0].proposed_value == "0x14"
    
    # 3. Validate IR
    issues = validate_efs_ir(ir)
    assert ir.validation is not None
    assert ir.validation.status == "FAILED"
    assert any("Duplicate register address offset detected" in iss["message"] for iss in issues)


def test_section_29_clean_specification():
    """
    Section 29:
    Clean specification with distinct registers at 0x10 and 0x14:
    - Zero conflicts.
    - Validation status='PASSED'.
    - Extraction quality correctly reflects completeness.
    """
    meta = EFSMetadata(
        design_id="DES_TEST_CLEAN",
        design_name="CleanTest",
        canonical_id="CANON_CLEAN",
        source_documents=[EFSSourceDocument(document_id="DOC_001", document_name="clean_spec.md")]
    )
    sig_clk = EFSSignal(signal_id="SIG_CLK", name="clk", width="1", direction="input", semantic_role="clock")
    reg_x = EFSRegister(
        register_id="REG_X",
        name="REG_X",
        address="0x10",
        offset="0x10",
        width=32,
        access_type="RW",
        access="RW"
    )
    reg_y = EFSRegister(
        register_id="REG_Y",
        name="REG_Y",
        address="0x14",
        offset="0x14",
        width=32,
        access_type="RW",
        access="RW"
    )
    
    ir = EFSIR(
        metadata=meta,
        signals=[sig_clk],
        clocks=[sig_clk],
        registers=[reg_x, reg_y],
        extraction_quality=EFSExtractionQuality(
            source_completeness="partial",
            ir_completeness="partial",
            extraction_status="success"
        )
    )
    
    issues = validate_efs_ir(ir)
    assert len(ir.conflicts) == 0
    assert ir.validation.status == "PASSED"
    assert ir.validation.summary.critical == 0
    assert ir.extraction_quality.ir_completeness == "partial"


def test_section_30_signal_references():
    """
    Section 30:
    Clock and reset reference canonical signal_id:
    - Serialization emits clocks: [{'signal_id': 'SIG_CLK'}].
    - Dangling reference SIG_999 is detected as validation ERROR.
    """
    meta = EFSMetadata(design_id="DES_SIG", design_name="SigTest", canonical_id="CANON_SIG")
    sig_clk = EFSSignal(signal_id="SIG_CLK", name="clk", width="1", direction="input", semantic_role="clock")
    sig_rst = EFSSignal(signal_id="SIG_RST", name="rst_n", width="1", direction="input", semantic_role="reset")
    
    ir = EFSIR(
        metadata=meta,
        signals=[sig_clk, sig_rst],
        clocks=[sig_clk],
        resets=[sig_rst]
    )
    
    ir_dict = ir.to_dict()
    assert ir_dict["clocks"] == [{"signal_id": "SIG_CLK"}]
    assert ir_dict["resets"] == [{"signal_id": "SIG_RST"}]
    
    # Deserialization resolves correctly
    loaded = EFSIR.from_dict(ir_dict)
    assert len(loaded.clocks) == 1
    assert loaded.clocks[0].signal_id == "SIG_CLK"
    assert loaded.clocks[0].name == "clk"
    
    # Now test dangling reference
    dangling_clock = EFSSignal(signal_id="SIG_999", name="phantom_clk")
    ir_dangling = EFSIR(
        metadata=meta,
        signals=[sig_clk],
        clocks=[dangling_clock]
    )
    issues = validate_efs_ir(ir_dangling)
    assert any(i["category"] == "Clocks" and "SIG_999" in i["message"] for i in issues)
    assert ir_dangling.validation.status == "FAILED"


def test_section_31_traceability_and_confidence():
    """
    Section 31:
    - extraction_method == 'unknown' defaults confidence to None.
    - Source documents are properly structured.
    """
    trace_unknown = EFSTraceability(
        source_document_id="DOC_123",
        source_document="datasheet.pdf",
        source_text="baud rate generator",
        extraction_method="unknown"
    )
    assert trace_unknown.confidence is None
    
    trace_dict = trace_unknown.to_dict()
    assert trace_dict["confidence"] is None
    assert trace_dict["source_document"] == "datasheet.pdf"
    
    # Deserializing None confidence
    reloaded = EFSTraceability.from_dict(trace_dict)
    assert reloaded.confidence is None


def test_section_32_schema_validation():
    """
    Section 32:
    Validates EFS IR against formal JSON Schema.
    """
    meta = EFSMetadata(
        design_id="DES_SCHEMA",
        design_name="SchemaTest",
        canonical_id="CANON_SCHEMA",
        source_documents=[EFSSourceDocument(document_id="DOC_1", document_name="test.md")]
    )
    sig_clk = EFSSignal(signal_id="SIG_CLK", name="clk", width="1", direction="input", semantic_role="clock")
    ir = EFSIR(metadata=meta, signals=[sig_clk], clocks=[sig_clk])
    validate_efs_ir(ir)
    
    data = ir.to_dict()
    is_valid, errors = validate_efs_ir_schema(data)
    assert is_valid, f"Schema validation failed: {errors}"


def test_structured_resolution_preservation():
    """
    Verify EFSResolution preserves original_value and resolved_value.
    """
    res = EFSResolution(
        resolution_id="RES_001",
        conflict_id="CONF_001",
        target_entity="REG_Y",
        property="offset",
        original_value="0x10",
        resolved_value="0x14",
        resolution_type="PARAM_OVERRIDE",
        resolution_source="HUMAN"
    )
    assert res.original_value == "0x10"
    assert res.resolved_value == "0x14"
    assert res.proposed_value == "0x14"
    
    d = res.to_dict()
    assert d["original_value"] == "0x10"
    assert d["resolved_value"] == "0x14"
    
    reloaded = EFSResolution.from_dict(d)
    assert reloaded.original_value == "0x10"
    assert reloaded.resolved_value == "0x14"


def test_section_32_multiple_documents_traceability():
    """
    Section 32: Multiple documents traceability isolation:
    Entities extracted from different source documents reference their respective document_id.
    """
    doc1 = EFSSourceDocument(document_id="DOC_AXI_SPEC", document_name="axi_protocol.pdf")
    doc2 = EFSSourceDocument(document_id="DOC_CSR_SPEC", document_name="csr_memory_map.pdf")
    meta = EFSMetadata(
        design_id="DES_SOC",
        design_name="SoCTop",
        canonical_id="CANON_SOC",
        source_documents=[doc1, doc2]
    )
    sig_axi = EFSSignal(
        signal_id="SIG_AWADDR",
        name="s_axi_awaddr",
        width="32",
        direction="input",
        semantic_role="address",
        traceability=EFSTraceability(
            source_document_id=doc1.document_id,
            source_document=doc1.document_name,
            extraction_method="llm_extraction",
            confidence=0.95
        )
    )
    reg_status = EFSRegister(
        register_id="REG_STATUS",
        name="STATUS",
        address="0x04",
        offset="0x04",
        width=32,
        access_type="RO",
        access="RO",
        traceability=EFSTraceability(
            source_document_id=doc2.document_id,
            source_document=doc2.document_name,
            extraction_method="table_parser",
            confidence=0.99
        )
    )
    ir = EFSIR(metadata=meta, signals=[sig_axi], registers=[reg_status])
    validate_efs_ir(ir)
    
    assert ir.signals[0].traceability.source_document_id == "DOC_AXI_SPEC"
    assert ir.registers[0].traceability.source_document_id == "DOC_CSR_SPEC"
    
    ir_dict = ir.to_dict()
    assert ir_dict["signals"][0]["traceability"]["source_document_id"] == "DOC_AXI_SPEC"
    assert ir_dict["registers"][0]["traceability"]["source_document_id"] == "DOC_CSR_SPEC"


def test_section_33_legitimate_empty_collections_preserved():
    """
    Section 33:
    Legitimate empty collections must be preserved as empty lists without fabricated values.
    """
    meta = EFSMetadata(design_id="DES_FIFO", design_name="AsyncFIFO", canonical_id="CANON_FIFO")
    sig_clk = EFSSignal(signal_id="SIG_CLK", name="clk", width="1", direction="input", semantic_role="clock")
    ir = EFSIR(
        metadata=meta,
        signals=[sig_clk],
        clocks=[sig_clk],
        registers=[],
        instructions=[],
        fsms=[],
        constraints=[]
    )
    
    assert ir.registers == []
    assert ir.instructions == []
    assert ir.fsms == []
    
    d = ir.to_dict()
    assert d["registers"] == []
    assert d["instructions"] == []
    assert d["fsms"] == []
    assert d["constraints"] == []
    
    loaded = EFSIR.from_dict(d)
    assert loaded.registers == []
    assert loaded.instructions == []
    assert loaded.fsms == []


def test_section_34_run_and_design_identity_consistency():
    """
    Section 34:
    Preserve run and design identity (design_id, canonical_id, generation_run_id, document_id, ingestion_id)
    across metadata and validation result.
    """
    meta = EFSMetadata(
        design_id="DES_DMA_001",
        design_name="ScatterGatherDMA",
        canonical_id="CANON_DMA",
        generation_run_id="RUN_2026_09_21_001",
        document_id="DOC_DMA_SPEC",
        ingestion_id="ING_DMA_SPEC",
        source_documents=[
            EFSSourceDocument(document_id="DOC_DMA_SPEC", ingestion_id="ING_DMA_SPEC", document_name="dma.md")
        ]
    )
    sig_clk = EFSSignal(signal_id="SIG_CLK", name="clk", width="1", direction="input", semantic_role="clock")
    ir = EFSIR(metadata=meta, signals=[sig_clk], clocks=[sig_clk])
    
    # Validate IR
    validate_efs_ir(ir)
    
    assert ir.validation is not None
    assert ir.validation.design_id == "DES_DMA_001"
    assert ir.validation.canonical_id == "CANON_DMA"
    assert ir.validation.generation_run_id == "RUN_2026_09_21_001"
    assert ir.validation.document_id == "DOC_DMA_SPEC"
    assert ir.validation.ingestion_id == "ING_DMA_SPEC"
    
    # Check serialization
    d = ir.to_dict()
    val_d = d["validation"]
    assert val_d["design_id"] == "DES_DMA_001"
    assert val_d["canonical_id"] == "CANON_DMA"
    assert val_d["generation_run_id"] == "RUN_2026_09_21_001"
    assert val_d["document_id"] == "DOC_DMA_SPEC"
    assert val_d["ingestion_id"] == "ING_DMA_SPEC"


def test_section_35_builder_end_to_end_duplicate_registers_and_naming():
    """
    Section 35:
    Test EFSIRBuilder:
    - Normalizes display_name by filtering leading articles like 'The'.
    - Assigns deterministic canonical_id.
    - Preserves duplicate register offsets at 0x00 and 0x0.
    - Detects DUPLICATE_REGISTER_ADDRESS and generates resolution candidates.
    - Populates ir.clocks and ir.resets from signal objects.
    """
    from core.efs_ir.builder import EFSIRBuilder
    
    req_model = {
        "design_spec_title": "The High Speed SPI Controller",
        "clocks_resets": [
            {"name": "sys_clk", "type": "clock", "description": "Core system clock"},
            {"name": "sys_rst_n", "type": "reset", "description": "Active-low reset"}
        ],
        "registers": [
            {"name": "CTRL", "offset": "0x00", "width": 32, "access": "RW", "description": "Control register"},
            {"name": "CONFIG", "offset": "0x0", "width": 32, "access": "RW", "description": "Config register"}
        ]
    }
    
    builder = EFSIRBuilder(
        requirement_model=req_model,
        design_spec="Module The High Speed SPI Controller with registers CTRL and CONFIG at 0x00.",
        req_model=req_model
    )
    ir = builder.build()
    
    # Display name shouldn't start with 'The'
    assert not ir.metadata.display_name.lower().startswith("the ")
    assert "SPI Controller" in ir.metadata.display_name
    assert ir.metadata.canonical_id.startswith("EFS_")
    
    # Clocks and resets should be populated
    assert len(ir.clocks) >= 1
    assert ir.clocks[0].name == "sys_clk"
    assert len(ir.resets) >= 1
    assert ir.resets[0].name == "sys_rst_n"
    
    # Offsets must be preserved as 0x00 and 0x0 (no auto-fixing)
    assert ir.registers[0].offset == "0x00"
    assert ir.registers[1].offset == "0x0"
    
    # Duplicate address conflict must be flagged
    dup_confs = [c for c in ir.conflicts if c.type == "DUPLICATE_REGISTER_ADDRESS"]
    assert len(dup_confs) == 1
    conf = dup_confs[0]
    assert conf.blocking is True
    assert conf.requires_human_decision is True
    assert len(conf.resolution_candidates) == 2
    assert conf.generation_run_id == ir.metadata.generation_run_id


def test_section_36_json_roundtrip_and_schema_validation():
    """
    Section 36:
    Full JSON serialization round-trip and validation against JSON Schema.
    """
    import json
    from core.efs_ir.builder import EFSIRBuilder
    
    req_model = {
        "design_spec_title": "AXI Arbiter",
        "clocks_resets": [
            {"name": "clk", "type": "clock", "description": "Clock"}
        ],
        "interfaces": [
            {"name": "req_0", "direction": "input", "width": "1"},
            {"name": "gnt_0", "direction": "output", "width": "1"}
        ]
    }
    builder = EFSIRBuilder(
        requirement_model=req_model,
        design_spec="AXI Arbiter module",
        req_model=req_model
    )
    ir = builder.build()
    validate_efs_ir(ir)
    
    ir_dict = ir.to_dict()
    is_valid, errors = validate_efs_ir_schema(ir_dict)
    assert is_valid, f"Initial schema validation failed: {errors}"
    
    json_str = json.dumps(ir_dict, indent=2)
    assert '"confidence": null' in json_str or '"confidence":' in json_str
    
    parsed_dict = json.loads(json_str)
    reloaded_ir = EFSIR.from_dict(parsed_dict)
    
    reloaded_dict = reloaded_ir.to_dict()
    is_valid_reloaded, errors_reloaded = validate_efs_ir_schema(reloaded_dict)
    assert is_valid_reloaded, f"Reloaded schema validation failed: {errors_reloaded}"
    assert reloaded_ir.metadata.canonical_id == ir.metadata.canonical_id
    assert len(reloaded_ir.signals) == len(ir.signals)

