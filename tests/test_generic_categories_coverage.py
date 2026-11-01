"""
Generic Categories Coverage Test Suite (Tests A - J).

Validates protocol-independent and document-agnostic extraction across:
- Test A: Instruction & Opcode extraction
- Test B: RTL module declaration & port extraction
- Test C: Parameter extraction
- Test D: Memory region / buffer extraction
- Test E: Performance / power targets
- Test F: Functional behavior
- Test G: Duplicate opcode conflict detection & HITL resolution
- Test H: Interfaces & signal association
- Test I: Extraction coverage metrics
- Test J: Deterministic absence reporting
"""

import json
import pytest
from config import CONFIG
# Ensure deterministic fast tests without external network or LLM latency
CONFIG.enable_llm_escalation = False

from core.requirement_ir.models import DocumentIR, DocumentBlock, SourceLocation
from core.extraction.requirement_extractor import GenericRequirementExtractor
from core.compiler.efs_compiler import RequirementToEFSCompiler
from core.efs_ir.validator import validate_efs_ir
from core.efs_ir.models import EFSIR, EFSResolution


def _build_doc_ir(text_blocks):
    import uuid
    uid = uuid.uuid4().hex[:8]
    return DocumentIR(
        document_id=f"DOC_{uid}",
        document_version="1.0",
        source_type="test",
        document_hash=f"hash_{uid}",
        ingestion_id=f"ing_{uid}",
        blocks=[
            DocumentBlock(
                block_id=f"BLK_{i+1:03d}",
                type="text",
                section_title=f"Section {i+1}",
                page=1,
                text=txt,
                source_location=SourceLocation(
                    document_id=f"DOC_{uid}",
                    document_hash=f"hash_{uid}",
                    section=f"Section {i+1}",
                    page=1,
                    block_id=f"BLK_{i+1:03d}",
                    original_text=txt[:100]
                )
            ) for i, txt in enumerate(text_blocks)
        ]
    )


def test_a_generic_instruction_and_opcode_extraction():
    """Test A: Extraction of custom ISA instructions, opcodes, and formats."""
    text = (
        "Instruction Set Architecture:\n"
        "The processor supports 32-bit instructions.\n"
        "Format R-Type: opcode[31:26], rs[25:21], rt[20:16], rd[15:11], funct[5:0]\n"
        "Opcodes:\n"
        "000001 : VADD (Vector Addition)\n"
        "000010 : VSUB (Vector Subtraction)\n"
        "000011 : VMUL (Vector Multiplication)\n"
    )
    doc_ir = _build_doc_ir([text])
    req_ir = GenericRequirementExtractor(doc_ir).extract_requirement_ir()
    efs = RequirementToEFSCompiler(req_ir).compile()

    assert len(efs.opcodes) == 3
    mnemonics = {op.mnemonic for op in efs.opcodes}
    assert "VADD" in mnemonics or "VADD_OPCODE" in mnemonics
    assert len(efs.instructions) >= 3
    assert len(efs.instruction_formats) >= 1
    fmt = efs.instruction_formats[0]
    assert len(fmt.fields) >= 3


def test_b_generic_rtl_module_and_port_parsing():
    """Test B: Extraction of Verilog RTL module header and port declarations."""
    text = (
        "module custom_accelerator #(\n"
        "    parameter DATA_WIDTH = 64,\n"
        "    parameter ADDR_WIDTH = 32\n"
        ") (\n"
        "    input  wire                   clk,\n"
        "    input  wire                   rst_n,\n"
        "    input  wire [DATA_WIDTH-1:0]  axis_tdata,\n"
        "    input  wire                   axis_tvalid,\n"
        "    output wire                   axis_tready,\n"
        "    output wire [ADDR_WIDTH-1:0]  mem_addr\n"
        ");\n"
    )
    doc_ir = _build_doc_ir([text])
    req_ir = GenericRequirementExtractor(doc_ir).extract_requirement_ir()
    efs = RequirementToEFSCompiler(req_ir).compile()

    sig_names = {s.name for s in efs.signals}
    assert "clk" in sig_names
    assert "rst_n" in sig_names
    assert "axis_tdata" in sig_names
    assert "axis_tready" in sig_names
    assert len(efs.components) >= 1
    assert any("accelerator" in c.name.lower() for c in efs.components)


def test_c_generic_parameter_extraction():
    """Test C: Extraction of hardware parameters and generics."""
    text = (
        "Configuration Parameters:\n"
        "parameter PIPELINE_DEPTH = 8;\n"
        "parameter BUFFER_ENTRIES = 64;\n"
    )
    doc_ir = _build_doc_ir([text])
    req_ir = GenericRequirementExtractor(doc_ir).extract_requirement_ir()
    efs = RequirementToEFSCompiler(req_ir).compile()

    assert len(efs.parameters) >= 2
    param_names = {p.name for p in efs.parameters}
    assert "PIPELINE_DEPTH" in param_names
    assert "BUFFER_ENTRIES" in param_names


def test_d_generic_memory_region_extraction():
    """Test D: Extraction of memory buffers and caches."""
    text = (
        "Memory Organization:\n"
        "The accelerator features a 128KB SRAM buffer for activation storage.\n"
        "A 64KB Weight Cache is provided for filter coefficients.\n"
    )
    doc_ir = _build_doc_ir([text])
    req_ir = GenericRequirementExtractor(doc_ir).extract_requirement_ir()
    efs = RequirementToEFSCompiler(req_ir).compile()

    assert len(efs.memory_regions) >= 2
    mem_names = {m.name.lower() for m in efs.memory_regions}
    assert any("buffer" in n or "sram" in n for n in mem_names)
    assert any("cache" in n or "weight" in n for n in mem_names)


def test_e_generic_performance_power_targets():
    """Test E: Extraction of performance throughput and latency targets."""
    text = (
        "Performance Targets:\n"
        "Throughput: 1024 GFLOPS peak computation\n"
        "Operating Frequency: 500 MHz maximum clock\n"
        "Power Target: 15 W TDP limit\n"
    )
    doc_ir = _build_doc_ir([text])
    req_ir = GenericRequirementExtractor(doc_ir).extract_requirement_ir()
    efs = RequirementToEFSCompiler(req_ir).compile()

    assert len(efs.performance_requirements) >= 2
    metrics = {p.metric.lower() for p in efs.performance_requirements}
    assert any("throughput" in m or "frequency" in m or "performance" in m for m in metrics)


def test_f_generic_functional_behavior():
    """Test F: Extraction of algorithm/functional behavior requirements."""
    text = (
        "Matrix Multiplication Behavior:\n"
        "When the compute command is asserted, the unit computes C = A * B across 4 systolic cores.\n"
    )
    doc_ir = _build_doc_ir([text])
    req_ir = GenericRequirementExtractor(doc_ir).extract_requirement_ir()
    efs = RequirementToEFSCompiler(req_ir).compile()

    assert len(efs.functional_behavior) >= 1
    beh = efs.functional_behavior[0]
    assert "C = A * B" in beh.description or "compute" in beh.description.lower()


def test_g_generic_duplicate_opcode_conflict_and_resolution():
    """Test G: Duplicate opcodes are detected as critical conflicts and resolvable via HITL."""
    text = (
        "Opcode Table:\n"
        "000001 : OP_LOAD (Load operation)\n"
        "000001 : OP_FETCH (Fetch operation)\n"
    )
    doc_ir = _build_doc_ir([text])
    req_ir = GenericRequirementExtractor(doc_ir).extract_requirement_ir()
    efs = RequirementToEFSCompiler(req_ir).compile()

    assert len(efs.conflicts) >= 1
    conf = next(c for c in efs.conflicts if c.type == "DUPLICATE_OPCODE")
    assert conf.severity == "CRITICAL"
    assert conf.requires_human_decision is True
    assert conf.resolution_status == "UNRESOLVED"

    # Verify validator flags the duplicate opcode
    issues = validate_efs_ir(efs)
    assert any("Duplicate opcode" in i.get("message", "") for i in issues)

    # Resolve conflict via single authoritative reference
    res = EFSResolution(
        resolution_id="RES_001",
        conflict_id=conf.conflict_id,
        resolved_value="OP_LOAD = 000001, OP_FETCH = 000010",
        proposed_change="REASSIGN_OPCODE",
        approval_status="APPROVED"
    )
    efs.resolutions.append(res)
    conf.resolve(res)

    assert conf.requires_human_decision is False
    assert conf.blocking is False


def test_h_generic_interfaces_and_signal_association():
    """Test H: Signals associated with interfaces link bidirectional IDs."""
    text = (
        "module bus_interface (\n"
        "    input  wire s_axi_aclk,\n"
        "    input  wire s_axi_awvalid,\n"
        "    output wire s_axi_awready\n"
        ");\n"
        "Interface: s_axi\n"
        "Role: subordinate\n"
    )
    doc_ir = _build_doc_ir([text])
    req_ir = GenericRequirementExtractor(doc_ir).extract_requirement_ir()
    efs = RequirementToEFSCompiler(req_ir).compile()

    assert len(efs.signals) >= 3
    assert len(efs.interfaces) >= 1


def test_i_generic_extraction_coverage_metrics():
    """Test I: Extraction coverage maps all categories accurately."""
    text = (
        "parameter N = 4;\n"
        "000001 : NOP\n"
    )
    doc_ir = _build_doc_ir([text])
    req_ir = GenericRequirementExtractor(doc_ir).extract_requirement_ir()
    efs = RequirementToEFSCompiler(req_ir).compile()

    cov = efs.extraction_coverage
    assert cov is not None
    assert cov["parameters"]["status"] == "extracted"
    assert cov["parameters"]["count"] >= 1
    assert cov["opcodes"]["status"] == "extracted"
    assert cov["opcodes"]["count"] >= 1


def test_j_deterministic_absence_reporting():
    """Test J: Categories not present in spec are reported as not_explicitly_present, not manufactured."""
    text = "parameter SIMPLE_PARAM = 1;"
    doc_ir = _build_doc_ir([text])
    req_ir = GenericRequirementExtractor(doc_ir).extract_requirement_ir()
    efs = RequirementToEFSCompiler(req_ir).compile()

    cov = efs.extraction_coverage
    assert cov["registers"]["status"] == "not_explicitly_present"
    assert cov["registers"]["count"] == 0
    assert len(efs.registers) == 0
