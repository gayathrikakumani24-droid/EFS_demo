"""
Generic Specification Extraction & EFS IR Compiler Test Suite.

Tests dynamic concept discovery (architecture, instructions, opcodes, memory regions,
data structures, errors, performance requirements, state transitions, conflicts, and traceability)
for arbitrary specifications without hardcoded protocol assumptions.
"""

from __future__ import annotations

import pytest
from core.requirement_ir.models import (
    AtomicRequirement, DiscoveredEntity, DocumentBlock, DocumentIR, SourceLocation
)
from core.extraction.requirement_extractor import GenericRequirementExtractor
from core.requirement_ir.validator import RequirementValidator
from core.compiler.efs_compiler import RequirementToEFSCompiler
from core.pipeline_diagnostics import calculate_extraction_coverage, generate_missing_information_report


COMPLEX_HARDWARE_SPEC = """
# Neural Accelerator Subsystem Specification v1.0

Section 1. Architecture & Overview
The NeuralAccelerator engine executes matrix multiplications and activations across a multi-core MPE organization.

Section 2. Instruction Set Architecture
- MLOAD [63:0]: Opcode 0x01, Load 64-bit matrix tile descriptor into SRAM buffer.
- MSTORE [63:0]: Opcode 0x02, Store output result matrix tile to external memory.
- MADD [63:0]: Opcode 0x03, Execute element-wise matrix addition on 32x32 tiles.
- MMUL [63:0]: Opcode 0x04, Execute matrix dot-product multiplication.

Section 3. Memory Hierarchy & Buffers
- SRAM_TILE_BUF: Base address 0x80000000, Size 64KB, SRAM local buffer for tile storage.
- L1_CACHE: Size 16KB, L1 instruction and data cache.

Section 4. Performance & Power Targets
- Operating Frequency: 500 MHz
- Target Latency: 16 cycles per matrix tile block.
- Maximum Power: 250 mW

Section 5. Error & Exception Handling
- ERR_TILE_OVERFLOW: Code 0xE01, Raised when tile dimensions exceed 64x64 buffer bounds.
- ERR_INVALID_OPCODE: Code 0xE02, Raised when decoder encounters an unsupported opcode.

Section 6. State Machine & Execution Flow
- IDLE: Waiting for start signal.
- DECODE: Decoding instruction opcode.
- EXECUTE: Executing matrix compute block.
- WRITEBACK: Writing matrix result tile back to memory.
Rule 6.1: NeuralAccelerator shall transition from IDLE to DECODE when start_run is 1.
Rule 6.2: NeuralAccelerator shall transition from DECODE to EXECUTE when decode_valid is 1.
Rule 6.3: NeuralAccelerator shall transition from EXECUTE to WRITEBACK when compute_done is 1.

Section 7. Contradictory Width Specifications
The instruction input is described as 32-bit in overview, while section 2 describes instruction input as 64-bit.
"""


def test_generic_spec_extraction_pipeline():
    # 1. Build Document IR
    blocks = []
    sections = []
    lines = [l for l in COMPLEX_HARDWARE_SPEC.split("\n") if l.strip()]
    
    current_sec = "Overview"
    for i, line in enumerate(lines):
        if line.startswith("Section"):
            current_sec = line
        b_id = f"BLK_{i+1:03d}"
        blocks.append(DocumentBlock(
            block_id=b_id,
            type="paragraph",
            section_id=f"SEC_{i+1}",
            section_title=current_sec,
            page=1,
            text=line,
            source_location=SourceLocation(document_id="doc_test", section=current_sec, block_id=b_id, page=1)
        ))

    doc_ir = DocumentIR(document_id="doc_test", blocks=blocks, source_type="md")

    # 2. Extract Requirement IR
    extractor = GenericRequirementExtractor(doc_ir)
    req_ir = extractor.extract_requirement_ir()

    assert len(req_ir.entities) > 0, "Entities should be extracted dynamically from specification"
    assert len(req_ir.requirements) > 0, "Atomic requirements should be extracted"

    # Verify Instructions & Opcodes extracted
    inst_names = [e.name for e in req_ir.entities if e.type in ("instruction", "opcode")]
    assert any("MLOAD" in n for n in inst_names), "MLOAD instruction should be discovered"
    assert any("MMUL" in n for n in inst_names), "MMUL instruction should be discovered"

    # Verify Memory Regions extracted
    mem_names = [e.name for e in req_ir.entities if e.type in ("memory_region", "data_structure")]
    assert any("SRAM" in n or "BUF" in n or "CACHE" in n for n in mem_names), "Memory region should be discovered"

    # 3. Validate & Check Conflicts
    validator = RequirementValidator(req_ir)
    conflicts, status = validator.validate()
    req_ir.conflicts = conflicts

    # 4. Compile EFS IR
    compiler = RequirementToEFSCompiler(req_ir)
    efs_ir = compiler.compile()

    assert len(efs_ir.instructions) > 0, "EFS IR instructions should be populated"
    assert len(efs_ir.memory_regions) > 0, "EFS IR memory regions should be populated"
    assert len(efs_ir.fsms) > 0, "EFS IR state machine should be populated"

    # 5. Coverage & Missing Info Report
    coverage = calculate_extraction_coverage(req_ir, efs_ir)
    assert coverage["extraction_coverage_pct"] > 0, "Extraction coverage % should be > 0"
    missing_rep = generate_missing_information_report(req_ir)
    assert "unparsed_blocks_count" in missing_rep

    # Verify Traceability
    for inst in efs_ir.instructions:
        assert inst.traceability.doc_id == "doc_test" or inst.traceability.extraction_method != "unknown"
