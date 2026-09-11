"""
Multi-Specification Compiler Pipeline Test Suite (Section 32 & 33).

Tests the generic, protocol-independent pipeline against three structurally
different specifications without any hardcoded protocol rules or terms:
  1. Specification A: Protocol-heavy custom handshake protocol
  2. Specification B: Register/Control-heavy peripheral controller
  3. Specification C: FSM/Transaction-heavy algorithmic engine
"""

from __future__ import annotations

import os
import pytest
from core.requirement_ir.models import (
    AtomicRequirement, DiscoveredEntity, DocumentBlock, DocumentIR,
    RequirementIR, SourceLocation
)
from core.extraction.requirement_extractor import GenericRequirementExtractor
from core.requirement_ir.reference_resolver import CrossReferenceResolver
from core.requirement_ir.graph import RequirementDependencyGraph
from core.requirement_ir.chunker import build_semantic_chunks
from core.requirement_ir.validator import RequirementValidator
from core.compiler.efs_compiler import RequirementToEFSCompiler
from core.retrieval.context_compiler import compile_task_context
from core.retrieval.query_analyzer import analyze_query


# --------------------------------------------------------------------------
# Sample Specifications (Structurally Different & Protocol-Independent)
# --------------------------------------------------------------------------

SPEC_A_PROTOCOL_HEAVY = """
# Custom Stream Handshake Specification v1.0

Section 1. System Overview
The StreamAccelerator block processes streaming packet transfers across a point-to-point interface.

Section 2. Signal Definitions
- str_data: input, 64 bits, Streaming data payload.
- str_valid: input, 1 bit, Asserted by sender when data payload is valid.
- str_ready: output, 1 bit, Asserted by receiver when ready to accept payload.
- str_last: input, 1 bit, Asserted on final packet beat.
- str_clock: input, 1 bit, System clock reference.
- str_reset_n: input, 1 bit, Active low system reset.

Section 3. Protocol Handshake Rules
Rule 3.1: An initiator shall assert str_valid when data is placed on str_data.
Rule 3.2: The signal str_valid must remain asserted until str_ready is asserted by the target.
Rule 3.3: A data transfer occurs on any clock cycle where both str_valid and str_ready are asserted.
Rule 3.4: See Section 2 for signal pin directions.
"""

SPEC_B_REGISTER_HEAVY = """
# Custom Crypto Core Register Map Spec v2.1

Section 1. Architecture
The CryptoCore block contains control, status, and key configuration registers.

Section 2. Register Definitions
- CTRL_REG: offset 0x00, access RW, description "Main control register".
  - Bit 0: START (RW, reset 0, Start crypto operation)
  - Bit 1: ENABLE (RW, reset 0, Module enable bit)
  - Bit 2: KEY_LEN (RW, reset 0, Select key length 128 or 256)
- STATUS_REG: offset 0x04, access RO, description "Status register".
  - Bit 0: BUSY (RO, reset 0, Operation in progress)
  - Bit 1: DONE (RO, reset 0, Operation completed)
  - Bit 2: ERROR (RO, reset 0, Key or parity error)

Section 3. Operational Requirements
Rule 3.1: The host shall write START bit to 1 after configuring KEY_LEN.
Rule 3.2: STATUS_REG BUSY bit shall remain 1 while encryption is executing.
Rule 3.3: See Section 2 for register bitfield layouts.
"""

SPEC_C_FSM_HEAVY = """
# Matrix Processing Engine State Machine Spec v3.0

Section 1. Subsystem Description
The MatrixEngine computes matrix multiplication using a 4-state Finite State Machine.

Section 2. FSM States & Behaviors
- IDLE: Waiting for trigger signal compute_start.
- FETCH: Loading matrix descriptor from memory. Output fetch_active asserted.
- COMPUTE: Executing dot-product calculations. Output compute_active asserted.
- DONE: Writing completion status and asserting compute_done signal.

Section 3. Transitions
Rule 3.1: MatrixEngine shall transition from IDLE to FETCH when compute_start is 1.
Rule 3.2: MatrixEngine shall transition from FETCH to COMPUTE when descriptor_loaded is 1.
Rule 3.3: MatrixEngine shall transition from COMPUTE to DONE when matrix_complete is 1.
Rule 3.4: MatrixEngine shall transition from DONE to IDLE when ack_received is 1.
Rule 3.5: See Section 2 for state descriptions.
"""

SPEC_CONFLICING_WIDTHS = """
# Faulty Peripheral Specification

Section 1. Interface
- data_bus: input, 16 bits, Data bus interface.
Section 2. Processing Unit
Rule 2.1: The data_bus is 32 bits wide for high throughput transfers.
"""


def _create_doc_ir(spec_text: str, doc_id: str) -> DocumentIR:
    lines = spec_text.strip().split("\n")
    sections = []
    blocks = []

    curr_sec_title = "General"
    curr_text_lines = []
    b_idx = 1

    for line in lines:
        if line.startswith("# ") or line.startswith("Section "):
            if curr_text_lines:
                text = "\n".join(curr_text_lines)
                b_id = f"BLK_{b_idx:03d}"
                blocks.append(DocumentBlock(
                    block_id=b_id,
                    type="paragraph",
                    section_title=curr_sec_title,
                    text=text,
                    source_location=SourceLocation(document_id=doc_id, section=curr_sec_title, block_id=b_id)
                ))
                b_idx += 1
                curr_text_lines = []
            curr_sec_title = line.strip("# ").strip()
            sections.append({"section_id": f"SEC_{len(sections)+1}", "title": curr_sec_title})
        else:
            curr_text_lines.append(line)

    if curr_text_lines:
        b_id = f"BLK_{b_idx:03d}"
        blocks.append(DocumentBlock(
            block_id=b_id,
            type="paragraph",
            section_title=curr_sec_title,
            text="\n".join(curr_text_lines),
            source_location=SourceLocation(document_id=doc_id, section=curr_sec_title, block_id=b_id)
        ))

    return DocumentIR(
        document_id=doc_id,
        source_type="md",
        sections=sections,
        blocks=blocks
    )


# --------------------------------------------------------------------------
# Test Cases
# --------------------------------------------------------------------------

def test_specification_a_protocol_heavy():
    """Test Specification A: Protocol-heavy stream handshake extraction and compilation."""
    doc_ir = _create_doc_ir(SPEC_A_PROTOCOL_HEAVY, "SPEC_A")
    extractor = GenericRequirementExtractor(doc_ir)
    req_ir = extractor.extract_requirement_ir()

    assert len(req_ir.requirements) > 0, "Failed to extract requirements from Spec A"
    assert len(req_ir.entities) > 0, "Failed to discover entities from Spec A"

    # Verify dynamic entity discovery without protocol hardcoding
    ent_names = [e.name.lower() for e in req_ir.entities]
    assert any("str_valid" in n or "valid" in n for n in ent_names), "Failed to discover str_valid signal"
    assert any("str_ready" in n or "ready" in n for n in ent_names), "Failed to discover str_ready signal"

    # Cross reference resolution
    resolver = CrossReferenceResolver(doc_ir)
    ref_edges, _ = resolver.resolve_references(req_ir.requirements)
    req_ir.edges.extend(ref_edges)

    # Graph and Chunks
    graph = RequirementDependencyGraph(req_ir)
    chunks = build_semantic_chunks(req_ir.requirements, req_ir.entities)
    assert len(chunks) > 0, "Failed to create semantic chunks"

    # EFS Compiler
    compiler = RequirementToEFSCompiler(req_ir)
    efs = compiler.compile()
    assert efs is not None
    assert len(efs.signals) > 0, "Compiled EFS IR should contain discovered signals"

    # Check Traceability
    for sig in efs.signals:
        assert sig.traceability.doc_id == "SPEC_A", "Signal traceability doc_id mismatch"


def test_specification_b_register_heavy():
    """Test Specification B: Register/Control-heavy peripheral extraction."""
    doc_ir = _create_doc_ir(SPEC_B_REGISTER_HEAVY, "SPEC_B")
    extractor = GenericRequirementExtractor(doc_ir)
    req_ir = extractor.extract_requirement_ir()

    assert len(req_ir.requirements) > 0
    ent_names = [e.name.lower() for e in req_ir.entities]
    assert any("ctrl_reg" in n or "status_reg" in n or "busy" in n for n in ent_names)

    # Compiler
    compiler = RequirementToEFSCompiler(req_ir)
    efs = compiler.compile()
    assert len(efs.registers) > 0 or len(efs.signals) > 0


def test_specification_c_fsm_heavy():
    """Test Specification C: FSM/Transaction-heavy algorithmic engine extraction."""
    doc_ir = _create_doc_ir(SPEC_C_FSM_HEAVY, "SPEC_C")
    extractor = GenericRequirementExtractor(doc_ir)
    req_ir = extractor.extract_requirement_ir()

    assert len(req_ir.requirements) > 0

    # Verify state transitions extracted
    compiler = RequirementToEFSCompiler(req_ir)
    efs = compiler.compile()
    assert len(efs.fsms) > 0 or len(efs.flows) > 0, "Spec C should compile an FSM or behavioral Flow"


def test_validation_and_conflict_detection():
    """Test Pre-Generation Validation detecting width mismatch conflicts."""
    doc_ir = _create_doc_ir(SPEC_CONFLICING_WIDTHS, "SPEC_CONFLICT")
    extractor = GenericRequirementExtractor(doc_ir)
    req_ir = extractor.extract_requirement_ir()

    validator = RequirementValidator(req_ir)
    conflicts, status = validator.validate()

    assert len(conflicts) > 0, "Validator failed to detect width mismatch conflict"
    assert status["has_critical_conflicts"] is True, "Critical conflict status should be True"


def test_selective_retrieval_and_token_reduction():
    """Test Phase B selective requirement retrieval context reduction."""
    doc_ir = _create_doc_ir(SPEC_A_PROTOCOL_HEAVY, "SPEC_A")
    extractor = GenericRequirementExtractor(doc_ir)
    req_ir = extractor.extract_requirement_ir()

    compiler = RequirementToEFSCompiler(req_ir)
    efs = compiler.compile()

    intent = analyze_query("Generate synthesizable RTL for str_valid handshake behavior")
    pack, res_result = compile_task_context(intent, efs, max_tokens=3000)

    assert pack is not None
    # Token count of packed context should be bounded
    assert pack.estimated_tokens <= 3000
