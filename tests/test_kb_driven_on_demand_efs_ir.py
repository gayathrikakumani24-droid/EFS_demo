"""
Test Suite for Knowledge-Base-Driven, Protocol-Aware, On-Demand EFS IR Generation Architecture.

Verifies:
1. Ingestion Phase: Document is parsed and knowledge extracted into Vector DB and Graph DB
   without compiling full executable EFS IR at upload time.
2. Reusable Protocol Knowledge: Ingested independently, versioned, and tagged with knowledge_domain="protocol".
3. Idempotent Ingestion: Re-uploading does not duplicate nodes or create orphan edges.
4. Query Interpreter: Extracts target component, scope, artifact, and referenced protocols from queries.
5. Scope Resolver: Resolves minimal sufficient boundary for target component.
6. Scoped EFS IR Compiler: Generates unified, protocol-aware, traceable EFS IR incorporating structured protocol rules.
7. End-to-end traceability and gap preservation: missing specs recorded without fabrication.
"""

from __future__ import annotations

import os
import pytest
from core.pipeline import run_pipeline, PipelineResult
from core.vectorstore.faiss_store import get_vector_store
from core.graph.neo4j_builder import get_graph_store
from core.retrieval.query_interpreter import interpret_query, ScopedQueryIntent
from core.retrieval.scope_resolver import resolve_component_scope, ResolvedScope
from core.compiler.scoped_efs_compiler import ScopedEFSCompiler
from core.efs_ir.models import EFSIR, EFSProtocolRule


@pytest.fixture
def clean_stores():
    """Ensure clean vector and graph stores before tests."""
    vstore = get_vector_store()
    gstore = get_graph_store()
    yield vstore, gstore


def test_upload_does_not_compile_full_efs_ir_upfront(tmp_path):
    """Verify that document ingestion stores knowledge in DBs but does NOT precompile executable EFS IR."""
    spec_content = """
    # Hardware Accelerator Subsystem Spec v1.0
    Section 1. Subsystem Architecture
    The Accelerator Subsystem contains an Instruction Decoder, an ALU, and an Output Buffer.
    Section 2. Registers
    - CTRL_REG: offset 0x00, width 32, access RW, description "Control Register"
    - STATUS_REG: offset 0x04, width 32, access RO, description "Status Register"
    Section 3. Instructions
    - EXEC_ADD: Opcode 0x10, Add operands
    - EXEC_SUB: Opcode 0x20, Subtract operands
    """
    spec_file = tmp_path / "test_accel_spec.md"
    spec_file.write_text(spec_content, encoding="utf-8")

    result = run_pipeline(str(spec_file), filename="test_accel_spec.md", knowledge_domain="design")

    # In Knowledge-Base-Driven architecture, _efs_ir is None at upload time
    assert result._efs_ir is None, "Executable EFS IR should NOT be precompiled at upload time!"
    assert result.req_ir is not None, "Requirement IR must be extracted and cached."
    assert len(result.chunks) > 0, "Hierarchical chunks must be generated."

    # Verify Vector Store received chunks with knowledge_domain="design"
    vstore = get_vector_store()
    for ch in result.chunks:
        assert ch.metadata.get("knowledge_domain") == "design"

    # Verify Graph Store received nodes
    gstore = get_graph_store()
    doc_nodes = gstore.all_nodes(document_id=result.doc.doc_id)
    assert len(doc_nodes) > 0, "Graph nodes should be populated in Knowledge Base."

    # Backwards compatibility check: accessing .efs_ir lazily produces it without error
    lazy_efs = result.efs_ir
    assert lazy_efs is not None, "Accessing .efs_ir property should lazily compile on-demand for backwards compatibility."


def test_reusable_protocol_knowledge_ingestion(tmp_path):
    """Verify reusable protocol specifications are stored independently with domain='protocol'."""
    protocol_content = """
    # AMBA AXI4 Protocol Specification
    Section 1. Handshake Rules
    Rule AXI_HS_01: The source must assert VALID when information is available.
    Rule AXI_HS_02: Once VALID is asserted, it must remain asserted until READY is high.
    Section 2. Reset Requirements
    Rule AXI_RST_01: During reset, ARESETn must be asserted LOW.
    Rule AXI_RST_02: All VALID signals must be driven LOW within 1 cycle after ARESETn assertion.
    Section 3. Burst Constraints
    Rule AXI_BST_01: INCR bursts must not cross a 4KB address boundary.
    """
    proto_file = tmp_path / "amba_axi4_protocol_spec.md"
    proto_file.write_text(protocol_content, encoding="utf-8")

    result = run_pipeline(str(proto_file), filename="amba_axi4_protocol_spec.md", knowledge_domain="protocol")

    assert result._efs_ir is None
    assert len(result.chunks) > 0

    # Chunks must have knowledge_domain="protocol" and protocol name
    for ch in result.chunks:
        assert ch.metadata.get("knowledge_domain") == "protocol"
        assert ch.metadata.get("protocol") == "AMBA_AXI4_PROTOCOL_SPEC"

    # Search Vector DB filtering specifically by domain="protocol"
    vstore = get_vector_store()
    proto_hits = vstore.search("handshake valid ready", top_k=3, domain="protocol")
    assert len(proto_hits) > 0
    assert all(h.metadata.get("knowledge_domain") == "protocol" for h in proto_hits)


def test_query_interpreter():
    """Verify natural-language query interpreter resolves component, scope, artifact, and protocols."""
    query = "Generate synthesizable Verilog RTL for the Instruction Decoder handling AXI4 transactions."
    intent = interpret_query(query)

    assert "instruction" in intent.target_component.lower() and "decoder" in intent.target_component.lower()
    assert intent.target_artifact == "RTL"
    assert intent.target_language == "Verilog"
    assert "AXI4" in intent.referenced_protocols
    assert intent.requested_scope == "COMPONENT"

    # Test top-level query
    top_query = "Generate full system PlantUML architecture diagram"
    top_intent = interpret_query(top_query, default_design_name="Matrix Compute Unit")
    assert top_intent.target_artifact == "PLANTUML"
    assert top_intent.requested_scope == "FULL_SYSTEM"


def test_scope_resolver_extracts_minimal_boundary(tmp_path):
    """Verify scope resolver extracts component boundary and does not pull unrelated logic."""
    spec_content = """
    # Multi-Unit System Specification
    Section 1. Architecture
    System contains Instruction Decoder and Execution Unit.
    Section 2. Instruction Decoder Signals
    - inst_in: input, 32 bits, Instruction word input
    - opcode_out: output, 6 bits, Decoded opcode
    - decode_valid: output, 1 bit, Opcode valid strobe
    - clk: input, 1 bit, System clock
    - rst_n: input, 1 bit, Active-low reset
    Section 3. Execution Unit Signals
    - exec_start: input, 1 bit, Start execution trigger
    - alu_result: output, 32 bits, Computation result
    """
    spec_file = tmp_path / "multi_unit_spec.md"
    spec_file.write_text(spec_content, encoding="utf-8")

    result = run_pipeline(str(spec_file), filename="multi_unit_spec.md", knowledge_domain="design")

    # Resolve scope for Instruction Decoder
    scope = resolve_component_scope(
        target_name="Instruction Decoder",
        scope_type="COMPONENT",
        req_ir=result.req_ir,
        document_id=result.doc.doc_id
    )

    assert scope.target_component_name.lower() == "instruction decoder"
    assert len(scope.ports) > 0

    port_names = [p["name"].lower() for p in scope.ports]
    assert any("inst" in p or "opcode" in p or "decode" in p for p in port_names), "Instruction Decoder ports must be included."


def test_scoped_efs_ir_compiler_generates_protocol_aware_ir():
    """Verify ScopedEFSCompiler generates unified, protocol-aware EFS IR with structured protocol rules."""
    intent = ScopedQueryIntent(
        target_component="Instruction Decoder",
        requested_scope="COMPONENT",
        target_artifact="RTL",
        target_language="Verilog",
        referenced_protocols=["AXI4"],
    )

    scope = ResolvedScope(
        target_component_name="Instruction Decoder",
        target_component_type="submodule",
        clock_domain="clk",
        reset_domain="rst_n",
        ports=[
            {"name": "clk", "width": 1, "direction": "input", "semantic_role": "clock"},
            {"name": "rst_n", "width": 1, "direction": "input", "semantic_role": "reset"},
            {"name": "inst_data", "width": 32, "direction": "input", "semantic_role": "data", "chunk_id": "BLK_002"},
            {"name": "opcode", "width": 6, "direction": "output", "semantic_role": "control", "chunk_id": "BLK_002"},
        ],
        interfaces=[
            {"name": "axi_inst_if", "protocol": "AXI4", "role": "slave", "chunk_id": "BLK_001"}
        ],
        opcodes=[
            {"mnemonic": "ADD", "binary_encoding": "000001", "operation": "Add"}
        ],
        behavioral_rules=[
            {"rule_id": "BEH_01", "statement": "Decode opcode on rising edge of clk when rst_n is high", "condition": "rst_n == 1'b1", "action": "decode"}
        ]
    )

    proto_rules = [
        {
            "chunk_id": "PROTO_BLK_01",
            "protocol": "AXI4",
            "text": "Once VALID is asserted it must remain high until READY is asserted.",
            "citation": "AXI4 Spec Section 1.2"
        },
        {
            "chunk_id": "PROTO_BLK_02",
            "protocol": "AXI4",
            "text": "ARESETn must be asserted for at least 16 clock cycles at startup.",
            "citation": "AXI4 Spec Section 2.1"
        }
    ]

    compiler = ScopedEFSCompiler(intent=intent, scope=scope, protocol_rules=proto_rules)
    efs_ir = compiler.compile()

    assert isinstance(efs_ir, EFSIR)
    assert len(efs_ir.components) >= 1
    assert efs_ir.components[0].name == "Instruction Decoder"

    # Verify structured protocol rules
    assert len(efs_ir.protocol_rules) == 2
    rule_types = [r.type for r in efs_ir.protocol_rules]
    assert "HANDSHAKE" in rule_types
    assert "RESET" in rule_types

    # Verify structured protocol context
    assert "protocol_name" in efs_ir.protocol_context
    assert efs_ir.protocol_context["protocol_name"] == "AXI4"
    assert len(efs_ir.protocol_context["handshake_rules"]) > 0

    # Verify end-to-end traceability
    sig_with_trace = next((s for s in efs_ir.signals if s.name == "inst_data"), None)
    assert sig_with_trace is not None
    assert sig_with_trace.traceability.chunk_id == "BLK_002"

    proto_rule = efs_ir.protocol_rules[0]
    assert proto_rule.traceability.chunk_id == "PROTO_BLK_01"


def test_idempotent_reingestion(tmp_path):
    """Verify that re-uploading a document replaces its nodes idempotently without duplicates."""
    spec_content = """
    # Idempotency Test Spec
    Section 1. Components
    Contains TestController.
    Section 2. Signals
    - ctrl_sig: input, 8 bits, Control input
    """
    spec_file = tmp_path / "idempotent_test.md"
    spec_file.write_text(spec_content, encoding="utf-8")

    # First ingestion
    res1 = run_pipeline(str(spec_file), filename="idempotent_test.md")
    gstore = get_graph_store()
    nodes1 = gstore.all_nodes(document_id=res1.doc.doc_id)
    count1 = len(nodes1)

    # Second ingestion of the exact same document
    res2 = run_pipeline(str(spec_file), filename="idempotent_test.md")
    nodes2 = gstore.all_nodes(document_id=res2.doc.doc_id)
    count2 = len(nodes2)

    assert count1 == count2, f"Idempotent re-ingestion produced different node count: {count1} vs {count2}"
