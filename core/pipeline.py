"""
End-to-end ingestion pipeline orchestrator.

Wires together every stage of the Generic Specification Compiler Pipeline:
    Document Parsing -> Document IR -> Generic Requirement Extraction ->
    Cross-Reference Resolution -> Requirement Graph -> Semantic Chunking ->
    Pre-Validation & Spec Conflict Checking -> EFS IR Compiler -> Vector/Graph DB Index
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from config import CONFIG
from core.parsing import get_parser
from core.chunking.hierarchical_chunker import chunk_document
from core.graph.neo4j_builder import build_graph
from core.vectorstore.faiss_store import get_vector_store
from utils.models import Chunk, Entity, ParsedDocument, Relationship, new_id
from utils.logger import get_logger

# New Generic Pipeline Modules
from core.requirement_ir.models import (
    DocumentIR, DocumentBlock, RequirementIR, SourceLocation, new_req_id
)
from core.extraction.requirement_extractor import GenericRequirementExtractor
from core.requirement_ir.reference_resolver import CrossReferenceResolver
from core.requirement_ir.graph import RequirementDependencyGraph
from core.requirement_ir.chunker import build_semantic_chunks
from core.requirement_ir.validator import RequirementValidator
from core.compiler.efs_compiler import RequirementToEFSCompiler
from core.cache.doc_cache import (
    compute_document_hash, get_cached_requirement_ir, save_requirement_ir_to_cache
)
from core.efs_ir.models import EFSIR
from core.extraction.source_evidence import SourceEvidenceIndex
from core.extraction.source_recovery import SourceBackedRecovery
from core.verification.conflict_resolution_engine import ConflictResolutionEngine
from core.pipeline_diagnostics import calculate_extraction_quality_metrics

logger = get_logger("pipeline")

ProgressCallback = Callable[[str, float], None]


@dataclass
class PipelineResult:
    doc: ParsedDocument
    chunks: List[Chunk] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    graph_stats: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    req_ir: Optional[RequirementIR] = None
    _efs_ir: Optional[EFSIR] = None
    extraction_quality: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        doc: ParsedDocument,
        chunks: Optional[List[Chunk]] = None,
        entities: Optional[List[Entity]] = None,
        relationships: Optional[List[Relationship]] = None,
        graph_stats: Optional[Dict[str, int]] = None,
        errors: Optional[List[str]] = None,
        req_ir: Optional[RequirementIR] = None,
        efs_ir: Optional[EFSIR] = None,
        extraction_quality: Optional[Dict[str, Any]] = None,
    ):
        self.doc = doc
        self.chunks = chunks if chunks is not None else []
        self.entities = entities if entities is not None else []
        self.relationships = relationships if relationships is not None else []
        self.graph_stats = graph_stats if graph_stats is not None else {}
        self.errors = errors if errors is not None else []
        self.req_ir = req_ir
        self._efs_ir = efs_ir
        self.extraction_quality = extraction_quality if extraction_quality is not None else {}

    @property
    def efs_ir(self) -> Optional[EFSIR]:
        if self._efs_ir is None and self.req_ir is not None:
            try:
                from core.compiler.efs_compiler import RequirementToEFSCompiler
                compiler = RequirementToEFSCompiler(self.req_ir)
                self._efs_ir = compiler.compile()
            except Exception:
                pass
        return self._efs_ir

    @efs_ir.setter
    def efs_ir(self, val: Optional[EFSIR]):
        self._efs_ir = val

    def to_full_json(self) -> Dict[str, Any]:
        """Produce full complete combined JSON representation across all extracted categories."""
        req_ir = self.req_ir or RequirementIR()
        efs_ir = self.efs_ir or EFSIR()

        from core.pipeline_diagnostics import (
            calculate_extraction_coverage, generate_missing_information_report,
            generate_pipeline_validation_report, generate_semantic_inventory
        )

        coverage = calculate_extraction_coverage(req_ir, efs_ir)
        missing_report = generate_missing_information_report(req_ir)
        val_report = generate_pipeline_validation_report(req_ir, efs_ir)
        inventory = generate_semantic_inventory(req_ir, efs_ir)

        # Helper to combine efs_ir items with req_ir entities/requirements safely
        inst_list = [i.to_dict() if hasattr(i, "to_dict") else i for i in getattr(efs_ir, "instructions", [])]
        if not inst_list:
            inst_list = [e.to_dict() if hasattr(e, "to_dict") else e for e in getattr(req_ir, "entities", []) if getattr(e, "type", "").lower() in ("instruction", "inst", "mnemonic")]

        op_list = [i.to_dict() if hasattr(i, "to_dict") else i for i in getattr(efs_ir, "opcodes", [])]
        if not op_list:
            op_list = [e.to_dict() if hasattr(e, "to_dict") else e for e in getattr(req_ir, "entities", []) if getattr(e, "type", "").lower() in ("opcode", "encoding")]

        mem_list = [m.to_dict() if hasattr(m, "to_dict") else m for m in getattr(efs_ir, "memory_regions", [])]
        if not mem_list:
            mem_list = [e.to_dict() if hasattr(e, "to_dict") else e for e in getattr(req_ir, "entities", []) if getattr(e, "type", "").lower() in ("memory", "memory_region", "sram", "dram", "cache", "buffer")]

        ds_list = [d.to_dict() if hasattr(d, "to_dict") else d for d in getattr(efs_ir, "data_structures", [])]
        if not ds_list:
            ds_list = [e.to_dict() if hasattr(e, "to_dict") else e for e in getattr(req_ir, "entities", []) if getattr(e, "type", "").lower() in ("data_structure", "descriptor", "packet", "message")]

        err_list = [e.to_dict() if hasattr(e, "to_dict") else e for e in getattr(efs_ir, "errors", [])]
        if not err_list:
            err_list = [e.to_dict() if hasattr(e, "to_dict") else e for e in getattr(req_ir, "entities", []) if getattr(e, "type", "").lower() in ("error", "exception", "fault")]

        perf_list = [p.to_dict() if hasattr(p, "to_dict") else p for p in getattr(efs_ir, "performance_requirements", []) if getattr(p, "metric", "") in ("throughput", "latency", "frequency", "performance")]
        if not perf_list:
            perf_list = [e.to_dict() if hasattr(e, "to_dict") else e for e in getattr(req_ir, "entities", []) if getattr(e, "type", "").lower() == "performance_requirement"]

        power_list = [p.to_dict() if hasattr(p, "to_dict") else p for p in getattr(efs_ir, "performance_requirements", []) if getattr(p, "metric", "") == "power"]
        if not power_list:
            power_list = [e.to_dict() if hasattr(e, "to_dict") else e for e in getattr(req_ir, "entities", []) if getattr(e, "type", "").lower() == "power_requirement"]

        sec_list = [p.to_dict() if hasattr(p, "to_dict") else p for p in getattr(efs_ir, "performance_requirements", []) if getattr(p, "metric", "") == "security"]
        if not sec_list:
            sec_list = [e.to_dict() if hasattr(e, "to_dict") else e for e in getattr(req_ir, "entities", []) if getattr(e, "type", "").lower() == "security_requirement"]

        # Safe extraction for nested FSM states and transitions
        fsm_states = []
        fsm_transitions = []
        for f in getattr(efs_ir, "fsms", []):
            for s in getattr(f, "states", []):
                fsm_states.append(s.to_dict() if hasattr(s, "to_dict") else (s if isinstance(s, dict) else {"name": str(s)}))
            for t in getattr(f, "transitions", []):
                fsm_transitions.append(t.to_dict() if hasattr(t, "to_dict") else t)

        return {
            "document": {
                "doc_id": getattr(self.doc, "doc_id", "unknown"),
                "filename": getattr(self.doc, "filename", "unknown"),
                "file_type": getattr(self.doc, "file_type", "unknown"),
                "total_pages": getattr(self.doc, "total_pages", 0),
                "raw_text_len": getattr(self.doc, "raw_text_len", 0)
            },
            "sections": [s.to_dict() if hasattr(s, "to_dict") else s for s in getattr(self.doc, "sections", [])],
            "entities": [e.to_dict() if hasattr(e, "to_dict") else e for e in getattr(req_ir, "entities", [])],
            "requirements": [r.to_dict() if hasattr(r, "to_dict") else r for r in getattr(req_ir, "requirements", [])],
            "relationships": [ed.to_dict() if hasattr(ed, "to_dict") else ed for ed in getattr(req_ir, "edges", [])],
            "behaviors": [r.to_dict() if hasattr(r, "to_dict") else r for r in getattr(req_ir, "requirements", []) if getattr(r, "type", "") in ("BEHAVIOR", "SEQUENCE", "FLOW")],
            "functional_behavior": [fb.to_dict() if hasattr(fb, "to_dict") else fb for fb in getattr(efs_ir, "functional_behavior", [])],
            "transactions": [t.to_dict() if hasattr(t, "to_dict") else t for t in getattr(efs_ir, "transactions", [])],
            "states": fsm_states,
            "transitions": fsm_transitions,
            "conditions": [c.to_dict() if hasattr(c, "to_dict") else c for c in getattr(efs_ir, "constraints", []) if getattr(c, "condition", None)],
            "events": [r.to_dict() if hasattr(r, "to_dict") else r for r in getattr(req_ir, "requirements", []) if getattr(r, "event", None)],
            "timing": [t.to_dict() if hasattr(t, "to_dict") else t for t in getattr(efs_ir, "timing_rules", [])],
            "interfaces": [i.to_dict() if hasattr(i, "to_dict") else i for i in getattr(efs_ir, "interfaces", [])],
            "registers": [r.to_dict() if hasattr(r, "to_dict") else r for r in getattr(efs_ir, "registers", [])],
            "parameters": [p.to_dict() if hasattr(p, "to_dict") else p for p in getattr(efs_ir, "parameters", [])],
            "operations": op_list,
            "instructions": inst_list,
            "instruction_formats": [fmt.to_dict() if hasattr(fmt, "to_dict") else fmt for fmt in getattr(efs_ir, "instruction_formats", [])],
            "opcodes": [op.to_dict() if hasattr(op, "to_dict") else op for op in getattr(efs_ir, "opcodes", [])],
            "memory": mem_list,
            "data_structures": ds_list,
            "errors": err_list,
            "performance": perf_list,
            "power": power_list,
            "security": sec_list,
            "subflows": [f.to_dict() if hasattr(f, "to_dict") else f for f in getattr(efs_ir, "flows", [])],
            "constraints": [c.to_dict() if hasattr(c, "to_dict") else c for c in getattr(efs_ir, "constraints", [])],
            "conflicts": [c.to_dict() if hasattr(c, "to_dict") else c for c in getattr(req_ir, "conflicts", [])],
            "source_traceability": [c.to_dict() if hasattr(c, "to_dict") else c for c in getattr(req_ir, "chunks", [])],
            "efs": efs_ir.to_dict() if hasattr(efs_ir, "to_dict") else {},
            "extraction_coverage": coverage,
            "missing_information_report": missing_report,
            "pipeline_validation_report": val_report,
            "semantic_inventory": inventory,
            "extraction_stats": getattr(req_ir, "stats", {}),
            "extraction_quality_metrics": self.extraction_quality
        }


def _noop_progress(msg: str, pct: float) -> None:
    logger.debug(f"[{pct*100:5.1f}%] {msg}")


def run_pipeline(
    filepath: str,
    filename: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
    knowledge_domain: Optional[str] = None
) -> PipelineResult:
    """Run the knowledge ingestion pipeline for an uploaded specification document."""
    progress_cb = progress_cb or _noop_progress
    errors: List[str] = []

    # Determine knowledge domain (reusable protocol vs design specification)
    if knowledge_domain:
        domain = knowledge_domain
    else:
        fn_lower = (filename or os.path.basename(filepath)).lower()
        if any(k in fn_lower for k in ["protocol", "amba", "axi", "apb", "ahb", "pcie", "uart", "spi", "i2c"]):
            domain = "protocol"
        else:
            domain = "design"

    # -- Step 1: Document Parsing -----------------------------------------
    progress_cb("Parsing document structure...", 0.05)
    try:
        parser = get_parser(filepath, filename)
        parsed_doc = parser.parse()
    except Exception as e:
        logger.exception("Document parsing failed")
        errors.append(f"Parsing failed: {e}")
        empty_doc = ParsedDocument(doc_id="unknown", filename=filename or filepath, file_type="unknown")
        return PipelineResult(doc=empty_doc, errors=errors)

    progress_cb(f"Parsed {len(parsed_doc.sections)} sections across {parsed_doc.total_pages} pages.", 0.10)

    from core.registry import get_registry
    reg = get_registry()
    reg.reset_design_session(parsed_doc.doc_id)

    # -- Step 2: Build Document IR & Check Cache ---------------------------
    doc_hash = compute_document_hash(filepath)
    cached_bundle = get_cached_requirement_ir(doc_hash)
    ingestion_id = f"ING_{uuid.uuid4().hex[:8]}"
    parsed_doc.document_hash = doc_hash
    parsed_doc.ingestion_id = ingestion_id

    blocks = [
        DocumentBlock(
            block_id=f"BLK_{i+1:03d}",
            type=getattr(s, "content_type", "text") or "text",
            section_id=getattr(s, "section_id", f"SEC_{i+1:03d}") or f"SEC_{i+1:03d}",
            section_title=getattr(s, "title", "") or f"Section {i+1}",
            page=getattr(s, "page_start", 1) or 1,
            text=s.text,
            source_location=SourceLocation(
                document_id=parsed_doc.doc_id,
                document_hash=doc_hash,
                section=getattr(s, "title", "") or f"Section {i+1}",
                section_id=getattr(s, "section_id", f"SEC_{i+1:03d}") or f"SEC_{i+1:03d}",
                page=getattr(s, "page_start", 1) or 1,
                block_id=f"BLK_{i+1:03d}",
                original_text=s.text[:200]
            )
        ) for i, s in enumerate(parsed_doc.sections) if getattr(s, "text", None) and s.text.strip()
    ]

    # Guarantee at least one valid DocumentBlock if sections contain titles or doc has text
    if not blocks:
        fallback_text = "\n\n".join(s.title for s in parsed_doc.sections if getattr(s, "title", None)) or "Empty specification document."
        blocks = [
            DocumentBlock(
                block_id="BLK_001",
                type="text",
                section_id="SEC_001",
                section_title="Document Root",
                page=1,
                text=fallback_text,
                source_location=SourceLocation(
                    document_id=parsed_doc.doc_id,
                    document_hash=doc_hash,
                    section="Document Root",
                    section_id="SEC_001",
                    page=1,
                    block_id="BLK_001",
                    original_text=fallback_text[:200]
                )
            )
        ]

    doc_ir = DocumentIR(
        document_id=parsed_doc.doc_id,
        document_version="1.0",
        source_type=parsed_doc.file_type,
        document_hash=doc_hash,
        ingestion_id=ingestion_id,
        metadata={"filename": parsed_doc.filename, "document_hash": doc_hash, "ingestion_id": ingestion_id},
        sections=[s.to_dict() if hasattr(s, "to_dict") else s for s in parsed_doc.sections],
        blocks=blocks
    )

    req_ir: Optional[RequirementIR] = None
    efs_ir: Optional[EFSIR] = None

    if cached_bundle:
        progress_cb("Reusing cached Requirement IR representation...", 0.35)
        _, req_ir = cached_bundle
        req_ir.document_id = parsed_doc.doc_id
        req_ir.document_hash = doc_hash
        req_ir.ingestion_id = ingestion_id
        if hasattr(req_ir, "doc_ir") and req_ir.doc_ir:
            req_ir.doc_ir.document_id = parsed_doc.doc_id
            req_ir.doc_ir.document_hash = doc_hash
            req_ir.doc_ir.ingestion_id = ingestion_id
        if hasattr(req_ir, "entities") and req_ir.entities:
            for ent in req_ir.entities:
                for ref in getattr(ent, "source_references", []):
                    ref.document_id = parsed_doc.doc_id
        if hasattr(req_ir, "requirements") and req_ir.requirements:
            for req in req_ir.requirements:
                for ref in getattr(req, "source_references", []):
                    ref.document_id = parsed_doc.doc_id
        if hasattr(req_ir, "edges") and req_ir.edges:
            for edge in req_ir.edges:
                ref = getattr(edge, "source_reference", None)
                if ref and hasattr(ref, "document_id"):
                    ref.document_id = parsed_doc.doc_id
        efs_ir = None
    else:
        # -- Step 3: Generic Semantic & Atomic Requirement Extraction -----------
        progress_cb("Extracting atomic requirements and dynamic entities...", 0.25)
        try:
            extractor = GenericRequirementExtractor(doc_ir)
            req_ir = extractor.extract_requirement_ir()
            req_ir.document_id = doc_ir.document_id
            req_ir.document_hash = doc_hash
            req_ir.ingestion_id = ingestion_id
        except Exception as e:
            logger.exception("Requirement extraction failed")
            errors.append(f"Requirement extraction failed: {e}")
            req_ir = RequirementIR(doc_ir=doc_ir, document_id=doc_ir.document_id, document_hash=doc_hash, ingestion_id=ingestion_id)

        # -- Step 4: Cross-Reference Resolution ------------------------------
        progress_cb("Resolving cross-section and table references...", 0.40)
        try:
            resolver = CrossReferenceResolver(doc_ir)
            ref_edges, _ = resolver.resolve_references(req_ir.requirements)
            req_ir.edges.extend(ref_edges)
        except Exception as e:
            logger.warning(f"Cross-reference resolution warning: {e}")

        # -- Step 5: Requirement Graph & Semantic Chunking -------------------
        progress_cb("Building Requirement Dependency Graph and semantic chunks...", 0.55)
        try:
            graph_builder = RequirementDependencyGraph(req_ir)
            semantic_chunks = build_semantic_chunks(req_ir.requirements, req_ir.entities)
            req_ir.chunks = semantic_chunks
        except Exception as e:
            logger.warning(f"Requirement graph building warning: {e}")

        # -- Step 6: Pre-Validation & Conflict Checking -----------------------
        progress_cb("Performing specification consistency and contradiction check...", 0.70)
        try:
            validator = RequirementValidator(req_ir)
            conflicts, val_status = validator.validate()
            req_ir.conflicts = conflicts
            req_ir.validation_status = val_status
        except Exception as e:
            logger.warning(f"Validation warning: {e}")

        # -- Step 7: Persist Extracted Intermediate Knowledge ----------------
        progress_cb("Persisting extracted Requirement IR to persistent knowledge cache...", 0.80)
        try:
            save_requirement_ir_to_cache(doc_hash, doc_ir, req_ir)
        except Exception as ex:
            logger.warning(f"Could not persist cache: {ex}")

        # In Knowledge-Base-Driven architecture, complete executable EFS IR
        # is NOT compiled during document ingestion. It is generated dynamically at query-time.
        efs_ir = None

    extraction_quality: Dict[str, Any] = {}

    # -- Step 8: Hierarchical Chunking & Knowledge Base Ingestion -------------
    progress_cb("Building hierarchical chunks with knowledge domain metadata...", 0.85)
    try:
        chunks = chunk_document(parsed_doc)
        for ch in chunks:
            ch.knowledge_domain = domain
            if not hasattr(ch, "metadata") or ch.metadata is None:
                ch.metadata = {}
            ch.metadata["knowledge_domain"] = domain
            ch.metadata["document_version"] = getattr(parsed_doc, "version", "1.0")
            ch.metadata["document_id"] = parsed_doc.doc_id
            ch.metadata["source_location"] = f"page_{ch.page}_sec_{ch.section}"
            ch.metadata["chunk_type"] = getattr(ch, "content_type", "text")
            if domain == "protocol":
                proto_name = (filename or os.path.basename(filepath)).split(".")[0].upper()
                ch.metadata["protocol"] = proto_name
    except Exception as e:
        logger.exception("Chunking failed")
        chunks = []

    normalized_entities: List[Entity] = []
    all_relationships: List[Relationship] = []

    # Map discovered entities to legacy Entity model for backward compatibility
    if req_ir and getattr(req_ir, "entities", None):
        for ent in req_ir.entities:
            ref = ent.source_references[0] if (getattr(ent, "source_references", None) and ent.source_references) else None
            normalized_entities.append(Entity(
                entity_id=getattr(ent, "entity_id", new_id("ent")),
                name=getattr(ent, "name", "unknown"),
                raw_name=getattr(ent, "name", "unknown"),
                entity_type=getattr(ent, "type", "entity"),
                chunk_id=getattr(ref, "block_id", None) or "BLK_001",
                doc_id=parsed_doc.doc_id,
                page=getattr(ref, "page", None) or 1,
                section=getattr(ref, "section", None) or "",
                knowledge_domain=domain
            ))

    if req_ir and getattr(req_ir, "edges", None):
        for ed in req_ir.edges:
            ref = getattr(ed, "source_reference", None)
            all_relationships.append(Relationship(
                rel_id=new_req_id("rel"),
                source=getattr(ed, "source", ""),
                relation=getattr(ed, "relationship", "RELATED_TO"),
                target=getattr(ed, "target", ""),
                chunk_id=getattr(ref, "block_id", None) or "BLK_001",
                doc_id=parsed_doc.doc_id,
                knowledge_domain=domain
            ))

    try:
        graph_stats = build_graph(
            normalized_entities,
            all_relationships,
            efs_ir=None,
            req_ir=req_ir,
            knowledge_domain=domain
        )
    except Exception as e:
        logger.warning(f"Knowledge graph building error: {e}")
        graph_stats = {}

    try:
        vector_store = get_vector_store()
        vector_store.clear_document(parsed_doc.doc_id)
        vector_store.add_chunks(chunks)
    except Exception as e:
        errors.append(f"Vector store update failed: {e}")

    progress_cb("Knowledge ingestion complete.", 1.0)

    return PipelineResult(
        doc=parsed_doc,
        chunks=chunks,
        entities=normalized_entities,
        relationships=all_relationships,
        graph_stats=graph_stats,
        errors=errors,
        req_ir=req_ir,
        efs_ir=None,
        extraction_quality=extraction_quality,
    )
