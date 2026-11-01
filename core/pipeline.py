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
    efs_ir: Optional[EFSIR] = None
    extraction_quality: Dict[str, Any] = field(default_factory=dict)

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


def run_pipeline(filepath: str, filename: Optional[str] = None,
                  progress_cb: Optional[ProgressCallback] = None) -> PipelineResult:
    """Run the full ingestion and compilation pipeline for a single uploaded document."""
    progress_cb = progress_cb or _noop_progress
    errors: List[str] = []

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
        compiler = RequirementToEFSCompiler(req_ir)
        efs_ir = compiler.compile()
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

        # -- Step 7: EFS IR Compiler ------------------------------------------
        progress_cb("Compiling Requirement IR into canonical EFS IR...", 0.80)
        try:
            compiler = RequirementToEFSCompiler(req_ir)
            efs_ir = compiler.compile()
        except Exception as e:
            logger.exception("EFS IR compilation failed")
            errors.append(f"EFS IR compilation failed: {e}")
            efs_ir = EFSIR()

        # Cache compiled representations
        try:
            save_requirement_ir_to_cache(doc_hash, doc_ir, req_ir)
        except Exception as ex:
            logger.warning(f"Could not persist cache: {ex}")

    # -- Step 7.5: Source Evidence Index, Source-Backed Recovery & Completeness Validation
    extraction_quality: Dict[str, Any] = {}
    if efs_ir and doc_ir:
        progress_cb("Performing source-backed recovery and completeness validation...", 0.82)
        try:
            evidence_index = SourceEvidenceIndex(doc_ir)
            SourceBackedRecovery.recover_efs_ir(efs_ir, doc_ir, evidence_index)

            curr_docs = [d.name if hasattr(d, "name") else str(d) for d in (efs_ir.metadata.source_documents or [])]
            if not curr_docs and getattr(parsed_doc, "filename", None):
                curr_docs = [parsed_doc.filename]
            curr_comps = [c.name for c in (efs_ir.components or [])]
            curr_rels = [
                f"{r.get('source')} -> {r.get('target')}" if isinstance(r, dict) else f"{getattr(r, 'source', '')} -> {getattr(r, 'target', '')}"
                for r in (getattr(efs_ir, "relationships", []) or [])
            ]
            summary_log = (
                f"\nCURRENT DESIGN DOCUMENTS: {curr_docs}\n"
                f"CURRENT CANONICAL COMPONENTS: {curr_comps}\n"
                f"CURRENT INTER-MODULE RELATIONSHIPS: {curr_rels}\n"
            )
            print(summary_log)
            logger.info(summary_log)

            completeness_gaps = ConflictResolutionEngine.validate_hardware_completeness(
                efs_ir, doc_ir=doc_ir, evidence_index=evidence_index
            )
            existing_keys = {c.semantic_key or ConflictResolutionEngine.compute_semantic_key(c) for c in efs_ir.conflicts}
            for c in completeness_gaps:
                k = c.semantic_key or ConflictResolutionEngine.compute_semantic_key(c)
                if k not in existing_keys:
                    existing_keys.add(k)
                    efs_ir.conflicts.append(c)

            efs_ir.conflicts = ConflictResolutionEngine.deduplicate_conflicts(efs_ir.conflicts)

            extraction_quality = calculate_extraction_quality_metrics(efs_ir, doc_ir, evidence_index)
            logger.info(f"Extraction Quality Metrics: {extraction_quality}")
        except Exception as ex:
            logger.warning(f"Source-backed recovery pass error: {ex}")

    # Save active EFS IR representation to disk so UI displays current document IR
    if efs_ir:
        try:
            ir_dir = os.path.join(CONFIG.data_dir, "efs_ir")
            os.makedirs(ir_dir, exist_ok=True)
            active_ir_path = os.path.join(ir_dir, "active_design_ir.json")
            with open(active_ir_path, "w", encoding="utf-8") as f:
                json.dump(efs_ir.to_dict(), f, indent=2, default=str)
            logger.info(f"Persisted active EFS IR to '{active_ir_path}'")
        except Exception as ex:
            logger.warning(f"Could not write active_design_ir.json: {ex}")

    # -- Step 8: Legacy Chunking & Graph/Vector Indexing ----------------------
    progress_cb("Building hierarchical chunks for legacy RAG & vector store...", 0.85)
    try:
        chunks = chunk_document(parsed_doc)
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
                section=getattr(ref, "section", None) or ""
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
                doc_id=parsed_doc.doc_id
            ))

    try:
        graph_stats = build_graph(normalized_entities, all_relationships, efs_ir=efs_ir)
    except Exception as e:
        logger.warning(f"Knowledge graph building error: {e}")
        graph_stats = {}

    try:
        vector_store = get_vector_store()
        vector_store.clear_document(parsed_doc.doc_id)
        vector_store.add_chunks(chunks)
    except Exception as e:
        errors.append(f"Vector store update failed: {e}")

    progress_cb("Pipeline complete.", 1.0)

    return PipelineResult(
        doc=parsed_doc,
        chunks=chunks,
        entities=normalized_entities,
        relationships=all_relationships,
        graph_stats=graph_stats,
        errors=errors,
        req_ir=req_ir,
        efs_ir=efs_ir,
        extraction_quality=extraction_quality,
    )
