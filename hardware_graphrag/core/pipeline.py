"""
End-to-end ingestion pipeline orchestrator.

Wires together every stage of the Generic Specification Compiler Pipeline:
    Document Parsing -> Document IR -> Generic Requirement Extraction ->
    Cross-Reference Resolution -> Requirement Graph -> Semantic Chunking ->
    Pre-Validation & Spec Conflict Checking -> EFS IR Compiler -> Vector/Graph DB Index
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from core.parsing import get_parser
from core.chunking.hierarchical_chunker import chunk_document
from core.extraction.entity_extractor import extract_entities
from core.extraction.relationship_extractor import extract_relationships
from core.normalization.entity_normalizer import normalize_entities
from core.graph.neo4j_builder import build_graph
from core.vectorstore.faiss_store import get_vector_store
from utils.models import Chunk, Entity, ParsedDocument, Relationship
from utils.logger import get_logger

# New Generic Pipeline Modules
from core.requirement_ir.models import DocumentIR, DocumentBlock, RequirementIR, SourceLocation
from core.extraction.requirement_extractor import GenericRequirementExtractor
from core.requirement_ir.reference_resolver import CrossReferenceResolver
from core.requirement_ir.graph import RequirementDependencyGraph
from core.requirement_ir.chunker import build_semantic_chunks
from core.requirement_ir.validator import RequirementValidator
from core.compiler.efs_compiler import RequirementToEFSCompiler
from core.cache.doc_cache import compute_document_hash, get_cached_requirement_ir, save_requirement_ir_to_cache
from core.efs_ir.models import EFSIR

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

        # Helper to combine efs_ir items with req_ir entities/requirements
        inst_list = [i.to_dict() for i in efs_ir.instructions]
        if not inst_list:
            inst_list = [e.to_dict() for e in req_ir.entities if e.type.lower() in ("instruction", "inst", "mnemonic")]

        op_list = [i.to_dict() for i in efs_ir.opcodes]
        if not op_list:
            op_list = [e.to_dict() for e in req_ir.entities if e.type.lower() in ("opcode", "encoding")]

        mem_list = [m.to_dict() for m in efs_ir.memory_regions]
        if not mem_list:
            mem_list = [e.to_dict() for e in req_ir.entities if e.type.lower() in ("memory", "memory_region", "sram", "dram", "cache", "buffer")]

        ds_list = [d.to_dict() for d in efs_ir.data_structures]
        if not ds_list:
            ds_list = [e.to_dict() for e in req_ir.entities if e.type.lower() in ("data_structure", "descriptor", "packet", "message")]

        err_list = [e.to_dict() for e in efs_ir.errors]
        if not err_list:
            err_list = [e.to_dict() for e in req_ir.entities if e.type.lower() in ("error", "exception", "fault")]

        perf_list = [p.to_dict() for p in efs_ir.performance_requirements if p.metric in ("throughput", "latency", "frequency", "performance")]
        if not perf_list:
            perf_list = [e.to_dict() for e in req_ir.entities if e.type.lower() == "performance_requirement"]

        power_list = [p.to_dict() for p in efs_ir.performance_requirements if p.metric == "power"]
        if not power_list:
            power_list = [e.to_dict() for e in req_ir.entities if e.type.lower() == "power_requirement"]

        sec_list = [p.to_dict() for p in efs_ir.performance_requirements if p.metric == "security"]
        if not sec_list:
            sec_list = [e.to_dict() for e in req_ir.entities if e.type.lower() == "security_requirement"]

        return {
            "document": {
                "doc_id": self.doc.doc_id,
                "filename": self.doc.filename,
                "file_type": self.doc.file_type,
                "total_pages": self.doc.total_pages,
                "raw_text_len": self.doc.raw_text_len
            },
            "sections": [s.to_dict() for s in self.doc.sections],
            "entities": [e.to_dict() for e in req_ir.entities],
            "requirements": [r.to_dict() for r in req_ir.requirements],
            "relationships": [ed.to_dict() for ed in req_ir.edges],
            "behaviors": [r.to_dict() for r in req_ir.requirements if r.type in ("BEHAVIOR", "SEQUENCE", "FLOW")],
            "transactions": [t.to_dict() for t in efs_ir.transactions],
            "states": [s.to_dict() for f in efs_ir.fsms for s in f.states],
            "transitions": [t.to_dict() for t in efs_ir.fsms for t in f.transitions],
            "conditions": [c.to_dict() for c in efs_ir.constraints if c.condition],
            "events": [r.to_dict() for r in req_ir.requirements if r.event],
            "timing": [t.to_dict() for t in efs_ir.timing_rules],
            "interfaces": [i.to_dict() for i in efs_ir.interfaces],
            "registers": [r.to_dict() for r in efs_ir.registers],
            "operations": op_list,
            "instructions": inst_list,
            "memory": mem_list,
            "data_structures": ds_list,
            "errors": err_list,
            "performance": perf_list,
            "power": power_list,
            "security": sec_list,
            "subflows": [f.to_dict() for f in efs_ir.flows],
            "constraints": [c.to_dict() for c in efs_ir.constraints],
            "conflicts": [c.to_dict() for c in req_ir.conflicts],
            "source_traceability": [c.to_dict() for c in req_ir.chunks],
            "efs": efs_ir.to_dict(),
            "extraction_coverage": coverage,
            "missing_information_report": missing_report,
            "pipeline_validation_report": val_report,
            "semantic_inventory": inventory
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

    # -- Step 2: Build Document IR & Check Cache ---------------------------
    import uuid
    doc_hash = compute_document_hash(filepath)
    cached_bundle = get_cached_requirement_ir(doc_hash)
    ingestion_id = f"ING_{uuid.uuid4().hex[:8]}"
    parsed_doc.document_hash = doc_hash
    parsed_doc.ingestion_id = ingestion_id

    doc_ir = DocumentIR(
        document_id=parsed_doc.doc_id,
        document_version="1.0",
        source_type=parsed_doc.file_type,
        document_hash=doc_hash,
        ingestion_id=ingestion_id,
        metadata={"filename": parsed_doc.filename, "document_hash": doc_hash, "ingestion_id": ingestion_id},
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
                    document_hash=doc_hash,
                    section=s.title,
                    section_id=s.section_id,
                    page=s.page_start,
                    block_id=f"BLK_{i+1:03d}",
                    original_text=s.text[:200]
                )
            ) for i, s in enumerate(parsed_doc.sections) if s.text and s.text.strip()
        ]
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
        save_requirement_ir_to_cache(doc_hash, doc_ir, req_ir)

    # Save active EFS IR representation to disk so UI displays current document IR
    if efs_ir:
        try:
            import json, os
            from config import CONFIG
            ir_dir = os.path.join(CONFIG.data_dir, "efs_ir")
            os.makedirs(ir_dir, exist_ok=True)
            active_ir_path = os.path.join(ir_dir, "active_design_ir.json")
            with open(active_ir_path, "w", encoding="utf-8") as f:
                json.dump(efs_ir.to_dict(), f, indent=2)
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
    for ent in req_ir.entities:
        normalized_entities.append(Entity(
            entity_id=ent.entity_id,
            name=ent.name,
            raw_name=ent.name,
            entity_type=ent.type,
            chunk_id=ent.source_references[0].block_id if ent.source_references else "BLK_001",
            doc_id=parsed_doc.doc_id,
            page=ent.source_references[0].page if ent.source_references and ent.source_references[0].page else 1,
            section=ent.source_references[0].section if ent.source_references and ent.source_references[0].section else ""
        ))

    for ed in req_ir.edges:
        all_relationships.append(Relationship(
            rel_id=new_req_id("rel"),
            source=ed.source,
            relation=ed.relationship,
            target=ed.target,
            chunk_id=ed.source_reference.block_id or "BLK_001",
            doc_id=parsed_doc.doc_id
        ))

    try:
        graph_stats = build_graph(normalized_entities, all_relationships, efs_ir=efs_ir)
    except Exception as e:
        graph_stats = {}

    try:
        vector_store = get_vector_store()
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
    )
