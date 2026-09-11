"""
End-to-end ingestion pipeline orchestrator.

Wires together every stage of the GraphRAG pipeline described in the
spec, exposing a single `run_pipeline()` function (used by the Upload
page) that accepts a file path and yields progress updates suitable for
driving a Streamlit progress bar / status log.

    Parsing -> Chunking -> [Vector DB]
                        -> Entity Extraction -> Relationship Extraction
                        -> Entity Normalization -> [Neo4j Knowledge Graph]

Each stage is wrapped in try/except with logging so a failure in one
document doesn't crash the whole app; partial results are still surfaced
where possible.
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


def _noop_progress(msg: str, pct: float) -> None:
    logger.debug(f"[{pct*100:5.1f}%] {msg}")


def run_pipeline(filepath: str, filename: Optional[str] = None,
                  progress_cb: Optional[ProgressCallback] = None) -> PipelineResult:
    """Run the full ingestion pipeline for a single uploaded document."""
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

    progress_cb(f"Parsed {len(parsed_doc.sections)} sections across {parsed_doc.total_pages} pages.", 0.15)

    # -- Step 2: Hierarchical Chunking --------------------------------------
    progress_cb("Building hierarchical chunks...", 0.2)
    try:
        chunks = chunk_document(parsed_doc)
    except Exception as e:
        logger.exception("Chunking failed")
        errors.append(f"Chunking failed: {e}")
        chunks = []
    progress_cb(f"Created {len(chunks)} chunks.", 0.3)

    # -- Step 3 & 4: Entity + Relationship Extraction -----------------------
    all_entities: List[Entity] = []
    chunks_with_entities = []
    total = max(len(chunks), 1)
    for i, chunk in enumerate(chunks):
        try:
            entities = extract_entities(chunk)
        except Exception as e:
            logger.warning(f"Entity extraction failed for chunk {chunk.chunk_id}: {e}")
            entities = []
        all_entities.extend(entities)
        chunks_with_entities.append((chunk, entities))
        if i % max(1, total // 20) == 0:
            progress_cb(f"Extracting entities... ({i+1}/{total} chunks)", 0.3 + 0.25 * (i / total))

    progress_cb(f"Extracted {len(all_entities)} raw entity mentions.", 0.55)

    all_relationships: List[Relationship] = []
    for i, (chunk, entities) in enumerate(chunks_with_entities):
        try:
            rels = extract_relationships(chunk, entities)
        except Exception as e:
            logger.warning(f"Relationship extraction failed for chunk {chunk.chunk_id}: {e}")
            rels = []
        all_relationships.extend(rels)
        if i % max(1, total // 20) == 0:
            progress_cb(f"Extracting relationships... ({i+1}/{total} chunks)", 0.55 + 0.2 * (i / total))

    progress_cb(f"Extracted {len(all_relationships)} relationship triples.", 0.75)

    # -- Step 5: Entity Normalization ----------------------------------------
    progress_cb("Normalizing and merging duplicate entities...", 0.8)
    try:
        normalized_entities = normalize_entities(all_entities)
        # remap relationship source/target from raw names to normalized canonical names
        raw_to_final: Dict[str, str] = {e.raw_name: e.name for e in normalized_entities}
        for r in all_relationships:
            r.source = raw_to_final.get(r.source, r.source)
            r.target = raw_to_final.get(r.target, r.target)
    except Exception as e:
        logger.exception("Entity normalization failed")
        errors.append(f"Normalization failed: {e}")
        normalized_entities = all_entities

    # -- Step 6: Build Neo4j Knowledge Graph ---------------------------------
    progress_cb("Building knowledge graph...", 0.85)
    try:
        graph_stats = build_graph(normalized_entities, all_relationships)
    except Exception as e:
        logger.exception("Graph build failed")
        errors.append(f"Graph build failed: {e}")
        graph_stats = {}

    # -- Step 7: Build Vector Database ---------------------------------------
    progress_cb("Generating embeddings and updating vector index...", 0.92)
    try:
        entity_names_by_chunk: Dict[str, List[str]] = {}
        for e in normalized_entities:
            entity_names_by_chunk.setdefault(e.chunk_id, [])
            if e.name not in entity_names_by_chunk[e.chunk_id]:
                entity_names_by_chunk[e.chunk_id].append(e.name)

        vector_store = get_vector_store()
        vector_store.add_chunks(chunks, entity_names_by_chunk=entity_names_by_chunk)
    except Exception as e:
        logger.exception("Vector store update failed")
        errors.append(f"Vector store update failed: {e}")

    progress_cb("Pipeline complete.", 1.0)

    return PipelineResult(
        doc=parsed_doc,
        chunks=chunks,
        entities=normalized_entities,
        relationships=all_relationships,
        graph_stats=graph_stats,
        errors=errors,
    )
