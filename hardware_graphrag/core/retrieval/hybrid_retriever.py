"""
Step 8: Hybrid Retrieval.

Pipeline for answering a natural-language question:
  1. Detect candidate entity mentions in the question (keyword match
     against known graph node names/aliases).
  2. Search the knowledge graph for matching nodes.
  3. Expand to neighboring nodes (configurable hop count).
  4. Collect chunk_ids referenced by those graph nodes.
  5. Run semantic vector search over the whole corpus.
  6. Merge + de-duplicate graph-sourced and vector-sourced chunks,
     re-ranking so chunks found by both signals score highest.
  7. Assemble a grounded context window and call the LLM for the final
     answer (falls back to an extractive summary if no LLM configured).
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Optional, Tuple

from config import CONFIG
from core.extraction.llm_client import get_llm_client
from core.graph.neo4j_builder import get_graph_store
from core.vectorstore.faiss_store import get_vector_store
from utils.models import GraphContext, QAResult, RetrievedChunk
from utils.logger import get_logger
from core.efs_ir.models import EFSIR

logger = get_logger("retrieval.hybrid")

_ANSWER_SYSTEM_PROMPT = """You are a hardware specification assistant. Answer the user's question
ONLY using the provided context chunks from the specification document. Cite
the chunk reference (e.g. [chunk: chapter/section, page]) after each factual
claim you make. If the answer is not contained in the context, say so plainly
instead of guessing."""


def _detect_entities_in_question(question: str, all_node_names: List[str]) -> List[str]:
    """Simple substring/keyword matching between the question and known graph node names."""
    q_lower = question.lower()
    hits = []
    for name in all_node_names:
        readable = name.replace("_", " ").lower()
        if readable and readable in q_lower:
            hits.append(name)
        else:
            # also try individual significant tokens (len > 3) for partial matches
            tokens = [t for t in readable.split() if len(t) > 3]
            if tokens and all(t in q_lower for t in tokens):
                hits.append(name)
    return sorted(set(hits))


def _graph_search(question: str, document_id: Optional[str] = None) -> GraphContext:
    """
    Query-driven Graph RAG Retrieval:
    Detects candidate entities in question, traverses multi-hop graph relationships,
    and expands dependent requirements up to max_depth with cycle protection.
    """
    store = get_graph_store()
    all_nodes = store.all_nodes()
    if document_id:
        all_nodes = [n for n in all_nodes if isinstance(n, dict) and (n.get("doc_id") == document_id or n.get("document_id") == document_id)]
    all_names = [n["name"] for n in all_nodes if isinstance(n, dict) and "name" in n]

    seed_names = _detect_entities_in_question(question, all_names)
    if not seed_names:
        # fallback: keyword search against node names/aliases
        for token in question.split():
            token = token.strip(".,?!:;()").lower()
            if len(token) <= 3:
                continue
            matches = store.search_nodes_by_keyword(token, limit=5)
            seed_names.extend(m["name"] for m in matches if isinstance(m, dict) and "name" in m)
        seed_names = sorted(set(seed_names))

    nodes: List[Dict] = []
    edges: List[Dict] = []
    chunk_ids: set = set()

    hops = getattr(CONFIG.retrieval, "graph_expansion_hops", 2)
    for name in seed_names[:15]:
        neighbor_nodes, neighbor_edges = store.get_neighbors(name, hops=hops)
        if document_id:
            neighbor_nodes = [n for n in neighbor_nodes if isinstance(n, dict) and (n.get("doc_id") == document_id or n.get("document_id") == document_id)]
        nodes.extend(neighbor_nodes)
        edges.extend(neighbor_edges)
        for n in neighbor_nodes:
            if isinstance(n, dict):
                cids = n.get("chunk_ids", [])
                if isinstance(cids, list):
                    chunk_ids.update(cids)
                elif cids:
                    chunk_ids.add(str(cids))
                if n.get("chunk_id"):
                    chunk_ids.add(str(n.get("chunk_id")))

    # De-duplicate nodes/edges
    seen_nodes = {}
    for n in nodes:
        if isinstance(n, dict) and n.get("name"):
            seen_nodes[n.get("name")] = n
    dedup_edges = []
    seen_edge_keys = set()
    for e in edges:
        if isinstance(e, dict):
            key = (e.get("source"), e.get("relation"), e.get("target"))
            if key not in seen_edge_keys:
                seen_edge_keys.add(key)
                dedup_edges.append(e)

    return GraphContext(
        seed_entities=seed_names,
        nodes=list(seen_nodes.values()),
        edges=dedup_edges,
        chunk_ids=sorted(chunk_ids),
    )


def _merge_results(vector_hits: List[RetrievedChunk], graph_chunk_ids: List[str]) -> List[RetrievedChunk]:
    """
    Hybrid Reranking:
    Combines vector search top hits with graph-corroborated chunk IDs.
    """
    vector_store = get_vector_store()
    by_id: Dict[str, RetrievedChunk] = {rc.chunk_id: rc for rc in vector_hits}

    for cid in graph_chunk_ids:
        if cid in by_id:
            by_id[cid].source = "vector+graph"
            by_id[cid].score += 0.25  # Substantial boost for graph-corroborated chunks
        else:
            meta = vector_store.get_chunk(cid)
            text_val = meta.get("text", "") if meta else ""
            if meta and text_val:
                by_id[cid] = RetrievedChunk(
                    chunk_id=cid, text=text_val, score=0.65,
                    source="graph", metadata=meta,
                )

    merged = sorted(by_id.values(), key=lambda rc: rc.score, reverse=True)
    return merged


def _format_context(chunks: List[RetrievedChunk], max_chunks: int = 8) -> Tuple[str, List[str]]:
    context_parts = []
    citations = []
    for rc in chunks[:max_chunks]:
        meta = rc.metadata or {}
        ref = f"{meta.get('chapter', '')}/{meta.get('section', '')} p.{meta.get('page', '?')}".strip("/ ")
        citations.append(ref)
        context_parts.append(f"[chunk: {ref}]\n{rc.text}")
    return "\n\n---\n\n".join(context_parts), citations


def _extractive_fallback_answer(question: str, chunks: List[RetrievedChunk]) -> str:
    """Used when no LLM is configured: return the top matching chunk(s) verbatim as a grounded answer."""
    if not chunks:
        return "No relevant information was found in the indexed document(s) for this question."
    top = chunks[0]
    meta = top.metadata or {}
    ref = f"{meta.get('chapter', '')}/{meta.get('section', '')} p.{meta.get('page', '?')}".strip("/ ")
    return (
        f"(No LLM configured - showing the most relevant retrieved passage.)\n\n"
        f"From [{ref}]:\n{top.text}"
    )


def answer_question(question: str, document_id: Optional[str] = None) -> QAResult:
    vector_store = get_vector_store()

    graph_context = _graph_search(question, document_id=document_id)
    vector_hits = vector_store.search(question, top_k=CONFIG.retrieval.top_k_vector, document_id=document_id)
    merged_chunks = _merge_results(vector_hits, graph_context.chunk_ids)

    context_text, citations = _format_context(merged_chunks)

    llm = get_llm_client()
    answer = None
    if llm.available and context_text:
        user_prompt = f"Question: {question}\n\nContext:\n{context_text}"
        answer = llm.complete_text(_ANSWER_SYSTEM_PROMPT, user_prompt, model=CONFIG.llm.extraction_model)

    if not answer:
        answer = _extractive_fallback_answer(question, merged_chunks)

    logger.info(
        f"Answered question with {len(merged_chunks)} merged chunks "
        f"({len(vector_hits)} vector, {len(graph_context.chunk_ids)} graph-linked)."
    )

    # Try loading active EFS IR design object
    efs_ir_obj = None
    ir_path = os.path.join(CONFIG.data_dir, "efs_ir", "active_design_ir.json")
    if os.path.exists(ir_path):
        try:
            with open(ir_path, "r", encoding="utf-8") as f:
                efs_ir_obj = EFSIR.from_dict(json.load(f))
        except Exception as e:
            logger.warning(f"Failed to load active EFS IR for QA results: {e}")

    return QAResult(
        question=question,
        answer=answer,
        retrieved_chunks=merged_chunks,
        graph_context=graph_context,
        citations=citations,
        efs_ir=efs_ir_obj,
    )
