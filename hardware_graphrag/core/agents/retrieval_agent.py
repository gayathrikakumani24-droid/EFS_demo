"""
Protocol Retrieval Agent.

Retrieves protocol rules, interface specifications, and signal behaviors from the
active GraphRAG databases (FAISS + Neo4j) to inform generation and verification.
"""

from __future__ import annotations

from typing import Any, Dict, List
from config import CONFIG
from core.vectorstore.faiss_store import get_vector_store
from core.graph.neo4j_builder import get_graph_store
from core.retrieval.hybrid_retriever import _graph_search, _merge_results
from utils.models import RetrievedChunk
from utils.logger import get_logger

logger = get_logger("agents.retrieval")


def retrieve_protocol_rules(requirement_model: Dict[str, Any], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Query the active protocol graph/vector store for rules relevant to requirements and design plan."""
    logger.info("Retrieving relevant protocol rules from GraphRAG...")
    
    # Compile a list of search terms based on requirements and plans
    search_queries = []
    
    # Protocols mentioned
    for p in requirement_model.get("protocol_references", []):
        search_queries.append(f"{p} signaling handshake")
        search_queries.append(f"{p} reset phase rules")
        search_queries.append(f"{p} burst transfer address alignment")
        search_queries.append(f"{p} interface signals definition")

    # If FSM or interfaces are mentioned, add key protocol mechanics
    if requirement_model.get("fsm_info") or requirement_model.get("interfaces"):
        search_queries.append("handshake valid ready transition rules")
        search_queries.append("reset signal assertion deassertion behavior")
        search_queries.append("bus write transfers")
        search_queries.append("bus read transfers")
    
    # If registers are mentioned
    if requirement_model.get("registers"):
        search_queries.append("register read write transfer protocol")

    # Add default general terms just in case
    search_queries.extend([
        "signal timing constraints",
        "valid ready handshaking"
    ])
    
    # Deduplicate queries
    search_queries = list(set(search_queries))
    
    vector_store = get_vector_store()
    
    # If no chunks exist, return empty lists gracefully (un-uploaded or empty library)
    if not vector_store.chunk_ids:
        logger.warning("Active vector database is empty. No protocol rules retrieved.")
        return []

    all_merged: Dict[str, RetrievedChunk] = {}
    
    # Run retrieval for each key term (limit top-k per query so we don't blow up context size)
    for query in search_queries[:6]:  # Query top 6 queries to stay fast and selective
        try:
            graph_context = _graph_search(query)
            vector_hits = vector_store.search(query, top_k=3)
            merged = _merge_results(vector_hits, graph_context.chunk_ids)
            
            for chunk in merged:
                # Add or update score
                if chunk.chunk_id not in all_merged or chunk.score > all_merged[chunk.chunk_id].score:
                    all_merged[chunk.chunk_id] = chunk
        except Exception as e:
            logger.error(f"Error querying GraphRAG for '{query}': {e}")
            
    # Sort by score and take top 4 most relevant chunks to stay within model limits
    sorted_chunks = sorted(all_merged.values(), key=lambda c: c.score, reverse=True)[:4]
    
    import json
    import os
    from core.efs_ir.models import EFSIR
    
    # Try loading active EFS IR design object
    active_ir = None
    ir_path = os.path.join(CONFIG.data_dir, "efs_ir", "active_design_ir.json")
    if os.path.exists(ir_path):
        try:
            with open(ir_path, "r", encoding="utf-8") as f:
                active_ir = EFSIR.from_dict(json.load(f))
        except Exception as e:
            logger.warning(f"Failed to load active EFS IR for retrieval matching: {e}")

    results = []
    for chunk in sorted_chunks:
        meta = chunk.metadata or {}
        ref = f"{meta.get('chapter', '')}/{meta.get('section', '')} p.{meta.get('page', '?')}".strip("/ ")
        
        # Resolve linked EFS IR objects
        linked_efs = []
        if active_ir:
            linked_ids = []
            for key in ("component_ids", "interface_ids", "signal_ids", "register_ids", "instruction_ids", "opcode_ids", "fsm_ids", "constraint_ids", "timing_rule_ids", "protocol_rule_ids", "flow_ids"):
                linked_ids.extend(meta.get(key, []))
                
            for efs_id in set(linked_ids):
                obj = None
                if efs_id.startswith("COMP_"):
                    obj = next((c for c in active_ir.components if c.component_id == efs_id), None)
                elif efs_id.startswith("IF_"):
                    obj = next((i for i in active_ir.interfaces if i.interface_id == efs_id), None)
                elif efs_id.startswith("SIG_"):
                    obj = next((s for s in active_ir.signals if s.signal_id == efs_id), None)
                elif efs_id.startswith("REG_"):
                    obj = next((r for r in active_ir.registers if r.register_id == efs_id), None)
                elif efs_id.startswith("INSTR_"):
                    obj = next((i for i in getattr(active_ir, "instructions", []) if i.instruction_id == efs_id), None)
                elif efs_id.startswith("OPC_"):
                    obj = next((o for o in getattr(active_ir, "opcodes", []) if o.opcode_id == efs_id), None)
                elif efs_id.startswith("FSM_"):
                    obj = next((f for f in active_ir.fsms if f.fsm_id == efs_id), None)
                elif efs_id.startswith("CONSTRAINT_"):
                    obj = next((c for c in active_ir.constraints if c.constraint_id == efs_id), None)
                elif efs_id.startswith("FLOW_"):
                    obj = next((f for f in active_ir.flows if f.flow_id == efs_id), None)
                    
                if obj:
                    linked_efs.append({
                        "efs_id": efs_id,
                        "type": obj.__class__.__name__,
                        "name": getattr(obj, "name", ""),
                    })

        results.append({
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "score": chunk.score,
            "source": chunk.source,
            "citation": ref,
            "linked_efs_objects": linked_efs
        })
        
    logger.info(f"Retrieved {len(results)} protocol specification chunks with EFS links.")
    return results
