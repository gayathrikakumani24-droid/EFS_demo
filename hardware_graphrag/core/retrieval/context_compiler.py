"""
Context Compiler module.

Combines EFS IR structural objects, Neo4j dependency traversal, FAISS vector search evidence,
protocol rules, and timing constraints into a minimal, query-driven Task ContextPack.

Implements Target-Specific Protocol Context Resolution and Protocol Context Completeness Check.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from core.efs_ir.models import EFSIR, EFSConflict
from core.retrieval.query_analyzer import analyze_query, TaskIntent
from core.retrieval.target_resolver import resolve_target, TargetResolutionResult
from core.retrieval.dependency_resolver import resolve_dependencies, ResolvedDependencies
from core.retrieval.protocol_resolver import resolve_protocol_context, validate_protocol_context_completeness
from core.retrieval.token_budget import TokenBudgetManager
from core.retrieval.context_pack import ContextPack
from core.vectorstore.faiss_store import get_vector_store
from utils.logger import get_logger

logger = get_logger("retrieval.context_compiler")


def compile_task_context(
    query_or_intent: str | TaskIntent,
    efs_ir: EFSIR,
    all_protocol_rules: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 4000
) -> Tuple[ContextPack, TargetResolutionResult]:
    """
    Build a target-specific ContextPack for a user query.
    Enforces Target-Specific Protocol Retrieval and Protocol Context Completeness Check.
    
    Returns:
        (ContextPack, TargetResolutionResult)
    """
    # 1. Analyze query if passed as string
    if isinstance(query_or_intent, str):
        intent = analyze_query(query_or_intent)
    else:
        intent = query_or_intent

    all_protocol_rules = all_protocol_rules or []

    # 2. Resolve Target Component against canonical complete EFS IR
    res_result = resolve_target(intent.target, efs_ir)
    if not res_result.is_success():
        logger.warning(f"Target resolution failed: {res_result.error_message}")
        pack = ContextPack(
            task=intent,
            conflicts=[{"type": "TARGET_RESOLUTION_FAILED", "description": res_result.error_message}]
        )
        return pack, res_result

    target_comp = res_result.target_component

    # 3. Structural & Graph Dependency Resolution
    dep_res = resolve_dependencies(target_comp, efs_ir, max_depth=1)

    # 4. Target-Specific Protocol Context Slicing (Pass 1)
    relevant_proto_rules, proto_evidence, proto_chunks = resolve_protocol_context(
        dep_res,
        all_protocol_rules,
        requested_interface=intent.interface
    )

    # 5. FAISS Vector Search for User Source Evidence (user specification doc)
    user_source_evidence: List[Dict[str, Any]] = []
    try:
        vstore = get_vector_store()
        search_query = f"{target_comp.name} {intent.task} interface behavior requirements"
        vector_hits = vstore.search(search_query, top_k=5)
        for hit in vector_hits:
            meta = hit.metadata or {}
            user_source_evidence.append({
                "document_id": meta.get("doc_id", "user_spec"),
                "page": meta.get("page", 1),
                "section": meta.get("section", ""),
                "chunk_id": hit.chunk_id,
                "text": hit.text,
                "original_text": hit.text,
                "score": hit.score
            })
    except Exception as e:
        logger.warning(f"Vector search for user source evidence failed: {e}")

    # 6. Extract Functional Requirements for target from EFS IR constraints / specs
    func_reqs = []
    for const in efs_ir.constraints:
        if (target_comp.name.lower() in const.expected_behavior.lower() or
            any(target_comp.name.lower() in str(so).lower() for so in const.source_objects)):
            func_reqs.append(const.to_dict())

    # Extract timing rules
    timing_rules = [t.to_dict() for t in efs_ir.timing_rules]

    # Extract FSM dictionary for target
    fsm_dict = {}
    if dep_res.fsms:
        fsm_dict = dep_res.fsms[0].to_dict()

    # Extract conflicts
    conflicts_dict = [c.to_dict() for c in efs_ir.conflicts if c.resolution_status == "UNRESOLVED"]

    # Calculate full counts vs selected counts
    total_full_efs_objects = (
        len(efs_ir.components) + len(efs_ir.signals) + len(efs_ir.interfaces) +
        len(efs_ir.registers) + len(efs_ir.fsms) + len(efs_ir.transactions)
    )
    selected_efs_objects = (
        1 + len(dep_res.direct_submodules) + len(dep_res.signals) +
        len(dep_res.interfaces) + len(dep_res.registers) + len(dep_res.fsms)
    )

    pack = ContextPack(
        task=intent,
        target_component=target_comp.to_dict(),
        interfaces=[i.to_dict() for i in dep_res.interfaces],
        signals=[s.to_dict() for s in dep_res.signals],
        registers=[r.to_dict() for r in dep_res.registers],
        dependencies=[d.to_dict() for d in dep_res.direct_submodules],
        fsm=fsm_dict,
        transactions=[t.to_dict() for t in dep_res.transactions],
        functional_requirements=func_reqs,
        timing_rules=timing_rules,
        protocol_rules=relevant_proto_rules,
        constraints=[c.to_dict() for c in efs_ir.constraints],
        source_evidence=user_source_evidence,
        protocol_source_evidence=proto_evidence,
        relevant_protocol_chunks=proto_chunks,
        conflicts=conflicts_dict,
        total_full_efs_objects=total_full_efs_objects,
        selected_efs_objects=selected_efs_objects,
        total_source_chunks=len(user_source_evidence) + len(proto_chunks),
        selected_source_chunks=len(user_source_evidence) + len(proto_chunks),
        total_protocol_rules=len(all_protocol_rules),
        selected_protocol_rules=len(relevant_proto_rules),
    )

    # 7. Protocol Context Completeness Check & Mandatory Re-retrieval Pass
    is_complete, missing_semantics = validate_protocol_context_completeness(pack, dep_res)
    if not is_complete:
        logger.info(f"Protocol Context Completeness Check failed missing: {missing_semantics}. Executing expanded retrieval pass...")
        exp_rules, exp_evidence, exp_chunks = resolve_protocol_context(
            dep_res,
            all_protocol_rules,
            requested_interface=intent.interface,
            expansion_query_additions=missing_semantics
        )
        pack.protocol_rules = exp_rules
        pack.protocol_source_evidence = exp_evidence
        pack.relevant_protocol_chunks = exp_chunks
        pack.selected_protocol_rules = len(exp_rules)
        
        # Re-run completeness validation
        is_complete, _ = validate_protocol_context_completeness(pack, dep_res)
        logger.info(f"Expanded retrieval pass completed. Completeness status: {is_complete}")

    # 8. Token Budgeting & Pruning (Guarantees MINIMUM COMPLETE CONTEXT > MINIMUM TOKEN COUNT)
    budget_mgr = TokenBudgetManager(max_tokens=max_tokens)
    pack = budget_mgr.enforce_budget(pack)

    logger.info(
        f"Context Compiler successfully created ContextPack for target '{target_comp.name}' "
        f"({pack.selected_efs_objects}/{pack.total_full_efs_objects} EFS objects, "
        f"{pack.selected_protocol_rules}/{pack.total_protocol_rules} protocol rules, "
        f"{len(pack.relevant_protocol_chunks)} protocol chunks, "
        f"{pack.estimated_tokens} estimated tokens)."
    )

    return pack, res_result
