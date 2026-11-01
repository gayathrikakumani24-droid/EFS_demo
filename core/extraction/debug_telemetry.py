"""
Debug Telemetry for EFS Extraction Pipeline Tracing.

Emits structured telemetry for:
- [REGISTER TRACE]
- [CONSTRAINT TRACE]
Zero hardcoding; protocol and document agnostic.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from utils.logger import get_logger

logger = get_logger("extraction.telemetry")


def emit_debug_telemetry(
    doc_ir: Optional[Any],
    req_ir: Optional[Any],
    efs_ir: Optional[Any],
    validator_conflicts: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """
    Compute and log structured debug telemetry for register and constraint extraction.
    """
    blocks = getattr(doc_ir, "blocks", []) if doc_ir else []
    source_chunks_retrieved = len(blocks)

    # 1. Register Source Search
    reg_chunks = [
        b for b in blocks
        if any(k in (b.text or "").lower() for k in ("register", "registers", "reg_name", "reg file", "csr", "m0-m7"))
    ]
    relevant_chunks_containing_register_terms = len(reg_chunks)

    # 2. Constraint Source Search
    constraint_chunks = [
        b for b in blocks
        if any(k in (b.text or "").lower() for k in ("frequency", "ghz", "mhz", "throughput", "power", "latency", "timing", "constraint", "mw", "clock"))
    ]
    relevant_chunks_containing_constraint_terms = len(constraint_chunks)

    # Candidates from Req IR
    candidates = getattr(req_ir, "candidates", []) if req_ir else []
    reg_candidates = [
        c.to_dict() if hasattr(c, "to_dict") else str(c)
        for c in candidates
        if getattr(c, "candidate_type", "").upper() == "REGISTER"
    ]
    constraint_candidates = [
        c.to_dict() if hasattr(c, "to_dict") else str(c)
        for c in candidates
        if getattr(c, "candidate_type", "").upper() in ("PERFORMANCE_REQUIREMENT", "TIMING", "CONSTRAINT")
    ]

    # EFS IR items
    registers = getattr(efs_ir, "registers", []) if efs_ir else []
    efs_ir_register_count = len(registers)
    efs_ir_registers = [
        {
            "name": r.name,
            "offset": getattr(r, "offset", None),
            "access": getattr(r, "access", getattr(r, "access_type", "RW")),
            "description": getattr(r, "description", "")
        }
        for r in registers
    ]

    timing_rules = getattr(efs_ir, "timing_rules", []) if efs_ir else []
    constraints = getattr(efs_ir, "constraints", []) if efs_ir else []
    perf_reqs = getattr(efs_ir, "performance_requirements", []) if efs_ir else []

    efs_ir_timing_rule_count = len(timing_rules)
    efs_ir_constraint_count = len(constraints)
    efs_ir_performance_requirement_count = len(perf_reqs)

    # Validator results
    conflicts = validator_conflicts or (getattr(efs_ir, "conflicts", []) if efs_ir else [])
    reg_conflicts = [
        c for c in conflicts
        if getattr(c, "property", "") in ("registers", "register") or "register" in getattr(c, "title", "").lower()
    ]
    constraint_conflicts = [
        c for c in conflicts
        if getattr(c, "property", "") in ("constraints", "timing", "clock") or "constraint" in getattr(c, "title", "").lower() or "timing" in getattr(c, "title", "").lower()
    ]

    validator_register_input = f"{efs_ir_register_count} register(s) in EFS IR"
    validator_register_result = f"PASSED ({efs_ir_register_count} registers)" if not reg_conflicts else f"FAILED: {[c.title for c in reg_conflicts]}"

    validator_constraint_input = f"{efs_ir_timing_rule_count} timing rule(s), {efs_ir_constraint_count} constraint(s), {efs_ir_performance_requirement_count} performance req(s)"
    validator_constraint_result = f"PASSED ({efs_ir_performance_requirement_count + efs_ir_timing_rule_count + efs_ir_constraint_count} items)" if not constraint_conflicts else f"FAILED: {[c.title for c in constraint_conflicts]}"

    # Format telemetry strings
    reg_trace_lines = [
        "[REGISTER TRACE]",
        f"source_chunks_retrieved = {source_chunks_retrieved}",
        f"relevant_chunks_containing_register_terms = {relevant_chunks_containing_register_terms}",
        f"raw_llm_register_output = {json.dumps(reg_candidates[:5]) if reg_candidates else '[]'}",
        f"normalized_register_output = {json.dumps(reg_candidates[:5]) if reg_candidates else '[]'}",
        f"efs_ir_register_count = {efs_ir_register_count}",
        f"efs_ir_registers = {json.dumps(efs_ir_registers)}",
        f"validator_register_input = {validator_register_input}",
        f"validator_register_result = {validator_register_result}"
    ]

    constraint_trace_lines = [
        "[CONSTRAINT TRACE]",
        f"source_chunks_retrieved = {source_chunks_retrieved}",
        f"relevant_chunks_containing_constraint_terms = {relevant_chunks_containing_constraint_terms}",
        f"raw_llm_constraint_output = {json.dumps(constraint_candidates[:5]) if constraint_candidates else '[]'}",
        f"normalized_constraint_output = {json.dumps(constraint_candidates[:5]) if constraint_candidates else '[]'}",
        f"efs_ir_timing_rule_count = {efs_ir_timing_rule_count}",
        f"efs_ir_constraint_count = {efs_ir_constraint_count}",
        f"efs_ir_performance_requirement_count = {efs_ir_performance_requirement_count}",
        f"validator_constraint_input = {validator_constraint_input}",
        f"validator_constraint_result = {validator_constraint_result}"
    ]

    reg_trace_str = "\n".join(reg_trace_lines)
    constraint_trace_str = "\n".join(constraint_trace_lines)

    logger.info("\n" + reg_trace_str + "\n\n" + constraint_trace_str)
    print("\n" + reg_trace_str + "\n\n" + constraint_trace_str)

    return {
        "register_trace": {
            "source_chunks_retrieved": source_chunks_retrieved,
            "relevant_chunks_containing_register_terms": relevant_chunks_containing_register_terms,
            "efs_ir_register_count": efs_ir_register_count,
            "efs_ir_registers": efs_ir_registers,
            "validator_register_input": validator_register_input,
            "validator_register_result": validator_register_result,
        },
        "constraint_trace": {
            "source_chunks_retrieved": source_chunks_retrieved,
            "relevant_chunks_containing_constraint_terms": relevant_chunks_containing_constraint_terms,
            "efs_ir_timing_rule_count": efs_ir_timing_rule_count,
            "efs_ir_constraint_count": efs_ir_constraint_count,
            "efs_ir_performance_requirement_count": efs_ir_performance_requirement_count,
            "validator_constraint_input": validator_constraint_input,
            "validator_constraint_result": validator_constraint_result,
        }
    }
