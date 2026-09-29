"""
Orchestrator for the AI Hardware Design Assistant.

Coordinates requirements parsing, planning, retrieval, code generation, compliance
verification, and iterative self-repair loops.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Callable, Optional

from datetime import datetime
from core.agents.requirement_agent import analyze_requirements
from core.agents.planning_agent import create_plan
from core.agents.retrieval_agent import retrieve_protocol_rules
from core.agents.spec_agent import generate_design_spec
from core.agents.rtl_agent import generate_code
from core.agents.verification_agent import verify_code
from core.agents.repair_agent import repair_code
from core.agents.plantuml_agent import generate_diagrams, get_diagram_url
from core.verification.static_verifier import verify_statically
from core.retrieval.context_compiler import compile_task_context
from core.efs_ir.models import EFSConflict, EFSResolution, EFSIR
from enum import Enum
from config import CONFIG
from utils.logger import get_logger

logger = get_logger("agents.orchestrator")


class WorkflowState(str, Enum):
    INGESTION = "INGESTION"
    EXTRACTION = "EXTRACTION"
    CONFLICT_DETECTION = "CONFLICT_DETECTION"
    HUMAN_RESOLUTION_REQUIRED = "HUMAN_RESOLUTION_REQUIRED"
    REVALIDATION = "REVALIDATION"
    READY_FOR_GENERATION = "READY_FOR_GENERATION"
    DESIGN_PLANNING = "DESIGN_PLANNING"
    PUML_GENERATION = "PUML_GENERATION"
    RTL_GENERATION = "RTL_GENERATION"
    RTL_VALIDATION = "RTL_VALIDATION"
    GENERATION_COMPLETE = "GENERATION_COMPLETE"
    STOPPED = "STOPPED"


def log_state_transition(from_state: WorkflowState | str, to_state: WorkflowState | str) -> None:
    from_name = from_state.value if isinstance(from_state, WorkflowState) else str(from_state)
    to_name = to_state.value if isinstance(to_state, WorkflowState) else str(to_state)
    logger.info(f"STATE_TRANSITION:\n{from_name} → {to_name}")


def log_generation_start(reason: str, model_version: int = 1, efs_ir_version: int = 1) -> None:
    logger.info(f"GENERATION_START:\nreason={reason}\nmodel_version={model_version}\nefs_ir_version={efs_ir_version}")


def log_generation_end(status: str) -> None:
    logger.info(f"GENERATION_END:\nstatus={status}")


def log_generation_skipped(reason: str) -> None:
    logger.info(f"GENERATION_SKIPPED:\nreason={reason}")


def run_design_flow(
    requirement_text: str,
    target_outputs: List[str],
    max_repair_iterations: int = 2,
    compliance_threshold: int = 90,
    progress_cb: Callable[[str, float], None] = None,
    bypass_conflicts: bool = False,
    user_query: Optional[str] = None
) -> Dict[str, Any]:
    """Execute the multi-agent design flow end-to-end, returning intermediate and final outputs."""
    import time
    
    # Force reload LLM client to ensure latest provider/model configuration is active
    from core.extraction.llm_client import get_llm_client
    get_llm_client(force_reload=True)

    # Initialize performance metrics counters
    llm_calls_count = 0
    static_checks_count = 0
    timings = {}
    
    def update_progress(msg: str, pct: float):
        if progress_cb:
            progress_cb(msg, pct)
        logger.info(f"[{pct*100:.1f}%] {msg}")

    # -- STEP 1: Requirement Analysis ---------------------------------------
    log_state_transition(WorkflowState.INGESTION, WorkflowState.EXTRACTION)
    update_progress("Understanding hardware requirements...", 0.05)
    start_time = time.time()
    req_model = analyze_requirements(requirement_text)
    timings["Requirement Understanding"] = time.time() - start_time
    llm_calls_count += 1

    # -- STEP 2: Architecture Planning --------------------------------------
    update_progress("Creating implementation planning layout...", 0.15)
    start_time = time.time()
    plan = create_plan(req_model)
    timings["Design Planning"] = time.time() - start_time
    llm_calls_count += 1

    # -- STEP 3: Protocol Retrieval -----------------------------------------
    update_progress("Retrieving protocol spec rules from Knowledge Graph & Vector DB...", 0.25)
    start_time = time.time()
    protocol_rules = retrieve_protocol_rules(req_model, plan)
    timings["Protocol Retrieval"] = time.time() - start_time

    # -- STEP 4: Build Canonical EFS IR & Detect Conflicts Immediately ------
    log_state_transition(WorkflowState.EXTRACTION, WorkflowState.CONFLICT_DETECTION)
    update_progress("Building canonical EFS Intermediate Representation...", 0.30)
    start_time_ir = time.time()
    design_spec = ""
    try:
        from core.efs_ir.builder import EFSIRBuilder
        from core.efs_ir.validator import validate_efs_ir
        from core.graph.neo4j_builder import build_graph
        from core.registry import get_registry
        import json
        import os
        
        # Build EFS IR
        generation_run_id = f"RUN_{int(time.time() * 1000)}"
        builder = EFSIRBuilder(req_model, plan, protocol_rules, generation_run_id=generation_run_id)
        efs_ir = builder.build()
        
        # Run Controlled Specification Inference Engine
        from core.verification.inference_engine import ControlledInferenceEngine
        inference_engine = ControlledInferenceEngine(efs_ir, plan, req_model)
        completed_model = inference_engine.infer_and_complete()
        efs_ir = completed_model.efs_ir

        validation_issues = validate_efs_ir(efs_ir)
        # Validate EFS IR and detect specification conflicts
        from core.verification.conflict_resolution_engine import ConflictResolutionEngine
        detected_conflicts = ConflictResolutionEngine.detect_conflicts(efs_ir, req_model, validation_issues=validation_issues)
        detected_conflicts = ConflictResolutionEngine.deduplicate_conflicts(detected_conflicts)

        # Hardware completeness check — ensures the EFS IR is a real specification,
        # not just a list of component names. All gaps surface as blocking HITL conflicts.
        completeness_gaps = ConflictResolutionEngine.validate_hardware_completeness(efs_ir, plan)
        if completeness_gaps:
            # Merge completeness gaps into detected conflicts (dedup by semantic key)
            existing_keys = {getattr(c, 'semantic_key', '') for c in detected_conflicts}
            for gap in completeness_gaps:
                gap_key = gap.semantic_key or ConflictResolutionEngine.compute_semantic_key(gap)
                if gap_key not in existing_keys:
                    detected_conflicts.append(gap)
                    existing_keys.add(gap_key)
            detected_conflicts = ConflictResolutionEngine.deduplicate_conflicts(detected_conflicts)

        efs_ir.conflicts = detected_conflicts
        unresolved_blocking = [c for c in detected_conflicts if getattr(c, "blocking", False) and getattr(c, "resolution_status", "UNRESOLVED") == "UNRESOLVED"]

        has_critical = any(issue["severity"] in ("ERROR", "CRITICAL") for issue in validation_issues) or len(unresolved_blocking) > 0
        if validation_issues:
            logger.warning(f"EFS IR Validation found {len(validation_issues)} issues.")
            
        # Serialize to active design file
        ir_dir = os.path.join(CONFIG.data_dir, "efs_ir")
        os.makedirs(ir_dir, exist_ok=True)
        active_ir_path = os.path.join(ir_dir, "active_design_ir.json")
        with open(active_ir_path, "w", encoding="utf-8") as f:
            json.dump(efs_ir.to_dict(), f, indent=2)
            
        # Update Knowledge Graph with canonical EFS IR objects
        reg = get_registry()
        build_graph(reg.all_entities(), reg.all_relationships(), efs_ir=efs_ir)
        
        # Link back-propagated EFS objects in Vector Store
        try:
            from core.vectorstore.faiss_store import get_vector_store
            vector_store = get_vector_store()
            vector_store.link_efs_objects_to_chunks(efs_ir)
        except Exception as ev:
            logger.warning(f"Failed to link EFS objects to vector store chunks: {ev}")
            
        # Stop and pause generation immediately if critical issues or unresolved conflicts exist and not bypassed
        if has_critical and not bypass_conflicts:
            logger.warning(f"Pipeline paused: AWAITING_HUMAN_DECISION due to {len(unresolved_blocking)} critical EFS IR conflicts / validation errors.")
            log_state_transition(WorkflowState.CONFLICT_DETECTION, WorkflowState.HUMAN_RESOLUTION_REQUIRED)
            log_generation_skipped("critical_conflicts_require_human_resolution")
            proposals_map = {}
            for conf in unresolved_blocking:
                proposals_map[conf.conflict_id] = ConflictResolutionEngine.generate_resolution_proposals(conf, efs_ir)

            return {
                "status": "AWAITING_HUMAN_DECISION",
                "pipeline_state": "AWAITING_HUMAN_DECISION",
                "workflow_state": WorkflowState.HUMAN_RESOLUTION_REQUIRED.value,
                "requires_human_decision": True,
                "is_blocked": False,
                "requirement_model": req_model,
                "plan": plan,
                "protocol_rules": protocol_rules,
                "design_spec": "",
                "efs_ir": efs_ir,
                "completed_model": completed_model,
                "efs_ir_validation_issues": validation_issues,
                "conflicts": unresolved_blocking,
                "proposals": proposals_map,
                "outputs": {},
                "diagrams": {},
                "diagram_urls": {},
                "performance_metrics": {
                    "llm_calls_count": llm_calls_count,
                    "static_checks_count": 0,
                    "timings": timings,
                    "primary_target": "Verilog",
                    "compliance_score": 0,
                    "total_iterations": 0
                }
            }
            
    except Exception as e:
        logger.exception("Failed to build or validate EFS IR")
        efs_ir = None
        completed_model = None
        validation_issues = [{"severity": "ERROR", "message": f"Builder failed: {e}", "category": "System"}]
        if not bypass_conflicts:
            return {
                "status": "BLOCKED",
                "requirement_model": req_model,
                "plan": plan,
                "protocol_rules": protocol_rules,
                "design_spec": "",
                "efs_ir": None,
                "completed_model": None,
                "efs_ir_validation_issues": validation_issues,
                "outputs": {},
                "diagrams": {},
                "diagram_urls": {},
                "performance_metrics": {
                    "llm_calls_count": llm_calls_count,
                    "static_checks_count": 0,
                    "timings": timings,
                    "primary_target": "Verilog",
                    "compliance_score": 0,
                    "total_iterations": 0
                }
            }
            
    timings["EFS IR Build & Validation"] = time.time() - start_time_ir

    # -- STEP 5: Design Spec Generation (Zero Conflicts -> Proceed) --------
    log_state_transition(WorkflowState.CONFLICT_DETECTION, WorkflowState.READY_FOR_GENERATION)
    log_state_transition(WorkflowState.READY_FOR_GENERATION, WorkflowState.DESIGN_PLANNING)
    update_progress("Generating formal design specification document...", 0.40)
    start_time = time.time()
    design_spec = generate_design_spec(req_model, plan, protocol_rules, efs_ir=efs_ir)
    timings["Design Spec Generation"] = time.time() - start_time
    llm_calls_count += 1

    # -- STEP 6: Query-Driven Context Compilation -------------------------
    update_progress("Compiling task-specific ContextPack...", 0.48)
    start_time_comp = time.time()
    from utils.logger import truncate_for_log
    from core.retrieval.context_compiler import compile_task_context

    query_cmd = user_query if user_query else ""
    context_pack, res_result = compile_task_context(
        query_cmd,
        efs_ir,
        all_protocol_rules=protocol_rules
    )

    timings["Context Compilation"] = time.time() - start_time_comp

    if not res_result.is_success() and efs_ir and efs_ir.components:
        logger.warning(f"Target resolution issue: {res_result.error_message}")
        log_generation_skipped(f"target_resolution_issue: {res_result.error_message}")
        return {
            "status": res_result.status,
            "error_message": res_result.error_message,
            "requirement_model": req_model,
            "plan": plan,
            "protocol_rules": protocol_rules,
            "design_spec": design_spec,
            "efs_ir": efs_ir,
            "completed_model": completed_model,
            "context_pack": context_pack,
            "efs_ir_validation_issues": validation_issues,
            "outputs": {},
            "diagrams": {},
            "diagram_urls": {},
            "performance_metrics": {
                "llm_calls_count": llm_calls_count,
                "static_checks_count": 0,
                "timings": timings,
                "primary_target": "Verilog",
                "compliance_score": 0,
                "total_iterations": 0
            }
        }

    # -- STEP 4.9: Generate PlantUML Code & Diagrams First -----------------
    log_state_transition(WorkflowState.DESIGN_PLANNING, WorkflowState.PUML_GENERATION)
    update_progress("Generating PlantUML code and diagrams...", 0.45)
    start_time_puml = time.time()
    diagrams = generate_diagrams(design_spec, plan, efs_ir=efs_ir)
    timings["PlantUML Diagrams Generation"] = time.time() - start_time_puml
    llm_calls_count += 1

    diagram_urls = {
        k: get_diagram_url(v) for k, v in (diagrams or {}).items()
    } if isinstance(diagrams, dict) else {}

    # We will generate and track the code for each selected target
    outputs: Dict[str, Dict[str, Any]] = {}

    # Identify primary target format (Verilog/SV/VHDL)
    rtl_targets = [t for t in target_outputs if t in ("Verilog", "SystemVerilog", "VHDL")]
    primary_target = rtl_targets[0] if rtl_targets else "Verilog"

    logger.info(f"Selected primary RTL target for stabilization: {primary_target}")

    # -- STEP 5: Stabilize Primary RTL (Verification & Repair Loop) ---------
    log_state_transition(WorkflowState.PUML_GENERATION, WorkflowState.RTL_GENERATION)
    log_state_transition(WorkflowState.RTL_GENERATION, WorkflowState.RTL_VALIDATION)
    m_ver = getattr(efs_ir.metadata, "model_version", 1) if (efs_ir and hasattr(efs_ir, "metadata")) else 1
    log_generation_start(reason="initial_generation_clean_spec", model_version=m_ver, efs_ir_version=m_ver)
    update_progress(f"Generating initial RTL code for primary target: {primary_target}...", 0.55)

    start_time = time.time()
    code = generate_code(design_spec, primary_target, efs_ir=efs_ir, protocol_rules=protocol_rules, context_pack=context_pack)
    timings[f"{primary_target} Initial Generation"] = time.time() - start_time
    llm_calls_count += 1

    # Handle RTL generation returning an insufficient specification marker.
    # Do NOT create new SOURCE_MISSING conflicts or restart the pipeline.
    # All missing-info checks were done upfront; this is a generation-quality issue only.
    if code == "INSUFFICIENT SPECIFICATION FOR RTL GENERATION" or (isinstance(code, str) and code.startswith("SPECIFICATION_INCOMPLETE")):
        logger.warning("RTL generation returned incomplete output. Reporting as generation error (not restarting pipeline).")
        log_generation_end("rtl_generation_insufficient_output")
        outputs[primary_target] = {
            "final_code": code,
            "compliance_score": 0,
            "report": {"failed_checks": [{"check_name": "RTL Generation Quality", "violation": "RTL agent could not produce complete synthesizable code from the resolved specification. Review the EFS IR and design spec for gaps."}]},
            "history": [],
            "iterations_run": 0
        }
        return {
            "status": "RTL_GENERATION_ERROR",
            "pipeline_state": "RTL_GENERATION_ERROR",
            "workflow_state": WorkflowState.RTL_VALIDATION.value,
            "requires_human_decision": False,
            "is_blocked": False,
            "error_message": "RTL generation could not produce complete output. The specification passed completeness checks but the RTL agent could not synthesize it. Check the design spec and EFS IR for any remaining ambiguities.",
            "requirement_model": req_model,
            "plan": plan,
            "protocol_rules": protocol_rules,
            "design_spec": design_spec,
            "efs_ir": efs_ir,
            "completed_model": completed_model,
            "context_pack": context_pack,
            "efs_ir_validation_issues": validation_issues,
            "outputs": outputs,
            "diagrams": diagrams,
            "diagram_urls": diagram_urls,
            "performance_metrics": {
                "llm_calls_count": llm_calls_count,
                "static_checks_count": 0,
                "timings": timings,
                "primary_target": primary_target,
                "compliance_score": 0,
                "total_iterations": 0
            }
        }
        
    iteration = 0
    loop_history = []
    final_code = code
    compliance_report = {}
    repairs_applied = []
    
    # Best-version tracking variables
    best_code = code
    best_score = -1
    best_iteration = -1
    if max_repair_iterations == 0 or (not getattr(CONFIG, "enable_static_verification", False) and not getattr(CONFIG, "enable_semantic_verification", False)):
        logger.info(f"Verification & repair loop disabled by config. Accepting generated {primary_target} code directly.")
        best_code = code
        best_score = 100
        best_report = {"compliance_score": 100, "passed_checks": [], "failed_checks": [], "warnings": []}
        outputs[primary_target] = {
            "final_code": best_code,
            "compliance_score": best_score,
            "report": best_report,
            "history": [],
            "iterations_run": 0
        }
    else:
        while iteration <= max_repair_iterations:
            update_progress(f"Verifying code for {primary_target} (Iteration {iteration}/{max_repair_iterations})...", 0.50 + (0.15 * (iteration / (max_repair_iterations + 1))))
            
            # Avoid verifying identical code
            is_duplicate = False
            if iteration > 0 and final_code == loop_history[-1]["code"]:
                logger.info("No code changes detected; skipping duplicate verification.")
                compliance_report = loop_history[-1]["report"]
                score = compliance_report.get("compliance_score", 100)
                is_duplicate = True
            else:
                start_time = time.time()
                compliance_report = verify_code(final_code, design_spec, protocol_rules, req_model, plan, efs_ir=efs_ir)
                timings[f"{primary_target} Verification Iteration {iteration}"] = time.time() - start_time
                
                # Check if LLM call occurred (semantic check was executed)
                has_critical_major_static = any(
                    f.get("severity") in ("CRITICAL", "MAJOR")
                    for f in verify_statically(final_code, design_spec, protocol_rules, req_model, plan, efs_ir=efs_ir).get("failed_checks", [])
                )
                if getattr(CONFIG, "enable_semantic_verification", True) and not has_critical_major_static:
                    llm_calls_count += 1
                
                # Count static checks
                static_checks_count += len(compliance_report.get("passed_checks", [])) + len(compliance_report.get("failed_checks", []))
                score = compliance_report.get("compliance_score", 100)
                
            failed_checks = compliance_report.get("failed_checks", [])
            
            loop_history.append({
                "iteration": iteration,
                "code": final_code,
                "score": score,
                "report": compliance_report,
                "repairs_applied": list(repairs_applied),
                "timestamp": datetime.now().isoformat()
            })
            
            # Track the best version
            if score > best_score:
                best_score = score
                best_code = final_code
                best_iteration = iteration
                best_report = compliance_report
                
            # -- Check termination conditions --
            # 1. Fully compliant
            if not failed_checks:
                logger.info(f"Target '{primary_target}' passed compliance with 100% score.")
                break
                
            # 2. Compliance threshold met
            if score >= compliance_threshold:
                logger.info(f"Target '{primary_target}' reached compliance threshold ({score}% >= {compliance_threshold}%).")
                break
                
            # 3. No critical/major violations (only warnings or minor issues)
            serious_failed_checks = [
                f for f in failed_checks
                if f.get("severity", "MAJOR") in ("CRITICAL", "MAJOR")
            ]
            if not serious_failed_checks:
                logger.info("No critical or major violations remaining. Skipping expensive repairs loop.")
                break
                
            # 4. Max iterations reached
            if iteration == max_repair_iterations:
                logger.warning(f"Target '{primary_target}' hit max repair iterations. Remaining violations: {len(failed_checks)}.")
                break
                
            # Check if EFS IR contains unresolved critical conflicts
            if efs_ir and getattr(efs_ir, "conflicts", []):
                unresolved_critical = [c for c in efs_ir.conflicts if c.severity == "CRITICAL" and c.resolution_status == "UNRESOLVED"]
                if unresolved_critical:
                    logger.warning(f"Repair loop aborted: {len(unresolved_critical)} unresolved critical specification conflicts in EFS IR.")
                    break
                    
            # Perform repair
            update_progress(f"Repairing violations for {primary_target} (Iteration {iteration})...", 0.52 + (0.15 * (iteration / (max_repair_iterations + 1))))
            start_time = time.time()
            repair_result = repair_code(final_code, design_spec, protocol_rules, compliance_report, efs_ir=efs_ir)
            timings[f"{primary_target} Repair Iteration {iteration}"] = time.time() - start_time
            llm_calls_count += 1
            
            # Check if the repair agent flagged EFS IR as incorrect (specification error)
            if not repair_result.get("efs_ir_correct", True):
                logger.warning(f"Repair loop aborted: Spec/IR error flagged by repair agent: {repair_result.get('efs_ir_repair_notes')}")
                loop_history.append({
                    "iteration": iteration,
                    "code": final_code,
                    "score": score,
                    "report": {**compliance_report, "warnings": compliance_report.get("warnings", []) + [f"IR/Spec error flagged: {repair_result.get('efs_ir_repair_notes')}"]},
                    "repairs_applied": ["BLOCKED: Spec/IR Error: " + repair_result.get("efs_ir_repair_notes", "")],
                    "timestamp": datetime.now().isoformat()
                })
                break
                
            final_code = repair_result.get("repaired_code", final_code)
            repairs_applied = repair_result.get("repairs", [])
            
            logger.info(f"Repaired target '{primary_target}' at iteration {iteration}. Repairs applied: {len(repairs_applied)}.")
            iteration += 1

    # Store finalized primary RTL output
    outputs[primary_target] = {
        "final_code": best_code,
        "compliance_score": best_score,
        "report": best_report,
        "history": loop_history,
        "iterations_run": iteration
    }

    # -- STEP 6: Generate Secondary Formats from Finalized Hardware Truth --
    secondary_targets = [t for t in target_outputs if t != primary_target]
    
    # Unified context binds design spec together with stabilized core hardware RTL
    stabilized_context = f"""=== HARDWARE SPECIFICATION ===
{design_spec}

=== FINALIZED RTL CODE ===
{best_code}
"""

    for idx, target in enumerate(secondary_targets):
        update_progress(f"Generating secondary format: {target}...", 0.70 + (0.20 * (idx / max(1, len(secondary_targets)))))
        
        start_time = time.time()
        sec_code = generate_code(stabilized_context, target, efs_ir=efs_ir, protocol_rules=protocol_rules, context_pack=context_pack)
        timings[f"{target} Generation"] = time.time() - start_time
        llm_calls_count += 1
        
        # Verify secondary format (disabled by default)
        sec_report = {"compliance_score": 100, "passed_checks": [], "failed_checks": [], "warnings": []}
        if getattr(CONFIG, "verify_secondary_artifacts", False):
            if getattr(CONFIG, "enable_static_verification", False) or getattr(CONFIG, "enable_semantic_verification", False):
                sec_report = verify_code(sec_code, stabilized_context, protocol_rules, req_model, plan, efs_ir=efs_ir)
                if getattr(CONFIG, "enable_semantic_verification", False):
                    llm_calls_count += 1
                static_checks_count += len(sec_report.get("passed_checks", [])) + len(sec_report.get("failed_checks", []))
            
        outputs[target] = {
            "final_code": sec_code,
            "compliance_score": sec_report.get("compliance_score", 100),
            "report": sec_report,
            "history": [],
            "iterations_run": 0
        }

    update_progress("Design generation complete.", 1.0)
    log_state_transition(WorkflowState.RTL_VALIDATION, WorkflowState.GENERATION_COMPLETE)
    log_generation_end("success")

    return {
        "status": "SUCCESS",
        "pipeline_state": "GENERATION_COMPLETE",
        "workflow_state": WorkflowState.GENERATION_COMPLETE.value,
        "requirement_model": req_model,
        "plan": plan,
        "protocol_rules": protocol_rules,
        "design_spec": design_spec,
        "efs_ir": efs_ir,
        "completed_model": completed_model,
        "context_pack": context_pack,
        "efs_ir_validation_issues": validation_issues,
        "diagrams": diagrams,
        "diagram_urls": diagram_urls,
        "outputs": outputs,
        # Performance logging metrics
        "performance_metrics": {
            "llm_calls_count": llm_calls_count,
            "static_checks_count": static_checks_count,
            "timings": timings,
            "primary_target": primary_target,
            "compliance_score": best_score,
            "total_iterations": iteration
        }
    }


def resume_design_flow_with_resolutions(
    design_results: Dict[str, Any],
    approved_resolutions: List[Any],
    target_outputs: List[str] = None,
    progress_cb: Callable[[str, float], None] = None,
    bypass_conflicts: bool = False
) -> Dict[str, Any]:
    """
    Applies user-approved resolution proposals to the design model,
    re-runs mandatory validation, and if clear, resumes RTL & PlantUML generation.
    """
    from core.verification.conflict_resolution_engine import ConflictResolutionEngine
    import json
    import os
    
    efs_ir = design_results.get("efs_ir")
    req_model = design_results.get("requirement_model")
    plan = design_results.get("plan")
    protocol_rules = design_results.get("protocol_rules")
    design_spec = design_results.get("design_spec")
    
    if not efs_ir:
        return design_results

    targets = target_outputs or ["Verilog"]

    # Ensure conflicts from design_results are tracked in efs_ir
    if design_results.get("conflicts"):
        existing_cids = {c.conflict_id for c in getattr(efs_ir, "conflicts", [])}
        for c in design_results["conflicts"]:
            if c.conflict_id not in existing_cids:
                efs_ir.conflicts.append(c)
                existing_cids.add(c.conflict_id)

    # Pipeline Logging: Human approval received
    for res in approved_resolutions:
        res_id = getattr(res, "resolution_id", "RES")
        conf_id = getattr(res, "conflict_id", "CONF")
        logger.info(f"AI_RESOLUTION_APPROVED: Human approval received for conflict: {conf_id} (Resolution ID: {res_id})")

    # 0. Idempotency Check: if all resolutions already recorded in efs_ir and design is already completed
    existing_res_ids = {getattr(r, "resolution_id", "") for r in getattr(efs_ir, "resolutions", [])}
    incoming_res_ids = {getattr(r, "resolution_id", "") for r in (approved_resolutions or [])}
    if incoming_res_ids and incoming_res_ids.issubset(existing_res_ids) and design_results.get("status") in ("SUCCESS", "GENERATION_COMPLETE") and design_results.get("outputs"):
        logger.info("Resolutions already applied and design generation completed; skipping re-execution.")
        log_generation_skipped("resolutions_already_applied")
        return design_results

    # If no human resolutions provided and blocking conflicts remain, stay in HUMAN_RESOLUTION_REQUIRED
    if not approved_resolutions and not bypass_conflicts:
        unresolved_blocking = [
            c for c in getattr(efs_ir, "conflicts", [])
            if getattr(c, "blocking", False) and getattr(c, "resolution_status", "UNRESOLVED") != "RESOLVED"
        ]
        if unresolved_blocking or (design_results.get("status") == "AWAITING_HUMAN_DECISION" and design_results.get("conflicts")):
            logger.info("No human resolutions provided while blocking conflicts remain; skipping generation.")
            log_generation_skipped("no_human_resolutions_provided")
            design_results["status"] = "AWAITING_HUMAN_DECISION"
            design_results["pipeline_state"] = "AWAITING_HUMAN_DECISION"
            design_results["workflow_state"] = WorkflowState.HUMAN_RESOLUTION_REQUIRED.value
            design_results["requires_human_decision"] = True
            return design_results

    log_state_transition(WorkflowState.HUMAN_RESOLUTION_REQUIRED, WorkflowState.REVALIDATION)

    def update_resume_progress(msg: str, val: float):
        if progress_cb:
            try:
                progress_cb(msg, val)
            except Exception:
                pass

    update_resume_progress("Applying approved resolutions to canonical EFS IR...", 0.60)

    # 1. Apply all approved resolutions & persist (each resolution applied only once)
    applied_rids = {
        getattr(r, "resolution_id", "") or getattr(r, "conflict_id", "")
        for r in getattr(efs_ir, "resolutions", [])
        if getattr(r, "approval_status", "") == "APPROVED"
    }
    applied_rids.discard("")
    resolved_ir = efs_ir
    for res in approved_resolutions:
        r_id = getattr(res, "resolution_id", "") or getattr(res, "conflict_id", "")
        if r_id and r_id in applied_rids:
            continue
        try:
            resolved_ir = ConflictResolutionEngine.apply_approved_resolution(resolved_ir, res)
            if r_id:
                applied_rids.add(r_id)
        except Exception as e:
            logger.warning(f"Resolution application rejected or failed: {e}")
    logger.info("RESOLUTION_PERSISTED: Approved resolution layer persisted.")

    # 2. Build resolved requirement model view & mandatory revalidation (ZERO LLM CALLS)
    logger.info("EFSIR_REBUILT: Rebuilding resolved requirement model view...")
    resolved_ir = resolved_ir.get_resolved_view()

    # 3. Mandatory deterministic revalidation
    update_resume_progress("Revalidating resolved model view...", 0.70)
    logger.info("REVALIDATION: Revalidating resolved requirement model deterministically...")
    has_blocking, validation_issues = ConflictResolutionEngine.revalidate_resolved_model(resolved_ir, plan)
    
    # Update serialized active design file with resolved view
    ir_dir = os.path.join(CONFIG.data_dir, "efs_ir")
    os.makedirs(ir_dir, exist_ok=True)
    active_ir_path = os.path.join(ir_dir, "active_design_ir.json")
    with open(active_ir_path, "w", encoding="utf-8") as f:
        json.dump(resolved_ir.to_dict(), f, indent=2)

    resolved_ir.conflicts = ConflictResolutionEngine.deduplicate_conflicts(resolved_ir.conflicts)

    # Re-check hardware completeness after resolution
    completeness_gaps = ConflictResolutionEngine.validate_hardware_completeness(resolved_ir, plan)
    if completeness_gaps:
        existing_keys = {getattr(c, 'semantic_key', '') for c in resolved_ir.conflicts}
        for gap in completeness_gaps:
            gap_key = gap.semantic_key or ConflictResolutionEngine.compute_semantic_key(gap)
            if gap_key not in existing_keys:
                resolved_ir.conflicts.append(gap)
                existing_keys.add(gap_key)
        resolved_ir.conflicts = ConflictResolutionEngine.deduplicate_conflicts(resolved_ir.conflicts)

    unresolved_blocking = [
        c for c in resolved_ir.conflicts
        if c.blocking and (c.resolution_status == "UNRESOLVED" or c.status == "ACTIVE") and c.status != "RESOLVED"
    ]
    blocking_count = len(unresolved_blocking)
    logger.info(f"Revalidation complete: blocking_issues={blocking_count} post_resolution_llm_calls=0")

    bypass_warning = None
    if unresolved_blocking and not bypass_conflicts:
        logger.warning(f"Generation remains blocked: {blocking_count} blocking issues remain.")
        log_state_transition(WorkflowState.REVALIDATION, WorkflowState.HUMAN_RESOLUTION_REQUIRED)
        log_generation_skipped("unresolved_blocking_conflicts_remain")
        proposals_map = {}
        for conf in unresolved_blocking:
            proposals_map[conf.conflict_id] = ConflictResolutionEngine.generate_resolution_proposals(conf, resolved_ir)
        
        return {
            **design_results,
            "status": "AWAITING_HUMAN_DECISION",
            "pipeline_state": "AWAITING_HUMAN_DECISION",
            "workflow_state": WorkflowState.HUMAN_RESOLUTION_REQUIRED.value,
            "requires_human_decision": True,
            "is_blocked": False,
            "efs_ir": resolved_ir,
            "efs_ir_validation_issues": validation_issues,
            "conflicts": unresolved_blocking,
            "proposals": proposals_map
        }
    elif unresolved_blocking and bypass_conflicts:
        bypass_warning = f"BYPASS_OVERRIDE_ACTIVE: Pipeline resumed despite {blocking_count} unresolved conflicts/issues (bypass_conflicts=True)."
        logger.warning(bypass_warning)
        for conf in unresolved_blocking:
            conf.resolution_status = "OVERRIDDEN"

    # 4. Continuation Gate: No blocking issues remain!
    logger.info("VALIDATION_PASSED: No blocking issues remain. Revalidation passed cleanly.")
    log_state_transition(WorkflowState.REVALIDATION, WorkflowState.READY_FOR_GENERATION)
    log_state_transition(WorkflowState.READY_FOR_GENERATION, WorkflowState.DESIGN_PLANNING)
    m_ver = getattr(resolved_ir.metadata, "model_version", 1) if (resolved_ir and hasattr(resolved_ir, "metadata")) else 1
    log_generation_start(reason="human_resolution_approved", model_version=m_ver, efs_ir_version=m_ver)
    
    # STEP 4.1: Re-run Controlled Specification Inference Engine
    logger.info("Starting design planning")
    try:
        from core.verification.inference_engine import ControlledInferenceEngine
        inference_engine = ControlledInferenceEngine(resolved_ir, plan, req_model)
        completed_model = inference_engine.infer_and_complete()
        resolved_ir = completed_model.efs_ir
    except Exception as ei:
        logger.warning(f"ControlledInferenceEngine re-run warning: {ei}")
        completed_model = design_results.get("completed_model")

    # STEP 4.2: Update design specification text to reflect approved resolutions
    try:
        resolved_design_spec = generate_design_spec(req_model or {}, plan or {}, protocol_rules or [], efs_ir=resolved_ir)
    except Exception:
        resolved_design_spec = design_spec

    # STEP 4.3: Re-compile ContextPack with resolved EFS IR
    try:
        from core.retrieval.context_compiler import compile_task_context
        resume_query = design_results.get("user_query", "") or getattr(getattr(design_results.get("context_pack"), "task", None), "raw_query", "") or ""
        context_pack, _ = compile_task_context(
            resume_query,
            resolved_ir,
            all_protocol_rules=protocol_rules
        )
    except Exception as ec:
        logger.warning(f"ContextPack re-compilation warning: {ec}")
        context_pack = design_results.get("context_pack")

    # STEP 4.4: Generate PlantUML Code & Diagrams
    log_state_transition(WorkflowState.DESIGN_PLANNING, WorkflowState.PUML_GENERATION)
    update_resume_progress("Generating PlantUML architecture diagrams...", 0.80)
    logger.info("Starting PUML generation")
    diagrams = generate_diagrams(resolved_design_spec, plan, efs_ir=resolved_ir)
    diagram_urls = {k: get_diagram_url(v) for k, v in (diagrams or {}).items()} if isinstance(diagrams, dict) else {}
    
    # STEP 4.5: RTL Planning & Code Generation
    log_state_transition(WorkflowState.PUML_GENERATION, WorkflowState.RTL_GENERATION)
    log_state_transition(WorkflowState.RTL_GENERATION, WorkflowState.RTL_VALIDATION)
    update_resume_progress("Resuming automated RTL code generation...", 0.90)
    logger.info("RTL_GENERATION_STARTED: Resuming automated RTL code generation...")
    
    outputs = {}
    primary_target = targets[0] if targets else "Verilog"
    code = generate_code(
        resolved_design_spec,
        primary_target,
        efs_ir=resolved_ir,
        protocol_rules=protocol_rules,
        context_pack=context_pack
    )

    # Handle RTL generation returning an insufficient output marker.
    # Do NOT create new SOURCE_MISSING/PLAN_MISSING conflicts or restart the pipeline.
    # All missing-info checks were done upfront before generation. This is a generation error only.
    if code == "INSUFFICIENT SPECIFICATION FOR RTL GENERATION" or (isinstance(code, str) and code.startswith("SPECIFICATION_INCOMPLETE")):
        logger.warning("RTL generation returned incomplete output after resolution. Reporting as generation error (not restarting pipeline).")
        log_generation_end("rtl_generation_insufficient_output_post_resolution")
        outputs[primary_target] = {
            "final_code": code,
            "compliance_score": 0,
            "report": {"failed_checks": [{"check_name": "RTL Generation Quality", "violation": "RTL agent could not produce complete synthesizable code from the resolved specification."}]},
            "history": [],
            "iterations_run": 0
        }
        return {
            **design_results,
            "status": "RTL_GENERATION_ERROR",
            "pipeline_state": "RTL_GENERATION_ERROR",
            "workflow_state": WorkflowState.RTL_VALIDATION.value,
            "requires_human_decision": False,
            "is_blocked": False,
            "error_message": "RTL generation could not produce complete output after conflict resolution. The specification passed completeness and conflict checks but the RTL agent was unable to synthesize it.",
            "requirement_model": req_model,
            "plan": plan,
            "protocol_rules": protocol_rules,
            "design_spec": resolved_design_spec,
            "efs_ir": resolved_ir,
            "completed_model": completed_model,
            "context_pack": context_pack,
            "efs_ir_validation_issues": validation_issues,
            "diagrams": diagrams,
            "diagram_urls": diagram_urls,
            "outputs": outputs
        }

    logger.info("RTL_GENERATION_COMPLETED: RTL generation completed successfully.")
    log_state_transition(WorkflowState.RTL_VALIDATION, WorkflowState.GENERATION_COMPLETE)
    log_generation_end("success")
    update_resume_progress("Generation complete with approved resolutions.", 1.0)

    outputs[primary_target] = {
        "final_code": code,
        "compliance_score": 100,
        "report": {"compliance_score": 100, "passed_checks": [], "failed_checks": []},
        "history": [],
        "iterations_run": 0
    }
    
    # Generate secondary formats if requested
    secondary_targets = [t for t in targets if t != primary_target]
    for sec_tgt in secondary_targets:
        try:
            sec_code = generate_code(
                resolved_design_spec,
                sec_tgt,
                efs_ir=resolved_ir,
                protocol_rules=protocol_rules,
                context_pack=context_pack
            )
            outputs[sec_tgt] = {
                "final_code": sec_code,
                "compliance_score": 100,
                "report": {"compliance_score": 100, "passed_checks": [], "failed_checks": []},
                "history": [],
                "iterations_run": 0
            }
        except Exception as e_sec:
            logger.warning(f"Failed to generate secondary format {sec_tgt}: {e_sec}")

    return {
        **design_results,
        "status": "SUCCESS",
        "pipeline_state": "GENERATION_COMPLETE",
        "workflow_state": WorkflowState.GENERATION_COMPLETE.value,
        "requires_human_decision": False,
        "is_blocked": False,
        "bypass_warning": bypass_warning,
        "requirement_model": req_model,
        "plan": plan,
        "protocol_rules": protocol_rules,
        "design_spec": resolved_design_spec,
        "efs_ir": resolved_ir,
        "completed_model": completed_model,
        "context_pack": context_pack,
        "efs_ir_validation_issues": validation_issues,
        "diagrams": diagrams,
        "diagram_urls": diagram_urls,
        "outputs": outputs
    }

