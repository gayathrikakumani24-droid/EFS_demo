with open("hardware_graphrag/core/agents/orchestrator.py", "r", encoding="utf-8") as f:
    text = f.read().replace("\r\n", "\n")

# 1. In run_design_flow: generate and attach rtl_summary
target1 = '''    # Store finalized primary RTL output
    outputs[primary_target] = {
        "final_code": best_code,
        "compliance_score": best_score,
        "report": best_report,
        "history": loop_history,
        "iterations_run": iteration
    }'''

replacement1 = '''    # Generate RTL architecture and implementation summary
    active_rtl_plan = getattr(efs_ir, "_active_rtl_plan", None)
    rtl_summary = ""
    try:
        from core.agents.rtl_summary_generator import generate_deterministic_rtl_summary
        rtl_summary = generate_deterministic_rtl_summary(
            best_code,
            efs_ir=efs_ir,
            rtl_plan=active_rtl_plan,
            architecture_classification=architecture_classification if 'architecture_classification' in locals() else None,
            verification_report=best_report,
            target_lang=primary_target
        )
    except Exception as e_sum:
        logger.warning(f"Failed to generate RTL summary: {e_sum}")

    # Store finalized primary RTL output
    outputs[primary_target] = {
        "final_code": best_code,
        "compliance_score": best_score,
        "report": best_report,
        "history": loop_history,
        "iterations_run": iteration,
        "rtl_summary": rtl_summary,
        "rtl_plan": active_rtl_plan
    }'''

assert target1 in text, "target1 not found in orchestrator.py"
text = text.replace(target1, replacement1, 1)

# 2. In run_design_flow return dict
target2 = '''        "diagrams": diagrams,
        "diagram_urls": diagram_urls,
        "outputs": outputs,'''

replacement2 = '''        "diagrams": diagrams,
        "diagram_urls": diagram_urls,
        "outputs": outputs,
        "rtl_summary": rtl_summary,
        "rtl_plan": active_rtl_plan,'''

assert target2 in text, "target2 not found in orchestrator.py"
text = text.replace(target2, replacement2, 1)

# 3. In resume_design_flow_with_resolutions: generate and attach rtl_summary
target3 = '''    outputs[primary_target] = {
        "final_code": code,
        "compliance_score": 100,
        "report": {"compliance_score": 100, "passed_checks": [], "failed_checks": []},
        "history": [],
        "iterations_run": 0
    }'''

replacement3 = '''    # Generate RTL architecture and implementation summary post-resolution
    res_active_plan = getattr(resolved_ir, "_active_rtl_plan", None) or getattr(efs_ir, "_active_rtl_plan", None)
    res_rtl_summary = ""
    try:
        from core.agents.rtl_summary_generator import generate_deterministic_rtl_summary
        res_rtl_summary = generate_deterministic_rtl_summary(
            code,
            efs_ir=resolved_ir,
            rtl_plan=res_active_plan,
            architecture_classification=architecture_classification if 'architecture_classification' in locals() else None,
            verification_report={"compliance_score": 100, "passed_checks": [], "failed_checks": []},
            target_lang=primary_target
        )
    except Exception as e_sum:
        logger.warning(f"Failed to generate RTL summary in resume flow: {e_sum}")

    outputs[primary_target] = {
        "final_code": code,
        "compliance_score": 100,
        "report": {"compliance_score": 100, "passed_checks": [], "failed_checks": []},
        "history": [],
        "iterations_run": 0,
        "rtl_summary": res_rtl_summary,
        "rtl_plan": res_active_plan
    }'''

assert target3 in text, "target3 not found in orchestrator.py"
text = text.replace(target3, replacement3, 1)

# 4. In resume_design_flow_with_resolutions return dict
target4 = '''        "diagrams": diagrams,
        "diagram_urls": diagram_urls,
        "outputs": outputs
    }'''

replacement4 = '''        "diagrams": diagrams,
        "diagram_urls": diagram_urls,
        "outputs": outputs,
        "rtl_summary": res_rtl_summary,
        "rtl_plan": res_active_plan
    }'''

assert target4 in text, "target4 not found in orchestrator.py"
text = text.replace(target4, replacement4, 1)

with open("hardware_graphrag/core/agents/orchestrator.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Updated orchestrator.py successfully")
