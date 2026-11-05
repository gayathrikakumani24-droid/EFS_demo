import re

print("Applying orchestrator and UI summary patches...")

# ==============================================================================
# Patch orchestrator.py: Generate & attach rtl_summary and rtl_plan
# ==============================================================================
orch_path = "hardware_graphrag/core/agents/orchestrator.py"
with open(orch_path, "r", encoding="utf-8") as f:
    orch_content = f.read()

orch_norm = orch_content.replace("\r\n", "\n")

# 1. In initial flow (run_design_flow)
target_store_primary = '''    # Store finalized primary RTL output
    outputs[primary_target] = {
        "final_code": best_code,
        "compliance_score": best_score,
        "report": best_report,
        "history": loop_history,
        "iterations_run": iteration
    }'''

replacement_store_primary = '''    # Generate RTL architecture and implementation summary
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

assert target_store_primary in orch_norm, "target_store_primary not found in orchestrator.py"
orch_norm = orch_norm.replace(target_store_primary, replacement_store_primary)

# Attach rtl_summary and rtl_plan to return dict of run_design_flow
target_return_success = '''    return {
        "status": "SUCCESS",
        "pipeline_state": "GENERATION_COMPLETE",
        "workflow_state": WorkflowState.GENERATION_COMPLETE.value,
        "requirement_model": req_model,
        "plan": plan,'''

replacement_return_success = '''    return {
        "status": "SUCCESS",
        "pipeline_state": "GENERATION_COMPLETE",
        "workflow_state": WorkflowState.GENERATION_COMPLETE.value,
        "requirement_model": req_model,
        "plan": plan,
        "rtl_plan": active_rtl_plan,
        "rtl_summary": rtl_summary,'''

assert target_return_success in orch_norm, "target_return_success not found in orchestrator.py"
orch_norm = orch_norm.replace(target_return_success, replacement_return_success)

# 2. In resume flow (resume_design_flow_with_resolutions)
target_resume_store = '''    outputs[primary_target] = {
        "final_code": code,
        "compliance_score": 100,
        "report": {"compliance_score": 100, "passed_checks": [], "failed_checks": []},
        "history": [],
        "iterations_run": 0
    }'''

replacement_resume_store = '''    # Generate RTL architecture and implementation summary post-resolution
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

assert target_resume_store in orch_norm, "target_resume_store not found in orchestrator.py"
orch_norm = orch_norm.replace(target_resume_store, replacement_resume_store)

target_resume_return = '''    return {
        **design_results,
        "status": "SUCCESS",
        "pipeline_state": "GENERATION_COMPLETE",
        "workflow_state": WorkflowState.GENERATION_COMPLETE.value,
        "requires_human_decision": False,
        "is_blocked": False,
        "bypass_warning": bypass_warning,
        "requirement_model": req_model,
        "plan": plan,'''

replacement_resume_return = '''    return {
        **design_results,
        "status": "SUCCESS",
        "pipeline_state": "GENERATION_COMPLETE",
        "workflow_state": WorkflowState.GENERATION_COMPLETE.value,
        "requires_human_decision": False,
        "is_blocked": False,
        "bypass_warning": bypass_warning,
        "requirement_model": req_model,
        "plan": plan,
        "rtl_plan": res_active_plan,
        "rtl_summary": res_rtl_summary,'''

assert target_resume_return in orch_norm, "target_resume_return not found in orchestrator.py"
orch_norm = orch_norm.replace(target_resume_return, replacement_resume_return)

with open(orch_path, "w", encoding="utf-8") as f:
    f.write(orch_norm)
print("1. Successfully updated orchestrator.py with rtl_summary and rtl_plan generation")


# ==============================================================================
# Patch hardware_design.py: Display RTL Summary expander & table docx parsing
# ==============================================================================
ui_path = "hardware_graphrag/ui/hardware_design.py"
with open(ui_path, "r", encoding="utf-8") as f:
    ui_content = f.read()

ui_norm = ui_content.replace("\r\n", "\n")

# 1. Update docx reading to also extract tables
old_docx_read = '''                elif file_ext == ".docx":
                    try:
                        import docx
                        doc = docx.Document(uploaded_doc)
                        raw_req_text = "\\n".join([p.text for p in doc.paragraphs])
                    except ImportError:
                        raw_req_text = str(uploaded_doc.read(), "utf-8", errors="ignore")'''

new_docx_read = '''                elif file_ext == ".docx":
                    try:
                        import docx
                        doc = docx.Document(uploaded_doc)
                        p_texts = [p.text for p in doc.paragraphs if p.text.strip()]
                        t_texts = []
                        for t in doc.tables:
                            for row in t.rows:
                                r_cells = [c.text.strip() for c in row.cells if c.text.strip()]
                                if r_cells:
                                    t_texts.append(" | ".join(r_cells))
                        raw_req_text = "\\n\\n".join(p_texts)
                        if t_texts:
                            raw_req_text += "\\n\\n### Specification Tables:\\n" + "\\n".join(t_texts)
                    except ImportError:
                        raw_req_text = str(uploaded_doc.read(), "utf-8", errors="ignore")'''

if old_docx_read in ui_norm:
    ui_norm = ui_norm.replace(old_docx_read, new_docx_read)
    print("2a. Updated docx parser to extract tables")

# 2. Add RTL Summary expander right after RTL Code expander
old_rtl_code_expander = '''            primary_target = results.get("performance_metrics", {}).get("primary_target", "Verilog")
            if primary_target in outputs:
                code_text = outputs[primary_target].get("final_code", "")
                if code_text and not code_text.startswith("SPECIFICATION_INCOMPLETE"):
                    with st.expander(f"📄 Generated {primary_target} RTL Code", expanded=True):
                        st.code(code_text, language="systemverilog" if "Verilog" in primary_target else "vhdl")'''

new_rtl_code_expander = '''            primary_target = results.get("performance_metrics", {}).get("primary_target", "Verilog")
            if primary_target in outputs:
                code_text = outputs[primary_target].get("final_code", "")
                if code_text and not code_text.startswith("SPECIFICATION_INCOMPLETE"):
                    with st.expander(f"📄 Generated {primary_target} RTL Code", expanded=True):
                        st.code(code_text, language="systemverilog" if "Verilog" in primary_target else "vhdl")

            # Render RTL Implementation Plan and Technical Documentation Summary
            rtl_summary_text = results.get("rtl_summary") or outputs.get(primary_target, {}).get("rtl_summary", "")
            if rtl_summary_text:
                with st.expander("📋 RTL Implementation Plan & Architecture Summary", expanded=True):
                    st.markdown(rtl_summary_text)
            elif results.get("rtl_plan"):
                with st.expander("📋 RTL Implementation Plan Schema", expanded=False):
                    st.json(results.get("rtl_plan"))'''

assert old_rtl_code_expander in ui_norm, "old_rtl_code_expander not found in hardware_design.py"
ui_norm = ui_norm.replace(old_rtl_code_expander, new_rtl_code_expander)
print("2b. Added RTL Implementation Plan & Architecture Summary expander in UI")

with open(ui_path, "w", encoding="utf-8") as f:
    f.write(ui_norm)
print("2. Successfully patched hardware_design.py")
print("All orchestrator and UI patches applied successfully!")
