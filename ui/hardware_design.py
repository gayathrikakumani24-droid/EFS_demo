"""Hardware Design Assistant Page: upload requirements, select protocols, configure options, and trigger generation."""

from __future__ import annotations

import os
import json
import streamlit as st
from config import CONFIG
from core.agents.orchestrator import run_design_flow
from core.efs_ir.models import EFSConflict, EFSResolution
from utils.protocol_manager import get_library_data, select_protocol
from utils.logger import get_logger

logger = get_logger("ui.hardware_design")


def _parse_uploaded_spec_doc(uploaded_doc) -> str:
    file_ext = os.path.splitext(uploaded_doc.name)[1].lower()
    if file_ext == ".txt":
        return str(uploaded_doc.read(), "utf-8")
    elif file_ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(uploaded_doc)
            return "\n".join([p.extract_text() for p in reader.pages])
        except ImportError:
            return str(uploaded_doc.read(), "utf-8", errors="ignore")
    elif file_ext == ".docx":
        try:
            import docx
            doc_obj = docx.Document(uploaded_doc)
            return "\n".join([p.text for p in doc_obj.paragraphs])
        except ImportError:
            return str(uploaded_doc.read(), "utf-8", errors="ignore")
    else:
        return str(uploaded_doc.read(), "utf-8", errors="ignore")


def _handle_doc_upload() -> None:
    doc = st.session_state.get("req_doc_upload")
    if doc:
        try:
            parsed = _parse_uploaded_spec_doc(doc)
            st.session_state["req_spec_text"] = parsed
            st.session_state["_last_parsed_doc_name"] = doc.name
            st.session_state["doc_upload_success"] = f"Parsed content from {doc.name}"
            st.session_state.pop("doc_upload_error", None)

            # Cleanly reset stale design session artifacts on new upload
            for k in [
                "design_results", "active_efs_ir", "active_req_ir", "active_plan",
                "current_conflicts", "extracted_components", "last_active_query",
                "user_target_query", "plantuml_diagrams", "generated_code",
                "compliance_score", "verification_results"
            ]:
                st.session_state.pop(k, None)
            from core.registry import get_registry
            get_registry().reset_design_session(new_doc_id=doc.name)
        except Exception as e:
            st.session_state["doc_upload_error"] = f"Failed to read file: {e}"
            st.session_state.pop("doc_upload_success", None)


def render() -> None:
    st.title("💻 AI Hardware Design Assistant")
    st.caption("Step-by-step wizard to upload specifications, select protocols, and generate compliant designs.")

    # 1. Load protocols library
    lib = get_library_data()
    ready_protocols = [name for name, info in lib.items() if info["status"] == "Ready"]
    
    if not ready_protocols:
        st.warning("⚠️ No protocols are compiled in the Protocol Library yet. Please go to the **Protocol Library** page to upload a protocol specification first.")
        return

    # Check if active protocol in session state is valid
    active_protocol = st.session_state.get("active_protocol", "AXI4")
    if active_protocol not in ready_protocols:
        active_protocol = ready_protocols[0]
        select_protocol(active_protocol)
        st.session_state["active_protocol"] = active_protocol

    # Sidebar: Target configuration
    st.sidebar.markdown("### Target Outputs")
    gen_verilog = st.sidebar.checkbox("Verilog RTL", value=True)
    gen_sv = st.sidebar.checkbox("SystemVerilog RTL", value=False)
    gen_vhdl = st.sidebar.checkbox("VHDL Design", value=False)

    st.sidebar.markdown("### Hybrid Generation Engine")
    generation_mode = st.sidebar.selectbox(
        "RTL Generation Mode",
        options=["HYBRID", "DETERMINISTIC", "DIRECT_LLM"],
        index=0,
        help="HYBRID combines deterministic structure with LLM behavior. DETERMINISTIC uses pure structural/library synthesis. DIRECT_LLM prompts LLM with locked header."
    )
    k_candidates = st.sidebar.slider("Number of Candidates (k)", min_value=1, max_value=3, value=1, help="Generate and rank k candidate RTL implementations.")
    allow_partial = st.sidebar.checkbox("Allow Safe Partial Generation", value=False, help="Safely generate partial RTL when requirements are incomplete.")

    st.sidebar.markdown("### Loop Configurations")
    max_repair = st.sidebar.slider("Max Repair Iterations", min_value=0, max_value=5, value=0)
    score_threshold = st.sidebar.slider("Compliance Target Score (%)", min_value=50, max_value=100, value=95)
    bypass_conflicts = st.sidebar.checkbox("Bypass Critical Conflicts / Override Block", value=False)
    if bypass_conflicts:
        st.sidebar.warning("⚠️ Override enabled: The pipeline will continue without resolving critical conflicts. Generated RTL may not conform to the specification requirements.")

    st.sidebar.markdown("### Refinement Prompt")
    refinement_instruction = st.sidebar.text_area("Guidance for regeneration/refinement:", height=80)

    # Main Panel Wizard
    st.markdown("### STEP 1: Provide User Requirement Specification")
    
    DEFAULT_REQ_SPEC = (
        "Design a Custom Acceleration Controller module.\n"
        "Interfaces:\n"
        "  - clk: input, 1 bit, System clock\n"
        "  - rst_n: input, 1 bit, Active-low reset\n"
        "  - instruction: input, 64 bits, Instruction word\n"
        "  - start: input, 1 bit, Start trigger\n"
        "  - busy: output, 1 bit, Busy status\n"
        "  - done: output, 1 bit, Execution complete"
    )
    # Check if a document is already uploaded in session state and hasn't been parsed yet
    uploaded_doc_in_state = st.session_state.get("req_doc_upload")
    if uploaded_doc_in_state:
        last_parsed = st.session_state.get("_last_parsed_doc_name")
        if last_parsed != uploaded_doc_in_state.name:
            _handle_doc_upload()

    if "req_spec_text" not in st.session_state:
        st.session_state["req_spec_text"] = DEFAULT_REQ_SPEC

    req_cols = st.columns([2, 1])
    with req_cols[0]:
        raw_req_text = st.text_area(
            "Input requirement text directly:",
            height=160,
            key="req_spec_text"
        )
    with req_cols[1]:
        uploaded_doc = st.file_uploader(
            "Or upload requirement spec (PDF/DOCX/TXT/MD):",
            type=["pdf", "docx", "txt", "md"],
            key="req_doc_upload",
            on_change=_handle_doc_upload
        )
        if st.session_state.get("doc_upload_success"):
            st.success(st.session_state["doc_upload_success"])
        if st.session_state.get("doc_upload_error"):
            st.error(st.session_state["doc_upload_error"])

    st.markdown("### STEP 2: Select Protocol Context")
    
    proto_cols = st.columns(2)
    with proto_cols[0]:
        # Option A: select from library
        active_index = ready_protocols.index(active_protocol) if active_protocol in ready_protocols else 0
        selected_proto = st.selectbox("Select from Protocol Library", ready_protocols, index=active_index)
        if selected_proto != active_protocol:
            select_protocol(selected_proto)
            st.session_state["active_protocol"] = selected_proto
            st.rerun()
            
    with proto_cols[1]:
        # Option B: upload new protocol specification
        uploaded_proto = st.file_uploader(
            "Or upload a new Protocol Specification (e.g. AXI5.pdf)",
            type=["pdf", "docx", "txt", "md"],
            key="new_proto_upload"
        )
        if uploaded_proto:
            st.info("💡 To ingest a new protocol, navigate to the **Protocol Library** page, upload it under the respective name block, and process it into the reusable knowledge base.")

    st.markdown("### STEP 3: Enter Targeted Generation Query")
    
    # Compute sensible suggested query based on current requirement or active design
    suggested_target = "target_module"
    current_req = st.session_state.get("req_spec_text", raw_req_text).strip()
    import re
    if re.search(r"\bFSM\b", current_req, re.IGNORECASE):
        suggested_target = "fsm_controller"
    else:
        m_name = re.search(r"(?:design\s+(?:a|an)\s+|module\s+)([A-Za-z0-9_ ]+?)(?:\s+module|\s+controller|\.|\n|$)", current_req, re.IGNORECASE)
        if m_name and len(m_name.group(1).strip()) > 2 and len(m_name.group(1).strip()) < 50:
            cand = m_name.group(1).strip()
            if cand.lower() not in ("simple", "custom", "new", "hardware"):
                suggested_target = re.sub(r"\s+", "_", cand.lower())

    # Fallback to session results if available
    res_in_state = st.session_state.get("design_results")
    if res_in_state and isinstance(res_in_state, dict):
        efs_in_state = res_in_state.get("efs_ir")
        if efs_in_state and hasattr(efs_in_state, "metadata") and efs_in_state.metadata.design_name:
            suggested_target = efs_in_state.metadata.design_name

    default_query_text = f"Generate synthesizable Verilog RTL for {suggested_target}."

    if "user_target_query" not in st.session_state:
        st.session_state["user_target_query"] = ""

    user_query = st.text_input(
        "Targeted Generation Query:",
        key="user_target_query",
        placeholder=f"e.g., {default_query_text}",
        help="Specify the target component, language, or format requested (e.g., 'Generate synthesizable Verilog RTL for Custom Acceleration Controller', 'Generate SVA assertions for AXI interface', 'Generate UVM testbench')."
    )
    effective_query = user_query.strip() if (user_query and user_query.strip()) else default_query_text
    st.session_state["last_active_query"] = effective_query

    st.markdown("### STEP 4: Review Target Outputs & Trigger Generation")
    targets_summary = []
    if gen_verilog: targets_summary.append("Verilog RTL")
    if gen_sv: targets_summary.append("SystemVerilog RTL")
    if gen_vhdl: targets_summary.append("VHDL Design")
    
    st.markdown(f"**Targets selected:** {', '.join(targets_summary) if targets_summary else 'None (Select in sidebar)'}")

    targets = []
    if gen_verilog: targets.append("Verilog")
    if gen_sv: targets.append("SystemVerilog")
    if gen_vhdl: targets.append("VHDL")

    generate_disabled = not targets or not raw_req_text

    col_btn, col_status = st.columns([1, 2])
    
    with col_btn:
        if st.button("🚀 Start Generation Flow", type="primary", disabled=generate_disabled):
            progress_bar = st.progress(0.0)
            status_container = st.empty()

            def progress_cb(message: str, pct: float) -> None:
                progress_bar.progress(min(max(pct, 0.0), 1.0))
                # Map progress ratio to steps list
                steps = [
                    ("Parsing requirement", 0.05),
                    ("Loading protocol knowledge", 0.15),
                    ("Retrieving relevant rules", 0.25),
                    ("Planning architecture", 0.35),
                    ("Building canonical EFS IR", 0.40),
                    ("Compiling task ContextPack", 0.42),
                    ("Generating PlantUML code & diagrams", 0.45),
                    ("Generating Primary RTL", 0.65),
                    ("Generating secondary formats", 0.90),
                ]
                
                checklist = []
                # Determine index of active step
                active_idx = -1
                for idx, (step_name, step_pct) in enumerate(steps):
                    if pct >= step_pct:
                        checklist.append(f"✓ {step_name}")
                        active_idx = idx
                    else:
                        if idx == active_idx + 1:
                            checklist.append(f"⏳ {step_name} ({message})")
                        else:
                            checklist.append(f"○ {step_name}")
                
                status_container.markdown("\n".join([f"- {item}" for item in checklist]))

            # Append refinement prompt if user added one
            final_req = raw_req_text
            if refinement_instruction:
                final_req += f"\n\nRefinement Instruction Guideline:\n{refinement_instruction}"

            try:
                # Run the orchestrator end-to-end
                results = run_design_flow(
                    final_req,
                    targets,
                    max_repair_iterations=max_repair,
                    compliance_threshold=score_threshold,
                    progress_cb=progress_cb,
                    bypass_conflicts=bypass_conflicts,
                    user_query=effective_query,
                    generation_mode=generation_mode,
                    k_candidates=k_candidates,
                    allow_partial=allow_partial
                )
                
                # Save results to session state
                st.session_state["design_results"] = results
                if results.get("status") in ("TARGET_NOT_FOUND", "AMBIGUOUS_TARGET"):
                    st.error(f"❌ Target Resolution Failed: {results.get('error_message')}")
                elif results.get("status") in ("AWAITING_HUMAN_DECISION", "BLOCKED_CONFLICT"):
                    st.warning("⚠️ Human Decision Required: Specification conflicts or ambiguities detected. The pipeline is safely paused at this decision point.")
                elif results.get("status") == "BLOCKED":
                    st.error("❌ Generation BLOCKED due to critical builder errors.")
                elif results.get("status") == "INSUFFICIENT_SPEC":
                    st.warning("⚠️ Human Decision Required: Incomplete specification details require clarification.")
                else:
                    st.success("✅ Design flow execution complete!")
                    st.balloons()
            except Exception as e:
                logger.exception("Design flow failed")
                st.error(f"❌ Design flow execution failed: {e}")

    # Display results and Context Preview
    if "design_results" in st.session_state:
        results = st.session_state["design_results"]
        pipe_state = results.get("pipeline_state", results.get("status", "PLANNING"))

        # Pipeline State Status Badge
        badge_style_map = {
            "PLANNING": ("#3b82f6", "#1e3a8a", "📐 PLANNING"),
            "AWAITING_HITL": ("#f59e0b", "#78350f", "⚠️ AWAITING_HITL"),
            "AWAITING_HUMAN_DECISION": ("#f59e0b", "#78350f", "⚠️ AWAITING_HITL"),
            "GENERATING": ("#8b5cf6", "#4c1d95", "⚙️ GENERATING"),
            "VERIFYING": ("#06b6d4", "#164e63", "🔍 VERIFYING"),
            "PARTIAL": ("#eab308", "#713f12", "⚡ PARTIAL (Compliant Subsets)"),
            "COMPLETED": ("#10b981", "#064e3b", "✅ COMPLETED"),
            "GENERATION_COMPLETE": ("#10b981", "#064e3b", "✅ COMPLETED"),
            "SUCCESS": ("#10b981", "#064e3b", "✅ COMPLETED"),
            "FAILED": ("#ef4444", "#7f1d1d", "❌ FAILED"),
            "RTL_VALIDATION_ERROR": ("#ef4444", "#7f1d1d", "❌ FAILED"),
            "RTL_GENERATION_ERROR": ("#ef4444", "#7f1d1d", "❌ FAILED"),
            "BLOCKED": ("#ef4444", "#7f1d1d", "❌ BLOCKED"),
        }
        b_fg, b_bg, b_label = badge_style_map.get(pipe_state, ("#64748b", "#1e293b", pipe_state))

        st.markdown(f"""
        <div style="background-color: #1e293b; padding: 12px 18px; border-radius: 8px; margin-bottom: 18px; border-left: 5px solid {b_fg};">
          <div style="font-weight: 700; font-size: 15px; margin-bottom: 4px; color: {b_fg};">
            Pipeline Execution State: <span style="background: {b_bg}; padding: 2px 10px; border-radius: 4px;">{b_label}</span>
          </div>
          <div style="font-size: 13px; color: #cbd5e1; line-height: 1.6;">
            <span style="color: #10b981; font-weight: 600;">✓ Ingestion</span> &nbsp;→&nbsp;
            <span style="color: #10b981; font-weight: 600;">✓ EFS IR & Validation</span> &nbsp;→&nbsp;
            <span style="color: #10b981; font-weight: 600;">✓ Classification & Planning</span> &nbsp;→&nbsp;
            <span style="color: {'#fbbf24' if 'HITL' in pipe_state else '#10b981'}; font-weight: 600;">{'⚠️ HITL Gate' if 'HITL' in pipe_state else '✓ HITL Gate'}</span> &nbsp;→&nbsp;
            <span style="color: {'#10b981' if pipe_state in ('COMPLETED', 'SUCCESS', 'GENERATION_COMPLETE', 'PARTIAL') else '#94a3b8'};">{'✓ Hybrid RTL Generation' if pipe_state in ('COMPLETED', 'SUCCESS', 'GENERATION_COMPLETE', 'PARTIAL') else '○ Hybrid Generation'}</span> &nbsp;→&nbsp;
            <span style="color: {'#10b981' if pipe_state in ('COMPLETED', 'SUCCESS', 'GENERATION_COMPLETE') else ('#ef4444' if pipe_state == 'FAILED' else '#94a3b8')};">{'✓ Verification & Selection' if pipe_state in ('COMPLETED', 'SUCCESS', 'GENERATION_COMPLETE') else '○ Verification'}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Architectural Classification & Plan Summary Cards
        pri_target = results.get("performance_metrics", {}).get("primary_target", "Verilog")
        out_dict = results.get("outputs", {}).get(pri_target, {})
        arch_val = out_dict.get("architecture_classification") or results.get("architecture_classification") or "CUSTOM_ARCHITECTURE"
        plan_dict = out_dict.get("plan") or results.get("rtl_generation_plan") or results.get("plan") or {}
        fsm_strat = plan_dict.get("fsm_strategy", "INFERRED") if isinstance(plan_dict, dict) else "N/A"

        card1, card2, card3, card4 = st.columns(4)
        card1.metric("Architecture Classification", str(arch_val))
        card2.metric("FSM Strategy", str(fsm_strat))
        card3.metric("HDL Target", str(pri_target))
        card4.metric("Generation Mode", str(generation_mode))

        # Expandable Plan Summary
        if plan_dict and isinstance(plan_dict, dict):
            with st.expander("📐 Structured RTL Generation Plan Summary", expanded=False):
                p_cols = st.columns(3)
                if "ports" in plan_dict or "internal_registers" in plan_dict:
                    p_cols[0].write(f"**Module Name:** `{plan_dict.get('module_name', 'mod')}`")
                    p_cols[1].write(f"**Ports Defined:** {len(plan_dict.get('ports', []))}")
                    p_cols[2].write(f"**Registers Defined:** {len(plan_dict.get('internal_registers', []))}")
                else:
                    p_cols[0].write(f"**System Architecture:** `{plan_dict.get('system_architecture', 'Standard Architecture')}`")
                    p_cols[1].write(f"**Submodules:** {len(plan_dict.get('submodules', []))}")
                    p_cols[2].write(f"**Clock Domains:** {len(plan_dict.get('clock_domains', []))}")
                st.json(plan_dict)
        
        if results.get("status") == "SUCCESS":
            st.success("✅ RTL and PlantUML Diagrams Generated Successfully")
            outputs = results.get("outputs", {})
            primary_target = results.get("performance_metrics", {}).get("primary_target", "Verilog")
            if primary_target in outputs:
                code_text = outputs[primary_target].get("final_code", "")
                if code_text and not code_text.startswith("SPECIFICATION_INCOMPLETE"):
                    with st.expander(f"📄 Generated {primary_target} RTL Code", expanded=True):
                        st.code(code_text, language="systemverilog" if "Verilog" in primary_target else "vhdl")

            # Comprehensive, Structured RTL Technical Documentation & Architecture Summary
            summary_text = (
                results.get("rtl_summary")
                or outputs.get(primary_target, {}).get("summary")
            )
            if not summary_text:
                try:
                    from core.agents.rtl_summary_generator import generate_comprehensive_rtl_summary
                    if code_text and not code_text.startswith("SPECIFICATION_INCOMPLETE"):
                        summary_text = generate_comprehensive_rtl_summary(
                            code=code_text,
                            efs_ir=results.get("efs_ir"),
                            rtl_plan=plan_dict,
                            architecture_classification=arch_val,
                            verification_report=outputs.get(primary_target, {}).get("report"),
                            target_lang=primary_target
                        )
                except Exception as e_sum:
                    logger.debug(f"UI dynamic summary generation exception: {e_sum}")

            if summary_text:
                with st.expander("📑 Comprehensive RTL Architecture & Technical Documentation", expanded=True):
                    st.markdown(summary_text)

            for tgt, out_data in outputs.items():
                if tgt != primary_target and isinstance(out_data, dict):
                    sec_code = out_data.get("final_code", "")
                    if sec_code:
                        with st.expander(f"📄 Generated {tgt} RTL Code", expanded=False):
                            st.code(sec_code, language="systemverilog" if "Verilog" in tgt else "vhdl")

            diagrams = results.get("diagrams", {})
            diagram_urls = results.get("diagram_urls", {})
            if diagrams or diagram_urls:
                with st.expander("📊 Generated PlantUML Architecture & Sequence Diagrams", expanded=True):
                    for k, u in diagram_urls.items():
                        if u:
                            st.image(u, caption=f"Generated {k.capitalize()} Diagram")
                    for k, code in diagrams.items():
                        st.markdown(f"**PlantUML ({k.capitalize()}):**")
                        st.code(code, language="plantuml")

        if results.get("status") == "RTL_GENERATION_ERROR":
            st.error(f"❌ RTL Generation Error: {results.get('error_message', 'RTL generation could not produce complete code.')}")
            outputs = results.get("outputs", {})
            for tgt, out_data in outputs.items():
                if isinstance(out_data, dict):
                    report = out_data.get("report", {})
                    failed = report.get("failed_checks", [])
                    if failed:
                        st.markdown("#### RTL Validation Issues:")
                        for fc in failed:
                            st.warning(f"• **{fc.get('check_name', 'Check')}**: {fc.get('violation', '')}")
                    code_text = out_data.get("final_code", "")
                    if code_text:
                        with st.expander(f"📄 Partial / Diagnostic {tgt} Output", expanded=False):
                            st.code(code_text)

            diagrams = results.get("diagrams", {})
            diagram_urls = results.get("diagram_urls", {})
            if diagrams or diagram_urls:
                with st.expander("📊 Generated PlantUML Architecture & Sequence Diagrams", expanded=True):
                    for k, u in diagram_urls.items():
                        if u:
                            st.image(u, caption=f"Generated {k.capitalize()} Diagram")
                    for k, code in diagrams.items():
                        st.markdown(f"**PlantUML ({k.capitalize()}):**")
                        st.code(code, language="plantuml")

        if results.get("status") in ("TARGET_NOT_FOUND", "AMBIGUOUS_TARGET"):
            st.error(f"⚠️ TARGET RESOLUTION FAILED: {results.get('error_message')}")
            if st.button("🔄 Reset Pipeline Session", use_container_width=True):
                del st.session_state["design_results"]
                st.rerun()
            return

        if results.get("status") in ("AWAITING_HUMAN_DECISION", "BLOCKED", "BLOCKED_CONFLICT", "INSUFFICIENT_SPEC"):
            # Live Pipeline Stage Tracker
            st.markdown("""
            <div style="background-color: #1e293b; padding: 12px 18px; border-radius: 8px; margin-bottom: 18px; border-left: 5px solid #f59e0b;">
              <div style="font-weight: 700; font-size: 15px; margin-bottom: 6px; color: #fbbf24;">
                Pipeline State: <code>AWAITING_HUMAN_DECISION</code>
              </div>
              <div style="font-size: 13px; color: #cbd5e1; line-height: 1.6;">
                <span style="color: #10b981; font-weight: 600;">✓ Parse Specification</span> &nbsp;→&nbsp;
                <span style="color: #10b981; font-weight: 600;">✓ Canonical EFS IR</span> &nbsp;→&nbsp;
                <span style="color: #10b981; font-weight: 600;">✓ Validate Spec</span> &nbsp;→&nbsp;
                <span style="color: #fbbf24; font-weight: 700; background: #78350f; padding: 2px 8px; border-radius: 4px;">⚠️ Awaiting Human Decision</span> &nbsp;→&nbsp;
                <span style="color: #94a3b8;">○ Revalidate</span> &nbsp;→&nbsp;
                <span style="color: #94a3b8;">○ PlantUML</span> &nbsp;→&nbsp;
                <span style="color: #94a3b8;">○ RTL Generation</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            conflicts = results.get("conflicts", [])
            proposals_map = results.get("proposals", {})
            efs_ir = results.get("efs_ir")
            req_model = results.get("requirement_model")
            
            if not conflicts and efs_ir:
                from core.verification.conflict_resolution_engine import ConflictResolutionEngine
                conflicts = ConflictResolutionEngine.detect_conflicts(
                    efs_ir, req_model, validation_issues=results.get("efs_ir_validation_issues")
                )

            # Build resolved conflict lookup from approved resolutions and efs_ir
            approved_res_list = st.session_state.get("approved_resolutions", [])
            approved_cids = {getattr(r, "conflict_id", None) for r in approved_res_list if getattr(r, "conflict_id", None)}
            if efs_ir and hasattr(efs_ir, "resolutions"):
                for r in efs_ir.resolutions:
                    if getattr(r, "approval_status", "") == "APPROVED" and getattr(r, "conflict_id", None):
                        approved_cids.add(r.conflict_id)

            approved_entity_props = set()
            for r in approved_res_list:
                r_prop = (getattr(r, "field_name", "") or getattr(r, "property", "") or "").lower()
                for req in getattr(r, "affected_requirements", []):
                    approved_entity_props.add((req.lower(), r_prop))
            if efs_ir and hasattr(efs_ir, "resolutions"):
                for r in efs_ir.resolutions:
                    if getattr(r, "approval_status", "") == "APPROVED":
                        r_prop = (getattr(r, "field_name", "") or getattr(r, "property", "") or "").lower()
                        for req in getattr(r, "affected_requirements", []):
                            approved_entity_props.add((req.lower(), r_prop))

            def is_conflict_resolved(c) -> bool:
                if getattr(c, "status", "") == "RESOLVED":
                    return True
                if getattr(c, "resolution_status", "UNRESOLVED") == "RESOLVED":
                    return True
                if getattr(c, "status", "") == "APPROVED":
                    return True
                if not getattr(c, "blocking", True) or not getattr(c, "requires_human_decision", True):
                    return True
                if getattr(c, "conflict_id", None) in approved_cids:
                    return True
                c_prop = (getattr(c, "property", "") or "").lower()
                for ent in (getattr(c, "affected_entities", []) or getattr(c, "entities", []) or []):
                    if c_prop and (ent.lower(), c_prop) in approved_entity_props:
                        return True
                return False

            unresolved_conflicts = [c for c in conflicts if not is_conflict_resolved(c)]

            if unresolved_conflicts:
                st.warning(f"### ⚠️ Human Decision Required ({len(unresolved_conflicts)} decision{'s' if len(unresolved_conflicts) != 1 else ''} pending)")
                st.info("The design pipeline identified hardware specification conflicts or ambiguities that require a design decision. The pipeline is safely paused at this decision point.")
            else:
                st.success("### ✓ All specification issues and conflicts resolved.")
                if st.button("▶ Resume Design & RTL Generation", type="primary", key="btn_resume_generation"):
                    with st.spinner("Resuming PlantUML and RTL generation..."):
                        from core.agents.orchestrator import resume_design_flow_with_resolutions
                        # Collect approved resolutions from both session_state AND efs_ir.resolutions
                        # (conflicts may have been resolved in-memory without being stored in session_state)
                        _ss_resolutions = list(st.session_state.get("approved_resolutions", []))
                        _ss_res_ids = {getattr(r, "resolution_id", "") for r in _ss_resolutions}
                        if efs_ir and hasattr(efs_ir, "resolutions"):
                            for _r in efs_ir.resolutions:
                                if (getattr(_r, "approval_status", "") == "APPROVED"
                                        and getattr(_r, "resolution_id", "") not in _ss_res_ids):
                                    _ss_resolutions.append(_r)
                        _resume_targets = targets if targets else ["Verilog"]
                        _res_query = (
                            st.session_state.get("user_target_query")
                            or st.session_state.get("last_active_query")
                            or results.get("user_query")
                            or default_query_text
                        )
                        res_results = resume_design_flow_with_resolutions(
                            results,
                            _ss_resolutions,
                            target_outputs=_resume_targets,
                            user_query=_res_query,
                            bypass_conflicts=bypass_conflicts,
                            generation_mode=generation_mode,
                            k_candidates=k_candidates,
                            allow_partial=allow_partial
                        )
                        st.session_state["design_results"] = res_results
                        if "outputs" in res_results:
                            st.session_state["generated_rtl"] = res_results["outputs"]
                        if "efs_ir" in res_results and res_results["efs_ir"]:
                            st.session_state["resolved_efsir"] = res_results["efs_ir"]
                        if "plan" in res_results:
                            st.session_state["design_plan"] = res_results["plan"]
                        st.session_state["generation_status"] = res_results.get("status", "SUCCESS")
                        st.rerun()

            if unresolved_conflicts:
                for conf_idx, conf in enumerate(unresolved_conflicts, 1):
                    with st.container(border=True):
                        st.markdown(f"### 🔴 Conflict #{conf_idx}: `{conf.title or conf.type}`")
                        col_meta1, col_meta2, col_meta3 = st.columns(3)
                        with col_meta1:
                            st.caption(f"**Severity:** `{conf.severity}`")
                        with col_meta2:
                            st.caption(f"**Conflict ID:** `{conf.conflict_id}`")
                        with col_meta3:
                            st.caption(f"**Property:** `{conf.property or ('opcode' if conf.type == 'DUPLICATE_OPCODE' else 'Specification')}`")

                        # 1. WHAT is wrong
                        st.markdown("#### 1. What is Wrong")
                        st.markdown(f"**Problem:** {conf.description or conf.evidence}")

                        # 2. WHY it is a problem
                        st.markdown("#### 2. Why This Matters")
                        st.markdown(f"**Impact:** {conf.explanation or conf.impact or 'Contradictory or missing parameters prevent unambiguous hardware synthesis.'}")

                        # 3. WHICH entities are affected & WHERE the conflicting info came from
                        st.markdown("#### 3. Affected Entities & Source Evidence")
                        raw_ents = conf.affected_entities or conf.entities or conf.target_entities or getattr(conf, "candidates", []) or []
                        clean_ents = []
                        for e in raw_ents:
                            if str(e).startswith("OPC_") and efs_ir:
                                m = next((o.mnemonic for o in getattr(efs_ir, "opcodes", []) if o.opcode_id == e and o.mnemonic), None)
                                clean_ents.append(m or str(e))
                            else:
                                clean_ents.append(str(e))
                        non_gen = [e for e in clean_ents if e.strip().upper() != "GENERAL"]
                        entities = non_gen or clean_ents or ["Unresolved_Entity"]
                        
                        source_refs = getattr(conf, "source_references", [])
                        if source_refs:
                            ref_rows = []
                            for ref in source_refs:
                                ref_rows.append({
                                    "Entity": ref.get("entity", "-"),
                                    "Source Document": ref.get("source", "Specification"),
                                    "Section / Context": ref.get("section", "Hardware Definition"),
                                    "Conflicting Value": str(ref.get("value", "-"))
                                })
                            st.table(ref_rows)
                        else:
                            col_e1, col_e2 = st.columns(2)
                            with col_e1:
                                st.markdown(f"**Affected Entities:** `{', '.join(entities)}`")
                            with col_e2:
                                st.markdown(f"**Conflicting Values:** `{', '.join(str(v) for v in conf.conflicting_values) if conf.conflicting_values else 'N/A'}`")
                            st.caption("Evidence Source: Extracted from Active Hardware Requirement Model.")

                        st.divider()

                        # 4. AI Recommendation & Rationale
                        st.markdown("#### 4. AI Recommendation & Proposed Resolution")
                        props = proposals_map.get(conf.conflict_id, [])
                        if not props and efs_ir:
                            from core.verification.conflict_resolution_engine import ConflictResolutionEngine
                            props = ConflictResolutionEngine.generate_resolution_proposals(conf, efs_ir)

                        ai_props = [p for p in props if p.resolution_type in ("ALLOW_AI", "ACCEPT_PROPOSAL", "PROPOSED", "AI_RECOMMENDED")]
                        if not ai_props:
                            ai_props = props

                        top_ai_proposal = ai_props[0] if ai_props else None

                        if top_ai_proposal:
                            target_ent = top_ai_proposal.affected_requirements[0] if top_ai_proposal.affected_requirements else (entities[0] if entities else "Entity")
                            orig_val = top_ai_proposal.original_value or conf.current_values.get(target_ent) or (conf.conflicting_values[0] if conf.conflicting_values else "N/A")
                            with st.container(border=True):
                                st.markdown(f"**Target Entity:** `{target_ent}`")
                                st.markdown(f"**Proposed Change:** `{target_ent}.{conf.property or 'value'} : {orig_val} → {top_ai_proposal.proposed_value}`")
                                st.markdown(f"**Reason / Rationale:** {top_ai_proposal.rationale}")
                        else:
                            st.info("⚠️ AI proposal unavailable for this conflict type. Please provide your custom solution below.")

                        st.divider()

                        # 5. HUMAN DECISION PATHS: OPTION 1 OR OPTION 2
                        st.markdown("#### 5. Human Decision Required")
                        strategy_key = f"strat_{conf.conflict_id}"
                        strategy_options = []
                        if top_ai_proposal:
                            strategy_options.append("OPTION 1: Accept AI Proposal")
                        strategy_options.append("OPTION 2: Provide My Own Solution")

                        chosen_strategy = st.radio(
                            "Choose your resolution path for this conflict:",
                            options=strategy_options,
                            key=strategy_key
                        )

                        if chosen_strategy == "OPTION 1: Accept AI Proposal" and top_ai_proposal:
                            st.markdown(f"Applying AI proposal: `{top_ai_proposal.proposed_change}`")
                            if st.button("✓ Accept AI Proposal", key=f"btn_accept_ai_{conf.conflict_id}", type="primary"):
                                if "approved_resolutions" not in st.session_state:
                                    st.session_state["approved_resolutions"] = []
                                top_ai_proposal.approval_status = "APPROVED"
                                if not any(getattr(r, "conflict_id", "") == top_ai_proposal.conflict_id or getattr(r, "resolution_id", "") == top_ai_proposal.resolution_id for r in st.session_state["approved_resolutions"]):
                                    st.session_state["approved_resolutions"].append(top_ai_proposal)

                                # Apply to efs_ir in memory using deterministic pipeline
                                if efs_ir:
                                    try:
                                        from core.verification.conflict_resolution_engine import ConflictResolutionEngine
                                        efs_ir = ConflictResolutionEngine.apply_approved_resolution(efs_ir, top_ai_proposal)
                                        results["efs_ir"] = efs_ir
                                        results["conflicts"] = efs_ir.conflicts
                                        st.session_state["resolved_efsir"] = efs_ir
                                    except Exception:
                                        pass

                                # Check remaining active unresolved conflicts from authoritative EFS IR
                                current_conf_pool = getattr(efs_ir, "conflicts", []) if efs_ir else unresolved_conflicts
                                remaining = [c for c in current_conf_pool if not is_conflict_resolved(c)]

                                if remaining:
                                    results["conflicts"] = remaining
                                    st.session_state["design_results"] = results
                                    st.success(f"✓ Conflict '{conf.title or conf.conflict_id}' resolved. ({len(remaining)} pending)")
                                    st.rerun()
                                else:
                                    with st.spinner("All conflicts resolved! Resuming PlantUML and RTL generation..."):
                                        from core.agents.orchestrator import resume_design_flow_with_resolutions
                                        _res_query = (
                                            st.session_state.get("user_target_query")
                                            or st.session_state.get("last_active_query")
                                            or results.get("user_query")
                                            or default_query_text
                                        )
                                        res_results = resume_design_flow_with_resolutions(
                                            results,
                                            st.session_state["approved_resolutions"],
                                            target_outputs=targets,
                                            user_query=_res_query
                                        )
                                        st.session_state["design_results"] = res_results
                                        if "outputs" in res_results:
                                            st.session_state["generated_rtl"] = res_results["outputs"]
                                        if "efs_ir" in res_results and res_results["efs_ir"]:
                                            st.session_state["resolved_efsir"] = res_results["efs_ir"]
                                        if "plan" in res_results:
                                            st.session_state["design_plan"] = res_results["plan"]
                                        st.session_state["generation_status"] = res_results.get("status", "SUCCESS")
                                        st.rerun()

                        elif chosen_strategy == "OPTION 2: Provide My Own Solution":
                            st.markdown("##### ✎ Provide My Own Solution")
                            col_sel_ent, col_sel_field = st.columns(2)
                            with col_sel_ent:
                                target_entity_choice = st.selectbox(
                                    "Select entity to modify:",
                                    options=entities,
                                    key=f"custom_ent_{conf.conflict_id}"
                                )
                            with col_sel_field:
                                current_val_show = conf.current_values.get(target_entity_choice) or (conf.conflicting_values[0] if conf.conflicting_values else "N/A")
                                st.text_input("Current Value:", value=str(current_val_show), disabled=True, key=f"curr_val_{conf.conflict_id}")

                            effective_prop = conf.property or ("opcode" if conf.type == "DUPLICATE_OPCODE" else "value")
                            if effective_prop in ("transition_condition", "fsm_definition"):
                                prop_placeholder = "e.g. start == 1'b1, req == 1'b1, timer_cnt >= 4'd4"
                                st.info("💡 **FSM Condition Guidance:** Enter an explicit Boolean condition (e.g. `start == 1'b1`, `timer_cnt >= 4'd10`, `data_valid && ready`) to define the state transition.")
                            elif effective_prop == "direction":
                                prop_placeholder = "e.g. input, output, or inout"
                            elif effective_prop == "width":
                                prop_placeholder = "e.g. 32, 16, 8, [7:0]"
                            elif effective_prop in ("offset", "address"):
                                prop_placeholder = "e.g. 0x08, 0x10, 0x14"
                            elif effective_prop in ("opcode", "binary_encoding"):
                                prop_placeholder = "e.g. 1011, 4'b1011, 0x0B"
                            else:
                                prop_placeholder = f"e.g. 1011 (opcode), 0x08 (offset), 32 (width), timer_cnt == 4'd4 (transition)..."

                            custom_new_val = st.text_input(
                                f"Enter new value for {effective_prop}:",
                                placeholder=prop_placeholder,
                                key=f"custom_val_{conf.conflict_id}"
                            )
                            custom_reason = st.text_area(
                                "Design rationale / reason for this change (optional):",
                                placeholder="Explain why this resolution conforms to the specification...",
                                key=f"custom_reason_{conf.conflict_id}",
                                height=70
                            )

                            if st.button("Apply My Solution", key=f"btn_custom_sol_{conf.conflict_id}", type="primary"):
                                if not custom_new_val or not custom_new_val.strip():
                                    st.error("❌ Resolution value cannot be empty. Please enter a valid value.")
                                else:
                                    from core.verification.conflict_resolution_engine import ConflictResolutionEngine
                                    # Ensure conflict has effective_prop populated
                                    if not conf.property:
                                        conf.property = effective_prop
                                    # Deterministic Server-Side Validation
                                    is_valid, val_err = ConflictResolutionEngine.validate_resolution(
                                        conf,
                                        custom_new_val.strip(),
                                        efs_ir,
                                        target_entity=target_entity_choice
                                    )
                                    if not is_valid:
                                        st.error(f"❌ Resolution Rejected by Validator: {val_err}")
                                        st.info("Please adjust the value or select an alternative to avoid colliding with other hardware entities.")
                                    else:
                                        # Valid resolution - apply and resume
                                        custom_res = ConflictResolutionEngine.create_custom_resolution(
                                            conf,
                                            f"{target_entity_choice}.{effective_prop} = {custom_new_val.strip()}",
                                            user_name="human_designer",
                                            target_entity=target_entity_choice
                                        )
                                        if custom_reason and custom_reason.strip():
                                            custom_res.rationale = custom_reason.strip()

                                        if "approved_resolutions" not in st.session_state:
                                            st.session_state["approved_resolutions"] = []
                                        if not any(getattr(r, "conflict_id", "") == custom_res.conflict_id or getattr(r, "resolution_id", "") == custom_res.resolution_id for r in st.session_state["approved_resolutions"]):
                                            st.session_state["approved_resolutions"].append(custom_res)

                                        # Apply to efs_ir in memory using deterministic pipeline
                                        if efs_ir:
                                            try:
                                                from core.verification.conflict_resolution_engine import ConflictResolutionEngine
                                                efs_ir = ConflictResolutionEngine.apply_approved_resolution(efs_ir, custom_res)
                                                results["efs_ir"] = efs_ir
                                                results["conflicts"] = efs_ir.conflicts
                                                st.session_state["resolved_efsir"] = efs_ir
                                            except Exception:
                                                pass

                                        # Check remaining active unresolved conflicts from authoritative EFS IR
                                        current_conf_pool = getattr(efs_ir, "conflicts", []) if efs_ir else unresolved_conflicts
                                        remaining = [c for c in current_conf_pool if not is_conflict_resolved(c)]

                                        if remaining:
                                            results["conflicts"] = remaining
                                            st.session_state["design_results"] = results
                                            st.success(f"✓ Conflict '{conf.title or conf.conflict_id}' resolved. ({len(remaining)} pending)")
                                            st.rerun()
                                        else:
                                            with st.spinner("All conflicts resolved! Resuming PlantUML and RTL generation..."):
                                                from core.agents.orchestrator import resume_design_flow_with_resolutions
                                                _res_query = (
                                                    st.session_state.get("user_target_query")
                                                    or st.session_state.get("last_active_query")
                                                    or results.get("user_query")
                                                    or default_query_text
                                                )
                                                res_results = resume_design_flow_with_resolutions(
                                                    results,
                                                    st.session_state["approved_resolutions"],
                                                    target_outputs=targets,
                                                    user_query=_res_query,
                                                    generation_mode=generation_mode,
                                                    k_candidates=k_candidates,
                                                    allow_partial=allow_partial
                                                )
                                                st.session_state["design_results"] = res_results
                                                if "outputs" in res_results:
                                                    st.session_state["generated_rtl"] = res_results["outputs"]
                                                if "efs_ir" in res_results and res_results["efs_ir"]:
                                                    st.session_state["resolved_efsir"] = res_results["efs_ir"]
                                                if "plan" in res_results:
                                                    st.session_state["design_plan"] = res_results["plan"]
                                                st.session_state["generation_status"] = res_results.get("status", "SUCCESS")
                                                st.rerun()

            # Resolution Provenance & Audit Log
            if st.session_state.get("approved_resolutions"):
                with st.expander("📜 Resolution Audit Log & Provenance", expanded=False):
                    hist_rows = []
                    for res in st.session_state["approved_resolutions"]:
                        hist_rows.append({
                            "Resolution ID": getattr(res, "resolution_id", "-"),
                            "Conflict ID": getattr(res, "conflict_id", "-"),
                            "Entity": ", ".join(res.affected_requirements) if getattr(res, "affected_requirements", None) else "-",
                            "Original Value": str(getattr(res, "original_value", "-")),
                            "Resolved Value": str(getattr(res, "proposed_value", "-")),
                            "Resolved By": getattr(res, "approved_by", "-"),
                            "Validation": getattr(res, "validation_status", "PASSED"),
                            "Reason": getattr(res, "rationale", "-")[:60]
                        })
                    st.dataframe(hist_rows, use_container_width=True, hide_index=True)

            if st.button("🔄 Reset Pipeline Session", use_container_width=True):
                if "design_results" in st.session_state:
                    del st.session_state["design_results"]
                if "approved_resolutions" in st.session_state:
                    del st.session_state["approved_resolutions"]
                st.rerun()
            return

        if results.get("status") == "INSUFFICIENT_SPEC":
            st.error("⚠️ DESIGN GENERATION STOPPED: Genuine Missing or Ambiguous Hardware Specification Requirements.")
            
            # Retrieve primary output error report
            primary_target = results.get("performance_metrics", {}).get("primary_target", "Verilog")
            out_info = results.get("outputs", {}).get(primary_target, {})
            final_code = out_info.get("final_code", "")
            
            if isinstance(final_code, str) and final_code.startswith("SPECIFICATION_INCOMPLETE"):
                st.markdown(final_code)
            else:
                st.warning("The specification is missing critical hardware details such as explicit FSM transition conditions or port definitions.")
                
            # Display Structured Requirements Classification Table
            req_model = results.get("requirement_model", {})
            issues_list = []
            if isinstance(req_model, dict) and "issues" in req_model:
                issues_list = req_model.get("issues", [])
            elif results.get("efs_ir_validation_issues"):
                issues_list = results.get("efs_ir_validation_issues", [])

            if issues_list:
                st.markdown("### 🔍 Requirement Validation & Classification Report")
                badge_map = {
                    "MISSING": "🔴 [MISSING]",
                    "SOURCE_MISSING": "🔴 [SOURCE MISSING]",
                    "EFSIR_EXTRACTION_GAP": "🟠 [EXTRACTION GAP]",
                    "GROUNDING_GAP": "🟡 [GROUNDING GAP]",
                    "TRACEABILITY_GAP": "🟡 [TRACEABILITY GAP]",
                    "VALIDATION_INTERNAL_ERROR": "⚠️ [INTERNAL ERROR]",
                    "CONFLICT": "🟠 [CONFLICT]",
                    "AMBIGUOUS": "🟡 [AMBIGUOUS]",
                    "COMPATIBLE": "🟢 [COMPATIBLE]",
                    "DERIVED": "🔵 [DERIVED]",
                    "IMPLEMENTATION_CHOICE": "⚙️ [IMPLEMENTATION CHOICE]",
                    "UNSUPPORTED": "🟣 [UNSUPPORTED]"
                }
                table_data = []
                conflicts_to_display = []

                for iss in issues_list:
                    cls = iss.get("classification") or ("GROUNDING_GAP" if iss.get("category") == "Grounding" else "MISSING")
                    badge = badge_map.get(cls, f"🔴 [{cls}]")
                    req_issue = iss.get("issue") or iss.get("message") or "Unspecified issue"
                    provenance = iss.get("evidence") or iss.get("user_spec_evidence") or (f"Object ID: {iss.get('object_id')}" if iss.get("object_id") else "N/A")
                    clarification = iss.get("required_information") or "None"
                    
                    table_data.append({
                        "Classification": badge,
                        "Category": iss.get("category", "GENERAL"),
                        "Requirement / Issue": req_issue,
                        "Source / Provenance": provenance,
                        "Required Clarification": clarification
                    })
                    if cls == "CONFLICT" or "user_spec_evidence" in iss:
                        conflicts_to_display.append(iss)

                if table_data:
                    st.dataframe(table_data, use_container_width=True)

                if conflicts_to_display:
                    st.markdown("#### 🟠 Detailed Conflict Analysis")
                    for c_iss in conflicts_to_display:
                        with st.expander(f"Conflict: {c_iss.get('issue', 'Interface/Protocol Conflict')}", expanded=True):
                            col_u, col_e = st.columns(2)
                            with col_u:
                                st.error("**USER SPECIFICATION:**")
                                st.code(c_iss.get("user_spec_evidence", c_iss.get("evidence", "Unspecified")))
                            with col_e:
                                st.warning("**EFS / PROTOCOL KNOWLEDGE:**")
                                st.code(c_iss.get("efs_protocol_evidence", c_iss.get("required_information", "Unspecified")))
                            st.info(f"**Required Action:** {c_iss.get('required_information', 'Review specification')}")
                
            if st.button("❌ Clear Design State", use_container_width=True):
                del st.session_state["design_results"]
                st.rerun()
            return

        # Display Context Compilation Summary & Inferred Architectural Decisions
        completed_model = results.get("completed_model")
        inferred_decisions = []
        if completed_model:
            if hasattr(completed_model, "inferred_decisions"):
                inferred_decisions = completed_model.inferred_decisions
            elif isinstance(completed_model, dict):
                inferred_decisions = completed_model.get("inferred_decisions", [])

        if inferred_decisions:
            st.divider()
            with st.expander("⚙️ Inferred Architectural Decisions & Safe Defaults", expanded=False):
                st.markdown("### ⚙️ Controlled Specification Inference Decisions")
                inf_table = []
                badge_map = {"DERIVED": "🔵 [DERIVED]", "IMPLEMENTATION_CHOICE": "⚙️ [IMPLEMENTATION CHOICE]"}
                for dec in inferred_decisions:
                    d_dict = dec.to_dict() if hasattr(dec, "to_dict") else dec
                    inf_table.append({
                        "Category / Type": badge_map.get(d_dict.get("decision_type"), d_dict.get("decision_type", "DERIVED")),
                        "Element": d_dict.get("target_element", ""),
                        "Decision / Action": d_dict.get("description", ""),
                        "Rationale": d_dict.get("rationale", ""),
                        "Evidence": d_dict.get("evidence", "N/A")
                    })
                st.dataframe(inf_table, use_container_width=True)

        cp = results.get("context_pack")
        if cp:
            st.divider()
            st.subheader("🎯 Query-Driven Context Compilation Summary")
            
            stats = cp.to_dict().get("stats", {})
            task_info = cp.task
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Resolved Target", task_info.target or "None")
            c2.metric("Task Type", task_info.task)
            c3.metric("Selected EFS Objects", f"{stats.get('selected_efs_objects', 0)} / {stats.get('total_full_efs_objects', 0)}")
            c4.metric("Est. ContextPack Tokens", f"{stats.get('estimated_tokens', 0)}")

            with st.expander("📜 Retrieved Protocol Chunks & Traceable Rules", expanded=False):
                st.markdown("### 📜 Target-Specific Retrieved Protocol Rules & Source Chunks")
                cp_dict = cp.to_dict()
                proto_rules = cp_dict.get("relevant_protocol_rules", [])
                proto_chunks = cp_dict.get("relevant_protocol_chunks", []) or cp_dict.get("protocol_source_evidence", [])

                if proto_rules:
                    st.markdown(f"**Retrieved Protocol Rules ({len(proto_rules)} rules):**")
                    for i, r in enumerate(proto_rules, 1):
                        doc_id = r.get("protocol_document_id") or r.get("document_id") or "protocol_spec"
                        chunk_id = r.get("chunk_id", "N/A")
                        page = r.get("page", "?")
                        sec = r.get("section", "")
                        st.info(f"**Rule {i}** `[Doc: {doc_id} | Sec: {sec} | Page: {page} | Chunk: {chunk_id}]`\n\n{r.get('text', '')}")
                else:
                    st.write("No specific protocol rules mapped for this target.")

                if proto_chunks:
                    st.markdown(f"**Retrieved Protocol Chunks ({len(proto_chunks)} dynamic chunks):**")
                    for i, chk in enumerate(proto_chunks, 1):
                        doc_id = chk.get("protocol_document_id") or chk.get("document_id") or "protocol_spec"
                        chunk_id = chk.get("chunk_id", "N/A")
                        page = chk.get("page", "?")
                        sec = chk.get("section", "")
                        score = chk.get("relevance_score", 1.0)
                        with st.container(border=True):
                            st.caption(f"Chunk #{i} — `Doc: {doc_id}` | `Section: {sec}` | `Page: {page}` | `Chunk ID: {chunk_id}` | `Relevance Score: {score:.2f}`")
                            st.text_area(f"Original Text (Chunk {chunk_id})", chk.get("original_text") or chk.get("text", ""), height=100, key=f"proto_chk_{i}_{chunk_id}")

            with st.expander("🔍 Context Preview (LLM Payload Inspector)", expanded=False):
                st.markdown("This preview demonstrates the exact task-specific ContextPack payload sent to the LLM generation agent:")
                st.code(cp.to_formatted_prompt_text(), language="markdown")
                st.json(cp.to_dict())
            
        metrics = results.get("performance_metrics", {})
        if metrics:
            st.divider()
            st.subheader("📊 Performance & Execution Metrics")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("LLM API Calls", metrics.get("llm_calls_count", 0))
            m2.metric("Static Rule Checks", metrics.get("static_checks_count", 0))
            m3.metric("Final Compliance Score", f"{metrics.get('compliance_score', 0)}%")
            m4.metric("Total Iterations Run", f"{metrics.get('total_iterations', 0)}")
            
            with st.expander("⏳ Phase Execution Timings", expanded=False):
                for phase, duration in metrics.get("timings", {}).items():
                    st.write(f"- **{phase}:** {duration:.2f} seconds")
                    
        st.divider()
        st.subheader("👨‍💻 Design Review & Actions")
        st.info("💡 A design is active in this session. You can browse all generated artifacts using the tabs in the sidebar: **Design Plan**, **Generated RTL**, **PlantUML Diagrams**, **Verification**, and **Repair History**.")

        # Download Canonical JSON option if available
        efs_ir_res = results.get("efs_ir")
        can_str = ""
        if efs_ir_res:
            try:
                can_str = json.dumps(efs_ir_res.to_dict() if hasattr(efs_ir_res, "to_dict") else efs_ir_res, indent=2)
            except Exception:
                pass
        if not can_str:
            ir_path = os.path.join(CONFIG.data_dir, "efs_ir", "active_design_ir.json")
            if os.path.exists(ir_path):
                try:
                    with open(ir_path, "r", encoding="utf-8") as f_act:
                        can_str = f_act.read()
                except Exception:
                    pass
        if can_str:
            st.download_button(
                label="📄 Download Canonical EFS IR (JSON)",
                data=can_str,
                file_name="canonical_efs_ir.json",
                mime="application/json",
                key="download_canonical_json_hw_design"
            )

        col_app, col_rej, col_ref = st.columns(3)
        if col_app.button("✅ Approve Design", use_container_width=True, type="primary"):
            st.success("Design approved! Use the sidebar pages to download files.")
        if col_rej.button("❌ Reject / Clear Design", use_container_width=True):
            del st.session_state["design_results"]
            st.warning("Active design cleared from session.")
            st.rerun()
        if col_ref.button("🔄 Request Refinement", use_container_width=True):
            st.info("Apply your instruction in the 'Refinement Prompt' in the sidebar and click 'Start Generation Flow' to refine.")

