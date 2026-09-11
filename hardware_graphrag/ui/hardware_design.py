"""Hardware Design Assistant Page: upload requirements, select protocols, configure options, and trigger generation."""

from __future__ import annotations

import os
import streamlit as st

from config import CONFIG
from core.agents.orchestrator import run_design_flow
from utils.protocol_manager import get_library_data, select_protocol
from utils.logger import get_logger

logger = get_logger("ui.hardware_design")


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

    st.sidebar.markdown("### Loop Configurations")
    max_repair = st.sidebar.slider("Max Repair Iterations", min_value=0, max_value=5, value=0)
    score_threshold = st.sidebar.slider("Compliance Target Score (%)", min_value=50, max_value=100, value=95)
    bypass_conflicts = st.sidebar.checkbox("Bypass Critical Conflicts / Override Block", value=False)

    st.sidebar.markdown("### Refinement Prompt")
    refinement_instruction = st.sidebar.text_area("Guidance for regeneration/refinement:", height=80)

    # Main Panel Wizard
    st.markdown("### STEP 1: Provide User Requirement Specification")
    
    req_cols = st.columns([2, 1])
    with req_cols[0]:
        raw_req_text = st.text_area(
            "Input requirement text directly:",
            value="Design a Custom Acceleration Controller module.\n"
                  "Interfaces:\n"
                  "  - clk: input, 1 bit, System clock\n"
                  "  - rst_n: input, 1 bit, Active-low reset\n"
                  "  - instruction: input, 64 bits, Instruction word\n"
                  "  - start: input, 1 bit, Start trigger\n"
                  "  - busy: output, 1 bit, Busy status\n"
                  "  - done: output, 1 bit, Execution complete",
            height=160
        )
    with req_cols[1]:
        uploaded_doc = st.file_uploader(
            "Or upload requirement spec (PDF/DOCX/TXT/MD):",
            type=["pdf", "docx", "txt", "md"],
            key="req_doc_upload"
        )
        if uploaded_doc:
            try:
                file_ext = os.path.splitext(uploaded_doc.name)[1].lower()
                if file_ext == ".txt":
                    raw_req_text = str(uploaded_doc.read(), "utf-8")
                elif file_ext == ".pdf":
                    try:
                        import pypdf
                        reader = pypdf.PdfReader(uploaded_doc)
                        raw_req_text = "\n".join([p.extract_text() for p in reader.pages])
                    except ImportError:
                        raw_req_text = str(uploaded_doc.read(), "utf-8", errors="ignore")
                elif file_ext == ".docx":
                    try:
                        import docx
                        doc = docx.Document(uploaded_doc)
                        raw_req_text = "\n".join([p.text for p in doc.paragraphs])
                    except ImportError:
                        raw_req_text = str(uploaded_doc.read(), "utf-8", errors="ignore")
                else:
                    raw_req_text = str(uploaded_doc.read(), "utf-8", errors="ignore")
                st.success(f"Parsed content from {uploaded_doc.name}")
            except Exception as e:
                st.error(f"Failed to read file: {e}")

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
    user_query = st.text_input(
        "Targeted Generation Query:",
        value="Generate synthesizable Verilog RTL for MatrixAdder.",
        help="Specify the target component, language, or format requested (e.g., 'Generate synthesizable Verilog RTL for MatrixAdder', 'Generate SVA assertions for AXI interface of MatrixAdder', 'Generate UVM testbench for MatrixMultiplier')."
    )

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
                    user_query=user_query
                )
                
                # Save results to session state
                st.session_state["design_results"] = results
                if results.get("status") in ("TARGET_NOT_FOUND", "AMBIGUOUS_TARGET"):
                    st.error(f"❌ Target Resolution Failed: {results.get('error_message')}")
                elif results.get("status") == "BLOCKED":
                    st.error("❌ Generation BLOCKED due to critical spec conflicts or validation errors. Resolve them or check 'Bypass Critical Conflicts / Override Block' in the sidebar.")
                elif results.get("status") == "INSUFFICIENT_SPEC":
                    st.error("❌ Generation STOPPED: Incomplete or ambiguous specification details.")
                else:
                    st.success("✅ Design flow execution complete!")
                    st.balloons()
            except Exception as e:
                logger.exception("Design flow failed")
                st.error(f"❌ Design flow execution failed: {e}")

    # Display results and Context Preview
    if "design_results" in st.session_state:
        results = st.session_state["design_results"]
        
        if results.get("status") == "SUCCESS":
            st.success("✅ RTL Generated Successfully")
            outputs = results.get("outputs", {})
            primary_target = results.get("performance_metrics", {}).get("primary_target", "Verilog")
            if primary_target in outputs:
                code_text = outputs[primary_target].get("final_code", "")
                if code_text and not code_text.startswith("SPECIFICATION_INCOMPLETE"):
                    with st.expander(f"📄 Generated {primary_target} RTL Code", expanded=True):
                        st.code(code_text, language="systemverilog" if "Verilog" in primary_target else "vhdl")

        if results.get("status") in ("TARGET_NOT_FOUND", "AMBIGUOUS_TARGET"):
            st.error(f"⚠️ TARGET RESOLUTION FAILED: {results.get('error_message')}")
            if st.button("❌ Clear State", use_container_width=True):
                del st.session_state["design_results"]
                st.rerun()
            return

        if results.get("status") in ("BLOCKED", "BLOCKED_CONFLICT", "INSUFFICIENT_SPEC"):
            st.warning("⚠️ Conflict detected")
            st.info("The design pipeline detected hardware specification conflicts or missing requirements that require human decision to proceed.")
            
            conflicts = results.get("conflicts", [])
            proposals_map = results.get("proposals", {})
            efs_ir = results.get("efs_ir")
            req_model = results.get("requirement_model")
            
            if not conflicts and efs_ir:
                from core.verification.conflict_resolution_engine import ConflictResolutionEngine
                conflicts = ConflictResolutionEngine.detect_conflicts(efs_ir, req_model)
                
            unresolved_conflicts = [c for c in conflicts if c.blocking and getattr(c, "resolution_status", "UNRESOLVED") == "UNRESOLVED"]
            
            if unresolved_conflicts:
                st.markdown("### ⚠️ Specification Conflict Detected")
                for conf in unresolved_conflicts:
                    with st.container(border=True):
                        st.markdown(f"#### ⚠️ Conflict: `{conf.type}` ({conf.severity})")
                        st.markdown(f"**Description:** {conf.description or conf.evidence}")
                        if conf.impact:
                            st.markdown(f"**Impact:** {conf.impact}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Conflicting Target Entities:**")
                            st.code(", ".join(conf.entities) if conf.entities else "Unspecified")
                        with col2:
                            st.markdown("**Conflicting Values / Evidence:**")
                            st.code(", ".join(conf.conflicting_values) if conf.conflicting_values else "N/A")

                        st.markdown("#### 🛠️ Choose Resolution Option:")
                        strategy_key = f"strat_{conf.conflict_id}"
                        strategy_choice = st.radio(
                            "Select how to resolve this blocking issue:",
                            options=[
                                "OPTION 1: Allow AI to make changes",
                                "OPTION 2: Add a custom requirement"
                            ],
                            key=strategy_key
                        )

                        if strategy_choice == "OPTION 1: Allow AI to make changes":
                            st.markdown("##### 💡 AI Proposed Resolution:")
                            props = proposals_map.get(conf.conflict_id, [])
                            if not props:
                                from core.verification.conflict_resolution_engine import ConflictResolutionEngine
                                props = ConflictResolutionEngine.generate_resolution_proposals(conf, efs_ir)

                            ai_props = [p for p in props if p.resolution_type in ("ALLOW_AI", "ACCEPT_PROPOSAL", "PROPOSED", "AI_RECOMMENDED")]
                            if not ai_props:
                                ai_props = props

                            for idx, p in enumerate(ai_props, 1):
                                target_entity = p.affected_requirements[0] if p.affected_requirements else (conf.entities[0] if conf.entities else "Entity")
                                with st.container(border=True):
                                    st.markdown(f"**AI Proposed Value:** `{target_entity} → {p.proposed_value}`")
                                    st.markdown(f"**Proposed Change:** `{p.proposed_change}`")
                                    st.markdown(f"**Reason:** {p.rationale}")
                                    
                                    col_app, col_rej = st.columns(2)
                                    with col_app:
                                        if st.button("Approve AI Resolution", key=f"app_prop_{conf.conflict_id}_{idx}", type="primary"):
                                            if "approved_resolutions" not in st.session_state:
                                                st.session_state["approved_resolutions"] = []
                                            st.session_state["approved_resolutions"].append(p)
                                            
                                            from core.agents.orchestrator import resume_design_flow_with_resolutions
                                            res_results = resume_design_flow_with_resolutions(
                                                results,
                                                st.session_state["approved_resolutions"],
                                                target_outputs=targets
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

                                    with col_rej:
                                        if st.button("Reject", key=f"rej_prop_{conf.conflict_id}_{idx}"):
                                            st.warning(f"Proposal rejected by user. Please select Option 2 or enter a custom requirement.")
                                            st.rerun()

                        elif strategy_choice == "OPTION 2: Add a custom requirement":
                            st.markdown("##### 📝 Add Custom Requirement / Specification Override:")
                            custom_input = st.text_area(
                                f"Enter custom requirement description for {conf.property}:",
                                placeholder="Describe explicit hardware requirement or specification override...",
                                key=f"cust_req_{conf.conflict_id}"
                            )
                            if st.button("Confirm Custom Requirement", key=f"sub_cust_{conf.conflict_id}", type="primary"):
                                if custom_input and custom_input.strip():
                                    from core.verification.conflict_resolution_engine import ConflictResolutionEngine
                                    from core.agents.orchestrator import resume_design_flow_with_resolutions

                                    custom_res = ConflictResolutionEngine.create_custom_resolution(
                                        conf,
                                        custom_input.strip(),
                                        user_name="user"
                                    )
                                    if "approved_resolutions" not in st.session_state:
                                        st.session_state["approved_resolutions"] = []
                                    st.session_state["approved_resolutions"].append(custom_res)

                                    res_results = resume_design_flow_with_resolutions(
                                        results,
                                        st.session_state["approved_resolutions"],
                                        target_outputs=targets
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

            else:
                st.info("Check **EFS IR Review** page for detailed validation report, or select **Bypass Critical Conflicts / Override Block** in the sidebar to bypass.")
                st.markdown("#### Blocking Reasons:")
                issues = results.get("efs_ir_validation_issues", [])
                for iss in issues:
                    if iss.get("severity") in ("ERROR", "CRITICAL"):
                        st.error(f"- **[{iss['category']}]** {iss['message']}")
            
            if st.button("❌ Clear Blocked State", use_container_width=True):
                del st.session_state["design_results"]
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
        
        col_app, col_rej, col_ref = st.columns(3)
        if col_app.button("✅ Approve Design", use_container_width=True, type="primary"):
            st.success("Design approved! Use the sidebar pages to download files.")
        if col_rej.button("❌ Reject / Clear Design", use_container_width=True):
            del st.session_state["design_results"]
            st.warning("Active design cleared from session.")
            st.rerun()
        if col_ref.button("🔄 Request Refinement", use_container_width=True):
            st.info("Apply your instruction in the 'Refinement Prompt' in the sidebar and click 'Start Generation Flow' to refine.")

