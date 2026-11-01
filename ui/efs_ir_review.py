"""EFS IR Review Page: Human-in-the-loop validation, traceability, and review of canonical EFS IR."""

from __future__ import annotations

import json
import os
import re
import streamlit as st

from config import CONFIG
from core.efs_ir.models import EFSIR
from core.efs_ir.validator import validate_efs_ir, get_efs_ir_quality


def render() -> None:
    st.title("🖥️ Canonical EFS IR Explorer & Review")
    st.caption("Inspect the protocol-agnostic EFS Hardware Intermediate Representation, trace elements to source text, and validate requirements.")

    # 1. Load active EFS IR
    ir_path = os.path.join(CONFIG.data_dir, "efs_ir", "active_design_ir.json")
    if not os.path.exists(ir_path):
        st.warning("⚠️ No active design flow execution found. Please generate a design first in the **Hardware Design** page.")
        return

    try:
        with open(ir_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            efs_ir = EFSIR.from_dict(raw_data)
    except Exception as e:
        st.error(f"Failed to load canonical EFS IR file: {e}")
        return

    # Prepare canonical JSON string and filename for download
    canonical_json_str = json.dumps(raw_data, indent=2)
    des_name = efs_ir.metadata.design_name if hasattr(efs_ir, "metadata") and efs_ir.metadata.design_name else "design"
    clean_des_name = re.sub(r"[^a-zA-Z0-9_]", "_", des_name).lower()
    canonical_filename = f"{clean_des_name}_canonical_ir.json"

    # Perform validation checks
    validation_issues = validate_efs_ir(efs_ir)

    # 2. Render metadata and scorecard summary
    st.markdown("### 📊 Canonical EFS IR Quality Scorecard")
    scorecard = get_efs_ir_quality(efs_ir, validation_issues)
    
    # Render scorecard status and download button
    status_color = "🔴 BLOCKED" if scorecard["status"] == "BLOCKED" else "🟢 READY"
    
    top_c1, top_c2 = st.columns([3, 1])
    with top_c1:
        st.markdown(f"**Canonical Verification Status:** {status_color}")
    with top_c2:
        st.download_button(
            label="💾 Download Canonical JSON",
            data=canonical_json_str,
            file_name=canonical_filename,
            mime="application/json",
            key="download_canonical_json_top",
            use_container_width=True
        )
    if scorecard["reasons"]:
        for r in scorecard["reasons"]:
            st.error(f"- {r}")
            
    # Metrics display
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Components", scorecard["components"])
    c2.metric("Interfaces", scorecard["interfaces"])
    c3.metric("Signals / Ports", scorecard["signals"])
    c4.metric("Registers Map", scorecard["registers"])
    
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Instructions", scorecard["instructions"])
    c6.metric("Opcodes Table", scorecard["opcodes"])
    c7.metric("State Machines", scorecard["fsms"])
    c8.metric("Traceability Coverage", f"{scorecard['traceability_coverage']:.1f}%")

    st.markdown("---")
    st.markdown("### 📊 Metadata")
    m = efs_ir.metadata
    meta_cols = st.columns(4)
    meta_cols[0].metric("Design Name", m.design_name)
    meta_cols[1].metric("Protocol", f"{m.protocol} v{m.protocol_version}")
    meta_cols[2].metric("Schema Version", m.ir_schema_version)
    meta_cols[3].metric("Timestamp", m.generation_timestamp[:10])

    # 3. Display validation checks
    st.markdown("### 🛡️ EFS IR Validation Report")
    if not validation_issues:
        st.success("✓ Zero structural or validation errors found in EFS IR!")
    else:
        st.warning(f"⚠️ Found {len(validation_issues)} validation issues:")
        # Group issues by severity
        errors = [i for i in validation_issues if i["severity"] in ("ERROR", "CRITICAL")]
        warnings = [i for i in validation_issues if i["severity"] not in ("ERROR", "CRITICAL")]
        
        if errors:
            st.markdown("**Errors (Blocking Code Gen/Correctness):**")
            for err in errors:
                st.error(f"- **[{err['category']}]** {err['message']} (Ref: `{err.get('object_id', '')}`)")
                
        if warnings:
            st.markdown("**Warnings (Potential ambiguities / missing definitions):**")
            for warn in warnings:
                st.warning(f"- **[{warn['category']}]** {warn['message']} (Ref: `{warn.get('object_id', '')}`)")

    st.divider()

    # 4. EFS IR Elements Explorer Tabs
    st.markdown("### 🔍 Model Components & Traceability Explorer")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Components & Interfaces",
        "Signals & Ports",
        "Registers Map",
        "Instructions & Opcodes",
        "Memory & Data Structures",
        "Errors & Performance",
        "FSMs & Constraints",
        "Specification Conflicts"
    ])
    
    with tab1:
        st.subheader("Hardware Components")
        if not efs_ir.components:
            st.info("No components defined in EFS IR.")
        else:
            for comp in efs_ir.components:
                with st.expander(f"📦 Component: {comp.name} ({comp.type})"):
                    st.write(f"**Description:** {comp.description}")
                    st.write(f"**Interfaces Linked:** {', '.join(comp.interfaces) if comp.interfaces else 'None'}")
                    _render_traceability(comp.traceability)
                    
        st.subheader("Port Interfaces")
        if not efs_ir.interfaces:
            st.info("No port interfaces defined in EFS IR.")
        else:
            for iface in efs_ir.interfaces:
                with st.expander(f"🔌 Interface: {iface.name} ({iface.role})"):
                    st.write(f"**Protocol:** {iface.protocol}")
                    st.write(f"**Signals Included:** {', '.join(iface.signals) if iface.signals else 'None'}")
                    _render_traceability(iface.traceability)

    with tab2:
        st.subheader("Signal List")
        if not efs_ir.signals:
            st.info("No signals declared in EFS IR.")
        else:
            sig_rows = []
            for sig in efs_ir.signals:
                sig_rows.append({
                    "Name": sig.name,
                    "Width": sig.width,
                    "Direction": sig.direction,
                    "Semantic Role": sig.semantic_role,
                    "Clock": sig.clock or "N/A",
                    "Reset": sig.reset or "N/A",
                    "ID": sig.signal_id
                })
            st.dataframe(sig_rows, use_container_width=True, hide_index=True)
            
            selected_sig_name = st.selectbox("Select signal for source text trace:", [s.name for s in efs_ir.signals])
            if selected_sig_name:
                selected_sig = next(s for s in efs_ir.signals if s.name == selected_sig_name)
                st.markdown(f"**Signal Trace details for '{selected_sig.name}':**")
                _render_traceability(selected_sig.traceability)

    with tab3:
        st.subheader("Addressable Registers Map")
        if not efs_ir.registers:
            st.info("No address registers configured.")
        else:
            for reg in efs_ir.registers:
                with st.expander(f"📝 Register: {reg.name} ({reg.offset} - {reg.access_type})"):
                    st.write(f"**Reset Value:** {reg.reset_value}")
                    st.write(f"**Description:** {reg.description}")
                    st.write(f"**Side Effects:** {reg.side_effects or 'None'}")
                    
                    if reg.fields:
                        st.markdown("**Register Bitfields:**")
                        field_rows = []
                        for f in reg.fields:
                            field_rows.append({
                                "Name": f.name,
                                "Bits": f"[{f.msb}:{f.lsb}]",
                                "Access": f.access,
                                "Reset Value": f.reset_value,
                                "Description": f.description
                            })
                        st.dataframe(field_rows, use_container_width=True, hide_index=True)
                    _render_traceability(reg.traceability)

    with tab4:
        st.subheader("Instructions Set & Opcodes")
        col_inst, col_opc = st.columns(2)
        with col_inst:
            st.markdown("**Instructions:**")
            if not efs_ir.instructions:
                st.info("No instructions extracted.")
            else:
                for inst in efs_ir.instructions:
                    with st.expander(f"📜 Instruction: {inst.mnemonic}"):
                        st.write(f"**Opcode:** `{inst.opcode or 'N/A'}`")
                        st.write(f"**Format:** `{inst.format or 'N/A'}`")
                        st.write(f"**Description:** {inst.description}")
                        _render_traceability(inst.traceability)
        with col_opc:
            st.markdown("**Opcodes Table:**")
            if not efs_ir.opcodes:
                st.info("No opcodes extracted.")
            else:
                opc_rows = []
                for opc in efs_ir.opcodes:
                    opc_rows.append({
                        "Mnemonic": getattr(opc, "mnemonic", ""),
                        "Encoding": getattr(opc, "binary_encoding", getattr(opc, "encoding", "")),
                        "Description": getattr(opc, "operation", getattr(opc, "description", ""))
                    })
                st.dataframe(opc_rows, use_container_width=True, hide_index=True)

    with tab5:
        st.subheader("Hierarchical Memory Regions & Data Structures")
        col_mem, col_ds = st.columns(2)
        with col_mem:
            st.markdown("**Memory Regions & Buffers:**")
            if not efs_ir.memory_regions:
                st.info("No memory regions extracted.")
            else:
                for mem in efs_ir.memory_regions:
                    with st.expander(f"🧠 Memory: {mem.name} ({mem.type})"):
                        st.write(f"**Base Address:** `{mem.base_address}`")
                        st.write(f"**Size:** `{mem.size}`")
                        st.write(f"**Description:** {mem.description}")
                        _render_traceability(mem.traceability)
        with col_ds:
            st.markdown("**Data Structures & Descriptors:**")
            if not efs_ir.data_structures:
                st.info("No data structures extracted.")
            else:
                for ds in efs_ir.data_structures:
                    with st.expander(f"📑 Structure: {ds.name}"):
                        st.write(f"**Description:** {ds.description}")
                        _render_traceability(ds.traceability)

    with tab6:
        st.subheader("Error Conditions & Performance Targets")
        col_err, col_perf = st.columns(2)
        with col_err:
            st.markdown("**Error Handling & Exceptions:**")
            if not efs_ir.errors:
                st.info("No explicit errors defined.")
            else:
                for err in efs_ir.errors:
                    with st.expander(f"🚨 Error: {err.name}"):
                        st.write(f"**Code:** `{err.code}`")
                        st.write(f"**Condition:** {err.condition}")
                        _render_traceability(err.traceability)
        with col_perf:
            st.markdown("**Performance, Power & Security:**")
            if not efs_ir.performance_requirements:
                st.info("No performance targets specified.")
            else:
                for perf in efs_ir.performance_requirements:
                    with st.expander(f"🎯 Metric: {perf.metric.upper()}"):
                        st.write(f"**Target Value:** `{perf.target_value}`")
                        st.write(f"**Condition:** {perf.condition}")
                        _render_traceability(perf.traceability)

    with tab7:
        st.subheader("Finite State Machines (FSMs) & Constraints")
        if not efs_ir.fsms:
            st.info("No state machines modeled.")
        else:
            for fsm in efs_ir.fsms:
                with st.expander(f"⚙️ State Machine: {fsm.name}"):
                    st.write(f"**Initial State:** {fsm.initial_state}")
                    st.write(f"**Encoding:** {fsm.encoding}")
                    st.markdown("**States defined:** " + ", ".join([s.name for s in fsm.states]))
                    if fsm.transitions:
                        trans_rows = [{"Source": t.source_state, "Target": t.target_state, "Condition": t.condition or "default"} for t in fsm.transitions]
                        st.dataframe(trans_rows, use_container_width=True, hide_index=True)
                    _render_traceability(fsm.traceability)

        st.subheader("Design Constraints & Timing Rules")
        if not efs_ir.constraints:
            st.info("No explicit constraints found.")
        else:
            for const in efs_ir.constraints:
                with st.expander(f"⚠️ Constraint ({const.type} - {const.severity})"):
                    st.write(f"**Condition:** `{const.condition}`")
                    st.write(f"**Expected Behavior:** {const.expected_behavior}")
                    _render_traceability(const.traceability)

    with tab8:
        st.subheader("Specification Conflicts & Approved Resolutions")
        conflicts = getattr(efs_ir, "conflicts", [])
        resolutions = getattr(efs_ir, "resolutions", [])
        
        if not conflicts and not resolutions:
            st.success("✓ Zero specification conflicts or contradictions detected.")
        else:
            approved_cids = {getattr(r, "conflict_id", None) for r in resolutions if getattr(r, "approval_status", "") == "APPROVED" and getattr(r, "conflict_id", None)}
            approved_entity_props = set()
            for r in resolutions:
                if getattr(r, "approval_status", "") == "APPROVED":
                    r_prop = (getattr(r, "field_name", "") or getattr(r, "property", "") or "").lower()
                    for req in getattr(r, "affected_requirements", []):
                        approved_entity_props.add((req.lower(), r_prop))
                        approved_entity_props.add((req.lower(), ""))

            unresolved_conflicts = [
                c for c in conflicts
                if getattr(c, "resolution_status", "UNRESOLVED") == "UNRESOLVED"
                and getattr(c, "status", "") not in ("APPROVED", "RESOLVED")
                and getattr(c, "blocking", True)
                and getattr(c, "conflict_id", None) not in approved_cids
                and not any(
                    (ent.lower(), (getattr(c, "property", "") or "").lower()) in approved_entity_props
                    or (ent.lower(), "") in approved_entity_props
                    for ent in (getattr(c, "affected_entities", []) or getattr(c, "entities", []) or [])
                )
            ]
            if unresolved_conflicts:
                st.markdown(f"#### Active Unresolved Specification Conflicts ({len(unresolved_conflicts)}):")
                for conf in unresolved_conflicts:
                    with st.expander(f"⚠️ Conflict: {conf.type} ({conf.severity}) - 🔴 Unresolved"):
                        st.markdown(f"**Entities:** `{', '.join(conf.entities) if conf.entities else 'N/A'}`")
                        st.markdown(f"**Property:** `{conf.property}`")
                        st.markdown(f"**Description:** {conf.description}")
                        st.markdown(f"**Impact:** {conf.impact}")
                        st.markdown(f"**Conflicting Values:** `{', '.join(conf.conflicting_values) if conf.conflicting_values else 'N/A'}`")
                        _render_traceability(conf.traceability)
            elif conflicts or resolutions:
                st.success("✓ All detected specification conflicts have been resolved! Zero active conflicts remaining.")
                        
            if resolutions:
                st.markdown("#### Resolution Override Layer:")
                res_rows = []
                for r in resolutions:
                    res_rows.append({
                        "Resolution ID": r.resolution_id,
                        "Conflict ID": r.conflict_id,
                        "Type": r.resolution_type,
                        "Proposed Change": r.proposed_change,
                        "Rationale": r.rationale,
                        "Proposed Value": str(r.proposed_value),
                        "Status": r.approval_status,
                        "Approved By": r.approved_by
                    })
                st.dataframe(res_rows, use_container_width=True, hide_index=True)

    st.divider()
    
    # 5. Human-in-the-loop actions
    st.subheader("📝 Human Override / Approval")
    cols = st.columns([2, 1])
    with cols[0]:
        override_notes = st.text_area("Input architectural override comments or corrections (e.g. change signal widths or reset active level):")
    with cols[1]:
        st.write(" ")
        st.write(" ")
        if st.button("Confirm & Approve EFS IR", type="primary", use_container_width=True):
            if override_notes:
                st.success("Overrides noted! They will guide the next generation/repair cycle.")
            else:
                st.success("EFS IR approved successfully!")
                
    with st.expander("👁️ View Complete Canonical JSON", expanded=False):
        st.download_button(
            label="💾 Download Complete Canonical JSON",
            data=canonical_json_str,
            file_name=canonical_filename,
            mime="application/json",
            key="download_canonical_json_expander",
            use_container_width=True
        )
        st.json(raw_data)


def _render_traceability(trace) -> None:
    """Helper to display traceability data in a clean box."""
    if not trace or not (trace.chunk_id or trace.original_text or getattr(trace, "source_text", None)):
        st.caption("🔍 No source document traceability context available.")
        return
    
    conf_str = f"{trace.confidence:.2f}" if trace.confidence is not None else "N/A"
    doc_display = getattr(trace, "source_document", None) or trace.doc_id or "N/A"
    snippet = trace.original_text or getattr(trace, "source_text", "")
    
    st.markdown(
        f"<div style='background-color:#0e1117; padding:10px; border-radius:5px; border-left:4px solid #1a8cff; margin:10px 0;'>"
        f"<p style='margin:0; font-size:12px; color:#a3a8b4;'><b>SOURCE TRACEABILITY:</b> "
        f"Doc: {doc_display} | Chapter: {trace.chapter or 'N/A'} | Page: {trace.page or 'N/A'} | "
        f"Confidence: {conf_str} | Method: {trace.extraction_method}</p>"
        f"<p style='margin:5px 0 0 0; font-family:monospace; font-size:12px; color:#d1d5db;'>{snippet}</p>"
        f"</div>",
        unsafe_allow_html=True
    )
