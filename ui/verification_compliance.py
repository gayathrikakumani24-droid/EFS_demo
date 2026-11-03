"""Verification & Compliance Page: view compliance reports and inspect checks and violations."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("🛡️ Compliance & Verification Dashboard")
    st.caption("Inspect candidate verification checks, EDA tool compilation, functional tests, and requirement traceability.")

    if "design_results" not in st.session_state:
        st.info("💡 No active design results found in this session. Head to the **Hardware Design** page to start a new generation.")
        return

    results = st.session_state["design_results"]
    outputs = results.get("outputs", {})

    out_keys = list(outputs.keys())
    if not out_keys:
        st.warning("⚠️ No generation targets were output in the design results.")
        return

    selected_target = st.selectbox("Select Target Collateral to Inspect Compliance", out_keys)

    out_data = outputs[selected_target]
    report = out_data.get("report", {})
    score = out_data.get("compliance_score", 0)
    readiness = out_data.get("readiness_status", "UNKNOWN")

    passed_checks = report.get("passed_checks", [])
    failed_checks = report.get("failed_checks", [])
    warnings = report.get("warnings", [])

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Target Name", selected_target)
    col2.metric("Compliance Score", f"{score}%" if score is not None else "N/A")
    col3.metric("Readiness Status", readiness)
    col4.metric("Violations / Failures", len(failed_checks) if isinstance(failed_checks, list) else len(report.get("errors", [])))

    st.divider()

    # -- Section: EDA Tool Compilation & Lint Status --
    st.subheader("🛠️ HDL Compiler & Tool Verification")
    tool_comp = out_data.get("tool_compilation", {}) or report.get("tool_compilation", {})
    if tool_comp and isinstance(tool_comp, dict):
        tool_cols = st.columns(len(tool_comp) if len(tool_comp) > 0 else 1)
        for idx, (tname, tinfo) in enumerate(tool_comp.items()):
            with tool_cols[idx % len(tool_cols)]:
                t_status = tinfo.get("status", "NOT_AVAILABLE")
                t_ver = tinfo.get("version", "N/A")
                if t_status == "PASS":
                    st.success(f"**{tname.upper()}**: PASS (v{t_ver})")
                elif t_status == "FAIL":
                    st.error(f"**{tname.upper()}**: FAIL (v{t_ver})")
                elif t_status == "NOT_AVAILABLE":
                    st.info(f"**{tname.upper()}**: NOT_AVAILABLE (Tool not installed)")
                else:
                    st.warning(f"**{tname.upper()}**: {t_status}")
    else:
        st.info("ℹ️ EDA Tools (Icarus Verilog, Verilator) status: NOT_AVAILABLE on this host environment.")

    st.divider()

    # -- Section: Requirement Traceability Matrix --
    st.subheader("📋 Requirement Traceability Matrix")
    traceability = out_data.get("traceability", []) or report.get("traceability", [])
    if traceability and isinstance(traceability, list):
        badge_status_map = {
            "IMPLEMENTED_AND_VERIFIED": "🟢 IMPLEMENTED_AND_VERIFIED",
            "IMPLEMENTED_NOT_VERIFIED": "🟡 IMPLEMENTED_NOT_VERIFIED",
            "PARTIALLY_IMPLEMENTED": "🟠 PARTIALLY_IMPLEMENTED",
            "NOT_IMPLEMENTED": "🔴 NOT_IMPLEMENTED",
            "BLOCKED_BY_AMBIGUITY": "🟣 BLOCKED_BY_AMBIGUITY",
            "NOT_APPLICABLE": "⚪ NOT_APPLICABLE",
        }
        trace_rows = []
        for item in traceability:
            if isinstance(item, dict):
                st_val = item.get("status", "UNKNOWN")
                trace_rows.append({
                    "Requirement ID": item.get("requirement_id", "REQ"),
                    "Description": item.get("description", "")[:60],
                    "Verification Status": badge_status_map.get(st_val, st_val),
                    "RTL Signals / Logic": ", ".join(item.get("rtl_signals", [])) or "None",
                    "Source Evidence": item.get("evidence", "spec evidence")[:40],
                })
        st.dataframe(trace_rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No requirement traceability items generated for this target.")

    st.divider()

    # -- Section: Candidate Repair History --
    repairs = out_data.get("repair_history", []) or report.get("repair_history", [])
    if repairs:
        st.subheader("🔧 Localized Surgical Recovery History")
        for i, rep in enumerate(repairs, 1):
            if isinstance(rep, dict):
                cat = rep.get("category", "DEFECT")
                act = rep.get("action", "Repair applied")
                diff = rep.get("diff", "")
                with st.expander(f"Repair #{i}: [{cat}] {act}", expanded=False):
                    if diff:
                        st.code(diff, language="diff")
                    else:
                        st.write(str(rep))

    st.divider()

    # -- Section: Violations & Details --
    col_left, col_right = st.columns([1, 1.5])

    with col_left:
        st.subheader("❌ Active Violations / Diagnostics")
        diagnostics = report.get("diagnostics", []) or failed_checks
        if not diagnostics and not report.get("errors"):
            st.success("✅ This candidate satisfies all structural and behavioral checks.")
        else:
            all_errs = list(report.get("errors", []))
            for idx, err in enumerate(all_errs):
                with st.container(border=True):
                    st.markdown(f"**Error #{idx+1}:** {err[:80]}")

    with col_right:
        st.subheader("🔍 Violation Details & Diagnostics")
        diags = report.get("diagnostics", [])
        if diags:
            for idx, d in enumerate(diags):
                with st.container(border=True):
                    if isinstance(d, dict):
                        st.markdown(f"**[{d.get('classification', d.get('category', 'DIAG'))}]** {d.get('issue', str(d))}")
                        if d.get("required_information"):
                            st.caption(f"Required: {d.get('required_information')}")
                    else:
                        st.write(str(d))
        else:
            st.caption("No diagnostic violations to inspect.")
