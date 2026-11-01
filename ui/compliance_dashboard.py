"""Compliance Dashboard Page: view and inspect compliance reports, checks, and violations."""

from __future__ import annotations

import streamlit as st
from utils.logger import get_logger

logger = get_logger("ui.compliance_dashboard")


def render() -> None:
    st.title("🛡️ Compliance & Verification Dashboard")
    st.caption("Inspect protocol compliance metrics, code check results, and detailed violation parameters.")

    if "design_results" not in st.session_state:
        st.info("💡 No active design results found in this session. Go to the **Hardware Design** page to specify requirements and generate a protocol-compliant design first.")
        return

    results = st.session_state["design_results"]
    outputs = results["outputs"]

    st.subheader("Design Verification Summaries")
    
    # Render multiple tabs for each generated collateral format
    out_keys = list(outputs.keys())
    tabs = st.tabs(out_keys)
    
    for idx, key in enumerate(out_keys):
        tab = tabs[idx]
        out_data = outputs[key]
        report = out_data["report"]
        score = out_data["compliance_score"]
        
        with tab:
            # Metric indicators
            passed_checks = report.get("passed_checks", [])
            failed_checks = report.get("failed_checks", [])
            warnings = report.get("warnings", [])
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Compliance Score", f"{score}%")
            m2.metric("Passed Checks", len(passed_checks))
            m3.metric("Violations/Failed", len(failed_checks))
            m4.metric("Warnings", len(warnings))
            
            # Interactive layout
            col_left, col_right = st.columns([1, 1.5])
            
            with col_left:
                st.markdown("### ❌ Active Violations")
                if not failed_checks:
                    st.success("✅ No violations detected! This target is fully compliant.")
                else:
                    selected_violation_idx = None
                    for v_idx, f in enumerate(failed_checks):
                        # Card representation of the violation
                        with st.container(border=True):
                            st.markdown(f"**Check:** {f['check_name']}")
                            st.markdown(f"*{f['violation']}*")
                            if st.button("🔍 Inspect Violation Details", key=f"inspect_{key}_{v_idx}"):
                                st.session_state[f"selected_violation_{key}"] = v_idx
                                
                    selected_violation_idx = st.session_state.get(f"selected_violation_{key}", 0)
                    if selected_violation_idx >= len(failed_checks):
                        selected_violation_idx = 0

            with col_right:
                st.markdown("### 🔍 Violation Inspector")
                if not failed_checks:
                    st.caption("No issues to inspect. All checks passed!")
                elif selected_violation_idx is not None and selected_violation_idx < len(failed_checks):
                    violation = failed_checks[selected_violation_idx]
                    
                    with st.container(border=True):
                        st.markdown(f"### Check: `{violation['check_name']}`")
                        st.divider()
                        st.markdown(f"**Violation Description:**\n{violation['violation']}")
                        st.markdown(f"**Protocol Rule Reference:**\n> {violation['protocol_rule']}")
                        st.markdown(f"**Design Spec Reference:** `{violation['spec_section']}`")
                        
                        st.markdown("**Violating Code Snippet:**")
                        st.code(violation["rtl_snippet"], language="verilog")
                        
                        st.markdown("**Suggested Correction:**")
                        st.code(violation["suggested_correction"], language="verilog")
                else:
                    st.caption("Select a violation to inspect its properties.")

            # List of passed checks
            st.markdown("---")
            col_pass, col_warn = st.columns(2)
            
            with col_pass:
                st.markdown("### 🟢 Passed Checks")
                if not passed_checks:
                    st.caption("No passed checks reported.")
                else:
                    for p in passed_checks:
                        st.markdown(f"**✔️ {p['check_name']}** — *{p['description']}*")
            
            with col_warn:
                st.markdown("### ⚠️ Warnings")
                if not warnings:
                    st.caption("No warnings reported.")
                else:
                    for w in warnings:
                        st.markdown(f"- {w}")
