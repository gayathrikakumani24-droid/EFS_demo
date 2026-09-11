"""Verification & Compliance Page: view compliance reports and inspect checks and violations."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("🛡️ Compliance & Verification Dashboard")
    st.caption("Inspect check logs, protocol warnings, and violation mappings across generated targets.")

    if "design_results" not in st.session_state:
        st.info("💡 No active design results found in this session. Head to the **Hardware Design** page to start a new generation.")
        return

    results = st.session_state["design_results"]
    outputs = results["outputs"]

    out_keys = list(outputs.keys())
    
    if not out_keys:
        st.warning("⚠️ No generation targets were output in the design results.")
        return

    selected_target = st.selectbox("Select Target Collateral to Inspect Compliance", out_keys)
    
    out_data = outputs[selected_target]
    report = out_data["report"]
    score = out_data["compliance_score"]
    
    passed_checks = report.get("passed_checks", [])
    failed_checks = report.get("failed_checks", [])
    warnings = report.get("warnings", [])

    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Target Name", selected_target)
    col2.metric("Compliance Score", f"{score}%")
    col3.metric("Passed Checks", len(passed_checks))
    col4.metric("Violations / Failures", len(failed_checks))

    st.divider()

    col_left, col_right = st.columns([1, 1.5])

    with col_left:
        st.subheader("❌ Active Violations")
        if not failed_checks:
            st.success("✅ This design satisfies all compliance checks.")
        else:
            for idx, f in enumerate(failed_checks):
                with st.container(border=True):
                    st.markdown(f"**Check Name:** {f.get('check_name', 'Unnamed')}")
                    st.caption(f.get("violation", "No detail"))
                    if st.button(f"🔍 Inspect Violation #{idx+1}", key=f"btn_{selected_target}_{idx}"):
                        st.session_state[f"selected_violation_{selected_target}"] = idx
                        
            # Select default violation index
            selected_violation_idx = st.session_state.get(f"selected_violation_{selected_target}", 0)
            if selected_violation_idx >= len(failed_checks):
                selected_violation_idx = 0

    with col_right:
        st.subheader("🔍 Violation Details")
        if not failed_checks:
            st.caption("No violations to inspect.")
        elif selected_violation_idx is not None and selected_violation_idx < len(failed_checks):
            violation = failed_checks[selected_violation_idx]
            
            with st.container(border=True):
                st.markdown(f"### Check: `{violation.get('check_name')}`")
                st.write(f"**Detailed Violation:**\n{violation.get('violation')}")
                st.markdown(f"**Protocol Rule Reference:**\n> {violation.get('protocol_rule')}")
                st.markdown(f"**Design Spec Reference:** `{violation.get('spec_section')}`")
                
                st.markdown("**Violating Code Block:**")
                st.code(violation.get("rtl_snippet", ""), language="verilog")
                
                st.markdown("**Suggested Correction:**")
                st.code(violation.get("suggested_correction", ""), language="verilog")
        else:
            st.caption("Select a violation to view details.")

    st.divider()
    col_passed, col_warnings = st.columns(2)
    
    with col_passed:
        st.subheader("🟢 Passed Checks")
        if not passed_checks:
            st.caption("No passed checks reported.")
        else:
            for check in passed_checks:
                st.markdown(f"**✔️ {check.get('check_name')}**")
                st.write(check.get("description", ""))
                
    with col_warnings:
        st.subheader("⚠️ Warnings")
        if not warnings:
            st.caption("No warnings reported.")
        else:
            for w in warnings:
                st.markdown(f"- {w}")
