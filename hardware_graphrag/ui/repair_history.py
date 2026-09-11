"""Repair History Page: browse design iterations and repair logs."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("📜 Repair History & Version Iterations")
    st.caption("Track the self-correction progress and compare design iterations across the generation pipeline.")

    if "design_results" not in st.session_state:
        st.info("💡 No active design results found in this session. Head to the **Hardware Design** page to start a new generation.")
        return

    results = st.session_state["design_results"]
    outputs = results["outputs"]

    out_keys = list(outputs.keys())
    
    if not out_keys:
        st.warning("⚠️ No generation targets were output in the design results.")
        return

    selected_target = st.selectbox("Select Target Collateral to View History", out_keys)
    out_data = outputs[selected_target]
    history = out_data.get("history", [])

    if not history:
        st.info("No iterations history recorded for this target.")
        return

    st.divider()
    
    # Render loop iterations
    iteration_labels = [f"Design v{item['iteration'] + 1} (Score: {item['score']}%)" for item in history]
    selected_iter_label = st.radio("Navigate Iterations", iteration_labels, horizontal=True)
    selected_idx = iteration_labels.index(selected_iter_label)
    
    iter_data = history[selected_idx]
    
    st.markdown(f"### Design version: `v{iter_data['iteration'] + 1}`")
    st.markdown(f"**Timestamp:** `{iter_data.get('timestamp', 'N/A')}` | **Compliance Score:** `{iter_data['score']}%`")

    # Display repairs applied to reach this iteration (recorded in the previous loop step)
    st.markdown("#### Repairs Applied to Reach this State:")
    repairs = iter_data.get("repairs_applied", [])
    if not repairs:
        st.success("No repairs applied (this is the initial design iteration or it passed compliance immediately).")
    else:
        for idx, repair in enumerate(repairs):
            with st.container(border=True):
                st.markdown(f"**Repair #{idx+1}: {repair.get('check_name')}**")
                st.write(f"**What Changed:** {repair.get('what_changed')}")
                st.write(f"**Why Changed:** {repair.get('why_changed')}")
                st.write(f"**Rule Mapped:** `{repair.get('protocol_rule_or_spec')}`")

    st.divider()
    
    # Download code for this iteration
    st.download_button(
        label=f"💾 Download Design v{iter_data['iteration'] + 1} Code",
        data=iter_data["code"],
        file_name=f"design_v{iter_data['iteration'] + 1}.sv",
        mime="text/plain",
        key=f"dl_iter_{selected_target}_{selected_idx}"
    )

    st.code(iter_data["code"], language="verilog")
