"""Testbench and Assertions Page: display generated verification testbenches and SVA assertions."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("🧪 Testbenches & Assertions")
    st.caption("Browse and export generated SystemVerilog UVM test environments and SVA check assertions.")

    if "design_results" not in st.session_state:
        st.info("💡 No active design results found in this session. Head to the **Hardware Design** page to start a new generation.")
        return

    results = st.session_state["design_results"]
    outputs = results["outputs"]

    targets = [t for t in ["UVM", "Assertions"] if t in outputs]

    if not targets:
        st.warning("⚠️ No verification targets (UVM, Assertions) were selected during the generation session. Go back to the **Hardware Design** page to regenerate them.")
        return

    tabs = st.tabs(targets)

    for idx, target in enumerate(targets):
        with tabs[idx]:
            out_data = outputs[target]
            code = out_data["final_code"]
            
            st.subheader(f"Verification Output: {target}")
            st.markdown(f"**Compliance verification score:** `{out_data['compliance_score']}%` (computed at iteration `{out_data['iterations_run']}`)")
            
            exts = {"UVM": "sv", "Assertions": "sv"}
            prefix = "uvm_tb" if target == "UVM" else "assertions"
            filename = f"{prefix}.{exts.get(target, 'sv')}"
            
            st.download_button(
                label=f"💾 Download {filename}",
                data=code,
                file_name=filename,
                mime="text/plain",
                key=f"download_{target}"
            )
            
            st.code(code, language="verilog")
