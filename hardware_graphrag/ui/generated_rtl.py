"""Generated RTL Page: browse and download generated HDL code files."""

from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("💻 Generated HDL / RTL Files")
    st.caption("View and download synthesizable Verilog, SystemVerilog, and VHDL implementations.")

    if "design_results" not in st.session_state:
        st.info("💡 No active design results found in this session. Head to the **Hardware Design** page to start a new generation.")
        return

    results = st.session_state["design_results"]
    outputs = results["outputs"]

    # Filter outputs that are RTL targets
    rtl_targets = [t for t in ["Verilog", "SystemVerilog", "VHDL"] if t in outputs]

    if not rtl_targets:
        st.warning("⚠️ No RTL outputs (Verilog, SystemVerilog, VHDL) were selected during the generation session. Go back to the **Hardware Design** page to regenerate them.")
        return

    tabs = st.tabs(rtl_targets)

    for idx, target in enumerate(rtl_targets):
        with tabs[idx]:
            out_data = outputs[target]
            code = out_data["final_code"]
            
            st.subheader(f"{target} Implementation")
            st.markdown(f"**Compliance verification score:** `{out_data['compliance_score']}%` (computed at iteration `{out_data['iterations_run']}`)")
            
            # File extension lookup
            exts = {"Verilog": "v", "SystemVerilog": "sv", "VHDL": "vhd"}
            filename = f"design.{exts.get(target, 'txt')}"
            
            # Download button
            st.download_button(
                label=f"💾 Download {filename}",
                data=code,
                file_name=filename,
                mime="text/plain",
                key=f"download_{target}"
            )
            
            st.code(code, language="verilog" if target != "VHDL" else "vhdl")
