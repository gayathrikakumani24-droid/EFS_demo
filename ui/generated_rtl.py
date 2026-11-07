"""Generated RTL Page: browse and download generated HDL code files."""

from __future__ import annotations

import json
import os
import streamlit as st
from config import CONFIG


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
            readiness = out_data.get("readiness_status", "RTL_GENERATED_UNVERIFIED")
            score = out_data.get("compliance_score")
            iterations = out_data.get("iterations_run", 0)
            
            st.subheader(f"{target} Implementation")
            
            if readiness == "RTL_VERIFIED":
                st.success(f"✅ **Readiness Status: RTL_VERIFIED** — Passed verification checks (Score: `{score}%` at iteration `{iterations}`)")
            elif readiness == "PARTIAL":
                st.warning("⚡ **Readiness Status: PARTIAL** — Compliant partial subsets generated safely for incomplete specification.")
            elif readiness == "RTL_REJECTED":
                st.error(f"❌ **Readiness Status: RTL_REJECTED** — Candidate failed mandatory verification checks (Score: `{score}%`).")
            elif readiness == "RTL_GENERATED_UNVERIFIED":
                if score is not None:
                    st.warning(f"⚠️ **Readiness Status: RTL_GENERATED_UNVERIFIED** — Generated RTL has unverified items (Score: `{score}%`)")
                else:
                    st.info("ℹ️ **Readiness Status: RTL_GENERATED_UNVERIFIED** — RTL generated; automated verification was not executed.")
            elif readiness in ("SPECIFICATION_INCOMPLETE", "PLAN_INVALID", "GENERATION_FAILED"):
                st.error(f"🔴 **Readiness Status: {readiness}** — RTL synthesis was blocked due to incomplete specification or invalid implementation plan.")
            else:
                st.markdown(f"**Readiness Status:** `{readiness}`")

            # Check if code is a specification diagnostic rather than synthesizable HDL
            is_incomplete = isinstance(code, str) and (code.startswith("SPECIFICATION_INCOMPLETE") or code.startswith("INSUFFICIENT SPECIFICATION"))

            exts = {"Verilog": "v", "SystemVerilog": "sv", "VHDL": "vhd"}
            filename = f"diagnostics_{target.lower()}.txt" if is_incomplete else f"design.{exts.get(target, 'txt')}"
            
            # Check if canonical IR is available to provide download option
            canonical_json_str = ""
            efs_ir_obj = results.get("efs_ir")
            if efs_ir_obj:
                try:
                    canonical_json_str = json.dumps(efs_ir_obj.to_dict() if hasattr(efs_ir_obj, "to_dict") else efs_ir_obj, indent=2)
                except Exception:
                    pass
            if not canonical_json_str:
                ir_path = os.path.join(CONFIG.data_dir, "efs_ir", "active_design_ir.json")
                if os.path.exists(ir_path):
                    try:
                        with open(ir_path, "r", encoding="utf-8") as f_ir:
                            canonical_json_str = f_ir.read()
                    except Exception:
                        pass

            if canonical_json_str:
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        label=f"💾 Download {filename}",
                        data=code,
                        file_name=filename,
                        mime="text/plain",
                        key=f"download_{target}",
                        use_container_width=True
                    )
                with col_dl2:
                    st.download_button(
                        label="📄 Download Canonical JSON",
                        data=canonical_json_str,
                        file_name="canonical_ir.json",
                        mime="application/json",
                        key=f"download_canonical_{target}",
                        use_container_width=True
                    )
            else:
                st.download_button(
                    label=f"💾 Download {filename}",
                    data=code,
                    file_name=filename,
                    mime="text/plain",
                    key=f"download_{target}"
                )
            
            if is_incomplete:
                st.text_area("Specification Incompleteness Report", code, height=300)
            else:
                st.code(code, language="verilog" if target != "VHDL" else "vhdl")

                # Comprehensive Technical Architecture Documentation
                summary_text = (
                    out_data.get("summary")
                    or results.get("rtl_summary")
                )
                if not summary_text:
                    try:
                        from core.agents.rtl_summary_generator import generate_comprehensive_rtl_summary
                        summary_text = generate_comprehensive_rtl_summary(
                            code=code,
                            efs_ir=results.get("efs_ir"),
                            rtl_plan=out_data.get("plan"),
                            architecture_classification=out_data.get("architecture_classification"),
                            verification_report=out_data.get("report"),
                            target_lang=target
                        )
                    except Exception:
                        pass
                if summary_text:
                    with st.expander("📑 Comprehensive Technical Architecture Documentation", expanded=False):
                        st.markdown(summary_text)

                # Expandable RTL Generation Plan Inspector
                plan_obj = out_data.get("plan") or results.get("rtl_generation_plan")
                if plan_obj and isinstance(plan_obj, dict):
                    with st.expander("📐 View RTL Generation Plan & Directives", expanded=False):
                        st.caption("Deterministic implementation contract enforced for this target HDL code.")
                        st.json(plan_obj)

                # Deterministic Post-Generation Contract Validation Report
                contract_val = out_data.get("contract_validation")
                if contract_val and isinstance(contract_val, dict):
                    is_compliant = contract_val.get("is_compliant", True)
                    val_title = "🛡️ Contract Validation: PASS" if is_compliant else "🛡️ Contract Validation: VIOLATIONS DETECTED"
                    with st.expander(val_title, expanded=not is_compliant):
                        if is_compliant:
                            st.success("✅ Synthesized HDL strictly complies with the RTL Generation Plan contract (Ports, Registers, and Logic Directives verified).")
                        else:
                            st.error("❌ Discrepancies detected between synthesized HDL and the RTL Generation Plan contract:")
                            for viol in contract_val.get("violations", []):
                                st.markdown(f"- **[{viol.get('category', 'VIOLATION')}]** {viol.get('issue', str(viol))}")
                        st.json(contract_val)
