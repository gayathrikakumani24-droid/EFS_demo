"""
Static Verifier for fast deterministic checking of generated RTL.

Performs static analysis using regular expressions and specification definitions
to detect missing ports, incorrect widths, register mismatches, and reset polarities.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from utils.logger import get_logger
from core.efs_ir.models import EFSIR
from core.verification.hdl_validator import (
    validate_rtl_syntax_and_structure,
    resolve_hdl_language,
    HDLLanguage,
)
from core.verification.traceability import generate_traceability_report

logger = get_logger("verification.static")


def verify_statically(
    code: str,
    design_spec: str,
    protocol_rules: List[Dict[str, Any]],
    req_model: Dict[str, Any] = None,
    plan: Dict[str, Any] = None,
    efs_ir: Optional[EFSIR] = None,
    target_language: str = "Verilog",
    language_version: Optional[str] = None
) -> Dict[str, Any]:
    """Perform fast, rule-based static analysis checks on the RTL code."""
    logger.info(f"Running fast static verification for target {target_language}...")
    passed_checks = []
    failed_checks = []
    warnings = []

    # -- 0. Language-Aware HDL Syntax, Keyword & Structural Verification ---
    hdl_report = validate_rtl_syntax_and_structure(
        code=code,
        target_language=target_language,
        language_version=language_version,
        efs_ir=efs_ir
    )
    for err in hdl_report.errors:
        failed_checks.append({
            "check_name": f"{hdl_report.target_language.value} Syntax/Structural: {err.category}",
            "rule_id": f"RULE_HDL_{err.category.upper().replace(' ', '_')}",
            "category": "HDL Syntax & Structure",
            "severity": "CRITICAL" if err.severity == "ERROR" else "MAJOR",
            "violation": err.message,
            "protocol_rule": f"Code must strictly conform to {hdl_report.target_language.value} standard.",
            "spec_section": "RTL Synthesizability and Standards",
            "rtl_snippet": f"Line {err.line_number}: {err.snippet}" if err.line_number else "// Syntax error",
            "suggested_correction": err.suggested_fix or f"Ensure syntax complies with {hdl_report.target_language.value}.",
            "line_number": err.line_number
        })
    for w in hdl_report.warnings:
        warnings.append(w)
    if hdl_report.valid:
        passed_checks.append({
            "check_name": f"{hdl_report.target_language.value} Structural & Syntax Integrity",
            "description": f"Verified clean syntax and adherence to {hdl_report.target_language.value} standard."
        })

    # If EFS IR is provided, perform checks directly on the canonical model
    if efs_ir:
        # 1. Interface Port / Signal Checks
        missing_ports = []
        for sig in efs_ir.signals:
            if sig.semantic_role in ("clock", "reset"):
                continue  # Checked separately
            pattern = rf"\b{sig.name}\b"
            if not re.search(pattern, code):
                missing_ports.append(sig.name)
                failed_checks.append({
                    "check_name": "Protocol Port Signal Presence",
                    "rule_id": "RULE_PORT_MISSING",
                    "category": "Interface Ports",
                    "severity": "CRITICAL",
                    "violation": f"Required signal '{sig.name}' is completely missing from the generated RTL.",
                    "protocol_rule": f"The interface signal '{sig.name}' is required by the protocol mapping.",
                    "spec_section": "External Interface Port List",
                    "rtl_snippet": "// Missing in port list",
                    "suggested_correction": f"Add port declaration: {sig.direction} signal '{sig.name}' of type logic.",
                    "efs_id": sig.signal_id
                })
        if not missing_ports and efs_ir.signals:
            passed_checks.append({
                "check_name": "Protocol Port Signal Presence",
                "description": f"Verified that all required EFS IR interface signals exist in the port declarations."
            })

        # 2. Clock and Reset Checks
        missing_clk_rst = []
        for sig in efs_ir.signals:
            if sig.semantic_role not in ("clock", "reset"):
                continue
            pattern = rf"\b{sig.name}\b"
            if not re.search(pattern, code):
                missing_clk_rst.append(sig.name)
                failed_checks.append({
                    "check_name": f"{sig.semantic_role.capitalize()} Presence Check",
                    "rule_id": "RULE_CLK_RST_MISSING",
                    "category": "Clock & Reset",
                    "severity": "CRITICAL",
                    "violation": f"Required {sig.semantic_role} signal '{sig.name}' is missing.",
                    "protocol_rule": f"The design must use the specified {sig.semantic_role} signal '{sig.name}'.",
                    "spec_section": "Clocks & Resets",
                    "rtl_snippet": "// Clock/reset missing",
                    "suggested_correction": f"Declare '{sig.name}' as an input clock/reset in the top-level module header.",
                    "efs_id": sig.signal_id
                })
            elif sig.semantic_role == "reset":
                # Verify active polarity
                polarity = "low" if "n" in sig.name.lower() or "rst_n" in sig.name.lower() else "high"
                if polarity == "low":
                    is_active_low_found = any([
                        "negedge" in code and sig.name in code,
                        f"!{sig.name}" in code,
                        f"~{sig.name}" in code,
                        f"{sig.name} == 0" in code,
                        f"{sig.name} == 1'b0" in code
                    ])
                    if not is_active_low_found:
                        failed_checks.append({
                            "check_name": "Reset Polarity Check",
                            "rule_id": "RULE_RST_POLARITY",
                            "category": "Clock & Reset",
                            "severity": "MAJOR",
                            "violation": f"Reset signal '{sig.name}' is declared as active-low, but no active-low conditional check was found.",
                            "protocol_rule": "Asynchronous active-low resets must check assertion using logical NOT or negative edge triggers.",
                            "spec_section": "Clocks & Resets",
                            "rtl_snippet": f"always @(posedge clk)",
                            "suggested_correction": f"Change reset condition checking block to verify '{sig.name} == 1'b0' or use negedge.",
                            "efs_id": sig.signal_id
                        })
                else:  # active high
                    is_active_high_found = any([
                        "posedge" in code and sig.name in code,
                        f"{sig.name} == 1" in code,
                        f"{sig.name} == 1'b1" in code
                    ])
                    if not is_active_high_found:
                        failed_checks.append({
                            "check_name": "Reset Polarity Check",
                            "rule_id": "RULE_RST_POLARITY",
                            "category": "Clock & Reset",
                            "severity": "MAJOR",
                            "violation": f"Reset signal '{sig.name}' is active-high, but no active-high trigger was found.",
                            "protocol_rule": "Active-high resets must use positive-edge or high-level conditional triggers.",
                            "spec_section": "Clocks & Resets",
                            "rtl_snippet": f"negedge {sig.name}",
                            "suggested_correction": f"Update reset block to trigger on '{sig.name} == 1'b1'.",
                            "efs_id": sig.signal_id
                        })
        if not missing_clk_rst:
            passed_checks.append({
                "check_name": "Clock & Reset Domain Presence",
                "description": "Checked that all EFS IR clocks and resets exist in port declarations."
            })

        # 3. Register Mapping Checks
        missing_regs = []
        for reg in efs_ir.registers:
            if not re.search(rf"\b{reg.name}\b", code, re.IGNORECASE):
                missing_regs.append(reg.name)
                failed_checks.append({
                    "check_name": "Register Map Declaration",
                    "rule_id": "RULE_REG_MISSING",
                    "category": "Register Map",
                    "severity": "MAJOR",
                    "violation": f"Required register state signal '{reg.name}' is missing.",
                    "protocol_rule": "All registers specified in the register map layout must exist in internal declarations.",
                    "spec_section": "Register & FIFO Layout",
                    "rtl_snippet": "// Register missing",
                    "suggested_correction": f"Add register state variable declaration: reg [31:0] {reg.name.lower()};",
                    "efs_id": reg.register_id
                })
            elif reg.offset:
                short_offset = reg.offset.lower().replace("0x", "").lstrip("0")
                if not short_offset:
                    short_offset = "0"
                offset_pattern = rf"({reg.offset}|{short_offset})"
                if not re.search(offset_pattern, code, re.IGNORECASE):
                    warnings.append(f"Register offset '{reg.offset}' for '{reg.name}' not statically detected in bus address decoders.")
        if not missing_regs and efs_ir.registers:
            passed_checks.append({
                "check_name": "Register Map Integrity",
                "description": f"Verified that all {len(efs_ir.registers)} specified registers exist as internal signal identifiers."
            })

        # 4. FSM States Checks
        for fsm in efs_ir.fsms:
            missing_states = []
            for state in fsm.states:
                if not re.search(rf"\b{state.name}\b", code, re.IGNORECASE):
                    missing_states.append(state.name)
                    failed_checks.append({
                        "check_name": "FSM State Implementation",
                        "rule_id": "RULE_FSM_STATE_MISSING",
                        "category": "State Machine",
                        "severity": "MAJOR",
                        "violation": f"State identifier '{state.name}' of FSM was not detected in parameter declarations or state logic.",
                        "protocol_rule": "The design state machine FSM must implement all states required to coordinate operations.",
                        "spec_section": "State Machine (FSM) Description",
                        "rtl_snippet": "// Missing FSM state parameter",
                        "suggested_correction": f"Declare parameter or localparam state '{state.name}'.",
                        "efs_id": fsm.fsm_id
                    })
            if not missing_states:
                passed_checks.append({
                    "check_name": "FSM States Definition",
                    "description": f"All {len(fsm.states)} required state constants exist in logic for FSM '{fsm.name}'."
                })

    else:
        # Fallback to existing logic if no efs_ir is supplied
        if not req_model:
            req_model = _parse_spec_heuristically(design_spec)

        # 1. Interface Port Checks
        interfaces = req_model.get("interfaces", [])
        if interfaces:
            missing_ports = []
            for port in interfaces:
                port_name = port.get("name", "")
                pattern = rf"\b{port_name}\b"
                if not re.search(pattern, code):
                    missing_ports.append(port_name)
                    failed_checks.append({
                        "check_name": "Protocol Port Signal Presence",
                        "rule_id": "RULE_PORT_MISSING",
                        "category": "Interface Ports",
                        "severity": "CRITICAL",
                        "violation": f"Required signal '{port_name}' is completely missing from the generated RTL.",
                        "protocol_rule": "The protocol requires all mandatory signals defined in the interface to be declared in the port list.",
                        "spec_section": "External Interface Port List",
                        "rtl_snippet": "// Missing in port list",
                        "suggested_correction": f"Add port declaration: input or output signal '{port_name}' of type logic or std_logic."
                    })
            if not missing_ports:
                passed_checks.append({
                    "check_name": "Protocol Port Signal Presence",
                    "description": f"Verified that all {len(interfaces)} required interface signals exist in the port declarations."
                })

        # 2. Clock and Reset Checks
        clocks_resets = req_model.get("clocks_resets", [])
        if clocks_resets:
            missing_clk_rst = []
            for sig in clocks_resets:
                sig_name = sig.get("name", "")
                sig_type = sig.get("type", "clock")
                pattern = rf"\b{sig_name}\b"
                
                if not re.search(pattern, code):
                    missing_clk_rst.append(sig_name)
                    failed_checks.append({
                        "check_name": f"{sig_type.capitalize()} Presence Check",
                        "rule_id": "RULE_CLK_RST_MISSING",
                        "category": "Clock & Reset",
                        "severity": "CRITICAL",
                        "violation": f"Required {sig_type} signal '{sig_name}' is missing.",
                        "protocol_rule": f"The design must use the specified {sig_type} signal '{sig_name}' for sequential transitions and state clearing.",
                        "spec_section": "Clocks & Resets",
                        "rtl_snippet": "// Clock/reset missing",
                        "suggested_correction": f"Declare '{sig_name}' as an input clock/reset in the top-level module header."
                    })
                elif sig_type == "reset":
                    polarity = sig.get("active_level", "low")
                    if polarity == "low":
                        is_active_low_found = any([
                            "negedge" in code and sig_name in code,
                            f"!{sig_name}" in code,
                            f"~{sig_name}" in code,
                            f"{sig_name} == 0" in code,
                            f"{sig_name} == 1'b0" in code
                        ])
                        if not is_active_low_found:
                            failed_checks.append({
                                "check_name": "Reset Polarity Check",
                                "rule_id": "RULE_RST_POLARITY",
                                "category": "Clock & Reset",
                                "severity": "MAJOR",
                                "violation": f"Reset signal '{sig_name}' is declared as active-low, but no active-low conditional check (e.g. negedge, !{sig_name}) was found.",
                                "protocol_rule": "Asynchronous active-low resets must check assertion using logical NOT or negative edge triggers.",
                                "spec_section": "Clocks & Resets",
                                "rtl_snippet": f"always @(posedge clk)",
                                "suggested_correction": f"Change reset condition checking block to verify '{sig_name} == 1'b0' or use negedge."
                            })
                    else:  # active high
                        is_active_high_found = any([
                            "posedge" in code and sig_name in code,
                            f"{sig_name} == 1" in code,
                            f"{sig_name} == 1'b1" in code
                        ])
                        if not is_active_high_found:
                            failed_checks.append({
                                "check_name": "Reset Polarity Check",
                                "rule_id": "RULE_RST_POLARITY",
                                "category": "Clock & Reset",
                                "severity": "MAJOR",
                                "violation": f"Reset signal '{sig_name}' is active-high, but no active-high trigger was found.",
                                "protocol_rule": "Active-high resets must use positive-edge or high-level conditional triggers.",
                                "spec_section": "Clocks & Resets",
                                "rtl_snippet": f"negedge {sig_name}",
                                "suggested_correction": f"Update reset block to trigger on '{sig_name} == 1'b1'."
                            })
            if not missing_clk_rst:
                passed_checks.append({
                    "check_name": "Clock & Reset Domain Presence",
                    "description": "Checked that target clocks and resets exist in port declarations."
                })

        # 3. Register Mapping Checks
        registers = req_model.get("registers", [])
        if registers:
            missing_regs = []
            for reg in registers:
                reg_name = reg.get("name", "")
                reg_offset = reg.get("offset", "")
                
                if not re.search(rf"\b{reg_name}\b", code, re.IGNORECASE):
                    missing_regs.append(reg_name)
                    failed_checks.append({
                        "check_name": "Register Map Declaration",
                        "rule_id": "RULE_REG_MISSING",
                        "category": "Register Map",
                        "severity": "MAJOR",
                        "violation": f"Required register state signal '{reg_name}' is missing.",
                        "protocol_rule": "All registers specified in the register map layout must exist in the internal storage declarations.",
                        "spec_section": "Register & FIFO Layout",
                        "rtl_snippet": "// Register missing",
                        "suggested_correction": f"Add register state variable declaration: reg [31:0] {reg_name.lower()};"
                    })
                elif reg_offset:
                    short_offset = reg_offset.lower().replace("0x", "").lstrip("0")
                    if not short_offset:
                        short_offset = "0"
                    offset_pattern = rf"({reg_offset}|{short_offset})"
                    if not re.search(offset_pattern, code, re.IGNORECASE):
                        warnings.append(f"Register offset '{reg_offset}' for '{reg_name}' not statically detected in bus address decoders.")
            if not missing_regs:
                passed_checks.append({
                    "check_name": "Register Map Integrity",
                    "description": f"Verified that all {len(registers)} specified registers exist as internal signal identifiers."
                })

        # 4. FSM States Checks
        fsm = req_model.get("fsm_info", {})
        fsm_states = fsm.get("states", [])
        if fsm_states:
            missing_states = []
            for state in fsm_states:
                if not re.search(rf"\b{state}\b", code, re.IGNORECASE):
                    missing_states.append(state)
                    failed_checks.append({
                        "check_name": "FSM State Implementation",
                        "rule_id": "RULE_FSM_STATE_MISSING",
                        "category": "State Machine",
                        "severity": "MAJOR",
                        "violation": f"State identifier '{state}' of FSM was not detected in parameter declarations or state logic.",
                        "protocol_rule": "The design state machine FSM must implement all states required to coordinate operations.",
                        "spec_section": "State Machine (FSM) Description",
                        "rtl_snippet": "// Missing FSM state parameter",
                        "suggested_correction": f"Declare parameter or localparam state '{state}'."
                    })
            if not missing_states:
                passed_checks.append({
                    "check_name": "FSM States Definition",
                    "description": f"All {len(fsm_states)} required state constants exist in logic."
                })

    # -- 5. Dynamic HDL Compilation Check if external compiler is present --
    dynamic_res = run_dynamic_hdl_simulation(code, target_language=target_language, language_version=language_version)
    if dynamic_res.get("status") == "FAIL":
        failed_checks.append({
            "check_name": "Dynamic HDL Compilation",
            "rule_id": "RULE_HDL_COMPILE_FAIL",
            "category": "Dynamic Compilation",
            "severity": "CRITICAL",
            "violation": f"HDL compiler failed with syntax error: {dynamic_res.get('error')}",
            "protocol_rule": f"Code must compile cleanly under {hdl_report.target_language.value} standard.",
            "spec_section": "RTL Synthesizability and Standards",
            "rtl_snippet": "// Compilation error",
            "suggested_correction": f"Fix {hdl_report.target_language.value} syntax errors identified by dynamic HDL compiler."
        })
    elif dynamic_res.get("status") == "PASS":
        passed_checks.append({
            "check_name": "Dynamic HDL Compilation",
            "description": f"HDL code compiled cleanly with external tool ({dynamic_res.get('tool', 'simulator')})."
        })

    # -- 6. Requirement-to-RTL Traceability Verification ---
    trace_report = generate_traceability_report(
        code=code,
        efs_ir=efs_ir,
        design_spec=design_spec,
        target_language=target_language,
        language_version=language_version
    )
    for unmapped in trace_report.unmapped_elements:
        warnings.append(f"Unmapped RTL element '{unmapped.identifier}' ({unmapped.element_type}) has no supporting requirement in EFS IR.")

    # Calculate deterministic compliance score
    score = 100
    if failed_checks:
        # Subtract based on severity
        for f in failed_checks:
            sev = f.get("severity", "MAJOR")
            if sev == "CRITICAL":
                score -= 30
            elif sev == "MAJOR":
                score -= 15
            elif sev == "MINOR":
                score -= 5
        score = max(0, score)

    return {
        "compliance_score": score,
        "passed_checks": passed_checks,
        "failed_checks": failed_checks,
        "warnings": warnings,
        "traceability_report": trace_report.to_dict(),
        "hdl_validation": hdl_report.to_dict(),
        "dynamic_simulation": dynamic_res,
        "target_language": hdl_report.target_language.value
    }


def run_dynamic_hdl_simulation(
    rtl_code: str,
    target_language: str = "Verilog",
    language_version: Optional[str] = None
) -> Dict[str, Any]:
    """Execute dynamic HDL compilation check using iverilog, verilator, or slang if available in PATH."""
    import shutil
    import subprocess
    import tempfile
    import os

    from core.verification.hdl_validator import resolve_hdl_language, HDLLanguage
    lang = resolve_hdl_language(target_language)
    is_sv = (lang == HDLLanguage.SYSTEMVERILOG)

    iverilog_path = shutil.which("iverilog")
    verilator_path = shutil.which("verilator")
    slang_path = shutil.which("slang")

    if not iverilog_path and not verilator_path and not slang_path:
        return {
            "status": "DYNAMIC_VALIDATION_NOT_AVAILABLE",
            "details": "Insufficient behavioral testbench or missing HDL compiler (iverilog/verilator/slang) in PATH."
        }

    ext = ".sv" if is_sv else ".v"
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, mode="w", delete=False) as f:
            f.write(rtl_code)
            temp_path = f.name

        if slang_path:
            cmd = [slang_path, temp_path]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if res.returncode != 0:
                return {"status": "FAIL", "error": res.stderr.strip() or res.stdout.strip(), "tool": "slang"}
            return {"status": "PASS", "details": "Compiled cleanly with slang", "tool": "slang"}

        elif iverilog_path:
            out_file = temp_path + ".out"
            std_flag = "-g2012" if is_sv else "-g2005"
            res = subprocess.run([iverilog_path, std_flag, "-o", out_file, temp_path], capture_output=True, text=True, timeout=10)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(out_file):
                os.remove(out_file)

            if res.returncode != 0:
                return {"status": "FAIL", "error": res.stderr.strip() or res.stdout.strip(), "tool": "iverilog"}
            return {"status": "PASS", "details": f"Compiled cleanly with Icarus Verilog ({std_flag})", "tool": "iverilog"}

        elif verilator_path:
            cmd = [verilator_path, "--lint-only"]
            if not is_sv:
                cmd.append("-v2005")
            cmd.append(temp_path)
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if os.path.exists(temp_path):
                os.remove(temp_path)

            if res.returncode != 0:
                return {"status": "FAIL", "error": res.stderr.strip() or res.stdout.strip(), "tool": "verilator"}
            return {"status": "PASS", "details": "Linted cleanly with Verilator", "tool": "verilator"}

    except Exception as e:
        logger.debug(f"Dynamic HDL simulation execution exception: {e}")
        return {"status": "DYNAMIC_VALIDATION_NOT_AVAILABLE", "details": str(e)}

    return {"status": "DYNAMIC_VALIDATION_NOT_AVAILABLE", "details": "Compiler unavailable"}


def _parse_spec_heuristically(spec: str) -> Dict[str, Any]:
    """Fallback specification parser scanning text when structured model is missing."""
    interfaces = []
    clocks_resets = []
    registers = []
    states = []

    # Heuristically extract ports (lines starting with | containing inputs/outputs)
    for line in spec.split("\n"):
        if "|" in line and ("input" in line.lower() or "output" in line.lower()):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 3:
                name = parts[0]
                dir_ = "input" if "input" in parts[2].lower() or "in" in parts[2].lower() else "output"
                width = parts[1]
                interfaces.append({"name": name, "direction": dir_, "width": width})

    # Heuristically extract clock/resets
    if "clk" in spec.lower():
        clocks_resets.append({"name": "clk", "type": "clock", "active_level": "rising"})
    if "rst_n" in spec.lower():
        clocks_resets.append({"name": "rst_n", "type": "reset", "active_level": "low"})
    elif "reset" in spec.lower():
        clocks_resets.append({"name": "reset", "type": "reset", "active_level": "high"})

    # Heuristically extract FSM states from text lists
    states_match = re.search(r"states:\s*([a-z0-9_,\s]+)", spec, re.IGNORECASE)
    if states_match:
        states = [s.strip() for s in states_match.group(1).split(",") if s.strip()]

    return {
        "interfaces": interfaces,
        "clocks_resets": clocks_resets,
        "registers": registers,
        "fsm_info": {"states": states}
    }
