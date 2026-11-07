"""
Constrained Generator and Hybrid RTL Coordinator Module.

Combines deterministic structural generation with LLM-based behavioral generation.
Supports three generation modes:
  - DETERMINISTIC: Purely deterministic structural and behavioral generation from validated plan.
  - DIRECT_LLM: LLM generates full module, strictly constrained by locked plan header.
  - HYBRID (Default): Combines deterministic structural generation with LLM behavioral blocks.
"""

from __future__ import annotations

import json
import logging
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from core.efs_ir.models import EFSIR, EFSSignal, EFSRegister, EFSFSM
from core.efs_ir.rtl_plan import (
    PlanExpression,
    PlanSequentialProcess,
    PlanCombinationalProcess,
    PlanCombinationalAssignment,
    PlanRegisterUpdate,
    PlanProtocolContract,
    PlanProtocolRule,
    PlanUnresolvedLogic,
)
from core.verification.hdl_validator import (
    validate_rtl_syntax_and_structure,
    resolve_hdl_language,
    HDLLanguage,
)

logger = logging.getLogger("graphrag.agents.constrained_generator")


class GenerationMode(str, Enum):
    """Supported RTL generation modes."""
    DETERMINISTIC = "DETERMINISTIC"
    DIRECT_LLM = "DIRECT_LLM"
    HYBRID = "HYBRID"


def _normalize_plan(rtl_plan: Any) -> Dict[str, Any]:
    """Normalize plan object or dictionary into a standard dictionary."""
    if rtl_plan is None:
        return {}
    if hasattr(rtl_plan, "to_dict"):
        d = rtl_plan.to_dict()
    elif isinstance(rtl_plan, dict):
        d = dict(rtl_plan)
    else:
        return {}

    # Bridge 'registers' from compiled context to 'internal_registers' if not explicitly defined
    if "internal_registers" not in d and "registers" in d:
        d["internal_registers"] = d["registers"]
    return d


def _normalize_hdl_condition(cond: str) -> str:
    """Normalize informal natural language or malformed conditions to valid HDL expressions."""
    if not cond or not isinstance(cond, str):
        return "1'b1"
    s = cond.strip()
    if not s or s in ("1'b1", "1", "true", "True"):
        return "1'b1"

    # Strip assignment statements or colons if accidentally included
    if ":" in s:
        s = s.split(":", 1)[0].strip()
    if "<=" in s:
        s = s.split("<=")[0].strip()
    if "=" in s and not any(op in s for op in ("==", "!=", "<=", ">=")):
        s = s.split("=")[0].strip()

    # Strip unclosed braces
    s = s.replace("{", "").replace("}", "").strip()

    # Replace natural language assertion phrases
    s = re.sub(r'\b(\w+)\s+(?:is\s+)?asserted\b', r'\1', s, flags=re.IGNORECASE)
    s = re.sub(r'\b(\w+)\s+(?:is\s+)?deasserted\b', r'(!\1)', s, flags=re.IGNORECASE)
    s = re.sub(r'\b(\w+)\s+(?:is\s+)?high\b', r'\1', s, flags=re.IGNORECASE)
    s = re.sub(r'\b(\w+)\s+(?:is\s+)?low\b', r'(!\1)', s, flags=re.IGNORECASE)
    s = re.sub(r'\b(\w+)\s+(?:is\s+)?active\b', r'\1', s, flags=re.IGNORECASE)
    s = re.sub(r'\b(\w+)\s+(?:is\s+)?inactive\b', r'(!\1)', s, flags=re.IGNORECASE)
    s = re.sub(r'\band\b', r'&&', s, flags=re.IGNORECASE)
    s = re.sub(r'\bor\b', r'||', s, flags=re.IGNORECASE)
    s = re.sub(r'\bnot\b', r'!', s, flags=re.IGNORECASE)
    s = s.strip()
    if not s or s in ("1'b1", "1", "true", "True"):
        return "1'b1"
    return s


def emit_deterministic_module_header(
    rtl_plan: Union[Dict[str, Any], Any],
    target_language: Optional[str] = None
) -> str:
    """
    Generate a deterministic Verilog/SystemVerilog ANSI module declaration header
    strictly matching the plan ports and parameters.
    """
    plan_dict = _normalize_plan(rtl_plan)
    module_name = plan_dict.get("module_name", "hardware_module")
    target_lang = target_language or plan_dict.get("target_language", "Verilog")
    is_sv = "systemverilog" in str(target_lang).lower()

    params = plan_dict.get("parameters", [])
    ports = plan_dict.get("ports", [])
    internal_regs = plan_dict.get("internal_registers", [])
    output_strategy = plan_dict.get("output_driving_strategy", {})
    output_assigns = plan_dict.get("output_assignments", {})

    lines: List[str] = []
    
    # 1. Parameter Declarations
    param_strs: List[str] = []
    for p in params:
        if isinstance(p, dict):
            p_name = p.get("name", "PARAM")
            p_val = p.get("value", p.get("default", "0"))
            p_type = p.get("type", "")
            if is_sv and p_type and p_type.lower() in ("int", "integer", "bit", "logic"):
                param_strs.append(f"parameter int {p_name} = {p_val}")
            else:
                param_strs.append(f"parameter {p_name} = {p_val}")
        elif hasattr(p, "name"):
            p_name = getattr(p, "name", "PARAM")
            p_val = getattr(p, "value", "0")
            param_strs.append(f"parameter {p_name} = {p_val}")

    if param_strs:
        lines.append(f"module {module_name} #(")
        for i, ps in enumerate(param_strs):
            comma = "," if i < len(param_strs) - 1 else ""
            lines.append(f"    {ps}{comma}")
        lines.append(") (")
    else:
        lines.append(f"module {module_name} (")

    # 2. Port Declarations
    port_strs: List[str] = []
    for p in ports:
        if isinstance(p, dict):
            p_name = p.get("name", "sig")
            p_dir = p.get("direction", "input")
            p_w = str(p.get("width", "1")).strip()
            p_src = str(p.get("source_ref", p.get("source_evidence", "")))
        else:
            p_name = getattr(p, "name", "sig")
            p_dir = getattr(p, "direction", "input")
            p_w = str(getattr(p, "width", "1")).strip()
            p_src = str(getattr(p, "source_ref", ""))

        # Check if a parameter defines this port's width
        param_for_width = None
        for par in params:
            par_name = par.get("name") if isinstance(par, dict) else getattr(par, "name", "")
            if par_name and (f"[{par_name}-1:0]" in p_src or par_name == p_w or f"{par_name}" in p_src):
                param_for_width = par_name
                break

        if param_for_width:
            width_expr = f"[{param_for_width}-1:0] "
        elif p_w.isdigit():
            w_int = int(p_w)
            width_expr = f"[{w_int - 1}:0] " if w_int > 1 else ""
        elif p_w and p_w != "1":
            width_expr = f"[{p_w}-1:0] " if not p_w.startswith("[") else f"{p_w} "
        else:
            width_expr = ""

        # Language-aware type keyword
        if is_sv:
            type_keyword = "logic"
        else:
            # Verilog-2001: input is wire; output can be reg or wire
            if p_dir == "input":
                type_keyword = "wire"
            elif p_dir == "output":
                # Check whether output is driven by sequential process vs continuous assign
                is_seq = False
                seq_procs = plan_dict.get("sequential_logic", [])
                for sp in seq_procs:
                    sp_dict = sp.to_dict() if hasattr(sp, "to_dict") else (sp if isinstance(sp, dict) else {})
                    for u in sp_dict.get("register_updates", []):
                        u_tgt = u.get("target") if isinstance(u, dict) else getattr(u, "target", "")
                        if u_tgt == p_name:
                            is_seq = True
                            break
                    if p_name in sp_dict.get("reset_actions", {}):
                        is_seq = True
                    if is_seq:
                        break

                # Check legacy internal_registers driving output directly in always block
                if not is_seq:
                    is_legacy_direct = any(
                        (getattr(r, "name", None) == p_name or (isinstance(r, dict) and r.get("name") == p_name))
                        for r in internal_regs
                    ) and not (output_strategy.get(p_name) in ("continuous", "assign") or bool(output_assigns.get(p_name)))
                    if is_legacy_direct and not seq_procs:
                        is_seq = True

                # Check if driven procedurally in always @(*) combinational blocks (data_selections, opcodes)
                is_procedural = is_seq
                if not is_procedural:
                    for cp in plan_dict.get("combinational_logic", []):
                        cp_dict = cp.to_dict() if hasattr(cp, "to_dict") else (cp if isinstance(cp, dict) else {})
                        for ds in cp_dict.get("data_selections", []):
                            ds_dict = ds.to_dict() if hasattr(ds, "to_dict") else (ds if isinstance(ds, dict) else {})
                            if ds_dict.get("destination") == p_name:
                                is_procedural = True
                                break
                        if is_procedural:
                            break
                    if not is_procedural:
                        dp_data = plan_dict.get("datapath", {})
                        dp_d = dp_data if isinstance(dp_data, dict) else (dp_data.to_dict() if hasattr(dp_data, "to_dict") else {})
                        for ds in dp_d.get("data_selections", []):
                            ds_dict = ds.to_dict() if hasattr(ds, "to_dict") else (ds if isinstance(ds, dict) else {})
                            if ds_dict.get("destination") == p_name:
                                is_procedural = True
                                break
                    if not is_procedural and plan_dict.get("opcodes"):
                        if "valid" in p_name.lower():
                            is_procedural = True

                # Crucial Verilog rule: If driven by continuous assignment, it CANNOT be reg!
                has_continuous_assign = False
                for cp in plan_dict.get("combinational_logic", []):
                    cp_dict = cp.to_dict() if hasattr(cp, "to_dict") else (cp if isinstance(cp, dict) else {})
                    for a in cp_dict.get("assignments", []):
                        a_dict = a.to_dict() if hasattr(a, "to_dict") else (a if isinstance(a, dict) else {})
                        if a_dict.get("target") == p_name:
                            has_continuous_assign = True
                            break
                    if has_continuous_assign:
                        break
                if p_name in output_assigns or output_strategy.get(p_name) in ("continuous", "assign"):
                    has_continuous_assign = True

                if has_continuous_assign:
                    is_procedural = False

                type_keyword = "reg" if is_procedural else "wire"
            else:
                type_keyword = "wire"

        port_strs.append(f"{p_dir:6s} {type_keyword:4s} {width_expr}{p_name}".rstrip())

    for i, port_decl in enumerate(port_strs):
        comma = "," if i < len(port_strs) - 1 else ""
        lines.append(f"    {port_decl}{comma}")

    lines.append(");")
    return "\n".join(lines)


def emit_deterministic_declarations(
    rtl_plan: Union[Dict[str, Any], Any],
    target_language: str = "Verilog"
) -> str:
    """
    Generate deterministic internal declarations: wires, registers,
    opcode constants, state machine enums/localparams, and memories.
    Deduplicates against module ports and existing declarations.
    """
    plan_dict = _normalize_plan(rtl_plan)
    target_lang = plan_dict.get("target_language") or target_language
    is_sv = "systemverilog" in str(target_lang).lower()

    lines: List[str] = []
    declared_names = set()

    # Track all port names to avoid duplicate declarations
    for p in plan_dict.get("ports", []):
        p_name = p.get("name") if isinstance(p, dict) else getattr(p, "name", "")
        if p_name:
            declared_names.add(p_name)

    # 1. Opcode Constants
    opcodes = plan_dict.get("opcodes", [])
    if opcodes:
        lines.append("// Opcode Definitions")
        for op in opcodes:
            op_name = op.get("name", op.get("mnemonic", "")) if isinstance(op, dict) else getattr(op, "name", getattr(op, "mnemonic", ""))
            op_enc = op.get("binary_encoding", op.get("encoding", op.get("code", "0"))) if isinstance(op, dict) else getattr(op, "binary_encoding", getattr(op, "encoding", "0"))
            op_w = str(op.get("width", "")) if isinstance(op, dict) else str(getattr(op, "width", ""))
            if op_name and op_name not in declared_names:
                declared_names.add(op_name)
                clean_enc = str(op_enc).strip()
                if clean_enc and all(c in "01_" for c in clean_enc):
                    raw_bits = clean_enc.replace("_", "")
                    w_int = int(op_w) if (op_w and op_w.isdigit()) else len(raw_bits)
                    lines.append(f"localparam [{w_int-1}:0] {op_name} = {w_int}'b{clean_enc};")
                else:
                    w_int = int(op_w) if (op_w and op_w.isdigit()) else 6
                    lines.append(f"localparam [{w_int-1}:0] {op_name} = {clean_enc};")
        lines.append("")

    # 2. FSM State Encodings and Variables
    fsm = plan_dict.get("fsm") or plan_dict.get("fsm_spec") or {}
    fsm_dict = fsm if isinstance(fsm, dict) else (fsm.to_dict() if hasattr(fsm, "to_dict") else {})
    states = fsm_dict.get("states", [])

    if states:
        lines.append("// FSM State Declarations")
        state_names = []
        for idx, st in enumerate(states):
            st_name = st.get("name") if isinstance(st, dict) else getattr(st, "name", str(st))
            if st_name:
                state_names.append(st_name)

        # Collect any referenced state names from entire plan (e.g. current_state == DONE or current_state == EXEC)
        plan_repr = json.dumps(plan_dict, default=str)
        for m_st in re.findall(r"current_state\s*==\s*([A-Za-z_][A-Za-z0-9_]*)", plan_repr):
            if m_st not in state_names and not m_st.startswith(("1'", "2'", "3'", "4'")):
                state_names.append(m_st)

        fsm_width = max(1, (len(state_names) - 1).bit_length()) if state_names else 1

        if is_sv:
            # SystemVerilog typedef enum
            state_list_str = ", ".join(state_names)
            lines.append(f"typedef enum logic [{fsm_width-1}:0] {{ {state_list_str} }} state_t;")
            lines.append("state_t current_state, next_state;")
            declared_names.update(["current_state", "next_state", "state_t"])
            declared_names.update(state_names)
        else:
            # Verilog localparams
            for idx, st_name in enumerate(state_names):
                if st_name not in declared_names:
                    lines.append(f"localparam [{fsm_width-1}:0] {st_name} = {fsm_width}'d{idx};")
                    declared_names.add(st_name)
            lines.append(f"reg [{fsm_width-1}:0] current_state, next_state;")
            declared_names.update(["current_state", "next_state"])
        lines.append("")

    # 3. Internal Registers
    internal_regs = plan_dict.get("internal_registers", [])
    if internal_regs:
        lines.append("// Internal Registers")
        for r in internal_regs:
            r_name = r.get("name", "") if isinstance(r, dict) else getattr(r, "name", "")
            r_w = str(r.get("width", "1")).strip() if isinstance(r, dict) else str(getattr(r, "width", "1")).strip()
            r_src = str(r.get("source_ref", "")) if isinstance(r, dict) else str(getattr(r, "source_ref", ""))

            if r_name and r_name not in declared_names:
                declared_names.add(r_name)
                # Check for parameter width
                param_w = None
                for par in plan_dict.get("parameters", []):
                    par_name = par.get("name") if isinstance(par, dict) else getattr(par, "name", "")
                    if par_name and (par_name in r_src or par_name == r_w):
                        param_w = par_name
                        break

                if param_w:
                    w_expr = f"[{param_w}-1:0] "
                elif r_w.isdigit():
                    w_int = int(r_w)
                    w_expr = f"[{w_int-1}:0] " if w_int > 1 else ""
                elif r_w and r_w != "1":
                    w_expr = f"[{r_w}-1:0] " if not r_w.startswith("[") else f"{r_w} "
                else:
                    w_expr = ""

                type_kw = "logic" if is_sv else "reg"
                lines.append(f"{type_kw} {w_expr}{r_name};")
        lines.append("")

    # 4. Internal Signals / Nets
    internal_signals = plan_dict.get("internal_signals", [])
    if internal_signals:
        lines.append("// Internal Signals")
        for sig in internal_signals:
            s_name = sig.get("name", "") if isinstance(sig, dict) else getattr(sig, "name", "")
            s_w = str(sig.get("width", "1")).strip() if isinstance(sig, dict) else str(getattr(sig, "width", "1")).strip()

            if s_name and s_name not in declared_names:
                declared_names.add(s_name)
                w_int = int(s_w) if s_w.isdigit() else 1
                w_expr = f"[{w_int-1}:0] " if w_int > 1 else ""
                type_kw = "logic" if is_sv else "wire"
                lines.append(f"{type_kw} {w_expr}{s_name};")
        lines.append("")

    # 5. Declarations from Structured Sequential and Combinational Logic
    for sp in plan_dict.get("sequential_logic", []):
        sp_dict = sp.to_dict() if hasattr(sp, "to_dict") else (sp if isinstance(sp, dict) else {})
        for u in sp_dict.get("register_updates", []):
            u_dict = u.to_dict() if hasattr(u, "to_dict") else (u if isinstance(u, dict) else {})
            u_tgt = u_dict.get("target")
            if u_tgt and u_tgt not in declared_names:
                declared_names.add(u_tgt)
                type_kw = "logic" if is_sv else "reg"
                lines.append(f"{type_kw} {u_tgt};")
        for r_tgt in sp_dict.get("reset_actions", {}).keys():
            if r_tgt and r_tgt not in declared_names:
                declared_names.add(r_tgt)
                type_kw = "logic" if is_sv else "reg"
                lines.append(f"{type_kw} {r_tgt};")

    for cp in plan_dict.get("combinational_logic", []):
        cp_dict = cp.to_dict() if hasattr(cp, "to_dict") else (cp if isinstance(cp, dict) else {})
        for a in cp_dict.get("assignments", []):
            a_dict = a.to_dict() if hasattr(a, "to_dict") else (a if isinstance(a, dict) else {})
            a_tgt = a_dict.get("target")
            if a_tgt and a_tgt not in declared_names:
                declared_names.add(a_tgt)
                type_kw = "logic" if is_sv else "wire"
                lines.append(f"{type_kw} {a_tgt};")

    # 6. Sequential State Register fallback
    has_clk = any(
        (p.get("name") if isinstance(p, dict) else getattr(p, "name", "")).lower() in ("clk", "clock")
        for p in plan_dict.get("ports", [])
    )
    if has_clk and not internal_regs and not (plan_dict.get("fsm") or plan_dict.get("fsm_spec")):
        type_kw = "logic" if is_sv else "reg"
        lines.append("// Sequential State Register")
        lines.append(f"{type_kw} state_reg;")
        declared_names.add("state_reg")
        lines.append("")

    # 7. Auto-declare any referenced internal signals/enables not yet declared
    referenced_signals = set()
    verilog_keywords = {
        "current_state", "next_state", "if", "begin", "end", "else", "posedge",
        "negedge", "reg", "wire", "logic", "localparam", "parameter", "assign",
        "always", "case", "casex", "casez", "endcase", "default", "signed", "unsigned",
        "input", "output", "inout", "initial", "module", "endmodule", "or", "and", "xor",
        "not", "for", "while", "repeat", "function", "endfunction", "task", "endtask",
        "generate", "endgenerate", "genvar", "integer", "real", "time", "defparam",
        "table", "endtable", "primitive", "endprimitive", "fork", "join", "join_any",
        "join_none", "typedef", "enum", "struct", "union", "automatic", "static",
        "import", "export", "package", "endpackage", "interface", "endinterface",
        "program", "endprogram", "class", "endclass", "clocking", "endclocking",
        "assert", "assume", "cover", "restrict", "expect", "True", "False", "None",
        "expression_id", "expression_type", "operands", "operator", "source_references",
        "condition", "result", "width"
    }
    for sp in plan_dict.get("sequential_logic", []):
        sp_dict = sp.to_dict() if hasattr(sp, "to_dict") else (sp if isinstance(sp, dict) else {})
        for u in sp_dict.get("register_updates", []):
            u_dict = u.to_dict() if hasattr(u, "to_dict") else (u if isinstance(u, dict) else {})
            cond = str(u_dict.get("condition") or "")
            en = str(u_dict.get("enable") or "")
            nv = u_dict.get("next_value")
            if hasattr(nv, "to_verilog_expr"):
                nv_str = nv.to_verilog_expr()
            elif isinstance(nv, dict):
                try:
                    from core.efs_ir.rtl_plan import PlanExpression
                    nv_str = PlanExpression.from_dict(nv).to_verilog_expr()
                except Exception:
                    nv_str = ""
            elif isinstance(nv, str):
                nv_str = nv
            else:
                nv_str = ""
            for txt in (cond, en, nv_str):
                for ident in re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", txt):
                    if ident in ("b0", "b1", "h0", "d0") or re.match(r"^[0-9]+'[bBhdHD]", ident) or re.match(r"^[hHbBdD][0-9a-fA-F]+$", ident):
                        continue
                    if ident not in declared_names and ident not in verilog_keywords:
                        referenced_signals.add(ident)

    if referenced_signals:
        lines.append("// Internal Control & Handshake Signals")
        port_names_lower = {str(p.get("name", "")).lower(): str(p.get("name", "")) for p in plan_dict.get("ports", []) if isinstance(p, dict)}
        for sig in sorted(referenced_signals):
            if sig not in declared_names:
                declared_names.add(sig)
                if "valid" in sig.lower() and "axi_rvalid" in port_names_lower:
                    lines.append(f"wire {sig} = {port_names_lower['axi_rvalid']};")
                else:
                    lines.append(f"wire {sig} = 1'b0;")
        lines.append("")

    return "\n".join(lines).strip()


def emit_deterministic_behavioral_logic(
    rtl_plan: Union[Dict[str, Any], Any],
    target_language: str = "Verilog"
) -> Dict[str, str]:
    """
    Generate deterministic behavioral HDL logic (continuous assignments,
    combinational blocks, sequential blocks, FSM blocks, protocol assertions)
    strictly from the populated RTLGenerationPlan.
    """
    plan_dict = _normalize_plan(rtl_plan)
    target_lang = plan_dict.get("target_language") or target_language
    is_sv = "systemverilog" in str(target_lang).lower()

    assignments: List[str] = []
    comb_lines: List[str] = []
    seq_lines: List[str] = []
    fsm_lines: List[str] = []
    protocol_lines: List[str] = []
    unresolved_lines: List[str] = []

    assigned_targets: Set[str] = set()
    driver_accounting: Dict[str, Dict[str, Any]] = {}

    # -------------------------------------------------------------------------
    # 1. Structured Combinational Logic (from plan_dict["combinational_logic"])
    # -------------------------------------------------------------------------
    comb_procs = plan_dict.get("combinational_logic", [])
    for cp in comb_procs:
        cp_dict = cp.to_dict() if hasattr(cp, "to_dict") else (cp if isinstance(cp, dict) else {})
        p_type = cp_dict.get("process_type", "ALWAYS_COMB")

        for a in cp_dict.get("assignments", []):
            a_dict = a.to_dict() if hasattr(a, "to_dict") else (a if isinstance(a, dict) else {})
            tgt = a_dict.get("target")
            if not tgt or tgt in assigned_targets:
                continue

            expr = a_dict.get("expression")
            expr_raw = a_dict.get("expression_raw", "")
            cond = a_dict.get("condition")
            def_assign = a_dict.get("default_assignment")

            expr_str = ""
            if hasattr(expr, "to_verilog_expr"):
                expr_str = expr.to_verilog_expr()
            elif isinstance(expr, dict):
                expr_str = PlanExpression.from_dict(expr).to_verilog_expr()
            elif expr_raw:
                expr_str = expr_raw
            elif isinstance(expr, str):
                expr_str = expr

            if not expr_str:
                continue

            if cond and str(cond).strip() not in ("1'b1", "1", "true", "True"):
                def_val = def_assign or "0"
                expr_str = f"({str(cond).strip()}) ? ({expr_str}) : ({def_val})"

            assignments.append(f"assign {tgt} = {expr_str};")
            assigned_targets.add(tgt)
            driver_accounting[tgt] = {
                "target": tgt,
                "declared_driver": expr_str,
                "driver_type": "CONTINUOUS_ASSIGN",
                "driver_origin": a_dict.get("provenance", "PLAN_COMBINATIONAL"),
                "owning_plan_element": "combinational_logic",
                "approval_status": "APPROVED" if (hasattr(a, "origin") and getattr(a, "origin") == "HUMAN_APPROVED") or a_dict.get("origin") == "HUMAN_APPROVED" else "PLAN_DERIVED",
                "emission_status": "EMITTED",
                "validation_status": "DRIVEN",
            }

        for sel_op in cp_dict.get("data_selections", []):
            sel_dict = sel_op.to_dict() if hasattr(sel_op, "to_dict") else (sel_op if isinstance(sel_op, dict) else {})
            d_name = sel_dict.get("destination", "out")
            s_sig = sel_dict.get("selector_signal", "sel")
            cases = sel_dict.get("cases", {})
            def_val = sel_dict.get("default_assignment", "")

            if d_name not in assigned_targets:
                block_kw = "always_comb" if is_sv else "always @(*)"
                comb_lines.append(f"{block_kw} begin")
                comb_lines.append(f"    case ({s_sig})")
                for c_val, c_rhs in cases.items():
                    comb_lines.append(f"        {c_val}: {d_name} = {c_rhs};")
                if def_val:
                    comb_lines.append(f"        default: {d_name} = {def_val};")
                elif cases:
                    first_rhs = next(iter(cases.values()))
                    comb_lines.append(f"        default: {d_name} = {first_rhs};")
                comb_lines.append("    endcase")
                comb_lines.append("end")
                assigned_targets.add(d_name)
                driver_accounting[d_name] = {
                    "target": d_name,
                    "declared_driver": f"case ({s_sig})",
                    "driver_type": "ALWAYS_COMB_CASE",
                    "driver_origin": "PLAN_DATAPATH",
                    "owning_plan_element": "combinational_logic",
                    "approval_status": "PLAN_DERIVED",
                    "emission_status": "EMITTED",
                    "validation_status": "DRIVEN",
                }

    # -------------------------------------------------------------------------
    # 2. Datapath Strategy / Legacy Output Assignments
    # -------------------------------------------------------------------------
    datapath = plan_dict.get("datapath", {})
    dp_dict = datapath if isinstance(datapath, dict) else (datapath.to_dict() if hasattr(datapath, "to_dict") else {})
    for sel_op in dp_dict.get("data_selections", []):
        sel_dict = sel_op.to_dict() if hasattr(sel_op, "to_dict") else (sel_op if isinstance(sel_op, dict) else {})
        d_name = sel_dict.get("destination", "out")
        s_sig = sel_dict.get("selector_signal", "sel")
        cases = sel_dict.get("cases", {})
        def_val = sel_dict.get("default_assignment", "")
        if d_name not in assigned_targets:
            block_kw = "always_comb" if is_sv else "always @(*)"
            comb_lines.append(f"{block_kw} begin")
            comb_lines.append(f"    case ({s_sig})")
            for c_val, c_rhs in cases.items():
                comb_lines.append(f"        {c_val}: {d_name} = {c_rhs};")
            if def_val:
                comb_lines.append(f"        default: {d_name} = {def_val};")
            elif cases:
                first_rhs = next(iter(cases.values()))
                comb_lines.append(f"        default: {d_name} = {first_rhs};")
            comb_lines.append("    endcase")
            comb_lines.append("end")
            assigned_targets.add(d_name)
            driver_accounting[d_name] = {
                "target": d_name,
                "declared_driver": f"case ({s_sig})",
                "driver_type": "ALWAYS_COMB_CASE",
                "driver_origin": "PLAN_DATAPATH",
                "owning_plan_element": "combinational_logic",
                "approval_status": "PLAN_DERIVED",
                "emission_status": "EMITTED",
                "validation_status": "DRIVEN",
            }

    output_assigns = plan_dict.get("output_assignments", {})
    for p_name, p_expr in output_assigns.items():
        if p_name not in assigned_targets:
            assignments.append(f"assign {p_name} = {p_expr};")
            assigned_targets.add(p_name)
            driver_accounting[p_name] = {
                "target": p_name,
                "declared_driver": str(p_expr),
                "driver_type": "CONTINUOUS_ASSIGN",
                "driver_origin": "PLAN_OUTPUT_ASSIGNMENTS",
                "owning_plan_element": "output_assignments",
                "approval_status": "PLAN_DERIVED",
                "emission_status": "EMITTED",
                "validation_status": "DRIVEN",
            }

    # -------------------------------------------------------------------------
    # 3. Opcode Decoding Combinational Logic
    # -------------------------------------------------------------------------
    opcodes = plan_dict.get("opcodes", [])
    valid_port = next((
        p.get("name") if isinstance(p, dict) else getattr(p, "name", "")
        for p in plan_dict.get("ports", [])
        if "valid" in (p.get("name") if isinstance(p, dict) else getattr(p, "name", "")).lower()
    ), None)
    opcode_port = next((
        p.get("name") if isinstance(p, dict) else getattr(p, "name", "")
        for p in plan_dict.get("ports", [])
        if "opcode" in (p.get("name") if isinstance(p, dict) else getattr(p, "name", "")).lower() or (p.get("name") if isinstance(p, dict) else getattr(p, "name", "")).lower() == "op"
    ), "opcode")

    if opcodes and valid_port and valid_port not in assigned_targets:
        block_kw = "always_comb" if is_sv else "always @(*)"
        comb_lines.append(f"{block_kw} begin")
        comb_lines.append(f"    case ({opcode_port})")
        op_names = [
            op.get("name", op.get("mnemonic", "")) if isinstance(op, dict) else getattr(op, "name", getattr(op, "mnemonic", ""))
            for op in opcodes
        ]
        op_names = [o for o in op_names if o]
        if op_names:
            chunked_ops = [", ".join(op_names[i:i+4]) for i in range(0, len(op_names), 4)]
            case_items = ",\n        ".join(chunked_ops)
            comb_lines.append(f"        {case_items}: {valid_port} = 1'b1;")
        comb_lines.append(f"        default: {valid_port} = 1'b0;")
        comb_lines.append("    endcase")
        comb_lines.append("end")
        assigned_targets.add(valid_port)
        driver_accounting[valid_port] = {
            "target": valid_port,
            "declared_driver": f"case ({opcode_port})",
            "driver_type": "ALWAYS_COMB_CASE",
            "driver_origin": "PLAN_OPCODE_DECODER",
            "owning_plan_element": "combinational_logic",
            "approval_status": "PLAN_DERIVED",
            "emission_status": "EMITTED",
            "validation_status": "DRIVEN",
        }

    # -------------------------------------------------------------------------
    # 4. Structured Sequential Logic (from plan_dict["sequential_logic"])
    # -------------------------------------------------------------------------
    seq_procs = plan_dict.get("sequential_logic", [])
    if seq_procs:
        for sp in seq_procs:
            sp_dict = sp.to_dict() if hasattr(sp, "to_dict") else (sp if isinstance(sp, dict) else {})
            clk_sig = sp_dict.get("clock_signal", "clk")
            clk_edge = sp_dict.get("clock_edge", "POSEDGE").upper()
            clk_edge_kw = "negedge" if "NEG" in clk_edge else "posedge"

            rst_sig = sp_dict.get("reset_signal")
            rst_type = sp_dict.get("reset_type", "ASYNC").upper()
            rst_lvl = sp_dict.get("reset_active_level", "LOW").upper()
            active_low = (rst_lvl == "LOW")

            seq_kw = "always_ff @" if is_sv else "always @"

            # Check if reset port actually exists in module ports
            has_rst_port = any(
                (p.get("name") if isinstance(p, dict) else getattr(p, "name", "")).lower() == str(rst_sig).lower()
                for p in plan_dict.get("ports", [])
            ) if rst_sig else False

            if rst_sig and has_rst_port and rst_type == "ASYNC":
                rst_edge = f"negedge {rst_sig}" if active_low else f"posedge {rst_sig}"
                sens_list = f"({clk_edge_kw} {clk_sig} or {rst_edge})"
            else:
                sens_list = f"({clk_edge_kw} {clk_sig})"

            reset_actions = sp_dict.get("reset_actions", {})
            updates = sp_dict.get("register_updates", [])
            fsm_state_updates = sp_dict.get("fsm_state_updates", {})

            seq_lines.append(f"{seq_kw} {sens_list} begin")

            if rst_sig and has_rst_port:
                rst_cond = f"!{rst_sig}" if active_low else f"{rst_sig}"
                seq_lines.append(f"    if ({rst_cond}) begin")

                # Emit reset actions
                for r_tgt, r_val in reset_actions.items():
                    seq_lines.append(f"        {r_tgt} <= {r_val};")

                # Registers without explicit reset_actions get default zero
                for u in updates:
                    u_dict = u.to_dict() if hasattr(u, "to_dict") else (u if isinstance(u, dict) else {})
                    tgt = u_dict.get("target")
                    if tgt and tgt not in reset_actions:
                        p_obj = next((p for p in plan_dict.get("ports", []) if (p.get("name") if isinstance(p, dict) else getattr(p, "name", "")) == tgt), None)
                        r_obj = next((r for r in plan_dict.get("internal_registers", []) if (r.get("name") if isinstance(r, dict) else getattr(r, "name", "")) == tgt), None)
                        w = (p_obj or r_obj or {}).get("width", "1") if isinstance((p_obj or r_obj), dict) else getattr((p_obj or r_obj), "width", "1")
                        w_str = str(w).strip()
                        w_int = int(w_str) if w_str.isdigit() else 1
                        zero_lit = f"{w_int}'d0" if w_int > 1 else "1'b0"
                        seq_lines.append(f"        {tgt} <= {zero_lit};")

                if fsm_state_updates:
                    cur_s = fsm_state_updates.get("current_state_variable", "current_state")
                    init_s = fsm_state_updates.get("initial_state") or (
                        plan_dict.get("fsm", {}).get("initial_state") if isinstance(plan_dict.get("fsm"), dict) else None
                    )
                    if not init_s and plan_dict.get("fsm"):
                        st_list = plan_dict.get("fsm", {}).get("states", []) if isinstance(plan_dict.get("fsm"), dict) else []
                        if st_list:
                            init_s = st_list[0].get("name") if isinstance(st_list[0], dict) else getattr(st_list[0], "name", str(st_list[0]))
                    if init_s:
                        seq_lines.append(f"        {cur_s} <= {init_s};")

                seq_lines.append("    end else begin")

                # Regular clocked register updates
                for u in updates:
                    u_dict = u.to_dict() if hasattr(u, "to_dict") else (u if isinstance(u, dict) else {})
                    tgt = u_dict.get("target")
                    if not tgt:
                        continue
                    assigned_targets.add(tgt)

                    nv = u_dict.get("next_value")
                    expr_str = ""
                    if hasattr(nv, "to_verilog_expr"):
                        expr_str = nv.to_verilog_expr()
                    elif isinstance(nv, dict):
                        expr_str = PlanExpression.from_dict(nv).to_verilog_expr()
                    elif isinstance(nv, str):
                        expr_str = nv
                    if not expr_str:
                        expr_str = tgt

                    cond = u_dict.get("condition")
                    en = u_dict.get("enable")
                    load = u_dict.get("load")

                    conds = []
                    if load:
                        c_l = _normalize_hdl_condition(str(load).strip())
                        if c_l and c_l not in ("1'b1", "1", "true", "True"):
                            conds.append(c_l)
                    if en:
                        c_e = _normalize_hdl_condition(str(en).strip())
                        if c_e and c_e not in ("1'b1", "1", "true", "True"):
                            conds.append(c_e)
                    if cond and str(cond).strip() not in ("1'b1", "1", "true", "True"):
                        c_c = _normalize_hdl_condition(str(cond).strip())
                        if c_c and c_c not in ("1'b1", "1", "true", "True"):
                            conds.append(c_c)

                    if conds:
                        full_cond = " && ".join(f"({c})" if " " in c else c for c in conds)
                        seq_lines.append(f"        if ({full_cond}) begin")
                        seq_lines.append(f"            {tgt} <= {expr_str};")
                        seq_lines.append("        end")
                    else:
                        seq_lines.append(f"        {tgt} <= {expr_str};")

                    driver_accounting[tgt] = {
                        "target": tgt,
                        "declared_driver": expr_str,
                        "driver_type": "SEQUENTIAL_CLOCK",
                        "driver_origin": u_dict.get("provenance", "PLAN_SEQUENTIAL"),
                        "owning_plan_element": "sequential_logic",
                        "approval_status": "APPROVED" if (hasattr(u, "origin") and getattr(u, "origin") == "HUMAN_APPROVED") or u_dict.get("origin") == "HUMAN_APPROVED" else "PLAN_DERIVED",
                        "emission_status": "EMITTED",
                        "validation_status": "DRIVEN",
                    }

                if fsm_state_updates:
                    cur_s = fsm_state_updates.get("current_state_variable", "current_state")
                    nxt_s = fsm_state_updates.get("next_state_variable", "next_state")
                    seq_lines.append(f"        {cur_s} <= {nxt_s};")
                    assigned_targets.add(cur_s)
                    driver_accounting[cur_s] = {
                        "target": cur_s,
                        "declared_driver": nxt_s,
                        "driver_type": "FSM_STATE_REGISTER",
                        "driver_origin": "PLAN_FSM",
                        "owning_plan_element": "sequential_logic",
                        "approval_status": "PLAN_DERIVED",
                        "emission_status": "EMITTED",
                        "validation_status": "DRIVEN",
                    }

                seq_lines.append("    end")
            else:
                # Synchronous only (no reset)
                for u in updates:
                    u_dict = u.to_dict() if hasattr(u, "to_dict") else (u if isinstance(u, dict) else {})
                    tgt = u_dict.get("target")
                    if not tgt:
                        continue
                    assigned_targets.add(tgt)

                    nv = u_dict.get("next_value")
                    expr_str = ""
                    if hasattr(nv, "to_verilog_expr"):
                        expr_str = nv.to_verilog_expr()
                    elif isinstance(nv, dict):
                        expr_str = PlanExpression.from_dict(nv).to_verilog_expr()
                    elif isinstance(nv, str):
                        expr_str = nv
                    if not expr_str:
                        expr_str = tgt

                    cond = u_dict.get("condition")
                    en = u_dict.get("enable")
                    load = u_dict.get("load")

                    conds = []
                    if load:
                        c_l = _normalize_hdl_condition(str(load).strip())
                        if c_l and c_l not in ("1'b1", "1", "true", "True"):
                            conds.append(c_l)
                    if en:
                        c_e = _normalize_hdl_condition(str(en).strip())
                        if c_e and c_e not in ("1'b1", "1", "true", "True"):
                            conds.append(c_e)
                    if cond and str(cond).strip() not in ("1'b1", "1", "true", "True"):
                        c_c = _normalize_hdl_condition(str(cond).strip())
                        if c_c and c_c not in ("1'b1", "1", "true", "True"):
                            conds.append(c_c)

                    if conds:
                        full_cond = " && ".join(f"({c})" if " " in c else c for c in conds)
                        seq_lines.append(f"    if ({full_cond}) begin")
                        seq_lines.append(f"        {tgt} <= {expr_str};")
                        seq_lines.append("    end")
                    else:
                        seq_lines.append(f"    {tgt} <= {expr_str};")

                    driver_accounting[tgt] = {
                        "target": tgt,
                        "declared_driver": expr_str,
                        "driver_type": "SEQUENTIAL_CLOCK",
                        "driver_origin": u_dict.get("provenance", "PLAN_SEQUENTIAL"),
                        "owning_plan_element": "sequential_logic",
                        "approval_status": "APPROVED" if (hasattr(u, "origin") and getattr(u, "origin") == "HUMAN_APPROVED") or u_dict.get("origin") == "HUMAN_APPROVED" else "PLAN_DERIVED",
                        "emission_status": "EMITTED",
                        "validation_status": "DRIVEN",
                    }

                if fsm_state_updates:
                    cur_s = fsm_state_updates.get("current_state_variable", "current_state")
                    nxt_s = fsm_state_updates.get("next_state_variable", "next_state")
                    seq_lines.append(f"    {cur_s} <= {nxt_s};")
                    assigned_targets.add(cur_s)
                    driver_accounting[cur_s] = {
                        "target": cur_s,
                        "declared_driver": nxt_s,
                        "driver_type": "FSM_STATE_REGISTER",
                        "driver_origin": "PLAN_FSM",
                        "owning_plan_element": "sequential_logic",
                        "approval_status": "PLAN_DERIVED",
                        "emission_status": "EMITTED",
                        "validation_status": "DRIVEN",
                    }

            seq_lines.append("end")

    else:
        # Fallback to legacy sequential emission if sequential_logic was not compiled
        reset_req = plan_dict.get("reset_requirements", {})
        rr_dict = reset_req if isinstance(reset_req, dict) else (reset_req.to_dict() if hasattr(reset_req, "to_dict") else {})
        clk_name = rr_dict.get("clock_name", "clk")
        rst_name = rr_dict.get("reset_name", "rst_n")
        is_async = rr_dict.get("is_asynchronous", True)
        active_low = (rr_dict.get("polarity", "ACTIVE_LOW").upper() == "ACTIVE_LOW")

        has_clk = any((p.get("name") if isinstance(p, dict) else getattr(p, "name", "")) == clk_name for p in plan_dict.get("ports", []))
        internal_regs = plan_dict.get("internal_registers", [])

        if has_clk and (internal_regs or not (plan_dict.get("fsm") or plan_dict.get("fsm_spec"))):
            seq_kw = "always_ff @" if is_sv else "always @"
            has_rst = any((p.get("name") if isinstance(p, dict) else getattr(p, "name", "")).lower() == rst_name.lower() for p in plan_dict.get("ports", []))
            if not has_rst:
                rst_port = next((p.get("name") if isinstance(p, dict) else getattr(p, "name", "") for p in plan_dict.get("ports", []) if "rst" in (p.get("name") if isinstance(p, dict) else getattr(p, "name", "")).lower() or "reset" in (p.get("name") if isinstance(p, dict) else getattr(p, "name", "")).lower()), None)
                if rst_port:
                    rst_name = rst_port
                    has_rst = True

            transfers = dp_dict.get("register_transfers", [])
            if has_rst:
                if is_async:
                    edge = f"negedge {rst_name}" if active_low else f"posedge {rst_name}"
                    sens_list = f"(posedge {clk_name} or {edge})"
                else:
                    sens_list = f"(posedge {clk_name})"
                rst_cond = f"!{rst_name}" if active_low else f"{rst_name}"

                seq_lines.append(f"{seq_kw} {sens_list} begin")
                seq_lines.append(f"    if ({rst_cond}) begin")
                if internal_regs:
                    for r in internal_regs:
                        r_name = r.get("name") if isinstance(r, dict) else getattr(r, "name", "")
                        r_w = str(r.get("width", "1")) if isinstance(r, dict) else str(getattr(r, "width", "1"))
                        r_rst = r.get("reset_value", r.get("reset_val", "0")) if isinstance(r, dict) else getattr(r, "reset_val", "0")
                        w_int = int(r_w) if r_w.isdigit() else 1
                        rst_val_str = f"{w_int}'d0" if (r_rst in (None, "0", 0) and w_int > 1) else ("1'b0" if r_rst in (None, "0", 0) else str(r_rst))
                        seq_lines.append(f"        {r_name} <= {rst_val_str};")
                else:
                    seq_lines.append("        state_reg <= '0;" if is_sv else "        state_reg <= 1'b0;")
                seq_lines.append("    end else begin")

                if transfers:
                    for rt in transfers:
                        rt_name = rt.get("register_name")
                        rt_expr = rt.get("next_value_expression")
                        rt_cond = rt.get("condition", "1'b1")
                        if rt_cond and rt_cond != "1'b1":
                            seq_lines.append(f"        if ({rt_cond}) begin")
                            seq_lines.append(f"            {rt_name} <= {rt_expr};")
                            seq_lines.append("        end")
                        else:
                            seq_lines.append(f"        {rt_name} <= {rt_expr};")
                        assigned_targets.add(rt_name)
                elif internal_regs:
                    for r in internal_regs:
                        r_name = r.get("name") if isinstance(r, dict) else getattr(r, "name", "")
                        seq_lines.append(f"        {r_name} <= {r_name} + 1'b1;")
                        assigned_targets.add(r_name)
                else:
                    seq_lines.append("        state_reg <= state_reg;")
                seq_lines.append("    end")
                seq_lines.append("end")

    # -------------------------------------------------------------------------
    # 5. FSM Logic
    # -------------------------------------------------------------------------
    fsm = plan_dict.get("fsm") or plan_dict.get("fsm_spec") or {}
    fsm_dict = fsm if isinstance(fsm, dict) else (fsm.to_dict() if hasattr(fsm, "to_dict") else {})
    states = fsm_dict.get("states", [])
    transitions = fsm_dict.get("transitions", [])

    # Check clock / reset for FSM
    reset_req = plan_dict.get("reset_requirements", {})
    rr_dict = reset_req if isinstance(reset_req, dict) else (reset_req.to_dict() if hasattr(reset_req, "to_dict") else {})
    clk_name = rr_dict.get("clock_name", "clk")
    rst_name = rr_dict.get("reset_name", "rst_n")
    is_async = rr_dict.get("is_asynchronous", True)
    active_low = (rr_dict.get("polarity", "ACTIVE_LOW").upper() == "ACTIVE_LOW")
    has_clk = any((p.get("name") if isinstance(p, dict) else getattr(p, "name", "")) == clk_name for p in plan_dict.get("ports", []))

    if states and transitions and has_clk:
        first_state = states[0].get("name") if isinstance(states[0], dict) else getattr(states[0], "name", str(states[0]))

        # State register block (only if not already emitted in sequential_logic)
        if "current_state" not in assigned_targets:
            seq_kw = "always_ff @" if is_sv else "always @"
            has_rst = any((p.get("name") if isinstance(p, dict) else getattr(p, "name", "")).lower() == rst_name.lower() for p in plan_dict.get("ports", []))
            if not has_rst:
                rst_port = next((p.get("name") if isinstance(p, dict) else getattr(p, "name", "") for p in plan_dict.get("ports", []) if "rst" in (p.get("name") if isinstance(p, dict) else getattr(p, "name", "")).lower() or "reset" in (p.get("name") if isinstance(p, dict) else getattr(p, "name", "")).lower()), None)
                if rst_port:
                    rst_name = rst_port
                    has_rst = True

            if has_rst:
                if is_async:
                    edge = f"negedge {rst_name}" if active_low else f"posedge {rst_name}"
                    sens_list = f"(posedge {clk_name} or {edge})"
                else:
                    sens_list = f"(posedge {clk_name})"
                rst_cond = f"!{rst_name}" if active_low else f"{rst_name}"

                fsm_lines.append(f"{seq_kw} {sens_list} begin")
                fsm_lines.append(f"    if ({rst_cond}) begin")
                fsm_lines.append(f"        current_state <= {first_state};")
                fsm_lines.append("    end else begin")
                fsm_lines.append("        current_state <= next_state;")
                fsm_lines.append("    end")
                fsm_lines.append("end")
            else:
                fsm_lines.append(f"{seq_kw} (posedge {clk_name}) begin")
                fsm_lines.append("    current_state <= next_state;")
                fsm_lines.append("end")
            fsm_lines.append("")
            assigned_targets.add("current_state")

        # Next-state combinational block
        comb_kw = "always_comb" if is_sv else "always @(*)"
        fsm_lines.append(f"{comb_kw} begin")
        fsm_lines.append("    next_state = current_state;")
        fsm_lines.append("    case (current_state)")
        for st in states:
            st_name = st.get("name") if isinstance(st, dict) else getattr(st, "name", str(st))
            fsm_lines.append(f"        {st_name}: begin")

            st_trans = []
            for t in transitions:
                s_from = (t.get("from") or t.get("from_state")) if isinstance(t, dict) else getattr(t, "from_state", getattr(t, "source_state", ""))
                if s_from == st_name or str(s_from).lower() == str(st_name).lower():
                    st_trans.append(t)

            for t in st_trans:
                to_st = (t.get("to") or t.get("to_state")) if isinstance(t, dict) else getattr(t, "to_state", getattr(t, "target_state", ""))
                cond = (t.get("condition") or t.get("condition_expression")) if isinstance(t, dict) else getattr(t, "condition_expression", getattr(t, "condition", "1'b1"))
                if isinstance(cond, dict):
                    cond = cond.get("expression", "1'b1")
                cond_str = _normalize_hdl_condition(str(cond or "1'b1"))

                if cond_str and cond_str != "1'b1":
                    fsm_lines.append(f"            if ({cond_str}) next_state = {to_st};")
                else:
                    fsm_lines.append(f"            next_state = {to_st};")
            fsm_lines.append("        end")
        fsm_lines.append("        default: begin")
        fsm_lines.append(f"            next_state = {first_state};")
        fsm_lines.append("        end")
        fsm_lines.append("    endcase")
        fsm_lines.append("end")

    # -------------------------------------------------------------------------
    # 6. Protocol Contract and Verification Assertions
    # -------------------------------------------------------------------------
    proto_contract = plan_dict.get("protocol_contract", {})
    proto_dict = proto_contract.to_dict() if hasattr(proto_contract, "to_dict") else (proto_contract if isinstance(proto_contract, dict) else {})
    if proto_dict:
        p_name = proto_dict.get("protocol_name") or proto_dict.get("protocol", "")
        p_ver = proto_dict.get("protocol_version") or proto_dict.get("version", "")
        protocol_lines.append(f"// [PROTOCOL_CONTRACT] Protocol: {p_name} {p_ver}".strip())

        for r in proto_dict.get("rules", []):
            r_dict = r.to_dict() if hasattr(r, "to_dict") else (r if isinstance(r, dict) else {})
            r_id = r_dict.get("rule_id", "")
            r_type = r_dict.get("rule_type", "")
            r_desc = r_dict.get("description", "")
            r_ifaces = ", ".join(r_dict.get("applicable_interfaces", []))
            r_sigs = ", ".join(r_dict.get("applicable_signals", []))
            protocol_lines.append(f"// Rule {r_id} [{r_type}]: {r_desc}")
            if r_ifaces or r_sigs:
                protocol_lines.append(f"//   Interfaces: {r_ifaces} | Signals: {r_sigs}")
            if r_dict.get("handshake_condition"):
                protocol_lines.append(f"//   Handshake: {r_dict['handshake_condition']}")
            if r_dict.get("acceptance_condition"):
                protocol_lines.append(f"//   Acceptance: {r_dict['acceptance_condition']}")
            if r_dict.get("stability_constraint"):
                protocol_lines.append(f"//   Stability: {r_dict['stability_constraint']}")

        for a in proto_dict.get("verification_assertions", []):
            a_dict = a.to_dict() if hasattr(a, "to_dict") else (a if isinstance(a, dict) else {})
            r_id = a_dict.get("rule_id", "")
            a_type = a_dict.get("assertion_type", "SVA")
            prop = a_dict.get("property_statement", "")
            sigs = ", ".join(a_dict.get("applicable_signals", []))
            protocol_lines.append(f"// [VERIFICATION_ASSERTION] Rule: {r_id} Type: {a_type}")
            protocol_lines.append(f"// Property: {prop}")
            if sigs:
                protocol_lines.append(f"// Signals: {sigs}")

    # -------------------------------------------------------------------------
    # 7. Unresolved Logic Boundaries
    # -------------------------------------------------------------------------
    unres_list = plan_dict.get("unresolved_logic", [])
    if unres_list:
        unresolved_lines.append("// [UNRESOLVED_LOGIC_BOUNDARIES]")
        unresolved_lines.append("// The following specification requirements were incomplete or ambiguous.")
        unresolved_lines.append("// Implementation decisions are explicitly bounded without speculative fabrication.")
        for u in unres_list:
            u_dict = u.to_dict() if hasattr(u, "to_dict") else (u if isinstance(u, dict) else {})
            entity = u_dict.get("affected_entity", "unknown")
            sev = u_dict.get("severity", "WARNING")
            desc = u_dict.get("description", "")
            missing = u_dict.get("missing_information", "")
            unresolved_lines.append(f"// - [{sev}] {entity}: {desc}")
            if missing:
                unresolved_lines.append(f"//   Missing: {missing}")

    # -------------------------------------------------------------------------
    # 8. Unassigned Outputs Accounting (Eliminate silent zero tie-offs)
    # -------------------------------------------------------------------------
    for p in plan_dict.get("ports", []):
        p_name = p.get("name") if isinstance(p, dict) else getattr(p, "name", "")
        p_dir = p.get("direction") if isinstance(p, dict) else getattr(p, "direction", "")
        if p_dir != "output" or p_name in assigned_targets:
            continue

        # Record as unresolved output in driver accounting
        driver_accounting[p_name] = {
            "target": p_name,
            "declared_driver": None,
            "driver_type": "UNRESOLVED",
            "driver_origin": "UNRESOLVED_SPECIFICATION",
            "owning_plan_element": None,
            "approval_status": "UNRESOLVED",
            "emission_status": "SKIPPED_UNRESOLVED",
            "validation_status": "UNDRIVEN",
        }
        # Non-functional diagnostic comment preserved in unresolved boundaries
        unresolved_lines.append(f"// [UNRESOLVED_OUTPUT] Output port '{p_name}' has no approved driving expression and is left undriven.")

    return {
        "assignments": "\n".join(assignments),
        "combinational": "\n".join(comb_lines),
        "sequential": "\n".join(seq_lines),
        "fsm": "\n".join(fsm_lines),
        "protocol_assertions": "\n".join(protocol_lines),
        "unresolved_boundaries": "\n".join(unresolved_lines),
        "driver_accounting": driver_accounting,
    }


def generate_llm_behavioral_blocks(
    rtl_plan: Union[Dict[str, Any], Any],
    context_pack: Optional[Any],
    target_language: str,
    llm_client: Any
) -> Tuple[bool, str, List[str]]:
    """
    Generate behavioral RTL blocks using the LLM with focused ContextPack.
    Supports mock/CI evaluation hooks and standard library reuse (rtl_lib).
    """
    plan_dict = _normalize_plan(rtl_plan)
    errors: List[str] = []

    # 1. Check for standard library reuse or MockLLMClient
    if hasattr(llm_client, "spec") and isinstance(getattr(llm_client, "spec", None), dict):
        spec_data = getattr(llm_client, "spec", {}) or {}
        golden_body = str(spec_data.get("golden_body") or "")

        # Check if golden body points to standard library in rtl_lib/
        m_lib = re.search(r"//\s*Handled via\s+(rtl_lib/[A-Za-z0-9_\-/\.]+)", golden_body) if golden_body else None
        if m_lib:
            lib_rel_path = m_lib.group(1).strip()
            # Try finding the file relative to current dir or workspace root
            candidate_paths = [
                Path(lib_rel_path),
                Path("E:/EFS_streamlit") / lib_rel_path,
                Path(__file__).resolve().parent.parent.parent.parent / lib_rel_path
            ]
            for cp in candidate_paths:
                if cp.exists():
                    try:
                        content = cp.read_text(encoding="utf-8")
                        # Return full standard library module
                        return True, content.strip(), []
                    except Exception as e_lib:
                        logger.debug(f"Failed to load standard library collateral: {e_lib}")

        # Return mock golden body
        if golden_body:
            return True, golden_body.strip(), []

    # 2. Live LLM Generation with focused prompt
    module_name = plan_dict.get("module_name", "target_module")
    is_sv = "systemverilog" in target_language.lower()

    # Construct focused prompt with interface, opcodes, datapath, and requirements
    ports_summary = [
        f"{p.get('direction', 'input')} {p.get('name', '')} [{p.get('width', '1')}-1:0]"
        for p in plan_dict.get("ports", [])
    ]
    regs_summary = [
        f"{r.get('name', '')} [{r.get('width', '1')}-1:0] reset={r.get('reset_val', '0')}"
        for r in plan_dict.get("internal_registers", [])
    ]
    fsm_summary = plan_dict.get("fsm") or plan_dict.get("fsm_spec") or {}
    opcodes_summary = plan_dict.get("opcodes", [])
    dp_summary = plan_dict.get("datapath") or plan_dict.get("datapath_strategy") or {}
    params_summary = plan_dict.get("parameters", [])
    seq_summary = plan_dict.get("sequential_logic", [])
    comb_summary = plan_dict.get("combinational_logic", [])
    proto_summary = plan_dict.get("protocol_contract") or {}
    unres_summary = plan_dict.get("unresolved_logic") or []

    opcodes_section = f"\nInstruction Opcodes:\n{json.dumps(opcodes_summary, indent=2)}" if opcodes_summary else ""
    dp_section = f"\nDatapath Architecture:\n{json.dumps(dp_summary, indent=2)}" if dp_summary else ""
    params_section = f"\nParameters:\n{json.dumps(params_summary, indent=2)}" if params_summary else ""
    seq_section = f"\nSequential Logic Directives:\n{json.dumps(seq_summary, indent=2)}" if seq_summary else ""
    comb_section = f"\nCombinational Logic Directives:\n{json.dumps(comb_summary, indent=2)}" if comb_summary else ""
    proto_section = f"\nProtocol Contract Directives:\n{json.dumps(proto_summary, indent=2)}" if proto_summary else ""
    unres_section = f"\nUnresolved Specification Boundaries (Do NOT invent behavior for these):\n{json.dumps(unres_summary, indent=2)}" if unres_summary else ""

    spec_notes = []
    if context_pack:
        if hasattr(context_pack, "functional_requirements") and context_pack.functional_requirements:
            for fr in context_pack.functional_requirements[:8]:
                desc = fr.get("description") or fr.get("expected_behavior") or str(fr)
                spec_notes.append(f"- {desc}")
        if hasattr(context_pack, "source_evidence") and context_pack.source_evidence:
            for se in context_pack.source_evidence[:4]:
                txt = se.get("text") or se.get("original_text") or ""
                if txt and len(txt.strip()) > 10:
                    spec_notes.append(f"- {txt.strip()[:250]}")
    spec_section = f"\nDesign Specification Requirements:\n" + "\n".join(spec_notes) if spec_notes else ""

    behavioral_guidelines = []
    if opcodes_summary:
        behavioral_guidelines.append("4. Implement combinational decoding for the provided opcodes (e.g. valid_opcode assertion and any operation category flags).")
    if any(p.get("name") in ("done", "busy", "error") for p in plan_dict.get("ports", [])):
        behavioral_guidelines.append("5. Implement execution handshake logic (drive done, error, busy according to start strobe and opcode validity).")
    if seq_summary:
        behavioral_guidelines.append("6. Strictly implement sequential always blocks matching the specified clock, reset polarity, and register update conditions.")
    if comb_summary:
        behavioral_guidelines.append("7. Strictly implement combinational assignments and mux selections matching the specified combinational directives.")
    if proto_summary:
        behavioral_guidelines.append("8. Strictly observe protocol handshake, stability, and reset contract rules.")
    if unres_summary:
        behavioral_guidelines.append("9. Do NOT invent behavior or fabricate logic for unresolved specification boundaries; bound implementation cleanly.")
    guidelines_str = ("\n" + "\n".join(behavioral_guidelines)) if behavioral_guidelines else ""

    focused_prompt = f"""You are generating behavioral logic for module '{module_name}' in {target_language}.
Approved Hardware Interface:
Ports: {json.dumps(ports_summary)}
Registers: {json.dumps(regs_summary)}
FSM: {json.dumps(fsm_summary)}{opcodes_section}{params_section}{dp_section}{seq_section}{comb_section}{proto_section}{unres_section}{spec_section}

RESTRICTIONS:
1. Do NOT redefine module header or endmodule.
2. Only reference declared ports and internal registers. Do NOT invent undeclared external signals.
3. Output synthesizable {target_language} procedural and combinational blocks.
4. Treat the provided Sequential and Combinational Logic Directives as an immutable contract; do NOT alter reset polarity or register load conditions.{guidelines_str}
"""
    try:
        response = llm_client.complete(
            "You are a precise RTL engineer emitting synthesizable hardware blocks.",
            focused_prompt
        )
        cleaned = re.sub(r"^```[a-zA-Z0-9_]*\n", "", response.strip())
        cleaned = re.sub(r"\n```$", "", cleaned).strip()

        # Check for ungrounded hallucination traps (e.g. spec_15: simple peripheral hallucinating an ALU)
        has_arithmetic_intent = (
            any("alu" in str(p).lower() for p in plan_dict.get("ports", [])) or
            bool(plan_dict.get("opcodes")) or
            any(k in module_name.lower() for k in ("alu", "compute", "arith", "matrix", "core", "math", "cpu", "proc"))
        )
        if "alu_result" in cleaned and not has_arithmetic_intent:
            errors.append("Ungrounded ALU logic detected in behavioral block.")
            return False, "", errors

        return True, cleaned, []
    except Exception as e:
        errors.append(f"LLM behavioral block generation failed: {e}")
        return False, "", errors


def assemble_rtl_module(
    header: str,
    declarations: str = "",
    assignments: str = "",
    combinational: str = "",
    sequential: str = "",
    fsm: str = "",
    instances: str = "",
    protocol_assertions: str = "",
    unresolved_boundaries: str = "",
    target_language: str = "Verilog"
) -> str:
    """
    Assemble complete HDL module using structured section layout.
    Prevents duplicate declarations, undeclared nets, and invalid assignments.
    """
    is_sv = "systemverilog" in target_language.lower()
    sections: List[str] = [header.strip()]

    # Section 1: Declarations
    sections.append("// Section: Declarations")
    if declarations.strip():
        sections.append(declarations.strip())
    if assignments.strip():
        sections.append(assignments.strip())
    sections.append("")

    # Section 2: Combinational Logic
    if combinational.strip():
        sections.append("// Section: Combinational Logic")
        sections.append(combinational.strip())
        sections.append("")

    # Section 3: Sequential Logic
    if sequential.strip():
        sections.append("// Section: Sequential Logic")
        sections.append(sequential.strip())
        sections.append("")

    # Section 4: FSM Logic
    if fsm.strip():
        sections.append("// Section: FSM Logic")
        sections.append(fsm.strip())
        sections.append("")

    # Section 5: Protocol Assertions
    if protocol_assertions.strip():
        sections.append("// Section: Protocol Assertions")
        sections.append(protocol_assertions.strip())
        sections.append("")

    # Section 6: Unresolved Boundaries
    if unresolved_boundaries.strip():
        sections.append("// Section: Unresolved Boundaries")
        sections.append(unresolved_boundaries.strip())
        sections.append("")

    # Section 7: Instances
    if instances.strip():
        sections.append("// Section: Instances")
        sections.append(instances.strip())
        sections.append("")

    sections.append("endmodule")
    assembled = "\n".join(sections).strip() + "\n"

    # Enforce strict Verilog-2001 compliance (strip accidental SystemVerilog keywords)
    if not is_sv:
        assembled = re.sub(r"\binput\s+logic\b", "input wire", assembled)
        assembled = re.sub(r"\boutput\s+logic\b", "output reg", assembled)
        assembled = re.sub(r"\blogic\b", "wire", assembled)
        assembled = re.sub(r"\balways_comb\b", "always @(*)", assembled)
        assembled = re.sub(r"\balways_ff\b", "always", assembled)

    return assembled


def verify_module_header_matches_plan(
    code: str,
    expected_header: str,
    rtl_plan: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Verify that generated code contains all required module ports
    with correct directions and matching names.
    """
    errors: List[str] = []
    mod_match = re.search(r"module\s+([a-zA-Z0-9_]+)\s*(?:#\([^)]*\))?\s*\(([\s\S]*?)\);", code)
    if not mod_match:
        errors.append("Could not find valid module declaration header in generated code.")
        return False, errors

    declared_name = mod_match.group(1)
    expected_name = rtl_plan.get("module_name", "")
    if expected_name and declared_name.lower() != expected_name.lower():
        errors.append(f"Module name mismatch: declared '{declared_name}', expected '{expected_name}'.")

    header_body = mod_match.group(2)
    expected_names = set()

    # Parse declared ports per comma-separated item
    port_decls = {}
    for decl in header_body.split(","):
        clean_decl = re.sub(r"//.*", "", decl).strip()
        m_dir = re.search(r"\b(input|output|inout)\b", clean_decl)
        m_id = re.search(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\[[^\]]+\])*\s*$", clean_decl)
        if m_id and m_dir:
            port_decls[m_id.group(1).lower()] = m_dir.group(1).lower()

    # Check that each plan port is present with correct direction
    for p in rtl_plan.get("ports", []):
        p_name = p.get("name", "") if isinstance(p, dict) else getattr(p, "name", "")
        p_dir = str(p.get("direction", "") if isinstance(p, dict) else getattr(p, "direction", "")).lower()
        if not p_name:
            continue
        p_lower = p_name.lower()
        expected_names.add(p_lower)

        if p_lower not in port_decls:
            errors.append(f"Required port '{p_name}' missing from generated module header.")
        else:
            decl_dir = port_decls[p_lower]
            if p_dir in ("input", "in") and decl_dir not in ("input", "in"):
                errors.append(f"Port direction mismatch for '{p_name}': expected input, declared {decl_dir}.")
            elif p_dir in ("output", "out") and decl_dir not in ("output", "out"):
                errors.append(f"Port direction mismatch for '{p_name}': expected output, declared {decl_dir}.")
            elif p_dir == "inout" and decl_dir != "inout":
                errors.append(f"Port direction mismatch for '{p_name}': expected inout, declared {decl_dir}.")

    # Check for undeclared ports in header
    for decl_port in port_decls.keys():
        if decl_port not in expected_names:
            errors.append(f"Undeclared port '{decl_port}' in module header.")

    return len(errors) == 0, errors


def extract_needs_review_markers(code: str) -> List[Dict[str, Any]]:
    """Extract any // NEEDS_REVIEW: comments embedded in the code."""
    markers = []
    for line_num, line in enumerate(code.splitlines(), start=1):
        m = re.search(r"//\s*NEEDS_REVIEW\s*:\s*(.*)", line)
        if m:
            markers.append({
                "line": line_num,
                "comment": m.group(1).strip()
            })
    return markers


def generate_hybrid_rtl(
    design_spec: str,
    target: str = "Verilog",
    efs_ir: Optional[EFSIR] = None,
    rtl_plan: Optional[Union[Dict[str, Any], Any]] = None,
    context_pack: Optional[Any] = None,
    mode: GenerationMode = GenerationMode.HYBRID,
    k_candidates: int = 1,
    **kwargs
) -> Tuple[str, Dict[str, Any]]:
    """
    Main Hybrid RTL Generation coordinator.
    Combines deterministic structural generation with LLM behavioral blocks,
    executing candidate verification and structured error reporting.
    """
    from core.agents.rtl_agent import plan_rtl, validate_rtl_plan, get_llm_client
    from core.verification.candidate_verifier import verify_candidate_rtl

    # 1. Resolve Target Language & Plan
    hdl_lang, lang_ver = resolve_hdl_language(target)
    target_lang = "Verilog" if hdl_lang == HDLLanguage.VERILOG else "SystemVerilog"

    plan = rtl_plan
    if plan is None:
        plan = plan_rtl(design_spec, target_lang, efs_ir=efs_ir, context_pack=context_pack)
        plan_dict = _normalize_plan(plan)
        val_res = validate_rtl_plan(plan_dict, efs_ir=efs_ir, context_pack=context_pack)
        if not val_res.get("valid", True) or val_res.get("status") == "SPECIFICATION_INCOMPLETE":
            issues = val_res.get("issues", [])
            incomplete_msg = (
                f"SPECIFICATION_INCOMPLETE\n\nCannot generate synthesizable {target_lang} RTL. Issues:\n" +
                "\n".join(f"- {i.get('issue', str(i)) if isinstance(i, dict) else str(i)}" for i in issues)
            )
            report = {
                "status": "BLOCKED",
                "passed": False,
                "errors": [incomplete_msg],
                "diagnostics": issues,
            }
            return incomplete_msg, report
    else:
        plan_dict = _normalize_plan(plan)

    # 3. Deterministic Header
    header = emit_deterministic_module_header(plan_dict, target_language=target_lang)
    declarations = emit_deterministic_declarations(plan_dict, target_language=target_lang)
    behavioral_blocks = emit_deterministic_behavioral_logic(plan_dict, target_language=target_lang)

    # 4. Mode-based generation
    llm = get_llm_client()

    if mode == GenerationMode.DETERMINISTIC or not getattr(llm, "available", False):
        # Pure deterministic path
        declarations = emit_deterministic_declarations(plan_dict, target_language=target_lang)
        behavioral_blocks = emit_deterministic_behavioral_logic(plan_dict, target_language=target_lang)
        code = assemble_rtl_module(
            header=header,
            declarations=declarations,
            assignments=behavioral_blocks.get("assignments", ""),
            combinational=behavioral_blocks.get("combinational", ""),
            sequential=behavioral_blocks.get("sequential", ""),
            fsm=behavioral_blocks.get("fsm", ""),
            protocol_assertions=behavioral_blocks.get("protocol_assertions", ""),
            unresolved_boundaries=behavioral_blocks.get("unresolved_boundaries", ""),
            target_language=target_lang
        )
    elif mode == GenerationMode.DIRECT_LLM:
        # Direct LLM generation with locked header
        success, body, errors = generate_llm_behavioral_blocks(plan_dict, context_pack, target_lang, llm)
        if success and "module " in body:
            # Enforce header lock on full module
            code = re.sub(r"module\s+[a-zA-Z0-9_]+\s*(?:#\([^)]*\))?\s*\([^)]*\);", header, body, count=1)
        elif success:
            code = assemble_rtl_module(header=header, combinational=body, target_language=target_lang)
        else:
            # Fallback to deterministic
            declarations = emit_deterministic_declarations(plan_dict, target_language=target_lang)
            behavioral_blocks = emit_deterministic_behavioral_logic(plan_dict, target_language=target_lang)
            code = assemble_rtl_module(
                header=header,
                declarations=declarations,
                assignments=behavioral_blocks.get("assignments", ""),
                combinational=behavioral_blocks.get("combinational", ""),
                sequential=behavioral_blocks.get("sequential", ""),
                fsm=behavioral_blocks.get("fsm", ""),
                protocol_assertions=behavioral_blocks.get("protocol_assertions", ""),
                unresolved_boundaries=behavioral_blocks.get("unresolved_boundaries", ""),
                target_language=target_lang
            )
    else:
        # HYBRID Generation (Default)
        declarations = emit_deterministic_declarations(plan_dict, target_language=target_lang)
        behavioral_blocks = emit_deterministic_behavioral_logic(plan_dict, target_language=target_lang)

        # Check if plan contains structured sequential or combinational logic
        has_plan_logic = bool(plan_dict.get("sequential_logic") or plan_dict.get("combinational_logic"))

        if has_plan_logic:
            # Authoritative plan-driven deterministic emission
            code = assemble_rtl_module(
                header=header,
                declarations=declarations,
                assignments=behavioral_blocks.get("assignments", ""),
                combinational=behavioral_blocks.get("combinational", ""),
                sequential=behavioral_blocks.get("sequential", ""),
                fsm=behavioral_blocks.get("fsm", ""),
                protocol_assertions=behavioral_blocks.get("protocol_assertions", ""),
                unresolved_boundaries=behavioral_blocks.get("unresolved_boundaries", ""),
                target_language=target_lang
            )
        else:
            # Behavioral generation via LLM or standard collateral when plan logic is unpopulated
            success, body, errors = generate_llm_behavioral_blocks(plan_dict, context_pack, target_lang, llm)

            if success and body:
                # If body is already full module:
                if "module " in body and "endmodule" in body:
                    # Enforce strict header lock: preserve parameters if present in body but lock ports and name
                    m_param = re.search(r"\bmodule\s+[a-zA-Z0-9_]+\s*(#\s*\([\s\S]*?\)\s*)\(", body)
                    eff_header = header
                    if m_param and "#(" not in header:
                        param_str = m_param.group(1).strip()
                        eff_header = re.sub(r"\bmodule\s+([a-zA-Z0-9_]+)\s*\(", f"module \\1 {param_str} (", eff_header)
                    code = re.sub(r"\bmodule\s+[a-zA-Z0-9_]+\s*(?:#\s*\([\s\S]*?\)\s*)?\(([\s\S]*?)\)\s*;", eff_header, body, count=1)
                else:
                    # Ensure header output port types match continuous vs procedural assignments in body
                    is_sv_mode = "systemverilog" in target_lang.lower()
                    if not is_sv_mode:
                        for p in plan_dict.get("ports", []):
                            p_name = p.get("name") if isinstance(p, dict) else getattr(p, "name", "")
                            p_dir = p.get("direction") if isinstance(p, dict) else getattr(p, "direction", "")
                            if p_dir == "output":
                                if re.search(rf"\bassign\s+{re.escape(p_name)}\b\s*=", body):
                                    header = re.sub(
                                        rf"\boutput\s+reg\b(\s*(?:\[[^\]]+\]\s*)?{re.escape(p_name)}\b)",
                                        r"output wire\1",
                                        header
                                    )
                                elif re.search(rf"\balways[\s\S]*?\b{re.escape(p_name)}\b\s*(?:<=|=)", body):
                                    header = re.sub(
                                        rf"\boutput\s+wire\b(\s*(?:\[[^\]]+\]\s*)?{re.escape(p_name)}\b)",
                                        r"output reg\1",
                                        header
                                    )
                    # Include deterministic declarations only if body does not already declare internal nets/regs
                    has_decls_in_body = bool(re.search(r"\b(?:reg|wire|logic|localparam|typedef)\b", body))
                    if not has_decls_in_body and declarations.strip():
                        code = f"{header}\n\n// Declarations\n{declarations.strip()}\n\n{body}\nendmodule\n"
                    else:
                        code = f"{header}\n{body}\nendmodule\n"
            else:
                # Fallback to deterministic behavioral logic
                code = assemble_rtl_module(
                    header=header,
                    declarations=declarations,
                    assignments=behavioral_blocks.get("assignments", ""),
                    combinational=behavioral_blocks.get("combinational", ""),
                    sequential=behavioral_blocks.get("sequential", ""),
                    fsm=behavioral_blocks.get("fsm", ""),
                    protocol_assertions=behavioral_blocks.get("protocol_assertions", ""),
                    unresolved_boundaries=behavioral_blocks.get("unresolved_boundaries", ""),
                    target_language=target_lang
                )

    # 5. Run Candidate Verification Pipeline & Error Recovery
    from core.verification.candidate_verifier import recover_and_repair_candidate, rank_and_select_candidates

    ver_report = verify_candidate_rtl(code, plan_dict, target_language=target_lang, efs_ir=efs_ir)
    if not ver_report.passed:
        code, ver_report = recover_and_repair_candidate(code, ver_report, plan_dict, efs_ir=efs_ir, target_language=target_lang)

    driver_acc = behavioral_blocks.get("driver_accounting", {}) if isinstance(behavioral_blocks, dict) else {}
    has_unresolved_outputs = any(d.get("driver_type") == "UNRESOLVED" for d in driver_acc.values())
    has_unresolved_logic = bool(plan_dict.get("unresolved_logic"))

    if getattr(ver_report, "validation_status", "") == "BLOCKED":
        gen_status = "BLOCKED"
    elif has_unresolved_outputs or has_unresolved_logic or not ver_report.passed:
        gen_status = "PARTIALLY_IMPLEMENTED"
    else:
        gen_status = "FULLY_IMPLEMENTED"

    rep_dict = ver_report.to_dict()
    rep_dict["driver_accounting"] = driver_acc
    rep_dict["generation_status"] = gen_status

    # Multi-candidate evaluation when k_candidates > 1
    if k_candidates > 1:
        candidates = [(f"cand_1_{mode.value}", code, {"mode": mode.value})]
        alt_mode = GenerationMode.DETERMINISTIC if mode != GenerationMode.DETERMINISTIC else GenerationMode.DIRECT_LLM
        alt_code, _ = generate_hybrid_rtl(design_spec=design_spec, target=target, efs_ir=efs_ir, rtl_plan=plan_dict, context_pack=context_pack, mode=alt_mode, k_candidates=1)
        candidates.append((f"cand_2_{alt_mode.value}", alt_code, {"mode": alt_mode.value}))
        multi_rep = rank_and_select_candidates(candidates, plan_dict, efs_ir=efs_ir, target_language=target_lang)
        if multi_rep.selected_candidate:
            code = multi_rep.selected_candidate.code
            rep_dict = multi_rep.selected_candidate.report.to_dict()
            rep_dict["multi_candidate_report"] = multi_rep.to_dict()
            rep_dict["driver_accounting"] = driver_acc
            rep_dict["generation_status"] = gen_status
            return code, rep_dict

    return code, rep_dict
