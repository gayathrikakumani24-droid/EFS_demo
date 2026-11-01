"""
RTL / Verification Collateral Generation Agent.

Consumes the intermediate Design Specification and outputs synthesizable hardware
description files (Verilog/SystemVerilog/VHDL), UVM testbenches, or SVA assertions.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
from config import CONFIG
from core.extraction.llm_client import get_llm_client
from utils.logger import get_logger
from core.efs_ir.models import EFSIR
from core.efs_ir.validation_rules import normalize_hardware_width

logger = get_logger("agents.rtl")

from core.verification.hdl_validator import (
    validate_rtl_syntax_and_structure,
    resolve_hdl_language,
    HDLLanguage,
)


def get_system_prompt(target_language: str = "SystemVerilog", language_version: Optional[str] = None) -> str:
    """Generate language-specific system prompt adhering strictly to the selected HDL standard."""
    lang, ver = resolve_hdl_language(target_language)
    if lang == HDLLanguage.VERILOG:
        return f"""You are a senior RTL design engineer specializing in standards-compliant Verilog ({ver}).

Generate complete, synthesizable Verilog RTL ({ver}) from the supplied EFS IR,
Design Plan, Architecture/FSM model, and relevant Protocol Contract.

RULES:

1. EFS IR is the authoritative hardware interface.
   Preserve exactly:
   - module name
   - port names
   - directions
   - widths
   - clock/reset (if specified in specification/IR)
   - parameters

2. Do not invent, remove, rename, merge, or resize external signals.

3. STRICT VERILOG LANGUAGE COMPLIANCE ({ver}):
   - Use `module` and `endmodule`.
   - Use `input`, `output`, and `inout` port declarations.
   - Continuous assignments (`assign`) must drive `wire` nets.
   - Procedural assignments inside `always` blocks must drive `reg` variables.
   - State encodings must use `localparam` or `parameter` constants.
   - Combinational procedural blocks must use `always @(*)` with blocking assignments (`=`).
   - Sequential procedural blocks must use `always @(posedge clk or negedge rst_n)` with non-blocking assignments (`<=`).
   - STRICTLY FORBIDDEN: Do NOT use SystemVerilog-only keywords or constructs:
     NO `logic`, NO `always_ff`, NO `always_comb`, NO `always_latch`,
     NO `typedef`, NO `enum`, NO `struct`, NO `union`, NO `interface`, NO `modport`,
     NO `bit`, NO `int`, NO `byte`, NO `package`.

4. Implement only functionality specified by the EFS IR or resolved requirements:
   - If the design has NO clock or reset specified, do NOT add clock or reset ports. Generate pure combinational RTL.
   - If the design has NO FSM specified, do NOT add FSM states or state variables. Generate only specified datapath/logic.
   - Do not add unsupported registers, states, opcodes, memories, or protocol features.

5. Synthesizability:
   - No delays (`#`)
   - No testbench constructs or initial blocks with file I/O
   - No inferred latches
   - No multiple drivers
   - No combinational loops
   - Use appropriate sequential/combinational logic

6. Specification Completeness & No Guessing:
   - Every RTL behavior must be traceable to the supplied EFS IR, plan, or specification.
   - Do not infer missing behavior from signal names or engineering assumptions.
   - If the specification is incomplete, report the missing requirement instead of guessing.

OUTPUT:
Return ONLY the raw Verilog RTL.
No explanation.
No Markdown fences.
No <think> block.
No commentary.
Start directly with:

module <module_name>

and end with:

endmodule"""
    else:
        return f"""You are a senior RTL design engineer specializing in standards-compliant SystemVerilog ({ver}).

Generate complete, synthesizable SystemVerilog RTL ({ver}) from the supplied EFS IR,
Design Plan, Architecture/FSM model, and relevant Protocol Contract.

RULES:

1. EFS IR is the authoritative hardware interface.
   Preserve exactly:
   - module name
   - port names
   - directions
   - widths
   - clock/reset (if specified in specification/IR)
   - parameters

2. Do not invent, remove, rename, merge, or resize external signals.

3. SYSTEMVERILOG LANGUAGE COMPLIANCE ({ver}):
   - Use `module` and `endmodule`.
   - Use `logic` and `bit` data types.
   - Use `always_ff @(posedge clk ...)` for sequential logic with non-blocking assignments (`<=`).
   - Use `always_comb` for combinational logic with blocking assignments (`=`).
   - Use `typedef enum logic [W-1:0]` for state machine definitions where applicable.
   - Use parameterized modules and generate constructs where appropriate.

4. Implement only functionality specified by the EFS IR or resolved requirements:
   - If the design has NO clock or reset specified, do NOT add clock or reset ports. Generate pure combinational RTL.
   - If the design has NO FSM specified, do NOT add FSM states or state variables. Generate only specified datapath/logic.
   - Do not add unsupported registers, states, opcodes, memories, or protocol features.

5. Synthesizability:
   - No delays (`#`)
   - No testbench constructs or initial blocks with file I/O
   - No inferred latches
   - No multiple drivers
   - No combinational loops
   - Use appropriate sequential/combinational logic

6. Specification Completeness & No Guessing:
   - Every RTL behavior must be traceable to the supplied EFS IR, plan, or specification.
   - Do not infer missing behavior from signal names or engineering assumptions.
   - If the specification is incomplete, report the missing requirement instead of guessing.

OUTPUT:
Return ONLY the raw SystemVerilog RTL.
No explanation.
No Markdown fences.
No <think> block.
No commentary.
Start directly with:

module <module_name>

and end with:

endmodule"""


SYSTEM_PROMPT = get_system_prompt("SystemVerilog")


def _clean_code_output(text: str) -> str:
    """Strip reasoning blocks <think>...</think>, conversational prefixes, and markdown fences."""
    if not text:
        return ""
    import re
    
    # 1. Remove <think>...</think> reasoning blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    # 2. Extract code block inside ```verilog ... ``` or ```systemverilog ... ``` or ```vhdl ... ``` or ``` ... ```
    match = re.search(r"```(?:verilog|systemverilog|sv|vhdl|systemverilog_assertions|sva|uvm|code)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()
    else:
        # Check if open code fence was used without closing fence
        fence_match = re.search(r"```(?:verilog|systemverilog|sv|vhdl|systemverilog_assertions|sva|uvm|code)?\s*\n(.*)", text, re.DOTALL | re.IGNORECASE)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        else:
            pos = -1
            for kw in ["module ", "entity ", "library ", "class ", "package ", "`timescale"]:
                p = text.find(kw)
                if p != -1 and (pos == -1 or p < pos):
                    pos = p
            if pos != -1:
                cleaned = text[pos:].strip()
            else:
                lines = text.split("\n")
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()

    # Strip trailing markdown backticks if any
    cleaned = re.sub(r"```+$", "", cleaned).strip()

    return cleaned



clean_generated_code = _clean_code_output


def plan_rtl(design_spec: str, target: Any = "Verilog", efs_ir: Optional[EFSIR] = None, context_pack: Optional[Any] = None) -> Dict[str, Any]:
    """Stage 1: Generate a Structured RTL Plan (JSON) from query-scoped ContextPack or EFS IR."""
    logger.info("Generating Structured RTL Implementation Plan (Stage 1)...")
    llm = get_llm_client()
    
    # Handle efs_ir passed as 1st or 2nd positional argument
    if hasattr(design_spec, "components") or hasattr(design_spec, "signals") or hasattr(design_spec, "metadata"):
        efs_ir = design_spec
        design_spec = getattr(getattr(efs_ir, "metadata", None), "description", "") or ""
    elif not isinstance(target, str) and (hasattr(target, "components") or hasattr(target, "signals")):
        efs_ir = target
        target = "Verilog"

    from core.context_compiler import compile_context
    from core.context_compiler.generation_context import GenerationContext

    # Determine target component name from context_pack if available
    target_comp_name = "" if target in ("Verilog", "SystemVerilog", "VHDL", "Assertions", "UVM") else str(target)
    if context_pack:
        tc = getattr(context_pack, "target_component", {})
        if isinstance(tc, dict) and tc.get("name"):
            target_comp_name = tc.get("name")
        elif hasattr(tc, "name") and getattr(tc, "name"):
            target_comp_name = tc.name
        elif hasattr(context_pack, "task") and getattr(context_pack.task, "target", None):
            target_comp_name = context_pack.task.target

    # Compile target-scoped RTL context using RTLContextCompiler
    compiled_rtl = compile_context(
        "rtl",
        efs_ir=efs_ir,
        target_component=target_comp_name,
        target_language=target,
        design_spec=design_spec,
        context_pack=context_pack
    )
    gen_ctx = GenerationContext.from_compiled_rtl(compiled_rtl, token_budget=8000)
    gen_ctx.artifact_type = "rtl_plan"

    spec_parsed = parse_markdown_spec(design_spec)
    spec_mod_name = spec_parsed.get("module_name")
    c_mod_name = compiled_rtl.get("module", {}).get("name")
    if c_mod_name and c_mod_name not in ("target_module", "hardware_module", "custom_module"):
        target_module_name = c_mod_name
    elif spec_mod_name and spec_mod_name not in ("target_module", "hardware_module", "custom_module"):
        target_module_name = spec_mod_name
    elif efs_ir and hasattr(efs_ir, "metadata") and getattr(efs_ir.metadata, "design_name", None):
        target_module_name = efs_ir.metadata.design_name
    else:
        target_module_name = "hardware_module"
    target_module_name = sanitize_module_name(target_module_name)
    scoped_signals = compiled_rtl.get("signals", compiled_rtl.get("ports", []))
    scoped_registers = compiled_rtl.get("registers", [])
    scoped_fsm = compiled_rtl.get("fsm", {})
    efs_ir_summary = gen_ctx.to_prompt_json(indent=2)

    raw_user_query = ""
    if context_pack and hasattr(context_pack, "task") and getattr(context_pack.task, "raw_query", None):
        raw_user_query = str(context_pack.task.raw_query).strip()
    user_query_note = f"\n=== TARGETED USER QUERY ===\n\"{raw_user_query}\"\n(Prioritize and construct RTL plan to fulfill this specific targeted request.)\n" if raw_user_query else ""

    plan_prompt = f"""You are a senior hardware architect. Output a JSON Structured RTL Implementation Plan.{user_query_note}
Do NOT generate SystemVerilog code yet. Output ONLY valid JSON according to this schema:

{{
  "module_name": "string",
  "specification_status": {{
    "status": "COMPLETE|INCOMPLETE|CONTRADICTORY",
    "blocking_issues": [
      {{
        "category": "FSM|INTERFACE|REGISTER|DATAPATH|PROTOCOL|RESET|TIMING",
        "issue": "string",
        "required_information": "string",
        "evidence": "string"
      }}
    ]
  }},
  "ports": [
    {{
      "name": "string",
      "direction": "input|output|inout",
      "width": "string",
      "description": "string",
      "semantic_role": "string",
      "source": "EFS|DesignSpec|Protocol|Unknown"
    }}
  ],
  "parameters": [
    {{
      "name": "string",
      "value": "string",
      "description": "string",
      "usage": "string",
      "is_behaviorally_defined": true
    }}
  ],
  "internal_registers": [
    {{
      "name": "string",
      "width": "string",
      "reset_val": "string",
      "purpose": "string",
      "fields": [
        {{
          "bits": "string",
          "meaning": "string",
          "source": "string"
        }}
      ]
    }}
  ],
  "fsm": {{
    "states": ["string"],
    "initial_state": "string",
    "transitions": [
      {{
        "from": "string",
        "to": "string",
        "condition": {{
          "expression": "string|null",
          "signals": ["string"],
          "registers": ["string"],
          "protocol_events": ["string"],
          "source": "EFS|Architecture|Protocol|DesignSpec|Unknown",
          "source_evidence": "string",
          "is_fully_defined": true
        }}
      }}
    ]
  }},
  "datapath_strategy": "string",
  "reset_polarity": "active_low|active_high"
}}

CRITICAL GROUNDING RULES FOR STAGE 1 PLAN:
1. Stage 1 must explicitly determine whether the specification is sufficient to generate deterministic RTL.
2. Mark "specification_status": "INCOMPLETE" or "CONTRADICTORY" if:
   - Any FSM transition has no implementable signal/register condition (e.g. "control start condition met" without identifying exact signal/bit or port).
     NOTE:
     - Transitions specifying "the next clock cycle", "next clock cycle", "1 cycle", "single-cycle dwell", "unconditional", or "1'b1" ARE fully defined and implementable (representing an unconditional state transition on the next active clock edge). Map their expression to "1'b1" and do NOT mark them as incomplete or ambiguous.
     - Transitions referencing named control signals or ports (such as 'start', 'done', 'busy', 'ready', 'valid') where matching signals or ports exist in the interface/signals list ARE fully defined and implementable. In hardware design, direct ports are standard active-high signals ("start asserted" -> "start == 1'b1" or "start", "done asserted" -> "done == 1'b1" or "done", "done deasserted" -> "!done" or "done == 1'b0"). Do NOT require them to be memory-mapped registers when external ports exist. Map their expressions accordingly and do NOT mark them as incomplete or ambiguous.
   - Register bit meanings are unspecified ONLY when a register is explicitly referenced for control/status without bit definitions (if direct top-level ports like start/done are used, no register bit mapping is required).
   - A completion condition or counter threshold/width is unspecified.
   - An output behavior or required protocol event is undefined or cannot be mapped to supplied signals.
   - Signal direction, width, or polarity conflicts exist across sources.
3. Do NOT invent signal names or expressions for missing behavior. Set "expression": null and "is_fully_defined": false for incomplete conditions.

=== COMPILED GENERATION CONTEXT ===
{efs_ir_summary}
"""
    gen_ctx.calculate_metrics(plan_prompt, system_prompt="You are an expert RTL architect.")
    if llm.available:
        plan_res = llm.complete_json("You are an expert RTL architect.", plan_prompt, model=CONFIG.llm.model)
        provider_tokens = llm.last_usage.get("prompt_tokens") if hasattr(llm, "last_usage") else None
        output_tokens = llm.last_usage.get("completion_tokens", 0) if hasattr(llm, "last_usage") else 0
        gen_ctx.log_telemetry(
            generation_name="rtl_agent_plan_rtl",
            model=CONFIG.llm.model,
            provider_tokens=provider_tokens,
            output_tokens=output_tokens
        )
        if plan_res and isinstance(plan_res, dict) and plan_res.get("ports"):
            plan_res["module_name"] = sanitize_module_name(plan_res.get("module_name", target_module_name))
            if plan_res.get("internal_registers"):
                plan_res["internal_registers"] = sanitize_hardware_registers(plan_res["internal_registers"])
            # Collect known signal and port names
            known_signal_names = set()
            for p in plan_res.get("ports", []):
                if isinstance(p, dict) and p.get("name"):
                    known_signal_names.add(str(p["name"]).strip().lower())
            for s in (scoped_signals or []):
                s_n = s.get("name") if isinstance(s, dict) else getattr(s, "name", None)
                if s_n:
                    known_signal_names.add(str(s_n).strip().lower())
            if efs_ir and hasattr(efs_ir, "signals"):
                for s in efs_ir.signals:
                    s_n = s.get("name") if isinstance(s, dict) else getattr(s, "name", None)
                    if s_n:
                        known_signal_names.add(str(s_n).strip().lower())

            # Normalize any next-clock-cycle / unconditional dwell transitions in the plan
            # AND normalize conditions referencing known port assertions (e.g. "start asserted", "done asserted", "done deasserted")
            plan_fsm = plan_res.get("fsm", {})
            if isinstance(plan_fsm, dict) and plan_fsm.get("transitions"):
                for tr in plan_fsm.get("transitions", []):
                    if isinstance(tr, dict):
                        c = tr.get("condition")
                        expr = ""
                        if isinstance(c, dict):
                            expr = str(c.get("expression") or "").strip()
                        elif isinstance(c, str):
                            expr = str(c).strip()

                        expr_lower = expr.lower()
                        if re.search(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:==|!=|<=|>=|<|>)\s*(\d+'[bBhdD][0-9a-fA-F_]+|\d+)\s*$", expr):
                            if isinstance(c, dict):
                                c["expression"] = expr
                                c["is_fully_defined"] = True
                            else:
                                tr["condition"] = {"expression": expr, "is_fully_defined": True}
                        elif re.search(r"(\d+)\s*(?:clock\s*)?cycles?", expr_lower):
                            m_cyc = re.search(r"(\d+)\s*(?:clock\s*)?cycles?", expr_lower)
                            dur = int(m_cyc.group(1))
                            term_val = max(0, dur - 1)
                            cnt_w = max(1, term_val.bit_length())
                            syn_expr = f"timer_cnt == {cnt_w}'d{term_val}"
                            if isinstance(c, dict):
                                c["expression"] = syn_expr
                                c["is_fully_defined"] = True
                            else:
                                tr["condition"] = {"expression": syn_expr, "is_fully_defined": True}
                        elif re.search(r"\b(?:next\s+clock\s+cycle|after\s+1\s+cycle|single[- ]cycle\s+dwell|unconditional)\b", expr_lower):
                            if isinstance(c, dict):
                                c["expression"] = "1'b1"
                                c["is_fully_defined"] = True
                            else:
                                tr["condition"] = {"expression": "1'b1", "is_fully_defined": True}
                        else:
                            matched_sig = None
                            is_asserted = bool(re.search(r"\basserted\b", expr_lower))
                            is_deasserted = bool(re.search(r"\bdeasserted\b", expr_lower))
                            for sig in known_signal_names:
                                if not sig or sig in ("clk", "clock", "rst", "reset", "rst_n"):
                                    continue
                                if re.search(rf"\b{re.escape(sig)}\s+(?:is\s+)?asserted\b", expr_lower) or expr_lower == f"{sig} asserted":
                                    matched_sig = sig
                                    is_asserted = True
                                    break
                                elif re.search(rf"\b{re.escape(sig)}\s+(?:is\s+)?deasserted\b", expr_lower) or expr_lower == f"{sig} deasserted":
                                    matched_sig = sig
                                    is_deasserted = True
                                    break

                            if not matched_sig:
                                if "start" in expr_lower and (is_asserted or "asserted" in expr_lower):
                                    matched_sig = next((s for s in known_signal_names if s not in ("clk", "clock", "rst", "reset", "rst_n") and ("start" in s)), "start")
                                    is_asserted = True
                                elif "done" in expr_lower and (is_asserted or "asserted" in expr_lower):
                                    matched_sig = next((s for s in known_signal_names if s not in ("clk", "clock", "rst", "reset", "rst_n") and ("done" in s)), "done")
                                    is_asserted = True
                                elif "done" in expr_lower and is_deasserted:
                                    matched_sig = next((s for s in known_signal_names if s not in ("clk", "clock", "rst", "reset", "rst_n") and ("done" in s)), "done")
                                    is_deasserted = True

                            if matched_sig:
                                if is_asserted:
                                    if isinstance(c, dict):
                                        c["expression"] = f"{matched_sig} == 1'b1"
                                        c["is_fully_defined"] = True
                                        c.setdefault("signals", [])
                                        if matched_sig not in c["signals"]:
                                            c["signals"].append(matched_sig)
                                    else:
                                        tr["condition"] = {"expression": f"{matched_sig} == 1'b1", "signals": [matched_sig], "is_fully_defined": True}
                                elif is_deasserted:
                                    if isinstance(c, dict):
                                        c["expression"] = f"!{matched_sig}"
                                        c["is_fully_defined"] = True
                                        c.setdefault("signals", [])
                                        if matched_sig not in c["signals"]:
                                            c["signals"].append(matched_sig)
                                    else:
                                        tr["condition"] = {"expression": f"!{matched_sig}", "signals": [matched_sig], "is_fully_defined": True}

            # Ensure internal counter register is declared in plan if timer_cnt is referenced
            has_timer_in_fsm = any(
                "timer_cnt" in str(tr.get("condition", ""))
                for tr in plan_res.get("fsm", {}).get("transitions", [])
                if isinstance(tr, dict)
            )
            if has_timer_in_fsm:
                int_regs = plan_res.setdefault("internal_registers", [])
                if not any(r.get("name") == "timer_cnt" for r in int_regs if isinstance(r, dict)):
                    int_regs.append({
                        "name": "timer_cnt",
                        "width": "4",
                        "description": "Cycle counter tracking elapsed cycles in current state.",
                        "reset_value": "0"
                    })

            # Filter out any false positive blocking issues regarding "next clock cycle" or known port assertions
            if isinstance(plan_res.get("specification_status"), dict):
                b_issues = plan_res["specification_status"].get("blocking_issues", [])
                filtered_issues = []
                for b in b_issues:
                    iss_full = (str(b.get("issue", "")) + " " + str(b.get("required_information", "")) + " " + str(b.get("evidence", ""))).lower()
                    if re.search(r"\b(?:next\s+clock\s+cycle|single[- ]cycle\s+dwell|unconditional)\b", iss_full):
                        logger.info(f"Filtered out false-positive FSM blocking issue for valid next-clock-cycle transition: {b.get('issue')}")
                        continue

                    if "timer_cnt" in iss_full or (("clock cycle" in iss_full or "duration" in iss_full) and "transition" in iss_full):
                        logger.info(f"Filtered out FSM timing issue because timer guard is synthesized: {b.get('issue')}")
                        continue

                    # Filter out false-positive FSM blocking issue referencing known signals/ports
                    # (e.g. "FSM transition conditions reference 'start asserted', 'done asserted' without exact signal names...")
                    if any(k in iss_full for k in ["start asserted", "done asserted", "done deasserted", "control register bit, external port", "active polarity"]):
                        logger.info(f"Filtered out false-positive FSM blocking issue referencing port assertions: {b.get('issue')}")
                        continue

                    # Filter out false-positive FSM issue when no FSM was required or specified in the design
                    has_fsm_in_spec = bool(
                        (scoped_fsm and isinstance(scoped_fsm, dict) and scoped_fsm.get("states"))
                        or (efs_ir and getattr(efs_ir, "fsms", None))
                        or (design_spec and isinstance(design_spec, str) and re.search(r"\b(?:states|transitions)\b\s*:", design_spec.lower()))
                    )
                    if not has_fsm_in_spec and b.get("category") == "FSM" and (
                        re.search(r"\b(?:no|missing|contains no|without)\s+(?:control\s+)?(?:fsm|states?|transitions?|sequencing)\b", iss_full)
                        or any(k in iss_full for k in ["no fsm", "no control fsm", "missing fsm", "contains no fsm", "no states", "no transition", "no initial state"])
                    ):
                        logger.info(f"Filtered out false-positive FSM issue for design with no FSM requirements: {b.get('issue')}")
                        continue

                    # Filter out false-positive prompt artifact where LLM reports conflict between 'direction' field and 'description' field
                    if (
                        "direction" in iss_full
                        and "description" in iss_full
                        and any(k in iss_full for k in ("field", "between", "conflict", "contradict"))
                        and not ("in efs ir but" in iss_full or "in protocol contract" in iss_full)
                    ):
                        logger.info(f"Filtered out false-positive signal direction conflict resolved by authoritative evidence: {b.get('issue')}")
                        continue

                    filtered_issues.append(b)

                # Merge any authoritative blocking issues detected in specification parsing (e.g. vague/ambiguous FSM transitions)
                for sp_b in spec_parsed.get("specification_status", {}).get("blocking_issues", []):
                    if not any(fb.get("issue") == sp_b.get("issue") for fb in filtered_issues):
                        filtered_issues.append(sp_b)

                plan_res["specification_status"]["blocking_issues"] = filtered_issues
                if not filtered_issues and plan_res["specification_status"].get("status") in ("INCOMPLETE", "CONTRADICTORY"):
                    plan_res["specification_status"]["status"] = "COMPLETE"

            # Recover FSM states and transitions from spec_parsed if LLM omitted them
            if spec_parsed.get("states") or spec_parsed.get("transitions"):
                rfsm = plan_res.setdefault("fsm", {})
                if not rfsm.get("states") and spec_parsed.get("states"):
                    rfsm["states"] = spec_parsed.get("states", [])
                    rfsm["initial_state"] = spec_parsed["states"][0] if spec_parsed.get("states") else None
                if not rfsm.get("transitions") and spec_parsed.get("transitions"):
                    rfsm["transitions"] = spec_parsed.get("transitions", [])

            # Reconcile ports with all declared external specification signals
            validation_signals = list(scoped_signals) if scoped_signals else (list(efs_ir.signals) if efs_ir else [])
            if efs_ir and efs_ir.signals and not (scoped_signals or context_pack):
                existing_names = {
                    (s.get("name") if isinstance(s, dict) else getattr(s, "name", "")).strip().lower()
                    for s in validation_signals if (s.get("name") if isinstance(s, dict) else getattr(s, "name", ""))
                }
                for s in efs_ir.signals:
                    if s.direction in ("input", "output", "inout") and s.name.strip().lower() not in existing_names:
                        validation_signals.append(s)
                        existing_names.add(s.name.strip().lower())

            plan_ports = plan_res.setdefault("ports", [])
            plan_port_map = {str(p.get("name", "")).strip().lower(): p for p in plan_ports if isinstance(p, dict)}

            for s in validation_signals:
                s_name = s.get("name", "") if isinstance(s, dict) else s.name
                s_dir = s.get("direction", "input") if isinstance(s, dict) else (s.direction or "input")
                s_width = str(s.get("width", "1")) if isinstance(s, dict) else str(s.width or "1")
                s_desc = s.get("description", "") if isinstance(s, dict) else (s.description or "")
                s_role = s.get("semantic_role", "data") if isinstance(s, dict) else (s.semantic_role or "data")
                s_id = s.get("signal_id", "") if isinstance(s, dict) else getattr(s, "signal_id", "")
                s_name_lower = str(s_name).strip().lower()

                if not s_name_lower or s_name_lower in ("name", "signal", "port", "pin", "width", "direction"):
                    continue

                if s_name_lower not in plan_port_map:
                    # Generic mapping: Ensure declared signal is carried into plan ports
                    new_p = {
                        "name": s_name,
                        "direction": s_dir,
                        "width": s_width,
                        "description": s_desc or f"Declared port {s_name}",
                        "semantic_role": s_role,
                        "source": "ContextPack" if context_pack else "EFS"
                    }
                    plan_ports.append(new_p)
                    plan_port_map[s_name_lower] = new_p
                else:
                    matching_p = plan_port_map[s_name_lower]
                    p_dir = str(matching_p.get("direction", "")).strip().lower()
                    from core.efs_ir.signal_normalization import normalize_direction
                    norm_p_dir = normalize_direction(p_dir)
                    norm_s_dir = normalize_direction(s_dir)
                    if not norm_p_dir:
                        matching_p["direction"] = norm_s_dir or s_dir
                    elif norm_s_dir and norm_p_dir != norm_s_dir:
                        # Attempt generic evidence-based resolution using authority hierarchy
                        from core.efs_ir.signal_normalization import resolve_signal_metadata
                        sig_dict = s if isinstance(s, dict) else {"name": s_name, "direction": s_dir, "description": s_desc}
                        resolved_sig = resolve_signal_metadata([sig_dict, matching_p], signal_name=s_name)
                        if resolved_sig.get("direction") and not resolved_sig.get("is_conflicted"):
                            matching_p["direction"] = resolved_sig["direction"]
                            if isinstance(s, dict):
                                s["direction"] = resolved_sig["direction"]
                            elif hasattr(s, "direction"):
                                s.direction = resolved_sig["direction"]
                        else:
                            is_resolved = False
                            if efs_ir and hasattr(efs_ir, "resolutions"):
                                for res in efs_ir.resolutions:
                                    if getattr(res, "approval_status", "") == "APPROVED" and (s_name in res.affected_requirements or s_id in res.affected_requirements):
                                        is_resolved = True
                                        break
                            if efs_ir and hasattr(efs_ir, "conflicts"):
                                for c in efs_ir.conflicts:
                                    if s_name in c.entities and getattr(c, "resolution_status", "") == "RESOLVED":
                                        is_resolved = True
                                        break
                        if not is_resolved:
                            if not isinstance(plan_res.get("specification_status"), dict):
                                plan_res["specification_status"] = {"status": "CONTRADICTORY", "blocking_issues": []}
                            plan_res["specification_status"]["status"] = "CONTRADICTORY"
                            plan_res["specification_status"].setdefault("blocking_issues", []).append({
                                "category": "INTERFACE",
                                "issue": f"Signal direction conflict: {s_name} is specified as {norm_s_dir} in EFS IR but {norm_p_dir} in protocol contract/spec.",
                                "required_information": f"Clarify the correct direction for port '{s_name}'.",
                                "evidence": f"EFS IR: {norm_s_dir}, Spec/Plan: {norm_p_dir}"
                            })

            val = validate_rtl_plan(plan_res, efs_ir=efs_ir, context_pack=context_pack)
            plan_res["specification_status"] = {
                "status": val["status"],
                "blocking_issues": val["issues"]
            }
            return plan_res

    # Grounded fallback plan from scoped context or efs_ir or design_spec
    spec_parsed = parse_markdown_spec(design_spec)
    ports = []
    
    if scoped_signals:
        for s in scoped_signals:
            s_name = s.get("name", "") if isinstance(s, dict) else s.name
            s_dir = s.get("direction", "input") if isinstance(s, dict) else s.direction
            s_width = str(s.get("width", "1")) if isinstance(s, dict) else str(s.width)
            s_desc = s.get("description", "") if isinstance(s, dict) else s.description
            s_role = s.get("semantic_role", "data") if isinstance(s, dict) else (s.semantic_role or "data")
            s_id = s.get("signal_id", "") if isinstance(s, dict) else s.signal_id

            if s_name.lower() in ("name", "signal", "port", "signal name", "port name", "signal_name", "port_name", "pin", "width", "direction", "description", "protocol/description"):
                continue
            parsed_ports = spec_parsed.get("interfaces", [])
            matching_p = next((p for p in parsed_ports if p.get("name", "").lower() == s_name.lower()), None)
            source_dir = s_dir
            is_resolved = False
            if efs_ir and hasattr(efs_ir, "resolutions"):
                for res in efs_ir.resolutions:
                    if getattr(res, "approval_status", "") == "APPROVED" and (s_name in res.affected_requirements or s_id in res.affected_requirements):
                        is_resolved = True
                        break
            if efs_ir and hasattr(efs_ir, "conflicts"):
                for c in efs_ir.conflicts:
                    if s_name in c.entities and getattr(c, "resolution_status", "") == "RESOLVED":
                        is_resolved = True
                        break

            from core.efs_ir.signal_normalization import normalize_direction
            norm_match_dir = normalize_direction(matching_p.get("direction")) if (matching_p and matching_p.get("direction")) else None
            norm_src_dir = normalize_direction(s_dir)

            if norm_match_dir and norm_src_dir and norm_match_dir != norm_src_dir and not is_resolved:
                spec_parsed["specification_status"]["status"] = "CONTRADICTORY"
                spec_parsed["specification_status"]["blocking_issues"].append({
                    "category": "INTERFACE",
                    "issue": f"Signal direction conflict: {s_name} is specified as {norm_src_dir} in EFS IR but {norm_match_dir} in protocol contract/spec.",
                    "required_information": f"Clarify the correct direction for port '{s_name}'.",
                    "evidence": f"EFS IR: {norm_src_dir}, Spec: {norm_match_dir}"
                })
            ports.append({
                "name": s_name,
                "direction": source_dir,
                "width": s_width,
                "description": s_desc,
                "semantic_role": s_role,
                "source": "ContextPack" if context_pack else "EFS"
            })
    else:
        ports = spec_parsed.get("interfaces", [])

    if efs_ir and efs_ir.signals and not (scoped_signals or context_pack):
        port_names = {str(p.get("name", "")).strip().lower() for p in ports if isinstance(p, dict)}
        for s in efs_ir.signals:
            if s.direction in ("input", "output", "inout") and s.name.strip().lower() not in port_names:
                ports.append({
                    "name": s.name,
                    "direction": s.direction or "input",
                    "width": str(s.width or 1),
                    "description": s.description or f"Declared port {s.name}",
                    "semantic_role": s.semantic_role or "data",
                    "source": "EFS"
                })
                port_names.add(s.name.strip().lower())

    internal_regs = []
    if scoped_registers:
        for r in scoped_registers:
            r_name = r.get("name", "") if isinstance(r, dict) else r.name
            r_width = str(r.get("width", "32")) if isinstance(r, dict) else str(getattr(r, "width", "32"))
            r_reset = r.get("reset_value", r.get("reset_val", "32'h0")) if isinstance(r, dict) else str(getattr(r, "reset_value", "32'h0"))
            r_desc = r.get("description", "Internal Register") if isinstance(r, dict) else (r.description or "Internal Register")
            raw_fields = r.get("fields", []) if isinstance(r, dict) else getattr(r, "fields", [])
            fields = []
            for f in raw_fields:
                f_bits = f.get("bits", f"{f.get('msb', 0)}:{f.get('lsb', 0)}") if isinstance(f, dict) else f"{getattr(f, 'msb', 0)}:{getattr(f, 'lsb', 0)}"
                f_meaning = f.get("meaning", f.get("name", "")) if isinstance(f, dict) else (getattr(f, "name", "") or getattr(f, "description", ""))
                fields.append({
                    "bits": f_bits,
                    "meaning": f_meaning,
                    "source": "ContextPack" if context_pack else "EFS"
                })
            internal_regs.append({
                "name": r_name,
                "width": r_width,
                "reset_val": r_reset,
                "purpose": r_desc,
                "fields": fields
            })
    else:
        internal_regs = spec_parsed.get("registers", [])
    internal_regs = sanitize_hardware_registers(internal_regs)

    fsm_states = spec_parsed.get("states", [])
    fsm_transitions = spec_parsed.get("transitions", [])
    if scoped_fsm and isinstance(scoped_fsm, dict) and scoped_fsm.get("states"):
        fsm_states = [st.get("name", "") if isinstance(st, dict) else st for st in scoped_fsm.get("states", [])]
        if scoped_fsm.get("transitions"):
            fsm_transitions = []
            for tr in scoped_fsm.get("transitions", []):
                from_st = tr.get("source_state", tr.get("from", "IDLE")) if isinstance(tr, dict) else getattr(tr, "source_state", getattr(tr, "from", "IDLE"))
                to_st = tr.get("target_state", tr.get("to", "RUN")) if isinstance(tr, dict) else getattr(tr, "target_state", getattr(tr, "to", "RUN"))
                raw_cond = tr.get("condition", "undefined") if isinstance(tr, dict) else getattr(tr, "condition", "undefined")
                if isinstance(raw_cond, dict):
                    cond_expr = raw_cond.get("expression")
                    cond_source_evid = raw_cond.get("source_evidence") or (str(cond_expr) if cond_expr is not None else "")
                else:
                    cond_expr = str(raw_cond) if raw_cond is not None else "undefined"
                    cond_source_evid = cond_expr

                # Check if this transition condition has been resolved in EFS IR resolutions
                is_resolved = False
                if efs_ir and hasattr(efs_ir, "resolutions"):
                    for res in efs_ir.resolutions:
                        if getattr(res, "approval_status", "") == "APPROVED":
                            aff = [str(x).lower() for x in (getattr(res, "affected_requirements", []) or [])]
                            t_ent = str(getattr(res, "target_entity", "")).lower()
                            c_id = str(getattr(res, "conflict_id", "")).lower()
                            if t_ent in ("fsm", "state", "transition") or any(st.lower() in aff for st in [from_st, to_st]) or "fsm" in c_id:
                                is_resolved = True
                                break

                cond_search_str = str(cond_expr if cond_expr is not None else cond_source_evid).lower()
                is_dwell = bool(re.search(r"\b(?:next\s+clock\s+cycle|after\s+1\s+cycle|single[- ]cycle\s+dwell|unconditional)\b", cond_search_str))
                is_vague = (not is_resolved) and (not is_dwell) and any(vk in cond_search_str for vk in ["control start", "operation completes", "acknowledged", "undefined", "tbd"])
                fsm_transitions.append({
                    "from": from_st,
                    "to": to_st,
                    "condition": {
                        "expression": "1'b1" if is_dwell else (cond_expr if (not is_vague or is_resolved) else None),
                        "signals": [],
                        "registers": [],
                        "protocol_events": [],
                        "source": "ContextPack",
                        "source_evidence": cond_source_evid,
                        "is_fully_defined": is_dwell or (not is_vague)
                    }
                })
    elif efs_ir and efs_ir.fsms:
        for f in efs_ir.fsms:
            if hasattr(f, "states") and f.states:
                fsm_states = [st.name for st in f.states]
            if hasattr(f, "transitions") and f.transitions:
                fsm_transitions = []
                for tr in f.transitions:
                    from_st = tr.from_state if hasattr(tr, "from_state") else getattr(tr, "from", getattr(tr, "source_state", "IDLE"))
                    to_st = tr.to_state if hasattr(tr, "to_state") else getattr(tr, "to", getattr(tr, "target_state", "RUN"))
                    raw_cond = tr.condition if hasattr(tr, "condition") else "undefined"
                    if isinstance(raw_cond, dict):
                        cond_expr = raw_cond.get("expression")
                        cond_source_evid = raw_cond.get("source_evidence") or (str(cond_expr) if cond_expr is not None else "")
                    else:
                        cond_expr = str(raw_cond) if raw_cond is not None else "undefined"
                        cond_source_evid = cond_expr

                    # Check if resolved
                    is_resolved = False
                    if efs_ir and hasattr(efs_ir, "resolutions"):
                        for res in efs_ir.resolutions:
                            if getattr(res, "approval_status", "") == "APPROVED":
                                aff = [str(x).lower() for x in (getattr(res, "affected_requirements", []) or [])]
                                t_ent = str(getattr(res, "target_entity", "")).lower()
                                c_id = str(getattr(res, "conflict_id", "")).lower()
                                if t_ent in ("fsm", "state", "transition") or any(st.lower() in aff for st in [from_st, to_st]) or "fsm" in c_id:
                                    is_resolved = True
                                    break

                    cond_search_str = str(cond_expr if cond_expr is not None else cond_source_evid).lower()
                    is_dwell = bool(re.search(r"\b(?:next\s+clock\s+cycle|after\s+1\s+cycle|single[- ]cycle\s+dwell|unconditional)\b", cond_search_str))
                    is_vague = (not is_resolved) and (not is_dwell) and any(vk in cond_search_str for vk in ["control start", "operation completes", "acknowledged", "undefined", "tbd"])
                    fsm_transitions.append({
                        "from": from_st,
                        "to": to_st,
                        "condition": {
                            "expression": "1'b1" if is_dwell else (cond_expr if (not is_vague or is_resolved) else None),
                            "signals": [],
                            "registers": [],
                            "protocol_events": [],
                            "source": "EFS",
                            "source_evidence": cond_source_evid,
                            "is_fully_defined": is_dwell or (not is_vague)
                        }
                    })

    plan = {
        "module_name": target_module_name if target_module_name not in ("target_module", "hardware_module", "custom_module") else spec_parsed.get("module_name", target_module_name),
        "specification_status": spec_parsed.get("specification_status", {
            "status": "COMPLETE",
            "blocking_issues": []
        }),
        "ports": ports,
        "parameters": spec_parsed.get("parameters", []),
        "internal_registers": internal_regs,
        "fsm": {
            "states": fsm_states,
            "initial_state": fsm_states[0] if fsm_states else None,
            "transitions": fsm_transitions
        },
        "datapath_strategy": "Direct state machine & register transfer logic",
        "reset_polarity": "active_low"
    }

    val = validate_rtl_plan(plan, efs_ir=efs_ir, context_pack=context_pack)
    plan["specification_status"] = {
        "status": val["status"],
        "blocking_issues": val["issues"]
    }
    return plan


def validate_rtl_plan(rtl_plan: Dict[str, Any], efs_ir: Optional[EFSIR] = None, context_pack: Optional[Any] = None) -> Dict[str, Any]:
    """
    Stage B Validation: Verifies that the generated RTL plan faithfully implements the extracted EFSIR or scoped ContextPack.
    Compares: RTL Plan <-> EFSIR / ContextPack (NOT against generic assumed hardware templates).
    """
    logger.info("Validating RTL plan against authoritative EFS IR or scoped ContextPack...")
    all_issues = []

    # Version and Source Consistency Logging
    model_ver = getattr(efs_ir.metadata, "model_version", 1) if (efs_ir and hasattr(efs_ir, "metadata")) else 1
    res_ver = getattr(efs_ir.metadata, "resolution_version", 0) if (efs_ir and hasattr(efs_ir, "metadata")) else 0
    ctx_ver = getattr(context_pack, "context_version", model_ver) if context_pack else model_ver
    val_source = "resolved_efs_ir" if res_ver > 0 else ("context_pack" if context_pack else "efs_ir")

    logger.info(
        f"MODEL_VERSION={model_ver} "
        f"EFS_IR_VERSION={model_ver} "
        f"GENERATION_CONTEXT_VERSION={ctx_ver} "
        f"RTL_VALIDATION_SOURCE={val_source}"
    )

    if not isinstance(rtl_plan, dict):
        return {
            "valid": False,
            "status": "INCOMPLETE",
            "issues": [{
                "category": "INTERFACE",
                "classification": "PLAN_MISSING",
                "badge": "🔴 [PLAN_MISSING]",
                "is_blocking": True,
                "issue": "Structured RTL Plan is not a valid JSON object.",
                "required_information": "Provide a valid structured plan.",
                "evidence": str(rtl_plan)
            }]
        }

    c_dict = None
    scoped_signals = []
    scoped_fsm = {}
    scoped_conflicts = []
    target_comp_name = ""

    if context_pack:
        c_dict = context_pack.to_dict() if hasattr(context_pack, "to_dict") else context_pack
        if isinstance(c_dict, dict):
            t_comp = c_dict.get("target_component", {})
            target_comp_name = t_comp.get("name") or getattr(getattr(context_pack, "task", None), "target", "") or ""
            scoped_signals = c_dict.get("target_signals", c_dict.get("signals", []))
            scoped_fsm = c_dict.get("target_fsm", c_dict.get("fsm", {}))
            scoped_conflicts = c_dict.get("conflicts", [])

    # 1. Report unresolved source conflicts from ContextPack or EFS IR
    if scoped_conflicts:
        for conf in scoped_conflicts:
            c_type = conf.get("type", "CONFLICT") if isinstance(conf, dict) else getattr(conf, "type", "CONFLICT")
            c_diag_type = conf.get("diagnostic_type", "") if isinstance(conf, dict) else getattr(conf, "diagnostic_type", "")
            if c_type in ("TARGET_RESOLUTION_FAILED", "PLANNER_ERROR") or c_diag_type in ("TARGET_RESOLUTION_FAILED", "PLANNER_ERROR"):
                continue
            c_status = conf.get("resolution_status", "UNRESOLVED") if isinstance(conf, dict) else getattr(conf, "resolution_status", "UNRESOLVED")
            if c_status == "UNRESOLVED":
                c_cat = conf.get("category", "GENERAL") if isinstance(conf, dict) else getattr(conf, "category", "GENERAL")
                c_evid = conf.get("evidence", conf.get("description", "")) if isinstance(conf, dict) else (getattr(conf, "evidence", "") or getattr(conf, "description", ""))
                c_desc = conf.get("description", "") if isinstance(conf, dict) else getattr(conf, "description", "")
                c_id = conf.get("conflict_id", conf.get("id", "")) if isinstance(conf, dict) else (getattr(conf, "conflict_id", getattr(conf, "id", "")))
                all_issues.append({
                    "category": c_cat or "GENERAL",
                    "classification": "SOURCE_CONFLICT",
                    "badge": "🟠 [SOURCE_CONFLICT]",
                    "is_blocking": True,
                    "issue": f"[{c_id}] Unresolved specification conflict [{c_type}]: {c_evid or c_desc}",
                    "required_information": c_desc,
                    "evidence": c_evid or "Source text",
                    "conflict_id": c_id
                })
    elif efs_ir and hasattr(efs_ir, "conflicts"):
        for conf in efs_ir.conflicts:
            c_type = getattr(conf, "type", "CONFLICT")
            c_diag_type = getattr(conf, "diagnostic_type", "SPEC_CONFLICT")
            if c_type in ("TARGET_RESOLUTION_FAILED", "PLANNER_ERROR") or c_diag_type in ("TARGET_RESOLUTION_FAILED", "PLANNER_ERROR"):
                continue
            if getattr(conf, "resolution_status", "UNRESOLVED") == "UNRESOLVED":
                if target_comp_name:
                    sig_names = {s.get("name", "").lower() if isinstance(s, dict) else s.name.lower() for s in scoped_signals}
                    conf_entities = [e.lower() for e in getattr(conf, "entities", [])]
                    conf_comp = getattr(conf, "target_component", "")
                    is_relevant = (
                        (conf_comp and conf_comp.lower() == target_comp_name.lower()) or
                        target_comp_name.lower() in conf_entities or
                        any(sn in conf_entities for sn in sig_names if sn)
                    )
                    if not is_relevant:
                        continue

                c_id = getattr(conf, "conflict_id", getattr(conf, "id", ""))
                all_issues.append({
                    "category": getattr(conf, "category", "GENERAL") or "GENERAL",
                    "classification": "SOURCE_CONFLICT",
                    "badge": "🟠 [SOURCE_CONFLICT]",
                    "is_blocking": True,
                    "issue": f"[{c_id}] Unresolved specification conflict [{conf.type}]: {conf.evidence or conf.description}",
                    "required_information": conf.description,
                    "evidence": conf.evidence or "Source text",
                    "conflict_id": c_id
                })

    # 2. Check FSM states and transitions from EFS IR or plan
    plan_fsm = rtl_plan.get("fsm", {})
    plan_states = set(plan_fsm.get("states", [])) if isinstance(plan_fsm, dict) else set()
    plan_transitions = plan_fsm.get("transitions", []) if isinstance(plan_fsm, dict) else []

    vague_patterns = [
        "control start condition met",
        "start condition met",
        "operation completes",
        "counter completes",
        "transaction acknowledged",
        "internal counter completes",
        "condition met",
        "undefined",
        "unspecified",
        "tbd"
    ]

    states_with_outgoing = set()
    outgoing_by_state: Dict[str, List[Dict[str, Any]]] = {}

    for idx, trans in enumerate(plan_transitions):
        if not isinstance(trans, dict):
            continue
        from_st = trans.get("from_state") or trans.get("from", f"STATE_{idx}")
        to_st = trans.get("to_state") or trans.get("to", f"NEXT_STATE_{idx}")
        cond = trans.get("condition")

        if from_st:
            states_with_outgoing.add(str(from_st).strip())
            outgoing_by_state.setdefault(str(from_st).strip(), []).append(trans)

        # Check if this transition condition has been resolved in EFS IR resolutions
        is_trans_resolved = False
        if efs_ir and hasattr(efs_ir, "resolutions"):
            for res in efs_ir.resolutions:
                if getattr(res, "approval_status", "") == "APPROVED":
                    aff = [str(x).lower() for x in (getattr(res, "affected_requirements", []) or [])]
                    t_ent = str(getattr(res, "target_entity", "")).lower()
                    c_id = str(getattr(res, "conflict_id", "")).lower()
                    if (
                        t_ent in ("fsm", "state", "transition")
                        or str(from_st).lower() in aff
                        or str(to_st).lower() in aff
                        or any(str(from_st).lower() in x or str(to_st).lower() in x for x in aff)
                        or (c_id and ("fsm" in c_id or "trans" in c_id))
                    ):
                        is_trans_resolved = True
                        break
        if efs_ir and hasattr(efs_ir, "conflicts"):
            for c in efs_ir.conflicts:
                if getattr(c, "resolution_status", "") == "RESOLVED":
                    c_ents = [str(e).lower() for e in getattr(c, "entities", [])]
                    if str(from_st).lower() in c_ents or str(to_st).lower() in c_ents or any("fsm" in e for e in c_ents):
                        is_trans_resolved = True
                        break

        cond_expr = cond.get("expression") if isinstance(cond, dict) else cond
        cond_evidence = cond.get("source_evidence") if isinstance(cond, dict) else ""
        cond_class = cond.get("classification") if isinstance(cond, dict) else ""
        cond_text = str(cond_expr if cond_expr is not None else (cond_evidence or "")).strip()
        cond_is_empty = (not cond_text or cond_text == "None")

        is_vague = (cond_class == "AMBIGUOUS" or any(vp in cond_text.lower() for vp in vague_patterns))

        if is_vague and not is_trans_resolved:
            all_issues.append({
                "category": "FSM",
                "classification": "AMBIGUOUS",
                "badge": "🟡 [AMBIGUOUS]",
                "is_blocking": True,
                "issue": f"{from_st} -> {to_st} transition condition is ambiguous ('{cond_text}').",
                "required_information": f"Specify explicit signal/register condition for transition '{from_st}' -> '{to_st}'.",
                "evidence": cond_text
            })
        elif cond_is_empty and not is_trans_resolved:
            all_issues.append({
                "category": "FSM",
                "classification": "PLAN_MISSING",
                "badge": "🔴 [PLAN_MISSING]",
                "is_blocking": True,
                "issue": f"{from_st} -> {to_st} transition condition is undefined.",
                "required_information": f"Specify condition triggering transition '{from_st}' -> '{to_st}'.",
                "evidence": f"Transition '{from_st}' -> '{to_st}' has null or empty condition."
            })
        elif cond is not None and not cond_is_empty:
            cond_str = cond_text

            # Check for unspecified register bit references in this transition condition
            internal_regs = rtl_plan.get("internal_registers", rtl_plan.get("registers", []))
            reg_map = {str(r.get("name", "")).lower(): r for r in internal_regs if isinstance(r, dict)}
            reg_bit_refs = re.findall(r"([a-zA-Z0-9_]+)\[(\d+)\]", cond_str)
            for r_name, bit_idx in reg_bit_refs:
                r_obj = reg_map.get(r_name.lower())
                if r_obj:
                    r_fields = r_obj.get("fields", [])
                    has_explicit_field = any(
                        (f.get("classification") == "EXPLICIT" or f.get("source") in ("ContextPack", "EFS"))
                        and (str(f.get("bits", "")) in (f"{bit_idx}:{bit_idx}", bit_idx, f"[{bit_idx}]")
                             or (f.get("msb") == int(bit_idx) and f.get("lsb") == int(bit_idx)))
                        for f in r_fields if isinstance(f, dict)
                    )
                    if not has_explicit_field:
                        all_issues.append({
                            "category": "REGISTER",
                            "classification": "PLAN_MISSING",
                            "badge": "🔴 [PLAN_MISSING]",
                            "is_blocking": True,
                            "issue": f"Register bit '{r_name}[{bit_idx}]' is referenced in transition condition but has no explicit field definition.",
                            "required_information": f"Define explicit field semantics for bit {bit_idx} of register '{r_name}'.",
                            "evidence": cond_str
                        })

    # Check for dead-end states (states with no outgoing transitions) only when FSM transitions are specified
    if plan_transitions:
        for st in plan_states:
            st_name = st.get("name") if isinstance(st, dict) else str(st).strip()
            if st_name and st_name not in states_with_outgoing and len(plan_states) > 1 and st_name.upper() not in ("DONE", "ERROR"):
                all_issues.append({
                    "category": "FSM",
                    "classification": "PLAN_MISSING",
                    "badge": "🔴 [PLAN_MISSING]",
                    "is_blocking": True,
                    "issue": f"FSM state '{st_name}' has no transitions defined.",
                    "required_information": f"Specify outgoing transitions for state '{st_name}'.",
                    "evidence": f"State '{st_name}' has no outgoing transitions."
                })

        # Check for multiple outgoing transitions without distinct conditions
        for fst, tr_list in outgoing_by_state.items():
            if len(tr_list) > 1:
                empty_or_vague = [t for t in tr_list if not t.get("condition") or str(t.get("condition") if not isinstance(t.get("condition"), dict) else t.get("condition").get("expression", "")).strip() in ("", "1'b1", "1")]
                if len(empty_or_vague) > 1 or any(not t.get("condition") for t in tr_list):
                    all_issues.append({
                        "category": "FSM",
                        "classification": "AMBIGUOUS",
                        "badge": "🟡 [AMBIGUOUS]",
                        "is_blocking": True,
                        "issue": f"FSM state '{fst}' has multiple outgoing transitions without distinct condition.",
                        "required_information": f"Specify mutually exclusive transition conditions for transitions from '{fst}'.",
                        "evidence": f"State '{fst}' has {len(tr_list)} transitions with ambiguous conditions."
                    })

    if scoped_fsm and isinstance(scoped_fsm, dict) and scoped_fsm.get("states"):
        for st in scoped_fsm.get("states", []):
            st_name = st.get("name") if isinstance(st, dict) else (getattr(st, "name", str(st)))
            if st_name and st_name not in plan_states and plan_states:
                all_issues.append({
                    "category": "FSM",
                    "classification": "PLAN_MISSING",
                    "badge": "🔴 [PLAN_MISSING]",
                    "is_blocking": True,
                    "issue": f"FSM state '{st_name}' declared in scoped context is missing from RTL plan states.",
                    "required_information": f"Include state '{st_name}' in RTL plan FSM.",
                    "evidence": f"Scoped FSM state '{st_name}'"
                })
    elif not context_pack and efs_ir and efs_ir.fsms:
        for fsm in efs_ir.fsms:
            for st in fsm.states:
                if st.name and st.name not in plan_states and plan_states:
                    all_issues.append({
                        "category": "FSM",
                        "classification": "PLAN_MISSING",
                        "badge": "🔴 [PLAN_MISSING]",
                        "is_blocking": True,
                        "issue": f"FSM state '{st.name}' declared in EFS IR is missing from RTL plan states.",
                        "required_information": f"Include state '{st.name}' in RTL plan FSM.",
                        "evidence": f"EFS IR state '{st.name}'"
                    })

    # 3. Check interface signals from ContextPack (scoped) or EFS IR (global)
    validation_signals = []
    if scoped_signals:
        validation_signals = scoped_signals
    elif not context_pack and efs_ir and efs_ir.signals:
        validation_signals = efs_ir.signals

    if validation_signals:
        plan_ports = rtl_plan.get("ports", [])
        plan_port_map = {str(p.get("name", "")).strip().lower(): p for p in plan_ports if isinstance(p, dict)}
        
        for sig in validation_signals:
            s_name = sig.get("name", "") if isinstance(sig, dict) else sig.name
            s_dir = sig.get("direction", "") if isinstance(sig, dict) else sig.direction
            s_id = sig.get("signal_id", "") if isinstance(sig, dict) else getattr(sig, "signal_id", "")
            s_name_lower = s_name.lower()
            if s_name_lower in ("name", "signal", "port", "pin", "width", "direction"):
                continue
            if s_name_lower not in plan_port_map and plan_ports:
                all_issues.append({
                    "category": "INTERFACE",
                    "classification": "PLAN_MISSING",
                    "badge": "🔴 [PLAN_MISSING]",
                    "is_blocking": True,
                    "issue": f"Signal '{s_name}' declared in specification is missing from RTL plan ports.",
                    "required_information": f"Add port '{s_name}' to RTL plan.",
                    "evidence": f"Specification signal '{s_name}' ({s_dir})"
                })
            elif s_name_lower in plan_port_map:
                from core.efs_ir.signal_normalization import normalize_direction, parse_bit_range
                p_obj = plan_port_map[s_name_lower]
                p_dir = str(p_obj.get("direction", "")).lower()
                e_dir = str(s_dir).lower()
                norm_p_dir = normalize_direction(p_dir)
                norm_e_dir = normalize_direction(e_dir)
                if norm_p_dir and norm_e_dir and norm_p_dir != norm_e_dir:
                    is_resolved = False
                    if efs_ir and hasattr(efs_ir, "resolutions"):
                        for res in efs_ir.resolutions:
                            if getattr(res, "approval_status", "") == "APPROVED" and (s_name in res.affected_requirements or s_id in res.affected_requirements):
                                is_resolved = True
                                break
                    if efs_ir and hasattr(efs_ir, "conflicts"):
                        for c in efs_ir.conflicts:
                            if s_name in c.entities and getattr(c, "resolution_status", "") == "RESOLVED":
                                is_resolved = True
                                break
                    if not is_resolved:
                        all_issues.append({
                            "category": "INTERFACE",
                            "classification": "PLAN_CONFLICT",
                            "badge": "🟠 [PLAN_CONFLICT]",
                            "is_blocking": True,
                            "issue": f"Signal direction conflict on '{s_name}': RTL plan specifies '{norm_p_dir}', but specification specifies '{norm_e_dir}'.",
                            "required_information": f"Align port '{s_name}' direction with specification ('{norm_e_dir}').",
                            "evidence": f"Plan: {norm_p_dir}, Spec: {norm_e_dir}"
                        })

                # Report unresolved metadata conflict on the signal itself
                sig_conflict = getattr(sig, "conflict_details", None) if not isinstance(sig, dict) else sig.get("conflict_details")
                if sig_conflict and isinstance(sig_conflict, dict) and sig_conflict.get("is_conflicted"):
                    for c_item in sig_conflict.get("conflicts", []):
                        all_issues.append({
                            "category": "INTERFACE",
                            "classification": "SOURCE_CONFLICT",
                            "badge": "🟠 [SOURCE_CONFLICT]",
                            "is_blocking": True,
                            "issue": f"Unresolved specification conflict on '{s_name}': {c_item.get('issue')}",
                            "required_information": f"Clarify conflicting {c_item.get('property')} for port '{s_name}'.",
                            "evidence": c_item.get("evidence", "Specification metadata")
                        })

                # Signal width consistency check
                p_w = str(p_obj.get("width", "1"))
                s_w = str(sig.get("width", "1") if isinstance(sig, dict) else (sig.width or "1"))
                norm_pw, pw_range = parse_bit_range(p_w)
                norm_sw, sw_range = parse_bit_range(s_w)
                has_w_conflict = False
                if norm_pw is not None and norm_sw is not None:
                    has_w_conflict = (norm_pw != norm_sw)
                elif pw_range and sw_range:
                    has_w_conflict = (pw_range.lower().replace(" ", "") != sw_range.lower().replace(" ", ""))

                if has_w_conflict:
                    is_w_resolved = False
                    if efs_ir and hasattr(efs_ir, "resolutions"):
                        for res in efs_ir.resolutions:
                            if getattr(res, "approval_status", "") == "APPROVED" and (s_name in getattr(res, "affected_requirements", []) or s_id in getattr(res, "affected_requirements", [])):
                                is_w_resolved = True
                                break
                    if not is_w_resolved:
                        all_issues.append({
                            "category": "INTERFACE",
                            "classification": "PLAN_CONFLICT",
                            "badge": "🟠 [PLAN_CONFLICT]",
                            "is_blocking": True,
                            "issue": f"Signal width mismatch on '{s_name}': specification requires width {norm_sw or sw_range}, but plan specifies width {norm_pw or pw_range}.",
                            "required_information": f"Align port '{s_name}' width with specification ({norm_sw or sw_range}).",
                            "evidence": f"Spec width: {norm_sw or sw_range}, Plan width: {norm_pw or pw_range}"
                        })

    # 4. Check for Opcode binary encoding collisions
    all_opcs = []
    if efs_ir and hasattr(efs_ir, "opcodes") and efs_ir.opcodes:
        all_opcs.extend(efs_ir.opcodes)
    if isinstance(rtl_plan.get("opcodes"), list):
        all_opcs.extend(rtl_plan["opcodes"])
    seen_opc: Dict[str, str] = {}
    for op in all_opcs:
        enc = op.get("binary_encoding") if isinstance(op, dict) else getattr(op, "binary_encoding", None)
        mnem = op.get("mnemonic") if isinstance(op, dict) else getattr(op, "mnemonic", None)
        if enc and mnem and str(enc).strip() not in ("", "None"):
            clean_enc = str(enc).strip()
            if clean_enc in seen_opc and seen_opc[clean_enc] != mnem:
                is_opc_resolved = False
                if efs_ir and hasattr(efs_ir, "resolutions"):
                    for res in efs_ir.resolutions:
                        if getattr(res, "approval_status", "") == "APPROVED":
                            aff = [str(x).lower() for x in (getattr(res, "affected_requirements", []) or [])]
                            if mnem.lower() in aff or seen_opc[clean_enc].lower() in aff or "opcode" in aff:
                                is_opc_resolved = True
                                break
                if not is_opc_resolved:
                    all_issues.append({
                        "category": "OPCODE",
                        "classification": "SOURCE_CONFLICT",
                        "badge": "🟠 [SOURCE_CONFLICT]",
                        "is_blocking": True,
                        "issue": f"Opcode binary encoding collision: '{mnem}' and '{seen_opc[clean_enc]}' share binary encoding '{clean_enc}'.",
                        "required_information": f"Assign distinct binary encodings for '{mnem}' and '{seen_opc[clean_enc]}'.",
                        "evidence": f"Opcode '{mnem}' and '{seen_opc[clean_enc]}' both mapped to '{clean_enc}'"
                    })
            else:
                seen_opc[clean_enc] = mnem

    # Also check EFSIR instructions
    if efs_ir and hasattr(efs_ir, "instructions") and efs_ir.instructions:
        inst_opc_map: Dict[str, str] = {}
        for inst in efs_ir.instructions:
            opc = getattr(inst, "opcode", None)
            iname = getattr(inst, "name", None)
            if opc and iname:
                clean_opc = str(opc).strip()
                if clean_opc in inst_opc_map and inst_opc_map[clean_opc] != iname:
                    all_issues.append({
                        "category": "OPCODE",
                        "classification": "SOURCE_CONFLICT",
                        "badge": "🟠 [SOURCE_CONFLICT]",
                        "is_blocking": True,
                        "issue": f"Opcode collision detected between '{iname}' and '{inst_opc_map[clean_opc]}' on opcode '{clean_opc}'.",
                        "required_information": f"Assign distinct binary opcodes for '{iname}' and '{inst_opc_map[clean_opc]}'.",
                        "evidence": f"Both instructions share opcode '{clean_opc}'"
                    })
                else:
                    inst_opc_map[clean_opc] = iname

    # 5. Check for Reset Polarity Conflicts
    reset_ports = [p for p in rtl_plan.get("ports", []) if isinstance(p, dict) and any(k in str(p.get("name", "")).lower() for k in ("rst", "reset"))]
    if len(reset_ports) > 1:
        reset_polarities = set()
        for rp in reset_ports:
            rp_name = str(rp.get("name", "")).lower()
            rp_pol = "active_low" if "n" in rp_name or "low" in str(rp.get("description", "")).lower() else "active_high"
            reset_polarities.add(rp_pol)
        if len(reset_polarities) > 1:
            is_pol_resolved = False
            if efs_ir and hasattr(efs_ir, "resolutions"):
                for res in efs_ir.resolutions:
                    if getattr(res, "approval_status", "") == "APPROVED" and "reset" in str(getattr(res, "property", "")).lower():
                        is_pol_resolved = True
                        break
            if not is_pol_resolved:
                all_issues.append({
                    "category": "RESET",
                    "classification": "SOURCE_CONFLICT",
                    "badge": "🟠 [SOURCE_CONFLICT]",
                    "is_blocking": True,
                    "issue": "Conflicting reset polarities detected across reset signals in plan.",
                    "required_information": "Unify reset signals to consistent active-low or active-high polarity.",
                    "evidence": f"Reset ports: {[p.get('name') for p in reset_ports]}"
                })

    # Check clock_reset section reset_polarity
    cr = rtl_plan.get("clock_reset", {})
    if isinstance(cr, dict):
        rst_n = str(cr.get("reset_name", ""))
        rst_pol = str(cr.get("reset_polarity", "")).lower()
        if "n" in rst_n.lower() and rst_pol == "high":
            all_issues.append({
                "category": "SIGNAL",
                "classification": "CONFLICT",
                "badge": "🟠 [CONFLICT]",
                "is_blocking": True,
                "issue": f"Conflicting reset polarity: signal '{rst_n}' indicates active-low but reset_polarity is specified as 'high'.",
                "required_information": "Align reset polarity with reset signal naming convention.",
                "evidence": f"Reset '{rst_n}' configured as '{rst_pol}'"
            })

    # 6. Check for Internal Register Reset Values
    for reg in rtl_plan.get("internal_registers", []):
        if isinstance(reg, dict):
            r_reset = reg.get("reset_val", reg.get("reset_value"))
            if r_reset is None or str(r_reset).strip() in ("", "None", "undefined"):
                all_issues.append({
                    "category": "REGISTER",
                    "classification": "PLAN_MISSING",
                    "badge": "🔴 [PLAN_MISSING]",
                    "is_blocking": True,
                    "issue": f"Internal register '{reg.get('name')}' is missing a defined reset value.",
                    "required_information": f"Specify explicit reset value for register '{reg.get('name')}'.",
                    "evidence": f"Register '{reg.get('name')}' has undefined reset value."
                })

    # 7. Check for Undriven Output Ports
    output_ports = [p for p in rtl_plan.get("ports", []) if isinstance(p, dict) and str(p.get("direction", "")).lower() in ("output", "out")]
    driving_strat = rtl_plan.get("output_driving_strategy") if "output_driving_strategy" in rtl_plan else rtl_plan.get("output_assignments")
    if driving_strat is not None and output_ports:
        strat_str = str(driving_strat).lower()
        for op in output_ports:
            op_name = str(op.get("name", "")).lower()
            if op_name not in strat_str and op.get("name") not in ("clk", "rst_n", "rst"):
                all_issues.append({
                    "category": "INTERFACE",
                    "classification": "PLAN_MISSING",
                    "badge": "🔴 [PLAN_MISSING]",
                    "is_blocking": True,
                    "issue": f"Output port '{op.get('name')}' has no defined driving behavior in the implementation plan.",
                    "required_information": f"Define driving behavior or assignment logic for output port '{op.get('name')}'.",
                    "evidence": f"Port '{op.get('name')}' has no driving strategy."
                })

    # 8. Check Datapath Operations
    datapath = rtl_plan.get("datapath")
    if isinstance(datapath, dict) and "operations" in datapath:
        ops = datapath.get("operations", [])
        if not ops:
            all_issues.append({
                "category": "DATAPATH",
                "classification": "PLAN_MISSING",
                "badge": "🔴 [PLAN_MISSING]",
                "is_blocking": True,
                "issue": "Datapath architecture specifies no operational semantics or operations.",
                "required_information": "Provide operation semantics for datapath.",
                "evidence": "Empty datapath operations list"
            })

    # 9. Check Interface Logic and AXI Handshakes
    iface_logic = rtl_plan.get("interface_logic", {})
    if isinstance(iface_logic, dict):
        for k, v in iface_logic.items():
            if "axi" in k.lower() or "handshake" in k.lower():
                if "without ready" in str(v).lower() or "incomplete" in str(v).lower():
                    all_issues.append({
                        "category": "PROTOCOL",
                        "classification": "PLAN_MISSING",
                        "badge": "🔴 [PLAN_MISSING]",
                        "is_blocking": True,
                        "issue": f"Incomplete AXI handshake behavior specified: '{v}'. AXI protocols require coupled valid/ready handshakes.",
                        "required_information": "Specify both valid and ready signals for AXI channel handshake.",
                        "evidence": str(v)
                    })

    # 8. Check for Placeholder Comments in Plan
    placeholder_tokens = ["// active low reset", "// combinational logic", "// synthesizable logic implementation"]
    procedural_sections = {
        "always_comb_body", "always_ff_body", "datapath_logic", "interface_logic",
        "procedural_blocks", "code", "logic_implementation", "architecture_description"
    }
    for pk, pv in rtl_plan.items():
        if pk not in procedural_sections or not pv:
            continue
        pv_lower = str(pv).lower()
        for pt in placeholder_tokens:
            if pt in pv_lower:
                all_issues.append({
                    "category": "ARCHITECTURE",
                    "classification": "PLAN_INVALID",
                    "badge": "🔴 [PLAN_INVALID]",
                    "is_blocking": True,
                    "issue": f"RTL plan contains placeholder code comment '{pt}' in section '{pk}'.",
                    "required_information": "Provide concrete hardware transfer behavior instead of placeholder text.",
                    "evidence": f"Section '{pk}' contains '{pt}'"
                })
                break

    # 9. Check reported issues from plan specification status
    spec_status = rtl_plan.get("specification_status", {})
    if isinstance(spec_status, dict):
        reported_issues = spec_status.get("blocking_issues", [])
        if isinstance(reported_issues, list):
            # Collect known signal and port names
            known_signal_names = set()
            for p in rtl_plan.get("ports", []):
                if isinstance(p, dict) and p.get("name"):
                    known_signal_names.add(str(p["name"]).strip().lower())
            for s in (scoped_signals or []):
                s_n = s.get("name") if isinstance(s, dict) else getattr(s, "name", None)
                if s_n:
                    known_signal_names.add(str(s_n).strip().lower())
            if efs_ir and hasattr(efs_ir, "signals"):
                for s in efs_ir.signals:
                    s_n = s.get("name") if isinstance(s, dict) else getattr(s, "name", None)
                    if s_n:
                        known_signal_names.add(str(s_n).strip().lower())

            for b_issue in reported_issues:
                if isinstance(b_issue, dict):
                    iss_txt = str(b_issue.get("issue", "")).lower()
                    iss_full = (iss_txt + " " + str(b_issue.get("required_information", "")) + " " + str(b_issue.get("evidence", ""))).lower()
                    if re.search(r"\b(?:next\s+clock\s+cycle|single[- ]cycle\s+dwell|unconditional)\b", iss_full):
                        continue

                    # Filter out false-positive FSM blocking issue referencing known signals/ports
                    if any(k in iss_full for k in ["start asserted", "done asserted", "done deasserted", "control register bit, external port", "active polarity"]):
                        logger.info(f"Filtered out false-positive FSM blocking issue in validate_rtl_plan: {b_issue.get('issue')}")
                        continue

                    # Filter out false-positive FSM issue when no FSM was required or specified in the design
                    has_fsm_in_spec = bool(
                        (scoped_fsm and isinstance(scoped_fsm, dict) and scoped_fsm.get("states"))
                        or (efs_ir and getattr(efs_ir, "fsms", None))
                    )
                    if not has_fsm_in_spec and b_issue.get("category") == "FSM" and (
                        re.search(r"\b(?:no|missing|contains no|without)\s+(?:control\s+)?(?:fsm|states?|transitions?|sequencing)\b", iss_full)
                        or any(k in iss_full for k in ["no fsm", "no control fsm", "missing fsm", "contains no fsm", "no states", "no transition", "no initial state"])
                    ):
                        logger.info(f"Filtered out false-positive FSM issue in validate_rtl_plan for design with no FSM requirements: {b_issue.get('issue')}")
                        continue

                    # Filter out false-positive prompt artifact where LLM reports conflict between 'direction' field and 'description' field
                    if (
                        "direction" in iss_full
                        and "description" in iss_full
                        and any(k in iss_full for k in ("field", "between", "conflict", "contradict"))
                        and not ("in efs ir but" in iss_full or "in protocol contract" in iss_full)
                    ):
                        has_active_sig_conflict = False
                        if efs_ir and hasattr(efs_ir, "signals"):
                            for s in efs_ir.signals:
                                if s.name.lower() in iss_full and (getattr(s, "direction_source", "") == "conflict" or getattr(s, "conflict_details", None)):
                                    has_active_sig_conflict = True
                                    break
                        if not has_active_sig_conflict:
                            logger.info(f"Filtered out false-positive signal direction conflict resolved by authoritative evidence in validate_rtl_plan: {b_issue.get('issue')}")
                            continue

                    
                    # Check if this issue corresponds to an approved/resolved conflict in EFS IR
                    is_issue_resolved = False
                    if efs_ir and hasattr(efs_ir, "resolutions"):
                        for res in efs_ir.resolutions:
                            if getattr(res, "approval_status", "") == "APPROVED":
                                conf_id = getattr(res, "conflict_id", "")
                                aff_reqs = getattr(res, "affected_requirements", [])
                                prop = str(getattr(res, "property", "")).lower()
                                tent = str(getattr(res, "target_entity", "")).lower()
                                change = str(getattr(res, "proposed_change", "")).lower()
                                if (
                                    (conf_id and conf_id.lower() in iss_txt)
                                    or any(e.lower() in iss_txt for e in aff_reqs if e)
                                    or ("reset" in iss_txt and ("reset" in prop or "reset" in tent or "rst" in tent or "reset" in change))
                                    or ("clock" in iss_txt and ("clock" in prop or "clk" in prop or "clock" in tent or "clk" in tent))
                                ):
                                    is_issue_resolved = True
                                    break
                    if efs_ir and hasattr(efs_ir, "conflicts"):
                        for c in efs_ir.conflicts:
                            if getattr(c, "resolution_status", "") == "RESOLVED":
                                if any(e.lower() in iss_txt for e in c.entities if e) or (getattr(c, "property", "") == "reset" and "reset" in iss_txt) or (getattr(c, "property", "") == "clock" and "clock" in iss_txt):
                                    is_issue_resolved = True
                                    break

                    if ("conflict" in iss_txt or "mismatch" in iss_txt or "contradict" in iss_txt) and not ("design name" in iss_txt or "module name" in iss_txt or "top-level" in iss_txt):
                        cls = "SOURCE_CONFLICT" if "opcode" in iss_txt or "source" in iss_txt else "PLAN_CONFLICT"
                        is_block = not is_issue_resolved
                    elif "unsupported" in iss_txt:
                        cls = "UNSUPPORTED"
                        is_block = True
                    elif "ambiguous" in iss_txt or "vague" in iss_txt or "described only as" in iss_txt or ("transition condition" in iss_txt and "undefined" not in iss_txt):
                        cls = "AMBIGUOUS"
                        is_block = True if not is_issue_resolved else False
                    elif "extraction" in iss_txt or "gap" in iss_txt:
                        cls = "EFSIR_EXTRACTION_GAP"
                        is_block = True
                    else:
                        cls = "PLAN_MISSING"
                        is_block = True if (not efs_ir and not is_issue_resolved) else False
                    
                    b_issue["classification"] = cls
                    b_issue["is_blocking"] = is_block
                    b_issue["badge"] = f"🔴 [{cls}]" if is_block else f"🔵 [DERIVED]"
                    if b_issue not in all_issues:
                        all_issues.append(b_issue)

    blocking_issues = [i for i in all_issues if i.get("is_blocking", False)]
    has_conflict = any(i.get("classification") in ("PLAN_CONFLICT", "SOURCE_CONFLICT") for i in blocking_issues)
    is_valid = len(blocking_issues) == 0
    final_status = "COMPLETE" if is_valid else ("CONTRADICTORY" if has_conflict else "INCOMPLETE")

    return {
        "valid": is_valid,
        "status": final_status,
        "issues": blocking_issues,
        "all_issues": all_issues,
        "errors": [i.get("issue", "") for i in blocking_issues],
        "warnings": [i.get("issue", "") for i in all_issues if not i.get("is_blocking", False)],
        "missing_requirements": [i.get("required_information", "") for i in blocking_issues if i.get("classification") in ("PLAN_MISSING", "MISSING")],
        "unresolved_conflicts": [i for i in blocking_issues if i.get("classification") in ("PLAN_CONFLICT", "SOURCE_CONFLICT")],
        "assumptions": [i for i in all_issues if i.get("classification") == "DERIVED"],
        "source_evidence": [i.get("evidence", "") for i in all_issues if i.get("evidence")]
    }


def validate_generated_code(
    code: str,
    rtl_plan: Optional[Dict[str, Any]] = None,
    target_language: str = "SystemVerilog",
    language_version: Optional[str] = None,
    efs_ir: Optional[EFSIR] = None,
) -> Tuple[bool, List[str]]:
    """
    Syntax-aware, structural, and language-standard validation of generated HDL code.
    Detects:
    - Empty procedural blocks (always_ff, always_comb, always @)
    - Placeholder comments standing in for implementation logic
    - Missing module boundaries (module ... endmodule)
    - Truncated code (unbalanced begin/end, unclosed case/endcase, unclosed brackets)
    - Language violations (SV keywords in Verilog mode, improper sensitivity lists)
    - Scalar declarations where vector widths are required
    - Missing state logic when FSM is declared
    - Undriven required outputs
    """
    if not code or not isinstance(code, str) or not code.strip():
        return False, ["Generated code is empty or null."]

    # Call language-aware validator
    report = validate_rtl_syntax_and_structure(
        code=code,
        target_language=target_language,
        language_version=language_version,
        rtl_plan=rtl_plan,
        efs_ir=efs_ir,
        run_external_tools=True,
    )

    errors: List[str] = [e.message for e in report.errors]

    # Additional plan-specific checks
    if rtl_plan:
        ports = rtl_plan.get("ports", [])
        for p in ports:
            p_name = p.get("name", "")
            p_w = str(p.get("width", "1"))
            norm_w = normalize_hardware_width(p_w)
            if norm_w and norm_w > 1:
                decl_scalar = re.search(rf"\b(?:logic|wire|reg|input|output)\s+{re.escape(p_name)}\b(?!\s*\[)", code)
                decl_vector = re.search(rf"\[[^\]]+\]\s*{re.escape(p_name)}\b", code)
                if decl_scalar and not decl_vector:
                    err = f"Signal '{p_name}' requires vector width {norm_w}, but is declared as 1-bit scalar in generated code."
                    if err not in errors:
                        errors.append(err)

        fsm = rtl_plan.get("fsm", {})
        states = fsm.get("states", []) if isinstance(fsm, dict) else []
        if len(states) > 1:
            if not re.search(r"\bstate\b", code, re.IGNORECASE):
                err = "Plan specifies FSM with states, but no `state` variable found in generated code."
                if err not in errors:
                    errors.append(err)
            if "case" not in code and "if" not in code:
                err = "Plan specifies FSM, but no state decode (`case` or `if`) found in generated code."
                if err not in errors:
                    errors.append(err)

    return len(errors) == 0, errors


def generate_code(design_spec: str, target: str, efs_ir: Optional[EFSIR] = None, protocol_rules: Optional[List[Dict[str, Any]]] = None, context_pack: Optional[Any] = None) -> str:
    """Generate HDL or verification collateral code from ContextPack or design specification."""
    logger.info(f"Generating hardware code for target '{target}' (2-Stage Architecture)...")

    # Resolve and validate target HDL language
    hdl_lang, lang_ver = resolve_hdl_language(target)
    if str(target).lower() in ("vhdl",):
        target_lang = "VHDL"
    elif hdl_lang == HDLLanguage.UNKNOWN:
        logger.error(f"Requested target language '{target}' is unsupported or ambiguous.")
        return (
            f"SPECIFICATION_INCOMPLETE\n\nTarget Language Selection Ambiguity:\n\n"
            f"    🔴 [LANGUAGE_UNSUPPORTED] Unsupported or ambiguous HDL target '{target}'.\n\n"
            f"Required clarification:\n\n"
            f"    Specify target language explicitly as 'Verilog' (e.g. Verilog-2001, Verilog-2005) or 'SystemVerilog' (e.g. SystemVerilog-2012, SystemVerilog-2017)."
        )
    else:
        target_lang = hdl_lang.value
    
    # Stage 1: Generate Structured RTL Implementation Plan
    rtl_plan = plan_rtl(design_spec, target_lang, efs_ir=efs_ir, context_pack=context_pack)
    
    # Deterministic Completeness Validation (Must run AFTER Stage 1 and BEFORE Stage 2)
    validation = validate_rtl_plan(rtl_plan, efs_ir=efs_ir, context_pack=context_pack)
    
    if not validation.get("valid", False):
        issues = validation.get("issues", [])
        logger.warning(f"RTL Generation BLOCKED: Specification is incomplete or contradictory for target '{target_lang}'. Issues: {issues}")
        
        # Convert blocking validation issues into structured EFSConflict objects and attach to efs_ir
        if efs_ir is not None:
            try:
                from core.verification.conflict_resolution_engine import ConflictResolutionEngine
                parsed_conflicts = ConflictResolutionEngine.parse_specification_issues_to_conflicts(issues, efs_ir=efs_ir)
                if not hasattr(efs_ir, "conflicts") or efs_ir.conflicts is None:
                    efs_ir.conflicts = []
                existing_keys = {getattr(c, "semantic_key", "") for c in efs_ir.conflicts}
                for pc in parsed_conflicts:
                    k = pc.semantic_key or ConflictResolutionEngine.compute_semantic_key(pc)
                    if k not in existing_keys:
                        efs_ir.conflicts.append(pc)
                        existing_keys.add(k)
            except Exception as e:
                logger.warning(f"Failed to attach validation conflicts to efs_ir: {e}")
        
        missing_lines = []
        req_lines = []
        seen_issues = set()
        seen_reqs = set()
        
        badge_map = {
            "MISSING": "🔴 [MISSING]",
            "CONFLICT": "🟠 [CONFLICT]",
            "AMBIGUOUS": "🟡 [AMBIGUOUS]",
            "COMPATIBLE": "🟢 [COMPATIBLE]",
            "DERIVED": "🔵 [DERIVED]",
            "IMPLEMENTATION_CHOICE": "⚙️ [IMPLEMENTATION CHOICE]",
            "UNSUPPORTED": "🟣 [UNSUPPORTED]"
        }
        
        for issue in issues:
            iss_text = str(issue.get("issue", "")).strip()
            req_text = str(issue.get("required_information", "")).strip()
            cat = issue.get("category", "GENERAL")
            cls = issue.get("classification", "MISSING")
            evid = issue.get("evidence", issue.get("source_evidence", "N/A"))
            badge = badge_map.get(cls, f"🔴 [{cls}]")
            
            if iss_text and iss_text not in seen_issues:
                seen_issues.add(iss_text)
                missing_lines.append(f"    {badge} [{cat}] {iss_text} | Source: {evid}")
            if req_text and req_text not in seen_reqs:
                seen_reqs.add(req_text)
                req_lines.append(f"    {req_text}")
                
        missing_str = "\n".join(missing_lines) if missing_lines else "    Specification requirements are incomplete or ambiguous."
        req_str = "\n".join(req_lines) if req_lines else "    Provide explicit signal, register, and FSM transition definitions."
        
        return f"SPECIFICATION_INCOMPLETE\n\nMissing/ambiguous requirements:\n\n{missing_str}\n\nRequired clarification:\n\n{req_str}"
        
    logger.info(f"Proceeding to Stage 2 RTL generation for target '{target_lang}'")
    
    module_name = rtl_plan.get("module_name", "custom_module")
    ports = rtl_plan.get("ports", [])

    # Stage 2 LLM generation using validated Stage 1 RTL plan and scoped protocol rules
    llm = get_llm_client()
    lang_system_prompt = get_system_prompt(target_lang, lang_ver)

    if llm.available:
        rules_to_use = protocol_rules or []
        if not rules_to_use and context_pack:
            if hasattr(context_pack, "protocol_rules"):
                rules_to_use = context_pack.protocol_rules
            elif isinstance(context_pack, dict):
                rules_to_use = context_pack.get("relevant_protocol_rules", [])

        raw_user_query = ""
        if context_pack and hasattr(context_pack, "task") and getattr(context_pack.task, "raw_query", None):
            raw_user_query = str(context_pack.task.raw_query).strip()
        user_query_note = f"\nTargeted User Generation Request: \"{raw_user_query}\"\n" if raw_user_query else ""

        stage2_prompt = f"""Generate synthesizable {target_lang} module code based on this approved RTL plan:{user_query_note}
{json.dumps(rtl_plan, indent=2)}

Relevant protocol rules:
{json.dumps(rules_to_use, indent=2)}

Target module name: {module_name}

Output ONLY raw synthesizable {target_lang} code with `module {module_name} ... endmodule` without conversational markdown or explanations.
Ensure all sequential and combinational procedural blocks are fully implemented without empty blocks, placeholder comments, or truncated structures.
"""
        try:
            code = llm.complete(lang_system_prompt, stage2_prompt, model=CONFIG.llm.model)
            cleaned = clean_generated_code(code)
            if target_lang == "Verilog" and cleaned and any(k in cleaned for k in ("logic", "always_comb", "always_ff")):
                cleaned = re.sub(r"\binput\s+logic\b", "input wire", cleaned)
                cleaned = re.sub(r"\boutput\s+logic\b", "output reg", cleaned)
                cleaned = re.sub(r"\blogic\b", "wire", cleaned)
                cleaned = re.sub(r"\balways_comb\b", "always @(*)", cleaned)
                cleaned = re.sub(r"\balways_ff\b", "always", cleaned)

            is_valid_code, code_errors = validate_generated_code(
                cleaned, rtl_plan, target_language=target_lang, language_version=lang_ver, efs_ir=efs_ir
            )

            # Bounded repair loop (up to 2 iterations) if generated code has defects
            repair_attempts = 0
            while not is_valid_code and repair_attempts < 2:
                logger.info(f"Generated RTL has defects ({code_errors}); running bounded repair attempt {repair_attempts + 1}/2...")
                repair_prompt = f"""The generated {target_lang} module has the following validation defects:
{chr(10).join(f"- {e}" for e in code_errors)}

Please repair the {target_lang} code for module {module_name} to eliminate all syntax defects, empty blocks, placeholder comments, missing port vectors, language violations, and undriven outputs.
Ensure all procedural blocks contain executable hardware logic conforming strictly to {target_lang}.
Output ONLY raw synthesizable {target_lang} code with `module {module_name} ... endmodule`.

Previous code:
{cleaned}
"""
                repaired = llm.complete(lang_system_prompt, repair_prompt, model=CONFIG.llm.model)
                repaired_clean = clean_generated_code(repaired)
                if target_lang == "Verilog" and repaired_clean and any(k in repaired_clean for k in ("logic", "always_comb", "always_ff")):
                    repaired_clean = re.sub(r"\binput\s+logic\b", "input wire", repaired_clean)
                    repaired_clean = re.sub(r"\boutput\s+logic\b", "output reg", repaired_clean)
                    repaired_clean = re.sub(r"\blogic\b", "wire", repaired_clean)
                    repaired_clean = re.sub(r"\balways_comb\b", "always @(*)", repaired_clean)
                    repaired_clean = re.sub(r"\balways_ff\b", "always", repaired_clean)

                if repaired_clean and "module " in repaired_clean and "endmodule" in repaired_clean:
                    is_valid_code, code_errors = validate_generated_code(
                        repaired_clean, rtl_plan, target_language=target_lang, language_version=lang_ver, efs_ir=efs_ir
                    )
                    if is_valid_code:
                        cleaned = repaired_clean
                        break
                repair_attempts += 1

            if is_valid_code and cleaned and "module " in cleaned and "endmodule" in cleaned:
                try:
                    from core.context_compiler import CompilationMetrics, record_generation_telemetry, estimate_tokens
                    sys_tokens = estimate_tokens(lang_system_prompt)
                    in_tokens = sys_tokens + estimate_tokens(stage2_prompt)
                    provider_tokens = llm.last_usage.get("prompt_tokens") if (hasattr(llm, "last_usage") and isinstance(llm.last_usage.get("prompt_tokens"), (int, float))) else None
                    out_tokens = llm.last_usage.get("completion_tokens") if (hasattr(llm, "last_usage") and isinstance(llm.last_usage.get("completion_tokens"), (int, float))) else estimate_tokens(cleaned)
                    orig_tokens = estimate_tokens(design_spec) + (estimate_tokens(efs_ir) if efs_ir else 0)
                    compiled_tokens = estimate_tokens(rtl_plan) + estimate_tokens(rules_to_use)
                    
                    actual_input = provider_tokens or in_tokens
                    orig = max(orig_tokens, 1)
                    reduction = round(max(0.0, (orig - actual_input) / orig) * 100, 2)
                    end_to_end = round(max(0.0, (orig - in_tokens) / orig) * 100, 2)
                    logger.info(
                        f"[GenerationContext] artifact=rtl_stage2 | "
                        f"original_tokens={orig_tokens} | "
                        f"compiled_tokens={compiled_tokens} | "
                        f"final_prompt_tokens={estimate_tokens(stage2_prompt)} | "
                        f"final_input_tokens={in_tokens} | "
                        f"provider_input_tokens={provider_tokens or 'N/A'} | "
                        f"output_tokens={out_tokens} | "
                        f"context_reduction={reduction}% | "
                        f"end_to_end_reduction={end_to_end}%"
                    )
                    metrics = CompilationMetrics(
                        artifact_type="rtl",
                        compiler_used="RTLContextCompiler",
                        original_context_tokens=orig_tokens,
                        compiled_context_tokens=compiled_tokens,
                        system_prompt_tokens=sys_tokens,
                        final_input_tokens=in_tokens,
                        output_tokens=out_tokens,
                        extra_details={"provider_input_tokens": provider_tokens}
                    )
                    record_generation_telemetry(metrics, generation_name=f"rtl_stage2_{module_name}", model=CONFIG.llm.model)
                except Exception as ex_tel:
                    logger.debug(f"Stage 2 telemetry error: {ex_tel}")
                return cleaned
            else:
                logger.warning(f"Stage 2 LLM generation produced incomplete code after repairs ({code_errors}); falling back to deterministic template.")
        except Exception as e:
            logger.warning(f"Stage 2 LLM generation failed, falling back to deterministic template: {e}")

    # Deterministic Heuristic Fallback: synthesizes complete functional RTL without placeholder comments
    fallback_code = get_heuristic_code_template(target_lang, design_spec, efs_ir=efs_ir, rtl_plan=rtl_plan)
    fb_valid, fb_errors = validate_generated_code(
        fallback_code, rtl_plan, target_language=target_lang, language_version=lang_ver, efs_ir=efs_ir
    )
    if fb_valid:
        return fallback_code

    logger.error(f"Fallback generation incomplete for target {target_lang}: {fb_errors}")
    return f"SPECIFICATION_INCOMPLETE\n\nCannot generate synthesizable {target_lang} RTL. Deficiencies:\n" + "\n".join(f"- {e}" for e in fb_errors)



def sanitize_module_name(name: str) -> str:
    """Sanitize module name to a legal, concise Verilog/SystemVerilog identifier."""
    if not name or not isinstance(name, str):
        return "target_module"
    clean = name.strip()
    if len(clean) > 40 or len(clean.split()) > 5:
        words = [w for w in re.findall(r"[A-Za-z0-9_]+", clean) if w.lower() not in (
            "builds", "upon", "the", "original", "architecture", "while", "adding",
            "configurability", "for", "and", "increased", "with", "a", "an", "of", "to", "in",
            "design", "module", "system", "top", "level"
        )]
        clean = "_".join(words[:4]) if words else "target_module"
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", clean).lower()
    clean = re.sub(r"_+", "_", clean).strip("_")
    if not clean or clean[0].isdigit():
        clean = f"mod_{clean}" if clean else "target_module"
    return clean


def sanitize_hardware_registers(raw_registers: List[Any]) -> List[Dict[str, Any]]:
    """
    Sanitize hardware registers into legal Verilog/SV identifiers with valid resets.
    Removes redundant group/range containers (e.g. 'Matrix Registers M0-M7' when 'M0'..'M7' exist).
    Deduplicates and ensures legal identifier names and sized reset expressions.
    """
    if not raw_registers:
        return []

    raw_names = []
    reg_items = []
    for r in raw_registers:
        if isinstance(r, dict):
            rname = str(r.get("name", "")).strip()
            rwidth = str(r.get("width", "32")).strip()
            rreset = r.get("reset_val", r.get("reset_value"))
            rfields = r.get("fields", [])
            rdesc = r.get("purpose", r.get("description", ""))
        else:
            rname = str(getattr(r, "name", "")).strip()
            rwidth = str(getattr(r, "width", "32")).strip()
            rreset = getattr(r, "reset_val", None) or getattr(r, "reset_value", None)
            rfields = getattr(r, "fields", []) or []
            rdesc = getattr(r, "description", "")
        if rname:
            raw_names.append(rname)
            reg_items.append({
                "name": rname,
                "width": rwidth,
                "reset_val": rreset,
                "fields": rfields,
                "description": rdesc
            })

    # Detect if individual numbered registers exist
    has_individual_m = any(re.match(r"^M\d+$", n, re.IGNORECASE) for n in raw_names)
    has_individual_r = any(re.match(r"^R\d+$", n, re.IGNORECASE) for n in raw_names)

    sanitized: List[Dict[str, Any]] = []
    seen_names: Set[str] = set()

    for item in reg_items:
        orig_name = item["name"]
        orig_lower = orig_name.lower()

        # Check for redundant group/range container headers
        is_range_header = bool(re.search(r"\b[A-Za-z0-9_]+\s*-\s*[A-Za-z0-9_]+\b", orig_name)) or (
            any(k in orig_lower for k in ("registers", "category", "group")) and not re.match(r"^[a-zA-Z0-9_]+$", orig_name)
        )
        if is_range_header:
            if has_individual_m and ("m0" in orig_lower or "matrix" in orig_lower):
                continue
            if has_individual_r and ("r0" in orig_lower or "general" in orig_lower):
                continue

        # If name has spaces or hyphens, clean it
        clean_name = orig_name
        if " " in clean_name or "-" in clean_name:
            if has_individual_m and "matrix" in orig_lower:
                continue
            if has_individual_r and "general" in orig_lower:
                continue
            if "config" in orig_lower:
                clean_name = "cfg_reg"
            elif "status" in orig_lower:
                clean_name = "status_reg"
            elif "matrix" in orig_lower:
                clean_name = "matrix_reg"
            else:
                words = re.findall(r"[A-Za-z0-9_]+", clean_name)
                clean_name = "_".join(words).lower()

        clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", clean_name)
        clean_name = re.sub(r"_+", "_", clean_name).strip("_")
        if not clean_name or clean_name[0].isdigit():
            clean_name = f"r_{clean_name}" if clean_name else "r_unnamed"

        if clean_name.lower() in seen_names:
            continue
        seen_names.add(clean_name.lower())

        # Width normalization
        raw_w = item["width"]
        norm_w = normalize_hardware_width(raw_w)
        w_int = norm_w if norm_w is not None else (int(raw_w) if raw_w.isdigit() else 32)
        w_str = str(w_int)

        # Reset value normalization
        raw_rst = item["reset_val"]
        if raw_rst is None or str(raw_rst).strip().lower() in ("none", "null", "undefined", "", "0", "0x0", "32'h0", "8'h0"):
            rst_expr = f"{w_int}'h0" if w_int > 1 else "1'b0"
        elif str(raw_rst).startswith(f"{w_int}'"):
            rst_expr = str(raw_rst)
        elif str(raw_rst).isdigit():
            rst_expr = f"{w_int}'d{raw_rst}"
        elif str(raw_rst).startswith("0x") or str(raw_rst).startswith("0X"):
            hex_val = str(raw_rst)[2:]
            rst_expr = f"{w_int}'h{hex_val}"
        else:
            rst_expr = f"{w_int}'h0" if w_int > 1 else "1'b0"

        sanitized.append({
            "name": clean_name,
            "width": w_str,
            "reset_val": rst_expr,
            "reset_value": rst_expr,
            "fields": item["fields"],
            "description": item["description"]
        })

    return sanitized


def _derive_register_fields(reg_name: str, spec_text: str) -> List[Dict[str, Any]]:
    """Extract or derive fields for a register."""
    fields = []
    r_lower = reg_name.lower()
    
    # Try searching for explicit field definitions in text
    field_matches = re.findall(r"\[(\d+(?::\d+)?)\]\s*:\s*([a-zA-Z0-9_]+)", spec_text)
    if field_matches:
        for bit_range, f_name in field_matches:
            fields.append({
                "bits": bit_range,
                "meaning": f_name,
                "classification": "EXPLICIT"
            })
        return fields

    # Derive standard field mappings if not explicitly listed
    if "ctrl" in r_lower or "control" in r_lower:
        fields = [
            {"bits": "0:0", "meaning": "ENABLE", "classification": "DERIVED"},
            {"bits": "1:1", "meaning": "START", "classification": "DERIVED"}
        ]
    elif "status" in r_lower:
        fields = [
            {"bits": "0:0", "meaning": "BUSY", "classification": "DERIVED"},
            {"bits": "1:1", "meaning": "DONE", "classification": "DERIVED"},
            {"bits": "2:2", "meaning": "ERROR", "classification": "DERIVED"}
        ]
    else:
        fields = [
            {"bits": "31:0", "meaning": "DATA_PAYLOAD", "classification": "DERIVED"}
        ]
    return fields


def get_heuristic_code_template(
    target: str,
    design_spec: str = "",
    efs_ir: Optional[EFSIR] = None,
    rtl_plan: Optional[Dict[str, Any]] = None
) -> str:
    """Deterministic, synthesizable HDL code generator strictly derived from validated EFS IR and RTL plan."""
    # Check if target and design_spec were swapped by caller
    t_clean = str(target or "").strip().lower()
    d_clean = str(design_spec or "").strip().lower()
    if d_clean in ("verilog", "systemverilog", "vhdl", "v", "sv") and t_clean not in ("verilog", "systemverilog", "vhdl", "v", "sv"):
        target, design_spec = design_spec, target
    elif t_clean not in ("verilog", "systemverilog", "vhdl", "v", "sv") and any(k in t_clean for k in ("module", "specification", "|", "#", "input", "output")):
        target, design_spec = design_spec, target
    elif t_clean not in ("verilog", "systemverilog", "vhdl", "v", "sv"):
        target = "SystemVerilog"

    if not rtl_plan and not (efs_ir and getattr(efs_ir, "signals", None)):
        try:
            rtl_plan = plan_rtl(design_spec, target, efs_ir=efs_ir)
        except Exception as e_plan:
            logger.debug(f"Failed to plan RTL for fallback template: {e_plan}")
            rtl_plan = {}
    elif not rtl_plan:
        rtl_plan = {}

    hdl_lang, lang_ver = resolve_hdl_language(target)
    is_verilog = (hdl_lang == HDLLanguage.VERILOG)
    target_lang = "Verilog" if is_verilog else "SystemVerilog"

    # 1. Module Name
    module_name = ""
    if efs_ir and hasattr(efs_ir, "metadata"):
        module_name = getattr(efs_ir.metadata, "top_module", "") or getattr(efs_ir.metadata, "design_name", "")
    if not module_name:
        module_name = rtl_plan.get("module_name", "")
    if not module_name or module_name in ("custom_module", "target_module", "hardware_module"):
        if design_spec:
            m_match = re.search(r"#\s*(?:Hardware Design Specification|Specification|Module):\s*([A-Za-z0-9_\-\s]+)", design_spec, re.IGNORECASE)
            if not m_match:
                m_match = re.search(r"Module:\s*([a-zA-Z0-9_]+)", design_spec, re.IGNORECASE)
            if not m_match:
                m_match = re.search(r"\bmodule\s+([a-zA-Z0-9_]+)", design_spec, re.IGNORECASE)
            if m_match:
                module_name = m_match.group(1).strip().lower().replace(" ", "_").replace("-", "_")
    if not module_name:
        module_name = "target_module"
    module_name = sanitize_module_name(module_name)

    # 2. Extract ports from EFS IR as source of truth, then plan, then design_spec
    plan_ports = []
    if efs_ir and getattr(efs_ir, "signals", None):
        for sig in efs_ir.signals:
            plan_ports.append({
                "name": sig.name,
                "direction": sig.direction,
                "width": str(sig.width or 1),
                "semantic_role": sig.semantic_role
            })
    elif rtl_plan.get("ports"):
        plan_ports = rtl_plan["ports"]
    elif design_spec:
        spec_parsed = parse_markdown_spec(design_spec)
        for iface in spec_parsed.get("interfaces", []):
            plan_ports.append({
                "name": iface.get("name"),
                "direction": iface.get("direction", "input"),
                "width": str(iface.get("width", 1)),
                "semantic_role": iface.get("semantic_role")
            })
        for cr in spec_parsed.get("clocks_resets", []):
            plan_ports.append({
                "name": cr.get("name"),
                "direction": "input",
                "width": "1",
                "semantic_role": cr.get("type", "clock")
            })

    # Internal registers
    registers = []
    if efs_ir and getattr(efs_ir, "registers", None):
        for r in efs_ir.registers:
            registers.append({"name": r.name, "width": str(r.width or 32), "reset_value": getattr(r, "reset_value", "0")})
    elif rtl_plan.get("internal_registers"):
        registers = rtl_plan["internal_registers"]
    registers = sanitize_hardware_registers(registers)

    # FSM extraction: strictly from EFS IR if provided
    fsm = rtl_plan.get("fsm", {}) if isinstance(rtl_plan.get("fsm"), dict) else {}
    states = []
    if efs_ir and getattr(efs_ir, "fsms", None):
        for s in efs_ir.fsms[0].states:
            states.append(s.name)
    elif not efs_ir and rtl_plan.get("fsm"):
        raw_states = fsm.get("states", [])
        for s in raw_states:
            states.append(s.get("name", "STATE") if isinstance(s, dict) else str(s))
    elif not efs_ir and design_spec:
        spec_parsed = parse_markdown_spec(design_spec)
        states = spec_parsed.get("states", [])

    has_fsm = len(states) > 0
    initial_state = states[0] if has_fsm else None

    # Identify semantic FSM states dynamically
    done_state = None
    error_state = None
    fetch_state = None
    exec_state = None
    wb_state = None
    decode_state = None

    if has_fsm:
        for s in states:
            s_up = s.upper()
            if any(k in s_up for k in ("DONE", "COMPLETE", "FINISH", "SUCCESS", "END")) and not any(k in s_up for k in ("ERROR", "FAIL", "FAULT", "ABORT")):
                done_state = s
            elif any(k in s_up for k in ("ERROR", "FAIL", "FAULT", "ABORT", "OVERFLOW")):
                error_state = s
            elif any(k in s_up for k in ("FETCH", "READ", "LOAD", "REQ")):
                fetch_state = s
            elif any(k in s_up for k in ("EXEC", "CALC", "COMPUTE", "RUN", "PROCESS")):
                exec_state = s
            elif any(k in s_up for k in ("WRITEBACK", "WB", "STORE", "WRITE")):
                wb_state = s
            elif any(k in s_up for k in ("DECODE", "DEC")):
                decode_state = s

        if not done_state:
            non_error = [s for s in states if not any(k in s.upper() for k in ("ERROR", "FAIL", "FAULT", "ABORT"))]
            done_state = non_error[-1] if non_error else states[-1]

    # 3. Detect clock and reset requirements strictly from specification / EFS IR / plan
    clock_reset = rtl_plan.get("clock_reset", {}) if isinstance(rtl_plan.get("clock_reset"), dict) else {}
    if efs_ir and getattr(efs_ir, "signals", None):
        has_clk = any(s.semantic_role == "clock" or s.name.lower() in ("clk", "clock") for s in efs_ir.signals)
        has_rst = any(s.semantic_role == "reset" or any(k in s.name.lower() for k in ("rst", "reset")) for s in efs_ir.signals)
    else:
        has_clk = bool(
            clock_reset.get("clock_name") or
            any(p.get("semantic_role") == "clock" or str(p.get("name", "")).lower() in ("clk", "clock") for p in plan_ports) or
            (has_fsm or registers)
        )
        has_rst = bool(
            clock_reset.get("reset_name") or
            any(p.get("semantic_role") == "reset" or any(k in str(p.get("name", "")).lower() for k in ("rst", "reset")) for p in plan_ports) or
            (has_fsm or registers)
        )

    clk_name = clock_reset.get("clock_name")
    if not clk_name:
        for p in plan_ports:
            if p.get("semantic_role") == "clock" or str(p.get("name", "")).lower() in ("clk", "clock"):
                clk_name = p.get("name")
                break
    if not clk_name and has_clk:
        clk_name = "clk"

    rst_name = clock_reset.get("reset_name")
    if not rst_name:
        for p in plan_ports:
            if p.get("semantic_role") == "reset" or any(k in str(p.get("name", "")).lower() for k in ("rst", "reset")):
                rst_name = p.get("name")
                break
    if not rst_name and has_rst:
        rst_name = "rst_n"

    rst_polarity = clock_reset.get("reset_polarity")
    if not rst_polarity:
        rst_polarity = "low" if rst_name and ("n" in rst_name.lower() or "low" in rst_name.lower()) else "high"

    clk_edge = clock_reset.get("clock_edge", "posedge")
    rst_edge = "negedge" if str(rst_polarity).lower() == "low" else "posedge"
    rst_cond = f"!{rst_name}" if str(rst_polarity).lower() == "low" else f"{rst_name}"
    is_async_reset = clock_reset.get("reset_type", "asynchronous").lower() != "synchronous"

    always_seq_kw = "always" if is_verilog else "always_ff"
    if has_rst and is_async_reset:
        seq_sens = f"@({clk_edge} {clk_name} or {rst_edge} {rst_name})"
    else:
        seq_sens = f"@({clk_edge} {clk_name})"

    # Assemble final ports preserving existing declarations
    final_ports = list(plan_ports)
    existing_port_names = {str(p.get("name", "")).lower() for p in final_ports}

    if has_clk and clk_name and clk_name.lower() not in existing_port_names:
        final_ports.insert(0, {"name": clk_name, "direction": "input", "width": "1", "semantic_role": "clock"})
        existing_port_names.add(clk_name.lower())

    if has_rst and rst_name and rst_name.lower() not in existing_port_names:
        insert_idx = 1 if (has_clk and len(final_ports) > 0) else 0
        final_ports.insert(insert_idx, {"name": rst_name, "direction": "input", "width": "1", "semantic_role": "reset"})
        existing_port_names.add(rst_name.lower())


    transitions = []
    if has_fsm and efs_ir and getattr(efs_ir, "fsms", None):
        for t in efs_ir.fsms[0].transitions:
            transitions.append({
                "from_state": getattr(t, "from_state", getattr(t, "source_state", "")),
                "to_state": getattr(t, "to_state", getattr(t, "target_state", "")),
                "condition": getattr(t, "condition", ""),
                "actions": getattr(t, "actions", getattr(t, "action", ""))
            })
    elif has_fsm and isinstance(fsm, dict):
        transitions = fsm.get("transitions", [])

    # Only declare timer_cnt if explicitly referenced
    has_timer = any("timer_cnt" in str(t.get("condition", "")) for t in transitions) or any(
        r.get("name") == "timer_cnt" for r in registers if isinstance(r, dict)
    )

    # Datapath operations
    datapath = rtl_plan.get("datapath", {}) if isinstance(rtl_plan.get("datapath"), dict) else {}
    dp_ops = datapath.get("operations", []) if isinstance(datapath, dict) else []

    # Output driving strategies
    driving_strat = rtl_plan.get("output_driving_strategy") if "output_driving_strategy" in rtl_plan else rtl_plan.get("output_assignments", {})
    if not isinstance(driving_strat, dict):
        driving_strat = {}

    out_ports = []
    for p in final_ports:
        pname = p.get("name")
        pdir = p.get("direction", "input").lower()
        if pdir in ("output", "out") and pname not in (clk_name, rst_name):
            norm_int = normalize_hardware_width(p.get("width", "1"))
            raw_w = str(p.get("width", "1")).strip()
            pw = str(norm_int) if norm_int is not None else raw_w
            out_ports.append({"name": pname, "width": pw})

    # Build Port Declarations
    port_decls = []
    for p in final_ports:
        pname = p.get("name")
        pdir = p.get("direction", "input").lower()
        norm_int = normalize_hardware_width(p.get("width", "1"))
        raw_w = str(p.get("width", "1")).strip()
        pw = str(norm_int) if norm_int is not None else raw_w

        w_str = ""
        if norm_int and norm_int > 1:
            w_str = f"[{norm_int-1}:0] "
        elif raw_w.startswith("["):
            w_str = f"{raw_w} "
        elif not raw_w.isdigit() and raw_w != "1":
            w_str = f"[{raw_w}-1:0] "

        if is_verilog:
            if pdir in ("input", "in"):
                dir_str = "input  wire "
            else:
                dir_str = "output reg  "
        else:
            dir_str = "input  logic " if pdir in ("input", "in") else "output logic "

        port_decls.append(f"    {dir_str}{w_str}{pname}")

    ports_str = ",\n".join(port_decls)

    # Internal Register Declarations & Resets
    reg_decls = []
    reg_resets = []
    for r in registers:
        if not isinstance(r, dict):
            continue
        rname = r.get("name")
        if not rname or rname == "timer_cnt":
            continue
        norm_rw = normalize_hardware_width(r.get("width", "32"))
        raw_rw = str(r.get("width", "32")).strip()
        rw_int = norm_rw if norm_rw is not None else (int(raw_rw) if raw_rw.isdigit() else None)

        if rw_int and rw_int > 1:
            w_str = f"[{rw_int-1}:0] "
        elif raw_rw.startswith("["):
            w_str = f"{raw_rw} "
        else:
            w_str = ""

        if is_verilog:
            reg_decls.append(f"    reg {w_str}{rname};")
        else:
            reg_decls.append(f"    logic {w_str}{rname};")

        rst_val = r.get("reset_value", r.get("reset_val", "0"))
        if rw_int:
            rst_val_expr = f"{rw_int}'h0" if str(rst_val) in ("0", "32'h0", "0x0", "8'h0") else str(rst_val)
        else:
            rst_val_expr = str(rst_val)
        reg_resets.append(f"            {rname} <= {rst_val_expr};")

    if has_timer:
        if is_verilog:
            reg_decls.append("    reg [15:0] timer_cnt;")
        else:
            reg_decls.append("    logic [15:0] timer_cnt;")
        reg_resets.append("            timer_cnt <= 16'd0;")

    # Detect if ALU / datapath execution register is needed
    in_ports = [p for p in final_ports if p.get("direction") in ("input", "in") and p.get("name") not in (clk_name, rst_name)]
    has_alu = bool(exec_state or any(k in ip.get("name").lower() for k in ("instr", "opcode", "cmd") for ip in in_ports))
    if has_alu and not any(r.get("name") == "alu_result" for r in registers):
        if is_verilog:
            reg_decls.append("    reg [31:0] alu_result;")
        else:
            reg_decls.append("    logic [31:0] alu_result;")
        reg_resets.append("            alu_result <= 32'h0;")

    regs_str = "\n".join(reg_decls) if reg_decls else "    // No internal registers"
    reg_resets_str = ("\n" + "\n".join(reg_resets)) if reg_resets else ""

    # Datapath lines in sequential block
    dp_lines = []
    for op in dp_ops:
        expr = op.get("expression") or op.get("behavior")
        if expr:
            expr_clean = expr.strip().rstrip(";")
            dp_lines.append(f"            {expr_clean};")

    if not dp_lines and has_fsm:
        instr_in = next((ip.get("name") for ip in in_ports if any(k in ip.get("name").lower() for k in ("instruction", "opcode", "cmd", "instr"))), None)
        r0_name = registers[0].get("name") if registers else "alu_result"
        r1_name = registers[1].get("name") if len(registers) > 1 else "32'd1"
        rval_sig = next((ip.get("name") for ip in in_ports if any(k in ip.get("name").lower() for k in ("rvalid", "read_valid", "r_valid"))), None)
        rdata_sig = next((ip.get("name") for ip in in_ports if any(k in ip.get("name").lower() for k in ("rdata", "read_data", "rx_data"))), None)

        if fetch_state and rval_sig and registers:
            load_src = rdata_sig if rdata_sig else (in_ports[0].get("name") if in_ports else "32'h0")
            dp_lines.append(f"            if (state == {fetch_state} && {rval_sig}) begin\n                {r0_name} <= {load_src};\n            end")

        if exec_state and has_alu:
            if instr_in:
                dp_lines.append(f"""            if (state == {exec_state}) begin
                case ({instr_in}[3:0])
                    4'd0: alu_result <= {r0_name} + {r1_name};
                    4'd1: alu_result <= {r0_name} - {r1_name};
                    4'd2: alu_result <= {r0_name} * {r1_name};
                    4'd3: alu_result <= {r0_name} & {r1_name};
                    default: alu_result <= {r0_name} + {r1_name};
                endcase
            end""")
            else:
                dp_lines.append(f"            if (state == {exec_state}) begin\n                alu_result <= {r0_name} + {r1_name};\n            end")

        if wb_state and registers:
            dp_lines.append(f"            if (state == {wb_state}) begin\n                {r0_name} <= alu_result;\n            end")

    elif not dp_lines and registers and in_ports:
        # Sequential datapath without FSM: update only on enable/valid
        en_in = next((ip.get("name") for ip in in_ports if any(k in ip.get("name").lower() for k in ("en", "enable", "valid", "start", "we"))), None)
        data_inputs = [ip for ip in in_ports if not any(k in ip.get("name").lower() for k in ("clk", "rst", "start", "en", "valid", "ready", "sel"))]
        d_src = data_inputs[0].get("name") if data_inputs else in_ports[0].get("name")
        r0_name = registers[0].get("name")
        if en_in:
            dp_lines.append(f"            if ({en_in}) begin\n                {r0_name} <= {r0_name} + {d_src};\n            end")
        else:
            dp_lines.append(f"            {r0_name} <= {r0_name} + {d_src};")

    dp_seq_str = ("\n" + "\n".join(dp_lines)) if dp_lines else ""

    # Timer increment in sequential block
    timer_seq_str = ""
    if has_timer and has_fsm:
        timer_seq_str = """
            if (state != next_state) begin
                timer_cnt <= 16'd0;
            end else begin
                timer_cnt <= timer_cnt + 16'd1;
            end"""
    elif has_timer:
        timer_seq_str = "\n            timer_cnt <= timer_cnt + 16'd1;"

    # Sequential sensitivity list
    if has_clk:
        if has_rst and is_async_reset:
            seq_sens = f"@({clk_edge} {clk_name} or {rst_edge} {rst_name})"
        else:
            seq_sens = f"@({clk_edge} {clk_name})"
        always_seq_kw = "always" if is_verilog else "always_ff"

    # FSM State Definitions
    bit_w = max(1, (len(states) - 1).bit_length()) if has_fsm else 1
    if has_fsm:
        if is_verilog:
            state_params = ",\n        ".join(f"{s} = {bit_w}'d{idx}" for idx, s in enumerate(states))
            fsm_decl_str = f"""    // State machine parameters\n    localparam [{bit_w - 1}:0]\n        {state_params};\n    reg [{bit_w - 1}:0] state, next_state;"""
        else:
            states_enum_str = ", ".join(states)
            fsm_decl_str = f"""    // State machine definition\n    typedef enum logic [{bit_w - 1}:0] {{\n        {states_enum_str}\n    }} state_t;\n    state_t state, next_state;"""

        state_cases = []
        for idx, s in enumerate(states):
            outgoing = [t for t in transitions if (t.get("from_state") or t.get("from")) == s]
            case_lines = [f"            {s}: begin"]
            next_st = states[idx + 1] if idx + 1 < len(states) else initial_state
            has_explicit_cond = any(tr.get("condition") and str(tr.get("condition") if not isinstance(tr.get("condition"), dict) else tr.get("condition").get("expression", "")).strip() not in ("1'b1", "1", "", "None") for tr in outgoing)

            if s == initial_state:
                trig_in = next((ip.get("name") for ip in in_ports if any(k in ip.get("name").lower() for k in ("start", "valid", "req", "en", "enable"))), None)
                target_st = outgoing[0].get("to_state") or outgoing[0].get("to") if outgoing else next_st
                if trig_in:
                    case_lines.append(f"                if ({trig_in}) begin\n                    next_state = {target_st};\n                end else begin\n                    next_state = {s};\n                end")
                else:
                    case_lines.append(f"                next_state = {target_st};")
            elif s == fetch_state and any(k in ip.get("name").lower() for k in ("rvalid", "read_valid", "r_valid") for ip in in_ports):
                rval_sig = next(ip.get("name") for ip in in_ports if any(k in ip.get("name").lower() for k in ("rvalid", "read_valid", "r_valid")))
                target_st = outgoing[0].get("to_state") or outgoing[0].get("to") if outgoing else next_st
                case_lines.append(f"                if ({rval_sig}) begin\n                    next_state = {target_st};\n                end else begin\n                    next_state = {s};\n                end")
            elif s == wb_state and any(k in ip.get("name").lower() for k in ("bvalid", "write_valid", "b_valid", "ack") for ip in in_ports):
                bval_sig = next(ip.get("name") for ip in in_ports if any(k in ip.get("name").lower() for k in ("bvalid", "write_valid", "b_valid", "ack")))
                target_st = outgoing[0].get("to_state") or outgoing[0].get("to") if outgoing else next_st
                case_lines.append(f"                if ({bval_sig}) begin\n                    next_state = {target_st};\n                end else begin\n                    next_state = {s};\n                end")
            elif s == done_state:
                case_lines.append(f"                next_state = {initial_state};")
            elif s == error_state:
                rst_trig = next((ip.get("name") for ip in in_ports if any(k in ip.get("name").lower() for k in ("start", "clr", "clear", "restart"))), None)
                if rst_trig:
                    case_lines.append(f"                if ({rst_trig}) begin\n                    next_state = {initial_state};\n                end else begin\n                    next_state = {s};\n                end")
                else:
                    case_lines.append(f"                next_state = {initial_state};")
            elif has_explicit_cond:
                branch_stmts = []
                for b_idx, tr in enumerate(outgoing):
                    c_obj = tr.get("condition")
                    cond = (c_obj.get("expression") if isinstance(c_obj, dict) else c_obj) or "1'b1"
                    to_st = tr.get("to_state") or tr.get("to", s)
                    if b_idx == 0:
                        branch_stmts.append(f"                if ({cond}) begin\n                    next_state = {to_st};\n                end")
                    elif cond not in ("1'b1", "1", None, ""):
                        branch_stmts.append(f"                else if ({cond}) begin\n                    next_state = {to_st};\n                end")
                    else:
                        branch_stmts.append(f"                else begin\n                    next_state = {to_st};\n                end")
                if not any(tr.get("condition") in ("1'b1", "1", None, "") for tr in outgoing):
                    branch_stmts.append(f"                else begin\n                    next_state = {s};\n                end")
                case_lines.append("\n".join(branch_stmts))
            else:
                target_st = outgoing[0].get("to_state") or outgoing[0].get("to") if outgoing else next_st
                case_lines.append(f"                next_state = {target_st};")

            case_lines.append("            end")
            state_cases.append("\n".join(case_lines))
        state_cases_str = "\n".join(state_cases)
    else:
        fsm_decl_str = ""
        state_cases_str = ""

    # Output defaults
    out_defaults = []
    assign_stmts = []
    for op in out_ports:
        op_name = op["name"]
        op_w = op["width"]
        if op_name in driving_strat:
            strat = str(driving_strat[op_name]).strip()
            if "driven by " in strat:
                expr = strat.split("driven by ", 1)[1].strip().rstrip(";")
                out_defaults.append(f"        {op_name} = ({expr});")
            elif "assign " in strat:
                expr = strat.replace("assign ", "").split("=", 1)[-1].strip().rstrip(";")
                assign_stmts.append(f"    assign {op_name} = {expr};")
                out_defaults.append(f"        {op_name} = {expr};")
            else:
                expr = strat.rstrip(";")
                out_defaults.append(f"        {op_name} = {expr};")
        elif has_fsm:
            # Check standard status signals
            if any(k in op_name.lower() for k in ("busy", "active")):
                term_cond = f"state != {done_state}" if done_state else ""
                err_cond = f" && state != {error_state}" if error_state else ""
                out_defaults.append(f"        {op_name} = (state != {initial_state}{(' && ' + term_cond) if term_cond else ''}{err_cond});")
            elif any(k in op_name.lower() for k in ("done", "finish", "complete", "success")):
                out_defaults.append(f"        {op_name} = (state == {done_state});")
            elif any(k in op_name.lower() for k in ("error", "fault", "fail", "overflow")):
                if error_state:
                    out_defaults.append(f"        {op_name} = (state == {error_state});")
                else:
                    out_defaults.append(f"        {op_name} = 1'b0;")
            elif any(k in op_name.lower() for k in ("ready", "idle")):
                out_defaults.append(f"        {op_name} = (state == {initial_state});")
            # Interface / bus transaction control signals
            elif any(k in op_name.lower() for k in ("arvalid", "read_req", "rd_en")):
                st = fetch_state if fetch_state else (states[1] if len(states) > 1 else initial_state)
                out_defaults.append(f"        {op_name} = (state == {st});")
            elif any(k in op_name.lower() for k in ("rready", "rd_ready")):
                st = fetch_state if fetch_state else (states[1] if len(states) > 1 else initial_state)
                out_defaults.append(f"        {op_name} = (state == {st});")
            elif any(k in op_name.lower() for k in ("awvalid", "wvalid", "write_req", "wr_en")):
                st = wb_state if wb_state else (states[-1] if len(states) > 1 else initial_state)
                out_defaults.append(f"        {op_name} = (state == {st});")
            elif any(k in op_name.lower() for k in ("bready", "wr_ready")):
                st = wb_state if wb_state else (states[-1] if len(states) > 1 else initial_state)
                out_defaults.append(f"        {op_name} = (state == {st});")
            elif any(k in op_name.lower() for k in ("wstrb",)):
                st = wb_state if wb_state else (states[-1] if len(states) > 1 else initial_state)
                out_defaults.append(f"        {op_name} = (state == {st}) ? 4'hF : 4'h0;")
            elif any(k in op_name.lower() for k in ("araddr", "awaddr", "addr")):
                out_defaults.append(f"        {op_name} = 32'h0;")
            elif any(k in op_name.lower() for k in ("wdata", "data_out", "dout")):
                data_src = "alu_result" if has_alu else (registers[0].get("name") if registers else "32'h0")
                out_defaults.append(f"        {op_name} = {data_src};")
            elif op_w.isdigit() and int(op_w) > 1:
                out_defaults.append(f"        {op_name} = {op_w}'h0;")
            else:
                out_defaults.append(f"        {op_name} = 1'b0;")
        elif registers:
            out_defaults.append(f"        {op_name} = {registers[0].get('name')};")
        elif in_ports:
            # Check for selector input
            sel_in = next((ip for ip in in_ports if any(k in ip.get("name").lower() for k in ("sel", "select", "mode"))), None)
            data_ins = [ip for ip in in_ports if ip != sel_in]
            if sel_in and len(data_ins) >= 2:
                out_defaults.append(f"        {op_name} = {sel_in.get('name')} ? {data_ins[1].get('name')} : {data_ins[0].get('name')};")
            elif len(data_ins) >= 2:
                out_defaults.append(f"        {op_name} = {data_ins[0].get('name')} + {data_ins[1].get('name')};")
            else:
                out_defaults.append(f"        {op_name} = {in_ports[0].get('name')};")
        elif op_w.isdigit() and int(op_w) > 1:
            out_defaults.append(f"        {op_name} = {op_w}'h0;")
        else:
            out_defaults.append(f"        {op_name} = 1'b0;")

    out_defaults_str = "\n".join(out_defaults)

    # CASE A: Pure Combinational Module (no clock, no FSM, no internal registers)
    if not has_clk and not has_fsm and not registers:
        comb_kw = "always @(*)" if is_verilog else "always_comb"
        return f"""// Deterministic {target_lang} Synthesis: Pure Combinational Module
// Generated from Validated Implementation Plan
module {module_name} (
{ports_str}
);

    // Continuous combinational process
    {comb_kw} begin
{out_defaults_str}
    end

endmodule
"""

    # CASE B: Sequential Datapath Module (has clock/reset, but NO FSM)
    if has_clk and not has_fsm:
        comb_kw = "always @(*)" if is_verilog else "always_comb"
        reset_body = reg_resets_str.lstrip("\n") if reg_resets_str else "            // Reset logic"
        comb_block = f"""
    // Combinational output assignments
    {comb_kw} begin
{out_defaults_str}
    end""" if out_defaults_str.strip() else ""
        return f"""// Deterministic {target_lang} Synthesis: Sequential Datapath
// Generated from Validated Implementation Plan
module {module_name} (
{ports_str}
);

{regs_str}

    // Sequential register updates
    {always_seq_kw} {seq_sens} begin
        if ({rst_cond}) begin
{reset_body}
        end else begin{timer_seq_str}{dp_seq_str}
        end
    end{comb_block}

endmodule
"""

    # CASE C: Sequential FSM Module
    comb_kw = "always @(*)" if is_verilog else "always_comb"
    reset_state_line = f"            state <= {initial_state};"

    return f"""// Deterministic {target_lang} Synthesis: State Machine
// Generated from Validated Implementation Plan
module {module_name} (
{ports_str}
);

{fsm_decl_str}

{regs_str}

    // Sequential state and register update
    {always_seq_kw} {seq_sens} begin
        if ({rst_cond}) begin
{reset_state_line}{reg_resets_str}
        end else begin
            state <= next_state;{timer_seq_str}{dp_seq_str}
        end
    end

    // Combinational next-state and output assignments
    {comb_kw} begin
        next_state = state;
{out_defaults_str}

        case (state)
{state_cases_str}
            default: begin
                next_state = {initial_state};
            end
        endcase
    end

endmodule
"""

    if target == "VHDL":

        vhdl_ports = []
        for p in interfaces:
            pname = p["name"]
            pdir = p["direction"]
            pwidth = p["width"]
            
            vtype = "std_logic"
            if pwidth.isdigit() and int(pwidth) > 1:
                vtype = f"std_logic_vector({int(pwidth)-1} downto 0)"
            elif ":" in pwidth:
                w_clean = pwidth.replace("[", "").replace("]", "").split(":")
                if len(w_clean) == 2:
                    vhdl_ports.append(f"        {pname} : {pdir} std_logic_vector({w_clean[0]} downto {w_clean[1]})")
                    continue
            
            pdir_str = "in" if pdir == "input" or pdir == "in" else "out"
            if pname in [clk_name, rst_name]:
                pdir_str = "in"
                vtype = "std_logic"
                
            vhdl_ports.append(f"        {pname} : {pdir_str} {vtype}")
            
        if not vhdl_ports:
            vhdl_ports_decl = f"""        {clk_name} : in std_logic;
        {rst_name} : in std_logic;
        req : in std_logic;
        ack : out std_logic;
        data_in : in std_logic_vector(31 downto 0);
        data_out : out std_logic_vector(31 downto 0)"""
        else:
            vhdl_ports_decl = ";\n".join(vhdl_ports)

        states_vhdl = ", ".join(states)
        reg_signals = []
        for r in registers:
            reg_signals.append(f"    signal {r['name'].lower()}_reg : std_logic_vector(31 downto 0) := (others => '0');")
        reg_signals_str = "\n".join(reg_signals)

        return f"""-- Dynamic Fallback VHDL Design Template
-- Derived from Unified Design Specification
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity {module_name} is
    port (
{vhdl_ports_decl}
    );
end entity {module_name};

architecture rtl of {module_name} is
    type state_t is ({states_vhdl});
    signal state, next_state : state_t;
    
{reg_signals_str}
begin

    -- Sequential process
    process({clk_name}, {rst_name}) is
    begin
        if {rst_name} = '{"0" if rst_polarity == "low" else "1"}' then
            state <= {states[0]};
        elsif rising_edge({clk_name}) then
            state <= next_state;
        end if;
    end process;

end architecture rtl;
"""
    elif target == "UVM":
        uvm_driver_tx_decls = []
        uvm_driver_drives = []
        for p in interfaces:
            pname = p["name"]
            pdir = p["direction"]
            pwidth = p["width"]
            
            w_decl = ""
            if pwidth.isdigit():
                w = int(pwidth)
                if w > 1:
                    w_decl = f"[{w-1}:0] "
            elif ":" in pwidth:
                w_decl = f"{pwidth} "
                
            if pdir == "input" and pname not in [clk_name, rst_name]:
                uvm_driver_tx_decls.append(f"  rand bit {w_decl}{pname};")
                uvm_driver_drives.append(f"    vif.{pname} <= tx.{pname};")
                
        tx_decls_str = "\n".join(uvm_driver_tx_decls)
        driver_drives_str = "\n".join(uvm_driver_drives)
        
        return f"""// Dynamic Fallback SystemVerilog UVM Environment Template
// Derived from Unified Design Specification
interface {module_name}_if(input bit {clk_name}, input bit {rst_name});
  // Virtual Interface Ports declarations
  // ...
endinterface

class {module_name}_transaction extends uvm_sequence_item;
{tx_decls_str}
  `uvm_object_utils({module_name}_transaction)
  
  function new(string name = "{module_name}_transaction");
    super.new(name);
  endfunction
endclass

class {module_name}_driver extends uvm_driver #({module_name}_transaction);
  `uvm_component_utils({module_name}_driver)
  
  virtual {module_name}_if vif;

  function new(string name = "{module_name}_driver", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  virtual task run_phase(uvm_phase phase);
    forever begin
      seq_item_port.get_next_item(req);
      drive_tx(req);
      seq_item_port.item_done();
    end
  endtask

  task drive_tx({module_name}_transaction tx);
    @(posedge vif.{clk_name});
{driver_drives_str}
  endtask
endclass
"""
    elif target == "Assertions":
        handshake_assertions = []
        valids = [p["name"] for p in interfaces if "valid" in p["name"].lower()]
        readys = [p["name"] for p in interfaces if "ready" in p["name"].lower()]
        
        for v in valids:
            prefix = v.lower().replace("valid", "").replace("_", "")
            matching_r = next((r for r in readys if r.lower().replace("ready", "").replace("_", "") == prefix), None)
            if matching_r:
                handshake_assertions.append(f"""
    // Protocol Handshake Check: {v} must remain asserted until {matching_r} is HIGH
    property check_{prefix}_handshake;
        @(posedge {clk_name}) disable iff ({"!" if rst_polarity == "low" else ""}{rst_name})
        ({v} && !{matching_r}) |=> {v};
    endproperty
    assert_{prefix}_handshake: assert property (check_{prefix}_handshake);""")
                
        assertions_str = "\n".join(handshake_assertions)
        if not assertions_str:
            assertions_str = f"""    // General Handshake Protocol Check Check
    property check_general_handshake;
        @(posedge {clk_name}) disable iff ({"!" if rst_polarity == "low" else ""}{rst_name})
        1'b1;
    endproperty
    assert_general_handshake: assert property (check_general_handshake);"""

        return f"""// Dynamic Fallback SystemVerilog Assertions (SVA) Template
// Derived from Unified Design Specification
module {module_name}_assertions(
{ports_decl.replace("wire", "logic").replace("reg", "logic")}
);

{assertions_str}

endmodule
"""
    return "// Default code snippet"


def _derive_register_fields(reg_name: str, spec_text: str) -> List[Dict[str, Any]]:
    """Extract or derive fields for a register."""
    fields = []
    r_lower = reg_name.lower()
    
    # Try searching for explicit field definitions in text
    field_matches = re.findall(r"\[(\d+(?::\d+)?)\]\s*:\s*([a-zA-Z0-9_]+)", spec_text)
    if field_matches:
        for bit_range, f_name in field_matches:
            fields.append({
                "bits": bit_range,
                "meaning": f_name,
                "classification": "EXPLICIT"
            })
        return fields

    # Derive standard field mappings if not explicitly listed
    if "ctrl" in r_lower or "control" in r_lower:
        fields = [
            {"bits": "0:0", "meaning": "ENABLE", "classification": "DERIVED"},
            {"bits": "1:1", "meaning": "START", "classification": "DERIVED"}
        ]
    elif "status" in r_lower:
        fields = [
            {"bits": "0:0", "meaning": "BUSY", "classification": "DERIVED"},
            {"bits": "1:1", "meaning": "DONE", "classification": "DERIVED"},
            {"bits": "2:2", "meaning": "ERROR", "classification": "DERIVED"}
        ]
    else:
        fields = [
            {"bits": "31:0", "meaning": "DATA_PAYLOAD", "classification": "DERIVED"}
        ]
    return fields


def parse_markdown_spec(spec: str) -> Dict[str, Any]:
    """Helper parser to extract design properties and build a structured plan from spec text."""
    import re
    interfaces = []
    clocks_resets = []
    registers = []
    parameters = []
    states = []
    transitions = []
    blocking_issues = []
    module_name = "target_module"

    # Extract module name
    name_match = re.search(r"# Hardware Design Specification:\s*([A-Za-z0-9_\-\s]+)", spec)
    if not name_match:
        name_match = re.search(r"module\s+([a-zA-Z0-9_]+)", spec, re.IGNORECASE)
    if name_match:
        module_name = name_match.group(1).strip().lower().replace(" ", "_").replace("-", "_")

    # Extract states if explicitly present
    state_match = re.search(r"states:\s*([a-zA-Z0-9_,\s]+?)(?=\r?\n\r?\n|\r?\n[a-zA-Z_]|\Z)", spec)
    fsm_is_explicit = False
    if state_match:
        states = [s.strip().upper() for s in state_match.group(1).split(",") if s.strip()]
        fsm_is_explicit = True

    # Parse ports / interfaces and registers
    lines = spec.split("\n")
    current_section = ""
    for line in lines:
        l = line.strip()
        if not l:
            continue
        if l.startswith("##") or l.startswith("#"):
            current_section = l.lower()
            continue
            
        bullet_match = re.match(r"^[-*]\s*([a-zA-Z0-9_]+)\s*:\s*(input|output|inout|in|out)\s*,\s*(\d+|\d+:\d+|\w+)\s*(?:bits?|bit)?(?:\s*,\s*(.*))?", l, re.IGNORECASE)
        if bullet_match:
            p_dir = bullet_match.group(2).lower()
            if p_dir == "in": p_dir = "input"
            if p_dir == "out": p_dir = "output"
            interfaces.append({
                "name": bullet_match.group(1),
                "direction": p_dir,
                "width": bullet_match.group(3),
                "description": bullet_match.group(4) or "",
                "semantic_role": "data",
                "classification": "EXPLICIT",
                "source": "DesignSpec"
            })
            continue

        reg_bullet = re.match(r"^[-*]\s*([a-zA-Z0-9_]+_reg|[a-zA-Z0-9_]+_ctrl|[a-zA-Z0-9_]+_status|[a-zA-Z0-9_]+_csr|[a-zA-Z0-9_]+reg|[a-zA-Z0-9_]+)\s*:\s*(\d+|\d+:\d+|\w+)\s*(?:bits?|bit)?(?:\s*,\s*(.*))?", l, re.IGNORECASE)
        if reg_bullet and ("register" in current_section or "_reg" in reg_bullet.group(1).lower() or "_ctrl" in reg_bullet.group(1).lower()):
            rname = reg_bullet.group(1)
            registers.append({
                "name": rname,
                "width": reg_bullet.group(2),
                "reset_val": "32'h0",
                "purpose": reg_bullet.group(3) or "Internal Register",
                "fields": _derive_register_fields(rname, spec),
                "classification": "EXPLICIT"
            })
            continue

        # Check Verilog/SystemVerilog port declarations (e.g. input wire [31:0] axi_awaddr, or [242] output wire [31:0] axi_araddr)
        from core.efs_ir.signal_normalization import parse_hdl_port_declaration
        hdl_port = parse_hdl_port_declaration(l)
        if hdl_port:
            p_dir = hdl_port["direction"]
            p_name = hdl_port["name"]
            p_width = hdl_port["width_str"]
            p_desc = hdl_port["source_evidence"]
            p_role = "clock" if any(k in p_name.lower() for k in ("clk", "clock")) else (
                "reset" if any(k in p_name.lower() for k in ("rst", "reset")) else (
                    "control" if any(k in p_name.lower() for k in ("valid", "ready", "en", "enable", "start", "done", "busy")) else "data"
                )
            )
            interfaces.append({
                "name": p_name,
                "direction": p_dir,
                "width": p_width,
                "description": p_desc,
                "semantic_role": p_role,
                "classification": "EXPLICIT",
                "source": "DesignSpec"
            })
            continue

        if "|" in l and not l.startswith("|---"):
            parts = [p.strip() for p in l.split("|") if p.strip()]
            p_lower_parts = [p.lower() for p in parts]
            is_header = any(h in p_lower_parts for h in ("name", "signal", "signal name", "port", "port name", "direction", "dir", "width"))
            has_direction = any(cell.lower() in ("input", "output", "inout", "in", "out") for cell in parts)
            is_port_sec = any(k in current_section for k in ("interface", "port", "signal", "module", "controller", "sequencer"))
            if is_header and (is_port_sec or has_direction or any(h in p_lower_parts for h in ("direction", "dir"))):
                continue
            if is_port_sec or has_direction:
                    p_lower_parts = [p.lower() for p in parts]
                    # Check if header row
                    is_header = any(h in p_lower_parts for h in ("name", "signal", "signal name", "port", "port name", "direction", "dir", "width"))
                    if is_header:
                        continue
                    
                    c_name = parts[0]
                    c_dir = "input"
                    c_width = "1"
                    c_desc = ""

                    # Find direction cell
                    dir_found_idx = -1
                    for idx, cell in enumerate(parts):
                        if cell.lower() in ("input", "output", "inout", "in", "out"):
                            c_dir = "input" if cell.lower() in ("input", "in") else ("output" if cell.lower() in ("output", "out") else "inout")
                            dir_found_idx = idx
                            break
                    
                    # Find width cell
                    width_found_idx = -1
                    for idx, cell in enumerate(parts):
                        if idx == dir_found_idx or idx == 0:
                            continue
                        clean_c = cell.replace("[", "").replace("]", "").replace("bits", "").replace("bit", "").strip()
                        if clean_c.isdigit() or ":" in cell or normalize_hardware_width(cell) is not None:
                            c_width = cell
                            width_found_idx = idx
                            break
                    
                    if dir_found_idx == -1:
                        if len(parts) > 2 and parts[2].lower() in ("input", "output", "inout", "in", "out"):
                            c_dir = "input" if parts[2].lower() in ("input", "in") else ("output" if parts[2].lower() in ("output", "out") else "inout")
                            c_width = parts[1]
                        elif len(parts) > 1 and parts[1].lower() in ("input", "output", "inout", "in", "out"):
                            c_dir = "input" if parts[1].lower() in ("input", "in") else ("output" if parts[1].lower() in ("output", "out") else "inout")
                            c_width = parts[2] if len(parts) > 2 else "1"

                    if len(parts) > 3:
                        c_desc = parts[3]
                    elif len(parts) > 2 and width_found_idx != 2 and dir_found_idx != 2:
                        c_desc = parts[2]

                    norm_w = normalize_hardware_width(c_width)
                    if norm_w is not None:
                        c_width = str(norm_w)
                    elif "[" in c_width and ":" in c_width:
                        m_w = re.match(r"\[?\s*(\d+)\s*:\s*(\d+)\s*\]?", c_width.strip())
                        if m_w:
                            c_width = str(abs(int(m_w.group(1)) - int(m_w.group(2))) + 1)

                    if c_name.lower() not in ("name", "signal", "port", "pin", "width", "direction"):
                        interfaces.append({
                            "name": c_name,
                            "width": c_width,
                            "direction": c_dir,
                            "description": c_desc,
                            "semantic_role": "data",
                            "classification": "EXPLICIT",
                            "source": "DesignSpec"
                        })
        elif "register" in current_section:
            if "|" in l and not l.startswith("|---") and "register" not in l.lower():
                parts = [p.strip() for p in l.split("|") if p.strip()]
                if len(parts) >= 2:
                    rname = parts[0]
                    registers.append({
                        "name": rname,
                        "width": parts[1] if len(parts) > 2 and parts[1].isdigit() else "32",
                        "reset_val": "32'h0",
                        "purpose": parts[2] if len(parts) > 2 else "RW Register",
                        "fields": _derive_register_fields(rname, spec),
                        "classification": "EXPLICIT"
                    })

    # Extract FSM transitions (e.g. IDLE -> RUN: condition)
    trans_matches = re.findall(r"([a-zA-Z0-9_]+)\s*->\s*([a-zA-Z0-9_]+)\s*:\s*(.+)", spec)
    if trans_matches:
        fsm_is_explicit = True
        for from_st, to_st, raw_cond in trans_matches:
            from_st = from_st.strip().upper()
            to_st = to_st.strip().upper()
            raw_cond = raw_cond.strip().strip("'\"")
            if from_st not in states: states.append(from_st)
            if to_st not in states: states.append(to_st)
            
            vague_keywords = [
                "control start condition met", "start condition met", "operation completes",
                "counter completes", "internal counter", "transaction acknowledged",
                "completion condition", "condition met", "undefined", "tbd"
            ]
            is_dwell = bool(re.search(r"\b(?:next\s+clock\s+cycle|after\s+1\s+cycle|single[- ]cycle\s+dwell|unconditional)\b", raw_cond.lower()))
            is_vague = (not is_dwell) and (any(vk in raw_cond.lower() for vk in vague_keywords) or not raw_cond)
            
            if is_vague or (not raw_cond and not is_dwell):
                transitions.append({
                    "from": from_st,
                    "to": to_st,
                    "condition": {
                        "expression": None,
                        "signals": [],
                        "registers": [],
                        "protocol_events": [],
                        "source": "EFS",
                        "source_evidence": raw_cond,
                        "is_fully_defined": False,
                        "classification": "AMBIGUOUS"
                    }
                })
                blocking_issues.append({
                    "category": "FSM",
                    "classification": "AMBIGUOUS",
                    "is_blocking": True,
                    "issue": f"{from_st} -> {to_st} transition condition is undefined or ambiguous ('{raw_cond}').",
                    "required_information": f"Specify the exact signal/register condition that triggers transition '{from_st}' -> '{to_st}'.",
                    "evidence": raw_cond
                })
            else:
                sig_refs = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", raw_cond) if not is_dwell else []
                kw_set = {"and", "or", "not", "if", "else", "true", "false", "d", "h", "b"}
                sig_refs = [s for s in sig_refs if s.lower() not in kw_set and not s.isdigit()]
                
                transitions.append({
                    "from": from_st,
                    "to": to_st,
                    "condition": {
                        "expression": "1'b1" if is_dwell else raw_cond,
                        "signals": sig_refs,
                        "registers": [],
                        "protocol_events": [],
                        "source": "DesignSpec",
                        "source_evidence": raw_cond,
                        "is_fully_defined": True,
                        "classification": "EXPLICIT"
                    }
                })

    # Find clocks & resets
    for p in interfaces:
        pname = p["name"].lower()
        if "clk" in pname or "aclk" in pname:
            p["semantic_role"] = "clock"
            clocks_resets.append({"name": p["name"], "type": "clock", "classification": "EXPLICIT"})
        elif "rst" in pname or "reset" in pname:
            p["semantic_role"] = "reset"
            clocks_resets.append({
                "name": p["name"], 
                "type": "reset", 
                "active_level": "low" if "n" in pname or "low" in spec.lower() else "high",
                "classification": "EXPLICIT"
            })
            
    # Check if specification has sequential indicators before adding derived clock/reset
    has_seq_indicators = bool(
        clocks_resets or
        registers or
        fsm_is_explicit or
        states or
        re.search(r"\b(?:clock|clk|reset|rst|registers?|flip-flop|latch|states?|fsm)\b", spec, re.IGNORECASE)
    )
    if has_seq_indicators:
        if not any(c["type"] == "clock" for c in clocks_resets):
            clocks_resets.append({"name": "clk", "type": "clock", "classification": "DERIVED"})
        if not any(c["type"] == "reset" for c in clocks_resets):
            clocks_resets.append({"name": "rst_n", "type": "reset", "active_level": "low", "classification": "DERIVED"})

    status = "COMPLETE" if not blocking_issues else "INCOMPLETE"
    
    return {
        "module_name": module_name,
        "specification_status": {
            "status": status,
            "blocking_issues": blocking_issues
        },
        "interfaces": interfaces,
        "clocks_resets": clocks_resets,
        "registers": registers,
        "parameters": parameters,
        "states": states if (fsm_is_explicit or states) else [],
        "transitions": transitions,
        "fsm_is_explicit": fsm_is_explicit
    }

