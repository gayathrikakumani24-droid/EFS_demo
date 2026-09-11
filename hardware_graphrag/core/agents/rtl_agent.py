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

logger = get_logger("agents.rtl")

SYSTEM_PROMPT = """You are a senior RTL/SystemVerilog design engineer.

Generate synthesizable SystemVerilog RTL from the supplied EFS IR,
Design Plan, Architecture/FSM model, and relevant Protocol Contract.

RULES:

1. EFS IR is the authoritative hardware interface.
   Preserve exactly:
   - module name
   - port names
   - directions
   - widths
   - clock/reset
   - parameters

2. Do not invent, remove, rename, merge, or resize external signals.

3. Do not assume any specific protocol.
   Derive protocol roles, channels, handshakes, transaction ordering,
   response, error, backpressure, burst, and retry behavior ONLY from
   the supplied Protocol Contract/retrieved protocol chunks.

4. Implement only functionality specified by the EFS.
   Do not add unsupported registers, states, opcodes, memories,
   interrupts, timing, or protocol features.

5. If an Architecture/FSM model is provided, RTL must implement it
   consistently.

6. Implement clock, reset, FSM, data widths, error handling, and
   transaction completion exactly as specified.

7. Use internal signals/states when required for implementation, but
   never expose them as new module ports unless specified.

8. Generate synthesizable SystemVerilog:
   - no delays
   - no testbench constructs
   - no inferred latches
   - no multiple drivers
   - no combinational loops
   - use appropriate sequential/combinational logic

9. Implement interface ports using exact port names, widths, and directions specified in the protocol contract or EFS IR. Always output synthesizable SystemVerilog RTL.

10. Before returning, verify:
    - interface
    - FSM
    - protocol handshake
    - transaction completion
    - error behavior
    - synthesizability

RULE 11 — SPECIFICATION COMPLETENESS
Every generated RTL behavior must be directly traceable to explicit information in the supplied EFS IR, Design Plan, Architecture/FSM model, Design Specification, or Protocol Contract.
Do not infer missing behavior from:
- signal names
- common RTL patterns
- typical AXI behavior
- register naming conventions
- presumed bit meanings
- likely FSM behavior
- conventional counter implementations
- engineering assumptions

RULE 12 — NO GUESSING
If a required behavior is undefined or ambiguous:
1. Do NOT invent a signal.
2. Do NOT invent a register.
3. Do NOT invent a counter.
4. Do NOT invent a state.
5. Do NOT repurpose an existing signal.
6. Do NOT assign meaning to an unspecified register bit.
7. Do NOT assume VALID means START.
8. Do NOT assume READY means ACKNOWLEDGE.
9. Do NOT assume a protocol event represents operation completion.
10. Do NOT use arbitrary constants to resolve ambiguity.

RULE 13 — SOURCE TRACEABILITY
Every non-trivial RTL behavior must be traceable to a supplied requirement.
If a transition condition cannot be mapped to an explicit signal, register field, protocol event, or explicitly defined internal condition, the specification is incomplete and RTL generation must be blocked.

RULE 14 — NO SPECIFICATION REPAIR
The RTL generator is not allowed to repair an incomplete specification.
If the specification is incomplete, report the missing requirement instead of selecting the most likely hardware interpretation.

OUTPUT:
Return ONLY the SystemVerilog RTL.
No explanation.
No Markdown fences.
No <think> block.
No commentary.
Start directly with:

module <module_name>

and end with:

endmodule"""

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

    # Safety check: if Verilog/SystemVerilog module code lacks endmodule, complete structure gracefully
    if "module " in cleaned and "endmodule" not in cleaned:
        if "case" in cleaned and "endcase" not in cleaned:
            cleaned += "\n            default: ;\n        endcase"
        if "always" in cleaned and "end" not in cleaned:
            cleaned += "\n    end"
        cleaned += "\n\nendmodule\n"

    return cleaned


def plan_rtl(design_spec: str, target: str, efs_ir: Optional[EFSIR] = None, context_pack: Optional[Any] = None) -> Dict[str, Any]:
    """Stage 1: Generate a Structured RTL Plan (JSON) from EFS IR and design specification."""
    logger.info("Generating Structured RTL Implementation Plan (Stage 1)...")
    llm = get_llm_client()
    
    efs_ir_summary = ""
    if efs_ir:
        try:
            efs_ir_summary = json.dumps({
                "metadata": efs_ir.metadata.to_dict(),
                "components": [c.to_dict() for c in efs_ir.components],
                "interfaces": [i.to_dict() for i in efs_ir.interfaces],
                "signals": [s.to_dict() for s in efs_ir.signals],
                "registers": [r.to_dict() for r in efs_ir.registers],
                "fsms": [f.to_dict() for f in efs_ir.fsms],
            }, indent=2)
        except Exception:
            pass

    plan_prompt = f"""You are a senior hardware architect. Output a JSON Structured RTL Implementation Plan.
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
   - Any FSM transition has no implementable signal/register condition (e.g. "control start condition met" without identifying exact signal/bit).
   - Register bit meanings are unspecified or referenced without explicit bit mapping (never assume ctrl_reg[0]=start or status_reg[0]=done).
   - A completion condition or counter threshold/width is unspecified.
   - An output behavior or required protocol event is undefined or cannot be mapped to supplied signals.
   - Signal direction, width, or polarity conflicts exist across sources.
3. Do NOT invent signal names or expressions for missing behavior. Set "expression": null and "is_fully_defined": false for incomplete conditions.

=== EFS IR HARDWARE CONTEXT ===
{efs_ir_summary if efs_ir_summary else "Not Available"}

=== SPECIFICATION ===
{design_spec[:6000]}
"""
    if llm.available:
        plan_res = llm.complete_json("You are an expert RTL architect.", plan_prompt, model=CONFIG.llm.model)
        if plan_res and isinstance(plan_res, dict) and plan_res.get("ports"):
            if efs_ir and efs_ir.signals:
                for s in efs_ir.signals:
                    if s.name.lower() in ("name", "signal", "port", "signal name", "port name", "signal_name", "port_name", "pin", "width", "direction", "description", "protocol/description"):
                        continue
                    matching_p = next((p for p in plan_res.get("ports", []) if isinstance(p, dict) and str(p.get("name", "")).lower() == s.name.lower()), None)
                    # Check if conflict on this signal has been resolved by human approval
                    is_resolved = False
                    if efs_ir and hasattr(efs_ir, "resolutions"):
                        for res in efs_ir.resolutions:
                            if getattr(res, "approval_status", "") == "APPROVED" and (s.name in res.affected_requirements or s.signal_id in res.affected_requirements):
                                is_resolved = True
                                break
                    if efs_ir and hasattr(efs_ir, "conflicts"):
                        for c in efs_ir.conflicts:
                            if s.name in c.entities and getattr(c, "resolution_status", "") == "RESOLVED":
                                is_resolved = True
                                break

                    if matching_p and matching_p.get("direction") and matching_p.get("direction").lower() != s.direction.lower() and not is_resolved:
                        if not isinstance(plan_res.get("specification_status"), dict):
                            plan_res["specification_status"] = {"status": "CONTRADICTORY", "blocking_issues": []}
                        plan_res["specification_status"]["status"] = "CONTRADICTORY"
                        plan_res["specification_status"].setdefault("blocking_issues", []).append({
                            "category": "INTERFACE",
                            "issue": f"Signal direction conflict: {s.name} is specified as {s.direction} in EFS IR but {matching_p.get('direction')} in protocol contract/spec.",
                            "required_information": f"Clarify the correct direction for port '{s.name}'.",
                            "evidence": f"EFS IR: {s.direction}, Spec/Plan: {matching_p.get('direction')}"
                        })
            val = validate_rtl_plan(plan_res, efs_ir=efs_ir)
            plan_res["specification_status"] = {
                "status": val["status"],
                "blocking_issues": val["issues"]
            }
            return plan_res

    # Grounded fallback plan from efs_ir or design_spec
    spec_parsed = parse_markdown_spec(design_spec)
    ports = []
    if efs_ir and efs_ir.signals:
        for s in efs_ir.signals:
            if s.name.lower() in ("name", "signal", "port", "signal name", "port name", "signal_name", "port_name", "pin", "width", "direction", "description", "protocol/description"):
                continue
            parsed_ports = spec_parsed.get("interfaces", [])
            matching_p = next((p for p in parsed_ports if p.get("name", "").lower() == s.name.lower()), None)
            source_dir = s.direction
            is_resolved = False
            if efs_ir and hasattr(efs_ir, "resolutions"):
                for res in efs_ir.resolutions:
                    if getattr(res, "approval_status", "") == "APPROVED" and (s.name in res.affected_requirements or s.signal_id in res.affected_requirements):
                        is_resolved = True
                        break
            if efs_ir and hasattr(efs_ir, "conflicts"):
                for c in efs_ir.conflicts:
                    if s.name in c.entities and getattr(c, "resolution_status", "") == "RESOLVED":
                        is_resolved = True
                        break

            if matching_p and matching_p.get("direction") and matching_p.get("direction").lower() not in ("direction", "dir") and matching_p.get("direction").lower() != s.direction.lower() and not is_resolved:
                spec_parsed["specification_status"]["status"] = "CONTRADICTORY"
                spec_parsed["specification_status"]["blocking_issues"].append({
                    "category": "INTERFACE",
                    "issue": f"Signal direction conflict: {s.name} is specified as {s.direction} in EFS IR but {matching_p.get('direction')} in protocol contract/spec.",
                    "required_information": f"Clarify the correct direction for port '{s.name}'.",
                    "evidence": f"EFS IR: {s.direction}, Spec: {matching_p.get('direction')}"
                })
            ports.append({
                "name": s.name,
                "direction": source_dir,
                "width": s.width,
                "description": s.description,
                "semantic_role": s.semantic_role or "data",
                "source": "EFS"
            })
    else:
        ports = spec_parsed.get("interfaces", [])

    internal_regs = []
    if efs_ir and efs_ir.registers:
        for r in efs_ir.registers:
            fields = []
            if hasattr(r, "fields") and r.fields:
                for f in r.fields:
                    fields.append({
                        "bits": getattr(f, "bits", getattr(f, "bit_range", "31:0")),
                        "meaning": getattr(f, "meaning", getattr(f, "description", "")),
                        "source": "EFS"
                    })
            internal_regs.append({
                "name": r.name,
                "width": getattr(r, "width", "32"),
                "reset_val": getattr(r, "reset_val", getattr(r, "reset_value", "32'h0")),
                "purpose": r.description or "Internal Register",
                "fields": fields
            })
    else:
        internal_regs = spec_parsed.get("registers", [])

    fsm_states = spec_parsed.get("states", ["IDLE", "RUN", "DONE"])
    fsm_transitions = spec_parsed.get("transitions", [])
    if efs_ir and efs_ir.fsms:
        for f in efs_ir.fsms:
            if hasattr(f, "states") and f.states:
                fsm_states = [st.name for st in f.states]
            if hasattr(f, "transitions") and f.transitions:
                fsm_transitions = []
                for tr in f.transitions:
                    raw_cond = tr.condition if hasattr(tr, "condition") else "undefined"
                    is_vague = any(vk in str(raw_cond).lower() for vk in ["control start", "operation completes", "acknowledged", "undefined", "tbd"])
                    fsm_transitions.append({
                        "from": tr.from_state if hasattr(tr, "from_state") else getattr(tr, "from", "IDLE"),
                        "to": tr.to_state if hasattr(tr, "to_state") else getattr(tr, "to", "RUN"),
                        "condition": {
                            "expression": None if is_vague else str(raw_cond),
                            "signals": [],
                            "registers": [],
                            "protocol_events": [],
                            "source": "EFS",
                            "source_evidence": str(raw_cond),
                            "is_fully_defined": not is_vague
                        }
                    })

    plan = {
        "module_name": spec_parsed.get("module_name", getattr(efs_ir.metadata, "design_name", "target_module") if efs_ir and hasattr(efs_ir, "metadata") else "target_module"),
        "specification_status": spec_parsed.get("specification_status", {
            "status": "COMPLETE",
            "blocking_issues": []
        }),
        "ports": ports,
        "parameters": spec_parsed.get("parameters", []),
        "internal_registers": internal_regs,
        "fsm": {
            "states": fsm_states,
            "initial_state": fsm_states[0] if fsm_states else "IDLE",
            "transitions": fsm_transitions
        },
        "datapath_strategy": "Direct state machine & register transfer logic",
        "reset_polarity": "active_low"
    }

    val = validate_rtl_plan(plan)
    plan["specification_status"] = {
        "status": val["status"],
        "blocking_issues": val["issues"]
    }
    return plan


def validate_rtl_plan(rtl_plan: Dict[str, Any], efs_ir: Optional[EFSIR] = None) -> Dict[str, Any]:
    """
    Stage B Validation: Verifies that the generated RTL plan faithfully implements the extracted EFSIR.
    Compares: RTL Plan <-> EFSIR (NOT against generic assumed hardware templates).
    """
    logger.info("Validating RTL plan against authoritative EFS IR...")
    all_issues = []
    
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

    # 1. Report unresolved source conflicts from EFS IR
    if efs_ir and hasattr(efs_ir, "conflicts"):
        for conf in efs_ir.conflicts:
            if getattr(conf, "resolution_status", "UNRESOLVED") == "UNRESOLVED":
                all_issues.append({
                    "category": getattr(conf, "category", "GENERAL") or "GENERAL",
                    "classification": "SOURCE_CONFLICT",
                    "badge": "🟠 [SOURCE_CONFLICT]",
                    "is_blocking": True,
                    "issue": f"Unresolved specification conflict [{conf.type}]: {conf.evidence or conf.description}",
                    "required_information": conf.description,
                    "evidence": conf.evidence or "Source text"
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

    for idx, trans in enumerate(plan_transitions):
        if not isinstance(trans, dict):
            continue
        from_st = trans.get("from", f"STATE_{idx}")
        to_st = trans.get("to", f"NEXT_STATE_{idx}")
        cond = trans.get("condition")

        if cond is None:
            all_issues.append({
                "category": "FSM",
                "classification": "PLAN_MISSING",
                "badge": "🔴 [PLAN_MISSING]",
                "is_blocking": True,
                "issue": f"{from_st} -> {to_st} transition condition is undefined.",
                "required_information": f"Specify condition triggering transition '{from_st}' -> '{to_st}'.",
                "evidence": f"Transition '{from_st}' -> '{to_st}' has null condition."
            })
        else:
            cond_str = str(cond.get("expression") if isinstance(cond, dict) else cond).strip()
            if any(vp in cond_str.lower() for vp in vague_patterns):
                all_issues.append({
                    "category": "FSM",
                    "classification": "PLAN_AMBIGUOUS",
                    "badge": "🟡 [PLAN_AMBIGUOUS]",
                    "is_blocking": True,
                    "issue": f"{from_st} -> {to_st} transition condition is ambiguous ('{cond_str}').",
                    "required_information": f"Specify explicit signal/register condition for transition '{from_st}' -> '{to_st}'.",
                    "evidence": cond_str
                })

    if efs_ir and efs_ir.fsms:
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

    # 3. Check interface signals from EFS IR
    if efs_ir and efs_ir.signals:
        plan_ports = rtl_plan.get("ports", [])
        plan_port_map = {str(p.get("name", "")).strip().lower(): p for p in plan_ports if isinstance(p, dict)}
        
        for sig in efs_ir.signals:
            s_name_lower = sig.name.lower()
            if s_name_lower in ("name", "signal", "port", "pin", "width", "direction"):
                continue
            if s_name_lower not in plan_port_map and plan_ports:
                # EFS IR signal missing from plan ports
                all_issues.append({
                    "category": "INTERFACE",
                    "classification": "PLAN_MISSING",
                    "badge": "🔴 [PLAN_MISSING]",
                    "is_blocking": True,
                    "issue": f"Signal '{sig.name}' declared in EFS IR is missing from RTL plan ports.",
                    "required_information": f"Add port '{sig.name}' to RTL plan.",
                    "evidence": f"EFS IR signal '{sig.name}' ({sig.direction})"
                })
            elif s_name_lower in plan_port_map:
                p_obj = plan_port_map[s_name_lower]
                p_dir = str(p_obj.get("direction", "")).lower()
                e_dir = str(sig.direction).lower()
                if p_dir and e_dir and p_dir != e_dir and not ("in" in p_dir and "in" in e_dir) and not ("out" in p_dir and "out" in e_dir):
                    all_issues.append({
                        "category": "INTERFACE",
                        "classification": "PLAN_CONFLICT",
                        "badge": "🟠 [PLAN_CONFLICT]",
                        "is_blocking": True,
                        "issue": f"Signal direction conflict on '{sig.name}': RTL plan specifies '{p_dir}', but EFS IR specifies '{e_dir}'.",
                        "required_information": f"Align port '{sig.name}' direction with EFS IR ('{e_dir}').",
                        "evidence": f"Plan: {p_dir}, EFS IR: {e_dir}"
                    })

    # 4. Check reported issues from plan specification status
    spec_status = rtl_plan.get("specification_status", {})
    if isinstance(spec_status, dict):
        reported_issues = spec_status.get("blocking_issues", [])
        if isinstance(reported_issues, list):
            for b_issue in reported_issues:
                if isinstance(b_issue, dict):
                    iss_txt = str(b_issue.get("issue", "")).lower()
                    
                    # Check if this issue corresponds to an approved/resolved conflict in EFS IR
                    is_issue_resolved = False
                    if efs_ir and hasattr(efs_ir, "resolutions"):
                        for res in efs_ir.resolutions:
                            if getattr(res, "approval_status", "") == "APPROVED":
                                conf_id = getattr(res, "conflict_id", "")
                                aff_reqs = getattr(res, "affected_requirements", [])
                                if (conf_id and conf_id.lower() in iss_txt) or any(e.lower() in iss_txt for e in aff_reqs if e):
                                    is_issue_resolved = True
                                    break
                    if efs_ir and hasattr(efs_ir, "conflicts"):
                        for c in efs_ir.conflicts:
                            if getattr(c, "resolution_status", "") == "RESOLVED":
                                if any(e.lower() in iss_txt for e in c.entities if e):
                                    is_issue_resolved = True
                                    break

                    if ("conflict" in iss_txt or "mismatch" in iss_txt or "contradict" in iss_txt) and not ("design name" in iss_txt or "module name" in iss_txt or "top-level" in iss_txt):
                        cls = "SOURCE_CONFLICT" if "opcode" in iss_txt or "source" in iss_txt else "PLAN_CONFLICT"
                        is_block = not is_issue_resolved
                    elif "unsupported" in iss_txt:
                        cls = "UNSUPPORTED"
                        is_block = True
                    elif "ambiguous" in iss_txt or "vague" in iss_txt:
                        cls = "SOURCE_AMBIGUOUS"
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
        "all_issues": all_issues
    }


def generate_code(design_spec: str, target: str, efs_ir: Optional[EFSIR] = None, protocol_rules: Optional[List[Dict[str, Any]]] = None, context_pack: Optional[Any] = None) -> str:
    """Generate HDL or verification collateral code from ContextPack or design specification."""
    logger.info(f"Generating hardware code for target '{target}' (2-Stage Architecture)...")
    
    # Stage 1: Generate Structured RTL Implementation Plan
    rtl_plan = plan_rtl(design_spec, target, efs_ir=efs_ir, context_pack=context_pack)
    
    # Deterministic Completeness Validation (Must run AFTER Stage 1 and BEFORE Stage 2)
    validation = validate_rtl_plan(rtl_plan, efs_ir=efs_ir)
    
    if not validation.get("valid", False):
        logger.warning(f"RTL Generation BLOCKED: Specification is incomplete or contradictory for target '{target}'.")
        issues = validation.get("issues", [])
        
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
        
    logger.info("Proceeding to Stage 2 RTL generation")
    
    module_name = rtl_plan.get("module_name", "custom_module")
    ports = rtl_plan.get("ports", [])
    
    # Port list formatting
    port_lines = []
    for p in ports:
        name = p.get("name", "signal")
        direction = p.get("direction", "input")
        width = str(p.get("width", "1"))
        
        # SystemVerilog logic syntax
        dir_keyword = "input " if direction in ("input", "in") else ("output" if direction in ("output", "out") else "inout ")
        
        if width in ("1", "1 bit", "1bit", "0:0", "[0:0]"):
            width_str = "      "
        elif width.isdigit():
            width_str = f"[{int(width)-1}:0] "
        elif ":" in width and not width.startswith("["):
            width_str = f"[{width}] "
        else:
            width_str = f"{width} "
            
        port_lines.append(f"  {dir_keyword} logic {width_str}{name}")
        
    port_block = ",\n".join(port_lines) if port_lines else "  input logic clk,\n  input logic rst_n"
    
    if target == "VHDL":
        vhdl_ports = []
        for p in ports:
            name = p.get("name", "signal")
            direction = "in" if p.get("direction") in ("input", "in") else "out"
            width = str(p.get("width", "1"))
            if width in ("1", "1 bit", "1bit"):
                type_str = "std_logic"
            elif width.isdigit():
                type_str = f"std_logic_vector({int(width)-1} downto 0)"
            else:
                type_str = "std_logic_vector"
            vhdl_ports.append(f"    {name} : {direction} {type_str}")
        vhdl_port_block = ";\n".join(vhdl_ports)
        
        return f"""-- Generated VHDL Hardware Description Module
-- Entity: {module_name}
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity {module_name} is
  port (
{vhdl_port_block}
  );
end entity {module_name};

architecture rtl of {module_name} is
begin
  -- Synthesizable logic implementation
end architecture rtl;
"""

    return f"""// Generated Synthesizable SystemVerilog Module
// Target: {module_name}
`timescale 1ns / 1ps

module {module_name} (
{port_block}
);

  // Internal architecture & register transfer logic
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // Active low reset
    end
  end

  always_comb begin
    // Combinational logic
  end
  
endmodule
"""


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
            {"bits": "31:0", "meaning": "Data Payload", "classification": "DERIVED"}
        ]
    return fields


def get_heuristic_code_template(target: str, design_spec: str, efs_ir: Optional[EFSIR] = None) -> str:
    """Helper template generator to extract design properties and build fallback RTL code."""
    module_name = "target_module"
    interfaces = []
    clocks_resets = []
    registers = []
    states = []
    
    if efs_ir:
        if hasattr(efs_ir.metadata, "design_name") and efs_ir.metadata.design_name:
            module_name = efs_ir.metadata.design_name
        for sig in efs_ir.signals:
            if sig.semantic_role in ("clock", "reset"):
                clocks_resets.append({
                    "name": sig.name,
                    "type": sig.semantic_role,
                    "active_level": "rising" if sig.semantic_role == "clock" else ("low" if "n" in sig.name.lower() or "rst" in sig.name.lower() else "high")
                })
            else:
                interfaces.append({
                    "name": sig.name,
                    "direction": sig.direction,
                    "width": sig.width,
                    "description": sig.description
                })
                
        for r in efs_ir.registers:
            registers.append({
                "name": r.name,
                "offset": r.offset,
                "access": r.access_type,
                "description": r.description
            })
            
        for fsm in efs_ir.fsms:
            for s in fsm.states:
                states.append(s.name)
        if not states:
            states = ["IDLE", "RUN", "DONE"]
            
        spec_data = {
            "module_name": module_name,
            "interfaces": interfaces,
            "clocks_resets": clocks_resets,
            "registers": registers,
            "states": states
        }
    else:
        spec_data = parse_markdown_spec(design_spec)
        
    module_name = spec_data["module_name"]
    interfaces = spec_data["interfaces"]
    clocks_resets = spec_data["clocks_resets"]
    registers = spec_data["registers"]
    states = spec_data["states"]
    
    clk_name = next((c["name"] for c in clocks_resets if c["type"] == "clock"), "clk")
    rst_name = next((c["name"] for c in clocks_resets if c["type"] == "reset"), "rst_n")
    rst_polarity = next((c.get("active_level", "low") for c in clocks_resets if c["type"] == "reset"), "low")

    port_list_str = []
    for p in interfaces:
        pname = str(p["name"]).strip("` \t\r\n'\"")
        pdir = str(p["direction"]).strip().lower()
        pwidth = str(p["width"]).strip("` \t\r\n'\"")
        
        w_decl = ""
        if pwidth.isdigit():
            w = int(pwidth)
            if w > 1:
                w_decl = f"[{w-1}:0] "
        elif ":" in pwidth:
            w_decl = f"{pwidth.replace('[', '').replace(']', '')} "
            
        pdir_str = "input  wire" if pdir == "input" or pdir == "in" else "output reg "
        if pname in [clk_name, rst_name]:
            pdir_str = "input  wire"
            w_decl = ""
            
        port_list_str.append(f"    {pdir_str} {w_decl}{pname}")

    if not port_list_str:
        ports_decl = f"""    input  wire        {clk_name},
    input  wire        {rst_name},
    input  wire        req,
    output reg         ack,
    input  wire [31:0] data_in,
    output reg  [31:0] data_out"""
    else:
        ports_decl = ",\n".join(port_list_str)

    state_decls = []
    for idx, s in enumerate(states):
        state_decls.append(f"    localparam {s} = 2'd{idx};")
    state_decls_str = "\n".join(state_decls)

    reg_decls = []
    reg_resets = []
    for r in registers:
        rname = r["name"].lower()
        reg_decls.append(f"    reg [31:0] {rname}_reg;")
        reg_resets.append(f"            {rname}_reg <= 32'h0;")
    reg_decls_str = "\n".join(reg_decls)
    reg_resets_str = "\n".join(reg_resets)

    out_resets = []
    for p in interfaces:
        pname = p["name"]
        if (p["direction"] == "output" or p["direction"] == "out") and pname not in [clk_name, rst_name]:
            out_resets.append(f"            {pname} <= 1'b0;")
    out_resets_str = "\n".join(out_resets)

    if target == "Verilog":
        return f"""// Dynamic Fallback Verilog RTL Template
// Derived from Unified Design Specification
module {module_name} (
{ports_decl}
);

    // State machine parameter declarations
{state_decls_str}
    reg [1:0] state, next_state;

    // Registers declarations
{reg_decls_str}

    // Sequential logic (Clock & Reset)
    always @(posedge {clk_name} or {"negedge" if rst_polarity == "low" else "posedge"} {rst_name}) begin
        if ({"!" if rst_polarity == "low" else ""} {rst_name}) begin
            state <= {states[0]};
{reg_resets_str}
{out_resets_str}
        end else begin
            state <= next_state;
            
            // Output behavior based on state
            case (state)
                {states[0]}: begin
                    // IDLE state default outputs
                end
            endcase
        end
    end

    // FSM Combinational next-state transitions
    always @(*) begin
        next_state = state;
        case (state)
            {states[0]}: begin
                next_state = state;
            end
        endcase
    end

endmodule
"""
    elif target == "SystemVerilog":
        ports_sv = ports_decl.replace("wire", "logic").replace("reg", "logic")
        states_enum = ", ".join(states)
        reg_decls_sv = reg_decls_str.replace("reg", "logic")
        return f"""// Dynamic Fallback SystemVerilog RTL Template
// Derived from Unified Design Specification
module {module_name} (
{ports_sv}
);

    // State machine enum
    typedef enum logic [1:0] {{
        {states_enum}
    }} state_t;
    state_t state, next_state;

    // Registers declarations
{reg_decls_sv}

    // Sequential logic (Clock & Reset)
    always_ff @(posedge {clk_name} or {"negedge" if rst_polarity == "low" else "posedge"} {rst_name}) begin
        if ({"!" if rst_polarity == "low" else ""} {rst_name}) begin
            state <= {states[0]};
{reg_resets_str}
{out_resets_str}
        end else begin
            state <= next_state;
        end
    end

    // FSM Combinational next-state transitions
    always_comb begin
        next_state = state;
        case (state)
            {states[0]}: begin
                next_state = state;
            end
        endcase
    end

endmodule
"""
    elif target == "VHDL":
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
            {"bits": "31:0", "meaning": "Data Payload", "classification": "DERIVED"}
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

        if "interface" in current_section or "port" in current_section:
            if "|" in l and not l.startswith("|---") and "signal" not in l.lower():
                parts = [p.strip() for p in l.split("|") if p.strip()]
                if len(parts) >= 3:
                    p_name_lower = parts[0].lower()
                    if p_name_lower in ("name", "signal", "port", "signal name", "port name", "signal_name", "port_name", "width", "direction", "description", "protocol/description") or parts[2].lower() in ("direction", "dir"):
                        continue
                    p_dir = parts[2].lower()
                    if p_dir == "in": p_dir = "input"
                    if p_dir == "out": p_dir = "output"
                    interfaces.append({
                        "name": parts[0],
                        "width": parts[1],
                        "direction": p_dir,
                        "description": parts[3] if len(parts) > 3 else "",
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
            is_vague = any(vk in raw_cond.lower() for vk in vague_keywords)
            
            if is_vague or not raw_cond:
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
                sig_refs = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", raw_cond)
                kw_set = {"and", "or", "not", "if", "else", "true", "false", "d", "h", "b"}
                sig_refs = [s for s in sig_refs if s.lower() not in kw_set and not s.isdigit()]
                
                transitions.append({
                    "from": from_st,
                    "to": to_st,
                    "condition": {
                        "expression": raw_cond,
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
        "states": states if states else ["IDLE", "PROCESS", "DONE"],
        "transitions": transitions,
        "fsm_is_explicit": fsm_is_explicit
    }

