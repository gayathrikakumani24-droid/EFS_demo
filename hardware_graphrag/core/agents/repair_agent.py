"""
Repair Agent.

Analyzes verification failures and applies surgical fixes to the generated code
or design specification, explaining the modification.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from config import CONFIG
from core.extraction.llm_client import get_llm_client
from utils.logger import get_logger
from core.efs_ir.models import EFSIR

logger = get_logger("agents.repair")

SYSTEM_PROMPT = """You are a Principal RTL Repair Engineer.
Your task is to repair the provided hardware code to resolve specific violations reported by the verification agent.

Before repairing, check the provided CANONICAL HARDWARE IR (EFS IR) and DESIGN SPECIFICATION.
Decide: Is the EFS IR correct? If the specification itself contains a conflict or is incorrect/incomplete (e.g. duplicate opcodes, width mismatch), set `efs_ir_correct` to `false` and explain the spec issue in `efs_ir_repair_notes`. Do not attempt to repair the RTL if the underlying EFS IR / Spec is incorrect.
If the specification and EFS IR are correct, set `efs_ir_correct` to `true` and apply surgical edits only to the RTL sections affected by the violations. Do not rewrite the entire design.

You must return a valid JSON object matching the following structure:
{
  "efs_ir_correct": true|false,
  "efs_ir_repair_notes": "Explanation of any specification conflict or EFS IR model correctness issues if efs_ir_correct is false",
  "repaired_code": "The full updated code containing only the corrected blocks",
  "repairs": [
    {
      "check_name": "string",
      "what_changed": "Summary of edit",
      "why_changed": "Technical justification",
      "protocol_rule_or_spec": "The corresponding protocol rule or spec section"
    }
  ]
}
Do not add any markdown formatting or explanation outside the JSON object. Keep the output strictly conforming to the JSON schema.
"""


def repair_code(
    generated_code: str,
    design_spec: str,
    protocol_rules: List[Dict[str, Any]],
    compliance_report: Dict[str, Any],
    efs_ir: Optional[EFSIR] = None
) -> Dict[str, Any]:
    """Repair generated code to fix compliance violations and return the repaired code and explanations."""
    logger.info("Repairing code violations...")
    llm = get_llm_client()

    # Trace failed checks back to EFS IR and original source specifications
    tracing_info = []
    if efs_ir and "failed_checks" in compliance_report:
        for failure in compliance_report["failed_checks"]:
            efs_id = failure.get("efs_id")
            if not efs_id:
                continue
            
            found_obj = None
            obj_type = "Unknown"
            
            # Find the object in EFS IR
            for comp in efs_ir.components:
                if comp.component_id == efs_id:
                    found_obj, obj_type = comp, "Component"
            for iface in efs_ir.interfaces:
                if iface.interface_id == efs_id:
                    found_obj, obj_type = iface, "Interface"
            for sig in efs_ir.signals:
                if sig.signal_id == efs_id:
                    found_obj, obj_type = sig, "Signal"
            for reg in efs_ir.registers:
                if reg.register_id == efs_id:
                    found_obj, obj_type = reg, "Register"
            for fsm in efs_ir.fsms:
                if fsm.fsm_id == efs_id:
                    found_obj, obj_type = fsm, "FSM"
            for const in efs_ir.constraints:
                if const.constraint_id == efs_id:
                    found_obj, obj_type = const, "Constraint"
                    
            if found_obj:
                trace = getattr(found_obj, "traceability", None)
                orig_text = getattr(trace, "original_text", "N/A") if trace else "N/A"
                doc_name = getattr(trace, "doc_id", "N/A") if trace else "N/A"
                
                tracing_info.append({
                    "check_name": failure.get("check_name"),
                    "violation": failure.get("violation"),
                    "efs_id": efs_id,
                    "efs_type": obj_type,
                    "efs_name": getattr(found_obj, "name", "N/A"),
                    "source_spec_document": doc_name,
                    "source_spec_text": orig_text
                })

    tracing_str = json.dumps(tracing_info, indent=2) if tracing_info else "No detailed EFS IR tracing available."
    
    efs_ir_str = ""
    if efs_ir:
        try:
            efs_ir_str = json.dumps(efs_ir.to_dict(), indent=2)
        except Exception:
            pass

    rules_summary = "\n".join([f"- [{r.get('citation', r.get('chunk_id', 'spec'))}] {r.get('text', '')[:200]}..." for r in protocol_rules[:3]]) if protocol_rules else "None"
    user_prompt = f"""
=== CANONICAL HARDWARE IR (EFS IR) ===
{efs_ir_str if efs_ir_str else "Not Available"}

=== ORIGINAL CODE ===
{generated_code}

=== COMPLIANCE VIOLATIONS TRACED TO SOURCE SPEC ===
{tracing_str}

=== DESIGN SPECIFICATION ===
{design_spec}

=== RETRIEVED PROTOCOL RULES ===
{rules_summary}

=== COMPLIANCE REPORT ===
{json.dumps(compliance_report, indent=2)}
"""

    if len(user_prompt) > 4000:
        user_prompt = user_prompt[:4000] + "\n...[Payload capped for token safety]..."

    if llm.available:
        result = llm.complete_json(SYSTEM_PROMPT, user_prompt, model=CONFIG.llm.code_model)
        if result:
            code = result.get("repaired_code")
            if not isinstance(code, str):
                code = ""
            from core.agents.rtl_agent import _clean_code_output
            result["repaired_code"] = _clean_code_output(code)
            return result
        logger.warning("LLM failed to return a valid JSON repair. Falling back to heuristic repair.")

    return get_heuristic_repair(generated_code, compliance_report, efs_ir=efs_ir)


def get_heuristic_repair(code: str, compliance_report: Dict[str, Any], efs_ir: Optional[EFSIR] = None) -> Dict[str, Any]:
    """Heuristic repair when LLM is unavailable."""
    repaired_code = code
    repairs = []
    
    # Process failed checks and perform simple regex replacements
    for failure in compliance_report.get("failed_checks", []):
        check = failure.get("check_name")
        violation = failure.get("violation")
        correction = failure.get("suggested_correction")
        snippet = failure.get("rtl_snippet")
        rule = failure.get("protocol_rule", "General protocol rule")
        
        if snippet and correction and snippet in repaired_code:
            repaired_code = repaired_code.replace(snippet, correction)
            repairs.append({
                "check_name": check,
                "what_changed": f"Replaced '{snippet}' with '{correction}'",
                "why_changed": f"Resolved violation: {violation}",
                "protocol_rule_or_spec": rule
            })
            
    if not repairs:
        # Default fallback repair
        repairs.append({
            "check_name": "No-op Repair",
            "what_changed": "None",
            "why_changed": "No simple heuristics matched for automatic repair. Manual verification required.",
            "protocol_rule_or_spec": "N/A"
        })
        
    return {
        "repaired_code": repaired_code,
        "repairs": repairs
    }
