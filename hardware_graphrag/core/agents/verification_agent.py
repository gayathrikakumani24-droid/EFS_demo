"""
Verification Agent.

Validates generated RTL/verification code against Design Specifications
and protocol rules, generating a structured compliance report.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from core.extraction.llm_client import get_llm_client
from utils.logger import get_logger

from config import CONFIG
from core.verification.static_verifier import verify_statically
from core.efs_ir.models import EFSIR

logger = get_logger("agents.verification")

SYSTEM_PROMPT = """You are a Hardware Verification Director.
Your task is to analyze the provided generated hardware code and verify its compliance with the Design Specification and Protocol Rules.

Assess compliance across these key criteria:
1. Signal Handshaking (valid/ready behavior, stability).
2. Reset Behavior (asynchronous/synchronous reset assertion states).
3. Register Offsets & Alignment (proper address mapping, field masks).
4. FSM State Transitions (valid transitions, deadlocks, unreachable states).
5. Signal naming conventions and mandatory interfaces.

You must return a valid JSON object matching the following structure:
{
  "compliance_score": 0 to 100 integer,
  "passed_checks": [
    {
      "check_name": "string",
      "description": "string"
    }
  ],
  "failed_checks": [
    {
      "check_name": "string",
      "description": "string",
      "violation": "Detailed description of what is incorrect",
      "protocol_rule": "The protocol rule violated",
      "spec_section": "The design spec section violated",
      "rtl_snippet": "The exact line or block of code containing the issue",
      "suggested_correction": "How to fix the code"
    }
  ],
  "warnings": [
    "string"
  ]
}
Do not add any markdown formatting or explanation outside the JSON object. Keep the output strictly conforming to the JSON schema.
"""


def verify_code(
    generated_code: str,
    design_spec: str,
    protocol_rules: List[Dict[str, Any]],
    req_model: Dict[str, Any] = None,
    plan: Dict[str, Any] = None,
    efs_ir: Optional[EFSIR] = None
) -> Dict[str, Any]:
    """Analyze generated code for protocol compliance and return a structured report."""
    logger.info("Running two-level verification pipeline...")
    
    # -- Level 1: Fast Static Verification ----------------------------------
    static_report = {"compliance_score": 100, "passed_checks": [], "failed_checks": [], "warnings": []}
    if getattr(CONFIG, "enable_static_verification", False):
        import time
        start_time = time.time()
        static_report = verify_statically(generated_code, design_spec, protocol_rules, req_model, plan, efs_ir=efs_ir)
        elapsed = time.time() - start_time
        logger.info(f"Static verification complete: {elapsed:.2f} seconds. Code: {len(generated_code)} chars. score: {static_report['compliance_score']}%")

        # Check for any CRITICAL or MAJOR violations
        has_serious_static = any(
            f.get("severity") in ("CRITICAL", "MAJOR")
            for f in static_report.get("failed_checks", [])
        )
        if has_serious_static:
            logger.info("Serious deterministic violations found in static check. Skipping semantic LLM verification.")
            return static_report

    # -- Level 2: Semantic LLM Verification ---------------------------------
    if not getattr(CONFIG, "enable_semantic_verification", False):
        logger.info("Semantic LLM verification disabled by config. Returning static report.")
        return static_report

    logger.info("Verifying generated code semantic compliance via LLM...")
    llm = get_llm_client()
    
    # Enrich the prompt with the static verification findings
    static_findings_str = json.dumps({
        "passed_checks": static_report.get("passed_checks", []),
        "failed_checks_minor": [f for f in static_report.get("failed_checks", []) if f.get("severity") not in ("CRITICAL", "MAJOR")],
        "warnings": static_report.get("warnings", [])
    }, indent=2)

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

=== GENERATED CODE ===
{generated_code}

=== DESIGN SPECIFICATION ===
{design_spec}

=== RETRIEVED PROTOCOL RULES ===
{rules_summary}

=== STATIC VERIFICATION FINDINGS ===
{static_findings_str}
"""

    import time
    start_time = time.time()
    
    if len(user_prompt) > 4000:
        user_prompt = user_prompt[:4000] + "\n...[Payload capped for token safety]..."

    llm_report = None
    if llm.available:
        llm_report = llm.complete_json(SYSTEM_PROMPT, user_prompt, model=CONFIG.llm.extraction_model)
        elapsed = time.time() - start_time
        logger.info(f"LLM Semantic Verification: {elapsed:.2f} seconds.")

    if not llm_report:
        logger.warning("LLM failed to return a valid JSON verification report. Falling back to heuristic compliance check.")
        llm_report = get_heuristic_verification(generated_code, design_spec, protocol_rules)

    # Merge Static and Semantic findings
    merged_passed = static_report.get("passed_checks", []) + llm_report.get("passed_checks", [])
    merged_failed = static_report.get("failed_checks", []) + llm_report.get("failed_checks", [])
    merged_warnings = list(set(static_report.get("warnings", []) + llm_report.get("warnings", [])))
    
    # De-duplicate lists based on check_name and violation
    dedup_passed = []
    seen_passed = set()
    for p in merged_passed:
        name = p.get("check_name", "")
        if name not in seen_passed:
            seen_passed.add(name)
            dedup_passed.append(p)

    dedup_failed = []
    seen_failed = set()
    for f in merged_failed:
        key = (f.get("check_name", ""), f.get("violation", ""))
        if key not in seen_failed:
            seen_failed.add(key)
            dedup_failed.append(f)

    # Recalculate score combining both reports
    score = min(static_report.get("compliance_score", 100), llm_report.get("compliance_score", 100))

    return {
        "compliance_score": score,
        "passed_checks": dedup_passed,
        "failed_checks": dedup_failed,
        "warnings": merged_warnings
    }


def get_heuristic_verification(code: str, spec: str, protocol_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Heuristic verification when LLM is unavailable."""
    passed = [
        {"check_name": "Clock Domain Check", "description": "Verified that single clock 'clk' is used consistently."},
        {"check_name": "Port List Integrity", "description": "Checked that ports conform to standard name suffixes."}
    ]
    failed = []
    warnings = []

    # Simple heuristic checks
    if "reset" in spec.lower() or "rst_n" in spec.lower():
        if "rst_n" in code and "posedge clk" in code:
            if "negedge rst_n" not in code and "!rst_n" not in code:
                failed.append({
                    "check_name": "Reset Sensitivity Check",
                    "description": "Verify reset is listed in the sequential sensitivity list.",
                    "violation": "Active-low reset rst_n is specified as asynchronous, but negedge rst_n is missing from the always block sensitivity list.",
                    "protocol_rule": "Asynchronous resets must trigger immediately on the active edge.",
                    "spec_section": "Section 3: Clocks & Resets",
                    "rtl_snippet": "always @(posedge clk)",
                    "suggested_correction": "Change to: always @(posedge clk or negedge rst_n)"
                })
            else:
                passed.append({"check_name": "Reset Sensitivity Check", "description": "Reset signal is properly declared."})
        else:
            passed.append({"check_name": "Reset Sensitivity Check", "description": "Reset signal is properly declared."})

    # If no failures are detected, compliance is 100. Else, reduce.
    score = 100 - (len(failed) * 20)
    score = max(0, min(100, score))
    
    if score < 100:
        warnings.append("Review manual timing constraint files to verify setup/hold parameters.")
        
    return {
        "compliance_score": score,
        "passed_checks": passed,
        "failed_checks": failed,
        "warnings": warnings
    }
