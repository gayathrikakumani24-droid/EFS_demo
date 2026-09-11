"""
Design Specification Agent.

Consolidates the Requirement Model, High-Level Plan, and Retrieved Protocol Rules
into a single structured Design Specification document.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from config import CONFIG
from core.extraction.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("agents.spec")

SYSTEM_PROMPT = """You are a Principal Hardware Architect.
Your task is to write a detailed, robust Hardware Design Specification markdown document.

Inputs to combine:
1. Requirement Model: Functional requirements, user-specified interfaces, registers, state machines.
2. Architecture Plan: Module hierarchies, sub-blocks, clock domain plans.
3. Retrieved Protocol Rules: Specific rules (handshake, timing, timing constraints) pulled from standards specs.

The output specification must contain the following sections:
- Module Hierarchy: Sub-modules and how they interconnect.
- External Interface Port List: Direct, specific signal tables (Name, Width, Direction, Protocol Mapping).
- Clocks & Resets: Expected clock frequencies, active level of resets, synchronicity.
- State Machine (FSM) Description: State variables, transitions, state encodings, outputs.
- Register & FIFO Layout: Registers offsets, write/read behavior, field descriptions.
- Protocol Compliance Checklist: Specific rules from retrieved sections that this design guarantees compliance with.
- Key Design Assumptions: Any design limits, parameters, boundaries.

Your output must be a Markdown string containing this formal technical document. Use clear tables for registers and interfaces.
Provide the technical specification document in Markdown format directly, without wrapping it in a parent JSON or adding conversational preamble.
"""


def _format_protocol_rules_compact(rules: List[Dict[str, Any]]) -> str:
    if not rules:
        return "None Available"
    lines = []
    for r in rules:
        cit = r.get("citation") or r.get("chunk_id") or "spec"
        text = r.get("text", "").strip().replace("\n", " ")
        lines.append(f"- [{cit}] {text}")
    return "\n".join(lines)


def generate_design_spec(requirement_model: Dict[str, Any], plan: Dict[str, Any], protocol_rules: List[Dict[str, Any]]) -> str:
    """Generate a cohesive, structured Design Specification document."""
    logger.info("Generating intermediate hardware design specification...")
    llm = get_llm_client()
    
    # Preserving full requirement model details without array capping
    compact_model = {}
    for k, v in requirement_model.items():
        if isinstance(v, list):
            compact_model[k] = v
        else:
            compact_model[k] = v

    rules_str = _format_protocol_rules_compact(protocol_rules)
    user_prompt = f"""
=== REQUIREMENT MODEL ===
{json.dumps(compact_model, indent=2)}

=== ARCHITECTURE PLAN ===
{json.dumps(plan, indent=2)}

=== RETRIEVED PROTOCOL RULES ===
{rules_str}
"""

    if llm.available:
        result = llm.complete_text(SYSTEM_PROMPT, user_prompt, model=CONFIG.llm.extraction_model)
        if result:
            return result
        logger.warning("LLM failed to return a valid design spec text. Falling back to heuristic spec.")

    return get_heuristic_spec(requirement_model, plan, protocol_rules)


def get_heuristic_spec(req_model: Dict[str, Any], plan: Dict[str, Any], protocol_rules: List[Dict[str, Any]]) -> str:
    """Heuristic spec generator when LLM is unavailable."""
    ports_rows = []
    for port in req_model.get("interfaces", []):
        ports_rows.append(f"| {port['name']} | {port['width']} | {port['direction']} | {port['description']} |")
        
    regs_rows = []
    for r in req_model.get("registers", []):
        regs_rows.append(f"| {r['name']} | {r['offset']} | {r['access']} | {r['description']} |")

    rules_list = []
    for rule in protocol_rules:
        rules_list.append(f"- **Rule (from {rule['citation']})**: {rule['text']}")

    return f"""# Hardware Design Specification: {req_model.get('protocol_references', ['Custom'])[0]} Peripheral Controller

## 1. Module Hierarchy
The controller comprises the following sub-modules:
- **Register_File**: Interfaces with the external bus and houses the register fields.
- **FSM_Control_Unit**: Executes interface handshakes and manages state transactions.
- **FIFO/Memories**: Internal buffers (if configured).

## 2. External Interface Port List
| Signal Name | Width | Direction | Description |
|---|---|---|---|
{chr(10).join(ports_rows)}

## 3. Clocks & Resets
- Clock: `clk` (Rising edge active)
- Reset: `rst_n` (Active-low, asynchronous reset)

## 4. State Machine (FSM) Description
States: {", ".join(req_model.get("fsm_info", {}).get("states", [])) if req_model.get("fsm_info", {}).get("states") else "Unspecified in raw requirements."}
Transitions:
{chr(10).join([f"- {t.get('from', 'SRC')} -> {t.get('to', 'TGT')}: when {t.get('condition', 'condition is met')}." for t in req_model.get("fsm_info", {}).get("transitions", []) if isinstance(t, dict)]) if req_model.get("fsm_info", {}).get("transitions") else "- Transitions as specified in source requirements."}

## 5. Register Layout
| Register Name | Offset | Access | Description |
|---|---|---|---|
{chr(10).join(regs_rows)}

## 6. Protocol Compliance Checklist
{chr(10).join(rules_list) if rules_list else "- General interface compliance."}

## 7. Key Design Assumptions
- Target frequency: 100MHz.
- Standard protocol timings apply.
"""
