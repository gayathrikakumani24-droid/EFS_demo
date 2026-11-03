"""
Design Specification Agent.

Consolidates the Requirement Model, High-Level Plan, and Retrieved Protocol Rules
into a single structured Design Specification document.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from config import CONFIG
from core.efs_ir.validation_rules import is_valid_signal_candidate
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


def _clean_model_for_json(req_model: Any) -> Dict[str, Any]:
    """Ensure requirement_model is clean and JSON serializable."""
    if hasattr(req_model, "to_full_combined_dict"):
        data = req_model.to_full_combined_dict()
    elif hasattr(req_model, "to_dict"):
        data = req_model.to_dict()
    elif isinstance(req_model, dict):
        data = req_model
    else:
        return {}

    def _sanitize(val: Any) -> Any:
        if hasattr(val, "to_dict"):
            return _sanitize(val.to_dict())
        elif isinstance(val, dict):
            return {k: _sanitize(v) for k, v in val.items() if not str(k).startswith("_")}
        elif isinstance(val, (list, tuple, set)):
            return [_sanitize(x) for x in val]
        elif isinstance(val, (str, int, float, bool)) or val is None:
            return val
        return str(val)

    return _sanitize(data)


def generate_design_spec(requirement_model: Any, plan: Dict[str, Any], protocol_rules: List[Dict[str, Any]], efs_ir: Optional[Any] = None) -> str:
    """Generate a cohesive, structured Design Specification document using compiled GenerationContext."""
    logger.info("Generating intermediate hardware design specification using compiled GenerationContext...")
    llm = get_llm_client()

    from core.context_compiler.generation_context import GenerationContext
    model_to_use = efs_ir if efs_ir is not None else requirement_model
    gen_ctx = GenerationContext.from_spec_inputs(
        model_to_use,
        plan,
        protocol_rules,
        artifact_type="design_spec",
        token_budget=4000
    )

    user_prompt = f"""=== COMPILED DESIGN SPECIFICATION CONTEXT ===
{gen_ctx.to_prompt_json(indent=2)}
"""
    gen_ctx.calculate_metrics(user_prompt, system_prompt=SYSTEM_PROMPT)

    if llm.available:
        result = llm.complete_text(SYSTEM_PROMPT, user_prompt, model=CONFIG.llm.extraction_model)
        provider_tokens = llm.last_usage.get("prompt_tokens") if hasattr(llm, "last_usage") else None
        output_tokens = llm.last_usage.get("completion_tokens", 0) if hasattr(llm, "last_usage") else 0
        gen_ctx.log_telemetry(
            generation_name="spec_agent_generate_design_spec",
            model=CONFIG.llm.extraction_model,
            provider_tokens=provider_tokens,
            output_tokens=output_tokens
        )
        if result:
            return result
        logger.warning("LLM failed to return a valid design spec text. Falling back to heuristic spec.")

    clean_model = _clean_model_for_json(requirement_model)
    return get_heuristic_spec(clean_model, plan, protocol_rules, efs_ir=efs_ir)


def get_heuristic_spec(req_model: Dict[str, Any], plan: Dict[str, Any], protocol_rules: List[Dict[str, Any]], efs_ir: Optional[Any] = None) -> str:
    """Heuristic spec generator when LLM is unavailable."""
    ports_rows = []
    ports = req_model.get("signals") or []
    if not ports and req_model.get("interfaces"):
        ports = req_model.get("interfaces")

    known_comps = [m.get("name", "") if isinstance(m, dict) else str(m) for m in req_model.get("modules", []) + req_model.get("components", [])]
    known_ifaces = [i.get("name", "") if isinstance(i, dict) else str(i) for i in req_model.get("interfaces", [])]

    for port in ports:
        if isinstance(port, dict):
            p_name = port.get("name", "signal")
            p_width = port.get("width", "1")
            p_dir = port.get("direction", "input")
            p_desc = port.get("description", "-")
            if not is_valid_signal_candidate(p_name, port, known_components=known_comps, known_interfaces=known_ifaces):
                continue
            ports_rows.append(f"| {p_name} | {p_width} | {p_dir} | {p_desc} |")
        elif hasattr(port, "name"):
            p_name = getattr(port, "name", "signal")
            p_width = getattr(port, "width", "1")
            p_dir = getattr(port, "direction", "input")
            p_desc = getattr(port, "description", "-")
            if not is_valid_signal_candidate(p_name, getattr(port, "__dict__", {}), known_components=known_comps, known_interfaces=known_ifaces):
                continue
            ports_rows.append(f"| {p_name} | {p_width} | {p_dir} | {p_desc} |")
        
    # Opcodes & Instruction set table
    raw_opcs = []
    if efs_ir and hasattr(efs_ir, "opcodes") and efs_ir.opcodes:
        raw_opcs = efs_ir.opcodes
    elif req_model.get("opcodes"):
        raw_opcs = req_model.get("opcodes")

    opc_rows = []
    for op in raw_opcs:
        op_m = (op.mnemonic if hasattr(op, "mnemonic") else op.get("mnemonic", op.get("name", "")) or "").strip()
        op_e = str(op.binary_encoding if hasattr(op, "binary_encoding") else op.get("binary_encoding", op.get("encoding", "")) or "").strip()
        op_desc = (op.operation if hasattr(op, "operation") else op.get("operation", "")) or "-"
        if op_m:
            opc_rows.append(f"| {op_m} | {op_e or '-'} | {op_desc} |")

    opc_section = ""
    if opc_rows:
        opc_table = "\n".join(opc_rows)
        opc_section = f"\n\n## 6. Opcode and Instruction Definitions\n| Mnemonic | Binary Encoding | Operation |\n|---|---|---|\n{opc_table}"

    regs_rows = []
    for r in req_model.get("registers", []):
        if isinstance(r, dict):
            r_name = r.get("name", "REG")
            r_offset = r.get("offset", r.get("address", "-"))
            r_access = r.get("access", r.get("mode", "RW"))
            r_desc = r.get("description", "-")
            regs_rows.append(f"| {r_name} | {r_offset} | {r_access} | {r_desc} |")
        elif hasattr(r, "name"):
            r_name = getattr(r, "name", "REG")
            r_offset = getattr(r, "offset", getattr(r, "address", "-"))
            r_access = getattr(r, "access", getattr(r, "mode", "RW"))
            r_desc = getattr(r, "description", "-")
            regs_rows.append(f"| {r_name} | {r_offset} | {r_access} | {r_desc} |")

    rules_list = []
    for rule in protocol_rules:
        if isinstance(rule, dict):
            citation = rule.get("citation") or rule.get("source") or "Specification"
            text = rule.get("text") or rule.get("rule") or rule.get("description") or str(rule)
            rules_list.append(f"- **Rule (from {citation})**: {text}")
        else:
            rules_list.append(f"- **Rule**: {rule}")

    proto_refs = req_model.get("protocol_references", [])
    proto_title = proto_refs[0] if proto_refs else req_model.get("design_name", "Hardware")

    return f"""# Hardware Design Specification: {proto_title} Peripheral Controller

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

{opc_section}

## 7. Protocol Compliance Checklist
{chr(10).join(rules_list) if rules_list else "- General interface compliance."}

## 8. Key Design Assumptions
- Target frequency: 100MHz.
- Standard protocol timings apply.
"""
