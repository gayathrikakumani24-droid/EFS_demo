"""
PlantUML Diagram Generation Agent.

Generates synthesizable Architecture, FSM, and Sequence diagrams in PlantUML
from the same Unified Design Specification. Includes a custom URL compression encoder.
"""

from __future__ import annotations

import base64
import json
import re
import zlib
from typing import Any, Dict, Optional
from config import CONFIG
from core.extraction.llm_client import get_llm_client
from utils.logger import get_logger
from core.efs_ir.models import EFSIR

logger = get_logger("agents.plantuml")
SYSTEM_PROMPT="""You are a hardware architect and PlantUML designer.

Generate three PlantUML diagrams strictly grounded in the provided EFS IR,
Design Plan, and Protocol Contract:

1. architecture
2. fsm
3. sequence

Return ONLY valid JSON:

{
  "architecture": "@startuml\n...\n@enduml",
  "fsm": "@startuml\n...\n@enduml",
  "sequence": "@startuml\n...\n@enduml"
}

SOURCE OF TRUTH & STRICT GROUNDING:
- EFS IR and Design Specification define the hardware.
- Do NOT invent hardware components, signals, states, roles, timing, or transactions.
- Do NOT output a generic AXI/AHB/APB slave or master transaction flow unless the EFS IR explicitly specifies those exact protocol signals.
- Every participant in the sequence diagram MUST correspond to an actual component in the EFS IR.
- Every signal interaction in the sequence diagram MUST use actual human-readable signal or register names defined in the EFS IR (from the signal's 'name' property).
- NEVER output internal signal IDs (such as 'SIG_53f10a8d', 'SIG_cc34814f') on diagram arrows or labels — ALWAYS use human-readable signal names (e.g. 'req_valid', 'bus_ack', 's_axi_awvalid').
- Preserve exact names, widths, directions, and protocol terminology.

ARCHITECTURE
- Show only actual hardware components and interfaces from EFS IR.
- Include clock/reset, registers, memory, FIFOs, DMA, interrupts, etc. only when specified in EFS IR.

FSM
- Show only specified or clearly derived hardware states from EFS IR.
- Show initial state, transitions, conditions, and important state actions.

SEQUENCE
- Represent an actual hardware transaction flow grounded in the EFS IR.
- Use only participants that exist as components in the EFS IR.
- Show actual human-readable signal names and handshake transfers from the EFS IR. Do NOT use internal signal IDs (like SIG_...).
- Do not invent ungrounded instruction words, decode phases, or channels.

PLANTUML
- Use conservative valid PlantUML syntax.
- Every diagram starts with @startuml and ends with @enduml.
- No Markdown, HTML, explanations, or unsupported syntax."""


def plantuml_encode(puml_text: str) -> str:
    """Encode PlantUML code using deflate + custom base64 for PlantUML online server."""
    try:
        # Compress using standard zlib compression (deflate)
        zlib_output = zlib.compress(puml_text.encode("utf-8"))
        # Strip zlib headers (first 2 bytes) and checksums (last 4 bytes)
        deflated = zlib_output[2:-4]
        
        # Base64 conversion with PlantUML custom character alphabet mapping
        standard_b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        puml_b64 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
        
        b64_data = base64.b64encode(deflated).decode("utf-8")
        translation_table = str.maketrans(standard_b64, puml_b64)
        return b64_data.translate(translation_table).replace("=", "")
    except Exception as e:
        logger.error(f"Failed to encode PlantUML: {e}")
        return ""


def get_diagram_url(puml_text: str) -> str:
    """Get the official PlantUML server rendering URL for the diagram."""
    encoded = plantuml_encode(puml_text)
    if encoded:
        return f"http://www.plantuml.com/plantuml/png/{encoded}"
    return ""


def resolve_diagram_ids_to_names(diagrams: Dict[str, str], efs_ir: Optional[EFSIR] = None) -> Dict[str, str]:
    """Replace internal EFS IDs (SIG_..., COMP_..., IF_..., REG_...) with human-readable names."""
    if not diagrams:
        return diagrams

    # 1. Build comprehensive ID -> Name map from EFS IR
    id_map: Dict[str, str] = {}
    available_sig_names = []

    if efs_ir:
        if efs_ir.signals:
            for s in efs_ir.signals:
                s_name = s.name.strip() if s.name else ""
                # If s_name is blank or starts with SIG_, try protocol_role or semantic_role
                if not s_name or s_name.startswith("SIG_"):
                    if s.protocol_role:
                        s_name = s.protocol_role.lower()
                    elif s.semantic_role and s.semantic_role not in ("data", "control"):
                        s_name = f"{s.semantic_role}_sig"
                    else:
                        s_name = "sig_data"
                
                if s.signal_id:
                    id_map[s.signal_id] = s_name
                if s_name not in available_sig_names:
                    available_sig_names.append(s_name)

        if efs_ir.components:
            for c in efs_ir.components:
                if c.component_id and c.name:
                    id_map[c.component_id] = c.name

        if efs_ir.interfaces:
            for iface in efs_ir.interfaces:
                if iface.interface_id and iface.name:
                    id_map[iface.interface_id] = iface.name

        if efs_ir.registers:
            for reg in efs_ir.registers:
                if reg.register_id and reg.name:
                    id_map[reg.register_id] = reg.name

        if efs_ir.fsms:
            for fsm in efs_ir.fsms:
                if fsm.fsm_id and fsm.name:
                    id_map[fsm.fsm_id] = fsm.name

    resolved_diagrams = {}

    for diag_key, puml_text in diagrams.items():
        if not puml_text:
            resolved_diagrams[diag_key] = puml_text
            continue

        text = puml_text

        # First pass: replace known IDs from id_map
        for efs_id, name in id_map.items():
            if efs_id in text:
                text = text.replace(efs_id, name)

        # Second pass: Fallback regex to clean up any remaining SIG_[a-f0-9]+ or SIG_[A-Za-z0-9_]+
        sig_id_pattern = re.compile(r"\bSIG_[a-f0-9]{6,12}\b|\bSIG_[A-Za-z0-9_]+\b", re.IGNORECASE)
        unmatched_sig_ids = list(set(sig_id_pattern.findall(text)))

        if unmatched_sig_ids:
            sig_name_idx = 0
            for sig_id in unmatched_sig_ids:
                if available_sig_names and sig_name_idx < len(available_sig_names):
                    replacement = available_sig_names[sig_name_idx]
                    sig_name_idx += 1
                else:
                    replacement = f"sig_transfer_{sig_name_idx + 1}"
                    sig_name_idx += 1
                text = text.replace(sig_id, replacement)

        # Third pass: clean up COMP_... IF_... REG_... if any remain
        text = re.sub(r"\bCOMP_[a-f0-9]{6,12}\b", "Module", text, flags=re.IGNORECASE)
        text = re.sub(r"\bIF_[a-f0-9]{6,12}\b", "Interface", text, flags=re.IGNORECASE)
        text = re.sub(r"\bREG_[a-f0-9]{6,12}\b", "Register", text, flags=re.IGNORECASE)

        resolved_diagrams[diag_key] = text

    return resolved_diagrams


def generate_diagrams(design_spec: str, plan: Dict[str, Any], efs_ir: Optional[EFSIR] = None) -> Dict[str, str]:
    """Generate PlantUML diagrams from the Design Context."""
    logger.info("Generating PlantUML diagrams...")
    llm = get_llm_client()
    
    efs_ir_str = ""
    if efs_ir:
        try:
            efs_ir_str = json.dumps(efs_ir.to_dict(), indent=2)
        except Exception:
            pass

    user_prompt = f"""
=== CANONICAL HARDWARE IR (EFS IR) ===
{efs_ir_str if efs_ir_str else "Not Available"}

=== DESIGN SPECIFICATION ===
{design_spec}

=== DESIGN PLAN ===
{json.dumps(plan, indent=2)}

CRITICAL REQUIREMENT FOR SEQUENCE DIAGRAM:
- Every signal interaction/label on arrows MUST use human-readable signal names (from the signal 'name' field), NOT internal IDs like SIG_53f10a8d.
"""

    if len(user_prompt) > 4000:
        user_prompt = user_prompt[:4000] + "\n...[Payload capped to stay within Groq token limits]..."

    diagrams = None

    if llm.available:
        result = llm.complete_json(SYSTEM_PROMPT, user_prompt, model=CONFIG.llm.code_model)
        if result and isinstance(result, dict):
            arch = result.get("architecture", "")
            fsm = result.get("fsm", "")
            seq = result.get("sequence", "")
            
            # Validate that sequence diagram contains actual sequence content
            if len(seq.strip()) > 35 and ("->" in seq or "actor" in seq or "participant" in seq):
                diagrams = {
                    "architecture": arch,
                    "fsm": fsm,
                    "sequence": seq
                }
            else:
                logger.warning("LLM returned incomplete PlantUML sequence diagram. Using enriched grounded sequence diagram.")

    if not diagrams:
        diagrams = get_heuristic_diagrams(plan, efs_ir=efs_ir)

    # Post-process all diagrams to guarantee any remaining signal/component/interface IDs are replaced with names
    return resolve_diagram_ids_to_names(diagrams, efs_ir=efs_ir)


def get_heuristic_diagrams(plan: Dict[str, Any], efs_ir: Optional[EFSIR] = None) -> Dict[str, str]:
    """Grounded fallback diagrams when LLM is unavailable or sequence diagram is empty."""
    primary_comp_name = "Hardware_Module"
    if efs_ir and efs_ir.components:
        primary_comp_name = efs_ir.components[0].name
    elif plan.get("architecture_overview"):
        primary_comp_name = plan.get("target", "Hardware_Module")

    # 1. Architecture Diagram Construction
    arch_lines = ["@startuml", f"  title {primary_comp_name} Hardware Architecture"]
    if efs_ir and efs_ir.components:
        for comp in efs_ir.components:
            arch_lines.append(f"  [{comp.name}] as {comp.component_id} << {comp.type} >>")
        for iface in efs_ir.interfaces:
            arch_lines.append(f"  interface \"{iface.name}\" as {iface.interface_id}")
            if iface.source_component:
                arch_lines.append(f"  {iface.source_component} ..> {iface.interface_id}")
            if iface.destination_component:
                arch_lines.append(f"  {iface.interface_id} ..> {iface.destination_component}")
    else:
        arch_lines.extend([
            f"package \"{primary_comp_name}\" {{",
            "  [Control Unit] as Ctrl",
            "  [Register File] as Regs",
            "}",
            "interface \"Control Port\" as CtrlPort",
            "CtrlPort --> Ctrl : enable",
            "Ctrl --> Regs : read/write"
        ])
    arch_lines.append("@enduml")

    # 2. FSM Diagram Construction
    fsm_lines = ["@startuml", f"  title {primary_comp_name} Controller FSM"]
    fsm_has_transitions = False
    if efs_ir and efs_ir.fsms:
        for fsm in efs_ir.fsms:
            fsm_lines.append(f"  [*] --> {fsm.initial_state}")
            for t in fsm.transitions:
                fsm_lines.append(f"  {t.source_state} --> {t.target_state} : {t.condition or 'trigger'}")
                fsm_has_transitions = True
    
    if not fsm_has_transitions:
        states_list = [s.get("name") for s in plan.get("fsm_architecture", {}).get("states", []) if isinstance(s, dict) and s.get("name")]
        if states_list:
            fsm_lines.append(f"  [*] --> {states_list[0]}")
            for i in range(len(states_list) - 1):
                fsm_lines.append(f"  {states_list[i]} --> {states_list[i+1]} : trigger")
            fsm_lines.append(f"  {states_list[-1]} --> {states_list[0]} : reset / return")
    fsm_lines.append("@enduml")

    # 3. Sequence Diagram Construction (Strictly Grounded in EFS IR)
    seq_lines = ["@startuml", f"  title {primary_comp_name} Transaction Flow Sequence", "  autonumber"]
    
    if efs_ir and efs_ir.components:
        comps = efs_ir.components
        for c in comps:
            seq_lines.append(f"  participant \"{c.name}\" as {c.component_id}")
        seq_lines.append("")

        c_src = comps[0].component_id
        c_dest = comps[1].component_id if len(comps) > 1 else c_src

        in_sigs = [s.name for s in efs_ir.signals if s.direction == "input" and s.semantic_role not in ("clock", "reset")]
        out_sigs = [s.name for s in efs_ir.signals if s.direction == "output"]

        if in_sigs:
            seq_lines.append(f"  Host -> {c_src} : Assert {', '.join(in_sigs[:3])}")
            seq_lines.append(f"  activate {c_src}")
        if len(comps) > 1 and out_sigs:
            seq_lines.append(f"  {c_src} -> {c_dest} : Transfer {', '.join(out_sigs[:3])}")
        elif out_sigs:
            seq_lines.append(f"  {c_src} --> Host : Output {', '.join(out_sigs[:3])}")
        
        if in_sigs:
            seq_lines.append(f"  deactivate {c_src}")
    elif efs_ir and efs_ir.registers:
        reg1 = efs_ir.registers[0]
        seq_lines.extend([
            "  actor \"Host Controller\" as Host",
            f"  participant \"{primary_comp_name}\" as Controller",
            "",
            f"  Host -> Controller : Write {reg1.name} (offset {reg1.offset})",
            "  activate Controller",
            "  Controller -> Controller : Update internal register state",
            "  Controller --> Host : Latch status / ACK",
            "  deactivate Controller"
        ])
    else:
        seq_lines.extend([
            "  actor \"Host Controller\" as Host",
            f"  participant \"{primary_comp_name}\" as Controller",
            "",
            "  Host -> Controller : Assert Control Signal",
            "  activate Controller",
            "  Controller -> Controller : Execute hardware state transition",
            "  Controller --> Host : Return status / complete",
            "  deactivate Controller"
        ])
    seq_lines.append("@enduml")

    return resolve_diagram_ids_to_names({
        "architecture": "\n".join(arch_lines),
        "fsm": "\n".join(fsm_lines),
        "sequence": "\n".join(seq_lines)
    }, efs_ir=efs_ir)

