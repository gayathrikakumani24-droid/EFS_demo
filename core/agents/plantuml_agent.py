"""
PlantUML Diagram Generation Agent.

Generates synthesizable Architecture, FSM, and Sequence diagrams in PlantUML
from the same Unified Design Specification. Includes safe identifier sanitization,
alias registry, syntax validation, and custom URL compression encoder.
"""

from __future__ import annotations

import base64
import json
import re
import zlib
from typing import Any, Dict, List, Optional, Set, Tuple
from config import CONFIG
from core.extraction.llm_client import get_llm_client
from utils.logger import get_logger
from core.efs_ir.models import EFSIR
from core.agents.plantuml_utils import (
    PlantUMLAliasRegistry,
    to_plantuml_alias,
    resolve_canonical_design_name,
    validate_plantuml_syntax,
    validate_interaction,
    PUMLValidationResult
)

logger = get_logger("agents.plantuml")

SYSTEM_PROMPT = """You are a hardware architect and PlantUML designer.

Generate three PlantUML diagrams strictly grounded in the provided EFS IR,
Design Plan, and Protocol Contract:

1. architecture
2. fsm
3. sequence

Return ONLY valid JSON:

{
  "architecture": "@startuml\\n...\\n@enduml",
  "fsm": "@startuml\\n...\\n@enduml",
  "sequence": "@startuml\\n...\\n@enduml"
}

SOURCE OF TRUTH & STRICT GROUNDING:
- EFS IR is the single resolved source of truth for all diagrams.
- Architecture:
  * When subcomponents exist, render the top-level design as an enclosing container (package "<TopName>" as TopAlias <<top_level_design>> { ... }) with its subcomponents declared inside.
  * Connect declared interfaces and signals between their true owner and consumer modules.
  * When topology connections are not specified, do NOT invent arrows; output:
    note as N_topo
      Topology connections not specified in EFS IR.
    end note
- FSM:
  * Use the exact IR FSM name from context.
  * Show the initial reset transition from [*] to the initial state with the declared reset condition (e.g. [*] --> IDLE : reset_n == 0).
  * Render explicit transition guards as boolean expressions in square brackets (e.g. [timer_cnt == 4'd4]).
  * Render per-state outputs inside each state (e.g. StateAlias : output_sig = value).
- Sequence:
  * If sequence flow or transactions are not specified in the EFS IR, do NOT fabricate or invent messages or transactions (such as generic AXI or Assert Control Signal); emit:
    @startuml
    title <CanonicalName> Transaction Flow Sequence
    note "Sequence flow not specified in EFS IR for this design." as N_seq
    @enduml
  * Do NOT output protocol-specific messages (such as AXI/AHB/APB) unless the spec explicitly defines that protocol.
  * Every participant and signal interaction MUST use actual human-readable names defined in the EFS IR, never internal IDs.

CRITICAL PLANTUML SYNTAX & ALIAS RULES:
- Every diagram entity MUST have TWO representations:
  1. display_name (human-readable string in quotes)
  2. plantuml_alias (safe, deterministic identifier without spaces, matching ^[A-Za-z_][A-Za-z0-9_]*$)
- NEVER use a raw display name with spaces as an alias!
  CORRECT:   participant "AXI4 Slave Interface" as AXI4_Slave_Interface
  INCORRECT: participant "AXI4 Slave Interface" as AXI4 Slave Interface
  CORRECT:   component "DMA Controller" as DMA_Controller <<controller>>
  INCORRECT: [DMA Controller] as DMA Controller <<controller>>
- All arrows and messages MUST use the declared safe aliases.
- Use appropriate semantic constructs based on entity type:
  - interface: interface "Name" as Alias
  - memory/storage/fifo/buffer/cache: database "Name" as Alias <<stereotype>>
  - package/subsystem: package "Name" as Alias
  - component/module/controller: component "Name" as Alias <<stereotype>>
- Every diagram starts with @startuml and ends with @enduml.
"""


def plantuml_encode(puml_text: str) -> str:
    """Encode PlantUML code using deflate + custom base64 for PlantUML online server."""
    try:
        zlib_output = zlib.compress(puml_text.encode("utf-8"))
        deflated = zlib_output[2:-4]
        
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


def repair_plantuml_syntax(puml_text: str, efs_ir: Optional[EFSIR] = None) -> str:
    """
    Deterministically repair common PlantUML syntax flaws:
    1. Fix aliases with spaces in declarations: '... as Some Name With Spaces <<' or '... as Some Name With Spaces\n'
    2. Sync updated aliases to arrow lines
    3. Ensure balanced @startuml / @enduml
    """
    if not puml_text:
        return "@startuml\n@enduml"

    lines = puml_text.splitlines()
    repaired_lines: List[str] = []
    alias_replacements: Dict[str, str] = {}

    for line in lines:
        stripped = line.strip()

        # Fix sequence participants with spaces in alias:
        # e.g. participant "Name" as Name With Spaces
        m_seq = re.match(r'^(\s*(?:participant|actor|boundary|control|entity|database|collections|queue)\s+(?:"([^"]+)"|([^\s]+))\s+as\s+)(.+)$', line, re.IGNORECASE)
        if m_seq:
            prefix = m_seq.group(1)
            display = m_seq.group(2) or m_seq.group(3)
            raw_alias = m_seq.group(4).strip()
            if " " in raw_alias or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", raw_alias):
                safe_alias = to_plantuml_alias(display or raw_alias)
                alias_replacements[raw_alias] = safe_alias
                line = f'{prefix.rstrip()} {safe_alias}'
            repaired_lines.append(line)
            continue

        # Fix bracket components with spaces in alias:
        # e.g. [Name] as Name With Spaces <<type>>
        m_brk = re.match(r'^(\s*\[([^\]]+)\]\s+as\s+)([^<>\n\r]+)(\s*<<.*>>)?$', line)
        if m_brk:
            prefix = m_brk.group(1)
            display = m_brk.group(2)
            raw_alias = m_brk.group(3).strip()
            stereotype = m_brk.group(4) or ""
            if " " in raw_alias or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", raw_alias):
                safe_alias = to_plantuml_alias(display or raw_alias)
                alias_replacements[raw_alias] = safe_alias
                line = f'  component "{display}" as {safe_alias}{stereotype}'
            repaired_lines.append(line)
            continue

        # Fix keyword component/interface/database with spaces in alias:
        # e.g. component "Name" as Name With Spaces <<type>>
        m_kw = re.match(r'^(\s*(?:component|interface|database|storage|node|package|frame)\s+(?:"([^"]+)"|([^\s]+))\s+as\s+)([^<>\n\r]+)(\s*<<.*>>)?$', line, re.IGNORECASE)
        if m_kw:
            prefix = m_kw.group(1)
            display = m_kw.group(2) or m_kw.group(3)
            raw_alias = m_kw.group(4).strip()
            stereotype = m_kw.group(5) or ""
            if " " in raw_alias or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", raw_alias):
                safe_alias = to_plantuml_alias(display or raw_alias)
                alias_replacements[raw_alias] = safe_alias
                line = f'{prefix.rstrip()} {safe_alias}{stereotype}'
            repaired_lines.append(line)
            continue

        repaired_lines.append(line)

    # Second pass: propagate alias replacements to relationships, messages, and lifecycle directives
    final_lines: List[str] = []
    for line in repaired_lines:
        # Only replace on arrow/relationship or directive lines to preserve display names in quotes
        if any(sym in line for sym in ("->", "-->", "..>", "--", "activate", "deactivate", "note right of", "note left of")):
            for old_alias, new_alias in alias_replacements.items():
                if old_alias and old_alias != new_alias:
                    line = re.sub(rf'\b{re.escape(old_alias)}\b', new_alias, line)
        final_lines.append(line)
    result_text = "\n".join(final_lines)

    # Ensure @startuml and @enduml are balanced
    if "@startuml" not in result_text:
        result_text = f"@startuml\n{result_text}"
    if "@enduml" not in result_text:
        result_text = f"{result_text}\n@enduml"

    return result_text


def resolve_diagram_ids_to_names(diagrams: Dict[str, str], efs_ir: Optional[EFSIR] = None) -> Dict[str, str]:
    """
    Replace internal EFS IDs (SIG_..., COMP_..., IF_..., REG_...) with deterministic safe aliases
    and human-readable names. Guarantees that no spaces are introduced into aliases.
    """
    if not diagrams:
        return diagrams

    registry = PlantUMLAliasRegistry()
    sig_id_to_name: Dict[str, str] = {}
    available_sig_names: List[str] = []

    if efs_ir:
        if efs_ir.signals:
            for s in efs_ir.signals:
                s_name = s.name.strip() if s.name else ""
                if not s_name or s_name.startswith("SIG_"):
                    if s.protocol_role:
                        s_name = s.protocol_role.lower()
                    elif s.semantic_role and s.semantic_role not in ("data", "control"):
                        s_name = f"{s.semantic_role}_sig"
                    else:
                        s_name = "sig_data"
                
                if s.signal_id:
                    sig_id_to_name[s.signal_id] = s_name
                if s_name not in available_sig_names:
                    available_sig_names.append(s_name)

        if efs_ir.components:
            for c in efs_ir.components:
                if c.name:
                    registry.register(c.name, canonical_id=c.component_id, entity_type=c.type)

        if efs_ir.interfaces:
            for iface in efs_ir.interfaces:
                if iface.name:
                    registry.register(iface.name, canonical_id=iface.interface_id, entity_type="interface")

        if efs_ir.registers:
            for reg in efs_ir.registers:
                if reg.name:
                    registry.register(reg.name, canonical_id=reg.register_id, entity_type="register")

        if efs_ir.fsms:
            for fsm in efs_ir.fsms:
                if fsm.name:
                    registry.register(fsm.name, canonical_id=fsm.fsm_id, entity_type="fsm")

    resolved_diagrams = {}

    for diag_key, puml_text in diagrams.items():
        if not puml_text:
            resolved_diagrams[diag_key] = puml_text
            continue

        text = puml_text

        # First pass: replace signal IDs on labels/messages with human-readable signal names
        for sig_id, sig_name in sig_id_to_name.items():
            if sig_id in text:
                text = text.replace(sig_id, sig_name)

        # Replace entity IDs in alias positions ('as COMP_...') with safe_alias
        for canonical_id, safe_alias in registry._id_to_alias.items():
            disp_name = registry.get_display_name(safe_alias)
            # Replace 'as <canonical_id>' with 'as <safe_alias>'
            text = re.sub(rf'\bas\s+{re.escape(canonical_id)}\b', f'as {safe_alias}', text)
            # Replace arrows: '<canonical_id> ->' or '-> <canonical_id>'
            text = re.sub(rf'\b{re.escape(canonical_id)}\b', safe_alias, text)
            # Replace bracketed '[COMP_...]' with '[display_name]'
            text = text.replace(f"[{canonical_id}]", f'[{disp_name}]')
            text = text.replace(f'"{canonical_id}"', f'"{disp_name}"')

        # Fallback regex for unmapped SIG_[a-f0-9]+
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

        # Third pass: clean up residual unmapped COMP_... IF_... REG_...
        text = re.sub(r"\bCOMP_[a-f0-9]{6,12}\b", "Module", text, flags=re.IGNORECASE)
        text = re.sub(r"\bIF_[a-f0-9]{6,12}\b", "Interface", text, flags=re.IGNORECASE)
        text = re.sub(r"\bREG_[a-f0-9]{6,12}\b", "Register", text, flags=re.IGNORECASE)

        # Repair any syntax or aliases with spaces
        repaired = repair_plantuml_syntax(text, efs_ir=efs_ir)

        # Pre-render syntax validation
        val_result = validate_plantuml_syntax(repaired, diagram_type=diag_key)
        if not val_result.is_valid:
            logger.warning(
                f"PlantUML {diag_key} validation warning/errors: {val_result.errors}. "
                f"Attempting final structural stabilization."
            )
            # Ensure safe fallback diagram if validation failed severely
            if any("Invalid alias" in err or "Duplicate alias" in err for err in val_result.errors):
                repaired = repair_plantuml_syntax(repaired, efs_ir=efs_ir)

        resolved_diagrams[diag_key] = repaired

    return resolved_diagrams


def generate_diagnostic_fsm_diagram(primary_comp_name: str, message: str) -> str:
    """Generate a clean, valid PlantUML diagram when FSM context is insufficient."""
    safe_alias = to_plantuml_alias(primary_comp_name)
    return f"""@startuml
title {primary_comp_name} Controller FSM
state "{primary_comp_name}" as {safe_alias} {{
  [*] --> Idle
  note "INSUFFICIENT_DIAGRAM_CONTEXT: {message}" as N_fsm
}}
@enduml"""


def get_heuristic_diagrams(plan: Dict[str, Any], efs_ir: Optional[EFSIR] = None) -> Dict[str, str]:
    """
    Grounded fallback diagrams when LLM is unavailable or sequence diagram is empty.
    Strictly evidence-driven; generates REAL relationships from EFS IR interfaces,
    signals, and transactions. Never hallucinates missing connections or fake FSM states.
    Only falls back to an inventory note if truly zero relationships exist after full analysis.
    """
    primary_comp_name = resolve_canonical_design_name(efs_ir=efs_ir, design_plan=plan)
    if not primary_comp_name or primary_comp_name.lower() in ("target_module", "target module", "unnamed_design"):
        primary_comp_name = "Hardware_Module"

    registry = PlantUMLAliasRegistry()

    # 1. Architecture Diagram Construction (Top-level as container when submodules exist)
    arch_lines = ["@startuml", f"  title {primary_comp_name} Hardware Architecture"]
    has_relationships = False
    
    if efs_ir and efs_ir.components:
        top_comp = next((c for c in efs_ir.components if (c.type in ("top_level", "top_level_design") or c.name == primary_comp_name or c.name == efs_ir.design_name)), efs_ir.components[0])
        sub_comps = [c for c in efs_ir.components if c != top_comp]

        if sub_comps:
            top_alias = registry.register(top_comp.name, canonical_id=top_comp.component_id, entity_type="top_level_design")
            arch_lines.append(f'  package "{top_comp.name}" as {top_alias} << top_level_design >> {{')
            for sub in sub_comps:
                sub_alias = registry.register(sub.name, canonical_id=sub.component_id, entity_type=sub.type)
                clean_type = re.sub(r"[^A-Za-z0-9_]", "_", (sub.type or "module").lower()).strip("_") or "module"
                arch_lines.append(f'    component "{sub.name}" as {sub_alias} << {clean_type} >>')
            arch_lines.append('  }')
        else:
            top_alias = registry.register(top_comp.name, canonical_id=top_comp.component_id, entity_type=top_comp.type)
            clean_type = re.sub(r"[^A-Za-z0-9_]", "_", (top_comp.type or "component").lower()).strip("_") or "component"
            arch_lines.append(f'  component "{top_comp.name}" as {top_alias} << {clean_type} >>')

        # Generate REAL connections from interfaces (source -> interface -> destination)
        if efs_ir.interfaces:
            for iface in efs_ir.interfaces:
                iface_alias = registry.register(iface.name, canonical_id=iface.interface_id, entity_type="interface")
                arch_lines.append(f'  interface "{iface.name}" as {iface_alias}')
                
                src_alias = registry.get_alias(iface.source_component) if iface.source_component else None
                tgt_alias = registry.get_alias(getattr(iface, "destination_component", None) or getattr(iface, "target_component", None) or "") if (getattr(iface, "destination_component", None) or getattr(iface, "target_component", None)) else None
                
                if src_alias and registry.is_declared(src_alias):
                    protocol_label = f" : {iface.protocol}" if iface.protocol else ""
                    arch_lines.append(f"  {src_alias} ..> {iface_alias}{protocol_label}")
                    has_relationships = True
                if tgt_alias and registry.is_declared(tgt_alias):
                    arch_lines.append(f"  {iface_alias} ..> {tgt_alias}")
                    has_relationships = True

        # Derive inter-component connections from signal ownership and consumption
        if efs_ir.signals:
            sig_conns: Dict[Tuple[str, str], List[str]] = {}
            for s in efs_ir.signals:
                if s.owner and s.consumer and s.owner != s.consumer:
                    o_alias = registry.get_alias(s.owner) or to_plantuml_alias(s.owner)
                    c_alias = registry.get_alias(s.consumer) or to_plantuml_alias(s.consumer)
                    if o_alias and c_alias and o_alias != c_alias:
                        key = (o_alias, c_alias)
                        sig_conns.setdefault(key, []).append(s.name)

            for (src_a, dst_a), s_names in sig_conns.items():
                arch_lines.append(f"  {src_a} --> {dst_a} : {', '.join(s_names[:4])}")
                has_relationships = True

        # Show register map as notes if registers exist
        if efs_ir.registers and not has_relationships:
            reg_summary = [f"{r.name}@{r.offset}" for r in efs_ir.registers[:5]]
            top_alias_inst = registry.get_alias(top_comp.component_id) or to_plantuml_alias(primary_comp_name)
            if top_alias_inst:
                arch_lines.append(f'  note bottom of {top_alias_inst}')
                arch_lines.append(f'    Registers: {", ".join(reg_summary)}')
                arch_lines.append(f'  end note')
                has_relationships = True

        if not has_relationships:
            arch_lines.append("  note as N_topo")
            arch_lines.append("    Topology connections not specified in EFS IR.")
            arch_lines.append("  end note")
    else:
        prim_alias = registry.register(primary_comp_name, entity_type="top_level_design")
        arch_lines.extend([
            f'  component "{primary_comp_name}" as {prim_alias} << top_level_design >>',
            f"  note right of {prim_alias}",
            "    Topology connections not specified in EFS IR.",
            "  end note"
        ])
    arch_lines.append("@enduml")

    # 2. FSM Diagram Construction (IR name, reset transition, per-state outputs, explicit guards)
    fsm_lines = ["@startuml"]
    fsm_has_transitions = False
    
    if efs_ir and efs_ir.fsms:
        fsm = efs_ir.fsms[0]
        fsm_name = fsm.name or f"{primary_comp_name}_fsm"
        fsm_lines.append(f"  title {fsm_name}")

        # Reset transition from IR reset definition
        rst_cond_label = ""
        if hasattr(efs_ir, "reset") and efs_ir.reset and getattr(efs_ir.reset, "signal", None):
            r_sig = efs_ir.reset.signal
            r_desc = (getattr(efs_ir.reset, "description", "") or "").lower()
            is_active_low = False if ("active_high" in r_desc or "active-high" in r_desc) else getattr(efs_ir.reset, "active_low", True)
            r_val = "0" if is_active_low else "1"
            rst_cond_label = f" : {r_sig} == {r_val}"
        elif getattr(efs_ir, "resets", None):
            r_obj = efs_ir.resets[0]
            r_sig = r_obj.name or r_obj.signal_id
            r_desc = (getattr(r_obj, "description", "") or "").lower()
            r_name = (getattr(r_obj, "name", "") or "").lower()
            if "active_high" in r_desc or "active-high" in r_desc:
                is_active_low = False
            elif "active_low" in r_desc or "active-low" in r_desc:
                is_active_low = True
            elif r_name.endswith("_n") or r_name.endswith("_b") or r_name in ("rst_n", "reset_n") or r_name.startswith("aresetn"):
                is_active_low = True
            elif r_name.endswith("_p") or r_name in ("rst", "reset", "sys_rst"):
                is_active_low = False
            else:
                is_active_low = getattr(r_obj, "active_low", True)
            r_val = "0" if is_active_low else "1"
            rst_cond_label = f" : {r_sig} == {r_val}"

        init_st = fsm.initial_state or (fsm.states[0].name if fsm.states and hasattr(fsm.states[0], "name") else "IDLE")
        init_alias = to_plantuml_alias(init_st)
        fsm_lines.append(f"  [*] --> {init_alias}{rst_cond_label}")

        # Render states with per-state outputs
        for st in (fsm.states or []):
            st_name = st.name if hasattr(st, "name") else str(st)
            st_alias = to_plantuml_alias(st_name)
            outputs = getattr(st, "outputs", {}) or {}
            if outputs:
                fsm_lines.append(f'  state "{st_name}" as {st_alias} {{')
                for sig_name, sig_val in outputs.items():
                    fsm_lines.append(f'    {st_alias} : {sig_name} = {sig_val}')
                fsm_lines.append('  }')
            else:
                fsm_lines.append(f'  state "{st_name}" as {st_alias}')

        # Render transitions with explicit guards
        for t in (fsm.transitions or []):
            src_alias = to_plantuml_alias(t.source_state)
            tgt_alias = to_plantuml_alias(t.target_state)
            cond = t.condition
            cond_expr = cond.get("expression") if isinstance(cond, dict) else str(cond or "")
            cond_clean = cond_expr.replace("\n", " ").strip()
            label = f" : [{cond_clean}]" if cond_clean else ""
            fsm_lines.append(f"  {src_alias} --> {tgt_alias}{label}")
            fsm_has_transitions = True

    if not fsm_has_transitions:
        plan_fsm = plan.get("fsm_architecture", {})
        states_list = [s.get("name") for s in plan_fsm.get("states", []) if isinstance(s, dict) and s.get("name")]
        plan_transitions = plan_fsm.get("transitions", [])
        
        if states_list and plan_transitions:
            fsm_lines.append(f"  title {primary_comp_name} Controller FSM")
            init_alias = to_plantuml_alias(states_list[0])
            fsm_lines.append(f"  [*] --> {init_alias}")
            for tr in plan_transitions:
                src_alias = to_plantuml_alias(tr.get("from", states_list[0]))
                tgt_alias = to_plantuml_alias(tr.get("to", states_list[0]))
                cond = tr.get("condition", "trigger")
                fsm_lines.append(f"  {src_alias} --> {tgt_alias} : {cond}")
                fsm_has_transitions = True

    if not fsm_has_transitions:
        fsm_lines = [
            "@startuml",
            f"  title {primary_comp_name} Controller FSM",
            '  note "INSUFFICIENT_DIAGRAM_CONTEXT: Explicit FSM states and transitions not specified in source." as N_fsm',
            "@enduml"
        ]
    else:
        fsm_lines.append("@enduml")

    # 3. Sequence Diagram Construction (Strictly Grounded; No Fabricated Protocol Flows)
    seq_lines = ["@startuml", f"  title {primary_comp_name} Transaction Flow Sequence", "  autonumber"]
    has_sequence_content = False
    
    if efs_ir and efs_ir.components:
        comps = efs_ir.components
        c_src_alias = to_plantuml_alias(comps[0].name)

        # Priority 1: Use explicit flows from EFS IR
        if hasattr(efs_ir, "flows") and efs_ir.flows:
            for c in comps:
                c_alias = registry.register(c.name, canonical_id=c.component_id, entity_type=c.type)
                seq_lines.append(f'  participant "{c.name}" as {c_alias}')
            seq_lines.append("")
            seq_lines.insert(3, '  actor "Host Controller" as Host')
            for flow in efs_ir.flows[:2]:
                for step in (flow.ordered_steps or [])[:6]:
                    action = getattr(step, "action_description", "") or ""
                    comp_name = getattr(step, "component", "") or comps[0].name
                    comp_alias = registry.get_alias(comp_name) or to_plantuml_alias(comp_name)
                    seq_lines.append(f"  Host -> {comp_alias} : {action[:60]}")
                    has_sequence_content = True

        # Priority 2: Use explicit transactions from EFS IR
        if not has_sequence_content and efs_ir.transactions:
            for c in comps:
                c_alias = registry.register(c.name, canonical_id=c.component_id, entity_type=c.type)
                seq_lines.append(f'  participant "{c.name}" as {c_alias}')
            seq_lines.append("")
            seq_lines.insert(3, '  actor "Host Controller" as Host')
            for tx in efs_ir.transactions[:5]:
                src = getattr(tx, "initiator", "") or getattr(tx, "source", "") or "Host"
                tgt = getattr(tx, "target", "") or getattr(tx, "destination", "") or comps[0].name
                action = getattr(tx, "name", "") or getattr(tx, "type", "") or getattr(tx, "description", "Transfer")
                src_alias = registry.get_alias(src) or "Host"
                tgt_alias = registry.get_alias(tgt) or c_src_alias
                seq_lines.append(f"  {src_alias} -> {tgt_alias} : {action[:60]}")
                has_sequence_content = True

    if not has_sequence_content:
        seq_lines = [
            "@startuml",
            f"  title {primary_comp_name} Transaction Flow Sequence",
            '  note "Sequence flow not specified in EFS IR for this design." as N_seq',
            "@enduml"
        ]
    else:
        seq_lines.append("@enduml")

    raw_diagrams = {
        "architecture": "\n".join(arch_lines),
        "fsm": "\n".join(fsm_lines),
        "sequence": "\n".join(seq_lines)
    }
    return raw_diagrams




def generate_diagrams(design_spec: str, plan: Dict[str, Any], efs_ir: Optional[EFSIR] = None) -> Dict[str, str]:
    """Generate PlantUML diagrams from the compiled Diagram Context with full syntax validation."""
    logger.info("Generating PlantUML diagrams with Artifact-Specific Context Compiler...")
    llm = get_llm_client()
    
    from core.context_compiler import (
        compile_context,
        CompilationMetrics,
        record_generation_telemetry,
        estimate_tokens
    )

    # Compile artifact-specific compact diagram context
    compiled_context = compile_context(
        "diagram",
        efs_ir=efs_ir,
        design_plan=plan,
        design_spec=design_spec
    )
    raw_metrics = compiled_context.pop("_metrics", {})

    user_prompt = f"""
=== COMPILED HARDWARE DIAGRAM CONTEXT ===
{json.dumps(compiled_context, indent=2)}

CRITICAL REQUIREMENT FOR ALL DIAGRAMS:
- Architecture: Show components, interfaces, ports, registers, memories, and explicit connections.
- FSM: Show states, initial state, and transitions with explicit conditions.
- Sequence: Every participant and signal interaction MUST use canonical human-readable names from the context, NOT internal IDs.
- Syntax: Every alias must be an identifier without spaces (e.g. participant "Name" as Valid_Alias).
"""

    sys_tokens = estimate_tokens(SYSTEM_PROMPT)
    input_tokens = sys_tokens + estimate_tokens(user_prompt)

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

    # Post-process all diagrams to guarantee any remaining signal/component/interface IDs are replaced with safe aliases
    resolved = resolve_diagram_ids_to_names(diagrams, efs_ir=efs_ir)

    # Record Observability / Telemetry metrics
    output_tokens = (llm.last_usage.get("completion_tokens") if (hasattr(llm, "last_usage") and llm.last_usage.get("completion_tokens")) else sum(estimate_tokens(v) for v in resolved.values()))
    provider_tokens = llm.last_usage.get("prompt_tokens") if hasattr(llm, "last_usage") else None
    orig_tokens = raw_metrics.get("original_context_tokens", input_tokens)
    comp_tokens = raw_metrics.get("compiled_context_tokens", estimate_tokens(compiled_context))

    logger.info(
        f"[GenerationContext] artifact=diagram | "
        f"original_tokens={orig_tokens} | "
        f"compiled_tokens={comp_tokens} | "
        f"final_prompt_tokens={estimate_tokens(user_prompt)} | "
        f"final_input_tokens={input_tokens} | "
        f"provider_input_tokens={provider_tokens or 'N/A'} | "
        f"output_tokens={output_tokens}"
    )

    metrics = CompilationMetrics(
        artifact_type="diagram",
        compiler_used="DiagramContextCompiler",
        original_context_tokens=orig_tokens,
        compiled_context_tokens=comp_tokens,
        system_prompt_tokens=sys_tokens,
        final_input_tokens=input_tokens,
        output_tokens=output_tokens,
        budget_exceeded=raw_metrics.get("budget_exceeded", False),
        overflow_warning=raw_metrics.get("overflow_warning"),
        extra_details={"provider_input_tokens": provider_tokens}
    )
    record_generation_telemetry(metrics, generation_name="plantuml_generation", model=CONFIG.llm.code_model)

    return resolved
