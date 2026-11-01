"""
Diagram Context Compiler module.

Compiles a compact, strictly-grounded context representation for PlantUML diagram generation:
- Architecture diagram context
- FSM diagram context
- Sequence diagram context

Filters out unrelated raw specification text, massive transaction object dumps,
full contradiction/conflict dumps, and enforces canonical human-readable naming.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from config import CONFIG
from core.efs_ir.models import EFSIR
from core.context_compiler.base import (
    BaseContextCompiler, CompilationMetrics, estimate_tokens, register_compiler
)
from core.context_compiler.resolver import ContextResolver
from core.context_compiler.validator import GroundingValidator
from core.context_compiler.budget import TokenBudgetController
from utils.logger import get_logger

logger = get_logger("context_compiler.diagram")


@register_compiler
class DiagramContextCompiler(BaseContextCompiler):
    """Compiles compact, grounded context specifically for PlantUML Architecture, FSM, and Sequence diagrams."""

    artifact_types = ["diagram", "plantuml", "puml", "architecture", "fsm", "sequence"]

    def compile(
        self,
        efs_ir: Optional[EFSIR] = None,
        design_plan: Optional[Dict[str, Any]] = None,
        protocol_contract: Optional[Any] = None,
        retrieval_context: Optional[Any] = None,
        token_budget: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        max_budget = token_budget or getattr(
            getattr(CONFIG, "context_compiler", None),
            "plantuml_token_budget",
            4000
        )

        metrics = CompilationMetrics(
            artifact_type="diagram",
            compiler_used="DiagramContextCompiler"
        )

        # Measure original oversized context size (simulate raw dumps)
        orig_tokens = 0
        if efs_ir:
            orig_tokens += estimate_tokens(efs_ir)
        if design_plan:
            orig_tokens += estimate_tokens(design_plan)
        design_spec = kwargs.get("design_spec", "")
        if design_spec:
            orig_tokens += estimate_tokens(design_spec)
        metrics.original_context_tokens = max(orig_tokens, 1)

        resolver = ContextResolver(efs_ir)

        # 1. Architecture Context Construction
        arch_components: List[Dict[str, Any]] = []
        arch_interfaces: List[Dict[str, Any]] = []
        arch_signals: List[Dict[str, Any]] = []
        arch_registers: List[Dict[str, Any]] = []
        arch_memories: List[Dict[str, Any]] = []
        arch_connections: List[Dict[str, Any]] = []

        if efs_ir:
            for c in (efs_ir.components or []):
                c_name = resolver.resolve_component_name(c.name or c.component_id)
                arch_components.append({
                    "name": c_name,
                    "type": c.type or "module",
                    "description": (c.description or "")[:80],
                    "clock_domain": c.clock_domain or "",
                    "reset_domain": c.reset_domain or "",
                    "is_top_level": (c.type in ("top_level", "top_level_design") or c_name == getattr(efs_ir, "design_name", getattr(efs_ir.metadata, "design_name", "")))
                })

            for iface in (efs_ir.interfaces or []):
                i_name = resolver.resolve_interface_name(iface.name or iface.interface_id)
                src_comp = resolver.resolve_component_name(iface.source_component or "")
                tgt_comp = resolver.resolve_component_name(getattr(iface, "target_component", "") or "")
                arch_interfaces.append({
                    "name": i_name,
                    "protocol": iface.protocol or "Custom",
                    "role": iface.role or "unspecified",
                    "source": src_comp,
                    "target": tgt_comp,
                    "signals": [resolver.resolve_signal_name(s) for s in (iface.signals or [])]
                })
                if src_comp and tgt_comp:
                    arch_connections.append({
                        "from": src_comp,
                        "to": tgt_comp,
                        "interface": i_name,
                        "protocol": iface.protocol
                    })

            for s in (efs_ir.signals or []):
                s_name = resolver.resolve_signal_name(s.name or s.signal_id)
                s_owner = resolver.resolve_component_name(s.owner or "")
                s_cons = resolver.resolve_component_name(s.consumer or "")
                arch_signals.append({
                    "name": s_name,
                    "direction": s.direction or "internal",
                    "width": str(s.width or 1),
                    "semantic_role": s.semantic_role or "data",
                    "owner": s_owner,
                    "consumer": s_cons
                })
                if s_owner and s_cons and s_owner != s_cons:
                    arch_connections.append({
                        "from": s_owner,
                        "to": s_cons,
                        "signal": s_name,
                        "role": s.semantic_role or "data"
                    })

            for r in (efs_ir.registers or []):
                arch_registers.append({
                    "name": resolver.id_to_name.get(r.register_id, r.name),
                    "offset": r.offset or "0x0",
                    "width": r.width or 32,
                    "owner": resolver.resolve_component_name(r.owner or "")
                })

            for m in (getattr(efs_ir, "memory_regions", []) or []):
                arch_memories.append({
                    "name": getattr(m, "name", "Memory_Block"),
                    "base_address": getattr(m, "base_address", "0x0"),
                    "size": getattr(m, "size", "N/A"),
                    "access": getattr(m, "access", "RW")
                })

            arch_parameters = [
                {"name": p.name, "value": p.value or p.default_value}
                for p in (getattr(efs_ir, "parameters", []) or [])[:10]
            ]
        else:
            arch_parameters = []

        # 2. FSM Context Construction (Enriched with actions, priorities, reset transition, per-state outputs)
        fsm_blocks: List[Dict[str, Any]] = []
        if efs_ir and efs_ir.fsms:
            # Determine reset transition from declared reset definition
            rst_cond_str = ""
            if hasattr(efs_ir, "reset") and efs_ir.reset and getattr(efs_ir.reset, "signal", None):
                rst_sig = resolver.resolve_signal_name(efs_ir.reset.signal)
                r_desc = (getattr(efs_ir.reset, "description", "") or "").lower()
                is_active_low = False if ("active_high" in r_desc or "active-high" in r_desc) else getattr(efs_ir.reset, "active_low", True)
                rst_val = "0" if is_active_low else "1"
                rst_cond_str = f"{rst_sig} == {rst_val}"
            elif getattr(efs_ir, "resets", None):
                r_obj = efs_ir.resets[0]
                rst_sig = resolver.resolve_signal_name(r_obj.name or r_obj.signal_id)
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
                rst_val = "0" if is_active_low else "1"
                rst_cond_str = f"{rst_sig} == {rst_val}"

            for f in efs_ir.fsms:
                f_name = f.name or f"{primary_comp_name}_fsm" if 'primary_comp_name' in locals() else (f.name or "Primary_FSM")
                f_owner = resolver.resolve_component_name(getattr(f, "owner", "") or "")
                f_states = []
                for st in (f.states or []):
                    st_name = st.name if hasattr(st, "name") else str(st)
                    if st_name:
                        st_dict: Dict[str, Any] = {
                            "name": st_name,
                            "outputs": getattr(st, "outputs", {}) or {},
                            "duration_cycles": getattr(st, "duration_cycles", None)
                        }
                        f_states.append(st_dict)

                f_transitions = []
                out_degrees: Dict[str, int] = {}
                for tr in (f.transitions or []):
                    out_degrees[tr.source_state] = out_degrees.get(tr.source_state, 0) + 1

                for tr in (f.transitions or []):
                    cond = tr.condition
                    cond_str = cond.get("expression") if isinstance(cond, dict) else str(cond or "")
                    cond_clean = resolver.scrub_internal_ids(cond_str)
                    act_clean = resolver.scrub_internal_ids(getattr(tr, "action", "") or "")
                    
                    tr_dict: Dict[str, Any] = {
                        "from": tr.source_state,
                        "to": tr.target_state,
                        "condition": cond_clean
                    }
                    if act_clean:
                        tr_dict["action"] = act_clean
                    if getattr(tr, "priority", 0):
                        tr_dict["priority"] = tr.priority
                    if out_degrees.get(tr.source_state, 0) > 1:
                        tr_dict["is_alternate_branch"] = True

                    f_transitions.append(tr_dict)

                initial_st_name = f.initial_state or (f_states[0]["name"] if f_states else "IDLE")
                fsm_blocks.append({
                    "name": f_name,
                    "owner": f_owner,
                    "initial_state": initial_st_name,
                    "reset_condition": rst_cond_str,
                    "states": f_states,
                    "transitions": f_transitions,
                    "has_alternate_branches": any(d > 1 for d in out_degrees.values())
                })

        from core.agents.plantuml_utils import resolve_canonical_design_name, validate_interaction

        # 3. Sequence Context Construction
        # Grounded participants: unique component names from EFS IR
        participants: List[str] = []
        if efs_ir and efs_ir.components:
            for c in efs_ir.components:
                c_name = resolver.resolve_component_name(c.name)
                if c_name and c_name.lower() not in ("target_module", "target module", "unnamed_design"):
                    if c_name not in participants:
                        participants.append(c_name)

        if not participants:
            resolved_primary = resolve_canonical_design_name(
                efs_ir=efs_ir,
                design_plan=design_plan,
                design_spec=design_spec
            )
            if resolved_primary:
                participants = [resolved_primary, "Host"]
            elif design_plan and design_plan.get("target"):
                participants = [design_plan.get("target"), "Host"]
            else:
                participants = ["Host", "Controller"]

        valid_part_set = set(participants)
        valid_part_set.update({"Host", "Master", "Slave", "Client", "Controller", "Environment"})

        # Curated representative transaction flow (from flows, subflows, and transactions)
        representative_interactions: List[Dict[str, Any]] = []
        seen_actions = set()

        # First priority: explicit EFSFlow flows and subflows
        if efs_ir and hasattr(efs_ir, "flows") and efs_ir.flows:
            for fl in efs_ir.flows:
                for step in (fl.ordered_steps or [])[:6]:
                    action_txt = resolver.scrub_internal_ids(step.action_description)
                    if action_txt and action_txt not in seen_actions:
                        src = resolver.resolve_component_name(step.component or (participants[0] if participants else "Host"))
                        tgt = resolver.resolve_component_name(participants[1] if len(participants) > 1 else participants[0])
                        step_dict: Dict[str, Any] = {
                            "from": src,
                            "to": tgt,
                            "action": action_txt,
                            "condition": resolver.scrub_internal_ids(step.condition or ""),
                            "timing": step.timing_expectations or ""
                        }
                        is_valid, reason = validate_interaction(step_dict, valid_part_set)
                        if is_valid:
                            seen_actions.add(action_txt)
                            representative_interactions.append(step_dict)
                        else:
                            logger.debug(f"Filtered invalid sequence step: {reason} ({step_dict})")

        # Second priority: curated representative transactions
        if efs_ir and efs_ir.transactions and len(representative_interactions) < 8:
            for tx in efs_ir.transactions:
                src = resolver.resolve_component_name(getattr(tx, "initiator", "") or getattr(tx, "source", "") or (participants[0] if participants else "Host"))
                tgt = resolver.resolve_component_name(getattr(tx, "target", "") or getattr(tx, "destination", "") or (participants[1] if len(participants) > 1 else participants[0]))
                action = getattr(tx, "name", "") or getattr(tx, "type", "") or getattr(tx, "description", "")
                action_clean = resolver.scrub_internal_ids(action)
                if action_clean and action_clean not in seen_actions:
                    tx_dict = {
                        "from": src,
                        "to": tgt,
                        "action": action_clean,
                        "signals": [resolver.resolve_signal_name(s) for s in getattr(tx, "signals", [])],
                        "ordering": tx.ordering or "in-order",
                        "response": tx.response or ""
                    }
                    is_valid, reason = validate_interaction(tx_dict, valid_part_set)
                    if is_valid:
                        seen_actions.add(action_clean)
                        representative_interactions.append(tx_dict)
                    else:
                        logger.debug(f"Filtered fragmented or invalid transaction: {reason} ({tx_dict})")
                if len(representative_interactions) >= 10:
                    break

        if not representative_interactions and arch_interfaces:
            # Derive grounded sequence from protocol interface handshake signals only if explicit protocol is defined
            for iface in arch_interfaces[:3]:
                proto = (iface.get("protocol") or "").lower()
                if proto and proto not in ("custom", "generic", "unspecified", "none"):
                    src = iface.get("source") or (participants[0] if participants else "Master")
                    tgt = iface.get("target") or (participants[1] if len(participants) > 1 else "Slave")
                    sigs = iface.get("signals", [])
                    for sig in sigs[:3]:
                        step_dict = {
                            "from": src,
                            "to": tgt,
                            "action": f"Assert {sig}",
                            "signals": [sig]
                        }
                        is_valid, _ = validate_interaction(step_dict, valid_part_set)
                        if is_valid:
                            representative_interactions.append(step_dict)

        # 4. Filtered Design Plan (Diagram-relevant only)
        filtered_plan = {}
        if design_plan and isinstance(design_plan, dict):
            for k in ("target", "architecture_overview", "interfaces", "state_machine", "clock_reset_plan"):
                if k in design_plan:
                    filtered_plan[k] = design_plan[k]

        primary_module_resolved = resolve_canonical_design_name(
            efs_ir=efs_ir,
            design_plan=design_plan,
            design_spec=design_spec
        ) or (participants[0] if participants else "Hardware_Module")

        compiled_context = {
            "diagram_type": "plantuml_triple",
            "primary_module": primary_module_resolved,
            "architecture": {
                "components": arch_components,
                "interfaces": arch_interfaces,
                "signals": arch_signals,
                "registers": arch_registers,
                "memories": arch_memories,
                "parameters": arch_parameters,
                "connections": arch_connections,
                "topology_status": "EXPLICIT" if arch_connections else "INSUFFICIENT_TOPOLOGY_CONTEXT"
            },
            "fsm": fsm_blocks[0] if fsm_blocks else {
                "status": "INSUFFICIENT_DIAGRAM_CONTEXT",
                "message": "Explicit FSM states and transitions not provided in design context."
            },
            "all_fsms": fsm_blocks,
            "sequence": {
                "participants": list(dict.fromkeys(participants)) if representative_interactions else [],
                "interactions": representative_interactions,
                "not_specified": len(representative_interactions) == 0,
                "signals": [s["name"] for s in arch_signals if s.get("name")]
            },
            "design_plan": filtered_plan
        }

        # 5. Grounding Validation
        validation = GroundingValidator.validate_diagram_context(compiled_context, efs_ir)
        if not validation.is_valid:
            logger.warning(f"Diagram context grounding violations: {validation.violations}")

        # 6. Priority-Based Token Budget Enforcement
        pruned_context, exceeded, warning = TokenBudgetController.enforce_diagram_budget(
            compiled_context, max_tokens=max_budget
        )

        metrics.compiled_context_tokens = estimate_tokens(pruned_context)
        metrics.budget_exceeded = exceeded
        metrics.overflow_warning = warning
        metrics.calculate_reduction()

        pruned_context["_metrics"] = metrics.to_dict()
        return pruned_context
