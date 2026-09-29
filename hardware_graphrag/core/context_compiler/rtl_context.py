"""
RTL Context Compiler module.

Compiles a rich, target-scoped context specifically for RTL generation (SystemVerilog, Verilog, VHDL).
Extracts all required behavioral semantics (ports, clocks, resets, registers, FSM transitions,
conditions, datapath operations, interface handshakes, and scoped protocol rules).

CRITICAL RULE:
Token reduction MUST NOT remove behavioral semantics.
REMOVE IRRELEVANCE, NOT REQUIRED BEHAVIOR.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set
from config import CONFIG
from core.efs_ir.models import EFSIR, EFSComponent
from core.context_compiler.base import (
    BaseContextCompiler, CompilationMetrics, estimate_tokens, register_compiler
)
from core.context_compiler.resolver import ContextResolver
from core.context_compiler.validator import GroundingValidator
from core.context_compiler.budget import TokenBudgetController
from utils.logger import get_logger

logger = get_logger("context_compiler.rtl")


@register_compiler
class RTLContextCompiler(BaseContextCompiler):
    """Compiles a compact, rich behavioral context tailored strictly to the target RTL module."""

    artifact_types = ["rtl", "verilog", "systemverilog", "vhdl"]

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
            "rtl_token_budget",
            8000
        )

        metrics = CompilationMetrics(
            artifact_type="rtl",
            compiler_used="RTLContextCompiler"
        )

        # 1. Target Component Identification
        target_name = kwargs.get("target_component") or kwargs.get("target") or ""
        if not target_name and design_plan and isinstance(design_plan, dict):
            target_name = design_plan.get("target") or design_plan.get("module_name") or ""
        if not target_name and efs_ir and efs_ir.metadata and efs_ir.metadata.design_name:
            target_name = efs_ir.metadata.design_name

        resolver = ContextResolver(efs_ir)

        target_comp: Optional[EFSComponent] = None
        if efs_ir and efs_ir.components:
            if target_name:
                t_lower = target_name.lower()
                target_comp = next((c for c in efs_ir.components if (c.name and c.name.lower() == t_lower) or c.component_id == target_name), None)
            if not target_comp:
                # Default to top-level or primary component
                target_comp = next((c for c in efs_ir.components if c.type in ("top_level", "top_level_design")), efs_ir.components[0])

        mod_name = target_comp.name if target_comp and target_comp.name else (target_name or "hardware_module")

        # Measure original context size
        orig_tokens = 0
        if efs_ir:
            orig_tokens += estimate_tokens(efs_ir)
        if design_plan:
            orig_tokens += estimate_tokens(design_plan)
        design_spec = kwargs.get("design_spec", "")
        if design_spec:
            orig_tokens += estimate_tokens(design_spec)
        metrics.original_context_tokens = max(orig_tokens, 1)

        # 2. Extract Ports and Directly Relevant Signals
        ports: List[Dict[str, Any]] = []
        signals: List[Dict[str, Any]] = []
        clock_reset: List[Dict[str, Any]] = []
        seen_port_names: Set[str] = set()

        scoped_signals = resolver.get_component_signals(target_comp) if target_comp else (efs_ir.signals if efs_ir else [])

        for s in scoped_signals:
            s_name = resolver.resolve_signal_name(s.name or s.signal_id)
            if not s_name or s_name.lower() in seen_port_names:
                continue
            seen_port_names.add(s_name.lower())

            direction = s.direction or "input"
            width = str(s.width or 1)
            role = s.semantic_role or "data"

            is_clk_rst = role in ("clock", "reset") or any(k in s_name.lower() for k in ("clk", "rst", "clock", "reset"))
            if is_clk_rst:
                clock_reset.append({
                    "name": s_name,
                    "type": "clock" if "clk" in s_name.lower() or role == "clock" else "reset",
                    "direction": direction,
                    "polarity": "active_low" if "n" in s_name.lower() or "rst_n" in s_name.lower() else "active_high"
                })

            port_item = {
                "name": s_name,
                "direction": direction,
                "width": width,
                "semantic_role": role,
                "description": (s.description or "")[:80],
                "interface": resolver.resolve_interface_name(s.interface or "")
            }
            ports.append(port_item)
            signals.append(port_item)

        # If ports list is empty, synthesize standard clock/reset safely
        if not ports:
            ports = [
                {"name": "clk", "direction": "input", "width": "1", "semantic_role": "clock"},
                {"name": "rst_n", "direction": "input", "width": "1", "semantic_role": "reset"}
            ]
            clock_reset = [
                {"name": "clk", "type": "clock", "direction": "input", "polarity": "rising"},
                {"name": "rst_n", "type": "reset", "direction": "input", "polarity": "active_low"}
            ]

        # 3. Extract Registers & Fields (Preserve All Explicit Bit Meanings)
        registers: List[Dict[str, Any]] = []
        scoped_registers = resolver.get_component_registers(target_comp) if target_comp else (efs_ir.registers if efs_ir else [])

        for r in scoped_registers:
            r_name = resolver.id_to_name.get(r.register_id, r.name)
            fields: List[Dict[str, Any]] = []
            for fld in (r.fields or []):
                msb = fld.msb if fld.msb is not None else 0
                lsb = fld.lsb if fld.lsb is not None else 0
                bit_range = f"{msb}:{lsb}" if msb != lsb else str(msb)
                fields.append({
                    "name": fld.name,
                    "bits": bit_range,
                    "meaning": fld.description or fld.name,
                    "reset_value": getattr(fld, "reset_value", "0")
                })

            registers.append({
                "name": r_name,
                "offset": r.offset or "0x0",
                "width": str(r.width or 32),
                "reset_value": getattr(r, "reset_val", None) or getattr(r, "reset_value", None) or "0x0",
                "access": getattr(r, "access", "RW"),
                "description": (r.description or "")[:80],
                "fields": fields
            })

        # 4. Extract FSM (States, Transitions, Conditions, Actions, Alternate Branches, Timeouts)
        fsm_data: Dict[str, Any] = {}
        target_fsm = resolver.get_component_fsm(target_comp) if target_comp else None

        if target_fsm:
            states_list = [st.name if hasattr(st, "name") else str(st) for st in (target_fsm.states or [])]
            transitions_list: List[Dict[str, Any]] = []
            
            # Count outgoing transitions per source state to detect alternate branches
            out_degrees: Dict[str, int] = {}
            for tr in (target_fsm.transitions or []):
                s_st = tr.source_state
                out_degrees[s_st] = out_degrees.get(s_st, 0) + 1

            for tr in (target_fsm.transitions or []):
                cond = tr.condition
                cond_expr = cond.get("expression") if isinstance(cond, dict) else str(cond or "1'b1")
                cond_clean = resolver.scrub_internal_ids(cond_expr)
                action_clean = resolver.scrub_internal_ids(getattr(tr, "action", "") or "")
                
                tr_data: Dict[str, Any] = {
                    "from": tr.source_state,
                    "to": tr.target_state,
                    "condition": {
                        "expression": cond_clean,
                        "signals": [resolver.resolve_signal_name(s) for s in getattr(tr, "signals", [])],
                        "source": "EFS"
                    }
                }
                if action_clean:
                    tr_data["action"] = action_clean
                if getattr(tr, "priority", 0):
                    tr_data["priority"] = tr.priority
                if out_degrees.get(tr.source_state, 0) > 1:
                    tr_data["is_alternate_branch"] = True

                transitions_list.append(tr_data)

            fsm_data = {
                "name": target_fsm.name or "Primary_FSM",
                "initial_state": target_fsm.initial_state or (states_list[0] if states_list else "IDLE"),
                "states": states_list,
                "transitions": transitions_list,
                "has_alternate_branches": any(deg > 1 for deg in out_degrees.values())
            }
        elif kwargs.get("context_pack"):
            cp = kwargs.get("context_pack")
            cp_fsm = getattr(cp, "fsm", None) if not isinstance(cp, dict) else cp.get("fsm")
            if cp_fsm:
                fsm_data = cp_fsm if isinstance(cp_fsm, dict) else getattr(cp_fsm, "__dict__", {})
        elif efs_ir and efs_ir.fsms:
            fallback_fsm = efs_ir.fsms[0]
            st_list = [st.name if hasattr(st, "name") else str(st) for st in (fallback_fsm.states or [])]
            tr_list = []
            for tr in (fallback_fsm.transitions or []):
                cond = tr.condition
                cond_expr = cond.get("expression") if isinstance(cond, dict) else str(cond or "1'b1")
                tr_list.append({
                    "from": tr.source_state,
                    "to": tr.target_state,
                    "condition": {
                        "expression": resolver.scrub_internal_ids(cond_expr),
                        "source": "EFS"
                    }
                })
            fsm_data = {
                "name": fallback_fsm.name or "Primary_FSM",
                "initial_state": fallback_fsm.initial_state or (st_list[0] if st_list else "IDLE"),
                "states": st_list,
                "transitions": tr_list,
                "has_alternate_branches": False
            }

        # 5. Dependency-Based Semantic Relevance Selection
        # (Preserves opcodes, instructions, commands, transactions, handshakes, memory ops, timeouts, parameters)
        from core.context_compiler.semantic_selector import SemanticDependencySelector
        sem_selector = SemanticDependencySelector(efs_ir, resolver)
        sem_deps = sem_selector.select_dependencies(
            target_comp=target_comp,
            artifact_type="rtl",
            scoped_signals=scoped_signals,
            target_fsm=target_fsm
        )

        # Merge datapath operations from functional behavior
        datapath_operations: List[Dict[str, Any]] = list(sem_deps.datapath_operations)
        if efs_ir and hasattr(efs_ir, "functional_behavior") and efs_ir.functional_behavior:
            for fb in efs_ir.functional_behavior:
                subj = getattr(fb, "subject", "") or getattr(fb, "target_component", "")
                target_match = not subj or subj.lower() == mod_name.lower()
                if target_match:
                    desc = getattr(fb, "description", "")
                    expr = getattr(fb, "action", "") or getattr(fb, "expression", "") or getattr(fb, "behavior", "") or desc
                    dp_item = {
                        "name": getattr(fb, "name", "op"),
                        "description": resolver.scrub_internal_ids(desc),
                        "expression": resolver.scrub_internal_ids(expr)
                    }
                    if not any(d.get("name") == dp_item["name"] for d in datapath_operations):
                        datapath_operations.append(dp_item)

        # 6. Sliced Protocol Rules (strictly relevant to this module's interfaces)
        target_interfaces = set()
        if target_comp:
            for iface_ref in (target_comp.interfaces or []):
                target_interfaces.add(iface_ref.lower())
                i_obj = resolver.interface_by_id.get(iface_ref) or resolver.interface_by_name.get(iface_ref.lower())
                if i_obj:
                    if i_obj.name:
                        target_interfaces.add(i_obj.name.lower())
                    if i_obj.protocol:
                        target_interfaces.add(i_obj.protocol.lower())

        relevant_protocol_rules: List[Dict[str, Any]] = []
        all_rules = []
        if isinstance(protocol_contract, list):
            all_rules.extend(protocol_contract)
        elif isinstance(protocol_contract, dict):
            all_rules.extend(protocol_contract.get("rules", []))
        elif efs_ir and hasattr(efs_ir, "protocol_rules"):
            all_rules.extend([r.to_dict() if hasattr(r, "to_dict") else r for r in (efs_ir.protocol_rules or [])])

        seen_rule_texts = set()
        for rule in all_rules:
            r_name = rule.get("name", "")
            r_text = rule.get("text", "") or rule.get("rule_statement", "") or rule.get("description", "") or r_name
            r_citation = rule.get("citation", "")
            r_iface = (rule.get("interface", "") or rule.get("protocol", "")).lower()

            # Match on interface protocol names, rule names, or port signal stems
            port_stems = []
            for p in ports:
                p_raw = p["name"].lower()
                port_stems.append(p_raw)
                for pfx in ("m_axi_", "s_axi_", "axi_", "m_", "s_", "in_", "out_"):
                    if p_raw.startswith(pfx):
                        port_stems.append(p_raw[len(pfx):])

            is_relevant = (
                not target_interfaces or
                any(ti in r_iface for ti in target_interfaces) or
                any(ti in r_name.lower() for ti in target_interfaces) or
                any(ti in r_text.lower() for ti in target_interfaces) or
                any(stem in r_text.lower() for stem in port_stems if len(stem) > 3)
            )

            if is_relevant and r_text and r_text not in seen_rule_texts:
                seen_rule_texts.add(r_text)
                relevant_protocol_rules.append({
                    "text": resolver.scrub_internal_ids(r_text),
                    "citation": r_citation
                })

        # 7. Sliced Design Plan (Module-specific implementation constraints)
        filtered_plan = {}
        if design_plan and isinstance(design_plan, dict):
            for k in ("architecture_overview", "datapath_strategy", "reset_polarity", "fsm", "parameters"):
                if k in design_plan:
                    filtered_plan[k] = design_plan[k]

        # 8. Source Traceability References
        source_refs: List[str] = []
        if target_comp:
            source_refs.extend(resolver.get_source_refs(target_comp))
        for p in scoped_signals:
            source_refs.extend(resolver.get_source_refs(p))

        # Build Minimum Sufficient Compiled RTL Context
        compiled_context = {
            "module": {
                "name": mod_name,
                "type": target_comp.type if target_comp else "module",
                "description": (target_comp.description if target_comp else "") or f"Synthesizable module {mod_name}",
            },
            "parameters": sem_deps.parameters,
            "ports": ports,
            "clock_reset": clock_reset,
            "signals": signals,
            "registers": registers,
            "fsm": fsm_data,
            "datapath": datapath_operations,
            "protocol_rules": relevant_protocol_rules,
            "design_plan": filtered_plan,
            "source_refs": list(dict.fromkeys(source_refs))[:15]
        }

        # Inject semantically critical constructs whenever present by dependency
        if sem_deps.opcodes:
            compiled_context["opcodes"] = sem_deps.opcodes
        if sem_deps.instructions:
            compiled_context["instructions"] = sem_deps.instructions
        if sem_deps.instruction_formats:
            compiled_context["instruction_formats"] = sem_deps.instruction_formats
        if sem_deps.transactions:
            compiled_context["relevant_transactions"] = sem_deps.transactions
        if sem_deps.commands:
            compiled_context["commands"] = sem_deps.commands
        if sem_deps.memory_operations:
            compiled_context["memory_operations"] = sem_deps.memory_operations
        if sem_deps.handshakes_and_constraints:
            compiled_context["handshakes_and_constraints"] = sem_deps.handshakes_and_constraints
        if sem_deps.ordering_and_timeouts:
            compiled_context["ordering_and_timeouts"] = sem_deps.ordering_and_timeouts
        if sem_deps.flows_and_branches:
            compiled_context["flows_and_branches"] = sem_deps.flows_and_branches
        if sem_deps.errors_and_interrupts:
            compiled_context["errors_and_interrupts"] = sem_deps.errors_and_interrupts

        # 9. Grounding Validation
        validation = GroundingValidator.validate_rtl_context(compiled_context, efs_ir)
        if not validation.is_valid:
            logger.warning(f"RTL context grounding violations: {validation.violations}")

        # 10. Priority-Based Token Budget Enforcement
        pruned_context, exceeded, warning = TokenBudgetController.enforce_rtl_budget(
            compiled_context, max_tokens=max_budget
        )

        metrics.compiled_context_tokens = estimate_tokens(pruned_context)
        metrics.budget_exceeded = exceeded
        metrics.overflow_warning = warning
        metrics.calculate_reduction()

        pruned_context["_metrics"] = metrics.to_dict()
        return pruned_context
