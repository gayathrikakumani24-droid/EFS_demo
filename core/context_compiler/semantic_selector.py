"""
Semantic Dependency Selector module for the EFS Context Compiler.

Implements Dependency-Based Relevance Selection rather than field-based whitelisting.
Selects semantically critical constructs whenever they are relevant to the target
artifact, achieving MINIMUM SUFFICIENT CONTEXT (not minimum possible context).

Preserves:
- opcodes, instruction fields, instruction formats
- commands, messages, transactions, transaction types
- protocol operations, events, handshakes
- conditions, actions, timeouts, alternate branches
- subflows, sequence dependencies, ordering constraints, out-of-order behavior
- datapath operations, memory operations
- interrupt behavior, error conditions
- configuration fields, parameters
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from core.efs_ir.models import (
    EFSIR, EFSComponent, EFSInterface, EFSSignal, EFSRegister, EFSFSM,
    EFSTransaction, EFSConstraint, EFSTimingRule, EFSProtocolRule,
    EFSInstruction, EFSOpcode, EFSInstructionFormat, EFSMemoryRegion,
    EFSParameter, EFSFunctionalBehavior, EFSError, EFSFlow, EFSState, EFSTransition
)
from core.context_compiler.resolver import ContextResolver
from utils.logger import get_logger

logger = get_logger("context_compiler.semantic_selector")


@dataclass
class SemanticDependencies:
    """Encapsulates all semantically relevant hardware constructs for a target artifact."""
    target_component: Optional[EFSComponent] = None
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    instructions: List[Dict[str, Any]] = field(default_factory=list)
    opcodes: List[Dict[str, Any]] = field(default_factory=list)
    instruction_formats: List[Dict[str, Any]] = field(default_factory=list)
    transactions: List[Dict[str, Any]] = field(default_factory=list)
    commands: List[Dict[str, Any]] = field(default_factory=list)
    datapath_operations: List[Dict[str, Any]] = field(default_factory=list)
    memory_operations: List[Dict[str, Any]] = field(default_factory=list)
    handshakes_and_constraints: List[Dict[str, Any]] = field(default_factory=list)
    ordering_and_timeouts: List[Dict[str, Any]] = field(default_factory=list)
    flows_and_branches: List[Dict[str, Any]] = field(default_factory=list)
    errors_and_interrupts: List[Dict[str, Any]] = field(default_factory=list)


class SemanticDependencySelector:
    """Performs dependency-based relevance analysis over canonical EFS IR."""

    def __init__(self, efs_ir: Optional[EFSIR], resolver: Optional[ContextResolver] = None):
        self.efs_ir = efs_ir
        self.resolver = resolver or ContextResolver(efs_ir)

    def select_dependencies(
        self,
        target_comp: Optional[EFSComponent],
        artifact_type: str = "rtl",
        scoped_signals: Optional[List[EFSSignal]] = None,
        target_fsm: Optional[EFSFSM] = None
    ) -> SemanticDependencies:
        """
        Select all semantically required constructs relevant to the target component and artifact.
        Uses dependency tracing from target component, its interfaces, signals, and FSM.
        """
        if not self.efs_ir:
            return SemanticDependencies(target_component=target_comp)

        comp_name = (target_comp.name if target_comp else "") or ""
        comp_type = (target_comp.type if target_comp else "") or ""
        comp_desc = (target_comp.description if target_comp else "") or ""

        # Gather target identity tokens
        target_tokens = set(re.findall(r"\w+", f"{comp_name} {comp_type} {comp_desc}".lower()))
        
        # Gather target interface IDs and names
        target_interfaces: Set[str] = set()
        if target_comp and target_comp.interfaces:
            for iface_ref in target_comp.interfaces:
                target_interfaces.add(iface_ref.lower())
                i_name = self.resolver.resolve_interface_name(iface_ref)
                if i_name:
                    target_interfaces.add(i_name.lower())

        # Gather target signal names
        target_sig_names: Set[str] = set()
        if scoped_signals:
            for s in scoped_signals:
                s_raw_name = s.get("name") if isinstance(s, dict) else getattr(s, "name", None)
                s_raw_id = s.get("signal_id") if isinstance(s, dict) else getattr(s, "signal_id", "")
                s_name = self.resolver.resolve_signal_name(s_raw_name or s_raw_id)
                if s_name:
                    target_sig_names.add(s_name.lower())

        # Check if target is top-level or single-component
        is_top_level = (
            len(self.efs_ir.components) <= 1 or
            comp_type.lower() in ("top_level", "top_level_design", "top", "system") or
            (self.efs_ir.metadata and (
                comp_name.lower() == (self.efs_ir.metadata.top_module or "").lower() or
                comp_name.lower() == (self.efs_ir.metadata.design_name or "").lower()
            ))
        )

        deps = SemanticDependencies(target_component=target_comp)

        # 1. Parameters & Configuration Fields
        deps.parameters = self._select_parameters(target_comp, target_tokens, is_top_level)

        # 2. Instructions, Opcodes & Formats (Instruction Decoder, CPU, ALU, Coprocessor)
        deps.instructions, deps.opcodes, deps.instruction_formats = self._select_instructions_and_opcodes(
            target_comp, target_tokens, target_sig_names, is_top_level
        )

        # 3. Transactions & Transaction Types (including FSM & Protocol Handshakes)
        deps.transactions = self._select_transactions(
            target_comp, target_interfaces, target_sig_names, target_fsm, is_top_level
        )

        # 4. Commands & Datapath Operations
        deps.commands, deps.datapath_operations = self._select_commands_and_datapath(
            target_comp, target_tokens, target_sig_names, is_top_level
        )

        # 5. Memory Operations & Regions
        deps.memory_operations = self._select_memory_operations(
            target_comp, target_tokens, is_top_level
        )

        # 6. Handshakes, Protocol Rules, Ordering Constraints, Timeouts & Out-of-Order Behavior
        deps.handshakes_and_constraints, deps.ordering_and_timeouts = self._select_constraints_and_protocols(
            target_comp, target_interfaces, target_sig_names, is_top_level
        )

        # 7. Flows, Subflows, Sequence Dependencies & Alternate Branches
        deps.flows_and_branches = self._select_flows(
            target_comp, target_interfaces, is_top_level
        )

        # 8. Errors & Interrupt Behavior
        deps.errors_and_interrupts = self._select_errors(
            target_comp, target_tokens, target_sig_names, is_top_level
        )

        return deps

    def _select_parameters(
        self,
        target_comp: Optional[EFSComponent],
        target_tokens: Set[str],
        is_top_level: bool
    ) -> List[Dict[str, Any]]:
        """Select parameters relevant to the target component."""
        params: List[Dict[str, Any]] = []
        comp_id = target_comp.component_id if target_comp else ""
        comp_name_lower = (target_comp.name or "").lower() if target_comp else ""

        for p in (self.efs_ir.parameters or []):
            p_owner = (p.owner or "").lower()
            is_relevant = (
                is_top_level or
                not p.owner or
                p_owner == comp_id.lower() or
                p_owner == comp_name_lower or
                any(tok in p.name.lower() for tok in target_tokens if len(tok) > 3)
            )
            if is_relevant:
                params.append({
                    "name": p.name,
                    "value": p.value or p.default_value,
                    "default_value": p.default_value,
                    "datatype": p.datatype or "integer",
                    "description": self.resolver.scrub_internal_ids(p.description or "")
                })
        return params

    def _select_instructions_and_opcodes(
        self,
        target_comp: Optional[EFSComponent],
        target_tokens: Set[str],
        target_sig_names: Set[str],
        is_top_level: bool
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Select opcodes, instruction fields, and formats if the target module
        is an instruction decoder, CPU, ALU, or execution unit.
        """
        is_decoder_or_core = (
            is_top_level or
            any(k in target_tokens for k in (
                "decode", "decoder", "instruction", "instr", "opcode", "cpu", "core",
                "processor", "alu", "exec", "execution", "dispatch", "fetch", "isa"
            )) or
            any(s in target_sig_names for s in (
                "opcode", "instr", "instruction", "funct3", "funct7", "imm", "opcode_in"
            ))
        )

        instructions: List[Dict[str, Any]] = []
        opcodes: List[Dict[str, Any]] = []
        formats: List[Dict[str, Any]] = []

        if is_decoder_or_core:
            # Opcodes
            for opc in (self.efs_ir.opcodes or []):
                opcodes.append({
                    "mnemonic": opc.mnemonic,
                    "binary_encoding": opc.binary_encoding,
                    "operation": self.resolver.scrub_internal_ids(opc.operation or ""),
                    "operands": opc.operands or []
                })

            # Instruction Formats & Fields
            for fmt in (self.efs_ir.instruction_formats or []):
                fields_data = []
                for fld in (fmt.fields or []):
                    fields_data.append({
                        "name": fld.name,
                        "msb": fld.msb,
                        "lsb": fld.lsb,
                        "width": fld.width,
                        "description": fld.description
                    })
                formats.append({
                    "format_name": fmt.name,
                    "total_width": fmt.total_width,
                    "fields": fields_data
                })

            # Instructions
            for ins in (self.efs_ir.instructions or []):
                fields_data = []
                for fld in (ins.fields or []):
                    fields_data.append({
                        "name": fld.name,
                        "msb": fld.msb,
                        "lsb": fld.lsb,
                        "width": fld.width,
                        "description": fld.description
                    })
                instructions.append({
                    "name": ins.name or ins.mnemonic,
                    "mnemonic": ins.mnemonic,
                    "opcode": ins.opcode,
                    "format": ins.format,
                    "operands": ins.operands or [],
                    "fields": fields_data,
                    "description": self.resolver.scrub_internal_ids(ins.description or "")
                })

        return instructions, opcodes, formats

    def _select_transactions(
        self,
        target_comp: Optional[EFSComponent],
        target_interfaces: Set[str],
        target_sig_names: Set[str],
        target_fsm: Optional[EFSFSM],
        is_top_level: bool
    ) -> List[Dict[str, Any]]:
        """
        Select transactions that determine state transitions, handshake behavior,
        ordering constraints, or are directly participated in by target component.
        """
        comp_id = target_comp.component_id if target_comp else ""
        comp_name_lower = (target_comp.name or "").lower() if target_comp else ""

        # Extract transaction references from FSM transitions
        fsm_trans_refs = set()
        raw_transitions = target_fsm.get("transitions", []) if isinstance(target_fsm, dict) else (getattr(target_fsm, "transitions", []) or []) if target_fsm else []
        for tr in raw_transitions:
            cond_val = tr.get("condition") if isinstance(tr, dict) else getattr(tr, "condition", "")
            act_val = tr.get("action") if isinstance(tr, dict) else getattr(tr, "action", "")
            cond_text = str(cond_val or "").lower()
            act_text = str(act_val or "").lower()
            for word in re.findall(r"[A-Za-z0-9_]+", f"{cond_text} {act_text}"):
                fsm_trans_refs.add(word)

        tx_list: List[Dict[str, Any]] = []
        seen_tx_names = set()

        for tx in (self.efs_ir.transactions or []):
            tx_name = (tx.name or "").strip()
            tx_type = (tx.type or "").strip()
            tx_name_lower = tx_name.lower()
            tx_type_lower = tx_type.lower()

            is_direct_participant = (
                tx.initiator == comp_id or
                tx.target == comp_id or
                (tx.initiator and tx.initiator.lower() == comp_name_lower) or
                (tx.target and tx.target.lower() == comp_name_lower)
            )

            is_interface_match = any(
                iface.lower() in target_interfaces for iface in (tx.participating_interfaces or [])
            )

            is_signal_match = any(
                sig.lower() in target_sig_names for sig in (tx.signals or [])
            )

            is_fsm_dependency = (
                tx_name_lower in fsm_trans_refs or
                tx_type_lower in fsm_trans_refs
            )

            if is_top_level or is_direct_participant or is_interface_match or is_signal_match or is_fsm_dependency:
                if tx_name_lower not in seen_tx_names:
                    seen_tx_names.add(tx_name_lower)
                    tx_list.append({
                        "name": tx_name,
                        "type": tx_type,
                        "protocol": tx.protocol or "generic",
                        "ordered_steps": [self.resolver.scrub_internal_ids(s) for s in (tx.ordered_steps or [])],
                        "preconditions": [self.resolver.scrub_internal_ids(p) for p in (tx.preconditions or [])],
                        "completion_condition": self.resolver.scrub_internal_ids(tx.completion_condition or ""),
                        "response": tx.response or tx.expected_response or "",
                        "ordering": tx.ordering or "in-order",
                        "error_behavior": self.resolver.scrub_internal_ids(tx.error_behavior or ""),
                        "timing": tx.timing or ""
                    })

                # Limit to top 20 relevant transactions to avoid payload explosion while preserving all critical semantics
                if len(tx_list) >= 20:
                    break

        return tx_list

    def _select_commands_and_datapath(
        self,
        target_comp: Optional[EFSComponent],
        target_tokens: Set[str],
        target_sig_names: Set[str],
        is_top_level: bool
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Select commands and datapath operations relevant to target module."""
        commands: List[Dict[str, Any]] = []
        datapath: List[Dict[str, Any]] = []

        comp_name_lower = (target_comp.name or "").lower() if target_comp else ""

        for fb in (getattr(self.efs_ir, "functional_behavior", []) or []):
            subj = (getattr(fb, "subject", "") or getattr(fb, "target_component", "") or "").lower()
            name = (getattr(fb, "name", "") or "").lower()

            is_match = (
                is_top_level or
                not subj or
                subj == comp_name_lower or
                any(tok in subj for tok in target_tokens if len(tok) > 3) or
                any(tok in name for tok in target_tokens if len(tok) > 3)
            )

            if is_match:
                act = getattr(fb, "action", "") or getattr(fb, "expression", "") or getattr(fb, "description", "")
                trigger = getattr(fb, "trigger", "")
                exp_res = getattr(fb, "expected_result", "")
                
                # If triggered by a command or controls datapath
                if "cmd" in name or "command" in name or "cmd" in str(trigger).lower():
                    commands.append({
                        "name": getattr(fb, "name", "cmd"),
                        "trigger": self.resolver.scrub_internal_ids(str(trigger)),
                        "action": self.resolver.scrub_internal_ids(str(act)),
                        "expected_result": self.resolver.scrub_internal_ids(str(exp_res))
                    })
                else:
                    datapath.append({
                        "operation": getattr(fb, "name", "operation"),
                        "trigger": self.resolver.scrub_internal_ids(str(trigger)),
                        "expression": self.resolver.scrub_internal_ids(str(act)),
                        "inputs": [self.resolver.resolve_signal_name(s) for s in getattr(fb, "inputs", [])],
                        "outputs": [self.resolver.resolve_signal_name(s) for s in getattr(fb, "outputs", [])]
                    })

        return commands, datapath

    def _select_memory_operations(
        self,
        target_comp: Optional[EFSComponent],
        target_tokens: Set[str],
        is_top_level: bool
    ) -> List[Dict[str, Any]]:
        """Select memory regions, caches, FIFOs, and buffers relevant to target."""
        mem_ops: List[Dict[str, Any]] = []
        is_memory_related = (
            is_top_level or
            any(k in target_tokens for k in ("memory", "mem", "sram", "dram", "cache", "buffer", "fifo", "dma"))
        )

        for mem in (getattr(self.efs_ir, "memory_regions", []) or []):
            m_name = getattr(mem, "name", "")
            m_type = getattr(mem, "type", "")
            if is_memory_related or any(tok in m_name.lower() for tok in target_tokens if len(tok) > 3):
                mem_ops.append({
                    "name": m_name,
                    "type": m_type or "SRAM",
                    "base_address": getattr(mem, "base_address", "0x0"),
                    "size": getattr(mem, "size", "N/A"),
                    "depth": getattr(mem, "depth", 0),
                    "width": getattr(mem, "width", 0),
                    "description": self.resolver.scrub_internal_ids(getattr(mem, "description", "") or "")
                })
        return mem_ops

    def _select_constraints_and_protocols(
        self,
        target_comp: Optional[EFSComponent],
        target_interfaces: Set[str],
        target_sig_names: Set[str],
        is_top_level: bool
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Select handshake rules, stability constraints, ordering rules,
        out-of-order constraints, and timeouts.
        """
        handshakes: List[Dict[str, Any]] = []
        ordering_timeouts: List[Dict[str, Any]] = []

        # 1. Constraints
        for c in (self.efs_ir.constraints or []):
            c_type = (c.type or "").lower()
            c_cond = c.condition or ""
            c_exp = c.expected_behavior or ""

            # Check relevance
            src_objs = {s.lower() for s in (c.source_objects or [])}
            tgt_objs = {t.lower() for t in (c.target_objects or [])}
            is_relevant = (
                is_top_level or
                bool(src_objs.intersection(target_sig_names)) or
                bool(tgt_objs.intersection(target_sig_names)) or
                any(i in c_cond.lower() for i in target_interfaces) or
                any(i in c_exp.lower() for i in target_interfaces) or
                any(s in c_cond.lower() for s in target_sig_names if len(s) > 2) or
                any(s in c_exp.lower() for s in target_sig_names if len(s) > 2)
            )

            if is_relevant:
                item = {
                    "type": c.type or "constraint",
                    "condition": self.resolver.scrub_internal_ids(c_cond),
                    "expected_behavior": self.resolver.scrub_internal_ids(c_exp),
                    "severity": c.severity or "MAJOR"
                }
                if c_type in ("handshake", "stability"):
                    handshakes.append(item)
                elif c_type in ("ordering", "timeout", "dependency", "latency"):
                    ordering_timeouts.append(item)
                else:
                    handshakes.append(item)

        # 2. Timing Rules
        for tr in (self.efs_ir.timing_rules or []):
            ordering_timeouts.append({
                "type": tr.type or "timing",
                "cycle_relationship": tr.cycle_relationship,
                "ordering": tr.ordering,
                "min_delay": tr.minimum_delay,
                "max_delay": tr.maximum_delay
            })

        return handshakes, ordering_timeouts

    def _select_flows(
        self,
        target_comp: Optional[EFSComponent],
        target_interfaces: Set[str],
        is_top_level: bool
    ) -> List[Dict[str, Any]]:
        """Select flows, subflows, sequence dependencies, and alternate branches."""
        flows: List[Dict[str, Any]] = []
        comp_id = target_comp.component_id if target_comp else ""
        comp_name_lower = (target_comp.name or "").lower() if target_comp else ""

        for flow in (getattr(self.efs_ir, "flows", []) or []):
            flow_comps = getattr(flow, "components", None) or (flow.get("components", []) if isinstance(flow, dict) else []) or []
            flow_ifaces = getattr(flow, "interfaces", None) or (flow.get("interfaces", []) if isinstance(flow, dict) else []) or []
            comps = {str(c).lower() for c in flow_comps}
            ifaces = {str(i).lower() for i in flow_ifaces}

            is_match = (
                is_top_level or
                comp_id.lower() in comps or
                comp_name_lower in comps or
                bool(ifaces.intersection(target_interfaces))
            )

            if is_match:
                steps_data = []
                flow_steps = getattr(flow, "ordered_steps", None) or (flow.get("ordered_steps", []) if isinstance(flow, dict) else []) or []
                for step in flow_steps:
                    action_desc = getattr(step, "action_description", None) or (step.get("action_description", "") if isinstance(step, dict) else str(step))
                    cond_desc = getattr(step, "condition", None) or (step.get("condition", "") if isinstance(step, dict) else "")
                    steps_data.append({
                        "action": self.resolver.scrub_internal_ids(action_desc),
                        "condition": self.resolver.scrub_internal_ids(cond_desc or ""),
                        "state_transitions": step.state_transitions or [],
                        "expected_events": step.expected_events or [],
                        "timing": step.timing_expectations or ""
                    })
                flow_name = getattr(flow, "name", None) or (flow.get("name", "") if isinstance(flow, dict) else str(flow))
                flow_conds = getattr(flow, "conditions", None) or (flow.get("conditions", "") if isinstance(flow, dict) else "")
                flows.append({
                    "flow_name": flow_name,
                    "conditions": self.resolver.scrub_internal_ids(flow_conds or ""),
                    "steps": steps_data
                })

        return flows

    def _select_errors(
        self,
        target_comp: Optional[EFSComponent],
        target_tokens: Set[str],
        target_sig_names: Set[str],
        is_top_level: bool
    ) -> List[Dict[str, Any]]:
        """Select error handling and interrupt behaviors."""
        errors: List[Dict[str, Any]] = []
        for err in (getattr(self.efs_ir, "errors", []) or []):
            err_name = (err.name or "").lower()
            err_cond = (err.condition or "").lower()

            is_match = (
                is_top_level or
                any(tok in err_name for tok in target_tokens if len(tok) > 3) or
                any(s in err_cond for s in target_sig_names if len(s) > 3) or
                "interrupt" in err.behavior.lower() or
                "irq" in target_sig_names
            )
            if is_match:
                errors.append({
                    "name": err.name,
                    "code": err.code,
                    "condition": self.resolver.scrub_internal_ids(err.condition),
                    "behavior": self.resolver.scrub_internal_ids(err.behavior)
                })
        return errors
