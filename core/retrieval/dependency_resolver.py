"""
Dependency Resolver module.

Traverses structural hardware dependencies for a resolved target component
using EFS IR objects and Neo4j knowledge graph traversal.
Configurable graph depth (default depth = 1, expanding to 2+ if needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Optional
from core.efs_ir.models import (
    EFSIR, EFSComponent, EFSInterface, EFSSignal, EFSRegister, EFSFSM,
    EFSTransaction, EFSConstraint, EFSTimingRule, EFSProtocolRule
)
from core.graph.neo4j_builder import get_graph_store
from utils.logger import get_logger

logger = get_logger("retrieval.dependency_resolver")


@dataclass
class ResolvedDependencies:
    target_component: EFSComponent
    direct_submodules: List[EFSComponent] = field(default_factory=list)
    indirect_submodules: List[EFSComponent] = field(default_factory=list)
    interfaces: List[EFSInterface] = field(default_factory=list)
    signals: List[EFSSignal] = field(default_factory=list)
    registers: List[EFSRegister] = field(default_factory=list)
    fsms: List[EFSFSM] = field(default_factory=list)
    transactions: List[EFSTransaction] = field(default_factory=list)
    constraints: List[EFSConstraint] = field(default_factory=list)
    timing_rules: List[EFSTimingRule] = field(default_factory=list)
    protocol_rules: List[EFSProtocolRule] = field(default_factory=list)
    dependency_classifications: Dict[str, str] = field(default_factory=dict)
    unresolved_dependencies: List[Dict[str, Any]] = field(default_factory=list)
    depth_used: int = 1


def resolve_dependencies(
    target_comp: EFSComponent,
    efs_ir: EFSIR,
    max_depth: int = 1
) -> ResolvedDependencies:
    """
    Retrieve target component and its structural, timing, reset, and protocol dependencies.
    Dynamically expands traversal depth when target component dependencies are non-empty.
    """
    logger.info(f"Resolving dependencies for target '{target_comp.name}' (max depth={max_depth})...")

    resolved_comp_ids: Set[str] = {target_comp.component_id}
    resolved_comp_names: Set[str] = {target_comp.name.lower()}

    direct_submodules: List[EFSComponent] = []
    indirect_submodules: List[EFSComponent] = []

    # 1. Structural traversal using EFS IR relationships
    # Gather interfaces owned by or associated with target
    interfaces: List[EFSInterface] = []
    target_comp_ifaces = set(target_comp.interfaces or [])
    for iface in efs_ir.interfaces:
        if (iface.interface_id in target_comp_ifaces or
            iface.name.lower() in [i.lower() for i in target_comp_ifaces] or
            iface.source_component == target_comp.component_id or
            iface.source_component == target_comp.name or
            iface.destination_component == target_comp.component_id or
            iface.destination_component == target_comp.name):
            interfaces.append(iface)

    # Gather signals owned by target component or declared in its interfaces
    interface_ids = {iface.interface_id for iface in interfaces}
    interface_signal_ids = set()
    interface_signal_names = set()
    for iface in interfaces:
        for s in iface.signals:
            interface_signal_ids.add(s)
            interface_signal_names.add(s.lower())

    signals: List[EFSSignal] = []
    for sig in efs_ir.signals:
        sig_id = getattr(sig, "signal_id", "")
        sig_name_lower = sig.name.lower()
        if (sig.owner == target_comp.component_id or
            sig.owner == target_comp.name or
            sig.consumer == target_comp.component_id or
            sig.consumer == target_comp.name or
            (sig.interface and sig.interface in interface_ids) or
            sig_id in interface_signal_ids or
            sig_name_lower in interface_signal_names):
            signals.append(sig)

    # Ensure clock and reset signals feeding the component are included
    all_clk_rst_sigs = list(efs_ir.signals) + getattr(efs_ir, "clocks", []) + getattr(efs_ir, "resets", [])
    clk_domain = (target_comp.clock_domain or "").lower()
    rst_domain = (target_comp.reset_domain or "").lower()
    for sig in all_clk_rst_sigs:
        s_name_lower = sig.name.lower()
        if any(s.name == sig.name for s in signals):
            continue
        if clk_domain and (sig.signal_id.lower() == clk_domain or s_name_lower == clk_domain):
            signals.append(sig)
        elif rst_domain and (sig.signal_id.lower() == rst_domain or s_name_lower == rst_domain):
            signals.append(sig)
        elif not clk_domain and sig.semantic_role == "clock" and not any(s.semantic_role == "clock" for s in signals):
            signals.append(sig)
        elif not rst_domain and sig.semantic_role == "reset" and not any(s.semantic_role == "reset" for s in signals):
            signals.append(sig)

    # Fallback to top-level signals/interfaces ONLY IF target is top-level or single component
    is_top_level = (
        len(efs_ir.components) <= 1 or
        target_comp.type.lower() in ("top_level_design", "top", "system") or
        (efs_ir.metadata and (
            target_comp.name.lower() == (efs_ir.metadata.top_module or "").lower() or
            target_comp.name.lower() == (efs_ir.metadata.design_name or "").lower()
        ))
    )
    if not signals and is_top_level and efs_ir.signals:
        signals = list(efs_ir.signals)
    if not interfaces and is_top_level and efs_ir.interfaces:
        interfaces = list(efs_ir.interfaces)

    comp_name_lower = target_comp.name.lower()
    comp_type_lower = (getattr(target_comp, "type", "") or "").lower()

    # Track structural dependency classifications
    dep_classifications: Dict[str, str] = {}
    unresolved_deps: List[Dict[str, Any]] = []

    for s in signals:
        if s.owner == target_comp.component_id or (s.owner and s.owner.lower() == comp_name_lower):
            dep_classifications[s.signal_id] = "DIRECT_OWNED"
        elif s.interface and s.interface in interface_ids:
            dep_classifications[s.signal_id] = "INTERFACE_CONNECTED"
        elif s.semantic_role in ("clock", "reset"):
            dep_classifications[s.signal_id] = "SHARED_RESOURCE"
        else:
            dep_classifications[s.signal_id] = "EXTERNAL_DEPENDENCY"

    # Identify unresolved interface signals (declared in interface but missing from IR signals)
    known_sig_ids = {s.signal_id for s in efs_ir.signals}
    for ref_id in interface_signal_ids:
        if ref_id not in known_sig_ids and not any(s.name.lower() == ref_id.lower() for s in efs_ir.signals):
            unresolved_deps.append({
                "type": "MISSING_INTERFACE_SIGNAL",
                "reference": ref_id,
                "target_component": target_comp.component_id,
                "severity": "WARNING"
            })

    # Gather registers owned by target (metadata-driven, never suppressed by component type)
    registers: List[EFSRegister] = []
    for reg in (efs_ir.registers or []):
        reg_owner = (reg.owner or "").lower()
        if (reg.owner == target_comp.component_id or
            reg.owner == target_comp.name or
            reg_owner == target_comp.component_id.lower() or
            reg_owner == comp_name_lower or
            (is_top_level and not reg.owner)):
            registers.append(reg)
            dep_classifications[reg.register_id] = "DIRECT_OWNED"
    if not registers and is_top_level and efs_ir.registers:
        registers = list(efs_ir.registers)
        for r in registers:
            dep_classifications.setdefault(r.register_id, "DIRECT_OWNED")

    # Gather state machines (FSMs) for target (metadata-driven, never suppressed by component type)
    fsms: List[EFSFSM] = []
    for fsm in (efs_ir.fsms or []):
        fsm_owner = (getattr(fsm, "owner", "") or getattr(fsm, "component_id", "") or "").lower()
        if (fsm_owner == target_comp.component_id.lower() or
            fsm_owner == comp_name_lower or
            (is_top_level and not fsm_owner)):
            fsms.append(fsm)
            dep_classifications[getattr(fsm, "fsm_id", fsm.name)] = "DIRECT_OWNED"
    if not fsms and efs_ir.fsms and is_top_level:
        fsms = list(efs_ir.fsms)
        for f in fsms:
            dep_classifications.setdefault(getattr(f, "fsm_id", f.name), "DIRECT_OWNED")

    # Gather transactions relevant to target
    transactions: List[EFSTransaction] = []
    for tx in (efs_ir.transactions or []):
        if (tx.initiator == target_comp.component_id or
            tx.target == target_comp.component_id or
            is_top_level):
            transactions.append(tx)

    # Gather timing rules, constraints, and protocol rules (scoped by target relevance)
    target_sig_ids = {s.signal_id for s in signals} | {s.name.lower() for s in signals}
    target_iface_ids = {i.interface_id for i in interfaces} | {i.name.lower() for i in interfaces}

    constraints: List[EFSConstraint] = []
    for c in (efs_ir.constraints or []):
        c_comp = getattr(c, "component", "") or getattr(c, "component_id", "")
        c_scope = getattr(c, "scope", "")
        c_sigs = {s.lower() for s in (getattr(c, "signals", []) or [])}
        if is_top_level or c_comp in (target_comp.component_id, target_comp.name) or (c_sigs & target_sig_ids) or c_scope in ("global", "top"):
            constraints.append(c)

    timing_rules: List[EFSTimingRule] = []
    for tr in (efs_ir.timing_rules or []):
        tr_comp = getattr(tr, "component", "") or getattr(tr, "component_id", "")
        tr_sigs = {s.lower() for s in (getattr(tr, "signals", []) or [])}
        if is_top_level or tr_comp in (target_comp.component_id, target_comp.name) or (tr_sigs & target_sig_ids):
            timing_rules.append(tr)

    protocol_rules: List[EFSProtocolRule] = []
    for pr in (efs_ir.protocol_rules or []):
        pr_iface = (getattr(pr, "interface", "") or getattr(pr, "protocol", "")).lower()
        if is_top_level or pr_iface in target_iface_ids or any(ti in pr_iface for ti in target_iface_ids):
            protocol_rules.append(pr)

    # 2. Submodule / Hierarchy traversal (metadata-driven parent-child relationships)
    subcomp_ids = set(getattr(target_comp, "subcomponents", []) or [])
    for comp in (efs_ir.components or []):
        if comp.component_id == target_comp.component_id:
            continue
        is_submodule = (
            comp.component_id in subcomp_ids or
            comp.name in subcomp_ids or
            getattr(comp, "parent_id", None) == target_comp.component_id or
            getattr(comp, "parent_id", None) == target_comp.name
        )
        if is_submodule:
            direct_submodules.append(comp)
            resolved_comp_ids.add(comp.component_id)
            resolved_comp_names.add(comp.name.lower())
            dep_classifications[comp.component_id] = "CHILD_SUBMODULE"

    # 3. Neo4j / Knowledge Graph Traversal
    try:
        store = get_graph_store()
        neighbors, edges = store.get_neighbors(target_comp.name, hops=max_depth)
        for n in neighbors:
            n_name = n.get("name", "")
            for comp in efs_ir.components:
                if comp.name.lower() == n_name.lower() and comp.component_id not in resolved_comp_ids:
                    direct_submodules.append(comp)
                    resolved_comp_ids.add(comp.component_id)
            for iface in efs_ir.interfaces:
                if iface.name.lower() == n_name.lower() and not any(i.name == iface.name for i in interfaces):
                    interfaces.append(iface)
            for sig in efs_ir.signals:
                if sig.name.lower() == n_name.lower() and not any(s.name == sig.name for s in signals):
                    signals.append(sig)
            for reg in efs_ir.registers:
                if reg.name.lower() == n_name.lower() and not any(r.name == reg.name for r in registers):
                    registers.append(reg)
    except Exception as e:
        logger.debug(f"Knowledge graph dependency expansion skipped: {e}")

    # Expand to depth = 2 if max_depth >= 2 or if direct submodules exist
    effective_depth = max_depth
    if direct_submodules and max_depth < 2:
        effective_depth = 2

    if effective_depth >= 2:
        for sub in list(direct_submodules):
            for comp in efs_ir.components:
                if comp.component_id not in resolved_comp_ids:
                    if comp.name.lower() in sub.description.lower():
                        indirect_submodules.append(comp)
                        resolved_comp_ids.add(comp.component_id)

    logger.info(
        f"Target '{target_comp.name}' dependency resolution complete: "
        f"{len(direct_submodules)} direct submodules, {len(interfaces)} interfaces, "
        f"{len(signals)} signals, {len(registers)} registers, {len(fsms)} FSMs, "
        f"{len(constraints)} constraints."
    )

    return ResolvedDependencies(
        target_component=target_comp,
        direct_submodules=direct_submodules,
        indirect_submodules=indirect_submodules,
        interfaces=interfaces,
        signals=signals,
        registers=registers,
        fsms=fsms,
        transactions=transactions,
        constraints=constraints,
        timing_rules=timing_rules,
        protocol_rules=protocol_rules,
        dependency_classifications=dep_classifications,
        unresolved_dependencies=unresolved_deps,
        depth_used=effective_depth
    )
