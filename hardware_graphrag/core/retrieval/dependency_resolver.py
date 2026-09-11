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
    # Gather interfaces owned by target
    interfaces: List[EFSInterface] = []
    for iface in efs_ir.interfaces:
        if (iface.source_component == target_comp.component_id or
            iface.destination_component == target_comp.component_id or
            iface.name.lower() in [i.lower() for i in target_comp.interfaces] or
            target_comp.name.lower() in iface.name.lower()):
            interfaces.append(iface)

    # Gather signals owned by target component or its interfaces
    interface_ids = {iface.interface_id for iface in interfaces}
    signals: List[EFSSignal] = []
    for sig in efs_ir.signals:
        if (sig.owner == target_comp.component_id or
            sig.consumer == target_comp.component_id or
            (sig.interface and sig.interface in interface_ids) or
            target_comp.name.lower() in sig.name.lower() or
            any(w in sig.name.lower() for w in target_comp.name.lower().split() if len(w) > 3)):
            signals.append(sig)

    # Dynamic expansion: Fallback to top-level signals/interfaces if 0 signals specific to component found
    if not signals and efs_ir.signals:
        signals = list(efs_ir.signals)
    if not interfaces and efs_ir.interfaces:
        interfaces = list(efs_ir.interfaces)

    # Gather registers owned by target
    registers: List[EFSRegister] = []
    for reg in efs_ir.registers:
        if reg.owner == target_comp.component_id or target_comp.name.lower() in reg.name.lower():
            registers.append(reg)
    if not registers and efs_ir.registers:
        registers = list(efs_ir.registers)

    # Gather state machines (FSMs) for target
    fsms: List[EFSFSM] = []
    for fsm in efs_ir.fsms:
        if target_comp.name.lower() in fsm.name.lower():
            fsms.append(fsm)

    # Gather transactions relevant to target
    transactions: List[EFSTransaction] = []
    for tx in efs_ir.transactions:
        if (tx.initiator == target_comp.component_id or
            tx.target == target_comp.component_id or
            target_comp.name.lower() in tx.name.lower()):
            transactions.append(tx)

    # Gather timing rules, constraints, and protocol rules
    constraints = list(efs_ir.constraints)
    timing_rules = list(efs_ir.timing_rules)
    protocol_rules = list(efs_ir.protocol_rules)

    # 2. Submodule / Dependency traversal (depth 1)
    for comp in efs_ir.components:
        if comp.component_id == target_comp.component_id:
            continue
        if (comp.name.lower() in target_comp.description.lower() or
            target_comp.name.lower() in comp.description.lower()):
            direct_submodules.append(comp)
            resolved_comp_ids.add(comp.component_id)
            resolved_comp_names.add(comp.name.lower())

    # 3. Neo4j Graph Traversal (if available)
    try:
        store = get_graph_store()
        neighbors, edges = store.get_neighbors(target_comp.name, hops=max_depth)
        for n in neighbors:
            n_name = n.get("name", "")
            for comp in efs_ir.components:
                if comp.name.lower() == n_name.lower() and comp.component_id not in resolved_comp_ids:
                    direct_submodules.append(comp)
                    resolved_comp_ids.add(comp.component_id)
    except Exception as e:
        logger.debug(f"Neo4j graph dependency expansion skipped: {e}")

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
        depth_used=effective_depth
    )
