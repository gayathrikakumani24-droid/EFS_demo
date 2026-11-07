"""
Scope Resolver module.

Determines the minimal sufficient knowledge boundary for a target hardware component:
1. Target component metadata, clock domains, and reset domains.
2. Ports and signals belonging directly to or driven/consumed by the component.
3. Registers and register fields owned by or mapped to the component.
4. Opcodes and instructions decoded or executed by the component.
5. FSM states and transitions implemented by the component.
6. Protocol interfaces attached to the component.
7. Connected adjacent components (at interface boundary level only, without pulling internal logic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from core.graph.neo4j_builder import get_graph_store
from utils.logger import get_logger

logger = get_logger("retrieval.scope_resolver")


@dataclass
class ResolvedScope:
    target_component_name: str
    target_component_type: str = "submodule"
    clock_domain: str = ""
    reset_domain: str = ""
    ports: List[Dict[str, Any]] = field(default_factory=list)
    interfaces: List[Dict[str, Any]] = field(default_factory=list)
    registers: List[Dict[str, Any]] = field(default_factory=list)
    instructions: List[Dict[str, Any]] = field(default_factory=list)
    opcodes: List[Dict[str, Any]] = field(default_factory=list)
    fsms: List[Dict[str, Any]] = field(default_factory=list)
    behavioral_rules: List[Dict[str, Any]] = field(default_factory=list)
    connected_components: List[Dict[str, Any]] = field(default_factory=list)
    source_chunk_ids: List[str] = field(default_factory=list)
    specification_gaps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_component_name": self.target_component_name,
            "target_component_type": self.target_component_type,
            "clock_domain": self.clock_domain,
            "reset_domain": self.reset_domain,
            "ports": self.ports,
            "interfaces": self.interfaces,
            "registers": self.registers,
            "instructions": self.instructions,
            "opcodes": self.opcodes,
            "fsms": self.fsms,
            "behavioral_rules": self.behavioral_rules,
            "connected_components": self.connected_components,
            "source_chunk_ids": self.source_chunk_ids,
            "specification_gaps": self.specification_gaps,
        }


def resolve_component_scope(
    target_name: str,
    scope_type: str = "COMPONENT",
    req_ir: Optional[Any] = None,
    document_id: Optional[str] = None
) -> ResolvedScope:
    """
    Resolve the minimal sufficient knowledge boundary for target_name using Graph DB and RequirementIR.
    """
    logger.info(f"Resolving scope for '{target_name}' (scope_type={scope_type}, doc_id={document_id})...")

    store = get_graph_store()
    target_clean = (target_name or "").strip()
    target_lower = target_clean.lower()
    from core.normalization.target_normalizer import normalize_target_name
    target_norm = normalize_target_name(target_clean)

    # Search for matching component node in graph
    all_nodes = store.all_nodes(document_id=document_id)
    candidate_node = None
    for n in all_nodes:
        n_name = n.get("name", "")
        n_type = n.get("type", "")
        if n_type.lower() in ("component", "design", "module", "subsystem"):
            n_norm = normalize_target_name(n_name)
            if n_norm == target_norm:
                candidate_node = n
                break
            elif target_norm and (target_norm in n_norm or n_norm in target_norm):
                if candidate_node is None:
                    candidate_node = n

    # Fallback if no direct component node match
    comp_name = candidate_node["name"] if candidate_node else target_clean
    comp_type = candidate_node.get("component_type", "submodule") if candidate_node else "submodule"
    clock_domain = candidate_node.get("clock_domain", "") if candidate_node else ""
    reset_domain = candidate_node.get("reset_domain", "") if candidate_node else ""

    chunk_ids: Set[str] = set()
    if candidate_node:
        chunk_ids.update(candidate_node.get("chunk_ids", []))
        cid = candidate_node.get("chunk_id")
        if cid:
            chunk_ids.add(cid)

    # 1. Traverse Graph Neighbors up to 2 hops for the target component
    neighbors, edges = store.get_neighbors(comp_name, hops=2)

    ports: List[Dict[str, Any]] = []
    interfaces: List[Dict[str, Any]] = []
    registers: List[Dict[str, Any]] = []
    instructions: List[Dict[str, Any]] = []
    opcodes: List[Dict[str, Any]] = []
    fsms: List[Dict[str, Any]] = []
    connected_components: List[Dict[str, Any]] = []

    seen_ports = set()
    seen_ifaces = set()
    seen_regs = set()
    seen_insts = set()
    seen_opcodes = set()
    seen_fsms = set()

    for n in neighbors:
        ntype = str(n.get("type", "")).lower()
        nname = n.get("name", "")
        cids = n.get("chunk_ids", [])
        if cids:
            chunk_ids.update(cids)
        if n.get("chunk_id"):
            chunk_ids.add(n["chunk_id"])

        if ntype in ("signal", "port"):
            if nname.lower() not in seen_ports:
                seen_ports.add(nname.lower())
                ports.append({
                    "name": nname,
                    "width": n.get("width", 1),
                    "direction": n.get("direction", "input"),
                    "datatype": n.get("datatype", "logic"),
                    "semantic_role": n.get("semantic_role", "data"),
                    "description": n.get("description", ""),
                    "chunk_id": n.get("chunk_id") or (cids[0] if cids else ""),
                    "doc_id": n.get("doc_id", ""),
                })
        elif ntype == "interface":
            if nname.lower() not in seen_ifaces:
                seen_ifaces.add(nname.lower())
                interfaces.append({
                    "name": nname,
                    "protocol": n.get("protocol", "GENERIC"),
                    "role": n.get("role", "slave"),
                    "description": n.get("description", ""),
                    "chunk_id": n.get("chunk_id") or (cids[0] if cids else ""),
                })
        elif ntype in ("register", "reg"):
            if nname.lower() not in seen_regs:
                seen_regs.add(nname.lower())
                registers.append({
                    "name": nname,
                    "address": n.get("address", "0x0"),
                    "offset": n.get("offset", "0x0"),
                    "width": n.get("width", 32),
                    "access_type": n.get("access_type", "RW"),
                    "description": n.get("description", ""),
                    "chunk_id": n.get("chunk_id") or (cids[0] if cids else ""),
                })
        elif ntype in ("instruction", "inst"):
            if nname.lower() not in seen_insts:
                seen_insts.add(nname.lower())
                instructions.append({
                    "name": nname,
                    "width": n.get("width", 32),
                    "description": n.get("description", ""),
                    "chunk_id": n.get("chunk_id") or (cids[0] if cids else ""),
                })
        elif ntype in ("opcode", "opc"):
            if nname.lower() not in seen_opcodes:
                seen_opcodes.add(nname.lower())
                opcodes.append({
                    "mnemonic": nname,
                    "binary_encoding": n.get("binary_encoding", ""),
                    "operation": n.get("operation", ""),
                    "chunk_id": n.get("chunk_id") or (cids[0] if cids else ""),
                })
        elif ntype == "fsm":
            if nname.lower() not in seen_fsms:
                seen_fsms.add(nname.lower())
                fsms.append({
                    "name": nname,
                    "initial_state": n.get("initial_state", "IDLE"),
                    "states": [],
                    "transitions": [],
                })
        elif ntype in ("component", "module", "submodule") and nname != comp_name:
            connected_components.append({
                "name": nname,
                "type": n.get("component_type", "module"),
                "boundary_only": True,
            })

    # 2. Extract Scope from RequirementIR if available
    behavioral_rules: List[Dict[str, Any]] = []
    gaps: List[str] = []

    if req_ir:
        # Pull entities directly relevant to the component
        for ent in getattr(req_ir, "entities", []):
            ename = getattr(ent, "name", "")
            etype = getattr(ent, "type", "").lower()
            attrs = getattr(ent, "attributes", {}) or {}

            ent_refs = getattr(ent, "source_references", [])
            sec_text = " ".join([getattr(r, "section", "") for r in ent_refs]).lower()
            edge_connected = any(
                (getattr(ed, "source", "").lower() == comp_name.lower() and getattr(ed, "target", "") == ename) or
                (getattr(ed, "target", "").lower() == comp_name.lower() and getattr(ed, "source", "") == ename)
                for ed in getattr(req_ir, "edges", [])
            )

            is_relevant = (
                comp_name.lower() in ename.lower() or
                ename.lower() in comp_name.lower() or
                comp_name.lower() in sec_text or
                edge_connected or
                scope_type == "FULL_SYSTEM"
            )

            if etype in ("signal", "port", "clock", "reset") and is_relevant:
                if ename.lower() not in seen_ports:
                    seen_ports.add(ename.lower())
                    ports.append({
                        "name": ename,
                        "width": attrs.get("width", 1),
                        "direction": attrs.get("direction", "input"),
                        "datatype": attrs.get("datatype", "logic"),
                        "semantic_role": attrs.get("semantic_role", "data"),
                        "description": getattr(ent, "description", ""),
                        "chunk_id": getattr(ent.source_references[0], "block_id", "") if getattr(ent, "source_references", None) else "",
                    })
            elif etype in ("register", "reg") and is_relevant:
                if ename.lower() not in seen_regs:
                    seen_regs.add(ename.lower())
                    registers.append({
                        "name": ename,
                        "address": attrs.get("address", "0x0"),
                        "offset": attrs.get("offset", "0x0"),
                        "width": attrs.get("width", 32),
                        "access_type": attrs.get("access_type", "RW"),
                        "description": getattr(ent, "description", ""),
                    })
            elif etype in ("instruction", "inst", "mnemonic") and is_relevant:
                if ename.lower() not in seen_insts:
                    seen_insts.add(ename.lower())
                    instructions.append({
                        "name": ename,
                        "width": attrs.get("width", 32),
                        "description": getattr(ent, "description", ""),
                    })
            elif etype in ("opcode", "encoding") and is_relevant:
                if ename.lower() not in seen_opcodes:
                    seen_opcodes.add(ename.lower())
                    opcodes.append({
                        "mnemonic": ename,
                        "binary_encoding": attrs.get("binary_encoding", ""),
                        "operation": attrs.get("operation", ""),
                    })

        # Pull behavioral requirements for this component
        for req in getattr(req_ir, "requirements", []):
            stmt = getattr(req, "statement", "")
            subj = getattr(req, "subject", "")
            rtype = getattr(req, "type", "").upper()

            req_relevant = (
                comp_name.lower() in stmt.lower() or
                comp_name.lower() in subj.lower() or
                scope_type == "FULL_SYSTEM" or
                any(p["name"].lower() in stmt.lower() for p in ports)
            )

            if req_relevant:
                behavioral_rules.append({
                    "rule_id": getattr(req, "req_id", ""),
                    "statement": stmt,
                    "condition": getattr(req, "condition", ""),
                    "action": getattr(req, "action", ""),
                    "type": rtype,
                    "source_ref": getattr(req.source_location, "block_id", "") if getattr(req, "source_location", None) else "",
                })

    # Record specification gaps if critical elements are unstated
    if not ports and scope_type != "FULL_SYSTEM":
        gaps.append(f"No ports explicitly discovered for target component '{comp_name}'.")

    logger.info(
        f"Scope Resolved for '{comp_name}': {len(ports)} ports, {len(interfaces)} interfaces, "
        f"{len(registers)} registers, {len(opcodes)} opcodes, {len(behavioral_rules)} behavioral rules."
    )

    return ResolvedScope(
        target_component_name=comp_name,
        target_component_type=comp_type,
        clock_domain=clock_domain,
        reset_domain=reset_domain,
        ports=ports,
        interfaces=interfaces,
        registers=registers,
        instructions=instructions,
        opcodes=opcodes,
        fsms=fsms,
        behavioral_rules=behavioral_rules,
        connected_components=connected_components,
        source_chunk_ids=sorted(chunk_ids),
        specification_gaps=gaps,
    )
