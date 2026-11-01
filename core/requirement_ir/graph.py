"""
Requirement Dependency Graph Implementation (Sections 8 & 9).

Connects Entities, Atomic Requirements, Conditions, Events, and Actions.
Provides N-hop dependency expansion for query-driven selective retrieval.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple
from core.requirement_ir.models import (
    AtomicRequirement, DiscoveredEntity, RequirementGraphEdge, RequirementIR
)
from utils.logger import get_logger

logger = get_logger("requirement_ir.graph")


class RequirementDependencyGraph:
    """In-memory Graph structure managing requirement & entity relationships."""

    def __init__(self, req_ir: RequirementIR = None):
        self.nodes: Dict[str, Dict] = {}        # node_id -> metadata
        self.adj_out: Dict[str, List[Tuple[str, str]]] = {} # node_id -> [(target_id, rel_type)]
        self.adj_in: Dict[str, List[Tuple[str, str]]] = {}  # node_id -> [(source_id, rel_type)]
        self.edges: List[RequirementGraphEdge] = []

        if req_ir:
            self.build_from_ir(req_ir)

    def add_node(self, node_id: str, node_type: str, data: Dict = None) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = {"id": node_id, "type": node_type, "data": data or {}}
            self.adj_out[node_id] = []
            self.adj_in[node_id] = []

    def add_edge(self, source: str, relationship: str, target: str, edge_obj: RequirementGraphEdge = None) -> None:
        self.add_node(source, "ENTITY" if not source.startswith("REQ_") else "REQUIREMENT")
        self.add_node(target, "ENTITY" if not target.startswith("REQ_") else "REQUIREMENT")

        self.adj_out[source].append((target, relationship))
        self.adj_in[target].append((source, relationship))

        if edge_obj:
            self.edges.append(edge_obj)
        else:
            self.edges.append(RequirementGraphEdge(source=source, relationship=relationship, target=target))

    def build_from_ir(self, req_ir: RequirementIR) -> None:
        """Populate graph from a complete RequirementIR object."""
        # 1. Add Entity nodes
        for ent in req_ir.entities:
            self.add_node(ent.entity_id, "ENTITY", ent.to_dict())
            # Map canonical name to entity_id as well
            self.add_node(ent.name, "ENTITY_NAME", {"entity_id": ent.entity_id})

        # 2. Add Requirement nodes and connecting edges
        for req in req_ir.requirements:
            self.add_node(req.requirement_id, "REQUIREMENT", req.to_dict())

            # Connect requirement to its entities
            for ent_name in req.entities:
                self.add_edge(req.requirement_id, "USES", ent_name)

            if req.subject:
                self.add_edge(req.requirement_id, "SUBJECT", req.subject)

            # Connect requirement dependencies
            for dep_id in req.dependencies:
                self.add_edge(req.requirement_id, "DEPENDS_ON", dep_id)

        # 3. Add explicit edges
        for edge in req_ir.edges:
            self.add_edge(edge.source, edge.relationship, edge.target, edge)

    def get_related_requirements(self, seed_ids: List[str], max_depth: int = 2) -> Set[str]:
        """Expand seed requirement or entity IDs using N-hop graph traversal."""
        visited: Set[str] = set()
        frontier: Set[str] = set(seed_ids)

        for _ in range(max_depth):
            next_frontier: Set[str] = set()
            for current in frontier:
                if current in visited:
                    continue
                visited.add(current)

                # Traverse outgoing edges
                for neighbor, _ in self.adj_out.get(current, []):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)

                # Traverse incoming edges
                for neighbor, _ in self.adj_in.get(current, []):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)

            frontier = next_frontier

        visited.update(frontier)

        # Filter nodes to return requirement IDs
        req_ids = {node for node in visited if node.startswith("REQ_") or "requirement_id" in self.nodes.get(node, {}).get("data", {})}
        return req_ids

    def get_dependencies_for_requirement(self, req_id: str) -> List[str]:
        """Find all upstream/downstream requirement dependencies for a specific requirement."""
        deps = []
        for target, rel in self.adj_out.get(req_id, []):
            if rel in ("DEPENDS_ON", "PRECEDES", "TRIGGERS", "USES", "FOLLOWED_BY"):
                deps.append(target)
        return deps
