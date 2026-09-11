"""
Step 6: Build the Knowledge Graph.

Primary backend: Neo4j (via the official `neo4j` Python driver), using
MERGE-based upserts so re-processing a document or adding new chunks
performs incremental updates rather than duplicating nodes.

Fallback backend: a small in-memory graph store with an identical public
interface, automatically used when `CONFIG.neo4j.enabled` is False or the
Neo4j server is unreachable. This keeps the Streamlit app fully functional
in environments without a Neo4j instance (e.g. quick local trials), while
still exercising the same graph-building / querying code paths.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

from config import CONFIG
from utils.models import Entity, Relationship
from utils.logger import get_logger
from core.efs_ir.models import EFSIR

logger = get_logger("graph.builder")

def _get_inmemory_graph_cache_path() -> str:
    return os.path.join(CONFIG.graph_cache_dir, "inmemory_graph.pkl")


class InMemoryGraphStore:
    """A minimal graph store mirroring the subset of Cypher operations we need.

    Persists itself to disk (data/graph_cache/inmemory_graph.pkl) after every
    mutation so it survives Streamlit/process restarts, not just reruns
    within a single long-lived server process.
    """

    def __init__(self):
        # node key: (type, name) -> node dict
        self.nodes: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # adjacency list of edges
        self.edges: List[Dict[str, Any]] = []
        # fast lookup for edges: (source, relation, target) -> edge dict
        self.edge_index: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        path = _get_inmemory_graph_cache_path()
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                    self.nodes = data.get("nodes", {})
                    self.edges = data.get("edges", [])
                    self.edge_index = {
                        (e["source"], e["relation"], e["target"]): e for e in self.edges
                    }
                logger.info(f"Loaded cached in-memory graph: {len(self.nodes)} nodes, {len(self.edges)} edges.")
            except Exception as e:
                logger.warning(f"Failed to load cached in-memory graph: {e}")

    def _persist(self) -> None:
        path = _get_inmemory_graph_cache_path()
        tmp_path = path + ".tmp"
        try:
            os.makedirs(CONFIG.graph_cache_dir, exist_ok=True)
            with open(tmp_path, "wb") as f:
                pickle.dump({"nodes": self.nodes, "edges": self.edges}, f)
            try:
                os.replace(tmp_path, path)
            except (PermissionError, OSError):
                import time
                time.sleep(0.05)
                try:
                    os.replace(tmp_path, path)
                except (PermissionError, OSError):
                    # Windows file lock fallback: write directly to target
                    with open(path, "wb") as f:
                        pickle.dump({"nodes": self.nodes, "edges": self.edges}, f)
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
        except Exception as e:
            logger.error(f"Failed to persist in-memory graph cache: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def merge_node(self, name: str, entity_type: str, props: Dict[str, Any], auto_persist: bool = True) -> None:
        key = (entity_type, name)
        existing = self.nodes.get(key)
        if existing:
            existing["chunk_ids"] = sorted(set(existing.get("chunk_ids", []) + props.get("chunk_ids", [])))
            existing["pages"] = sorted(set(existing.get("pages", []) + props.get("pages", [])))
            existing.setdefault("aliases", [])
            existing["aliases"] = sorted(set(existing["aliases"] + props.get("aliases", [])))
        else:
            node = {"name": name, "type": entity_type, **props}
        if auto_persist:
            self._persist()

    def clear_document(self, document_id: str) -> None:
        """Remove all graph nodes and edges associated with a specific document_id."""
        if not document_id:
            return
        keys_to_remove = [k for k, node in self.nodes.items() if node.get("doc_id") == document_id or node.get("document_id") == document_id]
        for k in keys_to_remove:
            del self.nodes[k]
        self.edges = [e for e in self.edges if e.get("doc_id") != document_id and e.get("document_id") != document_id]
        self.edge_index = {(e["source"], e["relation"], e["target"]): e for e in self.edges}
        self._persist()
        logger.info(f"Cleared graph nodes for document '{document_id}'.")

    def merge_edge(self, source: str, source_type: str, relation: str, target: str, target_type: str, props: Dict[str, Any], auto_persist: bool = True) -> None:
        edge_key = (source, relation, target)
        existing = self.edge_index.get(edge_key)
        if existing:
            existing["evidence"] = list(set(existing.get("evidence", []) + props.get("evidence", [])))
            if auto_persist:
                self._persist()
            return
        edge_dict = {
            "source": source, "source_type": source_type,
            "relation": relation,
            "target": target, "target_type": target_type,
            **props,
        }
        self.edges.append(edge_dict)
        self.edge_index[edge_key] = edge_dict
        if auto_persist:
            self._persist()

    def get_node(self, name: str) -> Optional[Dict[str, Any]]:
        for (etype, n), node in self.nodes.items():
            if n == name:
                return node
        return None

    def get_neighbors(self, name: str, hops: int = 1) -> Tuple[List[Dict], List[Dict]]:
        visited_nodes = {}
        visited_edges = []
        frontier = {name}
        for _ in range(max(1, hops)):
            next_frontier = set()
            for e in self.edges:
                if e["source"] in frontier or e["target"] in frontier:
                    visited_edges.append(e)
                    next_frontier.add(e["source"])
                    next_frontier.add(e["target"])
            frontier |= next_frontier
        for n in frontier:
            node = self.get_node(n)
            if node:
                visited_nodes[n] = node
        return list(visited_nodes.values()), visited_edges

    def all_nodes(self) -> List[Dict[str, Any]]:
        return list(self.nodes.values())

    def all_edges(self) -> List[Dict[str, Any]]:
        return list(self.edges)

    def deduplicate_nodes(self) -> Tuple[int, int]:
        """Merge duplicate nodes sharing the same (name.lower(), type) and consolidate edges."""
        canonical_map = {}
        merged_nodes = {}
        nodes_removed = 0

        for (etype, name), node_dict in list(self.nodes.items()):
            canon_key = (etype, name.lower())
            if canon_key not in canonical_map:
                canonical_map[canon_key] = (etype, name)
                merged_nodes[(etype, name)] = node_dict
            else:
                orig_type, orig_name = canonical_map[canon_key]
                target_node = merged_nodes[(orig_type, orig_name)]
                target_node["chunk_ids"] = sorted(set(target_node.get("chunk_ids", []) + node_dict.get("chunk_ids", [])))
                target_node["pages"] = sorted(set(target_node.get("pages", []) + node_dict.get("pages", [])))
                target_node["aliases"] = sorted(set(target_node.get("aliases", []) + node_dict.get("aliases", [])))
                nodes_removed += 1

        self.nodes = merged_nodes

        # Remap edges
        name_alias = {k[1]: canonical_map.get((k[0], k[1].lower()), (k[0], k[1]))[1] for k in list(self.nodes.keys())}
        new_edges = []
        seen_edges = set()
        for e in self.edges:
            src = name_alias.get(e["source"], e["source"])
            tgt = name_alias.get(e["target"], e["target"])
            if src == tgt:
                continue
            edge_key = (src, e["relation"], tgt)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                e["source"] = src
                e["target"] = tgt
                new_edges.append(e)

        edges_consolidated = len(self.edges) - len(new_edges)
        self.edges = new_edges
        self._persist()
        logger.info(f"Graph deduplication complete: removed {nodes_removed} duplicate nodes, consolidated {edges_consolidated} edges.")
        return nodes_removed, edges_consolidated

    def clear(self) -> None:
        self.nodes = {}
        self.edges = []
        self._persist()

    def stats(self) -> Dict[str, int]:
        return {"nodes": len(self.nodes), "edges": len(self.edges)}

    def search_nodes_by_keyword(self, keyword: str, limit: int = 20) -> List[Dict[str, Any]]:
        kw = keyword.lower()
        hits = [n for (t, name), n in self.nodes.items() if kw in name.lower() or any(kw in a.lower() for a in n.get("aliases", []))]
        return hits[:limit]


class Neo4jGraphStore:
    """Real Neo4j-backed store. Same public interface as InMemoryGraphStore."""

    def __init__(self):
        from neo4j import GraphDatabase  # imported lazily so it's optional
        cfg = CONFIG.neo4j
        self.driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
        self.database = cfg.database
        self._ensure_constraints()

    def _ensure_constraints(self) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                "CREATE CONSTRAINT entity_name_type IF NOT EXISTS "
                "FOR (n:Entity) REQUIRE (n.name, n.type) IS UNIQUE"
            )

    def merge_node(self, name: str, entity_type: str, props: Dict[str, Any]) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MERGE (n:Entity {name: $name, type: $type})
                ON CREATE SET n += $props
                ON MATCH SET n.chunk_ids = apoc.coll.toSet(coalesce(n.chunk_ids, []) + $chunk_ids),
                              n.pages = apoc.coll.toSet(coalesce(n.pages, []) + $pages),
                              n.aliases = apoc.coll.toSet(coalesce(n.aliases, []) + $aliases)
                """,
                name=name, type=entity_type, props=props,
                chunk_ids=props.get("chunk_ids", []), pages=props.get("pages", []),
                aliases=props.get("aliases", []),
            )

    def merge_edge(self, source: str, source_type: str, relation: str, target: str, target_type: str, props: Dict[str, Any]) -> None:
        with self.driver.session(database=self.database) as session:
            session.run(
                f"""
                MATCH (s:Entity {{name: $source, type: $source_type}})
                MATCH (t:Entity {{name: $target, type: $target_type}})
                MERGE (s)-[r:{_safe_rel(relation)}]->(t)
                ON CREATE SET r.evidence = $evidence
                """,
                source=source, source_type=source_type,
                target=target, target_type=target_type,
                evidence=props.get("evidence", []),
            )

    def deduplicate_nodes(self) -> Tuple[int, int]:
        with self.driver.session(database=self.database) as session:
            try:
                res = session.run(
                    "MATCH (n:Entity) "
                    "WITH toLower(n.name) AS name, n.type AS type, collect(n) AS nodes "
                    "WHERE size(nodes) > 1 "
                    "CALL apoc.refactor.mergeNodes(nodes, {properties: 'combine', mergeRels: true}) YIELD node "
                    "RETURN count(node) AS c"
                ).single()
                count = res["c"] if res else 0
                return count, 0
            except Exception:
                return 0, 0

    def clear(self) -> None:
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def get_neighbors(self, name: str, hops: int = 1) -> Tuple[List[Dict], List[Dict]]:
        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"""
                MATCH (n:Entity {{name: $name}})
                CALL apoc.path.subgraphAll(n, {{maxLevel: $hops}}) YIELD nodes, relationships
                RETURN nodes, relationships
                """,
                name=name, hops=hops,
            )
            record = result.single()
            if not record:
                return [], []
            nodes = [dict(n) for n in record["nodes"]]
            edges = [
                {"source": r.start_node["name"], "relation": r.type, "target": r.end_node["name"]}
                for r in record["relationships"]
            ]
            return nodes, edges

    def all_nodes(self) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            result = session.run("MATCH (n:Entity) RETURN n")
            return [dict(r["n"]) for r in result]

    def all_edges(self) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            result = session.run("MATCH (s:Entity)-[r]->(t:Entity) RETURN s.name AS source, type(r) AS relation, t.name AS target")
            return [dict(r) for r in result]

    def stats(self) -> Dict[str, int]:
        with self.driver.session(database=self.database) as session:
            n = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            e = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            return {"nodes": n, "edges": e}

    def search_nodes_by_keyword(self, keyword: str, limit: int = 20) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            result = session.run(
                "MATCH (n:Entity) WHERE toLower(n.name) CONTAINS toLower($kw) "
                "RETURN n LIMIT $limit",
                kw=keyword, limit=limit,
            )
            return [dict(r["n"]) for r in result]

    def close(self):
        self.driver.close()


def _safe_rel(relation: str) -> str:
    """Sanitize a relation type string for safe interpolation into Cypher."""
    return "".join(c for c in relation.upper() if c.isalnum() or c == "_") or "RELATED_TO"


_store_singleton = None


def get_graph_store():
    """Return the configured graph store, falling back to in-memory on any failure."""
    global _store_singleton
    if _store_singleton is not None:
        return _store_singleton

    if CONFIG.neo4j.enabled:
        try:
            _store_singleton = Neo4jGraphStore()
            logger.info("Connected to Neo4j graph store.")
            return _store_singleton
        except Exception as e:
            logger.warning(f"Neo4j unavailable ({e}); falling back to in-memory graph store.")

    _store_singleton = InMemoryGraphStore()
    logger.info("Using in-memory graph store.")
    return _store_singleton


def reset_graph_store():
    global _store_singleton
    _store_singleton = None


# --------------------------------------------------------------------------
# High level build API
# --------------------------------------------------------------------------

def build_graph(entities: List[Entity], relationships: List[Relationship], efs_ir: Optional[EFSIR] = None) -> Dict[str, int]:
    """Upsert all normalized entities and relationships into the graph store."""
    store = get_graph_store()

    # Scope and isolate graph store per active document
    active_doc_id = None
    if efs_ir and efs_ir.metadata and efs_ir.metadata.document_id:
        active_doc_id = efs_ir.metadata.document_id
    elif entities and getattr(entities[0], "doc_id", None):
        active_doc_id = entities[0].doc_id

    if active_doc_id and hasattr(store, "clear_document"):
        store.clear_document(active_doc_id)

    # group entity mentions by (name, type) so we build one node per merged entity
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for e in entities:
        key = (e.name, e.entity_type)
        g = grouped.setdefault(key, {"chunk_ids": set(), "pages": set(), "aliases": set(), "sample": e})
        g["chunk_ids"].add(e.chunk_id)
        g["pages"].add(e.page)
        g["aliases"].update(e.aliases)

    for (name, etype), g in grouped.items():
        sample: Entity = g["sample"]
        store.merge_node(
            name=name,
            entity_type=etype,
            props={
                "chunk_ids": sorted(g["chunk_ids"]),
                "pages": sorted(p for p in g["pages"] if p),
                "aliases": sorted(g["aliases"]),
                "chapter": sample.chapter,
                "section": sample.section,
                "doc_id": sample.doc_id,
                "original_text": sample.original_text,
            },
            auto_persist=False,
        )

    # need type lookup for edge endpoints
    type_by_name = {name: etype for (name, etype) in grouped.keys()}

    n_edges = 0
    for r in relationships:
        s_type = type_by_name.get(r.source)
        t_type = type_by_name.get(r.target)
        if not s_type or not t_type:
            continue
        store.merge_edge(
            source=r.source, source_type=s_type,
            relation=r.relation,
            target=r.target, target_type=t_type,
            props={"evidence": [r.evidence] if r.evidence else []},
            auto_persist=False,
        )
        n_edges += 1

    # Insert EFS IR canonical objects if available
    if efs_ir:
        logger.info("Inserting EFS IR objects into knowledge graph with canonical EFS IDs...")
        
        # Mapping helper: raw entity lower-case name to (raw_name, raw_type)
        raw_names_lower = {k[0].lower(): (k[0], k[1]) for k in grouped.keys()}
        
        # Helper to link raw entity to canonical EFS ID if matching name exists
        def link_raw_to_canonical(name_val: str, canonical_id: str, canonical_type: str):
            if not name_val:
                return
            n_lower = name_val.lower()
            if n_lower in raw_names_lower:
                raw_name, raw_type = raw_names_lower[n_lower]
                store.merge_edge(
                    source=raw_name, source_type=raw_type,
                    relation="MAPS_TO",
                    target=canonical_id, target_type=canonical_type,
                    props={"evidence": ["Matches canonical EFS IR object name"]}
                )

        # 1. Merge EFS IR components and sub-objects as nodes using name as primary key
        for comp in efs_ir.components:
            store.merge_node(comp.name, "Component", {
                "name": comp.name,
                "efs_id": comp.component_id,
                "component_type": comp.type,
                "description": comp.description,
                "doc_id": comp.traceability.doc_id or "",
                "chunk_id": comp.traceability.chunk_id or "",
                "original_text": comp.traceability.original_text or "",
            })
            link_raw_to_canonical(comp.name, comp.name, "Component")
            
        for iface in efs_ir.interfaces:
            store.merge_node(iface.name, "Interface", {
                "name": iface.name,
                "efs_id": iface.interface_id,
                "protocol": iface.protocol,
                "role": iface.role,
                "doc_id": iface.traceability.doc_id or "",
                "chunk_id": iface.traceability.chunk_id or "",
            })
            link_raw_to_canonical(iface.name, iface.name, "Interface")
            
        for sig in efs_ir.signals:
            store.merge_node(sig.name, "Signal", {
                "name": sig.name,
                "efs_id": sig.signal_id,
                "width": sig.width,
                "direction": sig.direction,
                "semantic_role": sig.semantic_role,
                "doc_id": sig.traceability.doc_id or "",
                "chunk_id": sig.traceability.chunk_id or "",
            })
            link_raw_to_canonical(sig.name, sig.name, "Signal")
            
        for reg in efs_ir.registers:
            store.merge_node(reg.name, "Register", {
                "name": reg.name,
                "efs_id": reg.register_id,
                "offset": reg.offset,
                "width": reg.width,
                "access_type": reg.access_type,
                "reset_value": reg.reset_value,
                "doc_id": reg.traceability.doc_id or "",
                "chunk_id": reg.traceability.chunk_id or "",
            })
            link_raw_to_canonical(reg.name, reg.name, "Register")
            
            for f in reg.fields:
                f_id = f"{reg.name}_{f.name}"
                store.merge_node(f_id, "Field", {
                    "name": f.name,
                    "msb": f.msb,
                    "lsb": f.lsb,
                    "access": f.access,
                    "reset_value": f.reset_value,
                })
                store.merge_edge(reg.name, "Register", "HAS_FIELD", f_id, "Field", {"evidence": ["Derived from register field definition"]})
                
        for fsm in efs_ir.fsms:
            store.merge_node(fsm.fsm_id, "FSM", {
                "name": fsm.name,
                "efs_id": fsm.fsm_id,
                "initial_state": fsm.initial_state,
                "doc_id": fsm.traceability.doc_id or "",
                "chunk_id": fsm.traceability.chunk_id or "",
            })
            link_raw_to_canonical(fsm.name, fsm.fsm_id, "FSM")
            
            for s in fsm.states:
                store.merge_node(s.state_id, "State", {
                    "name": s.name,
                    "efs_id": s.state_id
                })
                store.merge_edge(fsm.fsm_id, "FSM", "HAS_STATE", s.state_id, "State", {"evidence": ["FSM state list"]})
                
            for t in fsm.transitions:
                # Find state IDs based on names
                src_state_obj = next((s for s in fsm.states if s.name == t.source_state), None)
                tgt_state_obj = next((s for s in fsm.states if s.name == t.target_state), None)
                if src_state_obj and tgt_state_obj:
                    store.merge_edge(
                        src_state_obj.state_id, "State", "TRANSITIONS_TO",
                        tgt_state_obj.state_id, "State", {
                            "evidence": [t.condition or "unconditional"],
                            "efs_id": t.transition_id
                        }
                    )
                
        for const in efs_ir.constraints:
            store.merge_node(const.constraint_id, "Constraint", {
                "name": f"Constraint_{const.constraint_id}",
                "efs_id": const.constraint_id,
                "constraint_type": const.type,
                "condition": const.condition,
                "expected_behavior": const.expected_behavior,
                "doc_id": const.traceability.doc_id or "",
                "chunk_id": const.traceability.chunk_id or "",
            })
            
            # Link constraint to target signals/components if we can resolve them
            for target_obj in const.source_objects:
                # check signals by name
                sig_obj = next((s for s in efs_ir.signals if s.name == target_obj), None)
                if sig_obj:
                    store.merge_edge(const.constraint_id, "Constraint", "CONSTRAINS", sig_obj.signal_id, "Signal", {"evidence": ["Constraint mapping signal"]})
                else:
                    # check components by name
                    comp_obj = next((c for c in efs_ir.components if c.name == target_obj), None)
                    if comp_obj:
                        store.merge_edge(const.constraint_id, "Constraint", "CONSTRAINS", comp_obj.component_id, "Component", {"evidence": ["Constraint mapping component"]})

        # Extended models: instructions, opcodes, memory regions
        for instr in getattr(efs_ir, "instructions", []):
            store.merge_node(instr.instruction_id, "Instruction", {
                "name": instr.name,
                "efs_id": instr.instruction_id,
                "width": instr.width,
                "description": instr.description,
                "doc_id": instr.traceability.doc_id or "",
                "chunk_id": instr.traceability.chunk_id or "",
            })
            link_raw_to_canonical(instr.name, instr.instruction_id, "Instruction")
            
        for opc in getattr(efs_ir, "opcodes", []):
            store.merge_node(opc.opcode_id, "Opcode", {
                "name": opc.mnemonic,
                "efs_id": opc.opcode_id,
                "binary_encoding": opc.binary_encoding,
                "operation": opc.operation,
                "doc_id": opc.traceability.doc_id or "",
                "chunk_id": opc.traceability.chunk_id or "",
            })
            link_raw_to_canonical(opc.mnemonic, opc.opcode_id, "Opcode")
            
        for mr in getattr(efs_ir, "memory_regions", []):
            store.merge_node(mr.region_id, "MemoryRegion", {
                "name": mr.name,
                "efs_id": mr.region_id,
                "type": mr.type,
                "size": mr.size,
                "depth": mr.depth,
                "width": mr.width,
                "doc_id": mr.traceability.doc_id or "",
                "chunk_id": mr.traceability.chunk_id or "",
            })
            link_raw_to_canonical(mr.name, mr.region_id, "MemoryRegion")

        # 2. Merge EFS structural relationships using EFS IDs
        for comp in efs_ir.components:
            for if_id in comp.interfaces:
                iface = next((i for i in efs_ir.interfaces if i.interface_id == if_id), None)
                if iface:
                    store.merge_edge(
                        comp.component_id, "Component",
                        "HAS_INTERFACE",
                        iface.interface_id, "Interface",
                        {"evidence": ["Component interface declaration"]}
                    )
                    
        for iface in efs_ir.interfaces:
            for sig_id in iface.signals:
                sig = next((s for s in efs_ir.signals if s.signal_id == sig_id), None)
                if sig:
                    store.merge_edge(
                        iface.interface_id, "Interface",
                        "HAS_SIGNAL",
                        sig.signal_id, "Signal",
                        {"evidence": ["Interface signal declaration"]}
                    )

    if hasattr(store, "_persist"):
        store._persist()

    stats = store.stats()
    logger.info(f"Graph build complete: {stats}")
    return stats
