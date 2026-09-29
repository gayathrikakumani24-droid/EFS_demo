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
            self.nodes[key] = node
        if auto_persist:
            self._persist()

    def clear_document(self, document_id: str) -> None:
        """Remove all graph nodes and edges associated with a specific document_id."""
        if not document_id:
            return
        keys_to_remove = [k for k, node in self.nodes.items() if node.get("doc_id") == document_id or node.get("document_id") == document_id]
        for k in keys_to_remove:
            del self.nodes[k]
        valid_node_names = {n["name"] for n in self.nodes.values()}
        self.edges = [
            e for e in self.edges
            if e.get("doc_id") != document_id and e.get("document_id") != document_id
            and e["source"] in valid_node_names and e["target"] in valid_node_names
        ]
        self.edge_index = {(e["source"], e["relation"], e["target"]): e for e in self.edges}
        self._persist()
        logger.info(f"Cleared graph nodes and orphan edges for document '{document_id}'.")

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
                if raw_name != canonical_id or raw_type != canonical_type:
                    store.merge_edge(
                        source=raw_name, source_type=raw_type,
                        relation="MAPS_TO",
                        target=canonical_id, target_type=canonical_type,
                        props={"evidence": ["Matches canonical EFS IR object name"], "doc_id": active_doc_id or ""}
                    )

        # ------------------------------------------------------------------
        # 0. Design Node (Root of the hardware system)
        # ------------------------------------------------------------------
        design_name = (
            (efs_ir.metadata.display_name if efs_ir.metadata else None) or 
            (efs_ir.metadata.design_name if efs_ir.metadata else None) or 
            (efs_ir.metadata.top_module if efs_ir.metadata else None) or 
            "TopDesign"
        )
        design_id = (efs_ir.metadata.design_id if efs_ir.metadata else None) or "DSN_TOP"
        store.merge_node(design_name, "Design", {
            "name": design_name,
            "efs_id": design_id,
            "top_module": efs_ir.metadata.top_module if efs_ir.metadata else "",
            "protocol": efs_ir.metadata.protocol if efs_ir.metadata else "",
            "protocol_version": efs_ir.metadata.protocol_version if efs_ir.metadata else "",
            "description": efs_ir.metadata.description if efs_ir.metadata else "",
            "doc_id": active_doc_id or "",
            "aliases": efs_ir.metadata.aliases if efs_ir.metadata else [],
        })
        link_raw_to_canonical(design_name, design_name, "Design")
        if efs_ir.metadata:
            for alias in efs_ir.metadata.aliases:
                link_raw_to_canonical(alias, design_name, "Design")

        # ------------------------------------------------------------------
        # 1. Components
        # ------------------------------------------------------------------
        comp_id_to_name: Dict[str, str] = {}
        for comp in efs_ir.components:
            comp_name = comp.name or comp.component_id
            comp_id_to_name[comp.component_id] = comp_name
            comp_id_to_name[comp.name] = comp_name

            store.merge_node(comp_name, "Component", {
                "name": comp_name,
                "efs_id": comp.component_id,
                "component_type": comp.type,
                "description": comp.description,
                "clock_domain": comp.clock_domain or "",
                "reset_domain": comp.reset_domain or "",
                "doc_id": comp.traceability.doc_id or active_doc_id or "",
                "chunk_id": comp.traceability.chunk_id or "",
                "original_text": comp.traceability.original_text or "",
            })
            link_raw_to_canonical(comp.name, comp_name, "Component")
            if comp.component_id != comp.name:
                link_raw_to_canonical(comp.component_id, comp_name, "Component")

            # Design -> CONTAINS_COMPONENT -> Component
            if comp_name != design_name:
                store.merge_edge(
                    design_name, "Design", "CONTAINS_COMPONENT",
                    comp_name, "Component",
                    {"evidence": ["Top-level architectural component"]}
                )

        # Identify top-level subsystem/component if one exists
        top_comp = None
        for comp in efs_ir.components:
            c_name_l = comp.name.lower()
            if any(k in c_name_l for k in ["unit", "subsystem", "system", "top", "core"]) and not any(k in c_name_l for k in ["alu", "file", "scheduler", "engine", "queue", "fpu"]):
                top_comp = comp
                break
        if not top_comp and efs_ir.components:
            top_comp = efs_ir.components[0]
        top_comp_name = comp_id_to_name.get(top_comp.component_id, top_comp.name) if top_comp else None

        if top_comp_name:
            for comp in efs_ir.components:
                c_name = comp_id_to_name.get(comp.component_id, comp.name)
                if c_name != top_comp_name:
                    store.merge_edge(
                        top_comp_name, "Component", "CONTAINS_COMPONENT",
                        c_name, "Component",
                        {"evidence": ["Subsystem submodule component"]}
                    )

        # ------------------------------------------------------------------
        # 2. Interfaces
        # ------------------------------------------------------------------
        iface_id_to_name: Dict[str, str] = {}
        for iface in efs_ir.interfaces:
            iface_name = iface.name or iface.interface_id
            iface_id_to_name[iface.interface_id] = iface_name
            iface_id_to_name[iface.name] = iface_name

            store.merge_node(iface_name, "Interface", {
                "name": iface_name,
                "efs_id": iface.interface_id,
                "protocol": iface.protocol,
                "role": iface.role,
                "description": iface.description,
                "doc_id": iface.traceability.doc_id or active_doc_id or "",
                "chunk_id": iface.traceability.chunk_id or "",
            })
            link_raw_to_canonical(iface.name, iface_name, "Interface")
            if iface.interface_id != iface.name:
                link_raw_to_canonical(iface.interface_id, iface_name, "Interface")

        # Connect Components to Interfaces
        for comp in efs_ir.components:
            comp_name = comp_id_to_name.get(comp.component_id, comp.name or comp.component_id)
            for if_ref in comp.interfaces:
                if_name = iface_id_to_name.get(if_ref)
                if not if_name:
                    iface_obj = next((i for i in efs_ir.interfaces if i.interface_id == if_ref or i.name == if_ref), None)
                    if iface_obj:
                        if_name = iface_obj.name or iface_obj.interface_id
                if if_name:
                    store.merge_edge(
                        comp_name, "Component", "HAS_INTERFACE",
                        if_name, "Interface",
                        {"evidence": ["Component interface declaration"]}
                    )

        for iface in efs_ir.interfaces:
            iface_name = iface_id_to_name.get(iface.interface_id, iface.name or iface.interface_id)
            if iface.source_component:
                src_name = comp_id_to_name.get(iface.source_component)
                if src_name:
                    store.merge_edge(
                        src_name, "Component", "EXPOSES_INTERFACE",
                        iface_name, "Interface",
                        {"evidence": ["Interface source component"]}
                    )
            if iface.destination_component:
                dst_name = comp_id_to_name.get(iface.destination_component)
                if dst_name:
                    store.merge_edge(
                        iface_name, "Interface", "CONNECTS_TO",
                        dst_name, "Component",
                        {"evidence": ["Interface destination component"]}
                    )

        # ------------------------------------------------------------------
        # 3. Signals (including Clocks and Resets)
        # ------------------------------------------------------------------
        all_signals_map: Dict[str, Any] = {}
        for s in efs_ir.signals:
            all_signals_map[s.signal_id] = s
        for s in getattr(efs_ir, "clocks", []):
            if s.signal_id not in all_signals_map:
                all_signals_map[s.signal_id] = s
        for s in getattr(efs_ir, "resets", []):
            if s.signal_id not in all_signals_map:
                all_signals_map[s.signal_id] = s

        sig_id_to_name: Dict[str, str] = {}
        for sig in all_signals_map.values():
            sig_name = sig.name or sig.signal_id
            sig_id_to_name[sig.signal_id] = sig_name
            sig_id_to_name[sig.name] = sig_name

            store.merge_node(sig_name, "Signal", {
                "name": sig_name,
                "efs_id": sig.signal_id,
                "width": sig.width,
                "direction": sig.direction,
                "datatype": sig.datatype,
                "semantic_role": sig.semantic_role,
                "protocol_role": sig.protocol_role or "",
                "doc_id": sig.traceability.doc_id or active_doc_id or "",
                "chunk_id": sig.traceability.chunk_id or "",
            })
            link_raw_to_canonical(sig.name, sig_name, "Signal")
            if sig.signal_id != sig.name:
                link_raw_to_canonical(sig.signal_id, sig_name, "Signal")

            # Interface association
            if sig.interface:
                if_name = iface_id_to_name.get(sig.interface)
                if if_name:
                    store.merge_edge(
                        if_name, "Interface", "HAS_SIGNAL",
                        sig_name, "Signal",
                        {"evidence": ["Signal belongs to interface"]}
                    )

            # Driver (owner) and Consumer
            if sig.owner:
                owner_name = comp_id_to_name.get(sig.owner)
                if owner_name:
                    store.merge_edge(
                        owner_name, "Component", "DRIVES",
                        sig_name, "Signal",
                        {"evidence": ["Component drives signal"]}
                    )
                    store.merge_edge(
                        owner_name, "Component", "HAS_SIGNAL",
                        sig_name, "Signal",
                        {"evidence": ["Component port/signal"]}
                    )
            elif top_comp_name:
                store.merge_edge(
                    top_comp_name, "Component", "HAS_SIGNAL",
                    sig_name, "Signal",
                    {"evidence": ["Subsystem port/signal"]}
                )
            if sig.consumer:
                consumer_name = comp_id_to_name.get(sig.consumer)
                if consumer_name:
                    store.merge_edge(
                        sig_name, "Signal", "FEEDS",
                        consumer_name, "Component",
                        {"evidence": ["Signal feeds component"]}
                    )

        # Interface -> HAS_SIGNAL edges from iface.signals
        for iface in efs_ir.interfaces:
            iface_name = iface_id_to_name.get(iface.interface_id, iface.name or iface.interface_id)
            for sig_ref in iface.signals:
                sig_name = sig_id_to_name.get(sig_ref)
                if sig_name:
                    store.merge_edge(
                        iface_name, "Interface", "HAS_SIGNAL",
                        sig_name, "Signal",
                        {"evidence": ["Interface signal declaration"]}
                    )

        # Clock & Reset wiring to components
        for sig in all_signals_map.values():
            sig_name = sig_id_to_name.get(sig.signal_id, sig.name or sig.signal_id)
            is_clk = (
                sig.semantic_role == "clock" or 
                "clk" in sig_name.lower() or 
                any(c.name == sig.name for c in getattr(efs_ir, "clocks", []))
            )
            is_rst = (
                sig.semantic_role == "reset" or 
                "rst" in sig_name.lower() or 
                any(r.name == sig.name for r in getattr(efs_ir, "resets", []))
            )

            if is_clk:
                for comp in efs_ir.components:
                    comp_name = comp_id_to_name.get(comp.component_id, comp.name or comp.component_id)
                    if not comp.clock_domain or comp.clock_domain == sig_name or comp.clock_domain == sig.signal_id:
                        store.merge_edge(
                            sig_name, "Signal", "CLOCKS",
                            comp_name, "Component",
                            {"evidence": ["Clock feeds component"]}
                        )
            if is_rst:
                for comp in efs_ir.components:
                    comp_name = comp_id_to_name.get(comp.component_id, comp.name or comp.component_id)
                    if not comp.reset_domain or comp.reset_domain == sig_name or comp.reset_domain == sig.signal_id:
                        store.merge_edge(
                            sig_name, "Signal", "RESETS",
                            comp_name, "Component",
                            {"evidence": ["Reset resets component"]}
                        )

        # ------------------------------------------------------------------
        # 4. Registers and Fields
        # ------------------------------------------------------------------
        reg_id_to_name: Dict[str, str] = {}
        for reg in efs_ir.registers:
            reg_name = reg.name or reg.register_id
            reg_id_to_name[reg.register_id] = reg_name
            reg_id_to_name[reg.name] = reg_name

            store.merge_node(reg_name, "Register", {
                "name": reg_name,
                "efs_id": reg.register_id,
                "address": reg.address,
                "offset": reg.offset,
                "width": reg.width,
                "access_type": reg.access_type,
                "reset_value": reg.reset_value,
                "description": reg.description,
                "doc_id": reg.traceability.doc_id or active_doc_id or "",
                "chunk_id": reg.traceability.chunk_id or "",
            })
            link_raw_to_canonical(reg.name, reg_name, "Register")
            if reg.register_id != reg.name:
                link_raw_to_canonical(reg.register_id, reg_name, "Register")

            # Owner component / Design edge
            if reg.owner and comp_id_to_name.get(reg.owner):
                owner_name = comp_id_to_name[reg.owner]
                store.merge_edge(
                    owner_name, "Component", "HAS_REGISTER",
                    reg_name, "Register",
                    {"evidence": ["Register owner component"]}
                )
            elif top_comp_name:
                store.merge_edge(
                    top_comp_name, "Component", "HAS_REGISTER",
                    reg_name, "Register",
                    {"evidence": ["Subsystem register"]}
                )
            else:
                rf_comp = next((c for c in efs_ir.components if "reg" in c.type.lower() or "reg" in c.name.lower() or "ctrl" in c.name.lower() or "controller" in c.name.lower()), None)
                parent_target = (comp_id_to_name.get(rf_comp.component_id, rf_comp.name)) if rf_comp else design_name
                parent_type = "Component" if rf_comp else "Design"
                store.merge_edge(
                    parent_target, parent_type, "HAS_REGISTER",
                    reg_name, "Register",
                    {"evidence": ["Register mapping"]}
                )

            if top_comp_name:
                store.merge_edge(
                    top_comp_name, "Component", "HAS_REGISTER",
                    reg_name, "Register",
                    {"evidence": ["Subsystem register"]}
                )

            # Bitfields
            for f in reg.fields:
                f_name = f.name or f"field_{f.msb}_{f.lsb}"
                f_node_name = f"{reg_name}_{f_name}"
                store.merge_node(f_node_name, "Field", {
                    "name": f_node_name,
                    "field_name": f_name,
                    "parent_register": reg_name,
                    "msb": f.msb,
                    "lsb": f.lsb,
                    "width": f.width,
                    "access": f.access,
                    "reset_value": f.reset_value,
                    "description": f.description,
                })
                store.merge_edge(
                    reg_name, "Register", "HAS_FIELD",
                    f_node_name, "Field",
                    {"evidence": ["Derived from register field definition"]}
                )

        # ------------------------------------------------------------------
        # 5. Transactions
        # ------------------------------------------------------------------
        for tx in getattr(efs_ir, "transactions", []):
            tx_name = tx.name or tx.transaction_id
            store.merge_node(tx_name, "Transaction", {
                "name": tx_name,
                "efs_id": tx.transaction_id,
                "transaction_type": tx.type,
                "protocol": tx.protocol,
                "description": tx.description,
                "completion_condition": tx.completion_condition,
                "timing": tx.timing,
                "ordering": tx.ordering,
                "doc_id": tx.traceability.doc_id or active_doc_id or "",
                "chunk_id": tx.traceability.chunk_id or "",
            })
            link_raw_to_canonical(tx.name, tx_name, "Transaction")
            if tx.transaction_id != tx.name:
                link_raw_to_canonical(tx.transaction_id, tx_name, "Transaction")

            # Initiator
            if tx.initiator:
                init_comp_name = comp_id_to_name.get(tx.initiator)
                if init_comp_name:
                    store.merge_edge(
                        init_comp_name, "Component", "INITIATES",
                        tx_name, "Transaction",
                        {"evidence": ["Transaction initiator component"]}
                    )
                else:
                    init_if_name = iface_id_to_name.get(tx.initiator)
                    if init_if_name:
                        store.merge_edge(
                            init_if_name, "Interface", "INITIATES",
                            tx_name, "Transaction",
                            {"evidence": ["Transaction initiator interface"]}
                        )

            # Target
            if tx.target:
                tgt_comp_name = comp_id_to_name.get(tx.target)
                if tgt_comp_name:
                    store.merge_edge(
                        tx_name, "Transaction", "TARGETS",
                        tgt_comp_name, "Component",
                        {"evidence": ["Transaction target component"]}
                    )
                else:
                    tgt_if_name = iface_id_to_name.get(tx.target)
                    if tgt_if_name:
                        store.merge_edge(
                            tx_name, "Transaction", "TARGETS",
                            tgt_if_name, "Interface",
                            {"evidence": ["Transaction target interface"]}
                        )

            # Involved Signals
            for s_ref in tx.signals:
                s_name = sig_id_to_name.get(s_ref)
                if s_name:
                    store.merge_edge(
                        tx_name, "Transaction", "INVOLVES_SIGNAL",
                        s_name, "Signal",
                        {"evidence": ["Transaction involves signal"]}
                    )

            # Participating Interfaces
            for i_ref in tx.participating_interfaces:
                i_name = iface_id_to_name.get(i_ref)
                if i_name:
                    store.merge_edge(
                        tx_name, "Transaction", "USES_INTERFACE",
                        i_name, "Interface",
                        {"evidence": ["Transaction participating interface"]}
                    )

        # ------------------------------------------------------------------
        # 6. FSMs and States
        # ------------------------------------------------------------------
        for fsm in efs_ir.fsms:
            fsm_name = fsm.name or fsm.fsm_id
            store.merge_node(fsm_name, "FSM", {
                "name": fsm_name,
                "efs_id": fsm.fsm_id,
                "initial_state": fsm.initial_state,
                "encoding": fsm.encoding,
                "doc_id": fsm.traceability.doc_id or active_doc_id or "",
                "chunk_id": fsm.traceability.chunk_id or "",
            })
            link_raw_to_canonical(fsm.name, fsm_name, "FSM")
            if fsm.fsm_id != fsm.name:
                link_raw_to_canonical(fsm.fsm_id, fsm_name, "FSM")

            # Link to owner component if name overlaps
            fsm_comp = next((c for c in efs_ir.components if c.name.lower() in fsm_name.lower() or fsm_name.lower() in c.name.lower()), None)
            if fsm_comp:
                store.merge_edge(
                    comp_id_to_name.get(fsm_comp.component_id, fsm_comp.name), "Component", "IMPLEMENTS_FSM",
                    fsm_name, "FSM",
                    {"evidence": ["Component implements FSM"]}
                )

            # States
            state_node_map: Dict[str, str] = {}
            for s in fsm.states:
                s_node_name = f"{fsm_name}_{s.name}" if s.name else s.state_id
                state_node_map[s.name] = s_node_name
                state_node_map[s.state_id] = s_node_name

                store.merge_node(s_node_name, "State", {
                    "name": s_node_name,
                    "state_name": s.name,
                    "fsm": fsm_name,
                    "efs_id": s.state_id,
                    "encoding": s.encoding,
                })
                store.merge_edge(
                    fsm_name, "FSM", "HAS_STATE",
                    s_node_name, "State",
                    {"evidence": ["FSM state list"]}
                )

            # Transitions
            for t in fsm.transitions:
                src_node = state_node_map.get(t.source_state)
                tgt_node = state_node_map.get(t.target_state)
                if src_node and tgt_node:
                    store.merge_edge(
                        src_node, "State", "TRANSITIONS_TO",
                        tgt_node, "State", {
                            "evidence": [t.condition or "unconditional"],
                            "efs_id": t.transition_id,
                            "action": t.action,
                        }
                    )

        # ------------------------------------------------------------------
        # 7. Constraints, Timing Rules, Protocol Rules, Flows
        # ------------------------------------------------------------------
        for const in efs_ir.constraints:
            const_name = f"Constraint_{const.constraint_id}"
            store.merge_node(const_name, "Constraint", {
                "name": const_name,
                "efs_id": const.constraint_id,
                "constraint_type": const.type,
                "severity": const.severity,
                "condition": const.condition,
                "expected_behavior": const.expected_behavior,
                "doc_id": const.traceability.doc_id or active_doc_id or "",
                "chunk_id": const.traceability.chunk_id or "",
            })
            for target_obj in (const.source_objects + const.target_objects):
                target_sig_name = sig_id_to_name.get(target_obj)
                if target_sig_name:
                    store.merge_edge(const_name, "Constraint", "CONSTRAINS", target_sig_name, "Signal", {"evidence": ["Constraint mapping signal"]})
                target_comp_name = comp_id_to_name.get(target_obj)
                if target_comp_name:
                    store.merge_edge(const_name, "Constraint", "CONSTRAINS", target_comp_name, "Component", {"evidence": ["Constraint mapping component"]})

        for tr in getattr(efs_ir, "timing_rules", []):
            tr_name = f"Timing_{tr.rule_id}"
            store.merge_node(tr_name, "TimingConstraint", {
                "name": tr_name,
                "efs_id": tr.rule_id,
                "timing_type": tr.type,
                "min_delay": tr.minimum_delay,
                "max_delay": tr.maximum_delay,
                "ordering": tr.ordering,
                "doc_id": tr.traceability.doc_id or active_doc_id or "",
                "chunk_id": tr.traceability.chunk_id or "",
            })
            store.merge_edge(design_name, "Design", "HAS_TIMING_RULE", tr_name, "TimingConstraint", {"evidence": ["Design timing rule"]})

        for flow in getattr(efs_ir, "flows", []):
            flow_name = flow.name or flow.flow_id
            store.merge_node(flow_name, "Flow", {
                "name": flow_name,
                "efs_id": flow.flow_id,
                "conditions": flow.conditions,
                "doc_id": flow.traceability.doc_id or active_doc_id or "",
                "chunk_id": flow.traceability.chunk_id or "",
            })
            for c_ref in flow.components:
                c_name = comp_id_to_name.get(c_ref)
                if c_name:
                    store.merge_edge(flow_name, "Flow", "INVOLVES_COMPONENT", c_name, "Component", {"evidence": ["Flow component participation"]})
            for i_ref in flow.interfaces:
                i_name = iface_id_to_name.get(i_ref)
                if i_name:
                    store.merge_edge(flow_name, "Flow", "INVOLVES_INTERFACE", i_name, "Interface", {"evidence": ["Flow interface participation"]})

        # ------------------------------------------------------------------
        # 8. Extended Canonical Models: Instructions, Opcodes, Memory Regions
        # ------------------------------------------------------------------
        for instr in getattr(efs_ir, "instructions", []):
            instr_name = instr.name or instr.instruction_id
            chunk = instr.traceability.chunk_id or ""
            store.merge_node(instr_name, "Instruction", {
                "name": instr_name,
                "efs_id": instr.instruction_id,
                "width": instr.width,
                "description": instr.description,
                "doc_id": instr.traceability.doc_id or active_doc_id or "",
                "chunk_id": chunk,
                "chunk_ids": [chunk] if chunk else [],
            })
            link_raw_to_canonical(instr.name, instr_name, "Instruction")
            if top_comp_name:
                store.merge_edge(
                    top_comp_name, "Component", "EXECUTES_INSTRUCTION",
                    instr_name, "Instruction",
                    {"evidence": ["Subsystem instruction set"]}
                )
            
        for opc in getattr(efs_ir, "opcodes", []):
            opc_name = opc.mnemonic or opc.opcode_id
            chunk = opc.traceability.chunk_id or ""
            store.merge_node(opc_name, "Opcode", {
                "name": opc_name,
                "efs_id": opc.opcode_id,
                "binary_encoding": opc.binary_encoding,
                "operation": opc.operation,
                "doc_id": opc.traceability.doc_id or active_doc_id or "",
                "chunk_id": chunk,
                "chunk_ids": [chunk] if chunk else [],
            })
            link_raw_to_canonical(opc.mnemonic, opc_name, "Opcode")
            if top_comp_name:
                store.merge_edge(
                    top_comp_name, "Component", "SUPPORTS_OPCODE",
                    opc_name, "Opcode",
                    {"evidence": ["Subsystem opcode"]}
                )
            
        for mr in getattr(efs_ir, "memory_regions", []):
            mr_name = mr.name or mr.region_id
            store.merge_node(mr_name, "MemoryRegion", {
                "name": mr_name,
                "efs_id": mr.region_id,
                "type": mr.type,
                "size": mr.size,
                "depth": mr.depth,
                "width": mr.width,
                "doc_id": mr.traceability.doc_id or active_doc_id or "",
                "chunk_id": mr.traceability.chunk_id or "",
            })
            link_raw_to_canonical(mr.name, mr_name, "MemoryRegion")
            if top_comp_name:
                store.merge_edge(
                    top_comp_name, "Component", "HAS_MEMORY_REGION",
                    mr_name, "MemoryRegion",
                    {"evidence": ["Subsystem memory resource"]}
                )

    if hasattr(store, "_persist"):
        store._persist()

    stats = store.stats()
    logger.info(f"Graph build complete: {stats}")
    return stats
