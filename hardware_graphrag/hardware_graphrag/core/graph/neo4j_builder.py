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

logger = get_logger("graph.builder")

_INMEMORY_GRAPH_CACHE_PATH = os.path.join(CONFIG.graph_cache_dir, "inmemory_graph.pkl")


class InMemoryGraphStore:
    """A minimal graph store mirroring the subset of Cypher operations we need.

    Persists itself to disk (data/graph_cache/inmemory_graph.pkl) after every
    mutation so it survives Streamlit/process restarts, not just reruns
    within a single long-lived server process.
    """

    def __init__(self):
        # node key: (type, name) -> node dict
        self.nodes: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # adjacency: (type, name) -> list of (relation, (type, name))
        self.edges: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(_INMEMORY_GRAPH_CACHE_PATH):
            try:
                with open(_INMEMORY_GRAPH_CACHE_PATH, "rb") as f:
                    data = pickle.load(f)
                    self.nodes = data.get("nodes", {})
                    self.edges = data.get("edges", [])
                logger.info(f"Loaded cached in-memory graph: {len(self.nodes)} nodes, {len(self.edges)} edges.")
            except Exception as e:
                logger.warning(f"Failed to load cached in-memory graph: {e}")

    def _persist(self) -> None:
        try:
            os.makedirs(CONFIG.graph_cache_dir, exist_ok=True)
            with open(_INMEMORY_GRAPH_CACHE_PATH, "wb") as f:
                pickle.dump({"nodes": self.nodes, "edges": self.edges}, f)
        except Exception as e:
            logger.error(f"Failed to persist in-memory graph cache: {e}")

    def merge_node(self, name: str, entity_type: str, props: Dict[str, Any]) -> None:
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
        self._persist()

    def merge_edge(self, source: str, source_type: str, relation: str, target: str, target_type: str, props: Dict[str, Any]) -> None:
        for e in self.edges:
            if (e["source"] == source and e["target"] == target and e["relation"] == relation):
                e["evidence"] = list(set(e.get("evidence", []) + props.get("evidence", [])))
                self._persist()
                return
        self.edges.append({
            "source": source, "source_type": source_type,
            "relation": relation,
            "target": target, "target_type": target_type,
            **props,
        })
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

def build_graph(entities: List[Entity], relationships: List[Relationship]) -> Dict[str, int]:
    """Upsert all normalized entities and relationships into the graph store."""
    store = get_graph_store()

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
        )
        n_edges += 1

    stats = store.stats()
    logger.info(f"Graph build complete: {stats}")
    return stats
