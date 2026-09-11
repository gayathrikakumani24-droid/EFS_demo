"""
Lightweight persistence layer for pipeline outputs.

Streamlit reruns the whole script on every interaction, and a fresh
browser tab starts a new session_state. The vector store and graph
store already persist themselves to disk, but the raw Chunk / Entity /
Relationship / ParsedDocument objects (needed by the Chunk Explorer and
Entity Explorer pages for filtering/browsing) are only produced during
ingestion. This module pickles them to data/graph_cache/registry.pkl so
they survive across reruns and app restarts without needing a real
database.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List

from config import CONFIG
from utils.models import Chunk, Entity, ParsedDocument, Relationship
from utils.logger import get_logger

logger = get_logger("registry")

def _get_registry_path() -> str:
    return os.path.join(CONFIG.graph_cache_dir, "registry.pkl")


@dataclass
class Registry:
    documents: Dict[str, ParsedDocument] = field(default_factory=dict)   # doc_id -> ParsedDocument
    chunks: Dict[str, List[Chunk]] = field(default_factory=dict)         # doc_id -> chunks
    entities: Dict[str, List[Entity]] = field(default_factory=dict)      # doc_id -> entities
    relationships: Dict[str, List[Relationship]] = field(default_factory=dict)  # doc_id -> relationships
    errors: Dict[str, List[str]] = field(default_factory=dict)           # doc_id -> error messages
    active_req_ir: Optional[Any] = None
    active_efs_ir: Optional[Any] = None

    def all_chunks(self) -> List[Chunk]:
        out: List[Chunk] = []
        for c in self.chunks.values():
            out.extend(c)
        return out

    def all_entities(self) -> List[Entity]:
        out: List[Entity] = []
        for e in self.entities.values():
            out.extend(e)
        return out

    def all_relationships(self) -> List[Relationship]:
        out: List[Relationship] = []
        for r in self.relationships.values():
            out.extend(r)
        return out


_singleton: Registry = None


def get_registry() -> Registry:
    global _singleton
    if _singleton is None:
        _singleton = _load()
    return _singleton


def reset_registry() -> None:
    global _singleton
    _singleton = None


def _load() -> Registry:
    path = _get_registry_path()
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load registry cache, starting fresh: {e}")
    return Registry()


def save_registry() -> None:
    registry = get_registry()
    path = _get_registry_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(registry, f)
    except Exception as e:
        logger.error(f"Failed to persist registry cache: {e}")


def register_result(result) -> None:
    """Store a PipelineResult (see core.pipeline) into the registry and persist it."""
    registry = get_registry()
    doc_id = result.doc.doc_id
    registry.documents[doc_id] = result.doc
    registry.chunks[doc_id] = result.chunks
    registry.entities[doc_id] = result.entities
    registry.relationships[doc_id] = result.relationships
    registry.errors[doc_id] = result.errors
    if getattr(result, "req_ir", None):
        registry.active_req_ir = result.req_ir
    if getattr(result, "efs_ir", None):
        registry.active_efs_ir = result.efs_ir
    save_registry()


def clear_registry() -> None:
    global _singleton
    _singleton = Registry()
    path = _get_registry_path()
    if os.path.exists(path):
        os.remove(path)
