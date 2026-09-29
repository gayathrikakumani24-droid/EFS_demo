"""
Chunk Extraction Cache & Deduplication Module (Step 9 & Step 10).

Caches chunk-level extraction results using SHA-256 hashes of (doc_hash, chunk_text, versions, model).
Detects exact & near-duplicate content to skip repeated LLM calls while maintaining
source provenance across multiple pages/sections.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple
from config import CONFIG
from utils.logger import get_logger

logger = get_logger("extraction.chunk_cache")

EXTRACTOR_VERSION = "2.1.0"
PROMPT_VERSION = "2.1.0"
SCHEMA_VERSION = "2.1.0"


def compute_chunk_hash(text: str) -> str:
    """Normalize text whitespace/case and compute SHA-256 hash."""
    norm = " ".join(text.strip().lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def build_cache_key(doc_hash: str, chunk_text: str, model: str) -> str:
    """Build a composite cache key combining doc hash, chunk hash, version tags, and model name."""
    c_hash = compute_chunk_hash(chunk_text)
    raw_key = f"{doc_hash}:{c_hash}:{EXTRACTOR_VERSION}:{PROMPT_VERSION}:{SCHEMA_VERSION}:{model}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


class ExtractionCache:
    """Disk-backed fine-grained extraction cache and deduplicator."""

    def __init__(self):
        self.cache_dir = os.path.join(CONFIG.graph_cache_dir, "chunk_extraction_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._exact_hashes: Dict[str, Dict[str, Any]] = {}

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if not getattr(CONFIG, "enable_extraction_cache", True):
            return None
        path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                    logger.debug(f"Chunk cache hit: {cache_key[:10]}")
                    return data
            except Exception as e:
                logger.warning(f"Failed to read chunk cache file: {e}")
        return None

    def put(self, cache_key: str, data: Dict[str, Any]) -> None:
        if not getattr(CONFIG, "enable_extraction_cache", True):
            return
        path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Failed to write chunk cache file: {e}")

    def check_duplicate(self, text: str, page: int, section: str) -> Optional[Dict[str, Any]]:
        """
        Detect exact duplicate chunk content. Returns existing extraction if duplicate,
        while registering new page/section provenance.
        """
        if not getattr(CONFIG, "enable_deduplication", True):
            return None

        h = compute_chunk_hash(text)
        if h in self._exact_hashes:
            cached = self._exact_hashes[h]
            logger.info(f"Duplicate chunk detected (hash {h[:8]}). Reusing extraction with updated provenance.")
            # Clone cached result to avoid mutating stored instance
            cloned = pickle.loads(pickle.dumps(cached))
            # Update provenance details
            for req in cloned.get("requirements", []):
                req["page"] = page
                req["section"] = section
            return cloned

        return None

    def register_chunk(self, text: str, extraction_res: Dict[str, Any]) -> None:
        if not getattr(CONFIG, "enable_deduplication", True):
            return
        h = compute_chunk_hash(text)
        self._exact_hashes[h] = extraction_res


_singleton_cache: Optional[ExtractionCache] = None


def get_chunk_cache() -> ExtractionCache:
    global _singleton_cache
    if _singleton_cache is None:
        _singleton_cache = ExtractionCache()
    return _singleton_cache
