"""
Document & Requirement IR Caching Module (Section 25).

Caches parser and requirement IR artifacts by computing SHA-256 hash of specification files
along with parser and compiler version tags. Reuses existing indexed representations
on identical document uploads.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from typing import Optional, Tuple
from config import CONFIG
from core.requirement_ir.models import DocumentIR, RequirementIR
from utils.logger import get_logger

logger = get_logger("cache.doc_cache")

PARSER_VERSION = "1.0.0"
EXTRACTOR_VERSION = "2.0.0"
COMPILER_VERSION = "2.0.0"


def compute_document_hash(filepath: str) -> str:
    """Compute SHA-256 hash of file content."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def _get_cache_dir() -> str:
    cache_dir = os.path.join(CONFIG.graph_cache_dir, "doc_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_cached_requirement_ir(doc_hash: str) -> Optional[Tuple[DocumentIR, RequirementIR]]:
    """Retrieve cached DocumentIR and RequirementIR if version tags match."""
    cache_file = os.path.join(_get_cache_dir(), f"{doc_hash}.pkl")
    if not os.path.exists(cache_file):
        return None

    try:
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
            if (data.get("parser_version") == PARSER_VERSION and
                data.get("extractor_version") == EXTRACTOR_VERSION and
                data.get("compiler_version") == COMPILER_VERSION):
                logger.info(f"Cache hit for document hash {doc_hash[:10]}")
                return data["doc_ir"], data["req_ir"]
            else:
                logger.info("Cache invalidated due to version mismatch.")
    except Exception as e:
        logger.warning(f"Failed to load cached doc IR: {e}")

    return None


def save_requirement_ir_to_cache(doc_hash: str, doc_ir: DocumentIR, req_ir: RequirementIR) -> None:
    """Persist DocumentIR and RequirementIR to disk cache."""
    cache_file = os.path.join(_get_cache_dir(), f"{doc_hash}.pkl")
    data = {
        "doc_hash": doc_hash,
        "parser_version": PARSER_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "compiler_version": COMPILER_VERSION,
        "doc_ir": doc_ir,
        "req_ir": req_ir,
    }
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Persisted Requirement IR to cache for hash {doc_hash[:10]}")
    except Exception as e:
        logger.error(f"Failed to write cache file: {e}")
