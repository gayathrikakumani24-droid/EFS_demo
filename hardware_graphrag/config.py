"""
Central configuration for the Hardware Specification GraphRAG application.

All settings can be overridden via environment variables (or a local .env
file, loaded automatically if python-dotenv is installed). Nothing here
requires secrets to be hard-coded; the app works with sensible defaults
even if no LLM / Neo4j credentials are supplied (it degrades gracefully
to heuristic/rule-based extraction and an in-memory graph store).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


@dataclass
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "openai")   # openai | groq | gemini | none
    model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    api_key: str = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    base_url: str = os.getenv("LLM_BASE_URL", "")          # for OpenAI-compatible / Groq endpoints
    temperature: float = _env_float("LLM_TEMPERATURE", 0.0)
    max_tokens: int = _env_int("LLM_MAX_TOKENS", 4000)
    timeout_s: int = _env_int("LLM_TIMEOUT_S", 60)


@dataclass
class EmbeddingConfig:
    model_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    batch_size: int = _env_int("EMBEDDING_BATCH_SIZE", 32)
    dim: int = _env_int("EMBEDDING_DIM", 384)               # matches MiniLM-L6-v2
    cache_dir: str = os.getenv("EMBEDDING_CACHE_DIR", "data/embedding_cache")


@dataclass
class Neo4jConfig:
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "")
    database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    enabled: bool = _env_bool("NEO4J_ENABLED", False)       # falls back to in-memory graph if False/unreachable


@dataclass
class ChunkingConfig:
    max_chunk_chars: int = _env_int("MAX_CHUNK_CHARS", 1800)
    min_chunk_chars: int = _env_int("MIN_CHUNK_CHARS", 200)
    overlap_ratio: float = _env_float("CHUNK_OVERLAP_RATIO", 0.15)   # 10-20%


@dataclass
class RetrievalConfig:
    top_k_vector: int = _env_int("TOP_K_VECTOR", 6)
    top_k_graph_neighbors: int = _env_int("TOP_K_GRAPH_NEIGHBORS", 15)
    graph_expansion_hops: int = _env_int("GRAPH_EXPANSION_HOPS", 1)


@dataclass
class AppConfig:
    app_title: str = "Hardware Spec GraphRAG"
    data_dir: str = os.getenv("DATA_DIR", "data")
    upload_dir: str = os.getenv("UPLOAD_DIR", "data/uploads")
    vector_index_dir: str = os.getenv("VECTOR_INDEX_DIR", "data/vector_index")
    graph_cache_dir: str = os.getenv("GRAPH_CACHE_DIR", "data/graph_cache")
    supported_extensions: List[str] = field(default_factory=lambda: [".pdf", ".docx", ".md", ".markdown", ".txt"])

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.upload_dir, self.vector_index_dir, self.graph_cache_dir):
            os.makedirs(d, exist_ok=True)


CONFIG = AppConfig()
CONFIG.ensure_dirs()
