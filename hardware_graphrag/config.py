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
    load_dotenv(override=True)
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


def _resolve_llm_api_key(provider: str) -> str:
    prov = (provider or os.getenv("LLM_PROVIDER", "openrouter")).lower().strip()
    llm_key = os.getenv("LLM_API_KEY", "").strip()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")).strip()

    if prov == "openrouter":
        if openrouter_key:
            return openrouter_key
        if llm_key and not llm_key.startswith("gsk_"):
            return llm_key
        return openai_key or groq_key
    elif prov == "groq":
        if groq_key:
            return groq_key
        if llm_key and (llm_key.startswith("gsk_") or not openrouter_key):
            return llm_key
        return ""
    elif prov == "openai":
        if openai_key:
            return openai_key
        if llm_key and not (llm_key.startswith("gsk_") or llm_key.startswith("sk-or-v1-")):
            return llm_key
        return ""
    elif prov == "gemini":
        if gemini_key:
            return gemini_key
        if llm_key:
            return llm_key
        return ""

    return llm_key or openrouter_key or groq_key or openai_key or gemini_key


@dataclass
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "groq")   # openai | groq | openrouter | gemini | none
    model: str = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    extraction_model: str = os.getenv("LLM_EXTRACTION_MODEL", os.getenv("LLM_MODEL", "openai/gpt-oss-20b"))
    code_model: str = os.getenv("LLM_CODE_MODEL", os.getenv("LLM_MODEL", "openai/gpt-oss-120b"))
    api_key: str = field(default_factory=lambda: _resolve_llm_api_key(os.getenv("LLM_PROVIDER", "groq")))
    base_url: str = os.getenv("LLM_BASE_URL", "")          # for OpenAI-compatible / Groq / OpenRouter endpoints
    temperature: float = _env_float("LLM_TEMPERATURE", 0.0)
    max_tokens: int = _env_int("LLM_MAX_TOKENS", 8192)
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

    # Verification / Repair optimization configuration settings (Disabled by default for maximum speed)
    max_repair_iterations: int = _env_int("MAX_REPAIR_ITERATIONS", 0)
    compliance_threshold: int = _env_int("COMPLIANCE_THRESHOLD", 90)
    enable_static_verification: bool = _env_bool("ENABLE_STATIC_VERIFICATION", False)
    enable_semantic_verification: bool = _env_bool("ENABLE_SEMANTIC_VERIFICATION", False)
    verify_secondary_artifacts: bool = _env_bool("VERIFY_SECONDARY_ARTIFACTS", False)
    skip_duplicate_verification: bool = _env_bool("SKIP_DUPLICATE_VERIFICATION", True)

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.upload_dir, self.vector_index_dir, self.graph_cache_dir):
            os.makedirs(d, exist_ok=True)


CONFIG = AppConfig()
CONFIG.ensure_dirs()
