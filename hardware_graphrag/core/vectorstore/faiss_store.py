"""
Step 7: Vector Database.

Generates sentence embeddings for every chunk (via sentence-transformers,
if installed) and stores them in a FAISS index for fast approximate
nearest-neighbor semantic search. Alongside the index we keep a parallel
metadata store (chunk_id -> {text, metadata, entity names, linked graph
node ids}) so retrieval results can be enriched without a second lookup.

If neither `faiss` nor `sentence-transformers` is installed, the store
transparently falls back to:
  * TF-IDF-ish bag-of-words vectors (via a tiny built-in hashing vectorizer)
  * brute-force cosine similarity search (numpy)
so semantic search still works, just with lower quality, in minimal
environments.

Embeddings are cached to disk per chunk_id (keyed by a hash of the chunk
text) so re-running the pipeline on an unchanged document doesn't
recompute embeddings.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import CONFIG
from utils.models import Chunk, RetrievedChunk
from utils.logger import get_logger

logger = get_logger("vector.faiss_store")

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


class _HashingEmbedder:
    """Deterministic fallback embedder: hashed bag-of-words -> fixed-size vector."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for token in text.lower().split():
                idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                vectors[i, idx] += 1.0
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        return vectors


class VectorStore:
    """FAISS-backed (or brute-force fallback) semantic search over chunks."""

    def __init__(self):
        self.cfg = CONFIG.embedding
        self.index_dir = CONFIG.vector_index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self.cfg.cache_dir, exist_ok=True)

        self._model = None
        self._dim = self.cfg.dim
        self._init_model()

        self._index = None                      # faiss.IndexFlatIP or None (brute-force)
        self._vectors: Optional[np.ndarray] = None   # used in brute-force mode
        self.chunk_ids: List[str] = []          # row order matches index/_vectors
        self.metadata: Dict[str, dict] = {}     # chunk_id -> {text, chapter, section, entities, linked_node_ids}

        self._load()

    # ------------------------------------------------------------------
    def _init_model(self) -> None:
        if _HAS_ST:
            try:
                self._model = SentenceTransformer(self.cfg.model_name)
                self._dim = self._model.get_sentence_embedding_dimension()
                logger.info(f"Loaded SentenceTransformer model '{self.cfg.model_name}' (dim={self._dim}).")
                return
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer '{self.cfg.model_name}': {e}. Using hashing fallback.")
        else:
            logger.warning("sentence-transformers not installed; using hashing fallback embedder.")
        self._model = _HashingEmbedder(dim=self._dim)

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts, using an on-disk cache keyed by text hash."""
        cache_path_fn = lambda h: os.path.join(self.cfg.cache_dir, f"{h}.npy")
        to_compute_idx, to_compute_texts, hashes = [], [], []
        vectors: List[Optional[np.ndarray]] = [None] * len(texts)

        for i, text in enumerate(texts):
            h = _text_hash(text)
            hashes.append(h)
            path = cache_path_fn(h)
            if os.path.exists(path):
                vectors[i] = np.load(path)
            else:
                to_compute_idx.append(i)
                to_compute_texts.append(text)

        if to_compute_texts:
            computed = self._model.encode(
                to_compute_texts,
                batch_size=self.cfg.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            ) if _HAS_ST and not isinstance(self._model, _HashingEmbedder) else self._model.encode(to_compute_texts)
            computed = np.asarray(computed, dtype="float32")
            for j, idx in enumerate(to_compute_idx):
                vectors[idx] = computed[j]
                np.save(cache_path_fn(hashes[idx]), computed[j])

        return np.vstack(vectors).astype("float32")

    # ------------------------------------------------------------------
    def add_chunks(self, chunks: List[Chunk], entity_names_by_chunk: Optional[Dict[str, List[str]]] = None,
                   linked_node_ids_by_chunk: Optional[Dict[str, List[str]]] = None,
                   efs_metadata_by_chunk: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        if not chunks:
            return
        entity_names_by_chunk = entity_names_by_chunk or {}
        linked_node_ids_by_chunk = linked_node_ids_by_chunk or {}
        efs_metadata_by_chunk = efs_metadata_by_chunk or {}

        texts = [c.text for c in chunks]
        vectors = self._embed(texts)
        # normalize for cosine similarity via inner product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms

        for c in chunks:
            efs_meta = efs_metadata_by_chunk.get(c.chunk_id, {})
            self.metadata[c.chunk_id] = {
                "text": c.text,
                "doc_id": c.doc_id,
                "page": c.page,
                "chapter": c.chapter,
                "section": c.section,
                "subsection": c.subsection,
                "heading": c.heading,
                "content_type": c.content_type,
                "entities": entity_names_by_chunk.get(c.chunk_id, []),
                "linked_node_ids": linked_node_ids_by_chunk.get(c.chunk_id, []),
                # EFS IR metadata links
                "efs_ir_id": efs_meta.get("efs_ir_id"),
                "constraint_ids": efs_meta.get("constraint_ids", []),
                "signal_ids": efs_meta.get("signal_ids", []),
                "component_ids": efs_meta.get("component_ids", []),
                "transaction_ids": efs_meta.get("transaction_ids", []),
            }
        self.chunk_ids.extend(c.chunk_id for c in chunks)

        if _HAS_FAISS:
            if self._index is None:
                self._index = faiss.IndexFlatIP(vectors.shape[1])
            self._index.add(vectors)
        else:
            self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])

        self._save()
        logger.info(f"Added {len(chunks)} chunks to vector store (total: {len(self.chunk_ids)}).")

    def clear_document(self, document_id: str) -> None:
        """Remove all chunks associated with a specific document_id."""
        if not document_id:
            return
        keep_indices = [i for i, cid in enumerate(self.chunk_ids) if self.metadata.get(cid, {}).get("doc_id") != document_id and self.metadata.get(cid, {}).get("document_id") != document_id]
        if len(keep_indices) == len(self.chunk_ids):
            return
        
        self.chunk_ids = [self.chunk_ids[i] for i in keep_indices]
        self.metadata = {cid: self.metadata[cid] for cid in self.chunk_ids}
        
        if _HAS_FAISS and self._index is not None and self._index.ntotal > 0:
            if keep_indices:
                all_vecs = np.zeros((self._index.ntotal, self._dim), dtype="float32")
                for i in range(self._index.ntotal):
                    all_vecs[i] = self._index.reconstruct(i)
                keep_vecs = all_vecs[keep_indices]
                self._index = faiss.IndexFlatIP(self._dim)
                self._index.add(keep_vecs)
            else:
                self._index = faiss.IndexFlatIP(self._dim)
        elif self._vectors is not None:
            self._vectors = self._vectors[keep_indices] if keep_indices else None
        self._save()
        logger.info(f"Cleared document '{document_id}' from vector store.")

    def search(self, query: str, top_k: int = 6, document_id: Optional[str] = None) -> List[RetrievedChunk]:
        if not self.chunk_ids:
            return []
        q_vec = self._embed([query])
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        # Fetch more candidates if document filtering is active
        fetch_k = min(top_k * 5 if document_id else top_k, len(self.chunk_ids))

        if _HAS_FAISS and self._index is not None:
            scores, indices = self._index.search(q_vec, fetch_k)
            scores, indices = scores[0], indices[0]
        else:
            sims = (self._vectors @ q_vec.T).flatten()
            top_idx = np.argsort(-sims)[:fetch_k]
            scores, indices = sims[top_idx], top_idx

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            chunk_id = self.chunk_ids[idx]
            meta = self.metadata.get(chunk_id, {})
            
            # Strict document isolation filter
            if document_id:
                meta_doc = meta.get("doc_id") or meta.get("document_id")
                if meta_doc and meta_doc != document_id:
                    continue

            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=meta.get("text", ""),
                    score=float(score),
                    source="vector",
                    metadata=meta,
                )
            )
            if len(results) >= top_k:
                break
        return results

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        return self.metadata.get(chunk_id)

    def link_efs_objects_to_chunks(self, efs_ir: EFSIR) -> None:
        """Update existing chunk metadata to link them to EFS IR object IDs based on traceability."""
        updated = False
        
        # Helper to append to a list in metadata
        def add_ref(chunk_id: str, key: str, val: str):
            nonlocal updated
            if chunk_id in self.metadata:
                lst = self.metadata[chunk_id].setdefault(key, [])
                if val not in lst:
                    lst.append(val)
                    updated = True
        
        for comp in efs_ir.components:
            cid = comp.traceability.chunk_id
            if cid:
                add_ref(cid, "component_ids", comp.component_id)
                
        for iface in efs_ir.interfaces:
            cid = iface.traceability.chunk_id
            if cid:
                add_ref(cid, "interface_ids", iface.interface_id)
                
        for sig in efs_ir.signals:
            cid = sig.traceability.chunk_id
            if cid:
                add_ref(cid, "signal_ids", sig.signal_id)
                
        for reg in efs_ir.registers:
            cid = reg.traceability.chunk_id
            if cid:
                add_ref(cid, "register_ids", reg.register_id)
                
        for instr in getattr(efs_ir, "instructions", []):
            cid = instr.traceability.chunk_id
            if cid:
                add_ref(cid, "instruction_ids", instr.instruction_id)
                
        for opc in getattr(efs_ir, "opcodes", []):
            cid = opc.traceability.chunk_id
            if cid:
                add_ref(cid, "opcode_ids", opc.opcode_id)
                
        for fsm in efs_ir.fsms:
            cid = fsm.traceability.chunk_id
            if cid:
                add_ref(cid, "fsm_ids", fsm.fsm_id)
                
        for const in efs_ir.constraints:
            cid = const.traceability.chunk_id
            if cid:
                add_ref(cid, "constraint_ids", const.constraint_id)
                
        for rule in efs_ir.timing_rules:
            cid = rule.traceability.chunk_id
            if cid:
                add_ref(cid, "timing_rule_ids", rule.rule_id)
                
        for rule in efs_ir.protocol_rules:
            cid = rule.traceability.chunk_id
            if cid:
                add_ref(cid, "protocol_rule_ids", rule.rule_id)
                
        for flow in efs_ir.flows:
            cid = flow.traceability.chunk_id
            if cid:
                add_ref(cid, "flow_ids", flow.flow_id)
                
        for conf in getattr(efs_ir, "conflicts", []):
            cid = conf.traceability.chunk_id
            if cid:
                add_ref(cid, "conflict_ids", conf.conflict_id)
                
        if updated:
            self._save()
            logger.info("Updated vector store metadata with EFS IR object references.")

    def stats(self) -> Dict[str, int]:
        return {"total_chunks": len(self.chunk_ids), "dim": self._dim}

    # ------------------------------------------------------------------
    def _save(self) -> None:
        meta_path = os.path.join(self.index_dir, "metadata.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump({"chunk_ids": self.chunk_ids, "metadata": self.metadata}, f)

        if _HAS_FAISS and self._index is not None:
            faiss.write_index(self._index, os.path.join(self.index_dir, "index.faiss"))
        elif self._vectors is not None:
            np.save(os.path.join(self.index_dir, "vectors.npy"), self._vectors)

    def _load(self) -> None:
        meta_path = os.path.join(self.index_dir, "metadata.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
                self.chunk_ids = data.get("chunk_ids", [])
                self.metadata = data.get("metadata", {})

        if _HAS_FAISS:
            idx_path = os.path.join(self.index_dir, "index.faiss")
            if os.path.exists(idx_path):
                self._index = faiss.read_index(idx_path)
        else:
            vec_path = os.path.join(self.index_dir, "vectors.npy")
            if os.path.exists(vec_path):
                self._vectors = np.load(vec_path)

    def clear(self) -> None:
        self.chunk_ids = []
        self.metadata = {}
        self._index = None
        self._vectors = None
        for fn in ("metadata.pkl", "index.faiss", "vectors.npy"):
            p = os.path.join(self.index_dir, fn)
            if os.path.exists(p):
                os.remove(p)


_singleton: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _singleton
    if _singleton is None:
        _singleton = VectorStore()
    return _singleton


def reset_vector_store() -> None:
    global _singleton
    _singleton = None
