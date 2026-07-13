"""Document Upload page: upload PDF/DOCX/Markdown and run the ingestion pipeline."""

from __future__ import annotations

import os

import streamlit as st

from config import CONFIG
from core.pipeline import run_pipeline
from core.registry import register_result
from utils.logger import get_logger

logger = get_logger("ui.upload")


def render() -> None:
    st.title("📤 Document Upload")
    st.caption("Upload hardware specification documents (PDF, DOCX, or Markdown) to build the knowledge graph and vector index.")

    with st.expander("ℹ️ How this works", expanded=False):
        st.markdown(
            """
            Each uploaded document flows through the full GraphRAG pipeline:

            1. **Document Parsing** — hierarchy (chapters/sections/subsections), tables, figures, notes, warnings, registers.
            2. **Hierarchical Chunking** — split by structure (never fixed-size), with 10–20% overlap.
            3. **Entity Extraction** — Protocols, Interfaces, Channels, Signals, Registers, etc. (LLM or heuristic fallback).
            4. **Relationship Extraction** — structured (source, relation, target) triples.
            5. **Entity Normalization** — merges duplicate names (e.g. "WA Channel" → `WRITE_ADDRESS_CHANNEL`).
            6. **Knowledge Graph** — upserted into Neo4j (or an in-memory fallback graph).
            7. **Vector Database** — chunk embeddings indexed in FAISS for semantic search.
            """
        )

    uploaded_files = st.file_uploader(
        "Choose one or more files",
        type=["pdf", "docx", "md", "markdown", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("🚀 Start Processing", type="primary"):
        for uploaded_file in uploaded_files:
            _process_file(uploaded_file)
        st.success("All files processed. Check the Dashboard for a summary.")

    st.divider()
    st.subheader("Processing Log (this session)")
    if "upload_log" in st.session_state and st.session_state["upload_log"]:
        for entry in st.session_state["upload_log"]:
            st.text(entry)
    else:
        st.caption("No files processed yet in this session.")


def _process_file(uploaded_file) -> None:
    save_path = os.path.join(CONFIG.upload_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.markdown(f"**Processing `{uploaded_file.name}`...**")
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    log_lines = []

    def progress_cb(message: str, pct: float) -> None:
        progress_bar.progress(min(max(pct, 0.0), 1.0))
        status_text.text(message)
        log_lines.append(f"[{uploaded_file.name}] {message}")

    try:
        result = run_pipeline(save_path, filename=uploaded_file.name, progress_cb=progress_cb)
        register_result(result)

        if result.errors:
            st.warning(f"Completed with {len(result.errors)} warning(s)/error(s): {result.errors}")
        else:
            st.success(
                f"✅ `{uploaded_file.name}` processed: "
                f"{len(result.chunks)} chunks, {len({e.name for e in result.entities})} unique entities, "
                f"{len(result.relationships)} relationships."
            )
    except Exception as e:
        logger.exception(f"Pipeline crashed for {uploaded_file.name}")
        st.error(f"❌ Failed to process `{uploaded_file.name}`: {e}")
        log_lines.append(f"[{uploaded_file.name}] ERROR: {e}")

    st.session_state.setdefault("upload_log", [])
    st.session_state["upload_log"].extend(log_lines)
