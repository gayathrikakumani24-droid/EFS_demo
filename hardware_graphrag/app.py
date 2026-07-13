"""
Hardware Specification Knowledge Graph (GraphRAG) — Streamlit application.

Run with:
    streamlit run app.py

See README.md for setup instructions (API keys, Neo4j configuration, etc).
Every setting has a sensible fallback, so the app is fully usable out of
the box with no external services configured (heuristic entity/relation
extraction, in-memory graph store, hashing-based embeddings).
"""

from __future__ import annotations

import streamlit as st

from config import CONFIG
from utils.logger import get_logger

logger = get_logger("app")

st.set_page_config(
    page_title=CONFIG.app_title,
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "Dashboard": "ui.dashboard",
    "Document Upload": "ui.upload_page",
    "Chunk Explorer": "ui.chunk_explorer",
    "Entity Explorer": "ui.entity_explorer",
    "Knowledge Graph": "ui.graph_view",
    "Vector Search": "ui.vector_search_page",
    "Question Answering": "ui.qa_page",
    "Logs": "ui.logs_page",
}

PAGE_ICONS = {
    "Dashboard": "📊",
    "Document Upload": "📤",
    "Chunk Explorer": "🧩",
    "Entity Explorer": "🏷️",
    "Knowledge Graph": "🕸️",
    "Vector Search": "🔍",
    "Question Answering": "💬",
    "Logs": "📜",
}


def _render_sidebar_status() -> None:
    from core.extraction.llm_client import get_llm_client
    from core.graph.neo4j_builder import get_graph_store

    st.sidebar.markdown("### System Status")

    llm = get_llm_client()
    llm_status = "🟢 Connected" if llm.available else "🟡 Not configured (heuristic fallback)"
    st.sidebar.caption(f"**LLM ({CONFIG.llm.provider}):** {llm_status}")

    graph_store = get_graph_store()
    backend = "Neo4j" if graph_store.__class__.__name__ == "Neo4jGraphStore" else "In-Memory"
    st.sidebar.caption(f"**Graph backend:** {backend}")

    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        emb_status = "SentenceTransformers"
    except ImportError:
        emb_status = "Hashing fallback"
    st.sidebar.caption(f"**Embeddings:** {emb_status}")


def main() -> None:
    st.sidebar.title(f"🔩 {CONFIG.app_title}")
    st.sidebar.caption("Hardware Spec → Knowledge Graph + Vector RAG")

    page_names = list(PAGES.keys())
    labels = [f"{PAGE_ICONS[name]}  {name}" for name in page_names]
    choice_label = st.sidebar.radio("Navigate", labels, label_visibility="collapsed")
    choice = page_names[labels.index(choice_label)]

    st.sidebar.divider()
    _render_sidebar_status()

    st.sidebar.divider()
    st.sidebar.caption(
        "Configure LLM provider, Neo4j, and embedding settings via environment "
        "variables — see `.env.example` in the project root."
    )

    module_path = PAGES[choice]
    module = __import__(module_path, fromlist=["render"])
    try:
        module.render()
    except Exception as e:
        logger.exception(f"Rendering page '{choice}' failed")
        st.error(f"Something went wrong rendering this page: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
