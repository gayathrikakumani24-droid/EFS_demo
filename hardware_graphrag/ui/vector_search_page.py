"""Vector Search page: perform semantic searches over indexed chunks."""

from __future__ import annotations

import streamlit as st

from config import CONFIG
from core.vectorstore.faiss_store import get_vector_store


def render() -> None:
    st.title("🔍 Vector Search")
    st.caption("Run semantic (embedding-based) search over all indexed chunks.")

    vector_store = get_vector_store()
    stats = vector_store.stats()

    if stats.get("total_chunks", 0) == 0:
        st.info("The vector index is empty. Upload and process a document first.")
        return

    st.caption(f"Index contains {stats['total_chunks']} chunks (embedding dim: {stats['dim']}).")

    query = st.text_input("Enter a semantic search query", placeholder="e.g. write address channel handshake timing")
    top_k = st.slider("Number of results", min_value=1, max_value=20, value=CONFIG.retrieval.top_k_vector)

    if query.strip():
        results = vector_store.search(query, top_k=top_k)
        st.caption(f"Found {len(results)} results.")
        for rc in results:
            meta = rc.metadata or {}
            ref = f"{meta.get('chapter', '')} / {meta.get('section', '')} · p.{meta.get('page', '?')}"
            with st.expander(f"Score {rc.score:.3f} — {ref}"):
                st.markdown(f"**Heading:** {meta.get('heading', '—')}")
                st.markdown(f"**Content type:** {meta.get('content_type', 'text')}")
                if meta.get("entities"):
                    st.markdown(f"**Linked entities:** {', '.join(meta['entities'])}")
                st.text_area("Chunk text", rc.text, height=160, key=f"vs_{rc.chunk_id}")
    else:
        st.caption("Enter a query above to search.")
