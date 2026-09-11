"""Chunk Explorer page: browse, filter, and search hierarchical chunks."""

from __future__ import annotations

import streamlit as st

from core.registry import get_registry


def render() -> None:
    st.title("🧩 Chunk Explorer")
    st.caption("Browse the hierarchical chunks produced from your uploaded documents.")

    registry = get_registry()
    all_chunks = registry.all_chunks()

    if not all_chunks:
        st.info("No chunks available yet. Upload and process a document first.")
        return

    doc_options = {doc.filename: doc_id for doc_id, doc in registry.documents.items()}
    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        selected_doc_name = st.selectbox("Document", ["All"] + list(doc_options.keys()))
    chapters = sorted({c.chapter for c in all_chunks if c.chapter})
    with col2:
        selected_chapter = st.selectbox("Chapter", ["All"] + chapters)
    sections = sorted({c.section for c in all_chunks if c.section})
    with col3:
        selected_section = st.selectbox("Section", ["All"] + sections)

    search_query = st.text_input("🔎 Search chunk text", "")

    filtered = all_chunks
    if selected_doc_name != "All":
        doc_id = doc_options[selected_doc_name]
        filtered = [c for c in filtered if c.doc_id == doc_id]
    if selected_chapter != "All":
        filtered = [c for c in filtered if c.chapter == selected_chapter]
    if selected_section != "All":
        filtered = [c for c in filtered if c.section == selected_section]
    if search_query.strip():
        q = search_query.lower()
        filtered = [c for c in filtered if q in c.text.lower() or q in c.heading.lower()]

    st.caption(f"Showing {len(filtered)} of {len(all_chunks)} chunks.")

    for chunk in filtered[:200]:
        header = f"{chunk.heading or '(untitled)'} — p.{chunk.page} · {chunk.content_type}"
        with st.expander(header):
            st.markdown(
                f"**Chunk ID:** `{chunk.chunk_id}`  \n"
                f"**Chapter:** {chunk.chapter or '—'}  \n"
                f"**Section:** {chunk.section or '—'}  \n"
                f"**Subsection:** {chunk.subsection or '—'}  \n"
                f"**Previous chunk:** `{chunk.previous_chunk or '—'}`  \n"
                f"**Next chunk:** `{chunk.next_chunk or '—'}`"
            )
            if chunk.overlap_prefix:
                st.caption("⤴ Includes overlap carried over from the previous chunk.")
            st.text_area("Text", chunk.text, height=180, key=f"chunktext_{chunk.chunk_id}")

    if len(filtered) > 200:
        st.caption("Only the first 200 matching chunks are shown. Narrow your filters to see more precisely.")
