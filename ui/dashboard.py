"""Dashboard page: high-level overview of ingested documents and graph/vector stats."""

from __future__ import annotations

import streamlit as st

from core.graph.neo4j_builder import get_graph_store
from core.registry import get_registry
from core.vectorstore.faiss_store import get_vector_store


def render() -> None:
    st.title("📊 Dashboard")
    st.caption("Overview of everything ingested into the Hardware Spec GraphRAG system.")

    registry = get_registry()
    vector_store = get_vector_store()
    graph_store = get_graph_store()

    n_docs = len(registry.documents)
    n_chunks = len(registry.all_chunks())
    n_entities = len({e.name for e in registry.all_entities()})
    n_relationships = len(registry.all_relationships())
    graph_stats = graph_store.stats()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents", n_docs)
    col2.metric("Chunks", n_chunks)
    col3.metric("Unique Entities", n_entities)
    col4.metric("Relationships", n_relationships)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Knowledge Graph")
        st.metric("Graph Nodes", graph_stats.get("nodes", 0))
        st.metric("Graph Edges", graph_stats.get("edges", 0))
        backend = "Neo4j" if graph_store.__class__.__name__ == "Neo4jGraphStore" else "In-Memory (fallback)"
        st.caption(f"Backend: **{backend}**")

    with col2:
        st.subheader("Vector Database")
        vstats = vector_store.stats()
        st.metric("Indexed Chunks", vstats.get("total_chunks", 0))
        st.metric("Embedding Dimension", vstats.get("dim", 0))

    st.divider()
    st.subheader("Uploaded Documents")

    if not registry.documents:
        st.info("No documents have been processed yet. Head to **Document Upload** to get started.")
        return

    rows = []
    for doc_id, doc in registry.documents.items():
        n_c = len(registry.chunks.get(doc_id, []))
        n_e = len({e.name for e in registry.entities.get(doc_id, [])})
        n_r = len(registry.relationships.get(doc_id, []))
        errs = registry.errors.get(doc_id, [])
        rows.append({
            "Document": doc.filename,
            "Type": doc.file_type,
            "Pages": doc.total_pages,
            "Sections": len(doc.sections),
            "Chunks": n_c,
            "Entities": n_e,
            "Relationships": n_r,
            "Errors": len(errs),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    error_docs = {doc_id: errs for doc_id, errs in registry.errors.items() if errs}
    if error_docs:
        with st.expander(f"⚠️ {len(error_docs)} document(s) had processing errors"):
            for doc_id, errs in error_docs.items():
                doc_name = registry.documents[doc_id].filename
                st.write(f"**{doc_name}**")
                for e in errs:
                    st.write(f"- {e}")
