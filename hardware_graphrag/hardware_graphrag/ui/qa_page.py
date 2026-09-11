"""Question Answering page: ask natural language questions using hybrid GraphRAG retrieval."""

from __future__ import annotations

import streamlit as st

from core.extraction.llm_client import get_llm_client
from core.retrieval.hybrid_retriever import answer_question
from core.vectorstore.faiss_store import get_vector_store


def render() -> None:
    st.title("💬 Question Answering")
    st.caption("Ask a natural-language question. Retrieval combines Knowledge Graph traversal with semantic vector search.")

    vector_store = get_vector_store()
    if vector_store.stats().get("total_chunks", 0) == 0:
        st.info("Nothing has been indexed yet. Upload and process a document first.")
        return

    llm = get_llm_client()
    if not llm.available:
        st.warning(
            "No LLM API key is configured (set `LLM_API_KEY` / `LLM_PROVIDER` in your environment). "
            "Answers will fall back to showing the most relevant retrieved passage verbatim."
        )

    question = st.text_input("Your question", placeholder="e.g. What signals are asserted before the write response channel handshake?")

    if question.strip() and st.button("Ask", type="primary"):
        with st.spinner("Retrieving graph context and relevant chunks..."):
            result = answer_question(question)
        st.session_state["last_qa_result"] = result

    result = st.session_state.get("last_qa_result")
    if result and result.question == question:
        st.divider()
        st.subheader("Answer")
        st.markdown(result.answer)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🕸️ Retrieved Graph Context")
            gc = result.graph_context
            if gc and gc.nodes:
                st.markdown(f"**Seed entities detected:** {', '.join(gc.seed_entities) or '—'}")
                st.markdown(f"**Graph nodes retrieved:** {len(gc.nodes)}")
                st.markdown(f"**Relationships retrieved:** {len(gc.edges)}")
                with st.expander("View graph nodes"):
                    for n in gc.nodes[:30]:
                        st.write(f"- **{n.get('name')}** ({n.get('type')})")
                with st.expander("View relationships"):
                    for e in gc.edges[:30]:
                        st.write(f"`{e.get('source')}` **{e.get('relation')}** `{e.get('target')}`")
            else:
                st.caption("No matching graph entities were found for this question.")

        with col2:
            st.subheader("📄 Retrieved Chunks")
            for rc in result.retrieved_chunks[:8]:
                meta = rc.metadata or {}
                ref = f"{meta.get('chapter', '')}/{meta.get('section', '')} p.{meta.get('page', '?')}"
                with st.expander(f"[{rc.source}] score={rc.score:.2f} — {ref}"):
                    st.text(rc.text)
