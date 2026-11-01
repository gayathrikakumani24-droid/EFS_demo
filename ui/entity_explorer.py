"""Entity Explorer page: list, filter, and inspect extracted entities."""

from __future__ import annotations

from collections import defaultdict

import streamlit as st

from core.registry import get_registry
from utils.models import ENTITY_TYPES


def render() -> None:
    st.title("🏷️ Entity Explorer")
    st.caption("Browse entities extracted and normalized from your documents.")

    registry = get_registry()
    all_entities = registry.all_entities()

    if not all_entities:
        st.info("No entities available yet. Upload and process a document first.")
        return

    col1, col2 = st.columns([2, 3])
    with col1:
        selected_type = st.selectbox("Entity Type", ["All"] + ENTITY_TYPES)
    with col2:
        search_query = st.text_input("🔎 Search entity name", "")

    # group mentions by normalized name + type
    grouped = defaultdict(list)
    for e in all_entities:
        grouped[(e.name, e.entity_type)].append(e)

    rows = []
    for (name, etype), mentions in grouped.items():
        if selected_type != "All" and etype != selected_type:
            continue
        if search_query.strip() and search_query.lower() not in name.lower():
            continue
        aliases = sorted({a for m in mentions for a in m.aliases})
        sections = sorted({f"{m.chapter}/{m.section}".strip("/") for m in mentions if m.chapter or m.section})
        rows.append({
            "name": name,
            "type": etype,
            "mentions": len(mentions),
            "aliases": aliases,
            "sections": sections,
            "sample": mentions[0],
        })

    rows.sort(key=lambda r: r["mentions"], reverse=True)
    st.caption(f"Showing {len(rows)} unique entities ({len(all_entities)} total mentions).")

    table_data = [
        {
            "Name": r["name"],
            "Type": r["type"],
            "Mentions": r["mentions"],
            "Aliases": ", ".join(r["aliases"]) if r["aliases"] else "—",
        }
        for r in rows
    ]
    st.dataframe(table_data, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Inspect an entity")
    if rows:
        names = [r["name"] for r in rows]
        chosen = st.selectbox("Select entity", names)
        chosen_row = next(r for r in rows if r["name"] == chosen)
        st.markdown(f"**Type:** {chosen_row['type']}  \n**Total mentions:** {chosen_row['mentions']}")
        if chosen_row["aliases"]:
            st.markdown(f"**Merged aliases:** {', '.join(chosen_row['aliases'])}")
        st.markdown("**Appears in sections:**")
        for s in chosen_row["sections"]:
            st.write(f"- {s}")

        with st.expander("View original text snippets"):
            for m in grouped[(chosen_row["name"], chosen_row["type"])][:15]:
                st.markdown(f"*Page {m.page}, chunk `{m.chunk_id}`:*")
                st.text(m.original_text)
                st.markdown("---")
