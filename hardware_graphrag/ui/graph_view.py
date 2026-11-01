"""Knowledge Graph page: interactive Neo4j/in-memory graph visualization."""

from __future__ import annotations

from typing import Dict, List

import streamlit as st

from core.graph.neo4j_builder import get_graph_store
from utils.models import ENTITY_TYPES

_TYPE_COLORS = {
    "Protocol": "#e74c3c", "Interface": "#e67e22", "Channel": "#f1c40f",
    "Signal": "#2ecc71", "Register": "#1abc9c", "Field": "#3498db",
    "Transaction": "#9b59b6", "TimingConstraint": "#34495e", "ClockDomain": "#16a085",
    "MemoryRegion": "#d35400", "Address": "#7f8c8d", "Interrupt": "#c0392b",
    "State": "#8e44ad", "Command": "#2980b9", "DataStructure": "#27ae60", "Feature": "#f39c12",
}


def render() -> None:
    st.title("🕸️ Knowledge Graph")
    st.caption("Explore extracted entities and their relationships.")

    store = get_graph_store()
    all_nodes = store.all_nodes()
    all_edges = store.all_edges()

    if not all_nodes:
        st.info("The knowledge graph is empty. Upload and process a document first.")
        return

    col1, col2 = st.columns([2, 2])
    with col1:
        type_filter = st.multiselect("Filter by entity type", ENTITY_TYPES, default=[])
    with col2:
        search = st.text_input("🔎 Focus on entity (name contains)", "")

    nodes = all_nodes
    if type_filter:
        nodes = [n for n in nodes if n.get("type") in type_filter]

    focus_name = None
    if search.strip():
        matches = [n for n in nodes if search.lower() in n.get("name", "").lower()]
        if matches:
            focus_name = matches[0]["name"]
            neighbor_nodes, neighbor_edges = store.get_neighbors(focus_name, hops=1)
            nodes = neighbor_nodes
            edges = neighbor_edges
        else:
            st.warning(f"No entity matching '{search}' found.")
            edges = []
    else:
        node_names = {n["name"] for n in nodes}
        edges = [e for e in all_edges if e["source"] in node_names and e["target"] in node_names]

    st.caption(f"Displaying {len(nodes)} nodes and {len(edges)} edges.")

    _render_agraph(nodes, edges)

    st.divider()
    st.subheader("Node inspector")
    node_names = sorted({n["name"] for n in nodes})
    if node_names:
        default_idx = node_names.index(focus_name) if focus_name in node_names else 0
        chosen = st.selectbox("Select a node to inspect", node_names, index=default_idx)
        node = next((n for n in nodes if n["name"] == chosen), None)
        if node:
            st.json({
                "name": node.get("name"),
                "type": node.get("type"),
                "chapter": node.get("chapter"),
                "section": node.get("section"),
                "aliases": node.get("aliases", []),
                "pages": node.get("pages", []),
                "linked_chunk_ids": node.get("chunk_ids", []),
            })
            connected = [e for e in all_edges if e["source"] == chosen or e["target"] == chosen]
            if connected:
                st.markdown("**Connected relationships:**")
                for e in connected[:50]:
                    st.write(f"`{e['source']}` **{e['relation']}** `{e['target']}`")


def _render_agraph(nodes: List[Dict], edges: List[Dict]) -> None:
    try:
        from streamlit_agraph import agraph, Node, Edge, Config
    except ImportError:
        _render_fallback_table(nodes, edges)
        return

    agraph_nodes = [
        Node(
            id=n["name"],
            label=n["name"][:30],
            size=18,
            color=_TYPE_COLORS.get(n.get("type", ""), "#95a5a6"),
            title=f"{n.get('type', '')} — {n.get('name', '')}",
        )
        for n in nodes
    ]
    agraph_edges = [
        Edge(source=e["source"], target=e["target"], label=e.get("relation", ""), type="CURVE_SMOOTH")
        for e in edges
        if e["source"] in {n["name"] for n in nodes} and e["target"] in {n["name"] for n in nodes}
    ]
    config = Config(
        width="100%", height=550, directed=True, physics=True,
        hierarchical=False, nodeHighlightBehavior=True, highlightColor="#f5b301",
        collapsible=False,
    )
    agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)

    with st.expander("Legend"):
        legend_cols = st.columns(4)
        for i, (etype, color) in enumerate(_TYPE_COLORS.items()):
            with legend_cols[i % 4]:
                st.markdown(f"<span style='color:{color}'>●</span> {etype}", unsafe_allow_html=True)


def _render_fallback_table(nodes: List[Dict], edges: List[Dict]) -> None:
    st.warning(
        "`streamlit-agraph` isn't installed, so showing a plain table view instead. "
        "Install it (`pip install streamlit-agraph`) for an interactive graph visualization."
    )
    st.markdown("**Nodes**")
    st.dataframe(
        [{"Name": n.get("name"), "Type": n.get("type")} for n in nodes],
        use_container_width=True, hide_index=True,
    )
    st.markdown("**Edges**")
    st.dataframe(
        [{"Source": e.get("source"), "Relation": e.get("relation"), "Target": e.get("target")} for e in edges],
        use_container_width=True, hide_index=True,
    )
