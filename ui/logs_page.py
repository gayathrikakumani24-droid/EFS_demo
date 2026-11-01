"""Logs page: view parsing/extraction/graph/vector/error logs."""

from __future__ import annotations

import streamlit as st

from utils.logger import clear_logs, get_recent_logs

_CATEGORIES = ["all", "parsing", "chunking", "extraction", "graph", "vector", "retrieval", "general"]
_LEVELS = ["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def render() -> None:
    st.title("📜 Logs")
    st.caption("Parsing, extraction, graph creation, and error logs from the pipeline.")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        category = st.selectbox("Category", _CATEGORIES)
    with col2:
        level = st.selectbox("Level", _LEVELS)
    with col3:
        limit = st.number_input("Max entries", min_value=10, max_value=2000, value=300, step=10)

    if st.button("🗑️ Clear logs"):
        clear_logs()
        st.rerun()

    logs = get_recent_logs(category=category, level=level, limit=int(limit))
    st.caption(f"Showing {len(logs)} log entries.")

    if not logs:
        st.info("No logs to display yet.")
        return

    level_icons = {"DEBUG": "🔹", "INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌", "CRITICAL": "🔥"}
    log_text = "\n".join(
        f"{level_icons.get(l['level'], '')} {l['formatted']}" for l in logs
    )
    st.code(log_text, language="log")
