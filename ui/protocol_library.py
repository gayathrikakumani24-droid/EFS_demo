"""Protocol Library Page: browse, select, delete, or build protocol specifications."""

from __future__ import annotations

import os
import streamlit as st
from datetime import datetime

from config import CONFIG
from core.pipeline import run_pipeline
from core.registry import register_result, get_registry
from core.graph.neo4j_builder import get_graph_store
from core.vectorstore.faiss_store import get_vector_store
from utils.protocol_manager import (
    get_library_data,
    save_library_data,
    select_protocol,
    delete_protocol,
    bootstrap_library
)
from utils.logger import get_logger

logger = get_logger("ui.protocol_library")


def render() -> None:
    st.title("📚 Protocol Library")
    st.caption("Browse, select, or compile protocol specifications into the library context.")

    # Always ensure AXI4 is bootstrapped if root cache has data
    bootstrap_library()

    lib = get_library_data()

    # Dynamic active protocol selection
    active_protocol = st.session_state.get("active_protocol", "AXI4")
    
    st.subheader("Available Protocols")
    
    # Render protocols in a beautiful grid
    cols = st.columns(2)
    
    for idx, (p_name, p_info) in enumerate(lib.items()):
        col = cols[idx % 2]
        with col:
            with st.container(border=True):
                is_active = (p_name == active_protocol)
                badge = "🟢 Active" if is_active else ""
                
                header_cols = st.columns([3, 1])
                header_cols[0].markdown(f"### {p_name} {badge}")
                
                status = p_info.get("status", "Not Uploaded")
                if status == "Ready":
                    header_cols[1].success("Ready")
                    st.markdown(f"**Source File:** `{p_info.get('filename')}`")
                    st.markdown(f"**Ingested on:** {p_info.get('created_at', '')[:10]}")
                    st.markdown(f"**Reuse Count:** {p_info.get('reuse_count', 0)}")
                    
                    # Show statistics from DB
                    stat_cols = st.columns(3)
                    stat_cols[0].metric("Chunks", p_info.get("chunks", 0))
                    stat_cols[1].metric("Entities", p_info.get("entities", 0))
                    stat_cols[2].metric("Relationships", p_info.get("relationships", 0))
                    
                    btn_cols = st.columns(3)
                    if not is_active:
                        if btn_cols[0].button("🔌 Activate", key=f"act_{p_name}", use_container_width=True):
                            select_protocol(p_name)
                            st.session_state["active_protocol"] = p_name
                            st.success(f"Activated {p_name}")
                            st.rerun()
                    else:
                        btn_cols[0].button("🔌 Activated", key=f"act_{p_name}", disabled=True, use_container_width=True)
                        
                    if btn_cols[1].button("🗑️ Delete", key=f"del_{p_name}", use_container_width=True, type="secondary"):
                        delete_protocol(p_name)
                        # Fallback active protocol if we deleted the active one
                        if is_active:
                            for other in lib.keys():
                                if other != p_name and lib[other]["status"] == "Ready":
                                    select_protocol(other)
                                    st.session_state["active_protocol"] = other
                                    break
                            else:
                                select_protocol("AXI4")
                                st.session_state["active_protocol"] = "AXI4"
                        st.success(f"Deleted {p_name}")
                        st.rerun()
                else:
                    header_cols[1].error("Empty")
                    st.caption("No specifications uploaded yet for this protocol.")
                    
                    # Upload button and processor
                    uploaded_file = st.file_uploader(
                        f"Upload {p_name} Spec Document",
                        type=["pdf", "docx", "md", "markdown", "txt"],
                        key=f"upload_{p_name}"
                    )
                    
                    if uploaded_file and st.button(f"🚀 Ingest {p_name} Spec", key=f"ingest_{p_name}", type="primary"):
                        _process_protocol_file(p_name, uploaded_file)
                        st.rerun()

    st.divider()
    st.subheader("Protocol Library Diagnostics")
    
    diag_cols = st.columns(2)
    with diag_cols[0]:
        st.markdown(f"**Current Active Protocol Context:** `{active_protocol}`")
        st.markdown(f"**Vector Store Directory:** `{CONFIG.vector_index_dir}`")
        st.markdown(f"**Graph Cache Directory:** `{CONFIG.graph_cache_dir}`")
        
    with diag_cols[1]:
        # Diagnostic display of current active database sizes
        registry = get_registry()
        vector_store = get_vector_store()
        graph_store = get_graph_store()
        
        st.markdown(f"**Active Registry Documents:** {len(registry.documents)}")
        st.markdown(f"**Active Graph Nodes:** {graph_store.stats().get('nodes', 0)}")
        st.markdown(f"**Active Vector Chunks:** {vector_store.stats().get('total_chunks', 0)}")


def _process_protocol_file(p_name: str, uploaded_file) -> None:
    # 1. Switch active context to target protocol so pipeline writes there
    orig_protocol = st.session_state.get("active_protocol", "AXI4")
    select_protocol(p_name)

    # 2. Save file to upload directory
    save_path = os.path.join(CONFIG.upload_dir, uploaded_file.name)
    os.makedirs(CONFIG.upload_dir, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.info(f"Ingesting `{uploaded_file.name}` into `{p_name}` database. Please wait...")
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def progress_cb(message: str, pct: float) -> None:
        progress_bar.progress(min(max(pct, 0.0), 1.0))
        status_text.text(message)

    try:
        # Run pipeline
        result = run_pipeline(save_path, filename=uploaded_file.name, progress_cb=progress_cb)
        register_result(result)

        if result.errors:
            st.warning(f"Pipeline completed with warnings: {result.errors}")
        else:
            st.success(f"Successfully compiled {p_name} spec!")
            
        # Get count stats
        registry = get_registry()
        chunks_cnt = len(registry.all_chunks())
        entities_cnt = len({e.name for e in registry.all_entities()})
        rels_cnt = len(registry.all_relationships())
        
        # Save to library metadata
        lib = get_library_data()
        lib[p_name] = {
            "name": p_name,
            "filename": uploaded_file.name,
            "chunks": chunks_cnt,
            "entities": entities_cnt,
            "relationships": rels_cnt,
            "created_at": datetime.now().isoformat(),
            "reuse_count": 1,
            "status": "Ready"
        }
        save_library_data(lib)
        
        # Restore user's previous active protocol selection or set to the new one
        select_protocol(p_name)
        st.session_state["active_protocol"] = p_name
        
    except Exception as e:
        logger.exception(f"Failed to ingest protocol specification for {p_name}")
        st.error(f"Failed to process `{uploaded_file.name}`: {e}")
        # Restore original protocol context
        select_protocol(orig_protocol)
