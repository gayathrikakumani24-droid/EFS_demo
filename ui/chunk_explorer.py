"""Chunk Explorer page: browse, filter, and inspect hierarchical chunks enriched with entities, signals, registers, and interfaces."""

from __future__ import annotations

import json
import streamlit as st

from core.registry import get_registry
from core.chunking.chunk_enricher import enrich_chunk, get_all_enriched_chunks


def render() -> None:
    st.title("🧩 Chunk Explorer & Enriched JSON")
    st.caption("Inspect hierarchical chunks enriched with extracted entities, signals, registers, and interfaces.")

    registry = get_registry()
    all_chunks = registry.all_chunks()

    if not all_chunks:
        st.info("No chunks available yet. Upload and process a document first.")
        return

    doc_options = {doc.filename: doc_id for doc_id, doc in registry.documents.items()}
    
    # --- Top Filtering Bar ---
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    with col1:
        selected_doc_name = st.selectbox("📄 Document", ["All"] + list(doc_options.keys()))
    chapters = sorted({c.chapter for c in all_chunks if c.chapter})
    with col2:
        selected_chapter = st.selectbox("📖 Chapter", ["All"] + chapters)
    sections = sorted({c.section for c in all_chunks if c.section})
    with col3:
        selected_section = st.selectbox("📑 Section", ["All"] + sections)
    with col4:
        content_filter = st.selectbox("🔍 Filter by Content", ["All", "Has Signals", "Has Registers", "Has Interfaces", "Has Entities"])

    search_query = st.text_input("🔎 Search chunk text or entity / signal / register name", "")

    # Filter chunks
    filtered = all_chunks
    if selected_doc_name != "All":
        doc_id = doc_options[selected_doc_name]
        filtered = [c for c in filtered if c.doc_id == doc_id]
    if selected_chapter != "All":
        filtered = [c for c in filtered if c.chapter == selected_chapter]
    if selected_section != "All":
        filtered = [c for c in filtered if c.section == selected_section]

    # Pre-enrich filtered chunks
    enriched_filtered = [enrich_chunk(c, registry=registry) for c in filtered]

    # Apply search and content filter
    final_enriched = []
    for ec in enriched_filtered:
        if content_filter == "Has Signals" and not ec["signals"]:
            continue
        if content_filter == "Has Registers" and not ec["registers"]:
            continue
        if content_filter == "Has Interfaces" and not ec["interfaces"]:
            continue
        if content_filter == "Has Entities" and not ec["entities"]:
            continue

        if search_query.strip():
            q = search_query.lower()
            text_match = q in ec["text"].lower() or q in ec["heading"].lower()
            ent_match = any(q in e["name"].lower() or any(q in a.lower() for a in e.get("aliases", [])) for e in ec["entities"])
            sig_match = any(q in s["name"].lower() for s in ec["signals"])
            reg_match = any(q in r["name"].lower() for r in ec["registers"])
            if_match = any(q in i["name"].lower() for i in ec["interfaces"])
            if not (text_match or ent_match or sig_match or reg_match or if_match):
                continue
        final_enriched.append(ec)

    # --- Summary Metrics Bar ---
    total_entities = sum(len(ec["entities"]) for ec in final_enriched)
    total_signals = sum(len(ec["signals"]) for ec in final_enriched)
    total_registers = sum(len(ec["registers"]) for ec in final_enriched)
    total_interfaces = sum(len(ec["interfaces"]) for ec in final_enriched)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Chunks", len(final_enriched))
    m2.metric("🏷️ Entities", total_entities)
    m3.metric("⚡ Signals", total_signals)
    m4.metric("🗄️ Registers", total_registers)
    m5.metric("🔌 Interfaces", total_interfaces)

    # --- Export / Action Bar ---
    col_dl, col_view = st.columns([1, 1])
    with col_dl:
        json_export_str = json.dumps(final_enriched, indent=2)
        st.download_button(
            label="📥 Download Filtered Chunks (Enriched JSON)",
            data=json_export_str,
            file_name="enriched_chunks.json",
            mime="application/json",
            use_container_width=True
        )
    with col_view:
        show_all_json = st.toggle("👁️ View Full JSON Array", value=False)

    if show_all_json:
        st.subheader("📦 Complete Enriched JSON Payload")
        st.json(final_enriched[:50])
        if len(final_enriched) > 50:
            st.caption(f"Showing first 50 of {len(final_enriched)} chunks in JSON preview. Download file for complete dataset.")
        st.divider()

    st.markdown(f"**Showing {len(final_enriched)} of {len(all_chunks)} chunks:**")

    # --- Chunk List with Tabs ---
    for ec in final_enriched[:150]:
        num_ent = len(ec["entities"])
        num_sig = len(ec["signals"])
        num_reg = len(ec["registers"])
        num_if = len(ec["interfaces"])

        badge_str = f"[{num_ent} ent"
        if num_sig > 0:
            badge_str += f" | ⚡ {num_sig} sig"
        if num_reg > 0:
            badge_str += f" | 🗄️ {num_reg} reg"
        if num_if > 0:
            badge_str += f" | 🔌 {num_if} if"
        badge_str += "]"

        header = f"{ec['heading'] or '(untitled)'} — p.{ec['page']} · {ec['content_type']}  {badge_str}"
        with st.expander(header):
            tab_text, tab_sig, tab_reg, tab_if, tab_ent, tab_json = st.tabs([
                "📝 Text & Hierarchy",
                f"⚡ Signals ({num_sig})",
                f"🗄️ Registers ({num_reg})",
                f"🔌 Interfaces ({num_if})",
                f"🏷️ All Entities ({num_ent})",
                "📦 Enriched JSON"
            ])

            with tab_text:
                st.markdown(
                    f"**Chunk ID:** `{ec['chunk_id']}`  \n"
                    f"**Chapter:** {ec['chapter'] or '—'}  \n"
                    f"**Section:** {ec['section'] or '—'}  \n"
                    f"**Subsection:** {ec['subsection'] or '—'}  \n"
                    f"**Previous chunk:** `{ec['previous_chunk'] or '—'}`  \n"
                    f"**Next chunk:** `{ec['next_chunk'] or '—'}`"
                )
                if ec.get("overlap_prefix"):
                    st.caption("⤴ Includes overlap carried over from previous chunk.")
                st.text_area("Chunk Text", ec["text"], height=160, key=f"text_{ec['chunk_id']}")

            with tab_sig:
                if ec["signals"]:
                    sig_table = [
                        {
                            "Signal Name": s.get("name"),
                            "Width": s.get("width", 1),
                            "Direction": s.get("direction", "—"),
                            "Clock Domain": s.get("clock_domain") or "—",
                            "Description": s.get("description") or "—"
                        }
                        for s in ec["signals"]
                    ]
                    st.dataframe(sig_table, use_container_width=True, hide_index=True)
                else:
                    st.info("No signals mapped in this chunk.")

            with tab_reg:
                if ec["registers"]:
                    reg_table = [
                        {
                            "Register Name": r.get("name"),
                            "Address / Offset": r.get("address_offset") or "0x00",
                            "Size (Bits)": r.get("size_bits") or 32,
                            "Access": r.get("access_type") or "RW",
                            "Description": r.get("description") or "—"
                        }
                        for r in ec["registers"]
                    ]
                    st.dataframe(reg_table, use_container_width=True, hide_index=True)
                else:
                    st.info("No registers mapped in this chunk.")

            with tab_if:
                if ec["interfaces"]:
                    if_table = [
                        {
                            "Interface Name": i.get("name"),
                            "Protocol": i.get("protocol", "generic"),
                            "Role": i.get("role", "—"),
                            "Description": i.get("description") or "—"
                        }
                        for i in ec["interfaces"]
                    ]
                    st.dataframe(if_table, use_container_width=True, hide_index=True)
                else:
                    st.info("No interfaces mapped in this chunk.")

            with tab_ent:
                if ec["entities"]:
                    ent_table = [
                        {
                            "Entity Name": e.get("name"),
                            "Type": e.get("entity_type"),
                            "Raw Text Mention": e.get("raw_name") or e.get("name"),
                            "Aliases": ", ".join(e.get("aliases", [])) if e.get("aliases") else "—"
                        }
                        for e in ec["entities"]
                    ]
                    st.dataframe(ent_table, use_container_width=True, hide_index=True)
                else:
                    st.info("No entities detected in this chunk.")

            with tab_json:
                st.caption(f"Raw Enriched JSON for Chunk `{ec['chunk_id']}`:")
                st.json(ec)

    if len(final_enriched) > 150:
        st.caption("Displaying first 150 matching chunks. Use filters or download the JSON to review all chunks.")
