"""
Developer & Debug View Streamlit Page (Section 23, 24, 25, 26, 30).

Visualizes the end-to-end Specification Compiler Pipeline:
Document IR -> Sections -> Entities -> Requirements -> Dependency Graph -> Conflicts -> Coverage -> Missing Info Report -> EFS IR -> Full Combined JSON.
"""

from __future__ import annotations

import json
import streamlit as st
from core.registry import get_registry
from core.pipeline_diagnostics import calculate_extraction_coverage, generate_missing_information_report
from utils.logger import get_logger

logger = get_logger("ui.debug_view")


def render() -> None:
    st.title("🔍 Developer Debug View & Pipeline Lineage")
    st.caption("Inspect Document IR → Requirement IR → Requirement Graph → Spec Conflicts → Coverage → EFS IR → Combined JSON")

    registry = get_registry()
    req_ir = st.session_state.get("active_req_ir")
    efs_ir = st.session_state.get("active_efs_ir")

    if not req_ir and "active_req_ir" not in st.session_state:
        req_ir = getattr(registry, "active_req_ir", None)

    if not req_ir and not efs_ir:
        st.info("No specification has been ingested yet. Please upload a specification file on the **Document Upload** page.")
        return

    coverage = calculate_extraction_coverage(req_ir, efs_ir) if req_ir else {}
    missing_report = generate_missing_information_report(req_ir) if req_ir else {}

    # Metrics Section
    st.markdown("### 📊 Pipeline Metrics & Context Reduction")

    doc_blocks_count = len(req_ir.doc_ir.blocks) if req_ir else len(registry.all_chunks())
    reqs_count = len(req_ir.requirements) if req_ir else 0
    entities_count = len(req_ir.entities) if req_ir else len(registry.all_entities())
    rels_count = len(req_ir.edges) if req_ir else len(registry.all_relationships())
    conflicts_count = len(req_ir.conflicts) if req_ir else 0
    cov_pct = coverage.get("extraction_coverage_pct", 0.0)

    efs_constructs_count = 0
    if efs_ir:
        efs_constructs_count = (
            len(efs_ir.components) + len(efs_ir.signals) + len(efs_ir.interfaces) +
            len(efs_ir.registers) + len(efs_ir.instructions) + len(efs_ir.opcodes) +
            len(efs_ir.memory_regions) + len(efs_ir.data_structures) + len(efs_ir.errors) +
            len(efs_ir.performance_requirements) + len(efs_ir.fsms) + len(efs_ir.flows)
        )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Extraction Coverage", f"{cov_pct}%")
    m2.metric("Document Blocks", f"{doc_blocks_count}")
    m3.metric("Atomic Requirements", f"{reqs_count}")
    m4.metric("Discovered Entities", f"{entities_count}")
    m5.metric("Spec Conflicts", f"{conflicts_count}")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Discovered Relationships", f"{rels_count}")
    col_b.metric("Compiled EFS Constructs", f"{efs_constructs_count}")
    col_c.metric("Unresolved References", f"{coverage.get('references_unresolved', 0)}")

    st.divider()

    # Tabs for lineage breakdown
    t_doc, t_ent, t_req, t_graph, t_conf, t_cov, t_efs, t_full = st.tabs([
        "📄 Document IR",
        "🏷️ Entities",
        "⚡ Requirements",
        "🕸️ Graph",
        "⚠️ Conflicts",
        "📈 Coverage & Missing Info",
        "🖥️ EFS IR",
        "📦 Full Combined JSON"
    ])

    with t_doc:
        st.subheader("Normalized Document IR")
        if req_ir and req_ir.doc_ir:
            st.json({
                "document_id": req_ir.doc_ir.document_id,
                "version": req_ir.doc_ir.document_version,
                "sections_count": len(req_ir.doc_ir.sections),
                "blocks_count": len(req_ir.doc_ir.blocks),
            })
            with st.expander("View Document Blocks & Traceability", expanded=True):
                for b in req_ir.doc_ir.blocks[:15]:
                    st.markdown(f"**Block `{b.block_id}`** (Section: `{b.section_title}`, Page {b.page})")
                    st.text(b.text[:300] + ("..." if len(b.text) > 300 else ""))
                    st.caption(f"Source Location: {b.source_location.to_dict()}")
                    st.divider()
        else:
            st.write("Document IR not loaded.")

    with t_ent:
        st.subheader("Dynamic Entity Registry")
        if req_ir and req_ir.entities:
            ent_data = []
            for e in req_ir.entities:
                ent_data.append({
                    "ID": e.entity_id,
                    "Name": e.name,
                    "Type": e.type,
                    "Description": e.description,
                    "Attributes": str(e.attributes),
                    "Ref Count": len(e.source_references)
                })
            st.dataframe(ent_data, use_container_width=True)
        else:
            st.write("No entities discovered.")

    with t_req:
        st.subheader("Atomic Requirements IR")
        if req_ir and req_ir.requirements:
            for r in req_ir.requirements:
                with st.expander(f"[{r.requirement_id}] {r.type}: {r.statement[:80]}...", expanded=False):
                    st.markdown(f"**Statement:** {r.statement}")
                    c1, c2, c3 = st.columns(3)
                    c1.write(f"**Type:** `{r.type}`")
                    c2.write(f"**Knowledge Status:** `{r.knowledge_status}`")
                    c3.write(f"**Confidence:** `{r.confidence}`")

                    if r.subject or r.action:
                        st.write(f"**Subject/Action:** `{r.subject}` → `{r.action}`")
                    if r.condition:
                        st.write(f"**Condition:** `{r.condition}`")
                    if r.entities:
                        st.write(f"**Entities:** {r.entities}")

                    st.caption(f"Source: Doc `{r.source.document_id}`, Section `{r.source.section}`, Block `{r.source.block_id}`, Page `{r.source.page}`")
        else:
            st.write("No requirements extracted.")

    with t_graph:
        st.subheader("Requirement & Entity Dependency Graph")
        if req_ir and req_ir.edges:
            edge_data = [{"Source": ed.source, "Relationship": ed.relationship, "Target": ed.target} for ed in req_ir.edges]
            st.dataframe(edge_data, use_container_width=True)
        else:
            st.write("No graph edges populated.")

    with t_conf:
        st.subheader("Validation & Spec Conflicts")
        if req_ir and req_ir.conflicts:
            for conf in req_ir.conflicts:
                sev_color = "red" if conf.severity in ("CRITICAL", "HIGH") else "orange"
                st.markdown(f":{sev_color}[**[{conf.severity}] {conf.type}** - {conf.conflict_id}]")
                st.write(conf.description)
                if conf.claims:
                    st.write("**Claims:**", conf.claims)
                if conf.conflicting_values:
                    st.write("**Conflicting Values:**", conf.conflicting_values)
                st.divider()
        else:
            st.success("Zero specification conflicts detected. Validation passed cleanly!")

    with t_cov:
        st.subheader("Extraction Coverage & Missing Information Report")
        st.json(coverage)
        st.subheader("Missing Information Details")
        st.json(missing_report)

    with t_efs:
        st.subheader("Compiled Canonical EFS IR")
        if efs_ir:
            st.json(efs_ir.to_dict())
        else:
            st.write("EFS IR not compiled.")

    with t_full:
        st.subheader("Full Combined Structured JSON Output")
        pipeline_res = st.session_state.get("last_pipeline_result")
        if pipeline_res and hasattr(pipeline_res, "to_full_json"):
            st.json(pipeline_res.to_full_json())
        elif req_ir and efs_ir:
            from utils.models import ParsedDocument
            p_doc = ParsedDocument(
                doc_id=req_ir.doc_ir.document_id,
                filename=req_ir.doc_ir.metadata.get("filename", "specification"),
                file_type=req_ir.doc_ir.source_type,
                sections=[s if isinstance(s, dict) else s.to_dict() for s in req_ir.doc_ir.sections]
            )
            from core.pipeline import PipelineResult
            res = PipelineResult(doc=p_doc, req_ir=req_ir, efs_ir=efs_ir)
            st.json(res.to_full_json())
        else:
            st.write("Full JSON representation not available.")
