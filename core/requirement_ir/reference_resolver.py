"""
Cross-Section Reference Resolution Engine (Section 10).

Detects cross-references in specification text ("See Section 4.2", "Table 7",
"defined above", "register X") and resolves them into explicit relationships.
"""

from __future__ import annotations

import re
from typing import List, Tuple
from core.requirement_ir.models import (
    AtomicRequirement, DocumentIR, RequirementGraphEdge, SourceLocation
)
from utils.logger import get_logger

logger = get_logger("requirement_ir.reference_resolver")


class CrossReferenceResolver:
    """Engine resolving section, table, figure, and register references across requirements."""

    def __init__(self, doc_ir: DocumentIR = None):
        self.doc_ir = doc_ir
        self.section_map = {}
        self.table_map = {}
        if doc_ir:
            self._build_index()

    def _build_index(self):
        """Index section numbers, titles, and table identifiers from Document IR."""
        for sec in self.doc_ir.sections:
            sec_num = sec.get("section_number", "")
            sec_title = sec.get("title", "").lower()
            sec_id = sec.get("section_id", "")
            if sec_num:
                self.section_map[sec_num] = sec_id
            if sec_title:
                self.section_map[sec_title] = sec_id

        for tbl in self.doc_ir.tables:
            tbl_id = tbl.get("table_id") or tbl.get("name")
            if tbl_id:
                self.table_map[str(tbl_id).lower()] = tbl

    def resolve_references(self, requirements: List[AtomicRequirement]) -> Tuple[List[RequirementGraphEdge], List[str]]:
        """Scan requirements for references and resolve them to targets."""
        edges: List[RequirementGraphEdge] = []
        unresolved: List[str] = []

        section_pattern = re.compile(r"(?:see|in|according to|per)\s+(?:section|sec\.?)\s+([0-9]+(?:\.[0-9]+)*)", re.IGNORECASE)
        table_pattern = re.compile(r"(?:see|in|per)\s+(?:table)\s+([0-9]+|[a-z0-9_]+)", re.IGNORECASE)

        for req in requirements:
            text = req.statement

            # Check section references
            for match in section_pattern.finditer(text):
                sec_ref = match.group(1)
                target_id = self.section_map.get(sec_ref)
                if target_id:
                    edges.append(RequirementGraphEdge(
                        source=req.requirement_id,
                        relationship="SECTION_REFERENCE",
                        target=target_id,
                        source_reference=req.source
                    ))
                    req.references.append(f"SECTION_{sec_ref}")
                else:
                    unresolved.append(f"{req.requirement_id}: Section {sec_ref} target not found")

            # Check table references
            for match in table_pattern.finditer(text):
                tbl_ref = match.group(1).lower()
                if tbl_ref in self.table_map:
                    edges.append(RequirementGraphEdge(
                        source=req.requirement_id,
                        relationship="TABLE_REFERENCE",
                        target=f"TABLE_{tbl_ref}",
                        source_reference=req.source
                    ))
                    req.references.append(f"TABLE_{tbl_ref}")
                else:
                    unresolved.append(f"{req.requirement_id}: Table {tbl_ref} target not found")

        return edges, unresolved
