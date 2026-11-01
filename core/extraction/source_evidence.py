"""
Evidence-Backed Extraction and Source Evidence Index.

Provides structured provenance, evidence tracking, and source-backed retrieval
for all extracted hardware properties across the EFS pipeline.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from core.requirement_ir.models import DocumentBlock, DocumentIR, SourceLocation
from utils.logger import get_logger

logger = get_logger("extraction.source_evidence")


@dataclass
class SourceEvidence:
    """Verbatim source evidence supporting an extracted hardware property."""
    document: str = ""
    section: str = ""
    text: str = ""
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    block_id: Optional[str] = None
    evidence_type: str = "text"  # text | table | code_block | bullet | diagram

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SourceEvidence:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EvidenceBackedProperty:
    """A hardware specification property retaining full source provenance and confidence."""
    name: str = ""
    value: Any = None
    status: str = "explicit"  # explicit | partial | inferred | unresolved | conflicting
    evidence: List[SourceEvidence] = field(default_factory=list)
    confidence: float = 1.0
    evidence_type: str = "source_text"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "status": self.status,
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence": self.confidence,
            "evidence_type": self.evidence_type
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvidenceBackedProperty:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "evidence"}
        kwargs["evidence"] = [SourceEvidence.from_dict(e) for e in data.get("evidence", [])]
        return cls(**kwargs)


class SourceEvidenceIndex:
    """
    Searchable index over Document IR blocks, tables, and sections.
    Provides fast, protocol-agnostic evidence lookup for hardware components,
    ports, registers, clocks, resets, constraints, and relationships.
    """

    def __init__(self, doc_ir: Optional[DocumentIR] = None):
        self.doc_ir = doc_ir
        self.doc_name = ""
        self.blocks: List[DocumentBlock] = []
        self._block_lines: Dict[str, Tuple[int, int]] = {}
        self._section_map: Dict[str, List[DocumentBlock]] = {}

        if doc_ir:
            self.index_document(doc_ir)

    def index_document(self, doc_ir: DocumentIR) -> None:
        """Build positional and semantic index over DocumentIR blocks."""
        self.doc_ir = doc_ir
        self.doc_name = (
            doc_ir.metadata.get("filename")
            or getattr(doc_ir, "document_id", "specification")
        )
        self.blocks = list(doc_ir.blocks)
        self._block_lines.clear()
        self._section_map.clear()

        running_line = 1
        for b in self.blocks:
            lines = (b.text or "").split("\n")
            line_count = max(1, len(lines))
            start_line = running_line
            end_line = running_line + line_count - 1
            self._block_lines[b.block_id] = (start_line, end_line)
            running_line = end_line + 1

            sec_title = (b.section_title or "General").strip().lower()
            self._section_map.setdefault(sec_title, []).append(b)

        logger.info(
            f"SourceEvidenceIndex: Indexed {len(self.blocks)} blocks across {len(self._section_map)} sections for '{self.doc_name}'."
        )

    def search(
        self,
        keywords: List[str],
        section_hint: Optional[str] = None,
        max_results: int = 5,
        match_any: bool = False
    ) -> List[SourceEvidence]:
        """Search indexed document for blocks matching hardware keywords with source positioning."""
        results: List[SourceEvidence] = []
        if not keywords:
            return results

        clean_kws = [k.lower().strip() for k in keywords if k and k.strip()]
        if not clean_kws:
            return results

        candidate_blocks = self.blocks
        if section_hint:
            hint_lower = section_hint.lower()
            matching_blocks = [
                b for b in self.blocks
                if hint_lower in (b.section_title or "").lower()
            ]
            if matching_blocks:
                candidate_blocks = matching_blocks + [b for b in self.blocks if b not in matching_blocks]

        scored_matches: List[Tuple[float, DocumentBlock, str]] = []
        for b in candidate_blocks:
            b_text = b.text or ""
            b_text_lower = b_text.lower()
            sec_lower = (b.section_title or "").lower()

            hits = sum(1 for kw in clean_kws if kw in b_text_lower or kw in sec_lower)
            if hits == 0:
                continue

            if not match_any and hits < len(clean_kws):
                continue

            # Extract the most relevant snippet surrounding the keyword match
            best_snippet = ""
            for line in b_text.split("\n"):
                l_lower = line.lower()
                if any(kw in l_lower for kw in clean_kws):
                    best_snippet = line.strip()
                    break
            if not best_snippet:
                best_snippet = b_text[:200].strip()

            score = float(hits)
            if section_hint and section_hint.lower() in sec_lower:
                score += 2.0
            scored_matches.append((score, b, best_snippet))

        scored_matches.sort(key=lambda x: x[0], reverse=True)

        for _, b, snippet in scored_matches[:max_results]:
            line_start, line_end = self._block_lines.get(b.block_id, (1, 1))
            ev_type = "table" if b.type == "table" or "|" in snippet else "text"
            results.append(SourceEvidence(
                document=self.doc_name,
                section=b.section_title or "General",
                text=snippet,
                line_start=line_start,
                line_end=line_end,
                block_id=b.block_id,
                evidence_type=ev_type
            ))

        return results

    def find_entity_evidence(self, entity_name: str, property_name: Optional[str] = None) -> List[SourceEvidence]:
        """Find source evidence for a specific hardware entity and optional property."""
        kws = [entity_name]
        if property_name:
            kws.append(property_name)
        return self.search(kws, max_results=5, match_any=not bool(property_name))

    def get_full_text(self) -> str:
        """Return full concatenated document text."""
        return "\n\n".join(b.text for b in self.blocks if b.text)
