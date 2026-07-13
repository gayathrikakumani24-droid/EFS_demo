"""
Shared data models (schemas) used across the entire GraphRAG pipeline.

Using plain dataclasses (rather than a heavier framework) keeps the
pipeline dependency-light while still giving us type hints, defaults,
and easy (de)serialization to/from dict / JSON for caching and for
passing data between Streamlit pages via st.session_state.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


def new_id(prefix: str = "id") -> str:
    """Generate a short, human-readable unique id."""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# --------------------------------------------------------------------------
# Document structure
# --------------------------------------------------------------------------

@dataclass
class DocSection:
    """A single node in the parsed document hierarchy (chapter/section/etc)."""

    section_id: str
    title: str
    level: int                      # 0 = chapter, 1 = section, 2 = subsection, ...
    section_number: str = ""        # e.g. "3.2.1"
    page_start: int = 0
    page_end: int = 0
    chapter: str = ""
    parent_id: Optional[str] = None
    previous_id: Optional[str] = None
    next_id: Optional[str] = None
    content_type: str = "text"      # text | table | figure | timing_diagram | register | note | warning | example
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedDocument:
    """Result of Step 1 (Document Parsing)."""

    doc_id: str
    filename: str
    file_type: str                  # pdf | docx | md
    sections: List[DocSection] = field(default_factory=list)
    total_pages: int = 0
    raw_text_len: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "total_pages": self.total_pages,
            "raw_text_len": self.raw_text_len,
            "sections": [s.to_dict() for s in self.sections],
        }


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single hierarchical chunk (Step 2)."""

    chunk_id: str
    doc_id: str
    text: str
    page: int = 0
    chapter: str = ""
    section: str = ""
    subsection: str = ""
    heading: str = ""
    content_type: str = "text"
    previous_chunk: Optional[str] = None
    next_chunk: Optional[str] = None
    overlap_prefix: str = ""        # text carried over from previous chunk (10-20%)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Entities / Relationships (Steps 3 - 5)
# --------------------------------------------------------------------------

ENTITY_TYPES = [
    "Protocol", "Interface", "Channel", "Signal", "Register", "Field",
    "Transaction", "TimingConstraint", "ClockDomain", "MemoryRegion",
    "Address", "Interrupt", "State", "Command", "DataStructure", "Feature",
]

RELATIONSHIP_TYPES = [
    "BELONGS_TO", "PART_OF", "USES", "DEPENDS_ON", "HANDSHAKES_WITH",
    "CONNECTS_TO", "ASSERTED_BEFORE", "ASSERTED_AFTER", "TRANSITIONS_TO",
    "GENERATES", "REQUIRES", "CONFIGURES", "READS_FROM", "WRITES_TO",
    "REFERENCES", "DEFINED_IN",
]


@dataclass
class Entity:
    """A single extracted (and possibly normalized) entity (Step 3 / 5)."""

    entity_id: str
    name: str                       # normalized canonical name
    raw_name: str                   # name as it appeared in the source text
    entity_type: str
    chunk_id: str
    doc_id: str
    page: int = 0
    chapter: str = ""
    section: str = ""
    original_text: str = ""
    aliases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Relationship:
    """A single extracted relationship triple (Step 4)."""

    rel_id: str
    source: str                     # normalized entity name
    relation: str                   # one of RELATIONSHIP_TYPES
    target: str                     # normalized entity name
    chunk_id: str
    doc_id: str
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source: str                     # "vector" | "graph" | "vector+graph"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphContext:
    seed_entities: List[str]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    chunk_ids: List[str]


@dataclass
class QAResult:
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    graph_context: Optional[GraphContext]
    citations: List[str] = field(default_factory=list)
