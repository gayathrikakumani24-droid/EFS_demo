"""
Requirement IR Data Models & Canonical Schemas.

Defines protocol-independent schemas for Document IR, Atomic Requirements,
Entity Registry, Requirement Graph, Semantic Chunks, and Specification Conflicts.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def new_req_id(prefix: str = "req") -> str:
    """Generate a short unique identifier for requirement objects."""
    return f"{prefix.upper()}_{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------
# Source Location & Document IR Schema (Section 3)
# --------------------------------------------------------------------------

@dataclass
class SourceLocation:
    document_id: str = ""
    document_hash: Optional[str] = None
    section: str = ""
    section_id: Optional[str] = None
    page: Optional[int] = None
    block_id: Optional[str] = None
    paragraph: Optional[str] = None
    original_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> SourceLocation:
        if not data:
            return cls()
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DocumentBlock:
    block_id: str
    type: str                         # paragraph | heading | table | figure | code | reference
    section_id: str = ""
    source_label: str = ""             # displayed section label e.g. "5.1"
    section_title: str = ""
    page: int = 0
    paragraph_index: Optional[int] = None
    table_id: Optional[str] = None
    figure_id: Optional[str] = None
    text: str = ""
    source_location: SourceLocation = field(default_factory=SourceLocation)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source_location"] = self.source_location.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentBlock:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "source_location"}
        kwargs["source_location"] = SourceLocation.from_dict(data.get("source_location"))
        return cls(**kwargs)


@dataclass
class DocumentIR:
    document_id: str = field(default_factory=lambda: new_req_id("doc"))
    document_version: str = "1.0"
    source_type: str = "txt"           # pdf | docx | txt | md | image
    document_hash: str = ""
    ingestion_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    sections: List[Dict[str, Any]] = field(default_factory=list)
    blocks: List[DocumentBlock] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    figures: List[Dict[str, Any]] = field(default_factory=list)
    code_blocks: List[Dict[str, Any]] = field(default_factory=list)
    references: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_version": self.document_version,
            "source_type": self.source_type,
            "document_hash": self.document_hash,
            "ingestion_id": self.ingestion_id,
            "metadata": self.metadata,
            "sections": self.sections,
            "blocks": [b.to_dict() for b in self.blocks],
            "tables": self.tables,
            "figures": self.figures,
            "code_blocks": self.code_blocks,
            "references": self.references,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentIR:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "blocks"}
        kwargs["blocks"] = [DocumentBlock.from_dict(b) for b in data.get("blocks", [])]
        return cls(**kwargs)


# --------------------------------------------------------------------------
# Entity Registry Schema (Section 7)
# --------------------------------------------------------------------------

@dataclass
class DiscoveredEntity:
    entity_id: str = field(default_factory=lambda: new_req_id("ent"))
    name: str = ""                     # canonical discovered name
    type: str = "generic"              # signal | register | field | module | interface | instruction | opcode | memory_region | custom_entity | etc.
    description: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_references: List[SourceLocation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source_references"] = [s.to_dict() for s in self.source_references]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DiscoveredEntity:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "source_references"}
        kwargs["source_references"] = [SourceLocation.from_dict(s) for s in data.get("source_references", [])]
        return cls(**kwargs)


# --------------------------------------------------------------------------
# Requirement IR Schema (Sections 5 & 6)
# --------------------------------------------------------------------------

@dataclass
class AtomicRequirement:
    requirement_id: str = field(default_factory=lambda: new_req_id("req"))
    type: str = "BEHAVIOR"             # BEHAVIOR | HOLD | TRANSACTION | INTERFACE | STATE | TIMING | CONSTRAINT | etc.
    statement: str = ""
    entities: List[str] = field(default_factory=list)
    subject: Optional[str] = None
    action: Optional[str] = None
    condition: Optional[str] = None
    event: Optional[str] = None
    state: Optional[str] = None
    next_state: Optional[str] = None
    timing: Optional[str] = None
    value: Optional[str] = None
    unit: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    source: SourceLocation = field(default_factory=SourceLocation)
    confidence: float = 1.0
    extraction_method: str = "llm"     # RULE | NLP | LLM | DERIVED
    knowledge_status: str = "EXPLICIT" # EXPLICIT | DERIVED | INFERRED
    supporting_requirements: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source"] = self.source.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AtomicRequirement:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "source"}
        kwargs["source"] = SourceLocation.from_dict(data.get("source"))
        return cls(**kwargs)


# --------------------------------------------------------------------------
# Requirement Graph & Chunks (Sections 8, 9, 11)
# --------------------------------------------------------------------------

@dataclass
class RequirementGraphEdge:
    source: str
    relationship: str                  # DEPENDS_ON | REFERENCES | USES | PRODUCES | CONSUMES | TRIGGERS | etc.
    target: str
    source_reference: SourceLocation = field(default_factory=SourceLocation)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source_reference"] = self.source_reference.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RequirementGraphEdge:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "source_reference"}
        kwargs["source_reference"] = SourceLocation.from_dict(data.get("source_reference"))
        return cls(**kwargs)


@dataclass
class SemanticChunk:
    chunk_id: str = field(default_factory=lambda: new_req_id("chk"))
    chunk_type: str = "GENERAL"        # INTERFACE | TRANSACTION | STATE_MACHINE | SIGNAL_BEHAVIOR | REGISTER | TIMING | etc.
    requirement_ids: List[str] = field(default_factory=list)
    entity_ids: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    source_blocks: List[str] = field(default_factory=list)
    summary: str = ""
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SemanticChunk:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------
# Consistency & Conflict Detection (Section 13)
# --------------------------------------------------------------------------

@dataclass
class SpecConflict:
    conflict_id: str = field(default_factory=lambda: new_req_id("conf"))
    type: str = "CONTRADICTION"        # CONTRADICTION | WIDTH_MISMATCH | DUPLICATE_ID | DUPLICATE_ENCODING | etc.
    claims: List[str] = field(default_factory=list)
    sources: List[SourceLocation] = field(default_factory=list)
    severity: str = "HIGH"             # CRITICAL | HIGH | MAJOR | MINOR | WARNING
    status: str = "UNRESOLVED"         # UNRESOLVED | RESOLVED
    description: str = ""
    conflicting_values: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["sources"] = [s.to_dict() for s in self.sources]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SpecConflict:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "sources"}
        kwargs["sources"] = [SourceLocation.from_dict(s) for s in data.get("sources", [])]
        return cls(**kwargs)


# --------------------------------------------------------------------------
# Root Requirement IR Bundle
# --------------------------------------------------------------------------

@dataclass
class RequirementIR:
    doc_ir: DocumentIR = field(default_factory=DocumentIR)
    document_id: str = ""
    document_hash: str = ""
    ingestion_id: str = ""
    requirements: List[AtomicRequirement] = field(default_factory=list)
    entities: List[DiscoveredEntity] = field(default_factory=list)
    candidates: List[Any] = field(default_factory=list)
    edges: List[RequirementGraphEdge] = field(default_factory=list)
    chunks: List[SemanticChunk] = field(default_factory=list)
    conflicts: List[SpecConflict] = field(default_factory=list)
    validation_status: Dict[str, Any] = field(default_factory=dict)
    source_validation: Dict[str, Any] = field(default_factory=dict)
    ingestion_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_ir": self.doc_ir.to_dict(),
            "document_id": self.document_id or self.doc_ir.document_id,
            "document_hash": self.document_hash or self.doc_ir.document_hash,
            "ingestion_id": self.ingestion_id or self.doc_ir.ingestion_id,
            "requirements": [r.to_dict() for r in self.requirements],
            "entities": [e.to_dict() for e in self.entities],
            "candidates": [c.to_dict() if hasattr(c, "to_dict") else c for c in self.candidates],
            "edges": [ed.to_dict() for ed in self.edges],
            "chunks": [c.to_dict() for c in self.chunks],
            "conflicts": [cf.to_dict() for cf in self.conflicts],
            "validation_status": self.validation_status,
            "source_validation": self.source_validation,
            "ingestion_timestamp": self.ingestion_timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RequirementIR:
        doc_ir = DocumentIR.from_dict(data.get("doc_ir", {}))
        return cls(
            doc_ir=doc_ir,
            document_id=data.get("document_id", doc_ir.document_id),
            document_hash=data.get("document_hash", doc_ir.document_hash),
            ingestion_id=data.get("ingestion_id", doc_ir.ingestion_id),
            requirements=[AtomicRequirement.from_dict(r) for r in data.get("requirements", [])],
            entities=[DiscoveredEntity.from_dict(e) for e in data.get("entities", [])],
            candidates=data.get("candidates", []),
            edges=[RequirementGraphEdge.from_dict(ed) for ed in data.get("edges", [])],
            chunks=[SemanticChunk.from_dict(c) for c in data.get("chunks", [])],
            conflicts=[SpecConflict.from_dict(cf) for cf in data.get("conflicts", [])],
            validation_status=data.get("validation_status", {}),
            source_validation=data.get("source_validation", {}),
            ingestion_timestamp=data.get("ingestion_timestamp", datetime.now().isoformat()),
        )
