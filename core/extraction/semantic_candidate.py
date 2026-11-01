"""
Intermediate SemanticCandidate model for Stage 1 Candidate Extraction & Classification.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from core.requirement_ir.models import SourceLocation


@dataclass
class SemanticCandidate:
    """An uncommitted intermediate candidate extracted from specification text prior to classification."""
    candidate_id: str = field(default_factory=lambda: f"CAND_{uuid.uuid4().hex[:8]}")
    document_id: str = ""
    chunk_id: str = ""
    source_text: str = ""
    context: Dict[str, Any] = field(default_factory=dict)  # section_heading, previous_text, next_text, table_structure
    candidate_type: Optional[str] = None                    # e.g., INSTRUCTION, INSTRUCTION_FIELD, OPCODE, COMPONENT, MEMORY_RESOURCE
    name: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_status: str = "EXPLICIT"                      # EXPLICIT | DERIVED | INFERRED
    confidence: float = 0.0
    source: SourceLocation = field(default_factory=SourceLocation)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source"] = self.source.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SemanticCandidate:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "source"}
        kwargs["source"] = SourceLocation.from_dict(data.get("source"))
        return cls(**kwargs)
