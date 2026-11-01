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

    def to_full_combined_dict(self) -> Dict[str, Any]:
        """Produce full complete combined dictionary across all extracted entities and requirements."""
        from core.compiler.efs_compiler import RequirementToEFSCompiler
        try:
            compiler = RequirementToEFSCompiler(self)
            efs_ir = compiler.compile()
            efs_dict = efs_ir.to_dict()
        except Exception:
            efs_dict = {}

        # Synthesize modules / components
        modules = [e.to_dict() for e in self.entities if e.type.lower() in ("module", "component", "subsystem", "block", "unit", "controller")]
        if not modules and efs_dict.get("components"):
            modules = efs_dict["components"]

        # Synthesize signals / ports
        signals = [e.to_dict() for e in self.entities if e.type.lower() in ("signal", "port", "clock", "reset")]
        if not signals and efs_dict.get("signals"):
            signals = efs_dict["signals"]

        # Synthesize registers
        registers = [e.to_dict() for e in self.entities if e.type.lower() in ("register", "reg")]
        if not registers and efs_dict.get("registers"):
            registers = efs_dict["registers"]

        # Synthesize instructions & opcodes
        instructions = [e.to_dict() for e in self.entities if e.type.lower() in ("instruction", "inst", "mnemonic")]
        if not instructions and efs_dict.get("instructions"):
            instructions = efs_dict["instructions"]

        opcodes = [e.to_dict() for e in self.entities if e.type.lower() in ("opcode", "encoding")]
        if not opcodes and efs_dict.get("opcodes"):
            opcodes = efs_dict["opcodes"]

        # Synthesize functional behavior & transactions
        behaviors = [r.to_dict() for r in self.requirements if r.type.upper() in ("BEHAVIOR", "FUNCTIONAL_BEHAVIOR", "SEQUENCE", "FLOW", "ALGORITHM")]
        if not behaviors and efs_dict.get("functional_behavior"):
            behaviors = efs_dict["functional_behavior"]

        transactions = [r.to_dict() for r in self.requirements if r.type.upper() in ("TRANSACTION", "PROTOCOL_TRANSACTION")]
        if not transactions and efs_dict.get("transactions"):
            transactions = efs_dict["transactions"]

        # Synthesize FSM states & transitions
        states = []
        transitions = []
        for e in self.entities:
            if e.type.lower() in ("state", "fsm_state"):
                s_name = e.name
                s_attrs = getattr(e, "attributes", {}) or {}
                if not any(s.get("name") == s_name for s in states):
                    states.append({
                        "name": s_name,
                        "description": getattr(e, "description", ""),
                        "duration_cycles": s_attrs.get("duration_cycles"),
                        "outputs": s_attrs.get("outputs", {})
                    })
        for r in self.requirements:
            if r.state and not any(s.get("name") == r.state for s in states):
                states.append({"name": r.state, "duration_cycles": getattr(r, "timing", None)})
            if r.next_state and not any(s.get("name") == r.next_state for s in states):
                states.append({"name": r.next_state})
            if r.state and r.next_state:
                transitions.append({
                    "from": r.state,
                    "to": r.next_state,
                    "condition": r.condition or r.statement,
                    "action": r.action or ""
                })
        if hasattr(self.doc_ir, "extracted_states") and self.doc_ir.extracted_states:
            for s_info in self.doc_ir.extracted_states:
                s_name = s_info.get("name") if isinstance(s_info, dict) else str(s_info)
                if s_name and not any(s.get("name") == s_name for s in states):
                    states.append(s_info if isinstance(s_info, dict) else {"name": s_name})
        if hasattr(self.doc_ir, "extracted_transitions") and self.doc_ir.extracted_transitions:
            for t_info in self.doc_ir.extracted_transitions:
                if isinstance(t_info, dict) and not any(t.get("from") == t_info.get("from") and t.get("to") == t_info.get("to") for t in transitions):
                    transitions.append(t_info)
        if not states and efs_dict.get("fsms"):
            for f in efs_dict["fsms"]:
                states.extend(f.get("states", []))
                transitions.extend(f.get("transitions", []))

        # Synthesize conditions & constraints
        conditions = [r.to_dict() for r in self.requirements if r.condition or r.type.upper() in ("CONSTRAINT", "HOLD", "GUARD")]
        if not conditions and efs_dict.get("constraints"):
            conditions = efs_dict["constraints"]

        # Synthesize timing
        timing_rules = [r.to_dict() for r in self.requirements if r.timing or r.type.upper() in ("TIMING", "TIMEOUT")]
        if not timing_rules and efs_dict.get("timing_rules"):
            timing_rules = efs_dict["timing_rules"]

        # Synthesize performance requirements
        perf_reqs = [e.to_dict() for e in self.entities if e.type.lower() in ("performance_requirement", "performance", "power_requirement", "timing_rule")]
        if not perf_reqs and efs_dict.get("performance_requirements"):
            perf_reqs = efs_dict["performance_requirements"]
        elif not perf_reqs:
            for r in self.requirements:
                if r.type.upper() in ("PERFORMANCE", "POWER", "CONSTRAINT_PERFORMANCE"):
                    perf_reqs.append({
                        "metric": r.subject or "PERFORMANCE",
                        "target_value": getattr(r, "value", "") or r.statement,
                        "unit": getattr(r, "unit", ""),
                        "condition": r.statement
                    })

        # Synthesize parameters
        parameters = [e.to_dict() for e in self.entities if e.type.lower() in ("parameter", "param", "generic")]
        if not parameters and efs_dict.get("parameters"):
            parameters = efs_dict["parameters"]

        # Resolve design name from doc_ir, sections, filename, or entities
        design_name = ""
        if self.doc_ir and self.doc_ir.sections:
            first_sec = self.doc_ir.sections[0]
            sec_title = first_sec.get("title") if isinstance(first_sec, dict) else getattr(first_sec, "title", "")
            if sec_title and sec_title.strip() and sec_title.strip().lower() not in ("table of contents", "overview", "introduction"):
                design_name = sec_title.strip()
        if not design_name and self.doc_ir and self.doc_ir.metadata.get("filename"):
            fn = self.doc_ir.metadata["filename"]
            base = fn.rsplit(".", 1)[0].replace("_", " ").title().strip()
            if base and base.lower() not in ("specification", "spec"):
                design_name = base
        if not design_name and modules:
            m0 = modules[0]
            m0_name = m0.get("name") if isinstance(m0, dict) else getattr(m0, "name", "")
            if m0_name:
                design_name = m0_name
        if not design_name:
            design_name = efs_dict.get("metadata", {}).get("display_name") or efs_dict.get("metadata", {}).get("design_name") or ""

        return {
            "document_id": self.document_id or self.doc_ir.document_id,
            "document_hash": self.document_hash or self.doc_ir.document_hash,
            "ingestion_id": self.ingestion_id or self.doc_ir.ingestion_id,
            "design_name": design_name,
            "title": design_name,
            "entities": [e.to_dict() for e in self.entities],
            "requirements": [r.to_dict() for r in self.requirements],
            "modules": modules,
            "components": modules,
            "signals": signals,
            "interfaces": efs_dict.get("interfaces", []),
            "registers": registers,
            "instructions": instructions,
            "opcodes": opcodes,
            "memory_regions": efs_dict.get("memory_regions", []),
            "data_structures": efs_dict.get("data_structures", []),
            "transactions": transactions,
            "functional_behavior": behaviors,
            "states": states,
            "transitions": transitions,
            "fsms": efs_dict.get("fsms", []),
            "conditions": conditions,
            "constraints": conditions,
            "timing_rules": timing_rules,
            "timing_constraints": [r.get("cycle_relationship") or r.get("timing") or r.get("statement", "") for r in timing_rules],
            "performance_requirements": perf_reqs,
            "parameters": parameters,
            "relationships": [ed.to_dict() for ed in self.edges],
            "conflicts": [cf.to_dict() for cf in self.conflicts],
            "validation_status": self.validation_status,
            "doc_ir": self.doc_ir.to_dict() if hasattr(self.doc_ir, "to_dict") else self.doc_ir,
        }

    def to_approved_extraction(self) -> ApprovedExtractionArtifact:
        """Construct an ApprovedExtractionArtifact from this RequirementIR."""
        full_data = self.to_full_combined_dict()
        return ApprovedExtractionArtifact.from_dict(full_data)

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


# --------------------------------------------------------------------------
# Committed / Approved Extraction Artifact (Phase 3)
# --------------------------------------------------------------------------

@dataclass
class ApprovedExtractionArtifact:
    """
    Authoritative committed artifact storing the approved extracted knowledge
    from hardware specifications across approval lifecycles.
    """
    artifact_id: str = field(default_factory=lambda: new_req_id("appr"))
    document_id: str = ""
    design_name: str = "Target_Module"
    approval_state: Dict[str, str] = field(default_factory=lambda: {
        "ai_approval": "PENDING",
        "custom_approval": "PENDING",
        "status": "PENDING"
    })
    entities: List[Dict[str, Any]] = field(default_factory=list)
    modules: List[Dict[str, Any]] = field(default_factory=list)
    components: List[Dict[str, Any]] = field(default_factory=list)
    signals: List[Dict[str, Any]] = field(default_factory=list)
    interfaces: List[Dict[str, Any]] = field(default_factory=list)
    registers: List[Dict[str, Any]] = field(default_factory=list)
    instructions: List[Dict[str, Any]] = field(default_factory=list)
    opcodes: List[Dict[str, Any]] = field(default_factory=list)
    memory_regions: List[Dict[str, Any]] = field(default_factory=list)
    data_structures: List[Dict[str, Any]] = field(default_factory=list)
    transactions: List[Dict[str, Any]] = field(default_factory=list)
    functional_behavior: List[Dict[str, Any]] = field(default_factory=list)
    states: List[Dict[str, Any]] = field(default_factory=list)
    transitions: List[Dict[str, Any]] = field(default_factory=list)
    fsms: List[Dict[str, Any]] = field(default_factory=list)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    timing_rules: List[Dict[str, Any]] = field(default_factory=list)
    timing_constraints: List[Dict[str, Any]] = field(default_factory=list)
    performance_requirements: List[Dict[str, Any]] = field(default_factory=list)
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    resolutions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_functional_data(self) -> bool:
        """Check if artifact contains substantive extracted functional information."""
        return bool(
            self.signals
            or self.registers
            or self.components
            or self.modules
            or self.instructions
            or self.opcodes
            or self.fsms
            or self.states
            or self.transitions
            or self.transactions
            or self.functional_behavior
            or self.performance_requirements
            or self.timing_rules
            or self.constraints
        )

    def apply_resolution(self, res: Any, is_ai: bool = False, approved_by: str = "") -> None:
        """
        Apply an approved resolution proposal or custom decision to the functional artifact,
        updating the affected entity/property directly and logging provenance.
        """
        conf_id = getattr(res, "conflict_id", "") or (res.get("conflict_id") if isinstance(res, dict) else "")
        proposed_change = getattr(res, "proposed_change", "") or (res.get("proposed_change") if isinstance(res, dict) else "")
        proposed_val = getattr(res, "proposed_value", None)
        if proposed_val is None and isinstance(res, dict):
            proposed_val = res.get("proposed_value")
        target_ent = getattr(res, "target_entity", None) or (res.get("target_entity") if isinstance(res, dict) else None)
        affected_reqs = getattr(res, "affected_requirements", []) or (res.get("affected_requirements") if isinstance(res, dict) else [])
        prop = getattr(res, "property", "") or (res.get("property") if isinstance(res, dict) else "")

        # Extract target property and value from proposed_change if formatted e.g. "DST_ADDR.offset = 0x14"
        if proposed_change and "=" in proposed_change:
            lhs, rhs = proposed_change.split("=", 1)
            rhs_clean = rhs.strip()
            if proposed_val is None or proposed_val == "":
                proposed_val = rhs_clean
            lhs_clean = lhs.strip()
            if "." in lhs_clean:
                target_ent = target_ent or lhs_clean.split(".")[0].strip()
                prop = prop or lhs_clean.split(".")[-1].strip()
            else:
                prop = prop or lhs_clean

        val_str = str(proposed_val).strip() if proposed_val is not None else ""
        target_names = set()
        if target_ent:
            target_names.add(target_ent.lower())
        for req in affected_reqs:
            target_names.add(str(req).lower())

        # 1. Apply to Registers
        for reg in self.registers:
            r_name = str(reg.get("name", "")).lower()
            if not target_names or r_name in target_names or any(t in r_name for t in target_names):
                if "offset" in prop.lower() or "address" in prop.lower() or val_str.startswith("0x"):
                    reg["offset"] = val_str
                    reg["address"] = val_str
                elif "width" in prop.lower() and val_str.isdigit():
                    reg["width"] = int(val_str)
                elif "access" in prop.lower():
                    reg["access"] = val_str
                    reg["access_type"] = val_str

        # 2. Apply to Signals
        for sig in self.signals:
            s_name = str(sig.get("name", "")).lower()
            if not target_names or s_name in target_names or any(t in s_name for t in target_names):
                if "direction" in prop.lower() or val_str.lower() in ("input", "output", "inout"):
                    sig["direction"] = val_str.lower()
                elif "width" in prop.lower() or val_str.isdigit() or ":" in val_str:
                    sig["width"] = val_str

        # 3. Apply to Opcodes / Instructions
        for opc in self.opcodes:
            o_name = str(opc.get("mnemonic", "")).lower()
            if not target_names or o_name in target_names or any(t in o_name for t in target_names):
                if "encoding" in prop.lower() or "opcode" in prop.lower() or val_str:
                    opc["binary_encoding"] = val_str

        for inst in self.instructions:
            i_name = str(inst.get("mnemonic", inst.get("name", ""))).lower()
            if not target_names or i_name in target_names or any(t in i_name for t in target_names):
                if "encoding" in prop.lower() or "opcode" in prop.lower() or val_str:
                    inst["opcode"] = val_str

        # 4. Mark conflict as resolved in self.conflicts
        for conf in self.conflicts:
            c_id = conf.get("conflict_id") if isinstance(conf, dict) else getattr(conf, "conflict_id", "")
            if c_id == conf_id or (target_names and any(str(e).lower() in target_names for e in (conf.get("entities", []) if isinstance(conf, dict) else getattr(conf, "entities", [])))):
                if isinstance(conf, dict):
                    conf["status"] = "APPROVED"
                    conf["resolution_status"] = "RESOLVED"
                    conf["blocking"] = False
                    conf["requires_human_decision"] = False
                else:
                    conf.status = "APPROVED"
                    conf.resolution_status = "RESOLVED"
                    conf.blocking = False
                    conf.requires_human_decision = False

        # 5. Record resolution in self.resolutions
        res_entry = {
            "conflict_id": conf_id,
            "target_entity": target_ent or (affected_reqs[0] if affected_reqs else ""),
            "property": prop,
            "proposed_value": val_str,
            "proposed_change": proposed_change or f"{prop} = {val_str}",
            "approval_status": "APPROVED",
            "approved_by": approved_by or ("ai_recommendation" if is_ai else "human_designer"),
            "resolution_type": "AI_PROPOSAL" if is_ai else "CUSTOM_REQUIREMENT",
        }
        self.resolutions.append(res_entry)

        # 6. Update approval states
        if is_ai:
            self.approval_state["ai_approval"] = "COMPLETED"
        else:
            self.approval_state["custom_approval"] = "COMPLETED"

        unresolved = [
            c for c in self.conflicts
            if (c.get("blocking", True) if isinstance(c, dict) else getattr(c, "blocking", True))
            and (c.get("resolution_status", "UNRESOLVED") if isinstance(c, dict) else getattr(c, "resolution_status", "UNRESOLVED")) == "UNRESOLVED"
        ]
        if not unresolved:
            self.approval_state["status"] = "APPROVED"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ApprovedExtractionArtifact:
        if not data:
            return cls()
        fields = cls.__dataclass_fields__
        kwargs = {}
        for k, v in data.items():
            if k in fields:
                kwargs[k] = v
        # Ensure design_name is not left as Target_Module if title or display_name exists
        if not kwargs.get("design_name") or kwargs.get("design_name") in ("Target_Module", "target_module", "unnamed_design"):
            cand = data.get("title") or data.get("design_spec_title") or data.get("display_name") or data.get("name")
            if cand and str(cand).strip() not in ("Target_Module", "target_module", "unnamed_design"):
                kwargs["design_name"] = str(cand).strip()

        # Ensure modules & components are synced
        if not kwargs.get("modules") and kwargs.get("components"):
            kwargs["modules"] = kwargs["components"]
        if not kwargs.get("components") and kwargs.get("modules"):
            kwargs["components"] = kwargs["modules"]
        return cls(**kwargs)

