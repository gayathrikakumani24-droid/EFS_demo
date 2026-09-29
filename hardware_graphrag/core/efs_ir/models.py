"""
Canonical EFS Hardware Intermediate Representation (EFS IR) Models.

Defines protocol-agnostic data models representing hardware specifications,
components, interfaces, signals, registers, transactions, FSMs, and constraints.
All models support source traceability back to original specification chunks.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class DiagnosticType(str, Enum):
    """Classification of diagnostics and issues in the EFS synthesis pipeline."""
    SPEC_CONFLICT = "SPEC_CONFLICT"
    MISSING_SPECIFICATION = "MISSING_SPECIFICATION"
    TARGET_RESOLUTION_FAILED = "TARGET_RESOLUTION_FAILED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    PLANNER_ERROR = "PLANNER_ERROR"


class ConflictStatus(str):
    """Status string supporting interoperability between ACTIVE and UNRESOLVED statuses."""
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            if str(self) in ("UNRESOLVED", "ACTIVE") and str(other) in ("UNRESOLVED", "ACTIVE"):
                return True
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash("UNRESOLVED")



from core.efs_ir.validation_rules import normalize_hardware_offset


PRESERVED_NULL_KEYS = {
    "confidence",
    "address",
    "offset",
    "offset_source",
    "reset_value",
    "original_value",
    "resolved_value",
    "proposed_value",
    "current_value",
}


def strip_nulls(obj: Any, preserve_keys: Optional[set] = None) -> Any:
    """Recursively remove None/null values from dicts and lists to minimize JSON payload size,
    except for fields where null explicitly conveys uncertainty or unassigned state."""
    keys_to_preserve = preserve_keys if preserve_keys is not None else PRESERVED_NULL_KEYS
    if isinstance(obj, dict):
        return {
            k: strip_nulls(v, keys_to_preserve)
            for k, v in obj.items()
            if v is not None or k in keys_to_preserve
        }
    elif isinstance(obj, list):
        return [strip_nulls(v, keys_to_preserve) for v in obj if v is not None]
    return obj


def new_efs_id(prefix: str = "efs") -> str:
    """Generate a short unique identifier for EFS objects."""
    return f"{prefix.upper()}_{uuid.uuid4().hex[:8]}"


@dataclass
class EFSSourceDocument:
    """Structured representation of a source specification document."""
    document_id: str = field(default_factory=lambda: new_efs_id("doc"))
    name: str = ""
    document_name: str = ""
    type: str = "docx"
    document_type: str = "docx"
    hash: str = ""
    document_hash: str = ""
    revision: str = "1.0"
    ingestion_id: str = ""
    path: str = ""

    def __post_init__(self):
        if not self.document_name and self.name:
            self.document_name = self.name
        elif not self.name and self.document_name:
            self.name = self.document_name
            
        if self.document_type and not self.type:
            self.type = self.document_type
        elif self.type and not self.document_type:
            self.document_type = self.type
            
        if self.document_hash and not self.hash:
            self.hash = self.document_hash
        elif self.hash and not self.document_hash:
            self.document_hash = self.hash

        if not self.name:
            self.name = "specification.docx"
            self.document_name = self.name

        # Ensure document_hash / hash is never an empty placeholder string
        if not self.hash:
            import hashlib
            seed_content = f"{self.document_id}_{self.name}_{self.revision}"
            self.hash = hashlib.sha256(seed_content.encode("utf-8")).hexdigest()
            self.document_hash = self.hash

        # Ensure ingestion_id is never empty
        if not self.ingestion_id:
            self.ingestion_id = f"ING_{self.hash[:8]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_name": self.document_name or self.name,
            "name": self.name or self.document_name,
            "document_type": self.document_type or self.type,
            "type": self.type or self.document_type,
            "document_hash": self.document_hash or self.hash,
            "hash": self.hash or self.document_hash,
            "revision": self.revision,
            "ingestion_id": self.ingestion_id,
            "path": self.path
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSSourceDocument:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EFSMetadata:
    """Metadata for the EFS IR instance."""
    design_id: str = field(default_factory=lambda: new_efs_id("dsn"))
    design_name: str = "unnamed_design"
    canonical_id: str = ""
    display_name: str = ""
    aliases: List[str] = field(default_factory=list)
    project_name: str = "default_project"
    document_id: str = ""
    document_hash: str = ""
    ingestion_id: str = ""
    generation_run_id: str = ""
    specification_revision: str = "1.0"
    ir_schema_version: str = "1.0"
    model_version: int = 1
    resolution_version: int = 0
    protocol: str = "generic"
    protocol_version: str = "1.0"
    top_module: str = ""
    description: str = ""
    author: str = "Hardware GraphRAG"
    source_documents: List[EFSSourceDocument] = field(default_factory=list)
    generation_timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    def __post_init__(self):
        # Guarantee non-empty, non-generic display_name
        raw_display = (self.display_name or "").strip()
        if not raw_display or raw_display.lower() in ("the", "a", "an", "unnamed_design", "target_module", ""):
            if self.design_name and self.design_name.strip().lower() not in ("the", "a", "an", "unnamed_design", ""):
                raw_display = self.design_name.strip()
            elif self.top_module and self.top_module.strip().lower() not in ("the", "a", "an", ""):
                raw_display = self.top_module.strip()
            else:
                raw_display = "EFS_Hardware_Module"
        self.display_name = raw_display

        # Guarantee non-empty design_id
        if not self.design_id:
            self.design_id = new_efs_id("dsn")

        # Guarantee non-empty canonical_id adhering to EFS_xxxxx or CANON_xxxxx
        raw_canon = (self.canonical_id or "").strip()
        if not raw_canon or raw_canon.lower() in ("the", "a", "an", ""):
            clean_id = self.design_id.replace("DSN_", "")
            self.canonical_id = f"EFS_{clean_id}"
        else:
            self.canonical_id = raw_canon

        # Guarantee generation_run_id
        if not self.generation_run_id:
            self.generation_run_id = new_efs_id("run")

        # Synchronize document identities from source_documents if not set
        if self.source_documents:
            first_doc = self.source_documents[0]
            if not self.document_id and getattr(first_doc, "document_id", None):
                self.document_id = first_doc.document_id
            if not self.document_hash and getattr(first_doc, "hash", None):
                self.document_hash = first_doc.hash
            if not self.ingestion_id and getattr(first_doc, "ingestion_id", None):
                self.ingestion_id = first_doc.ingestion_id

    def to_dict(self) -> Dict[str, Any]:
        d = strip_nulls(asdict(self))
        d["source_documents"] = [
            doc.to_dict() if hasattr(doc, "to_dict") else {"name": str(doc), "path": str(doc)}
            for doc in self.source_documents
        ]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSMetadata:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "source_documents"}
        source_docs = []
        for d in data.get("source_documents", []):
            if isinstance(d, EFSSourceDocument):
                source_docs.append(d)
            elif isinstance(d, dict):
                source_docs.append(EFSSourceDocument.from_dict(d))
            elif isinstance(d, str) and d:
                source_docs.append(EFSSourceDocument(name=d))
        kwargs["source_documents"] = source_docs
        return cls(**kwargs)


@dataclass
class EFSTraceability:
    """Traceability mapping an EFS IR object back to its source specification origin."""
    doc_id: Optional[str] = None
    source_document_id: Optional[str] = None
    source_document: Optional[str] = None
    document_hash: Optional[str] = None
    ingestion_id: Optional[str] = None
    chunk_id: Optional[str] = None
    block_id: Optional[str] = None
    page: Optional[int] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    original_text: Optional[str] = None
    source_text: Optional[str] = None
    paragraph: Optional[str] = None
    source_entity_id: Optional[str] = None
    source_relationship_id: Optional[str] = None
    extraction_method: str = "unknown"  # e.g., "llm", "heuristic", "regex", "table_extraction", "document_parser", "graph_extraction", "inferred", "manual", "unknown"
    confidence: Optional[float] = None  # None when extraction_method == "unknown"

    def __post_init__(self):
        if self.doc_id and not self.source_document_id:
            self.source_document_id = self.doc_id
        elif self.source_document_id and not self.doc_id:
            self.doc_id = self.source_document_id
            
        if self.original_text and not self.source_text:
            self.source_text = self.original_text
        elif self.source_text and not self.original_text:
            self.original_text = self.source_text

        # If extraction method is unknown, confidence must not be falsely represented; set to None
        if self.extraction_method == "unknown":
            self.confidence = None

    def to_dict(self) -> Dict[str, Any]:
        d = strip_nulls(asdict(self))
        d["confidence"] = self.confidence
        return d

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> EFSTraceability:
        if not data:
            return cls()
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EFSComponent:
    """A distinct hardware module or subblock in the design architecture."""
    component_id: str = field(default_factory=lambda: new_efs_id("comp"))
    name: str = ""
    type: str = ""                     # e.g., "controller", "register_file", "fifo", "arbiter", "datapath"
    description: str = ""
    interfaces: List[str] = field(default_factory=list)  # list of interface_ids
    clock_domain: Optional[str] = None
    reset_domain: Optional[str] = None
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["traceability"] = self.traceability.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSComponent:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "traceability"}
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSInterface:
    """An explicit port interface group connecting hardware components or pins."""
    interface_id: str = field(default_factory=lambda: new_efs_id("if"))
    name: str = ""
    protocol: str = "generic"          # e.g., "discovered_protocol", "custom"
    role: str = ""                     # e.g., "master", "slave", "initiator", "target" (generic)
    source_component: Optional[str] = None       # component_id
    destination_component: Optional[str] = None  # component_id
    signals: List[str] = field(default_factory=list)  # list of signal_ids
    clock: Optional[str] = None
    reset: Optional[str] = None
    description: str = ""
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["traceability"] = self.traceability.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSInterface:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "traceability"}
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSSignal:
    """A port signal or internal net in the design module."""
    signal_id: str = field(default_factory=lambda: new_efs_id("sig"))
    name: str = ""
    width: str = "1"                   # e.g. "1", "32", "31:0"
    direction: str = "input"           # input | output | inout
    datatype: str = "logic"            # logic | wire | reg | std_logic
    signedness: bool = False
    clock: Optional[str] = None
    reset: Optional[str] = None
    protocol_role: Optional[str] = None # e.g. "AWADDR", "READY", "VALID"
    interface: Optional[str] = None    # interface_id
    owner: Optional[str] = None        # component_id
    consumer: Optional[str] = None     # component_id
    description: str = ""
    semantic_role: str = "data"        # data | control | clock | reset | address | handshake | response
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["traceability"] = self.traceability.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSSignal:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "traceability"}
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSRegisterField:
    """A single bitfield within a hardware register."""
    name: str = ""
    msb: int = 0
    lsb: int = 0
    width: int = 1
    access: str = "RW"                 # RW | RO | WO | RC | W1C
    reset_value: str = "0"
    description: str = ""
    enum_values: Dict[str, str] = field(default_factory=dict)  # map value string to description

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSRegisterField:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EFSRegister:
    """A control, status, or configuration register in the address space."""
    register_id: str = field(default_factory=lambda: new_efs_id("reg"))
    name: str = ""
    address: Optional[str] = None      # absolute hex address if known, e.g. "0x4000_0000" or None
    offset: Optional[str] = None       # relative offset, e.g. "0x0C" or None if missing
    offset_source: str = "missing"     # "explicit" | "missing" | "inferred"
    width: Optional[int] = 32
    access_type: str = "RW"            # RW | RO | WO
    access: str = "RW"
    reset_value: Optional[str] = None
    fields: List[EFSRegisterField] = field(default_factory=list)
    description: str = ""
    side_effects: str = ""
    owner: Optional[str] = None        # component_id
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    def __post_init__(self):
        # Normalize access and access_type so they cannot diverge
        if self.access != "RW" and self.access_type == "RW":
            self.access_type = self.access
        elif self.access_type != "RW" and self.access == "RW":
            self.access = self.access_type

        # Normalize width to integer or None
        if self.width is not None:
            if isinstance(self.width, str):
                try:
                    self.width = int(self.width)
                except ValueError:
                    self.width = None
            elif not isinstance(self.width, int) or self.width <= 0:
                self.width = None

        # Normalize address: empty string becomes None
        if self.address == "":
            self.address = None

        # Normalize offset: format numeric or hex string cleanly, or preserve None
        if self.offset is None or self.offset == "" or str(self.offset).lower() in ("none", "null", "tbd", "unassigned", "-", "--"):
            self.offset = None
            if self.offset_source != "inferred":
                self.offset_source = "missing"
        else:
            if isinstance(self.offset, int):
                self.offset = f"0x{self.offset:X}"
                if self.offset_source == "missing":
                    self.offset_source = "explicit"
            elif isinstance(self.offset, str):
                clean_off = self.offset.strip()
                if clean_off.isdigit():
                    self.offset = f"0x{int(clean_off):X}"
                    if self.offset_source == "missing":
                        self.offset_source = "explicit"
                elif clean_off.lower().startswith("0x"):
                    self.offset = clean_off
                    if self.offset_source == "missing":
                        self.offset_source = "explicit"
                else:
                    norm = normalize_hardware_offset(clean_off)
                    if norm is not None:
                        self.offset = norm
                        if self.offset_source == "missing":
                            self.offset_source = "explicit"
                    else:
                        self.offset = None
                        if self.offset_source != "inferred":
                            self.offset_source = "missing"

    def to_dict(self) -> Dict[str, Any]:
        d = strip_nulls(asdict(self))
        d["address"] = self.address
        d["offset"] = self.offset
        d["offset_source"] = self.offset_source
        d["reset_value"] = self.reset_value
        d["fields"] = [f.to_dict() for f in self.fields]
        d["traceability"] = self.traceability.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSRegister:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k not in ("fields", "traceability")}
        kwargs["fields"] = [EFSRegisterField.from_dict(f) for f in data.get("fields", [])]
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSTransaction:
    """A structured bus transfer or communication operation transaction."""
    transaction_id: str = field(default_factory=lambda: new_efs_id("tx"))
    name: str = ""
    type: str = "generic"
    description: str = ""
    protocol: str = "generic"
    initiator: Optional[str] = None    # interface_id or component_id
    target: Optional[str] = None       # interface_id or component_id
    participating_interfaces: List[str] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)      # signals involved
    ordered_steps: List[str] = field(default_factory=list)# execution step descriptions
    preconditions: List[str] = field(default_factory=list)# conditions required to start
    completion_condition: str = ""
    response: str = ""
    expected_response: str = ""
    error_behavior: str = ""
    timing: str = ""
    ordering: str = ""                 # ordering rules (e.g. out-of-order, in-order)
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    def __post_init__(self):
        if self.expected_response and not self.response:
            self.response = self.expected_response
        elif self.response and not self.expected_response:
            self.expected_response = self.response

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["traceability"] = self.traceability.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSTransaction:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "traceability"}
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSState:
    """A state defined inside a Finite State Machine."""
    state_id: str = field(default_factory=lambda: new_efs_id("st"))
    name: str = ""
    encoding: str = ""                 # binary / one-hot representation, e.g. "2'b00"
    outputs: Dict[str, str] = field(default_factory=dict) # signal_name -> output value expression
    actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSState:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EFSTransition:
    """A directed state transition between FSM states."""
    transition_id: str = field(default_factory=lambda: new_efs_id("trans"))
    source_state: str = ""             # state name or ID
    target_state: str = ""             # state name or ID
    condition: str = ""                # transition trigger expression
    action: str = ""                   # side effects executed during transition
    priority: int = 0
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    @property
    def from_state(self) -> str:
        return self.source_state

    @from_state.setter
    def from_state(self, val: str) -> None:
        self.source_state = val

    @property
    def to_state(self) -> str:
        return self.target_state

    @to_state.setter
    def to_state(self, val: str) -> None:
        self.target_state = val

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["traceability"] = self.traceability.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSTransition:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "traceability"}
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSFSM:
    """A Finite State Machine model orchestrating hardware state transitions."""
    fsm_id: str = field(default_factory=lambda: new_efs_id("fsm"))
    name: str = ""
    states: List[EFSState] = field(default_factory=list)
    initial_state: str = ""            # state name
    encoding: str = "binary"           # binary | onehot | gray
    transitions: List[EFSTransition] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["states"] = [s.to_dict() for s in self.states]
        d["transitions"] = [t.to_dict() for t in self.transitions]
        d["traceability"] = self.traceability.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSFSM:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k not in ("states", "transitions", "traceability")}
        kwargs["states"] = [EFSState.from_dict(s) for s in data.get("states", [])]
        kwargs["transitions"] = [EFSTransition.from_dict(t) for t in data.get("transitions", [])]
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSConstraint:
    """A generic hardware protocol or structural constraint definition."""
    constraint_id: str = field(default_factory=lambda: new_efs_id("const"))
    type: str = "handshake"            # handshake | ordering | dependency | stability | timeout | etc.
    source_objects: List[str] = field(default_factory=list) # signal / component IDs
    target_objects: List[str] = field(default_factory=list) # signal / component IDs
    condition: str = ""                # when the constraint is active
    expected_behavior: str = ""       # property statement
    severity: str = "MAJOR"            # CRITICAL | MAJOR | MINOR | WARNING
    protocol: str = "generic"
    related_transaction: Optional[str] = None # transaction_id
    related_fsm: Optional[str] = None         # fsm_id
    traceability: EFSTraceability = field(default_factory=EFSTraceability)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["traceability"] = self.traceability.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSConstraint:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "traceability"}
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSTimingRule:
    """Timing relationship or delay specification rule."""
    rule_id: str = field(default_factory=lambda: new_efs_id("time"))
    type: str = "latency"              # setup | hold | latency | timeout | cycle_relationship | clock_relationship
    cycle_relationship: str = ""
    clock_relationship: str = ""
    ordering: str = ""
    minimum_delay: str = ""            # cycles or time e.g., "1 cycle", "5ns"
    maximum_delay: str = ""
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["traceability"] = self.traceability.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSTimingRule:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "traceability"}
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSProtocolRule:
    """Abstract protocol specification rules extracted from golden protocol document."""
    rule_id: str = field(default_factory=lambda: new_efs_id("prule"))
    name: str = ""
    type: str = "handshake"            # handshake | transaction_ordering | response_behavior | error_handling
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["traceability"] = self.traceability.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSProtocolRule:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "traceability"}
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSFlowStep:
    """A single sequential step inside a behavioral Flow flow diagram."""
    step_id: str = field(default_factory=lambda: new_efs_id("fstep"))
    component: Optional[str] = None     # component_id
    interface: Optional[str] = None     # interface_id
    action_description: str = ""
    condition: str = ""
    expected_events: List[str] = field(default_factory=list)
    state_transitions: List[str] = field(default_factory=list)
    signal_behavior: str = ""
    timing_expectations: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSFlowStep:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EFSFlow:
    """A sequences of hardware events representing transaction or data flow through blocks."""
    flow_id: str = field(default_factory=lambda: new_efs_id("flow"))
    name: str = ""
    components: List[str] = field(default_factory=list)  # component_ids
    interfaces: List[str] = field(default_factory=list)  # interface_ids
    ordered_steps: List[EFSFlowStep] = field(default_factory=list)
    conditions: str = ""
    expected_events: List[str] = field(default_factory=list)
    state_transitions: List[str] = field(default_factory=list)
    signal_behavior: str = ""
    timing_expectations: str = ""
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ordered_steps"] = [s.to_dict() for s in self.ordered_steps]
        d["traceability"] = self.traceability.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSFlow:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k not in ("ordered_steps", "traceability")}
        kwargs["ordered_steps"] = [EFSFlowStep.from_dict(s) for s in data.get("ordered_steps", [])]
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSAssertionIntent:
    """Structured assertion specification intended for compilation into SVA checks."""
    assertion_id: str = field(default_factory=lambda: new_efs_id("ast"))
    derived_constraint_id: Optional[str] = None
    property_type: str = "assert"      # assert | assume | cover
    clock: str = "clk"
    reset_condition: str = "rst_n"
    antecedent: str = ""               # trigger condition
    consequent: str = ""               # outcome condition
    temporal_behavior: str = "implication" # implication | overlap | cycles_delay
    severity: str = "error"            # fatal | error | warning
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["traceability"] = self.traceability.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSAssertionIntent:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "traceability"}
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSCoverageIntent:
    """Coverage metric specification to check simulation completeness."""
    coverage_id: str = field(default_factory=lambda: new_efs_id("cov"))
    type: str = "state"                # state | transition | transaction | scenario | constraint | signal | cross
    target_object_id: str = ""         # ID of FSM, signal, register, or transaction
    description: str = ""
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["traceability"] = self.traceability.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSCoverageIntent:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "traceability"}
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSInstructionField:
    """A field within an instruction encoding."""
    name: str = ""
    width: int = 0
    msb: int = 0
    lsb: int = 0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSInstructionField:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EFSInstructionFormat:
    """An instruction layout format definition."""
    format_id: str = field(default_factory=lambda: new_efs_id("fmt"))
    name: str = ""
    total_width: int = 32
    fields: List[EFSInstructionField] = field(default_factory=list)
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["fields"] = [f.to_dict() for f in self.fields]
        d["traceability"] = self.traceability.to_dict()
        return strip_nulls(d)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSInstructionFormat:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k not in ("fields", "traceability")}
        kwargs["fields"] = [EFSInstructionField.from_dict(f) for f in data.get("fields", [])]
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSExecutionModel:
    """An execution model definition (e.g. pipeline, tiling, scheduling policy)."""
    execution_id: str = field(default_factory=lambda: new_efs_id("exec"))
    type: str = "PIPELINE"               # PIPELINE | TILING | PARALLEL_EXECUTION | SCHEDULING | WORK_STEALING
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["traceability"] = self.traceability.to_dict()
        return strip_nulls(d)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSExecutionModel:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "traceability"}
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSInstruction:
    """An instruction definition supported by the instruction decoder."""
    instruction_id: str = field(default_factory=lambda: new_efs_id("instr"))
    name: str = ""
    mnemonic: str = ""
    opcode: str = ""
    format: str = ""
    operation_id: Optional[str] = None
    format_id: Optional[str] = None
    opcode_id: Optional[str] = None
    operands: List[str] = field(default_factory=list)
    width: int = 32
    fields: List[EFSInstructionField] = field(default_factory=list)
    description: str = ""
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    def __post_init__(self):
        if self.mnemonic and not self.name:
            self.name = self.mnemonic
        elif self.name and not self.mnemonic:
            self.mnemonic = self.name

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["fields"] = [f.to_dict() for f in self.fields]
        d["traceability"] = self.traceability.to_dict()
        return strip_nulls(d)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSInstruction:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k not in ("fields", "traceability")}
        kwargs["fields"] = [EFSInstructionField.from_dict(f) for f in data.get("fields", [])]
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSOpcode:
    """An opcode representation in the instruction encoding table."""
    opcode_id: str = field(default_factory=lambda: new_efs_id("opc"))
    mnemonic: str = ""
    binary_encoding: str = ""
    operation: str = ""
    operands: List[str] = field(default_factory=list)
    instruction_id: Optional[str] = None
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    @property
    def encoding(self) -> str:
        return self.binary_encoding

    @encoding.setter
    def encoding(self, val: str) -> None:
        self.binary_encoding = val

    @property
    def description(self) -> str:
        return self.operation

    @description.setter
    def description(self, val: str) -> None:
        self.operation = val

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["traceability"] = self.traceability.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSOpcode:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "traceability"}
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSMemoryRegion:
    """A hardware memory region, cache, or buffer domain."""
    region_id: str = field(default_factory=lambda: new_efs_id("memreg"))
    name: str = ""
    type: str = ""                     # e.g., "SRAM", "cache_L1", "cache_L2", "cache_shared", "DMA_buffer"
    base_address: str = ""
    size: str = ""
    depth: int = 0
    width: int = 0
    description: str = ""
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["traceability"] = self.traceability.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSMemoryRegion:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "traceability"}
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSDataStructure:
    """A data structure representing protocol packets, tiling descriptors, etc."""
    structure_id: str = field(default_factory=lambda: new_efs_id("dstruct"))
    name: str = ""                      # e.g., "matrix_descriptor"
    fields: List[Dict[str, Any]] = field(default_factory=list)  # field list of dictionary descriptors
    description: str = ""
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["traceability"] = self.traceability.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSDataStructure:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "traceability"}
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSError:
    """An error condition definition mapped to error status bits."""
    error_id: str = field(default_factory=lambda: new_efs_id("err"))
    name: str = ""
    code: str = ""
    condition: str = ""
    behavior: str = ""
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["traceability"] = self.traceability.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSError:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "traceability"}
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSPerformanceRequirement:
    """A performance constraint target."""
    perf_id: str = field(default_factory=lambda: new_efs_id("perf"))
    metric: str = ""                    # throughput | latency | frequency | power
    target_value: str = ""
    condition: str = ""
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["traceability"] = self.traceability.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSPerformanceRequirement:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "traceability"}
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSParameter:
    """A generic hardware parameter (e.g. Verilog/VHDL parameter or generic)."""
    parameter_id: str = field(default_factory=lambda: new_efs_id("param"))
    name: str = ""
    value: str = ""
    default_value: str = ""
    datatype: str = "integer"
    owner: str = ""
    description: str = ""
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["traceability"] = self.traceability.to_dict()
        return strip_nulls(d)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSParameter:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "traceability"}
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSFunctionalBehavior:
    """A generic hardware functional behavior representation."""
    behavior_id: str = field(default_factory=lambda: new_efs_id("beh"))
    name: str = ""
    subject: str = ""
    action: str = ""
    trigger: str = ""
    expected_result: str = ""
    conditions: List[str] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    description: str = ""
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["traceability"] = self.traceability.to_dict()
        return strip_nulls(d)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSFunctionalBehavior:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "traceability"}
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSResolutionCandidate:
    """Structured proposal candidate for resolving an EFSConflict."""
    candidate_id: str = field(default_factory=lambda: new_efs_id("cand"))
    action: str = "REASSIGN_VALUE"  # REASSIGN_VALUE | MODIFY_PROPERTY | ADD_CONSTRUCT | OVERRIDE
    target_entity: str = ""
    property: str = ""
    current_value: Any = None
    proposed_value: Any = None
    reason: str = ""
    confidence: Optional[float] = 1.0
    validation_status: str = "PENDING"  # PENDING | VALID | REJECTED

    def to_dict(self) -> Dict[str, Any]:
        return strip_nulls(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSResolutionCandidate:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EFSResolution:
    """An approved or proposed resolution overriding an EFSConflict without mutating original source requirements."""
    resolution_id: str = field(default_factory=lambda: new_efs_id("res"))
    conflict_id: str = ""
    target_entity: str = ""
    property: str = ""
    resolution_type: str = "ACCEPT_PROPOSAL"  # ACCEPT_PROPOSAL | CUSTOM_OVERRIDE | CUSTOM_REQUIREMENT | REJECT
    proposed_change: str = ""
    rationale: str = ""
    reason: str = ""
    affected_requirements: List[str] = field(default_factory=list)
    evidence: str = ""
    source_values: List[str] = field(default_factory=list)
    proposed_value: Any = None
    resolved_value: Any = None
    original_value: Any = None
    field_name: str = ""
    resolution_source: str = "human"          # human | ai_approved | user_custom
    validation_status: str = "PASSED"         # PASSED | REJECTED | PENDING
    approval_status: str = "UNRESOLVED"       # UNRESOLVED | PROPOSED | APPROVED | REJECTED
    approved_by: str = "user"                 # user | AI_ASSISTANT
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Synchronize target_entity and affected_requirements
        if self.target_entity and not self.affected_requirements:
            self.affected_requirements = [self.target_entity]
        elif self.affected_requirements and not self.target_entity:
            self.target_entity = self.affected_requirements[0]

        # Synchronize property and field_name
        if self.property and not self.field_name:
            self.field_name = self.property
        elif self.field_name and not self.property:
            self.property = self.field_name

        # Synchronize rationale and reason
        if self.rationale and not self.reason:
            self.reason = self.rationale
        elif self.reason and not self.rationale:
            self.rationale = self.reason

        # Synchronize proposed_value and resolved_value
        if self.proposed_value is not None and self.resolved_value is None:
            self.resolved_value = self.proposed_value
        elif self.resolved_value is not None and self.proposed_value is None:
            self.proposed_value = self.resolved_value

        # Synchronize resolution_source and approved_by
        if self.approved_by == "AI_ASSISTANT" and self.resolution_source == "human":
            self.resolution_source = "ai_approved"

    def to_dict(self) -> Dict[str, Any]:
        d = strip_nulls(asdict(self))
        # Add standard schema compatibility aliases
        d["issue_id"] = self.conflict_id
        d["issue_type"] = self.resolution_type
        d["original_evidence"] = self.evidence
        d["resolution"] = self.proposed_change
        d["source"] = self.approved_by
        d["approved_by_user"] = (self.approval_status == "APPROVED")
        d["original_value"] = self.original_value
        d["resolved_value"] = self.resolved_value
        d["target_entity"] = self.target_entity
        d["property"] = self.property
        d["field_name"] = self.field_name
        d["resolution_source"] = self.resolution_source
        d["validation_status"] = self.validation_status
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSResolution:
        # Support aliases if present in input dict
        if "conflict_id" not in data and "issue_id" in data:
            data["conflict_id"] = data["issue_id"]
        if "proposed_change" not in data and "resolution" in data:
            data["proposed_change"] = data["resolution"]
        if "evidence" not in data and "original_evidence" in data:
            data["evidence"] = data["original_evidence"]
        if "approved_by" not in data and "source" in data:
            data["approved_by"] = data["source"]
        if "proposed_value" not in data and "resolved_value" in data:
            data["proposed_value"] = data["resolved_value"]
        if "resolved_value" not in data and "proposed_value" in data:
            data["resolved_value"] = data["proposed_value"]
        if "field_name" not in data and "property" in data:
            data["field_name"] = data["property"]
        if "property" not in data and "field_name" in data:
            data["property"] = data["field_name"]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EFSConflict:
    """A structured conflict or ambiguity detected in the specifications requiring human decision."""
    conflict_id: str = field(default_factory=lambda: new_efs_id("conf"))
    type: str = ""                      # e.g., DUPLICATE_OPCODE, DUPLICATE_REGISTER_ADDRESS, CONFLICTING_WIDTH, MISSING_SIGNAL
    severity: str = "CRITICAL"          # CRITICAL | MAJOR | MINOR | WARNING
    blocking: bool = True
    status: str = field(default_factory=lambda: ConflictStatus("UNRESOLVED"))          # UNRESOLVED | PENDING | RESOLVING | RESOLVED | REJECTED | APPROVED
    title: str = ""
    description: str = ""
    explanation: str = ""               # Why it matters / why this is a problem
    impact: str = ""
    why_it_matters: str = ""            # Alias / why this is a problem
    entities: List[str] = field(default_factory=list)
    affected_entities: List[str] = field(default_factory=list)
    target_entities: List[str] = field(default_factory=list)
    property: str = ""
    properties: List[str] = field(default_factory=list)
    conflicting_values: List[str] = field(default_factory=list)
    source_references: List[Dict[str, Any]] = field(default_factory=list)  # [{'entity': str, 'source': str, 'section': str, 'value': str}]
    current_values: Dict[str, Any] = field(default_factory=dict)           # e.g., {'LOAD': '1010', 'STORE': '1010'}
    source_objects: List[str] = field(default_factory=list)                # IDs of conflicting entities
    evidence: str = ""
    suggestions: List[str] = field(default_factory=list)
    resolution_candidates: List[EFSResolutionCandidate] = field(default_factory=list)
    ai_proposal: Optional[Dict[str, Any]] = None                           # Structured summary of AI proposal
    resolution: Optional[EFSResolution] = None
    resolution_id: Optional[str] = None
    candidates: List[str] = field(default_factory=list)
    resolution_status: str = "UNRESOLVED"                                  # UNRESOLVED | AWAITING_HUMAN_DECISION | RESOLVED
    requires_human_decision: bool = True
    resolved_reason: str = ""                                              # Audit reason when marked RESOLVED
    semantic_key: str = ""                                                 # Deterministic canonical semantic conflict key
    generation_run_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    traceability: EFSTraceability = field(default_factory=EFSTraceability)
    diagnostic_type: str = "SPEC_CONFLICT"                                  # SPEC_CONFLICT | MISSING_SPECIFICATION | TARGET_RESOLUTION_FAILED | VALIDATION_ERROR | PLANNER_ERROR
    resolver_diagnostic: Optional[Dict[str, Any]] = None                    # Structured diagnostic details when target resolution fails

    def __post_init__(self):
        if self.status in ("UNRESOLVED", "ACTIVE") and not isinstance(self.status, ConflictStatus):
            self.status = ConflictStatus(self.status)
        if not self.diagnostic_type:
            self.diagnostic_type = DiagnosticType.SPEC_CONFLICT.value
        # Sync entities, affected_entities, and target_entities
        all_ents = self.target_entities or self.entities or self.affected_entities
        if all_ents:
            if not self.entities:
                self.entities = list(all_ents)
            if not self.affected_entities:
                self.affected_entities = list(all_ents)
            if not self.target_entities:
                self.target_entities = list(all_ents)

        # Sync property and properties
        if self.properties and not self.property:
            self.property = self.properties[0]
        elif self.property and not self.properties:
            self.properties = [self.property]
            
        # Default title if empty
        if not self.title:
            clean_type = self.type.replace("_", " ").title()
            self.title = f"{clean_type} Detected" if not clean_type.endswith("Detected") else clean_type

        # Default description if empty
        if not self.description:
            if self.entities and self.conflicting_values:
                self.description = f"Conflict between {', '.join(self.entities)} with values: {', '.join(str(v) for v in self.conflicting_values)}."
            elif self.evidence:
                self.description = self.evidence

        # Build current_values mapping if empty
        if not self.current_values and self.entities and self.conflicting_values:
            for i, ent in enumerate(self.entities):
                val = self.conflicting_values[i] if i < len(self.conflicting_values) else self.conflicting_values[0]
                self.current_values[ent] = val

        if not self.why_it_matters:
            self.why_it_matters = self.explanation or self.impact or "Specification ambiguity requires human resolution."

        # Sync resolution reference
        if self.resolution:
            if not self.resolution_id:
                self.resolution_id = self.resolution.resolution_id
            if self.resolution.approval_status == "APPROVED":
                self.resolution_status = "RESOLVED"
                self.status = "APPROVED"

        if self.resolution_status == "RESOLVED" or self.status == "RESOLVED":
            self.requires_human_decision = False
            self.blocking = False
        else:
            # Synchronize blocking as derived state
            self.blocking = (self.severity.upper() in ("CRITICAL", "ERROR"))

    def resolve(self, resolution: Optional[EFSResolution] = None) -> None:
        """Resolve this conflict, linking resolution and clearing blocking flags."""
        if resolution:
            self.resolution = resolution
            self.resolution_id = resolution.resolution_id
        self.resolution_status = "RESOLVED"
        self.status = "APPROVED"
        self.requires_human_decision = False
        self.blocking = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["resolution_id"] = self.resolution_id
        if self.resolution:
            d["resolution"] = self.resolution.to_dict()
        d["traceability"] = self.traceability.to_dict()
        d["affected_entities"] = list(self.affected_entities or self.entities)
        d["entities"] = list(self.entities or self.affected_entities)
        d["target_entities"] = list(self.target_entities or self.entities)
        d["properties"] = list(self.properties or ([self.property] if self.property else []))
        d["title"] = self.title
        d["requires_human_decision"] = self.requires_human_decision
        d["blocking"] = self.blocking
        d["generation_run_id"] = self.generation_run_id
        d["current_values"] = dict(self.current_values)
        d["source_references"] = list(self.source_references)
        d["resolution_candidates"] = [
            c.to_dict() if isinstance(c, EFSResolutionCandidate) else c
            for c in self.resolution_candidates
        ]
        if self.ai_proposal:
            d["ai_proposal"] = dict(self.ai_proposal)
        d["diagnostic_type"] = self.diagnostic_type
        if self.resolver_diagnostic:
            d["resolver_diagnostic"] = dict(self.resolver_diagnostic)
        return strip_nulls(d)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSConflict:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k not in ("traceability", "resolution", "resolution_candidates")}
        if data.get("resolution"):
            kwargs["resolution"] = EFSResolution.from_dict(data["resolution"])
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        raw_cands = data.get("resolution_candidates", [])
        kwargs["resolution_candidates"] = [
            EFSResolutionCandidate.from_dict(c) if isinstance(c, dict) else c
            for c in raw_cands
        ]
        return cls(**kwargs)


@dataclass
class EFSValidationSummary:
    """Summary metrics of EFS IR validation checks."""
    total: int = 0
    critical: int = 0
    warning: int = 0
    info: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSValidationSummary:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EFSValidationResult:
    """First-class validation findings report for the canonical EFS IR."""
    status: str = "PASSED"  # PASSED | CONFLICTS_FOUND | BLOCKED | WARNINGS_FOUND | FAILED
    summary: EFSValidationSummary = field(default_factory=EFSValidationSummary)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    design_id: str = ""
    canonical_id: str = ""
    generation_run_id: str = ""
    document_id: str = ""
    ingestion_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "design_id": self.design_id,
            "canonical_id": self.canonical_id,
            "generation_run_id": self.generation_run_id,
            "document_id": self.document_id,
            "ingestion_id": self.ingestion_id,
            "summary": self.summary.to_dict() if hasattr(self.summary, "to_dict") else self.summary,
            "findings": list(self.findings)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSValidationResult:
        summary_raw = data.get("summary", {})
        summary = EFSValidationSummary.from_dict(summary_raw) if isinstance(summary_raw, dict) else EFSValidationSummary()
        return cls(
            status=data.get("status", "PASSED"),
            summary=summary,
            findings=data.get("findings", []),
            design_id=data.get("design_id", ""),
            canonical_id=data.get("canonical_id", ""),
            generation_run_id=data.get("generation_run_id", ""),
            document_id=data.get("document_id", ""),
            ingestion_id=data.get("ingestion_id", "")
        )


@dataclass
class EFSExtractionQuality:
    """Quality scorecard describing extraction and completeness status."""
    extraction_status: str = "success"        # success | partial | failed
    source_completeness: str = "unknown"     # unknown | partial | complete
    ir_completeness: str = "partial"         # partial | complete
    missing_information: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return strip_nulls(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSExtractionQuality:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EFSRequirementItem:
    """Generic first-class requirement representation with full source evidence and status."""
    requirement_id: str = field(default_factory=lambda: new_efs_id("req"))
    category: str = "INTERFACE"  # INTERFACE | REGISTER | FSM | DATAPATH | PROTOCOL | RESET | TIMING
    property: str = ""
    value: Any = None
    status: str = "COMPLETE"     # COMPLETE | MISSING | AMBIGUOUS | CONFLICT | EFSIR_EXTRACTION_GAP
    evidence: Dict[str, Any] = field(default_factory=dict)
    source_location: str = "N/A"

    def to_dict(self) -> Dict[str, Any]:
        return strip_nulls(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSRequirementItem:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EFSRequirement:
    """A requirement node representing a specification assertion."""
    requirement_id: str = field(default_factory=lambda: new_efs_id("req"))
    category: str = ""
    target_object: str = ""
    description: str = ""
    is_critical: bool = True
    title: str = ""
    requirement_type: str = ""
    req_id: str = ""
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    def __post_init__(self):
        if self.req_id and not self.requirement_id:
            self.requirement_id = self.req_id
        elif self.requirement_id and not self.req_id:
            self.req_id = self.requirement_id
        if self.title and not self.description:
            self.description = self.title
        elif self.description and not self.title:
            self.title = self.description[:60]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["traceability"] = self.traceability.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSRequirement:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "traceability"}
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


@dataclass
class EFSIR:
    """The root Intermediate Representation object containing the full hardware design model."""
    metadata: EFSMetadata = field(default_factory=EFSMetadata)
    components: List[EFSComponent] = field(default_factory=list)
    interfaces: List[EFSInterface] = field(default_factory=list)
    signals: List[EFSSignal] = field(default_factory=list)
    clocks: List[EFSSignal] = field(default_factory=list)
    resets: List[EFSSignal] = field(default_factory=list)
    registers: List[EFSRegister] = field(default_factory=list)
    transactions: List[EFSTransaction] = field(default_factory=list)
    fsms: List[EFSFSM] = field(default_factory=list)
    constraints: List[EFSConstraint] = field(default_factory=list)
    timing_rules: List[EFSTimingRule] = field(default_factory=list)
    protocol_rules: List[EFSProtocolRule] = field(default_factory=list)
    flows: List[EFSFlow] = field(default_factory=list)
    functional_behavior: List[EFSFunctionalBehavior] = field(default_factory=list)
    assertions: List[EFSAssertionIntent] = field(default_factory=list)
    coverage: List[EFSCoverageIntent] = field(default_factory=list)
    # Extended Canonical Models
    parameters: List[EFSParameter] = field(default_factory=list)
    instructions: List[EFSInstruction] = field(default_factory=list)
    instruction_formats: List[EFSInstructionFormat] = field(default_factory=list)
    opcodes: List[EFSOpcode] = field(default_factory=list)
    memory_regions: List[EFSMemoryRegion] = field(default_factory=list)
    data_structures: List[EFSDataStructure] = field(default_factory=list)
    execution_models: List[EFSExecutionModel] = field(default_factory=list)
    errors: List[EFSError] = field(default_factory=list)
    performance_requirements: List[EFSPerformanceRequirement] = field(default_factory=list)
    conflicts: List[EFSConflict] = field(default_factory=list)
    requirements: List[EFSRequirement] = field(default_factory=list)
    resolutions: List[EFSResolution] = field(default_factory=list)
    extraction_quality: EFSExtractionQuality = field(default_factory=EFSExtractionQuality)
    extraction_coverage: Dict[str, Any] = field(default_factory=dict)
    validation: Optional[EFSValidationResult] = None

    def __post_init__(self):
        if not self.clocks and self.signals:
            self.clocks = [s for s in self.signals if s.semantic_role == "clock"]
        if not self.resets and self.signals:
            self.resets = [s for s in self.signals if s.semantic_role == "reset"]

    def get_resolved_view(self) -> EFSIR:
        """
        Builds and returns a resolved view overlaying approved resolutions onto the base EFSIR
        without mutating the immutable original source requirements.
        """
        import copy
        import re
        resolved_ir = copy.deepcopy(self)
        resolved_ir.metadata.resolution_version = getattr(resolved_ir.metadata, "resolution_version", 0) + 1
        resolved_ir.metadata.model_version = getattr(resolved_ir.metadata, "model_version", 1) + 1

        approved_res = [r for r in resolved_ir.resolutions if r.approval_status == "APPROVED"]
        
        for res in approved_res:
            target_prop = res.proposed_change.split("=")[0].strip() if "=" in res.proposed_change else res.proposed_change
            if "." in target_prop:
                target_entity_from_prop = target_prop.split(".")[0].strip()
                target_prop_name = target_prop.split(".")[-1].strip()
            else:
                target_entity_from_prop = res.target_entity or None
                target_prop_name = res.property or target_prop

            if "=" in res.proposed_change:
                val = res.proposed_change.split("=")[1].strip()
            elif res.resolved_value is not None:
                val = res.resolved_value
            else:
                val = res.proposed_value
            
            # Match conflict and sync resolution status
            conf = next((c for c in resolved_ir.conflicts if c.conflict_id == res.conflict_id), None)
            if not conf:
                conf = next((c for c in resolved_ir.conflicts if any(e in c.entities for e in res.affected_requirements) or c.property == target_prop_name), None)
            if conf:
                conf.status = "APPROVED"
                conf.resolution_status = "RESOLVED"
                conf.blocking = False
                conf.requires_human_decision = False
                conf.resolution_id = res.resolution_id
                conf.resolution = res
                res.conflict_id = conf.conflict_id

            # Determine explicit target entities to update
            if res.affected_requirements:
                affected_ids = set(res.affected_requirements)
            elif target_entity_from_prop:
                affected_ids = {target_entity_from_prop}
            elif conf and conf.source_objects:
                affected_ids = set(conf.source_objects)
            elif conf and conf.entities:
                affected_ids = {conf.entities[-1]}  # Target second/last entity if ambiguous
            else:
                affected_ids = set()

            val_str = str(val).strip() if val is not None else ""

            # Apply to matching Signals
            for sig in resolved_ir.signals:
                if sig.signal_id in affected_ids or sig.name in affected_ids or any(sig.name.lower() == e.lower() for e in affected_ids):
                    if val_str.lower() in ("input", "in", "output", "out", "inout") or "direction" in target_prop_name.lower() or "direction" in res.proposed_change.lower():
                        sig.direction = "input" if val_str.lower() in ("input", "in") else ("output" if val_str.lower() in ("output", "out") else val_str)
                    if "width" in target_prop_name.lower() or "width" in res.proposed_change.lower() or val_str.isdigit():
                        if val_str.isdigit() or ":" in val_str:
                            sig.width = val_str

            # Apply to matching Registers
            for reg in resolved_ir.registers:
                if reg.register_id in affected_ids or reg.name in affected_ids or any(reg.name.lower() == e.lower() for e in affected_ids):
                    if "offset" in target_prop_name.lower() or "address" in target_prop_name.lower() or val_str.startswith("0x") or val_str.isdigit():
                        if val_str.startswith("0x") or val_str.isdigit():
                            reg.offset = val_str
                            reg.address = val_str
                    elif "width" in target_prop_name.lower():
                        if val_str.isdigit():
                            reg.width = int(val_str)

            # Apply to Opcodes / Instructions
            for opc in resolved_ir.opcodes:
                if opc.opcode_id in affected_ids or opc.mnemonic in affected_ids or any(opc.mnemonic.lower() == e.lower() for e in affected_ids):
                    if "encoding" in target_prop_name.lower() or "opcode" in target_prop_name.lower() or "'" in val_str or val_str.startswith("0x") or all(c in '01' for c in val_str):
                        opc.binary_encoding = val_str

            for inst in resolved_ir.instructions:
                if inst.instruction_id in affected_ids or inst.mnemonic in affected_ids or any(inst.mnemonic.lower() == e.lower() for e in affected_ids):
                    if "encoding" in target_prop_name.lower() or "opcode" in target_prop_name.lower() or "'" in val_str or val_str.startswith("0x") or all(c in '01' for c in val_str):
                        inst.opcode = val_str

            # Apply to FSMs (States, Transitions, and Conditions)
            is_fsm_target = (
                any(e.lower() in ("fsm", "state_machine", "statemachine") for e in affected_ids) or
                (target_entity_from_prop and target_entity_from_prop.lower() in ("fsm", "state_machine")) or
                (res.target_entity and res.target_entity.lower() in ("fsm", "state_machine")) or
                target_prop_name.lower() in ("fsm", "state_machine", "states", "transitions", "transition", "condition", "fsm_transition", "state") or
                "fsm" in res.proposed_change.lower() or
                (conf and (getattr(conf, "category", "") == "FSM" or "fsm" in getattr(conf, "title", "").lower()))
            )
            if is_fsm_target:
                if not resolved_ir.fsms:
                    new_fsm = EFSFSM(name="Primary_FSM", initial_state="")
                    if resolved_ir.components:
                        new_fsm.owner = resolved_ir.components[0].component_id
                    resolved_ir.fsms.append(new_fsm)
                fsm = resolved_ir.fsms[0]

                # Parse transitions if formatted like 'STATE_A -> STATE_B on condition' or 'STATE_A -> STATE_B: condition'
                trans_matches = re.findall(r"([A-Za-z0-9_]+)\s*(?:->|-->|=>)\s*([A-Za-z0-9_]+)(?:\s*(?:on|when|if|:)\s*([^,;\n]+))?", val_str)
                if not trans_matches:
                    trans_matches = re.findall(r"([A-Za-z0-9_]+)\s*(?:->|-->|=>)\s*([A-Za-z0-9_]+)(?:\s*(?:on|when|if|:)\s*([^,;\n]+))?", res.proposed_change)

                if trans_matches:
                    for src, dst, cond in trans_matches:
                        src_clean = src.strip()
                        dst_clean = dst.strip()
                        cond_clean = cond.strip() if cond else "1'b1"
                        
                        # Ensure states exist in fsm.states
                        for s_name in (src_clean, dst_clean):
                            if not any(s.name.lower() == s_name.lower() for s in fsm.states):
                                fsm.states.append(EFSState(name=s_name))
                        
                        # Update or add transition
                        existing_tr = next((t for t in fsm.transitions if t.source_state.lower() == src_clean.lower() and t.target_state.lower() == dst_clean.lower()), None)
                        if existing_tr:
                            existing_tr.condition = cond_clean
                        else:
                            fsm.transitions.append(EFSTransition(source_state=src_clean, target_state=dst_clean, condition=cond_clean))
                    if not fsm.initial_state and fsm.states:
                        fsm.initial_state = fsm.states[0].name
                else:
                    # Check if val_str provides state names (comma-separated or 'states: ...')
                    states_match = re.search(r"(?:states\s*[:=]\s*)?([A-Za-z0-9_]+(?:\s*,\s*[A-Za-z0-9_]+)+)", val_str, re.IGNORECASE)
                    if states_match and target_prop_name.lower() in ("states", "fsm", "state_machine", "state"):
                        raw_states = [s.strip() for s in states_match.group(1).split(",") if s.strip()]
                        for st_name in raw_states:
                            if not any(s.name.lower() == st_name.lower() for s in fsm.states):
                                fsm.states.append(EFSState(name=st_name))
                        if not fsm.initial_state and fsm.states:
                            fsm.initial_state = fsm.states[0].name
                    else:
                        # Value provides a condition expression for an ambiguous/missing transition
                        cond_expr = val_str
                        # Update any transition with null/vague condition
                        vague_kws = ["control start", "operation completes", "acknowledged", "undefined", "unspecified", "tbd", "met"]
                        updated = False
                        for t in fsm.transitions:
                            t_cond = str(getattr(t, "condition", "") or "").lower()
                            if not t_cond or any(vk in t_cond for vk in vague_kws):
                                t.condition = cond_expr
                                updated = True
                                break
                        if not updated and fsm.transitions:
                            fsm.transitions[0].condition = cond_expr
                        elif not updated and not fsm.transitions and len(fsm.states) >= 2:
                            for i in range(len(fsm.states) - 1):
                                fsm.transitions.append(EFSTransition(
                                    source_state=fsm.states[i].name,
                                    target_state=fsm.states[i+1].name,
                                    condition=cond_expr if i == 0 else "1'b1"
                                ))
                        elif not updated and not fsm.states:
                            # Parse state names mentioned in condition or create standard named states from specification context
                            fsm.states.append(EFSState(name="IDLE"))
                            fsm.states.append(EFSState(name="RUN"))
                            fsm.states.append(EFSState(name="DONE"))
                            fsm.initial_state = "IDLE"
                            fsm.transitions.append(EFSTransition(source_state="IDLE", target_state="RUN", condition=cond_expr))
                            fsm.transitions.append(EFSTransition(source_state="RUN", target_state="DONE", condition="done == 1'b1"))
                            fsm.transitions.append(EFSTransition(source_state="DONE", target_state="IDLE", condition="1'b1"))

            # Apply to Missing Signals / ACKs
            if conf and conf.type in ("MISSING_SIGNAL", "MISSING_ACK") or "signal" in res.proposed_change.lower():
                sig_name = val_str.split("=")[-1].strip() if "=" in val_str else val_str
                if sig_name and not any(s.name.lower() == sig_name.lower() for s in resolved_ir.signals):
                    role = "control" if "ack" in sig_name.lower() else "data"
                    direction = "input" if "req" in sig_name.lower() else "output"
                    resolved_ir.signals.append(EFSSignal(
                        name=sig_name,
                        direction=direction,
                        width="1",
                        semantic_role=role,
                        description=f"Resolved signal for {conf.conflict_id if conf else 'requirement'}"
                    ))

        return resolved_ir

    def to_dict(self) -> Dict[str, Any]:
        # Clocks and resets reference signal_id
        clocks = [{"signal_id": s.signal_id} for s in self.clocks] if self.clocks else [{"signal_id": s.signal_id} for s in self.signals if s.semantic_role == "clock"]
        resets = [{"signal_id": s.signal_id} for s in self.resets] if self.resets else [{"signal_id": s.signal_id} for s in self.signals if s.semantic_role == "reset"]
        
        # Build first-class validation object
        if self.validation:
            if not self.validation.design_id:
                self.validation.design_id = self.metadata.design_id
            if not self.validation.canonical_id:
                self.validation.canonical_id = self.metadata.canonical_id
            if not self.validation.generation_run_id:
                self.validation.generation_run_id = self.metadata.generation_run_id
            if not self.validation.document_id:
                self.validation.document_id = self.metadata.document_id
            if not self.validation.ingestion_id:
                self.validation.ingestion_id = self.metadata.ingestion_id
            val_dict = self.validation.to_dict()
        else:
            val_dict = {
                "status": "CONFLICTS_FOUND" if any(c.blocking and c.resolution_status == "UNRESOLVED" for c in self.conflicts) else "PASSED",
                "design_id": self.metadata.design_id,
                "canonical_id": self.metadata.canonical_id,
                "generation_run_id": self.metadata.generation_run_id,
                "document_id": self.metadata.document_id,
                "ingestion_id": self.metadata.ingestion_id,
                "summary": {
                    "total": len(self.conflicts),
                    "critical": len([c for c in self.conflicts if c.severity in ("CRITICAL", "ERROR")]),
                    "warning": len([c for c in self.conflicts if c.severity not in ("CRITICAL", "ERROR")])
                },
                "findings": [c.to_dict() for c in self.conflicts]
            }

        return strip_nulls({
            "metadata": self.metadata.to_dict(),
            "components": [c.to_dict() for c in self.components],
            "interfaces": [i.to_dict() for i in self.interfaces],
            "signals": [s.to_dict() for s in self.signals],
            "clocks": clocks,
            "resets": resets,
            "parameters": [p.to_dict() for p in self.parameters],
            "registers": [r.to_dict() for r in self.registers],
            "instructions": [i.to_dict() for i in self.instructions],
            "instruction_formats": [f.to_dict() for f in self.instruction_formats],
            "opcodes": [o.to_dict() for o in self.opcodes],
            "fsms": [f.to_dict() for f in self.fsms],
            "transactions": [t.to_dict() for t in self.transactions],
            "functional_behavior": [b.to_dict() for b in self.functional_behavior] if self.functional_behavior else [f.to_dict() for f in self.flows],
            "memory_regions": [m.to_dict() for m in self.memory_regions],
            "memory": [m.to_dict() for m in self.memory_regions],
            "protocol_rules": [p.to_dict() for p in self.protocol_rules],
            "timing_rules": [t.to_dict() for t in self.timing_rules],
            "timing_requirements": [t.to_dict() for t in self.timing_rules],
            "performance_requirements": [p.to_dict() for p in self.performance_requirements],
            "errors": [e.to_dict() for e in self.errors],
            "constraints": [c.to_dict() for c in self.constraints],
            "requirements": [r.to_dict() for r in self.requirements],
            "validation": val_dict,
            "resolutions": [r.to_dict() for r in self.resolutions],
            "examples": [],
            "ambiguities": [],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "contradictions": [c.to_dict() for c in self.conflicts],
            "extraction_quality": self.extraction_quality.to_dict() if hasattr(self.extraction_quality, "to_dict") else self.extraction_quality,
            "extraction_coverage": self.extraction_coverage
        })

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSIR:
        # Support both 'validation.findings' and 'contradictions' or 'conflicts'
        conflicts_data = data.get("conflicts", [])
        if not conflicts_data and "validation" in data and isinstance(data["validation"], dict):
            conflicts_data = data["validation"].get("findings", [])
        if not conflicts_data:
            conflicts_data = data.get("contradictions", [])

        # Parse validation block if present
        val_obj = None
        if "validation" in data and isinstance(data["validation"], dict):
            val_obj = EFSValidationResult.from_dict(data["validation"])

        # Parse extraction quality if present
        eq_raw = data.get("extraction_quality", {})
        eq_obj = EFSExtractionQuality.from_dict(eq_raw) if isinstance(eq_raw, dict) else EFSExtractionQuality()

        signals_list = [EFSSignal.from_dict(s) for s in data.get("signals", [])]
        signal_lookup = {s.signal_id: s for s in signals_list}
        clocks_list = []
        for c in data.get("clocks", []):
            if isinstance(c, dict):
                sig_id = c.get("signal_id")
                if sig_id and sig_id in signal_lookup:
                    clocks_list.append(signal_lookup[sig_id])
                else:
                    clocks_list.append(EFSSignal.from_dict(c))
            elif isinstance(c, str):
                if c in signal_lookup:
                    clocks_list.append(signal_lookup[c])
                else:
                    clocks_list.append(EFSSignal(signal_id=c, name=c))
        
        resets_list = []
        for r in data.get("resets", []):
            if isinstance(r, dict):
                sig_id = r.get("signal_id")
                if sig_id and sig_id in signal_lookup:
                    resets_list.append(signal_lookup[sig_id])
                else:
                    resets_list.append(EFSSignal.from_dict(r))
            elif isinstance(r, str):
                if r in signal_lookup:
                    resets_list.append(signal_lookup[r])
                else:
                    resets_list.append(EFSSignal(signal_id=r, name=r))

        if not clocks_list:
            clocks_list = [s for s in signals_list if s.semantic_role == "clock"]
        if not resets_list:
            resets_list = [s for s in signals_list if s.semantic_role == "reset"]

        return cls(
            metadata=EFSMetadata.from_dict(data.get("metadata", {})),
            components=[EFSComponent.from_dict(c) for c in data.get("components", [])],
            interfaces=[EFSInterface.from_dict(i) for i in data.get("interfaces", [])],
            signals=signals_list,
            clocks=clocks_list,
            resets=resets_list,
            registers=[EFSRegister.from_dict(r) for r in data.get("registers", [])],
            transactions=[EFSTransaction.from_dict(t) for t in data.get("transactions", [])],
            fsms=[EFSFSM.from_dict(f) for f in data.get("fsms", [])],
            constraints=[EFSConstraint.from_dict(c) for c in data.get("constraints", [])],
            timing_rules=[EFSTimingRule.from_dict(t) for t in data.get("timing_rules", [])],
            protocol_rules=[EFSProtocolRule.from_dict(p) for p in data.get("protocol_rules", [])],
            flows=[EFSFlow.from_dict(f) for f in data.get("flows", [])],
            functional_behavior=[EFSFunctionalBehavior.from_dict(b) for b in data.get("functional_behavior", [])],
            assertions=[EFSAssertionIntent.from_dict(a) for a in data.get("assertions", [])],
            coverage=[EFSCoverageIntent.from_dict(c) for c in data.get("coverage", [])],
            parameters=[EFSParameter.from_dict(p) for p in data.get("parameters", [])],
            instructions=[EFSInstruction.from_dict(i) for i in data.get("instructions", [])],
            instruction_formats=[EFSInstructionFormat.from_dict(f) for f in data.get("instruction_formats", [])],
            opcodes=[EFSOpcode.from_dict(o) for o in data.get("opcodes", [])],
            memory_regions=[EFSMemoryRegion.from_dict(m) for m in (data.get("memory_regions") or data.get("memory", []))],
            data_structures=[EFSDataStructure.from_dict(d) for d in data.get("data_structures", [])],
            execution_models=[EFSExecutionModel.from_dict(e) for e in data.get("execution_models", [])],
            errors=[EFSError.from_dict(e) for e in data.get("errors", [])],
            performance_requirements=[EFSPerformanceRequirement.from_dict(p) for p in data.get("performance_requirements", [])],
            conflicts=[EFSConflict.from_dict(c) for c in conflicts_data],
            requirements=[EFSRequirement.from_dict(r) for r in data.get("requirements", [])],
            resolutions=[EFSResolution.from_dict(r) for r in data.get("resolutions", [])],
            extraction_quality=eq_obj,
            extraction_coverage=data.get("extraction_coverage", {}),
            validation=val_obj
        )


@dataclass
class InferredDecision:
    """Represents an architectural decision or default applied by the Controlled Inference Engine."""
    target_element: str
    element_type: str  # "interface", "register", "fsm", "reset", "datapath"
    decision_type: str  # "DERIVED" or "IMPLEMENTATION_CHOICE"
    description: str
    rationale: str
    evidence: str = "N/A"

    def to_dict(self) -> Dict[str, Any]:
        return strip_nulls(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InferredDecision:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CompletedDesignModel:
    """
    Unified Completed Design Model.
    
    The single canonical source of truth shared by both RTL Generator and PlantUML Generator.
    Contains the completed EFS IR, all inferred architectural decisions, and validation status.
    """
    efs_ir: EFSIR
    design_plan: Dict[str, Any] = field(default_factory=dict)
    inferred_decisions: List[InferredDecision] = field(default_factory=list)
    validation_status: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "efs_ir": self.efs_ir.to_dict(),
            "design_plan": strip_nulls(self.design_plan),
            "inferred_decisions": [d.to_dict() for d in self.inferred_decisions],
            "validation_status": strip_nulls(self.validation_status)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CompletedDesignModel:
        return cls(
            efs_ir=EFSIR.from_dict(data.get("efs_ir", {})),
            design_plan=data.get("design_plan", {}),
            inferred_decisions=[InferredDecision.from_dict(d) for d in data.get("inferred_decisions", [])],
            validation_status=data.get("validation_status", {})
        )
