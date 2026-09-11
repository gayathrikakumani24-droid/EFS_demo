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
from typing import Any, Dict, List, Optional


def strip_nulls(obj: Any) -> Any:
    """Recursively remove None/null values from dicts and lists to minimize JSON payload size."""
    if isinstance(obj, dict):
        return {
            k: strip_nulls(v)
            for k, v in obj.items()
            if v is not None
        }
    elif isinstance(obj, list):
        return [strip_nulls(v) for v in obj if v is not None]
    return obj


def new_efs_id(prefix: str = "efs") -> str:
    """Generate a short unique identifier for EFS objects."""
    return f"{prefix.upper()}_{uuid.uuid4().hex[:8]}"


@dataclass
class EFSMetadata:
    """Metadata for the EFS IR instance."""
    design_id: str = field(default_factory=lambda: new_efs_id("dsn"))
    design_name: str = "unnamed_design"
    canonical_id: str = ""
    display_name: str = ""
    aliases: List[str] = field(default_factory=list)
    project_name: str = "default_project"
    protocol: str = "generic"
    protocol_version: str = "1.0"
    document_id: str = ""
    document_hash: str = ""
    ingestion_id: str = ""
    source_documents: List[str] = field(default_factory=list)
    specification_revision: str = "1.0"
    ir_schema_version: str = "1.0"
    generation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return strip_nulls(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSMetadata:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EFSTraceability:
    """Traceability mapping an EFS IR object back to its source specification origin."""
    doc_id: Optional[str] = None
    document_hash: Optional[str] = None
    ingestion_id: Optional[str] = None
    chunk_id: Optional[str] = None
    block_id: Optional[str] = None
    page: Optional[int] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    original_text: Optional[str] = None
    paragraph: Optional[str] = None
    source_entity_id: Optional[str] = None
    source_relationship_id: Optional[str] = None
    extraction_method: str = "unknown"  # e.g., "llm", "rule_fallback", "user_override"
    confidence: float = 1.0            # 0.0 to 1.0 confidence score

    def to_dict(self) -> Dict[str, Any]:
        return strip_nulls(asdict(self))

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
    address: str = ""                  # absolute hex address if known, e.g. "0x4000_0000"
    offset: str = "0x0"                # relative offset, e.g. "0x0C"
    width: int = 32
    access_type: str = "RW"            # RW | RO | WO
    access: str = "RW"
    reset_value: str = "0"
    fields: List[EFSRegisterField] = field(default_factory=list)
    description: str = ""
    side_effects: str = ""
    owner: Optional[str] = None        # component_id
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    def __post_init__(self):
        if self.access != "RW" and self.access_type == "RW":
            self.access_type = self.access
        elif self.access_type != "RW" and self.access == "RW":
            self.access = self.access_type

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
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
    error_behavior: str = ""
    timing: str = ""
    ordering: str = ""                 # ordering rules (e.g. out-of-order, in-order)
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

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
class EFSResolution:
    """An approved or proposed resolution overriding an EFSConflict without mutating original source requirements."""
    resolution_id: str = field(default_factory=lambda: new_efs_id("res"))
    conflict_id: str = ""
    resolution_type: str = "ACCEPT_PROPOSAL"  # ACCEPT_PROPOSAL | CUSTOM_OVERRIDE | REJECT
    proposed_change: str = ""
    rationale: str = ""
    affected_requirements: List[str] = field(default_factory=list)
    evidence: str = ""
    source_values: List[str] = field(default_factory=list)
    proposed_value: Any = None
    approval_status: str = "UNRESOLVED"        # UNRESOLVED | PROPOSED | APPROVED | REJECTED
    approved_by: str = "user"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = strip_nulls(asdict(self))
        # Add standard schema compatibility aliases
        d["issue_id"] = self.conflict_id
        d["issue_type"] = self.resolution_type
        d["original_evidence"] = self.evidence
        d["resolution"] = self.proposed_change
        d["source"] = self.approved_by
        d["approved_by_user"] = (self.approval_status == "APPROVED")
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
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EFSConflict:
    """A conflict or ambiguity detected in the specifications."""
    conflict_id: str = field(default_factory=lambda: new_efs_id("conf"))
    type: str = ""                      # e.g., DUPLICATE_OPCODE, RESET_POLARITY, SIGNAL_WIDTH
    severity: str = "MAJOR"             # CRITICAL | MAJOR | MINOR | WARNING
    blocking: bool = True
    status: str = "UNRESOLVED"          # UNRESOLVED | PROPOSED | APPROVED | REJECTED
    entities: List[str] = field(default_factory=list)
    property: str = ""
    conflicting_values: List[str] = field(default_factory=list)
    source_objects: List[str] = field(default_factory=list)  # IDs of conflicting signals/components/opcodes
    evidence: str = ""
    description: str = ""
    explanation: str = ""
    impact: str = ""
    suggestions: List[str] = field(default_factory=list)
    resolution: Optional[EFSResolution] = None
    candidates: List[str] = field(default_factory=list)
    resolution_status: str = "UNRESOLVED"  # UNRESOLVED | RESOLVED
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.resolution:
            d["resolution"] = self.resolution.to_dict()
        d["traceability"] = self.traceability.to_dict()
        return strip_nulls(d)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSConflict:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k not in ("traceability", "resolution")}
        if data.get("resolution"):
            kwargs["resolution"] = EFSResolution.from_dict(data["resolution"])
        kwargs["traceability"] = EFSTraceability.from_dict(data.get("traceability"))
        return cls(**kwargs)


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
    """An explicit hardware requirement extracted from specifications."""
    req_id: str = field(default_factory=lambda: new_efs_id("req"))
    title: str = ""
    category: str = "SPECIFIED"          # SPECIFIED | DERIVED | INFERRED | UNKNOWN | CONFLICTING
    requirement_type: str = "functional" # functional | interface | timing | register | reset | protocol
    target_object: str = ""
    description: str = ""
    is_critical: bool = True
    traceability: EFSTraceability = field(default_factory=EFSTraceability)

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
    registers: List[EFSRegister] = field(default_factory=list)
    transactions: List[EFSTransaction] = field(default_factory=list)
    fsms: List[EFSFSM] = field(default_factory=list)
    constraints: List[EFSConstraint] = field(default_factory=list)
    timing_rules: List[EFSTimingRule] = field(default_factory=list)
    protocol_rules: List[EFSProtocolRule] = field(default_factory=list)
    flows: List[EFSFlow] = field(default_factory=list)
    assertions: List[EFSAssertionIntent] = field(default_factory=list)
    coverage: List[EFSCoverageIntent] = field(default_factory=list)
    # Extended Canonical Models
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

    def get_resolved_view(self) -> EFSIR:
        """
        Builds and returns a resolved view overlaying approved resolutions onto the base EFSIR
        without mutating the immutable original source requirements.
        """
        import copy
        resolved_ir = copy.deepcopy(self)
        approved_res = [r for r in resolved_ir.resolutions if r.approval_status == "APPROVED"]
        
        for res in approved_res:
            target_prop = res.proposed_change.split("=")[0].strip() if "=" in res.proposed_change else res.proposed_change
            if "." in target_prop:
                target_entity_from_prop = target_prop.split(".")[0].strip()
                target_prop_name = target_prop.split(".")[-1].strip()
            else:
                target_entity_from_prop = None
                target_prop_name = target_prop

            val = res.proposed_value
            
            # Match conflict and sync resolution status
            conf = next((c for c in resolved_ir.conflicts if c.conflict_id == res.conflict_id), None)
            if not conf:
                conf = next((c for c in resolved_ir.conflicts if any(e in c.entities for e in res.affected_requirements) or c.property == target_prop_name), None)
            if conf:
                conf.status = "APPROVED"
                conf.resolution_status = "RESOLVED"
                conf.resolution = res

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

        return resolved_ir

    def to_dict(self) -> Dict[str, Any]:
        clocks = [s.to_dict() for s in self.signals if s.semantic_role == "clock"]
        resets = [s.to_dict() for s in self.signals if s.semantic_role == "reset"]
        return strip_nulls({
            "metadata": self.metadata.to_dict(),
            "components": [c.to_dict() for c in self.components],
            "interfaces": [i.to_dict() for i in self.interfaces],
            "signals": [s.to_dict() for s in self.signals],
            "clocks": clocks,
            "resets": resets,
            "parameters": [],
            "registers": [r.to_dict() for r in self.registers],
            "instructions": [i.to_dict() for i in self.instructions],
            "instruction_formats": [f.to_dict() for f in self.instruction_formats],
            "opcodes": [o.to_dict() for o in self.opcodes],
            "fsms": [f.to_dict() for f in self.fsms],
            "transactions": [t.to_dict() for t in self.transactions],
            "functional_behavior": [f.to_dict() for f in self.flows],
            "memory": [m.to_dict() for m in self.memory_regions],
            "protocol_rules": [p.to_dict() for p in self.protocol_rules],
            "timing_rules": [t.to_dict() for t in self.timing_rules],
            "timing_requirements": [t.to_dict() for t in self.timing_rules],
            "performance_requirements": [p.to_dict() for p in self.performance_requirements],
            "errors": [e.to_dict() for e in self.errors],
            "constraints": [c.to_dict() for c in self.constraints],
            "requirements": [r.to_dict() for r in self.requirements],
            "resolutions": [r.to_dict() for r in self.resolutions],
            "examples": [],
            "ambiguities": [],
            "contradictions": [c.to_dict() for c in self.conflicts],
            "extraction_quality": {
                "source_completeness": "complete",
                "ir_completeness": "complete",
                "extraction_status": "success",
                "missing_information": []
            }
        })

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EFSIR:
        return cls(
            metadata=EFSMetadata.from_dict(data.get("metadata", {})),
            components=[EFSComponent.from_dict(c) for c in data.get("components", [])],
            interfaces=[EFSInterface.from_dict(i) for i in data.get("interfaces", [])],
            signals=[EFSSignal.from_dict(s) for s in data.get("signals", [])],
            registers=[EFSRegister.from_dict(r) for r in data.get("registers", [])],
            transactions=[EFSTransaction.from_dict(t) for t in data.get("transactions", [])],
            fsms=[EFSFSM.from_dict(f) for f in data.get("fsms", [])],
            constraints=[EFSConstraint.from_dict(c) for c in data.get("constraints", [])],
            timing_rules=[EFSTimingRule.from_dict(t) for t in data.get("timing_rules", [])],
            protocol_rules=[EFSProtocolRule.from_dict(p) for p in data.get("protocol_rules", [])],
            flows=[EFSFlow.from_dict(f) for f in data.get("flows", [])],
            assertions=[EFSAssertionIntent.from_dict(a) for a in data.get("assertions", [])],
            coverage=[EFSCoverageIntent.from_dict(c) for c in data.get("coverage", [])],
            instructions=[EFSInstruction.from_dict(i) for i in data.get("instructions", [])],
            instruction_formats=[EFSInstructionFormat.from_dict(f) for f in data.get("instruction_formats", [])],
            opcodes=[EFSOpcode.from_dict(o) for o in data.get("opcodes", [])],
            memory_regions=[EFSMemoryRegion.from_dict(m) for m in data.get("memory_regions", [])],
            data_structures=[EFSDataStructure.from_dict(d) for d in data.get("data_structures", [])],
            execution_models=[EFSExecutionModel.from_dict(e) for e in data.get("execution_models", [])],
            errors=[EFSError.from_dict(e) for e in data.get("errors", [])],
            performance_requirements=[EFSPerformanceRequirement.from_dict(p) for p in data.get("performance_requirements", [])],
            conflicts=[EFSConflict.from_dict(c) for c in data.get("conflicts", [])],
            requirements=[EFSRequirement.from_dict(r) for r in data.get("requirements", [])],
            resolutions=[EFSResolution.from_dict(r) for r in data.get("resolutions", [])],
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
