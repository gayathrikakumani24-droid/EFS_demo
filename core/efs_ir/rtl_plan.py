"""
Structured RTL Generation Plan Data Models and Validation Schema.

Defines typed, structured hardware implementation plans consumed by
downstream RTL generation and verified by the RTL Plan Validator.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from core.efs_ir.models import strip_nulls


class EvidenceClassification(str, Enum):
    """Provenance and inference classification for plan elements."""
    EXPLICIT = "explicit"
    DERIVED = "derived"
    INFERRED = "inferred"


class PlanStatus(str, Enum):
    """Specification sufficiency status."""
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    CONTRADICTORY = "CONTRADICTORY"


@dataclass
class PlanPort:
    """Hardware module port definition."""
    name: str
    direction: str  # input, output, inout
    width: str = "1"
    signedness: bool = False
    semantic_role: str = "data"  # clock, reset, data, control, status, handshake
    clock: Optional[str] = None
    reset: Optional[str] = None
    description: str = ""
    source_ref: str = ""
    efs_id: Optional[str] = None
    classification: EvidenceClassification = EvidenceClassification.EXPLICIT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "width": str(self.width),
            "signedness": self.signedness,
            "semantic_role": self.semantic_role,
            "clock": self.clock,
            "reset": self.reset,
            "description": self.description,
            "source_ref": self.source_ref,
            "efs_id": self.efs_id,
            "classification": self.classification.value if isinstance(self.classification, EvidenceClassification) else str(self.classification),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanPort:
        c_val = data.get("classification", "explicit")
        try:
            c_enum = EvidenceClassification(c_val)
        except Exception:
            c_enum = EvidenceClassification.EXPLICIT
        return cls(
            name=data.get("name", "unnamed_port"),
            direction=data.get("direction", "input"),
            width=str(data.get("width", "1")),
            signedness=bool(data.get("signedness", False)),
            semantic_role=data.get("semantic_role", "data"),
            clock=data.get("clock"),
            reset=data.get("reset"),
            description=data.get("description", ""),
            source_ref=data.get("source_ref", data.get("source_evidence", "")),
            efs_id=data.get("efs_id"),
            classification=c_enum,
        )


@dataclass
class PlanParameter:
    """Hardware parameter / generic definition."""
    name: str
    value: Any = "0"
    type: str = "integer"
    description: str = ""
    usage: str = ""
    is_behaviorally_defined: bool = True
    source_ref: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": str(self.value),
            "type": self.type,
            "description": self.description,
            "usage": self.usage,
            "is_behaviorally_defined": self.is_behaviorally_defined,
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanParameter:
        return cls(
            name=data.get("name", "PARAM"),
            value=data.get("value", data.get("default", "0")),
            type=data.get("type", "integer"),
            description=data.get("description", ""),
            usage=data.get("usage", ""),
            is_behaviorally_defined=bool(data.get("is_behaviorally_defined", True)),
            source_ref=data.get("source_ref", ""),
        )


@dataclass
class PlanInternalComponent:
    """Internal submodule / component instantiation."""
    name: str
    component_type: str
    purpose: str = ""
    connections: Dict[str, str] = field(default_factory=dict)
    source_ref: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "component_type": self.component_type,
            "purpose": self.purpose,
            "connections": self.connections,
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanInternalComponent:
        return cls(
            name=data.get("name", "inst"),
            component_type=data.get("component_type", data.get("type", "module")),
            purpose=data.get("purpose", ""),
            connections=data.get("connections", {}),
            source_ref=data.get("source_ref", ""),
        )


@dataclass
class PlanInternalSignal:
    """Internal wire / combinational net."""
    name: str
    width: str = "1"
    signedness: bool = False
    driving_type: str = "wire"  # wire, logic, reg
    driver: str = ""
    source_ref: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "width": str(self.width),
            "signedness": self.signedness,
            "driving_type": self.driving_type,
            "driver": self.driver,
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanInternalSignal:
        return cls(
            name=data.get("name", "sig"),
            width=str(data.get("width", "1")),
            signedness=bool(data.get("signedness", False)),
            driving_type=data.get("driving_type", "wire"),
            driver=data.get("driver", ""),
            source_ref=data.get("source_ref", ""),
        )


@dataclass
class PlanRegisterField:
    """Bitfield within a hardware register."""
    name: str
    bits: str
    meaning: str = ""
    reset_value: str = "0"
    source: str = "EFS"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "bits": self.bits,
            "meaning": self.meaning,
            "reset_value": self.reset_value,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanRegisterField:
        return cls(
            name=data.get("name", "field"),
            bits=str(data.get("bits", "0")),
            meaning=data.get("meaning", data.get("description", "")),
            reset_value=str(data.get("reset_value", "0")),
            source=data.get("source", "EFS"),
        )


@dataclass
class PlanRegister:
    """Internal sequential register or register bank entry."""
    name: str
    width: str = "32"
    reset_val: str = "0"
    reset_polarity: str = "active_low"
    update_condition: str = "posedge clk"
    purpose: str = ""
    fields: List[PlanRegisterField] = field(default_factory=list)
    source_ref: str = ""
    efs_id: Optional[str] = None
    offset: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "width": str(self.width),
            "reset_val": str(self.reset_val),
            "reset_polarity": self.reset_polarity,
            "update_condition": self.update_condition,
            "purpose": self.purpose,
            "offset": self.offset,
            "fields": [f.to_dict() for f in self.fields],
            "source_ref": self.source_ref,
            "efs_id": self.efs_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanRegister:
        r_reset = data.get("reset_val", data.get("reset_value", "0"))
        flds = [
            PlanRegisterField.from_dict(f) if isinstance(f, dict) else f
            for f in data.get("fields", [])
        ]
        return cls(
            name=data.get("name", "reg"),
            width=str(data.get("width", "32")),
            reset_val=str(r_reset if r_reset is not None else "0"),
            reset_polarity=data.get("reset_polarity", "active_low"),
            update_condition=data.get("update_condition", "posedge clk"),
            purpose=data.get("purpose", data.get("description", "")),
            offset=data.get("offset"),
            fields=flds,
            source_ref=data.get("source_ref", ""),
            efs_id=data.get("efs_id"),
        )


@dataclass
class PlanResetRequirements:
    """Hardware reset domain and polarity specifications."""
    reset_polarity: str = "active_low"  # active_low, active_high, none
    reset_type: str = "async"  # async, sync, none
    reset_signal: Optional[str] = None
    target_registers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reset_polarity": self.reset_polarity,
            "reset_type": self.reset_type,
            "reset_signal": self.reset_signal,
            "target_registers": self.target_registers,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanResetRequirements:
        return cls(
            reset_polarity=data.get("reset_polarity", "active_low"),
            reset_type=data.get("reset_type", "async"),
            reset_signal=data.get("reset_signal", data.get("reset_name")),
            target_registers=data.get("target_registers", []),
        )


@dataclass
class PlanOpcode:
    """Hardware instruction or command opcode definition."""
    name: str
    binary_encoding: str
    associated_operations: List[str] = field(default_factory=list)
    evidence: str = ""
    efs_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "binary_encoding": self.binary_encoding,
            "associated_operations": self.associated_operations,
            "evidence": self.evidence,
            "efs_id": self.efs_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanOpcode:
        return cls(
            name=data.get("name", data.get("mnemonic", "OPC")),
            binary_encoding=str(data.get("binary_encoding", data.get("encoding", ""))),
            associated_operations=data.get("associated_operations", data.get("operations", [])),
            evidence=data.get("evidence", data.get("source_evidence", "")),
            efs_id=data.get("efs_id"),
        )


@dataclass
class PlanFSMTransition:
    """FSM state transition with explicit condition and classification."""
    from_state: str
    to_state: str
    condition_expression: str = "1'b1"
    signals: List[str] = field(default_factory=list)
    registers: List[str] = field(default_factory=list)
    is_fully_defined: bool = True
    state_actions: List[str] = field(default_factory=list)
    classification: EvidenceClassification = EvidenceClassification.EXPLICIT
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_state": self.from_state,
            "to_state": self.to_state,
            "condition": {
                "expression": self.condition_expression,
                "signals": self.signals,
                "registers": self.registers,
                "is_fully_defined": self.is_fully_defined,
            },
            "state_actions": self.state_actions,
            "classification": self.classification.value if isinstance(self.classification, EvidenceClassification) else str(self.classification),
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanFSMTransition:
        c_obj = data.get("condition", {})
        if isinstance(c_obj, dict):
            expr = c_obj.get("expression", "1'b1")
            sigs = c_obj.get("signals", [])
            regs = c_obj.get("registers", [])
            fully_def = bool(c_obj.get("is_fully_defined", True))
        else:
            expr = str(c_obj or "1'b1")
            sigs = []
            regs = []
            fully_def = bool(expr and expr.strip() not in ("", "None"))

        c_val = data.get("classification", "explicit")
        try:
            c_enum = EvidenceClassification(c_val)
        except Exception:
            c_enum = EvidenceClassification.EXPLICIT

        actions = data.get("state_actions", data.get("actions", []))
        if isinstance(actions, str):
            actions = [actions]

        return cls(
            from_state=data.get("from_state", data.get("from", "")),
            to_state=data.get("to_state", data.get("to", "")),
            condition_expression=str(expr if expr is not None else ""),
            signals=sigs,
            registers=regs,
            is_fully_defined=fully_def,
            state_actions=actions,
            classification=c_enum,
            evidence=data.get("evidence", data.get("source_evidence", "")),
        )


@dataclass
class PlanFSM:
    """Finite State Machine architectural definition."""
    states: List[str] = field(default_factory=list)
    initial_state: str = "IDLE"
    transitions: List[PlanFSMTransition] = field(default_factory=list)
    fsm_required: bool = True
    fsm_type: str = "explicit"  # explicit, inferred, none
    state_encodings: Dict[str, int] = field(default_factory=dict)
    classification: EvidenceClassification = EvidenceClassification.EXPLICIT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "states": self.states,
            "initial_state": self.initial_state,
            "fsm_required": self.fsm_required,
            "fsm_type": self.fsm_type,
            "state_encodings": self.state_encodings,
            "classification": self.classification.value if isinstance(self.classification, EvidenceClassification) else str(self.classification),
            "transitions": [t.to_dict() for t in self.transitions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanFSM:
        states = data.get("states", [])
        clean_states = [s.get("name") if isinstance(s, dict) else str(s) for s in states]
        trans_raw = data.get("transitions", [])
        transitions = [
            PlanFSMTransition.from_dict(t) if isinstance(t, dict) else t
            for t in trans_raw
        ]
        c_val = data.get("classification", "explicit")
        try:
            c_enum = EvidenceClassification(c_val)
        except Exception:
            c_enum = EvidenceClassification.EXPLICIT

        return cls(
            states=clean_states,
            initial_state=data.get("initial_state", clean_states[0] if clean_states else "IDLE"),
            transitions=transitions,
            fsm_required=bool(data.get("fsm_required", True)),
            fsm_type=data.get("fsm_type", "explicit"),
            state_encodings=data.get("state_encodings", {}),
            classification=c_enum,
        )


@dataclass
class PlanArithmeticOp:
    """Arithmetic operation in the datapath."""
    name: str
    expression: str
    operands: List[str] = field(default_factory=list)
    destination: str = ""
    condition: Optional[str] = None
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "expression": self.expression,
            "operands": self.operands,
            "destination": self.destination,
            "condition": self.condition,
            "evidence": self.evidence,
        }


@dataclass
class PlanLogicalOp:
    """Bitwise or boolean logical operation in the datapath."""
    name: str
    expression: str
    operands: List[str] = field(default_factory=list)
    destination: str = ""
    condition: Optional[str] = None
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "expression": self.expression,
            "operands": self.operands,
            "destination": self.destination,
            "condition": self.condition,
            "evidence": self.evidence,
        }


@dataclass
class PlanDataSelection:
    """Multiplexer or conditional data selection."""
    name: str
    selector_signal: str
    cases: Dict[str, str] = field(default_factory=dict)
    default_assignment: str = ""
    destination: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "selector_signal": self.selector_signal,
            "cases": self.cases,
            "default_assignment": self.default_assignment,
            "destination": self.destination,
        }


@dataclass
class PlanRegisterTransfer:
    """Explicit register transfer on active clock edge."""
    register_name: str
    next_value_expression: str
    condition: str = "1'b1"
    clock_domain: str = "clk"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "register_name": self.register_name,
            "next_value_expression": self.next_value_expression,
            "condition": self.condition,
            "clock_domain": self.clock_domain,
        }


@dataclass
class PlanDatapath:
    """Hardware datapath processing definitions."""
    arithmetic_operations: List[PlanArithmeticOp] = field(default_factory=list)
    logical_operations: List[PlanLogicalOp] = field(default_factory=list)
    data_selection: List[PlanDataSelection] = field(default_factory=list)
    intermediate_values: List[str] = field(default_factory=list)
    register_transfers: List[PlanRegisterTransfer] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arithmetic_operations": [op.to_dict() for op in self.arithmetic_operations],
            "logical_operations": [op.to_dict() for op in self.logical_operations],
            "data_selection": [ds.to_dict() for ds in self.data_selection],
            "intermediate_values": self.intermediate_values,
            "register_transfers": [rt.to_dict() for rt in self.register_transfers],
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanDatapath:
        if not isinstance(data, dict):
            return cls(description=str(data))
        ops = data.get("operations", [])
        arith_ops = []
        for op in ops:
            if isinstance(op, dict):
                arith_ops.append(PlanArithmeticOp(
                    name=op.get("name", "op"),
                    expression=op.get("expression", ""),
                    operands=op.get("operands", []),
                    destination=op.get("destination", ""),
                    condition=op.get("condition"),
                    evidence=op.get("evidence", ""),
                ))
        return cls(
            arithmetic_operations=arith_ops,
            intermediate_values=data.get("intermediate_values", []),
            description=data.get("description", data.get("datapath_strategy", "")),
        )


@dataclass
class PlanBehavioralOp:
    """High-level behavioral implementation rule."""
    operation_name: str
    semantics: str
    execution_conditions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    evidence: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_name": self.operation_name,
            "semantics": self.semantics,
            "execution_conditions": self.execution_conditions,
            "dependencies": self.dependencies,
            "expected_outputs": self.expected_outputs,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanBehavioralOp:
        return cls(
            operation_name=data.get("operation_name", "op"),
            semantics=data.get("semantics", ""),
            execution_conditions=data.get("execution_conditions", []),
            dependencies=data.get("dependencies", []),
            expected_outputs=data.get("expected_outputs", []),
            evidence=data.get("evidence", ""),
        )


@dataclass
class PlanTraceability:
    """Traceability metadata linking the plan to EFS IR and Source Evidence."""
    efs_object_ids: List[str] = field(default_factory=list)
    source_evidence_ids: List[str] = field(default_factory=list)
    context_pack_refs: List[str] = field(default_factory=list)
    confidence: float = 1.0
    evidence_classification: EvidenceClassification = EvidenceClassification.EXPLICIT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "efs_object_ids": self.efs_object_ids,
            "source_evidence_ids": self.source_evidence_ids,
            "context_pack_refs": self.context_pack_refs,
            "confidence": self.confidence,
            "evidence_classification": self.evidence_classification.value if isinstance(self.evidence_classification, EvidenceClassification) else str(self.evidence_classification),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanTraceability:
        c_val = data.get("evidence_classification", "explicit")
        try:
            c_enum = EvidenceClassification(c_val)
        except Exception:
            c_enum = EvidenceClassification.EXPLICIT
        return cls(
            efs_object_ids=data.get("efs_object_ids", []),
            source_evidence_ids=data.get("source_evidence_ids", []),
            context_pack_refs=data.get("context_pack_refs", []),
            confidence=float(data.get("confidence", 1.0)),
            evidence_classification=c_enum,
        )


@dataclass
class PlanBlockingIssue:
    """Structured issue identifying an incomplete or contradictory requirement."""
    category: str  # FSM, INTERFACE, REGISTER, DATAPATH, PROTOCOL, RESET, TIMING
    issue: str
    required_information: str
    evidence: str
    affected_object: Optional[str] = None
    severity: str = "CRITICAL"
    hitl_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "issue": self.issue,
            "required_information": self.required_information,
            "evidence": self.evidence,
            "affected_object": self.affected_object,
            "severity": self.severity,
            "hitl_required": self.hitl_required,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanBlockingIssue:
        return cls(
            category=data.get("category", "GENERAL"),
            issue=data.get("issue", ""),
            required_information=data.get("required_information", ""),
            evidence=data.get("evidence", ""),
            affected_object=data.get("affected_object"),
            severity=data.get("severity", "CRITICAL"),
            hitl_required=bool(data.get("hitl_required", True)),
        )


@dataclass
class PlanSpecificationStatus:
    """Status indicating whether specification was complete, incomplete, or contradictory."""
    status: PlanStatus = PlanStatus.COMPLETE
    blocking_issues: List[PlanBlockingIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value if isinstance(self.status, PlanStatus) else str(self.status),
            "blocking_issues": [b.to_dict() for b in self.blocking_issues],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanSpecificationStatus:
        st = data.get("status", "COMPLETE")
        try:
            st_enum = PlanStatus(st)
        except Exception:
            st_enum = PlanStatus.COMPLETE
        issues = [
            PlanBlockingIssue.from_dict(b) if isinstance(b, dict) else b
            for b in data.get("blocking_issues", [])
        ]
        return cls(status=st_enum, blocking_issues=issues)


@dataclass
class RTLGenerationPlan:
    """
    Authoritative Typed RTL Generation Plan.
    Represents the full architectural plan that subsequent generation stages consume.
    """
    module_name: str
    target_language: str = "Verilog"
    architecture_class: str = "CUSTOM_ARCHITECTURE"
    specification_status: PlanSpecificationStatus = field(default_factory=PlanSpecificationStatus)
    parameters: List[PlanParameter] = field(default_factory=list)
    ports: List[PlanPort] = field(default_factory=list)
    internal_components: List[PlanInternalComponent] = field(default_factory=list)
    internal_signals: List[PlanInternalSignal] = field(default_factory=list)
    internal_registers: List[PlanRegister] = field(default_factory=list)
    reset_requirements: PlanResetRequirements = field(default_factory=PlanResetRequirements)
    opcodes: List[PlanOpcode] = field(default_factory=list)
    fsm: Optional[PlanFSM] = None
    datapath: PlanDatapath = field(default_factory=PlanDatapath)
    behavioral_implementation: List[PlanBehavioralOp] = field(default_factory=list)
    traceability: PlanTraceability = field(default_factory=PlanTraceability)
    output_driving_strategy: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to clean JSON-serializable dictionary."""
        d = {
            "module_name": self.module_name,
            "target_language": self.target_language,
            "architecture_class": self.architecture_class,
            "specification_status": self.specification_status.to_dict(),
            "parameters": [p.to_dict() for p in self.parameters],
            "ports": [p.to_dict() for p in self.ports],
            "internal_components": [c.to_dict() for c in self.internal_components],
            "internal_signals": [s.to_dict() for s in self.internal_signals],
            "internal_registers": [r.to_dict() for r in self.internal_registers],
            "reset_requirements": self.reset_requirements.to_dict(),
            "opcodes": [o.to_dict() for o in self.opcodes],
            "fsm": self.fsm.to_dict() if self.fsm else None,
            "datapath": self.datapath.to_dict(),
            "behavioral_implementation": [b.to_dict() for b in self.behavioral_implementation],
            "traceability": self.traceability.to_dict(),
            "output_driving_strategy": self.output_driving_strategy,
            # Legacy compatibility fields
            "clock_reset": {
                "clock_name": next((p.name for p in self.ports if p.semantic_role == "clock" or "clk" in p.name.lower()), "clk"),
                "reset_name": self.reset_requirements.reset_signal or next((p.name for p in self.ports if p.semantic_role == "reset" or "rst" in p.name.lower()), "rst_n"),
                "reset_polarity": "low" if self.reset_requirements.reset_polarity == "active_low" else "high",
            } if any(p.semantic_role in ("clock", "reset") or any(k in p.name.lower() for k in ("clk", "rst")) for p in self.ports) else None
        }
        return strip_nulls(d)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RTLGenerationPlan:
        """Construct from raw dictionary or parsed JSON."""
        # Ports
        ports = [
            PlanPort.from_dict(p) if isinstance(p, dict) else p
            for p in data.get("ports", [])
        ]
        # Parameters
        parameters = [
            PlanParameter.from_dict(p) if isinstance(p, dict) else p
            for p in data.get("parameters", [])
        ]
        # Internal components
        components = [
            PlanInternalComponent.from_dict(c) if isinstance(c, dict) else c
            for c in data.get("internal_components", [])
        ]
        # Internal signals
        signals = [
            PlanInternalSignal.from_dict(s) if isinstance(s, dict) else s
            for s in data.get("internal_signals", data.get("signals", []))
        ]
        # Registers
        regs = [
            PlanRegister.from_dict(r) if isinstance(r, dict) else r
            for r in data.get("internal_registers", data.get("registers", []))
        ]
        # Reset requirements
        reset_req = (
            PlanResetRequirements.from_dict(data["reset_requirements"])
            if isinstance(data.get("reset_requirements"), dict)
            else (
                PlanResetRequirements.from_dict(data["clock_reset"])
                if isinstance(data.get("clock_reset"), dict)
                else PlanResetRequirements()
            )
        )
        # Opcodes
        opcodes = [
            PlanOpcode.from_dict(o) if isinstance(o, dict) else o
            for o in data.get("opcodes", [])
        ]
        # FSM
        fsm_obj = None
        if data.get("fsm") and isinstance(data["fsm"], dict):
            fsm_obj = PlanFSM.from_dict(data["fsm"])
        # Datapath
        datapath_obj = (
            PlanDatapath.from_dict(data["datapath"])
            if isinstance(data.get("datapath"), dict)
            else PlanDatapath(description=str(data.get("datapath_strategy", "")))
        )
        # Behavioral implementation
        behaviors = [
            PlanBehavioralOp.from_dict(b) if isinstance(b, dict) else b
            for b in data.get("behavioral_implementation", [])
        ]
        # Traceability
        traceability_obj = (
            PlanTraceability.from_dict(data["traceability"])
            if isinstance(data.get("traceability"), dict)
            else PlanTraceability()
        )
        # Status
        spec_status = (
            PlanSpecificationStatus.from_dict(data["specification_status"])
            if isinstance(data.get("specification_status"), dict)
            else PlanSpecificationStatus()
        )

        return cls(
            module_name=data.get("module_name", "hardware_module"),
            target_language=data.get("target_language", "Verilog"),
            architecture_class=data.get("architecture_class", "CUSTOM_ARCHITECTURE"),
            specification_status=spec_status,
            parameters=parameters,
            ports=ports,
            internal_components=components,
            internal_signals=signals,
            internal_registers=regs,
            reset_requirements=reset_req,
            opcodes=opcodes,
            fsm=fsm_obj,
            datapath=datapath_obj,
            behavioral_implementation=behaviors,
            traceability=traceability_obj,
            output_driving_strategy=data.get("output_driving_strategy", {}),
        )

