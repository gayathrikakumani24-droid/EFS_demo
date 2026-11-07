"""
Structured RTL Generation Plan Data Models and Validation Schema.

Defines typed, structured hardware implementation plans consumed by
downstream RTL generation and verified by the RTL Plan Validator.
"""

from __future__ import annotations

import json
import re
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


class PlanOrigin(str, Enum):
    """Origin classification for plan elements distinguishing requirements from implementation decisions."""
    INHERITED = "INHERITED"
    DERIVED = "DERIVED"
    IMPLEMENTATION_DECISION = "IMPLEMENTATION_DECISION"


@dataclass
class DerivationMetadata:
    """Metadata describing how a plan element was derived from specification entities."""
    description: str = ""
    input_entities: List[str] = field(default_factory=list)
    rule: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "input_entities": self.input_entities,
            "rule": self.rule,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DerivationMetadata:
        if not isinstance(data, dict):
            return cls(description=str(data))
        return cls(
            description=data.get("description", ""),
            input_entities=data.get("input_entities", []),
            rule=data.get("rule", ""),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass
class PlanExpression:
    """Structured expression representation for hardware logic."""
    expression_id: str = ""
    expression_type: str = "BOOLEAN"  # BOOLEAN, ARITHMETIC, MUX, SLICE, CONCAT, IDENTIFIER, LITERAL, CONDITIONAL, UNARY
    operator: str = ""                # "+", "-", "*", "&", "|", "^", "==", "!=", "<", ">", "<=", ">=", "&&", "||", "!", "~", "MUX", "SLICE", "CONCAT"
    operands: List[Any] = field(default_factory=list)
    condition: Optional[str] = None
    result: Optional[str] = None
    width: Optional[int] = None
    signed: bool = False
    source_references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        ops = []
        for op in self.operands:
            if hasattr(op, "to_dict"):
                ops.append(op.to_dict())
            elif isinstance(op, dict):
                ops.append(op)
            else:
                ops.append(op)
        d = {
            "expression_id": self.expression_id,
            "expression_type": self.expression_type,
            "operator": self.operator,
            "operands": ops,
            "condition": self.condition,
            "result": self.result,
            "width": self.width,
            "signed": self.signed,
            "source_references": self.source_references,
        }
        return {k: v for k, v in d.items() if v is not None}

    def to_verilog_expr(self) -> str:
        """Render machine-readable PlanExpression to synthesizable Verilog expression."""
        t = self.expression_type.upper()
        if t in ("IDENTIFIER", "LITERAL"):
            res = str(self.operands[0]) if self.operands else (self.result or "0")
        elif t == "SLICE" and len(self.operands) >= 3:
            res = f"{self.operands[0]}[{self.operands[1]}:{self.operands[2]}]"
        elif t == "SLICE" and len(self.operands) == 2:
            res = f"{self.operands[0]}[{self.operands[1]}]"
        elif t == "CONCAT":
            inner = ", ".join(op.to_verilog_expr() if hasattr(op, "to_verilog_expr") else str(op) for op in self.operands)
            res = f"{{{inner}}}"
        elif t in ("MUX", "CONDITIONAL"):
            cond = str(self.condition or (self.operands[0] if len(self.operands) > 2 else "1'b1"))
            true_v = self.operands[0].to_verilog_expr() if hasattr(self.operands[0], "to_verilog_expr") else str(self.operands[0])
            false_v = self.operands[1].to_verilog_expr() if (len(self.operands) > 1 and hasattr(self.operands[1], "to_verilog_expr")) else (str(self.operands[1]) if len(self.operands) > 1 else "0")
            res = f"({cond} ? {true_v} : {false_v})"
        elif t == "UNARY" and self.operands:
            op_str = self.operands[0].to_verilog_expr() if hasattr(self.operands[0], "to_verilog_expr") else str(self.operands[0])
            res = f"{self.operator}{op_str}"
        elif t in ("BOOLEAN", "ARITHMETIC") and len(self.operands) >= 2:
            op1 = self.operands[0].to_verilog_expr() if hasattr(self.operands[0], "to_verilog_expr") else str(self.operands[0])
            op2 = self.operands[1].to_verilog_expr() if hasattr(self.operands[1], "to_verilog_expr") else str(self.operands[1])
            op_sym = self.operator or "&"
            res = f"({op1} {op_sym} {op2})"
        elif self.operands:
            res = str(self.operands[0])
        else:
            res = self.result or "0"
        return re.sub(r"\b0x([0-9a-fA-F]+)\b", r"'h\1", res)

    @classmethod
    def parse_expression(cls, raw: str, expr_id: str = "") -> PlanExpression:
        """Parse raw hardware expression string into a structured PlanExpression."""
        if not raw or not isinstance(raw, str):
            return cls(expression_id=expr_id or "expr_lit", expression_type="LITERAL", operands=["0"])
        s = raw.strip()

        # Slice: sig[msb:lsb] or sig[bit]
        m_slice = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*\[\s*(\d+|[a-zA-Z0-9_+\-]+)\s*(?::\s*(\d+|[a-zA-Z0-9_+\-]+))?\s*\]$", s)
        if m_slice:
            sig = m_slice.group(1)
            msb = m_slice.group(2)
            lsb = m_slice.group(3)
            if lsb is not None:
                return cls(
                    expression_id=expr_id or f"expr_slice_{sig}",
                    expression_type="SLICE",
                    operator="SLICE",
                    operands=[sig, int(msb) if msb.isdigit() else msb, int(lsb) if lsb.isdigit() else lsb]
                )
            else:
                return cls(
                    expression_id=expr_id or f"expr_bit_{sig}",
                    expression_type="SLICE",
                    operator="SLICE",
                    operands=[sig, int(msb) if msb.isdigit() else msb]
                )

        # Concatenation: {a, b, ...}
        if s.startswith("{") and s.endswith("}"):
            parts = [p.strip() for p in s[1:-1].split(",") if p.strip()]
            return cls(
                expression_id=expr_id or "expr_concat",
                expression_type="CONCAT",
                operator="CONCAT",
                operands=parts
            )

        # Ternary conditional: cond ? val_true : val_false
        m_tern = re.match(r"^([^?]+)\s*\?\s*([^:]+)\s*:\s*(.+)$", s)
        if m_tern:
            cond = m_tern.group(1).strip()
            val_t = m_tern.group(2).strip()
            val_f = m_tern.group(3).strip()
            return cls(
                expression_id=expr_id or "expr_mux",
                expression_type="CONDITIONAL",
                operator="?",
                condition=cond,
                operands=[val_t, val_f]
            )

        # Binary operators
        bin_ops = ["==", "!=", "<=", ">=", "&&", "||", "<<", ">>", "+", "-", "*", "&", "|", "^", "<", ">"]
        for op in bin_ops:
            pat = rf"^(.+?)\s*{re.escape(op)}\s*(.+)$"
            m_op = re.match(pat, s)
            if m_op:
                left = m_op.group(1).strip().strip("()")
                right = m_op.group(2).strip().strip("()")
                expr_t = "ARITHMETIC" if op in ("+", "-", "*", "<<", ">>") else "BOOLEAN"
                return cls(
                    expression_id=expr_id or f"expr_{expr_t.lower()}",
                    expression_type=expr_t,
                    operator=op,
                    operands=[left, right]
                )

        # Unary: !a, ~a
        if s.startswith("!") or s.startswith("~"):
            return cls(
                expression_id=expr_id or "expr_unary",
                expression_type="UNARY",
                operator=s[0],
                operands=[s[1:].strip()]
            )

        # Literal or Identifier
        is_lit = re.match(r"^\d+'[bBhdD][0-9a-fA-F_]+|\b\d+\b", s)
        return cls(
            expression_id=expr_id or ("expr_lit" if is_lit else "expr_id"),
            expression_type="LITERAL" if is_lit else "IDENTIFIER",
            operands=[s]
        )

    @classmethod
    def from_dict(cls, data: Any) -> PlanExpression:
        if not isinstance(data, dict):
            return cls.parse_expression(str(data))
        ops = []
        for o in data.get("operands", []):
            if isinstance(o, dict) and ("expression_type" in o or "operator" in o):
                ops.append(cls.from_dict(o))
            else:
                ops.append(o)
        return cls(
            expression_id=data.get("expression_id", ""),
            expression_type=data.get("expression_type", "BOOLEAN"),
            operator=data.get("operator", ""),
            operands=ops,
            condition=data.get("condition"),
            result=data.get("result"),
            width=data.get("width"),
            signed=bool(data.get("signed", False)),
            source_references=data.get("source_references", []),
        )


@dataclass
class PlanRegisterUpdate:
    """Structured representation of sequential register update semantics."""
    target: str
    condition: str = "1'b1"
    enable: Optional[str] = None
    load: Optional[str] = None
    next_value: Optional[PlanExpression] = None
    hold_behavior: str = "PRESERVE"   # PRESERVE, DEFAULT, ZERO
    assignment_priority: int = 0
    reset_behavior: Optional[str] = None
    source_references: List[str] = field(default_factory=list)
    origin: PlanOrigin = PlanOrigin.INHERITED
    derivation: Optional[DerivationMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "condition": self.condition,
            "enable": self.enable,
            "load": self.load,
            "next_value": self.next_value.to_dict() if self.next_value else None,
            "hold_behavior": self.hold_behavior,
            "assignment_priority": self.assignment_priority,
            "reset_behavior": self.reset_behavior,
            "source_references": self.source_references,
            "origin": self.origin.value if isinstance(self.origin, PlanOrigin) else str(self.origin),
            "derivation": self.derivation.to_dict() if self.derivation else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanRegisterUpdate:
        if not isinstance(data, dict):
            return cls(target=str(data))
        nv = data.get("next_value")
        nv_obj = PlanExpression.from_dict(nv) if nv else None
        orig = PlanOrigin(data["origin"]) if data.get("origin") in PlanOrigin._value2member_map_ else PlanOrigin.INHERITED
        deriv = DerivationMetadata.from_dict(data["derivation"]) if data.get("derivation") else None
        return cls(
            target=data.get("target", ""),
            condition=data.get("condition", "1'b1"),
            enable=data.get("enable"),
            load=data.get("load"),
            next_value=nv_obj,
            hold_behavior=data.get("hold_behavior", "PRESERVE"),
            assignment_priority=int(data.get("assignment_priority", 0)),
            reset_behavior=data.get("reset_behavior"),
            source_references=data.get("source_references", []),
            origin=orig,
            derivation=deriv,
        )


@dataclass
class PlanSequentialProcess:
    """Explicit, structured sequential always/process block definition."""
    process_id: str
    process_type: str = "ALWAYS_FF"   # ALWAYS_FF, ALWAYS, PROCESS
    clock_signal: str = "clk"
    clock_edge: str = "POSEDGE"       # POSEDGE, NEGEDGE
    reset_signal: Optional[str] = None
    reset_type: str = "ASYNC"         # ASYNC, SYNC, NONE
    reset_active_level: str = "LOW"   # LOW, HIGH
    reset_priority: str = "HIGH"
    reset_actions: Dict[str, str] = field(default_factory=dict)
    register_updates: List[PlanRegisterUpdate] = field(default_factory=list)
    fsm_state_updates: Dict[str, Any] = field(default_factory=dict)
    source_references: List[str] = field(default_factory=list)
    origin: PlanOrigin = PlanOrigin.INHERITED
    derivation: Optional[DerivationMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "process_id": self.process_id,
            "process_type": self.process_type,
            "clock_signal": self.clock_signal,
            "clock_edge": self.clock_edge,
            "reset_signal": self.reset_signal,
            "reset_type": self.reset_type,
            "reset_active_level": self.reset_active_level,
            "reset_priority": self.reset_priority,
            "reset_actions": self.reset_actions,
            "register_updates": [u.to_dict() for u in self.register_updates],
            "fsm_state_updates": self.fsm_state_updates,
            "source_references": self.source_references,
            "origin": self.origin.value if isinstance(self.origin, PlanOrigin) else str(self.origin),
            "derivation": self.derivation.to_dict() if self.derivation else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanSequentialProcess:
        if not isinstance(data, dict):
            return cls(process_id="proc_seq")
        clk_obj = data.get("clock", {})
        rst_obj = data.get("reset", {})
        clk_sig = clk_obj.get("signal", data.get("clock_signal", "clk")) if isinstance(clk_obj, dict) else str(clk_obj or "clk")
        clk_edge = clk_obj.get("edge", data.get("clock_edge", "POSEDGE")) if isinstance(clk_obj, dict) else data.get("clock_edge", "POSEDGE")
        rst_sig = rst_obj.get("signal", data.get("reset_signal")) if isinstance(rst_obj, dict) else str(rst_obj or "")
        rst_type = rst_obj.get("type", data.get("reset_type", "ASYNC")) if isinstance(rst_obj, dict) else data.get("reset_type", "ASYNC")
        rst_lvl = rst_obj.get("active_level", data.get("reset_active_level", "LOW")) if isinstance(rst_obj, dict) else data.get("reset_active_level", "LOW")

        updates = [
            PlanRegisterUpdate.from_dict(u) if isinstance(u, dict) else u
            for u in data.get("register_updates", [])
        ]
        orig = PlanOrigin(data["origin"]) if data.get("origin") in PlanOrigin._value2member_map_ else PlanOrigin.INHERITED
        deriv = DerivationMetadata.from_dict(data["derivation"]) if data.get("derivation") else None
        return cls(
            process_id=data.get("process_id", "SEQ_001"),
            process_type=data.get("process_type", "ALWAYS_FF"),
            clock_signal=clk_sig,
            clock_edge=clk_edge,
            reset_signal=rst_sig or None,
            reset_type=rst_type,
            reset_active_level=rst_lvl,
            reset_priority=data.get("reset_priority", "HIGH"),
            reset_actions=data.get("reset_actions", {}),
            register_updates=updates,
            fsm_state_updates=data.get("fsm_state_updates", {}),
            source_references=data.get("source_references", []),
            origin=orig,
            derivation=deriv,
        )


@dataclass
class PlanCombinationalAssignment:
    """Explicit combinational target assignment."""
    target: str
    expression: Optional[PlanExpression] = None
    expression_raw: str = ""
    condition: Optional[str] = None
    default_assignment: Optional[str] = None
    source_references: List[str] = field(default_factory=list)
    origin: PlanOrigin = PlanOrigin.INHERITED
    derivation: Optional[DerivationMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "expression": self.expression.to_dict() if self.expression else None,
            "expression_raw": self.expression_raw or (self.expression.to_verilog_expr() if self.expression else ""),
            "condition": self.condition,
            "default_assignment": self.default_assignment,
            "source_references": self.source_references,
            "origin": self.origin.value if isinstance(self.origin, PlanOrigin) else str(self.origin),
            "derivation": self.derivation.to_dict() if self.derivation else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanCombinationalAssignment:
        if not isinstance(data, dict):
            return cls(target=str(data))
        expr = data.get("expression")
        expr_obj = PlanExpression.from_dict(expr) if expr else None
        expr_raw = str(data.get("expression_raw") or (expr.get("raw") if isinstance(expr, dict) else (expr or "")))
        if not expr_obj and expr_raw:
            expr_obj = PlanExpression.parse_expression(expr_raw)
        orig = PlanOrigin(data["origin"]) if data.get("origin") in PlanOrigin._value2member_map_ else PlanOrigin.INHERITED
        deriv = DerivationMetadata.from_dict(data["derivation"]) if data.get("derivation") else None
        return cls(
            target=data.get("target", ""),
            expression=expr_obj,
            expression_raw=expr_raw,
            condition=data.get("condition"),
            default_assignment=data.get("default_assignment"),
            source_references=data.get("source_references", []),
            origin=orig,
            derivation=deriv,
        )


@dataclass
class PlanCombinationalProcess:
    """Explicit, structured combinational always_comb or continuous assign block."""
    process_id: str
    process_type: str = "ALWAYS_COMB"  # ALWAYS_COMB, CONTINUOUS_ASSIGN
    input_dependencies: List[str] = field(default_factory=list)
    output_targets: List[str] = field(default_factory=list)
    assignments: List[PlanCombinationalAssignment] = field(default_factory=list)
    data_selections: List[Any] = field(default_factory=list)
    source_references: List[str] = field(default_factory=list)
    origin: PlanOrigin = PlanOrigin.INHERITED
    derivation: Optional[DerivationMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "process_id": self.process_id,
            "process_type": self.process_type,
            "input_dependencies": self.input_dependencies,
            "output_targets": self.output_targets,
            "assignments": [a.to_dict() for a in self.assignments],
            "data_selections": [ds.to_dict() if hasattr(ds, "to_dict") else ds for ds in self.data_selections],
            "source_references": self.source_references,
            "origin": self.origin.value if isinstance(self.origin, PlanOrigin) else str(self.origin),
            "derivation": self.derivation.to_dict() if self.derivation else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanCombinationalProcess:
        if not isinstance(data, dict):
            return cls(process_id="proc_comb")
        assigns = [
            PlanCombinationalAssignment.from_dict(a) if isinstance(a, dict) else a
            for a in data.get("assignments", [])
        ]
        orig = PlanOrigin(data["origin"]) if data.get("origin") in PlanOrigin._value2member_map_ else PlanOrigin.INHERITED
        deriv = DerivationMetadata.from_dict(data["derivation"]) if data.get("derivation") else None
        return cls(
            process_id=data.get("process_id", "COMB_001"),
            process_type=data.get("process_type", "ALWAYS_COMB"),
            input_dependencies=data.get("input_dependencies", []),
            output_targets=data.get("output_targets", []),
            assignments=assigns,
            data_selections=data.get("data_selections", []),
            source_references=data.get("source_references", []),
            origin=orig,
            derivation=deriv,
        )


@dataclass
class PlanUnresolvedLogic:
    """Represents an explicitly unresolved logic requirement or gap in specification."""
    affected_entity: str
    description: str
    missing_information: str
    affected_component: Optional[str] = None
    reason: Optional[str] = None
    source_evidence: Optional[str] = None
    blocks_generation: bool = False
    requires_human_decision: bool = False
    severity: str = "WARNING"  # BLOCKING, WARNING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "affected_entity": self.affected_entity,
            "description": self.description,
            "missing_information": self.missing_information,
            "affected_component": self.affected_component,
            "reason": self.reason,
            "source_evidence": self.source_evidence,
            "blocks_generation": self.blocks_generation,
            "requires_human_decision": self.requires_human_decision,
            "severity": self.severity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanUnresolvedLogic:
        if not isinstance(data, dict):
            return cls(affected_entity="unknown", description=str(data), missing_information=str(data))
        return cls(
            affected_entity=data.get("affected_entity", "general"),
            description=data.get("description", ""),
            missing_information=data.get("missing_information", ""),
            affected_component=data.get("affected_component"),
            reason=data.get("reason"),
            source_evidence=data.get("source_evidence"),
            blocks_generation=bool(data.get("blocks_generation", False)),
            requires_human_decision=bool(data.get("requires_human_decision", False)),
            severity=data.get("severity", "WARNING"),
        )


@dataclass
class PlanProtocolRule:
    """Individual structured protocol rule inside a component's protocol contract."""
    rule_id: str
    name: str = ""
    rule_type: str = "HANDSHAKE"  # HANDSHAKE, TRANSACTION_ACCEPTANCE, ORDERING, STABILITY, RESET, TIMING, VERIFICATION_ONLY
    description: str = ""
    applicable_interfaces: List[str] = field(default_factory=list)
    applicable_signals: List[str] = field(default_factory=list)
    handshake_condition: Optional[str] = None
    acceptance_condition: Optional[str] = None
    ordering_requirement: Optional[str] = None
    stability_constraint: Optional[str] = None
    reset_requirement: Optional[str] = None
    timing_constraint: Optional[str] = None
    is_synthesizable_behavior: bool = True
    source_references: List[str] = field(default_factory=list)
    source_evidence: str = ""
    origin: PlanOrigin = PlanOrigin.INHERITED
    derivation: Optional[DerivationMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "rule_type": self.rule_type,
            "description": self.description,
            "applicable_interfaces": self.applicable_interfaces,
            "applicable_signals": self.applicable_signals,
            "handshake_condition": self.handshake_condition,
            "acceptance_condition": self.acceptance_condition,
            "ordering_requirement": self.ordering_requirement,
            "stability_constraint": self.stability_constraint,
            "reset_requirement": self.reset_requirement,
            "timing_constraint": self.timing_constraint,
            "is_synthesizable_behavior": self.is_synthesizable_behavior,
            "source_references": self.source_references,
            "source_evidence": self.source_evidence,
            "origin": self.origin.value if isinstance(self.origin, PlanOrigin) else str(self.origin),
            "derivation": self.derivation.to_dict() if self.derivation else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanProtocolRule:
        if not isinstance(data, dict):
            return cls(rule_id=str(data))
        orig = PlanOrigin(data["origin"]) if data.get("origin") in PlanOrigin._value2member_map_ else PlanOrigin.INHERITED
        deriv = DerivationMetadata.from_dict(data["derivation"]) if data.get("derivation") else None
        return cls(
            rule_id=data.get("rule_id", "prule_0"),
            name=data.get("name", ""),
            rule_type=data.get("rule_type", "HANDSHAKE"),
            description=data.get("description", ""),
            applicable_interfaces=list(data.get("applicable_interfaces", [])),
            applicable_signals=list(data.get("applicable_signals", [])),
            handshake_condition=data.get("handshake_condition"),
            acceptance_condition=data.get("acceptance_condition"),
            ordering_requirement=data.get("ordering_requirement"),
            stability_constraint=data.get("stability_constraint"),
            reset_requirement=data.get("reset_requirement"),
            timing_constraint=data.get("timing_constraint"),
            is_synthesizable_behavior=bool(data.get("is_synthesizable_behavior", True)),
            source_references=list(data.get("source_references", [])),
            source_evidence=data.get("source_evidence", ""),
            origin=orig,
            derivation=deriv,
        )


@dataclass
class PlanProtocolContract:
    """Target component-scoped protocol contract containing applicable protocol specifications."""
    protocol_name: str = "GENERIC"
    protocol_version: str = "1.0"
    applicable_interfaces: List[str] = field(default_factory=list)
    rules: List[PlanProtocolRule] = field(default_factory=list)
    verification_assertions: List[Dict[str, Any]] = field(default_factory=list)
    source_evidence_ids: List[str] = field(default_factory=list)
    origin: PlanOrigin = PlanOrigin.INHERITED
    derivation: Optional[DerivationMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_name": self.protocol_name,
            "protocol_version": self.protocol_version,
            "applicable_interfaces": self.applicable_interfaces,
            "rules": [r.to_dict() for r in self.rules],
            "verification_assertions": self.verification_assertions,
            "source_evidence_ids": self.source_evidence_ids,
            "origin": self.origin.value if isinstance(self.origin, PlanOrigin) else str(self.origin),
            "derivation": self.derivation.to_dict() if self.derivation else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanProtocolContract:
        if not isinstance(data, dict):
            return cls(protocol_name=str(data))
        rules_list = [
            PlanProtocolRule.from_dict(r) if isinstance(r, dict) else r
            for r in data.get("rules", [])
        ]
        orig = PlanOrigin(data["origin"]) if data.get("origin") in PlanOrigin._value2member_map_ else PlanOrigin.INHERITED
        deriv = DerivationMetadata.from_dict(data["derivation"]) if data.get("derivation") else None
        return cls(
            protocol_name=data.get("protocol_name", "GENERIC"),
            protocol_version=data.get("protocol_version", "1.0"),
            applicable_interfaces=list(data.get("applicable_interfaces", [])),
            rules=rules_list,
            verification_assertions=list(data.get("verification_assertions", [])),
            source_evidence_ids=list(data.get("source_evidence_ids", [])),
            origin=orig,
            derivation=deriv,
        )


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
    origin: PlanOrigin = PlanOrigin.INHERITED
    derivation: Optional[DerivationMetadata] = None

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
            "origin": self.origin.value if isinstance(self.origin, PlanOrigin) else str(self.origin),
            "derivation": self.derivation.to_dict() if self.derivation else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanPort:
        c_val = data.get("classification", "explicit")
        try:
            c_enum = EvidenceClassification(c_val)
        except Exception:
            c_enum = EvidenceClassification.EXPLICIT

        o_val = data.get("origin", "INHERITED")
        try:
            o_enum = PlanOrigin(o_val)
        except Exception:
            o_enum = PlanOrigin.INHERITED

        drv = data.get("derivation")
        drv_obj = DerivationMetadata.from_dict(drv) if drv else None

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
            origin=o_enum,
            derivation=drv_obj,
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
    origin: PlanOrigin = PlanOrigin.INHERITED
    derivation: Optional[DerivationMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": str(self.value),
            "type": self.type,
            "description": self.description,
            "usage": self.usage,
            "is_behaviorally_defined": self.is_behaviorally_defined,
            "source_ref": self.source_ref,
            "origin": self.origin.value if isinstance(self.origin, PlanOrigin) else str(self.origin),
            "derivation": self.derivation.to_dict() if self.derivation else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanParameter:
        o_val = data.get("origin", "INHERITED")
        try:
            o_enum = PlanOrigin(o_val)
        except Exception:
            o_enum = PlanOrigin.INHERITED
        drv = data.get("derivation")
        drv_obj = DerivationMetadata.from_dict(drv) if drv else None
        return cls(
            name=data.get("name", "PARAM"),
            value=data.get("value", data.get("default", "0")),
            type=data.get("type", "integer"),
            description=data.get("description", ""),
            usage=data.get("usage", ""),
            is_behaviorally_defined=bool(data.get("is_behaviorally_defined", True)),
            source_ref=data.get("source_ref", ""),
            origin=o_enum,
            derivation=drv_obj,
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
    origin: PlanOrigin = PlanOrigin.IMPLEMENTATION_DECISION
    derivation: Optional[DerivationMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "width": str(self.width),
            "signedness": self.signedness,
            "driving_type": self.driving_type,
            "driver": self.driver,
            "source_ref": self.source_ref,
            "origin": self.origin.value if isinstance(self.origin, PlanOrigin) else str(self.origin),
            "derivation": self.derivation.to_dict() if self.derivation else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanInternalSignal:
        o_val = data.get("origin", "IMPLEMENTATION_DECISION")
        try:
            o_enum = PlanOrigin(o_val)
        except Exception:
            o_enum = PlanOrigin.IMPLEMENTATION_DECISION
        drv = data.get("derivation")
        drv_obj = DerivationMetadata.from_dict(drv) if drv else None
        return cls(
            name=data.get("name", "sig"),
            width=str(data.get("width", "1")),
            signedness=bool(data.get("signedness", False)),
            driving_type=data.get("driving_type", "wire"),
            driver=data.get("driver", ""),
            source_ref=data.get("source_ref", ""),
            origin=o_enum,
            derivation=drv_obj,
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
    origin: PlanOrigin = PlanOrigin.INHERITED
    derivation: Optional[DerivationMetadata] = None

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
            "origin": self.origin.value if isinstance(self.origin, PlanOrigin) else str(self.origin),
            "derivation": self.derivation.to_dict() if self.derivation else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanRegister:
        r_reset = data.get("reset_val", data.get("reset_value", "0"))
        flds = [
            PlanRegisterField.from_dict(f) if isinstance(f, dict) else f
            for f in data.get("fields", [])
        ]
        o_val = data.get("origin", "INHERITED")
        try:
            o_enum = PlanOrigin(o_val)
        except Exception:
            o_enum = PlanOrigin.INHERITED
        drv = data.get("derivation")
        drv_obj = DerivationMetadata.from_dict(drv) if drv else None
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
            origin=o_enum,
            derivation=drv_obj,
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
    origin: PlanOrigin = PlanOrigin.INHERITED
    derivation: Optional[DerivationMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "binary_encoding": self.binary_encoding,
            "associated_operations": self.associated_operations,
            "evidence": self.evidence,
            "efs_id": self.efs_id,
            "origin": self.origin.value if isinstance(self.origin, PlanOrigin) else str(self.origin),
            "derivation": self.derivation.to_dict() if self.derivation else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanOpcode:
        o_val = data.get("origin", "INHERITED")
        try:
            o_enum = PlanOrigin(o_val)
        except Exception:
            o_enum = PlanOrigin.INHERITED
        drv = data.get("derivation")
        drv_obj = DerivationMetadata.from_dict(drv) if drv else None
        return cls(
            name=data.get("name", data.get("mnemonic", "OPC")),
            binary_encoding=str(data.get("binary_encoding", data.get("encoding", ""))),
            associated_operations=data.get("associated_operations", data.get("operations", [])),
            evidence=data.get("evidence", data.get("source_evidence", "")),
            efs_id=data.get("efs_id"),
            origin=o_enum,
            derivation=drv_obj,
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
    sequential_logic: List[PlanSequentialProcess] = field(default_factory=list)
    combinational_logic: List[PlanCombinationalProcess] = field(default_factory=list)
    unresolved_logic: List[PlanUnresolvedLogic] = field(default_factory=list)
    traceability: PlanTraceability = field(default_factory=PlanTraceability)
    output_driving_strategy: Dict[str, str] = field(default_factory=dict)
    protocol_contract: Optional[PlanProtocolContract] = None

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
            "sequential_logic": [seq.to_dict() for seq in self.sequential_logic],
            "combinational_logic": [comb.to_dict() for comb in self.combinational_logic],
            "unresolved_logic": [u.to_dict() for u in self.unresolved_logic],
            "protocol_contract": self.protocol_contract.to_dict() if self.protocol_contract else None,
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
        # Sequential logic
        seq_logic = [
            PlanSequentialProcess.from_dict(s) if isinstance(s, dict) else s
            for s in data.get("sequential_logic", [])
        ]
        # Combinational logic
        comb_logic = [
            PlanCombinationalProcess.from_dict(c) if isinstance(c, dict) else c
            for c in data.get("combinational_logic", [])
        ]
        # Unresolved logic
        unresolved = [
            PlanUnresolvedLogic.from_dict(u) if isinstance(u, dict) else u
            for u in data.get("unresolved_logic", [])
        ]
        # Traceability
        traceability_obj = (
            PlanTraceability.from_dict(data["traceability"])
            if isinstance(data.get("traceability"), dict)
            else PlanTraceability()
        )
        # Protocol contract
        proto_contract = None
        if data.get("protocol_contract") and isinstance(data["protocol_contract"], dict):
            proto_contract = PlanProtocolContract.from_dict(data["protocol_contract"])

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
            sequential_logic=seq_logic,
            combinational_logic=comb_logic,
            unresolved_logic=unresolved,
            protocol_contract=proto_contract,
            traceability=traceability_obj,
            output_driving_strategy=data.get("output_driving_strategy", {}),
        )

