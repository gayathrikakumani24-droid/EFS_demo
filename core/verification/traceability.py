"""
Requirement-to-RTL Traceability Engine.

Maintains bidirectional mappings between:
- Original specification requirements
- EFS IR entities and relationships
- Generated RTL modules, ports, signals, functional logic, and FSM states

Produces comprehensive traceability reports detailing:
- Fully implemented requirements
- Partially implemented requirements
- Missing requirements
- Requirements awaiting clarification
- RTL elements without supporting requirements (detecting fabricated hardware)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from utils.logger import get_logger

logger = get_logger("verification.traceability")


class TraceabilityStatus(str, Enum):
    FULLY_IMPLEMENTED = "FULLY_IMPLEMENTED"
    PARTIALLY_IMPLEMENTED = "PARTIALLY_IMPLEMENTED"
    MISSING = "MISSING"
    AWAITING_CLARIFICATION = "AWAITING_CLARIFICATION"


@dataclass
class TraceabilityItem:
    requirement_id: str
    requirement_type: str  # Signal, Port, Register, FSMState, Transition, Opcode, ClockReset, Operation
    name: str
    status: TraceabilityStatus
    specification_evidence: str
    rtl_mapping: Optional[str] = None
    line_number: Optional[int] = None
    details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "requirement_type": self.requirement_type,
            "name": self.name,
            "status": self.status.value,
            "specification_evidence": self.specification_evidence,
            "rtl_mapping": self.rtl_mapping,
            "line_number": self.line_number,
            "details": self.details,
        }


@dataclass
class UnmappedRTLElement:
    element_type: str  # Port, Register, State, Wire, Logic
    name: str
    line_number: Optional[int] = None
    snippet: Optional[str] = None
    risk: str = "Potential fabricated or unsupported hardware construct."

    @property
    def identifier(self) -> str:
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_type": self.element_type,
            "name": self.name,
            "line_number": self.line_number,
            "snippet": self.snippet,
            "risk": self.risk,
        }


@dataclass
class TraceabilityReport:
    module_name: str
    target_language: str
    fully_implemented: List[TraceabilityItem] = field(default_factory=list)
    partially_implemented: List[TraceabilityItem] = field(default_factory=list)
    missing: List[TraceabilityItem] = field(default_factory=list)
    awaiting_clarification: List[TraceabilityItem] = field(default_factory=list)
    unmapped_rtl_elements: List[UnmappedRTLElement] = field(default_factory=list)
    coverage_percentage: float = 0.0

    @property
    def unmapped_elements(self) -> List[UnmappedRTLElement]:
        return self.unmapped_rtl_elements

    @property
    def items(self) -> List[TraceabilityItem]:
        return self.fully_implemented + self.partially_implemented + self.missing + self.awaiting_clarification

    @property
    def is_specification_compliant(self) -> bool:
        return len(self.missing) == 0 and len(self.awaiting_clarification) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_name": self.module_name,
            "target_language": self.target_language,
            "coverage_percentage": round(self.coverage_percentage, 2),
            "is_specification_compliant": self.is_specification_compliant,
            "counts": {
                "fully_implemented": len(self.fully_implemented),
                "partially_implemented": len(self.partially_implemented),
                "missing": len(self.missing),
                "awaiting_clarification": len(self.awaiting_clarification),
                "unmapped_rtl_elements": len(self.unmapped_rtl_elements),
                "total_requirements": len(self.fully_implemented) + len(self.partially_implemented) + len(self.missing) + len(self.awaiting_clarification),
            },
            "fully_implemented": [i.to_dict() for i in self.fully_implemented],
            "partially_implemented": [i.to_dict() for i in self.partially_implemented],
            "missing": [i.to_dict() for i in self.missing],
            "awaiting_clarification": [i.to_dict() for i in self.awaiting_clarification],
            "unmapped_rtl_elements": [e.to_dict() for e in self.unmapped_rtl_elements],
        }

    def to_markdown(self) -> str:
        counts = self.to_dict()["counts"]
        lines = [
            f"# Requirement-to-RTL Traceability Report: `{self.module_name}`",
            f"**Target Language:** {self.target_language} | **Implementation Coverage:** {round(self.coverage_percentage, 1)}%",
            "",
            "| Requirement | Type | EFS IR Entity | Status | RTL Mapping | Specification Citation |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for item in self.items:
            lines.append(f"| `{item.name}` | {item.requirement_type} | `{item.requirement_id}` | {item.status.value} | {item.rtl_mapping or 'N/A'} | {item.specification_evidence} |")
        lines.extend([
            "",
            "| Category | Count |",
            "| :--- | :--- |",
            f"| 🟢 Fully Implemented | {counts['fully_implemented']} |",
            f"| 🟡 Partially Implemented | {counts['partially_implemented']} |",
            f"| 🔴 Missing Requirements | {counts['missing']} |",
            f"| 🟠 Awaiting Clarification | {counts['awaiting_clarification']} |",
            f"| 🟣 Unmapped RTL Elements | {counts['unmapped_rtl_elements']} |",
            "",
        ])

        if self.missing:
            lines.extend([
                "### 🔴 Missing Requirements",
                "| ID | Type | Name | Spec Evidence |",
                "| :--- | :--- | :--- | :--- |",
            ])
            for m in self.missing:
                lines.append(f"| `{m.requirement_id}` | {m.requirement_type} | `{m.name}` | {m.specification_evidence} |")
            lines.append("")

        if self.awaiting_clarification:
            lines.extend([
                "### 🟠 Requirements Awaiting Clarification",
                "| ID | Type | Name | Details |",
                "| :--- | :--- | :--- | :--- |",
            ])
            for a in self.awaiting_clarification:
                lines.append(f"| `{a.requirement_id}` | {a.requirement_type} | `{a.name}` | {a.details or a.specification_evidence} |")
            lines.append("")

        if self.unmapped_rtl_elements:
            lines.extend([
                "### 🟣 Unmapped RTL Elements (Fabrication Risk)",
                "| Type | Name | Line | Risk |",
                "| :--- | :--- | :--- | :--- |",
            ])
            for u in self.unmapped_rtl_elements:
                lines.append(f"| {u.element_type} | `{u.name}` | {u.line_number or 'N/A'} | {u.risk} |")
            lines.append("")

        return "\n".join(lines)


def generate_traceability_report(
    code: str,
    efs_ir: Optional[Any] = None,
    rtl_plan: Optional[Dict[str, Any]] = None,
    target_language: str = "Verilog",
    design_spec: Optional[str] = None,
    language_version: Optional[str] = None,
    **kwargs: Any
) -> TraceabilityReport:
    """
    Generate requirement-to-RTL traceability report by matching EFS IR entities and
    plan items against the actual generated RTL source code.
    """
    module_name = "target_module"
    m_match = re.search(r"\bmodule\s+([a-zA-Z_][a-zA-Z0-9_]*)", code)
    if m_match:
        module_name = m_match.group(1)
    elif rtl_plan and rtl_plan.get("module_name"):
        module_name = rtl_plan["module_name"]
    elif efs_ir and getattr(getattr(efs_ir, "metadata", None), "design_name", None):
        module_name = efs_ir.metadata.design_name

    fully_implemented: List[TraceabilityItem] = []
    partially_implemented: List[TraceabilityItem] = []
    missing: List[TraceabilityItem] = []
    awaiting_clarification: List[TraceabilityItem] = []
    unmapped_rtl: List[UnmappedRTLElement] = []

    lines = code.split("\n")
    clean_code = re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL))

    # Known entity names from EFS IR
    known_efs_names: Set[str] = set()

    # Active unresolved conflict entities
    conflict_entities: Set[str] = set()
    if efs_ir and hasattr(efs_ir, "conflicts") and efs_ir.conflicts:
        for c in efs_ir.conflicts:
            if getattr(c, "resolution_status", "UNRESOLVED") == "UNRESOLVED":
                for ent in getattr(c, "entities", []):
                    conflict_entities.add(str(ent).lower())

    # 1. Trace Interface Ports / Signals
    signals_to_check = []
    if efs_ir and hasattr(efs_ir, "signals") and efs_ir.signals:
        for s in efs_ir.signals:
            signals_to_check.append({
                "id": getattr(s, "signal_id", s.name),
                "name": s.name,
                "direction": s.direction,
                "width": str(s.width or "1"),
                "evidence": getattr(getattr(s, "traceability", None), "original_text", f"Signal '{s.name}'"),
            })
    elif rtl_plan and rtl_plan.get("ports"):
        for p in rtl_plan["ports"]:
            signals_to_check.append({
                "id": p.get("name", ""),
                "name": p.get("name", ""),
                "direction": p.get("direction", "input"),
                "width": str(p.get("width", "1")),
                "evidence": p.get("description", f"Port '{p.get('name')}'"),
            })

    for sig in signals_to_check:
        s_name = sig["name"]
        s_id = sig["id"]
        s_dir = sig["direction"]
        s_evid = sig["evidence"]
        known_efs_names.add(s_name.lower())

        if s_name.lower() in conflict_entities:
            awaiting_clarification.append(
                TraceabilityItem(
                    requirement_id=s_id,
                    requirement_type="Port",
                    name=s_name,
                    status=TraceabilityStatus.AWAITING_CLARIFICATION,
                    specification_evidence=s_evid,
                    details=f"Signal '{s_name}' is associated with an active unresolved specification conflict.",
                )
            )
            continue

        # Find line number in code
        line_no = None
        for idx, l in enumerate(lines, 1):
            if re.search(rf"\b{re.escape(s_name)}\b", l):
                line_no = idx
                break

        # Check if declared as port
        is_declared_port = bool(
            re.search(rf"\b(?:input|output|inout)\s+(?:wire|reg|logic)?\s*(?:\[[^\]]+\])?\s*{re.escape(s_name)}\b", clean_code) or
            re.search(rf"\b(?:input|output|inout)\b[^\n,;]*\b{re.escape(s_name)}\b", clean_code)
        )

        if not is_declared_port:
            missing.append(
                TraceabilityItem(
                    requirement_id=s_id,
                    requirement_type="Port",
                    name=s_name,
                    status=TraceabilityStatus.MISSING,
                    specification_evidence=s_evid,
                    details=f"Declared port '{s_name}' is missing from RTL module port list.",
                )
            )
        else:
            # Check if driven (for output) or read (for input)
            if s_dir.lower() in ("output", "out"):
                is_driven = bool(
                    re.search(rf"\bassign\s+{re.escape(s_name)}\b\s*=", clean_code) or
                    re.search(rf"\b{re.escape(s_name)}\b\s*(?:<=|=)", clean_code)
                )
                if is_driven:
                    fully_implemented.append(
                        TraceabilityItem(
                            requirement_id=s_id,
                            requirement_type="Port",
                            name=s_name,
                            status=TraceabilityStatus.FULLY_IMPLEMENTED,
                            specification_evidence=s_evid,
                            rtl_mapping=f"Declared output port '{s_name}' driven in logic.",
                            line_number=line_no,
                        )
                    )
                else:
                    partially_implemented.append(
                        TraceabilityItem(
                            requirement_id=s_id,
                            requirement_type="Port",
                            name=s_name,
                            status=TraceabilityStatus.PARTIALLY_IMPLEMENTED,
                            specification_evidence=s_evid,
                            rtl_mapping=f"Declared port '{s_name}' exists but has no driving assignment.",
                            line_number=line_no,
                            details="Output port declared but undriven in logic.",
                        )
                    )
            else:
                fully_implemented.append(
                    TraceabilityItem(
                        requirement_id=s_id,
                        requirement_type="Port",
                        name=s_name,
                        status=TraceabilityStatus.FULLY_IMPLEMENTED,
                        specification_evidence=s_evid,
                        rtl_mapping=f"Declared input port '{s_name}'.",
                        line_number=line_no,
                    )
                )

    # 2. Trace Registers
    regs_to_check = []
    if efs_ir and hasattr(efs_ir, "registers") and efs_ir.registers:
        for r in efs_ir.registers:
            regs_to_check.append({
                "id": getattr(r, "register_id", r.name),
                "name": r.name,
                "evidence": getattr(getattr(r, "traceability", None), "original_text", f"Register '{r.name}'"),
            })
    elif rtl_plan and rtl_plan.get("internal_registers"):
        for r in rtl_plan["internal_registers"]:
            regs_to_check.append({
                "id": r.get("name", ""),
                "name": r.get("name", ""),
                "evidence": r.get("purpose", f"Register '{r.get('name')}'"),
            })

    for reg in regs_to_check:
        r_name = reg["name"]
        r_id = reg["id"]
        r_evid = reg["evidence"]
        known_efs_names.add(r_name.lower())

        if r_name.lower() in conflict_entities:
            awaiting_clarification.append(
                TraceabilityItem(
                    requirement_id=r_id,
                    requirement_type="Register",
                    name=r_name,
                    status=TraceabilityStatus.AWAITING_CLARIFICATION,
                    specification_evidence=r_evid,
                )
            )
            continue

        is_present = bool(re.search(rf"\b{re.escape(r_name)}\b", clean_code))
        is_updated = bool(re.search(rf"\b{re.escape(r_name)}\b\s*<=", clean_code))
        line_no = None
        for idx, l in enumerate(lines, 1):
            if re.search(rf"\b{re.escape(r_name)}\b", l):
                line_no = idx
                break

        if not is_present:
            missing.append(
                TraceabilityItem(
                    requirement_id=r_id,
                    requirement_type="Register",
                    name=r_name,
                    status=TraceabilityStatus.MISSING,
                    specification_evidence=r_evid,
                    details=f"Required register '{r_name}' is not declared in RTL.",
                )
            )
        elif is_updated:
            fully_implemented.append(
                TraceabilityItem(
                    requirement_id=r_id,
                    requirement_type="Register",
                    name=r_name,
                    status=TraceabilityStatus.FULLY_IMPLEMENTED,
                    specification_evidence=r_evid,
                    rtl_mapping=f"Declared and updated sequential register `{r_name}`.",
                    line_number=line_no,
                )
            )
        else:
            partially_implemented.append(
                TraceabilityItem(
                    requirement_id=r_id,
                    requirement_type="Register",
                    name=r_name,
                    status=TraceabilityStatus.PARTIALLY_IMPLEMENTED,
                    specification_evidence=r_evid,
                    rtl_mapping=f"Declared register `{r_name}` without detected sequential updates.",
                    line_number=line_no,
                )
            )

    # 3. Trace FSM States and Transitions
    fsm_states_to_check = []
    fsm_trans_to_check = []
    if efs_ir and hasattr(efs_ir, "fsms") and efs_ir.fsms:
        for f in efs_ir.fsms:
            for s in f.states:
                fsm_states_to_check.append({
                    "id": getattr(s, "state_id", s.name),
                    "name": s.name,
                    "evidence": f"FSM '{f.name}' state '{s.name}'",
                })
            for t in f.transitions:
                from_s = getattr(t, "from_state", getattr(t, "source_state", ""))
                to_s = getattr(t, "to_state", getattr(t, "target_state", ""))
                cond = getattr(t, "condition", "")
                fsm_trans_to_check.append({
                    "id": getattr(t, "transition_id", f"{from_s}->{to_s}"),
                    "name": f"{from_s} -> {to_s}",
                    "condition": cond,
                    "evidence": f"FSM transition from {from_s} to {to_s} when {cond}",
                })
    elif rtl_plan and rtl_plan.get("fsm"):
        plan_fsm = rtl_plan["fsm"]
        if isinstance(plan_fsm, dict):
            for st in plan_fsm.get("states", []):
                st_name = st.get("name") if isinstance(st, dict) else str(st)
                fsm_states_to_check.append({
                    "id": st_name,
                    "name": st_name,
                    "evidence": f"FSM state '{st_name}'",
                })
            for tr in plan_fsm.get("transitions", []):
                from_s = tr.get("from_state") or tr.get("from")
                to_s = tr.get("to_state") or tr.get("to")
                cond = tr.get("condition")
                cond_str = cond.get("expression") if isinstance(cond, dict) else str(cond)
                fsm_trans_to_check.append({
                    "id": f"{from_s}->{to_s}",
                    "name": f"{from_s} -> {to_s}",
                    "condition": cond_str,
                    "evidence": f"FSM transition {from_s} -> {to_s}",
                })

    for st in fsm_states_to_check:
        st_name = st["name"]
        known_efs_names.add(st_name.lower())
        line_no = None
        for idx, l in enumerate(lines, 1):
            if re.search(rf"\b{re.escape(st_name)}\b", l):
                line_no = idx
                break

        if re.search(rf"\b{re.escape(st_name)}\b", clean_code):
            fully_implemented.append(
                TraceabilityItem(
                    requirement_id=st["id"],
                    requirement_type="FSMState",
                    name=st_name,
                    status=TraceabilityStatus.FULLY_IMPLEMENTED,
                    specification_evidence=st["evidence"],
                    rtl_mapping=f"Declared and decoded state '{st_name}'.",
                    line_number=line_no,
                )
            )
        else:
            missing.append(
                TraceabilityItem(
                    requirement_id=st["id"],
                    requirement_type="FSMState",
                    name=st_name,
                    status=TraceabilityStatus.MISSING,
                    specification_evidence=st["evidence"],
                    details=f"FSM state '{st_name}' is missing from RTL.",
                )
            )

    for tr in fsm_trans_to_check:
        tr_name = tr["name"]
        cond = tr.get("condition") or ""
        # Check if condition expression or target state is present in RTL
        is_trans_present = bool(
            re.search(rf"\b{re.escape(tr_name.split('->')[-1].strip())}\b", clean_code)
        )
        if is_trans_present:
            fully_implemented.append(
                TraceabilityItem(
                    requirement_id=tr["id"],
                    requirement_type="Transition",
                    name=tr_name,
                    status=TraceabilityStatus.FULLY_IMPLEMENTED,
                    specification_evidence=tr["evidence"],
                    rtl_mapping=f"Transition to {tr_name.split('->')[-1].strip()} implemented.",
                )
            )
        else:
            missing.append(
                TraceabilityItem(
                    requirement_id=tr["id"],
                    requirement_type="Transition",
                    name=tr_name,
                    status=TraceabilityStatus.MISSING,
                    specification_evidence=tr["evidence"],
                )
            )

    # 4. Detect Unmapped RTL Elements (Signals, ports, registers not in EFS IR or plan)
    # Check declared signals in RTL
    all_decl_signals = re.finditer(
        r"\b(?:input|output|inout|wire|reg|logic)\s+(?:\[[^\]]+\]\s+)?([a-zA-Z_][a-zA-Z0-9_]*)", clean_code
    )
    for dm in all_decl_signals:
        sig_name = dm.group(1)
        if sig_name.lower() in ("clk", "clock", "rst", "rst_n", "reset", "state", "next_state", "timer_cnt"):
            continue
        if sig_name.lower() not in known_efs_names:
            line_no = None
            for idx, l in enumerate(lines, 1):
                if sig_name in l:
                    line_no = idx
                    break
            unmapped_rtl.append(
                UnmappedRTLElement(
                    element_type="Signal/Port",
                    name=sig_name,
                    line_number=line_no,
                    risk=f"Signal '{sig_name}' was declared in RTL but does not correspond to any validated requirement or signal in EFS IR.",
                )
            )

    # Calculate coverage
    total_reqs = len(fully_implemented) + len(partially_implemented) + len(missing) + len(awaiting_clarification)
    coverage = 100.0
    if total_reqs > 0:
        coverage = (len(fully_implemented) + 0.5 * len(partially_implemented)) / total_reqs * 100.0

    return TraceabilityReport(
        module_name=module_name,
        target_language=target_language,
        fully_implemented=fully_implemented,
        partially_implemented=partially_implemented,
        missing=missing,
        awaiting_clarification=awaiting_clarification,
        unmapped_rtl_elements=unmapped_rtl,
        coverage_percentage=coverage,
    )
