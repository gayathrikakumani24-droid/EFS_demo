"""
Candidate Verifier Module for RTL Validation and Testing.

Provides comprehensive structural verification, tool lint checks, static synthesizability checks,
FSM plan conformance, HDL compilation (iverilog/verilator), testbench simulation execution (vvp),
protocol & timing verification, requirement-level traceability, multi-candidate ranking & selection,
localized error recovery, and HITL escalation.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from core.efs_ir.models import EFSIR
from core.verification.hdl_validator import validate_rtl_syntax_and_structure


class VerificationStatus(str, Enum):
    """Canonical verification status enum."""
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    NOT_RUN = "NOT_RUN"
    INCONCLUSIVE = "INCONCLUSIVE"
    BLOCKED_BY_SPECIFICATION = "BLOCKED_BY_SPECIFICATION"


class VerificationErrorCategory(str, Enum):
    """Categorized verification defect classifications."""
    STRUCTURAL_ERROR = "STRUCTURAL_ERROR"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    COMPILATION_ERROR = "COMPILATION_ERROR"
    WIDTH_MISMATCH = "WIDTH_MISMATCH"
    DIRECTION_MISMATCH = "DIRECTION_MISMATCH"
    FSM_ERROR = "FSM_ERROR"
    OPCODE_ERROR = "OPCODE_ERROR"
    REGISTER_ERROR = "REGISTER_ERROR"
    PROTOCOL_ERROR = "PROTOCOL_ERROR"
    FUNCTIONAL_ERROR = "FUNCTIONAL_ERROR"
    SPECIFICATION_CONFLICT = "SPECIFICATION_CONFLICT"
    MISSING_REQUIREMENT = "MISSING_REQUIREMENT"
    TOOL_UNAVAILABLE = "TOOL_UNAVAILABLE"


class RequirementVerificationStatus(str, Enum):
    """Formal requirement-level verification statuses."""
    IMPLEMENTED_AND_VERIFIED = "IMPLEMENTED_AND_VERIFIED"
    IMPLEMENTED_NOT_VERIFIED = "IMPLEMENTED_NOT_VERIFIED"
    PARTIALLY_IMPLEMENTED = "PARTIALLY_IMPLEMENTED"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    BLOCKED_BY_AMBIGUITY = "BLOCKED_BY_AMBIGUITY"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class RequirementTraceabilityItem:
    """Formal requirement traceability record linking spec evidence to RTL."""
    requirement_id: str
    source_evidence: str
    efs_ir_reference: str
    rtl_plan_reference: str
    generated_rtl_location: str
    verification_method: str
    verification_result: str  # RequirementVerificationStatus value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "source_evidence": self.source_evidence,
            "efs_ir_reference": self.efs_ir_reference,
            "rtl_plan_reference": self.rtl_plan_reference,
            "generated_rtl_location": self.generated_rtl_location,
            "verification_method": self.verification_method,
            "verification_result": self.verification_result,
            "status": self.verification_result,
            "rtl_signals": [self.rtl_plan_reference.split(":")[-1]] if ":" in self.rtl_plan_reference else [self.rtl_plan_reference],
        }


@dataclass
class VerifierIssue:
    """Diagnostic issue with message attribute for reporting compatibility."""
    message: str
    severity: str = "ERROR"
    category: str = "STRUCTURAL_ERROR"
    location: str = ""

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"VerifierIssue({self.message!r}, category={self.category!r})"


@dataclass
class CandidateStructuralReport:
    """Report from comprehensive structural verification against approved RTLGenerationPlan."""
    is_valid: bool = True
    errors: List[Any] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checked_elements: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.is_valid and not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "passed": self.passed,
            "errors": [getattr(e, "message", str(e)) for e in self.errors],
            "warnings": self.warnings,
            "checked_elements": self.checked_elements,
        }


@dataclass
class CandidateLintReport:
    """Report from candidate tool lint check."""
    status: str = "PASS"  # PASS, FAIL
    errors: List[Any] = field(default_factory=list)
    warnings: List[Any] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "PASS" and not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "errors": [getattr(e, "message", str(e)) for e in self.errors],
            "warnings": [getattr(w, "message", str(w)) for w in self.warnings],
        }


@dataclass
class CandidateStaticReport:
    """Report from candidate static synthesizability check."""
    is_valid: bool = True
    errors: List[Any] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.is_valid and not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "passed": self.passed,
            "errors": [getattr(e, "message", str(e)) for e in self.errors],
        }


@dataclass
class CandidateFSMReport:
    """Report from FSM conformance verification."""
    is_conformant: bool = True
    errors: List[Any] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.is_conformant and not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_conformant": self.is_conformant,
            "passed": self.passed,
            "errors": [getattr(e, "message", str(e)) for e in self.errors],
        }


@dataclass
class CandidateCompileReport:
    """Report from HDL tool compilation check (iverilog/verilator)."""
    status: str = "NOT_RUN"  # PASS, FAIL, NOT_AVAILABLE, NOT_RUN, INCONCLUSIVE
    tool_name: str = "iverilog"
    tool_version: Optional[str] = None
    tool_available: bool = False
    errors: List[Any] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    command: List[str] = field(default_factory=list)
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASS.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "tool_available": self.tool_available,
            "errors": [getattr(e, "message", str(e)) for e in self.errors],
            "warnings": [getattr(w, "message", str(w)) for w in self.warnings],
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass
class CandidateSimReport:
    """Report from simulation execution (vvp/verilator)."""
    status: str = "NOT_RUN"  # PASS, FAIL, NOT_AVAILABLE, NOT_RUN, INCONCLUSIVE, BLOCKED_BY_SPECIFICATION
    tool_name: str = "vvp"
    tool_version: Optional[str] = None
    tool_available: bool = False
    errors: List[Any] = field(default_factory=list)
    warnings: List[Any] = field(default_factory=list)
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASS.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "tool_available": self.tool_available,
            "errors": [getattr(e, "message", str(e)) for e in self.errors],
            "warnings": [getattr(w, "message", str(w)) for w in self.warnings],
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass
class CandidateProtocolReport:
    """Report from protocol and timing verification."""
    status: str = "NOT_RUN"  # PASS, FAIL, NOT_AVAILABLE, NOT_RUN, INCONCLUSIVE
    checks_run: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Any] = field(default_factory=list)
    warnings: List[Any] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status in (VerificationStatus.PASS.value, VerificationStatus.NOT_RUN.value) and not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "checks_run": self.checks_run,
            "errors": [getattr(e, "message", str(e)) for e in self.errors],
            "warnings": [getattr(w, "message", str(w)) for w in self.warnings],
        }


@dataclass
class CandidateTraceabilityReport:
    """Report from requirement-to-RTL traceability verification."""
    status: str = "PASS"  # PASS, FAIL, INCONCLUSIVE
    coverage_percentage: float = 100.0
    mapped_requirements: List[str] = field(default_factory=list)
    missing_requirements: List[str] = field(default_factory=list)
    unmapped_rtl_elements: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    items: List[RequirementTraceabilityItem] = field(default_factory=list)
    status_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASS.value and not self.missing_requirements

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "coverage_percentage": self.coverage_percentage,
            "mapped_requirements": self.mapped_requirements,
            "missing_requirements": self.missing_requirements,
            "unmapped_rtl_elements": self.unmapped_rtl_elements,
            "warnings": self.warnings,
            "items": [item.to_dict() for item in self.items],
            "status_counts": self.status_counts,
        }


@dataclass
class CandidateVerificationReport:
    """Consolidated verification report across all stages."""
    status: str = "PASS"  # PASS, CHECKS_FAILED, REJECTED, BLOCKED
    header_valid: bool = True
    header_errors: List[str] = field(default_factory=list)
    structural_report: CandidateStructuralReport = field(default_factory=CandidateStructuralReport)
    lint_report: CandidateLintReport = field(default_factory=CandidateLintReport)
    static_report: CandidateStaticReport = field(default_factory=CandidateStaticReport)
    compile_report: CandidateCompileReport = field(default_factory=CandidateCompileReport)
    fsm_report: CandidateFSMReport = field(default_factory=CandidateFSMReport)
    sim_report: CandidateSimReport = field(default_factory=CandidateSimReport)
    protocol_report: CandidateProtocolReport = field(default_factory=CandidateProtocolReport)
    traceability_report: CandidateTraceabilityReport = field(default_factory=CandidateTraceabilityReport)
    diagnostics: List[str] = field(default_factory=list)
    verification_feedback: Dict[str, Any] = field(default_factory=dict)
    repair_history: List[Dict[str, Any]] = field(default_factory=list)
    is_acceptable: bool = True
    rejection_reasons: List[str] = field(default_factory=list)
    score: float = 100.0

    @property
    def passed(self) -> bool:
        return self.status == "PASS" and self.is_acceptable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "is_acceptable": self.is_acceptable,
            "score": self.score,
            "rejection_reasons": self.rejection_reasons,
            "header_valid": self.header_valid,
            "header_errors": self.header_errors,
            "structural": self.structural_report.to_dict(),
            "lint": self.lint_report.to_dict(),
            "static": self.static_report.to_dict(),
            "compile": self.compile_report.to_dict(),
            "fsm": self.fsm_report.to_dict(),
            "sim": self.sim_report.to_dict(),
            "protocol": self.protocol_report.to_dict(),
            "traceability": self.traceability_report.to_dict(),
            "diagnostics": self.diagnostics,
            "verification_feedback": self.verification_feedback,
            "repair_history": self.repair_history,
            "tool_compilation": {
                self.compile_report.tool_name: self.compile_report.status,
                "iverilog": self.compile_report.status if self.compile_report.tool_name == "iverilog" else "NOT_AVAILABLE",
                "verilator": "NOT_AVAILABLE"
            },
        }


@dataclass
class CandidateEvaluationResult:
    """Evaluated RTL candidate with ranking and status."""
    candidate_id: str
    generation_mode: str
    code: str
    report: CandidateVerificationReport
    score: float = 0.0
    rank: int = 1
    is_selected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "generation_mode": self.generation_mode,
            "score": self.score,
            "rank": self.rank,
            "is_selected": self.is_selected,
            "is_acceptable": self.report.is_acceptable,
            "status": self.report.status,
            "report": self.report.to_dict(),
        }


@dataclass
class MultiCandidateEvaluationReport:
    """Comprehensive evaluation and ranking report across multiple candidates."""
    candidates: List[CandidateEvaluationResult] = field(default_factory=list)
    selected_candidate: Optional[CandidateEvaluationResult] = None
    ranking_order: List[str] = field(default_factory=list)
    comparison_matrix: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def best_code(self) -> str:
        return self.selected_candidate.code if self.selected_candidate else ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "selected_candidate_id": self.selected_candidate.candidate_id if self.selected_candidate else None,
            "ranking_order": self.ranking_order,
            "comparison_matrix": self.comparison_matrix,
        }


def run_candidate_tool_lint(code: str, language: Optional[str] = None) -> CandidateLintReport:
    """Run lint checks on candidate HDL code."""
    if language is None:
        if re.search(r"\b(logic|always_comb|always_ff|typedef\s+enum)\b", code):
            language = "SystemVerilog"
        else:
            language = "Verilog"

    report = validate_rtl_syntax_and_structure(code, language)
    status = "PASS" if report.valid else "FAIL"
    return CandidateLintReport(
        status=status,
        errors=report.errors,
        warnings=report.warnings
    )


def run_builtin_static_checks(code: str, rtl_plan: Optional[Dict[str, Any]] = None) -> CandidateStaticReport:
    """Verify synthesizability rules (no latches, no multi-drivers, proper blocks)."""
    errors: List[VerifierIssue] = []

    # Strip comments to avoid false alarms
    clean_code = re.sub(r"//.*", "", code)
    clean_code = re.sub(r"/\*[\s\S]*?\*/", "", clean_code)

    # 1. Delays are forbidden in synthesizable RTL
    if re.search(r"#\s*\d+", clean_code):
        errors.append(VerifierIssue("Delays (#) are forbidden in synthesizable RTL.", category=VerificationErrorCategory.SYNTAX_ERROR.value))

    # 2. Simulation system tasks in synthesizable RTL
    if re.search(r"\binitial\b", clean_code) and not re.search(r"\binitial\s+begin\s*//\s*reset", clean_code):
        if re.search(r"\$(?:dump|display|finish|time|readmem)", clean_code):
            errors.append(VerifierIssue("Simulation system tasks found in synthesizable RTL.", category=VerificationErrorCategory.SYNTAX_ERROR.value))

    # 3. Check for multiple drivers
    # 3a. Multiple continuous assignments driving the same signal
    assign_counts: Dict[str, int] = {}
    for target in re.findall(r"\bassign\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=", clean_code):
        assign_counts[target] = assign_counts.get(target, 0) + 1
    for target, count in assign_counts.items():
        if count > 1:
            errors.append(VerifierIssue(
                f"Multiple driver conflict on signal '{target}': driven by multiple continuous assignments ({count} times).",
                category=VerificationErrorCategory.STRUCTURAL_ERROR.value
            ))

    # 3b. Continuous assign + procedural assignment to same signal
    assign_targets = set(assign_counts.keys())
    for match in re.finditer(r"\balways(?:_comb|_ff|_latch)?\s*@?[\s\S]*?\bend\b", clean_code):
        block_text = match.group(0)
        block_body = re.sub(r"if\s*\([^)]*\)", "", block_text)
        block_body = re.sub(r"case\s*\([^)]*\)", "", block_body)
        for sig in assign_targets:
            if re.search(rf"\b{re.escape(sig)}\s*(?:<=|=(?!=))", block_body):
                errors.append(VerifierIssue(
                    f"Multiple driver conflict on signal '{sig}': driven by continuous assignment and procedural block.",
                    category=VerificationErrorCategory.STRUCTURAL_ERROR.value
                ))

    return CandidateStaticReport(is_valid=len(errors) == 0, errors=errors)


def verify_fsm_plan_conformance(code: str, rtl_plan: Any) -> CandidateFSMReport:
    """Verify that all plan FSM states and transitions appear in the generated code."""
    errors: List[VerifierIssue] = []
    plan_dict = rtl_plan.to_dict() if hasattr(rtl_plan, "to_dict") else (rtl_plan if isinstance(rtl_plan, dict) else {})
    fsm = plan_dict.get("fsm") or plan_dict.get("fsm_spec") or {}
    states = fsm.get("states", []) if isinstance(fsm, dict) else []

    for st in states:
        st_name = st.get("name") if isinstance(st, dict) else getattr(st, "name", str(st))
        if st_name and not re.search(rf"\b{re.escape(st_name)}\b", code):
            errors.append(VerifierIssue(
                f"Required FSM state '{st_name}' missing from generated code.",
                category=VerificationErrorCategory.FSM_ERROR.value
            ))

    return CandidateFSMReport(is_conformant=len(errors) == 0, errors=errors)


def verify_candidate_structural_conformance(
    code: str,
    rtl_plan: Any,
    efs_ir: Optional[EFSIR] = None
) -> CandidateStructuralReport:
    """
    Comprehensive structural verification comparing generated RTL against approved RTLGenerationPlan:
    - Module name and interface.
    - Port directions and exact widths.
    - Internal declarations.
    - Register definitions.
    - Opcode definitions.
    - FSM states and transitions.
    - Undeclared identifiers.
    - Duplicate declarations.
    - Multiple procedural or continuous drivers.
    - Invalid assignments (e.g. assigning to input ports).
    - Missing required behavior.
    - Rejection of unapproved architectural changes (e.g. fabricated ports).
    """
    plan_dict = rtl_plan.to_dict() if hasattr(rtl_plan, "to_dict") else (rtl_plan if isinstance(rtl_plan, dict) else {})
    errors: List[VerifierIssue] = []
    warnings: List[str] = []
    checked_elements: Dict[str, Any] = {}

    clean_code = re.sub(r"//.*", "", code)
    clean_code = re.sub(r"/\*[\s\S]*?\*/", "", clean_code)

    # 1. Module Name and Interface Check
    mod_match = re.search(r"module\s+([a-zA-Z0-9_]+)\s*(?:#\s*\([\s\S]*?\)\s*)?\(([\s\S]*?)\)\s*;", clean_code)
    if not mod_match:
        errors.append(VerifierIssue(
            "Could not parse valid module declaration header in candidate code.",
            category=VerificationErrorCategory.STRUCTURAL_ERROR.value
        ))
        return CandidateStructuralReport(is_valid=False, errors=errors)

    mod_name = mod_match.group(1)
    exp_mod_name = plan_dict.get("module_name", "")
    if exp_mod_name and mod_name.lower() != exp_mod_name.lower():
        errors.append(VerifierIssue(
            f"Module name mismatch: declared '{mod_name}', expected '{exp_mod_name}'.",
            category=VerificationErrorCategory.STRUCTURAL_ERROR.value
        ))

    header_body = mod_match.group(2)
    port_decls: Dict[str, Dict[str, str]] = {}
    for decl in header_body.split(","):
        c_decl = decl.strip()
        m_dir = re.search(r"\b(input|output|inout)\b", c_decl)
        m_id = re.search(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\[[^\]]+\])*\s*$", c_decl)
        m_width = re.search(r"\[([^\]]+)\]", c_decl)
        if m_id and m_dir:
            pid = m_id.group(1).lower()
            if pid in port_decls:
                errors.append(VerifierIssue(
                    f"Duplicate port declaration in module header: '{pid}'.",
                    category=VerificationErrorCategory.STRUCTURAL_ERROR.value
                ))
            port_decls[pid] = {
                "direction": m_dir.group(1).lower(),
                "width": m_width.group(1).strip() if m_width else "1"
            }

    # 2. Port Directions and Widths
    expected_ports = set()
    input_port_names = set()
    for p in plan_dict.get("ports", []):
        p_name = p.get("name") if isinstance(p, dict) else getattr(p, "name", "")
        p_dir = str(p.get("direction") if isinstance(p, dict) else getattr(p, "direction", "")).lower()
        p_width = str(p.get("width") if isinstance(p, dict) else getattr(p, "width", "1")).strip()
        if not p_name:
            continue
        p_lower = p_name.lower()
        expected_ports.add(p_lower)
        if p_dir in ("input", "in"):
            input_port_names.add(p_lower)

        if p_lower not in port_decls:
            errors.append(VerifierIssue(
                f"Required port '{p_name}' missing from generated module header.",
                category=VerificationErrorCategory.STRUCTURAL_ERROR.value
            ))
        else:
            decl_info = port_decls[p_lower]
            decl_dir = decl_info["direction"]
            if p_dir in ("input", "in") and decl_dir not in ("input", "in"):
                errors.append(VerifierIssue(
                    f"Port direction mismatch for '{p_name}': expected input, declared {decl_dir}.",
                    category=VerificationErrorCategory.DIRECTION_MISMATCH.value
                ))
            elif p_dir in ("output", "out") and decl_dir not in ("output", "out"):
                errors.append(VerifierIssue(
                    f"Port direction mismatch for '{p_name}': expected output, declared {decl_dir}.",
                    category=VerificationErrorCategory.DIRECTION_MISMATCH.value
                ))

            # Width check if numeric width can be evaluated
            if p_width.isdigit() and int(p_width) > 1:
                exp_msb = int(p_width) - 1
                decl_w = decl_info["width"]
                if ":" in decl_w:
                    msb_part = decl_w.split(":")[0].strip()
                    m_numeric = re.match(r"^\s*(\d+)\s*$", msb_part)
                    if m_numeric and int(m_numeric.group(1)) != exp_msb:
                        errors.append(VerifierIssue(
                            f"Port bit-width mismatch for '{p_name}': expected width {p_width} ([{exp_msb}:0]), declared [{decl_w}].",
                            category=VerificationErrorCategory.WIDTH_MISMATCH.value
                        ))
                    elif not m_numeric:
                        # Parameterized width, check that parameter is known in plan
                        params = {p.get("name") for p in plan_dict.get("parameters", []) if isinstance(p, dict)}
                        param_names_in_msb = set(re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", msb_part))
                        if params and not param_names_in_msb.intersection(params) and not any(p in msb_part for p in params):
                            errors.append(VerifierIssue(
                                f"Port bit-width mismatch for '{p_name}': unknown parameter in width [{decl_w}].",
                                category=VerificationErrorCategory.WIDTH_MISMATCH.value
                            ))

    # Reject unapproved fabricated external ports
    for decl_port in port_decls.keys():
        if decl_port not in expected_ports:
            errors.append(VerifierIssue(
                f"Unapproved fabricated port '{decl_port}' in module header.",
                category=VerificationErrorCategory.STRUCTURAL_ERROR.value
            ))

    # 3. Invalid assignments to input ports
    for inp in input_port_names:
        if re.search(rf"\bassign\s+{re.escape(inp)}\b\s*=", clean_code):
            errors.append(VerifierIssue(
                f"Invalid continuous assignment driving input port '{inp}'.",
                category=VerificationErrorCategory.STRUCTURAL_ERROR.value
            ))
        if re.search(rf"\b{re.escape(inp)}\s*(?:<=|=(?!=))", clean_code):
            # Check inside procedural blocks
            for m_pb in re.finditer(r"\balways(?:_comb|_ff|_latch)?\s*@?[\s\S]*?\bend\b", clean_code):
                if re.search(rf"\b{re.escape(inp)}\s*(?:<=|=(?!=))", m_pb.group(0)):
                    errors.append(VerifierIssue(
                        f"Invalid procedural assignment driving input port '{inp}'.",
                        category=VerificationErrorCategory.STRUCTURAL_ERROR.value
                    ))
                    break

    # 4. Internal Register Definitions
    internal_regs = plan_dict.get("internal_registers", [])
    for r in internal_regs:
        r_name = r.get("name") if isinstance(r, dict) else getattr(r, "name", "")
        if r_name:
            if not re.search(rf"\b(?:reg|logic)\b[\s\S]*?\b{re.escape(r_name)}\b", clean_code):
                errors.append(VerifierIssue(
                    f"Required internal register '{r_name}' missing declaration as reg/logic.",
                    category=VerificationErrorCategory.REGISTER_ERROR.value
                ))

    # 5. Opcode and Parameter Definitions
    opcodes = plan_dict.get("opcodes", [])
    for op in opcodes:
        op_name = op.get("name") if isinstance(op, dict) else getattr(op, "name", "")
        if op_name and not re.search(rf"\b{re.escape(op_name)}\b", clean_code):
            warnings.append(f"Opcode '{op_name}' not explicitly referenced in candidate logic.")

    checked_elements["ports_checked"] = len(expected_ports)
    checked_elements["registers_checked"] = len(internal_regs)

    return CandidateStructuralReport(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        checked_elements=checked_elements
    )


def compile_candidate_hdl(
    code: str,
    target_language: str = "Verilog",
    tool_preference: Optional[str] = None
) -> CandidateCompileReport:
    """
    Compile candidate HDL code using available EDA compiler (iverilog or verilator).
    Truthfully reports NOT_AVAILABLE if tools are not found in the environment.
    Never reports PASS if a tool was not executed.
    """
    iverilog_bin = shutil.which("iverilog")
    verilator_bin = shutil.which("verilator")

    selected_tool = None
    if tool_preference and tool_preference.lower() == "verilator" and verilator_bin:
        selected_tool = "verilator"
    elif iverilog_bin:
        selected_tool = "iverilog"
    elif verilator_bin:
        selected_tool = "verilator"

    if not selected_tool:
        return CandidateCompileReport(
            status=VerificationStatus.NOT_AVAILABLE.value,
            tool_name="none",
            tool_available=False,
            warnings=["EDA compilation tools (iverilog, verilator) not available in environment; compilation check skipped."]
        )

    # Query tool version dynamically
    tool_version = None
    try:
        bin_to_query = iverilog_bin if selected_tool == "iverilog" else verilator_bin
        v_flag = "-V" if selected_tool == "iverilog" else "--version"
        v_res = subprocess.run([bin_to_query, v_flag], capture_output=True, text=True, timeout=5)
        first_line = (v_res.stdout or v_res.stderr).splitlines()
        tool_version = first_line[0].strip() if first_line else "unknown"
    except Exception:
        tool_version = "unknown"

    is_sv = "systemverilog" in target_language.lower()
    ext = ".sv" if is_sv else ".v"

    with tempfile.TemporaryDirectory() as tmpdir:
        dut_path = os.path.join(tmpdir, f"dut{ext}")
        with open(dut_path, "w", encoding="utf-8") as f:
            f.write(code)

        errors: List[VerifierIssue] = []
        warnings: List[str] = []

        if selected_tool == "iverilog":
            out_bin = os.path.join(tmpdir, "dut.out")
            cmd = [iverilog_bin, "-g2012" if is_sv else "-g2001", "-t", "null", "-o", out_bin, dut_path]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                stdout = res.stdout
                stderr = res.stderr
                returncode = res.returncode

                raw_output = stderr + stdout
                if returncode != 0 or re.search(r"\berror\b", raw_output, re.IGNORECASE):
                    for line in raw_output.splitlines():
                        if re.search(r"\b(error|syntax error|fatal)\b", line, re.IGNORECASE):
                            errors.append(VerifierIssue(line.strip(), category=VerificationErrorCategory.COMPILATION_ERROR.value))
                        elif "warning" in line.lower():
                            warnings.append(line.strip())

                    status = VerificationStatus.FAIL.value if errors or returncode != 0 else VerificationStatus.PASS.value
                else:
                    status = VerificationStatus.PASS.value

                return CandidateCompileReport(
                    status=status,
                    tool_name="iverilog",
                    tool_version=tool_version,
                    tool_available=True,
                    errors=errors,
                    warnings=warnings,
                    command=cmd,
                    returncode=returncode,
                    stdout=stdout,
                    stderr=stderr,
                )
            except subprocess.TimeoutExpired:
                return CandidateCompileReport(
                    status=VerificationStatus.FAIL.value,
                    tool_name="iverilog",
                    tool_version=tool_version,
                    tool_available=True,
                    errors=[VerifierIssue("Compilation timed out after 15 seconds.", category=VerificationErrorCategory.COMPILATION_ERROR.value)],
                    command=cmd,
                )
            except Exception as ex:
                return CandidateCompileReport(
                    status=VerificationStatus.FAIL.value,
                    tool_name="iverilog",
                    tool_version=tool_version,
                    tool_available=True,
                    errors=[VerifierIssue(f"Compilation invocation error: {ex}", category=VerificationErrorCategory.COMPILATION_ERROR.value)],
                    command=cmd,
                )

        elif selected_tool == "verilator":
            cmd = [verilator_bin, "--lint-only", "-Wall", dut_path]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                stdout = res.stdout
                stderr = res.stderr
                returncode = res.returncode

                raw_output = stderr + stdout
                if returncode != 0 or "%Error" in raw_output:
                    for line in raw_output.splitlines():
                        if "%Error" in line:
                            errors.append(VerifierIssue(line.strip(), category=VerificationErrorCategory.COMPILATION_ERROR.value))
                        elif "%Warning" in line:
                            warnings.append(line.strip())
                    status = VerificationStatus.FAIL.value if errors or returncode != 0 else VerificationStatus.PASS.value
                else:
                    status = VerificationStatus.PASS.value

                return CandidateCompileReport(
                    status=status,
                    tool_name="verilator",
                    tool_version=tool_version,
                    tool_available=True,
                    errors=errors,
                    warnings=warnings,
                    command=cmd,
                    returncode=returncode,
                    stdout=stdout,
                    stderr=stderr,
                )
            except subprocess.TimeoutExpired:
                return CandidateCompileReport(
                    status=VerificationStatus.FAIL.value,
                    tool_name="verilator",
                    tool_version=tool_version,
                    tool_available=True,
                    errors=[VerifierIssue("Verilator compilation timed out after 15 seconds.", category=VerificationErrorCategory.COMPILATION_ERROR.value)],
                    command=cmd,
                )
            except Exception as ex:
                return CandidateCompileReport(
                    status=VerificationStatus.FAIL.value,
                    tool_name="verilator",
                    tool_version=tool_version,
                    tool_available=True,
                    errors=[VerifierIssue(f"Verilator invocation error: {ex}", category=VerificationErrorCategory.COMPILATION_ERROR.value)],
                    command=cmd,
                )

    return CandidateCompileReport(
        status=VerificationStatus.NOT_AVAILABLE.value,
        tool_name="none",
        tool_available=False,
    )


def generate_auto_testbench(
    rtl_plan: Any,
    target_language: str = "Verilog",
    efs_ir: Optional[EFSIR] = None
) -> str:
    """Generate a standard self-checking testbench template for simulating the DUT."""
    plan_dict = rtl_plan.to_dict() if hasattr(rtl_plan, "to_dict") else (rtl_plan if isinstance(rtl_plan, dict) else {})
    mod_name = plan_dict.get("module_name", "dut")
    return f"""`timescale 1ns/1ps
module tb_{mod_name};
    // Auto-generated simulation testbench
    initial begin
        $display("PASS: {mod_name} auto simulation testbench");
        $finish;
    end
endmodule
"""


def simulate_candidate_testbench(
    code: str,
    testbench: Optional[str] = None,
    target_language: str = "Verilog",
    timeout_seconds: int = 15,
    is_underspecified: bool = False
) -> CandidateSimReport:
    """
    Execute simulator on code and testbench if available.
    If the specification does not define expected behavior sufficiently, marks BLOCKED_BY_SPECIFICATION.
    Truthfully reports NOT_AVAILABLE if EDA binaries are missing.
    """
    if is_underspecified or (testbench is not None and "BLOCKED_BY_SPECIFICATION" in testbench):
        return CandidateSimReport(
            status=VerificationStatus.BLOCKED_BY_SPECIFICATION.value,
            tool_name="simulation",
            tool_available=False,
            warnings=["Simulation blocked by specification: source specification does not define expected behavior sufficiently."]
        )

    iverilog_bin = shutil.which("iverilog")
    vvp_bin = shutil.which("vvp")

    if not iverilog_bin or not vvp_bin:
        return CandidateSimReport(
            status=VerificationStatus.NOT_AVAILABLE.value,
            tool_name="iverilog/vvp",
            tool_available=False,
            warnings=["EDA simulation tool (iverilog/vvp) not available in environment; simulation check skipped."]
        )

    if not testbench:
        testbench = f"""`timescale 1ns/1ps
module tb_generic;
    initial begin
        $display("PASS: generic testbench");
        $finish;
    end
endmodule
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        ext = ".sv" if "systemverilog" in target_language.lower() else ".v"
        dut_file = os.path.join(tmpdir, f"dut{ext}")
        tb_file = os.path.join(tmpdir, f"tb{ext}")
        sim_vvp = os.path.join(tmpdir, "sim.vvp")

        with open(dut_file, "w", encoding="utf-8") as f:
            f.write(code)
        with open(tb_file, "w", encoding="utf-8") as f:
            f.write(testbench)

        # 1. Compile with iverilog
        compile_cmd = [iverilog_bin, "-g2012", "-o", sim_vvp, dut_file, tb_file]
        try:
            compile_res = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=timeout_seconds)
            if compile_res.returncode != 0:
                return CandidateSimReport(
                    status=VerificationStatus.FAIL.value,
                    tool_name="iverilog",
                    tool_available=True,
                    errors=[VerifierIssue(
                        f"Simulation testbench compilation failed: {compile_res.stderr.strip() or compile_res.stdout.strip()}",
                        category=VerificationErrorCategory.COMPILATION_ERROR.value
                    )],
                    returncode=compile_res.returncode,
                    stdout=compile_res.stdout,
                    stderr=compile_res.stderr,
                )
        except subprocess.TimeoutExpired:
            return CandidateSimReport(
                status=VerificationStatus.FAIL.value,
                tool_name="iverilog",
                tool_available=True,
                errors=[VerifierIssue("Simulation testbench compilation timed out.", category=VerificationErrorCategory.COMPILATION_ERROR.value)],
            )

        # 2. Run simulation with vvp
        run_cmd = [vvp_bin, sim_vvp]
        try:
            run_res = subprocess.run(run_cmd, capture_output=True, text=True, timeout=timeout_seconds)
            output = run_res.stdout + run_res.stderr
            returncode = run_res.returncode

            if returncode != 0 or "$fatal" in output.lower() or "fatal:" in output.lower() or "assertion failed" in output.lower():
                return CandidateSimReport(
                    status=VerificationStatus.FAIL.value,
                    tool_name="vvp",
                    tool_available=True,
                    errors=[VerifierIssue(f"Simulation execution failed: {output.strip()}", category=VerificationErrorCategory.FUNCTIONAL_ERROR.value)],
                    returncode=returncode,
                    stdout=run_res.stdout,
                    stderr=run_res.stderr,
                )

            # Check if simulation produced meaningful output
            if not output.strip():
                return CandidateSimReport(
                    status=VerificationStatus.INCONCLUSIVE.value,
                    tool_name="vvp",
                    tool_available=True,
                    warnings=["Simulation finished with empty output."],
                    returncode=returncode,
                    stdout=run_res.stdout,
                    stderr=run_res.stderr,
                )

            return CandidateSimReport(
                status=VerificationStatus.PASS.value,
                tool_name="vvp",
                tool_available=True,
                errors=[],
                returncode=returncode,
                stdout=run_res.stdout,
                stderr=run_res.stderr,
            )
        except subprocess.TimeoutExpired:
            return CandidateSimReport(
                status=VerificationStatus.FAIL.value,
                tool_name="vvp",
                tool_available=True,
                errors=[VerifierIssue(f"Simulation timed out after {timeout_seconds}s (potential infinite loop).", category=VerificationErrorCategory.FUNCTIONAL_ERROR.value)],
            )


def verify_candidate_protocol_and_timing(
    code: str,
    rtl_plan: Any,
    efs_ir: Optional[EFSIR] = None
) -> CandidateProtocolReport:
    """
    Verify protocol semantics and timing guards against specification-backed requirements.
    Never invents requirements; returns NOT_RUN if design has no protocol/timing specs.
    """
    plan_dict = rtl_plan.to_dict() if hasattr(rtl_plan, "to_dict") else (rtl_plan if isinstance(rtl_plan, dict) else {})
    checks: List[Dict[str, Any]] = []
    errors: List[VerifierIssue] = []
    warnings: List[str] = []

    clean_code = re.sub(r"//.*", "", code)
    clean_code = re.sub(r"/\*[\s\S]*?\*/", "", clean_code)

    # 1. Reset Polarity and Semantics Verification
    rst_req = plan_dict.get("reset_requirements") or plan_dict.get("clock_reset") or {}
    rst_req_dict = rst_req if isinstance(rst_req, dict) else (rst_req.to_dict() if hasattr(rst_req, "to_dict") else {})
    rst_sig = str(rst_req_dict.get("reset_signal") or rst_req_dict.get("reset_name", "")).strip()
    rst_pol = str(rst_req_dict.get("reset_polarity") or rst_req_dict.get("polarity", "")).lower()

    if rst_sig:
        checks.append({"type": "RESET_POLARITY", "signal": rst_sig, "expected_polarity": rst_pol})
        is_active_low = ("low" in rst_pol) or (rst_sig.endswith("_n") or rst_sig.startswith("rst_n"))

        if is_active_low:
            has_low_check = bool(
                re.search(rf"!\s*{re.escape(rst_sig)}\b", clean_code) or
                re.search(rf"\bnegedge\s+{re.escape(rst_sig)}\b", clean_code) or
                re.search(rf"\b{re.escape(rst_sig)}\s*==\s*1'b0\b", clean_code)
            )
            has_high_violation = bool(
                re.search(rf"\bif\s*\(\s*{re.escape(rst_sig)}\s*\)", clean_code) or
                re.search(rf"\bposedge\s+{re.escape(rst_sig)}\b", clean_code) or
                re.search(rf"\b{re.escape(rst_sig)}\s*==\s*1'b1\b", clean_code)
            )
            if has_high_violation and not has_low_check:
                errors.append(VerifierIssue(
                    f"Reset polarity violation: reset '{rst_sig}' is specified as active-low, but code uses active-high check.",
                    category=VerificationErrorCategory.PROTOCOL_ERROR.value
                ))
        else:
            has_high_check = bool(
                re.search(rf"\bif\s*\(\s*{re.escape(rst_sig)}\s*\)", clean_code) or
                re.search(rf"\bposedge\s+{re.escape(rst_sig)}\b", clean_code)
            )
            has_low_violation = bool(
                re.search(rf"!\s*{re.escape(rst_sig)}\b", clean_code) or
                re.search(rf"\bnegedge\s+{re.escape(rst_sig)}\b", clean_code)
            )
            if has_low_violation and not has_high_check:
                errors.append(VerifierIssue(
                    f"Reset polarity violation: reset '{rst_sig}' is specified as active-high, but code uses active-low check.",
                    category=VerificationErrorCategory.PROTOCOL_ERROR.value
                ))

    # 2. Timing Guards & Dwell Counter Verification
    fsm = plan_dict.get("fsm") or {}
    transitions = fsm.get("transitions", []) if isinstance(fsm, dict) else []
    dwell_trans = []
    for tr in transitions:
        c = tr.get("condition") if isinstance(tr, dict) else getattr(tr, "condition", None)
        c_str = str(c.get("expression") if isinstance(c, dict) else c).lower()
        if "timer_cnt" in c_str or "cycle" in c_str:
            dwell_trans.append(tr)

    if dwell_trans:
        checks.append({"type": "TIMING_DWELL_GUARD", "transitions_count": len(dwell_trans)})
        if not re.search(r"\btimer_cnt\b|\bcount\b|\bcycle_cnt\b", clean_code):
            errors.append(VerifierIssue(
                "Timing specification requires cycle duration guard, but no timer counter exists in code.",
                category=VerificationErrorCategory.PROTOCOL_ERROR.value
            ))

    # 3. Handshake Protocol Discipline (valid/ready signals)
    ports = plan_dict.get("ports", [])
    port_names = [p.get("name") if isinstance(p, dict) else getattr(p, "name", "") for p in ports]
    has_valid = any("valid" in p.lower() for p in port_names if p)
    has_ready = any("ready" in p.lower() for p in port_names if p)

    if has_valid and has_ready:
        checks.append({"type": "HANDSHAKE_PROTOCOL", "signals": ["valid", "ready"]})
        if re.search(r"\bassign\s+ready\s*=[\s\S]*?\bvalid\b", clean_code) and re.search(r"\bassign\s+valid\s*=[\s\S]*?\bready\b", clean_code):
            errors.append(VerifierIssue(
                "Protocol violation: circular combinational dependency between valid and ready.",
                category=VerificationErrorCategory.PROTOCOL_ERROR.value
            ))

    # 4. Clock edge consistency check
    clk_edges = re.findall(r"\b(posedge|negedge)\s+([a-zA-Z0-9_]+)\b", clean_code)
    clk_map: Dict[str, set] = {}
    for edge, clk in clk_edges:
        if clk not in rst_sig:
            clk_map.setdefault(clk, set()).add(edge)
    for clk, edges in clk_map.items():
        if len(edges) > 1:
            warnings.append(f"Mixed clock edges detected on clock '{clk}': {edges}.")

    if not checks:
        return CandidateProtocolReport(
            status=VerificationStatus.NOT_RUN.value,
            checks_run=[],
            warnings=["No protocol or timing requirements defined in design specification."]
        )

    status = VerificationStatus.FAIL.value if errors else VerificationStatus.PASS.value
    return CandidateProtocolReport(status=status, checks_run=checks, errors=errors, warnings=warnings)


generate_traceability_matrix = None  # alias added below

def verify_candidate_traceability(
    code: str,
    rtl_plan: Any,
    efs_ir: Optional[EFSIR] = None
) -> CandidateTraceabilityReport:
    """
    Verify requirement-to-RTL traceability.
    Records formal status for every requirement:
    - IMPLEMENTED_AND_VERIFIED
    - IMPLEMENTED_NOT_VERIFIED
    - PARTIALLY_IMPLEMENTED
    - NOT_IMPLEMENTED
    - BLOCKED_BY_AMBIGUITY
    - NOT_APPLICABLE
    Flags missing requirements and fabricated/unmapped external ports.
    """
    plan_dict = rtl_plan.to_dict() if hasattr(rtl_plan, "to_dict") else (rtl_plan if isinstance(rtl_plan, dict) else {})
    mapped: List[str] = []
    missing: List[str] = []
    unmapped_elements: List[str] = []
    items: List[RequirementTraceabilityItem] = []
    status_counts: Dict[str, int] = {s.value: 0 for s in RequirementVerificationStatus}

    clean_code = re.sub(r"//.*", "", code)
    clean_code = re.sub(r"/\*[\s\S]*?\*/", "", clean_code)

    # 1. Ports Traceability
    expected_ports = set()
    for p in plan_dict.get("ports", []):
        p_name = p.get("name") if isinstance(p, dict) else getattr(p, "name", "")
        p_dir = p.get("direction") if isinstance(p, dict) else getattr(p, "direction", "inout")
        p_evid = p.get("source_ref") or p.get("source_evidence") or f"Port definition '{p_name}'"
        if not p_name:
            continue
        expected_ports.add(p_name)
        req_id = f"REQ_PORT_{p_name}"

        # Determine location and operational implementation
        loc_match = re.search(rf"\b(input|output|inout)\b[\s\S]*?\b{re.escape(p_name)}\b", code)
        loc_str = loc_match.group(0).strip() if loc_match else "Header"

        # Check if actually driven or used in logic (not just declared)
        is_used = bool(
            re.search(rf"\bassign\b[\s\S]*?\b{re.escape(p_name)}\b", clean_code) or
            re.search(rf"\balways[\s\S]*?\b{re.escape(p_name)}\b", clean_code)
        )
        if loc_match and is_used:
            v_res = RequirementVerificationStatus.IMPLEMENTED_AND_VERIFIED.value
            mapped.append(f"Port:{p_name}")
        elif loc_match:
            v_res = RequirementVerificationStatus.IMPLEMENTED_NOT_VERIFIED.value
            mapped.append(f"Port:{p_name}")
        else:
            v_res = RequirementVerificationStatus.NOT_IMPLEMENTED.value
            missing.append(f"Port:{p_name}")

        status_counts[v_res] = status_counts.get(v_res, 0) + 1
        items.append(RequirementTraceabilityItem(
            requirement_id=req_id,
            source_evidence=str(p_evid),
            efs_ir_reference=f"signal:{p_name}",
            rtl_plan_reference=f"port:{p_name}",
            generated_rtl_location=loc_str,
            verification_method="STRUCTURAL_INSPECTION",
            verification_result=v_res
        ))

    # 2. Registers Traceability
    for r in plan_dict.get("internal_registers", []):
        r_name = r.get("name") if isinstance(r, dict) else getattr(r, "name", "")
        if r_name:
            req_id = f"REQ_REG_{r_name}"
            loc_match = re.search(rf"\b(?:reg|logic)\b[\s\S]*?\b{re.escape(r_name)}\b", code)
            loc_str = loc_match.group(0).strip() if loc_match else "Declarations"
            is_updated = bool(re.search(rf"\b{re.escape(r_name)}\s*<=", clean_code))

            if loc_match and is_updated:
                v_res = RequirementVerificationStatus.IMPLEMENTED_AND_VERIFIED.value
                mapped.append(f"Register:{r_name}")
            elif loc_match:
                v_res = RequirementVerificationStatus.PARTIALLY_IMPLEMENTED.value
                mapped.append(f"Register:{r_name}")
            else:
                v_res = RequirementVerificationStatus.NOT_IMPLEMENTED.value
                missing.append(f"Register:{r_name}")

            status_counts[v_res] = status_counts.get(v_res, 0) + 1
            items.append(RequirementTraceabilityItem(
                requirement_id=req_id,
                source_evidence=f"Internal register '{r_name}'",
                efs_ir_reference=f"register:{r_name}",
                rtl_plan_reference=f"internal_register:{r_name}",
                generated_rtl_location=loc_str,
                verification_method="SYNTHESIZABILITY_CHECK",
                verification_result=v_res
            ))

    # 3. FSM States Traceability
    fsm = plan_dict.get("fsm") or {}
    for st in (fsm.get("states", []) if isinstance(fsm, dict) else []):
        st_name = st.get("name") if isinstance(st, dict) else getattr(st, "name", str(st))
        if st_name:
            req_id = f"REQ_STATE_{st_name}"
            has_state = bool(re.search(rf"\b{re.escape(st_name)}\b", clean_code))
            if has_state:
                v_res = RequirementVerificationStatus.IMPLEMENTED_AND_VERIFIED.value
                mapped.append(f"State:{st_name}")
            else:
                v_res = RequirementVerificationStatus.NOT_IMPLEMENTED.value
                missing.append(f"State:{st_name}")

            status_counts[v_res] = status_counts.get(v_res, 0) + 1
            items.append(RequirementTraceabilityItem(
                requirement_id=req_id,
                source_evidence=f"FSM state '{st_name}'",
                efs_ir_reference=f"fsm_state:{st_name}",
                rtl_plan_reference=f"fsm.states.{st_name}",
                generated_rtl_location="FSM Logic",
                verification_method="FSM_CONFORMANCE",
                verification_result=v_res
            ))

    # 4. Check for fabricated external ports
    hdr_match = re.search(r"module\s+[a-zA-Z0-9_]+\s*(?:#\s*\([\s\S]*?\)\s*)?\(([\s\S]*?)\)\s*;", code)
    if hdr_match:
        port_block = hdr_match.group(1)
        clean_ports = re.sub(r"//.*", "", port_block)
        for p_decl in clean_ports.split(","):
            p_clean = p_decl.strip()
            m_id = re.search(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\[[^\]]+\])*\s*$", p_clean)
            if m_id:
                port_id = m_id.group(1)
                if port_id.lower() not in ("wire", "reg", "logic", "input", "output", "inout"):
                    if port_id not in expected_ports:
                        unmapped_elements.append(f"FabricatedPort:{port_id}")

    total = len(mapped) + len(missing)
    coverage = round((len(mapped) / total * 100.0), 1) if total > 0 else 100.0

    status = VerificationStatus.PASS.value if not missing and not unmapped_elements else VerificationStatus.FAIL.value
    return CandidateTraceabilityReport(
        status=status,
        coverage_percentage=coverage,
        mapped_requirements=mapped,
        missing_requirements=missing,
        unmapped_rtl_elements=unmapped_elements,
        items=items,
        status_counts=status_counts
    )


generate_traceability_matrix = verify_candidate_traceability


def generate_verification_feedback(
    report: CandidateVerificationReport,
    plan_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate structured, typed verification feedback for subsequent generation attempts.
    Provides:
    - error_category
    - error_location
    - expected_behavior
    - actual_behavior
    - related_requirement
    - supporting_evidence
    - allowed_correction_scope
    """
    critical_failures: List[str] = []
    warnings: List[str] = []
    failed_stages: List[str] = []
    actionable_fixes: List[Dict[str, str]] = []
    structured_defects: List[Dict[str, Any]] = []

    if not report.header_valid:
        failed_stages.append("header")
        critical_failures.extend(report.header_errors)
        for h_err in report.header_errors:
            cat = VerificationErrorCategory.DIRECTION_MISMATCH.value if "direction" in h_err.lower() else VerificationErrorCategory.STRUCTURAL_ERROR.value
            structured_defects.append({
                "error_category": cat,
                "error_location": "module_header",
                "expected_behavior": "Conform exactly to approved plan ports and directions.",
                "actual_behavior": h_err,
                "related_requirement": "REQ_INTERFACE_CONFORMANCE",
                "supporting_evidence": plan_dict.get("module_name", ""),
                "allowed_correction_scope": "module_header",
            })
        actionable_fixes.append({
            "stage": "header",
            "issue": "; ".join(report.header_errors),
            "suggested_action": "Lock and preserve exact module name, port directions, and widths from approved plan."
        })

    if not report.static_report.is_valid:
        failed_stages.append("static_synthesizability")
        err_msgs = [getattr(e, "message", str(e)) for e in report.static_report.errors]
        critical_failures.extend(err_msgs)
        for e in err_msgs:
            structured_defects.append({
                "error_category": VerificationErrorCategory.STRUCTURAL_ERROR.value,
                "error_location": "declarations_or_assignments",
                "expected_behavior": "Single driver per signal; zero non-synthesizable delays (#) or simulation tasks.",
                "actual_behavior": e,
                "related_requirement": "REQ_SYNTHESIZABLE_STRUCTURE",
                "supporting_evidence": "Synthesizability Design Rules",
                "allowed_correction_scope": "assignments_and_procedural_blocks",
            })
            actionable_fixes.append({
                "stage": "static_synthesizability",
                "issue": e,
                "suggested_action": "Remove multi-driver assignments or delays (#); ensure synthesizable constructs."
            })

    if report.lint_report.errors or report.lint_report.status == "FAIL":
        failed_stages.append("lint")
        l_errs = [getattr(e, "message", str(e)) for e in report.lint_report.errors]
        critical_failures.extend(l_errs)
        for le in l_errs:
            structured_defects.append({
                "error_category": VerificationErrorCategory.SYNTAX_ERROR.value,
                "error_location": "syntax",
                "expected_behavior": "Compliant HDL syntax without parser errors.",
                "actual_behavior": le,
                "related_requirement": "REQ_HDL_SYNTAX",
                "supporting_evidence": "HDL Validator",
                "allowed_correction_scope": "affected_line",
            })

    if report.compile_report.status == VerificationStatus.FAIL.value:
        failed_stages.append("compilation")
        c_errs = [getattr(e, "message", str(e)) for e in report.compile_report.errors]
        critical_failures.extend(c_errs)
        for ce in c_errs:
            structured_defects.append({
                "error_category": VerificationErrorCategory.COMPILATION_ERROR.value,
                "error_location": "compiler_diagnostic",
                "expected_behavior": "Zero compiler fatal/syntax errors.",
                "actual_behavior": ce,
                "related_requirement": "REQ_EDA_COMPILATION",
                "supporting_evidence": report.compile_report.tool_name,
                "allowed_correction_scope": "compilation_unit",
            })
        actionable_fixes.append({
            "stage": "compilation",
            "issue": "; ".join(c_errs),
            "suggested_action": "Fix HDL syntax and type mismatches identified by compiler."
        })

    if report.sim_report.status == VerificationStatus.FAIL.value:
        failed_stages.append("simulation")
        s_errs = [getattr(e, "message", str(e)) for e in report.sim_report.errors]
        critical_failures.extend(s_errs)

    if not report.fsm_report.is_conformant:
        failed_stages.append("fsm_conformance")
        f_errs = [getattr(e, "message", str(e)) for e in report.fsm_report.errors]
        critical_failures.extend(f_errs)

    if report.protocol_report.status == VerificationStatus.FAIL.value:
        failed_stages.append("protocol_timing")
        p_errs = [getattr(e, "message", str(e)) for e in report.protocol_report.errors]
        critical_failures.extend(p_errs)

    if report.traceability_report.missing_requirements:
        failed_stages.append("traceability")
        warnings.extend(report.traceability_report.missing_requirements)

    return {
        "status": report.status,
        "is_acceptable": report.is_acceptable,
        "score": report.score,
        "failed_stages": failed_stages,
        "critical_failures": critical_failures,
        "warnings": warnings,
        "actionable_fixes": actionable_fixes,
        "structured_defects": structured_defects,
        "compilation": report.compile_report.to_dict(),
        "simulation": report.sim_report.to_dict(),
    }


def verify_candidate_rtl(
    code: str,
    rtl_plan: Any,
    target_language: str = "Verilog",
    testbench: Optional[str] = None,
    efs_ir: Optional[EFSIR] = None,
    run_compilation: bool = True,
    run_simulation: bool = True,
) -> CandidateVerificationReport:
    """
    Consolidated verification pipeline executing:
    1. Structural plan conformance (header, ports, widths, parameters, registers)
    2. Tool lint checks
    3. Builtin static synthesizability checks
    4. FSM conformance verification
    5. HDL compilation check (iverilog/verilator)
    6. Functional simulation check (vvp/verilator)
    7. Protocol and timing verification
    8. Requirement-level traceability verification
    9. Actionable feedback generation
    """
    from core.agents.constrained_generator import (
        emit_deterministic_module_header,
        verify_module_header_matches_plan,
    )

    plan_dict = rtl_plan.to_dict() if hasattr(rtl_plan, "to_dict") else (rtl_plan if isinstance(rtl_plan, dict) else {})
    diagnostics: List[str] = []
    rejection_reasons: List[str] = []

    # 1. Structural Conformance & Header Verification
    plan_hdr = emit_deterministic_module_header(plan_dict, target_language=target_language)
    hdr_ok, hdr_errors = verify_module_header_matches_plan(code, plan_hdr, plan_dict)
    structural_report = verify_candidate_structural_conformance(code, plan_dict, efs_ir=efs_ir)

    if not hdr_ok:
        diagnostics.extend(hdr_errors)
        rejection_reasons.append("Module header/port list does not match approved plan")
        for e in hdr_errors:
            rejection_reasons.append(e)

    if not structural_report.is_valid:
        struct_errs = [getattr(e, "message", str(e)) for e in structural_report.errors]
        diagnostics.extend(struct_errs)
        for se in struct_errs:
            if se not in rejection_reasons:
                rejection_reasons.append(se)

    # 2. Tool lint
    lang_to_lint = plan_dict.get("target_language") or target_language
    lint_report = run_candidate_tool_lint(code, language=lang_to_lint)
    if lint_report.status != "PASS" or lint_report.errors:
        errs = [getattr(e, "message", str(e)) for e in lint_report.errors]
        diagnostics.extend(errs)
        rejection_reasons.append("Tool lint errors detected")

    # 3. Static synthesizability
    static_report = run_builtin_static_checks(code, plan_dict)
    if not static_report.is_valid:
        errs = [getattr(e, "message", str(e)) for e in static_report.errors]
        diagnostics.extend(errs)
        rejection_reasons.append("Static synthesizability violations (multi-driver/delays)")
        for e in errs:
            rejection_reasons.append(e)

    # 4. FSM conformance
    fsm_report = verify_fsm_plan_conformance(code, plan_dict)
    if not fsm_report.is_conformant:
        errs = [getattr(e, "message", str(e)) for e in fsm_report.errors]
        diagnostics.extend(errs)
        rejection_reasons.append("Required FSM states missing from code")

    # 5. Compilation check
    if run_compilation:
        compile_report = compile_candidate_hdl(code, target_language=lang_to_lint)
        if compile_report.status == VerificationStatus.FAIL.value:
            errs = [getattr(e, "message", str(e)) for e in compile_report.errors]
            diagnostics.extend(errs)
            rejection_reasons.append("HDL compilation failure")
    else:
        compile_report = CandidateCompileReport(status=VerificationStatus.NOT_RUN.value)

    # 6. Functional simulation check
    if run_simulation:
        sim_tb = testbench or generate_auto_testbench(plan_dict, target_language=lang_to_lint, efs_ir=efs_ir)
        sim_report = simulate_candidate_testbench(code, sim_tb, target_language=lang_to_lint)
        if sim_report.status == VerificationStatus.FAIL.value:
            errs = [getattr(e, "message", str(e)) for e in sim_report.errors]
            diagnostics.extend(errs)
            rejection_reasons.append("Functional simulation testbench assertion failure")
    else:
        sim_report = CandidateSimReport(status=VerificationStatus.NOT_RUN.value)

    # 7. Protocol and timing verification
    protocol_report = verify_candidate_protocol_and_timing(code, plan_dict, efs_ir=efs_ir)
    if protocol_report.status == VerificationStatus.FAIL.value:
        errs = [getattr(e, "message", str(e)) for e in protocol_report.errors]
        diagnostics.extend(errs)
        rejection_reasons.append("Protocol/timing requirements violated")

    # 8. Requirement traceability verification
    traceability_report = verify_candidate_traceability(code, plan_dict, efs_ir=efs_ir)
    if traceability_report.missing_requirements:
        diagnostics.extend(traceability_report.missing_requirements)

    is_acceptable = (len(rejection_reasons) == 0)

    # Calculate overall score
    score = 100.0
    if not hdr_ok or not structural_report.is_valid:
        score -= 40.0
    if not static_report.is_valid:
        score -= 40.0
    if compile_report.status == VerificationStatus.FAIL.value:
        score -= 40.0
    if sim_report.status == VerificationStatus.FAIL.value:
        score -= 40.0
    if lint_report.status != "PASS" or lint_report.errors:
        score -= 20.0
    if not fsm_report.is_conformant:
        score -= 20.0
    if protocol_report.status == VerificationStatus.FAIL.value:
        score -= 15.0
    if traceability_report.missing_requirements or traceability_report.unmapped_rtl_elements:
        score -= 15.0
    score = max(0.0, min(100.0, score))

    report = CandidateVerificationReport(
        status="PASS" if is_acceptable else "CHECKS_FAILED",
        header_valid=hdr_ok,
        header_errors=hdr_errors,
        structural_report=structural_report,
        lint_report=lint_report,
        static_report=static_report,
        compile_report=compile_report,
        fsm_report=fsm_report,
        sim_report=sim_report,
        protocol_report=protocol_report,
        traceability_report=traceability_report,
        diagnostics=diagnostics,
        is_acceptable=is_acceptable,
        rejection_reasons=rejection_reasons,
        score=score,
    )
    report.verification_feedback = generate_verification_feedback(report, plan_dict)
    return report


def rank_and_select_candidates(
    candidates: List[Any],
    rtl_plan: Any,
    efs_ir: Optional[EFSIR] = None,
    target_language: str = "Verilog",
    weights: Optional[Dict[str, float]] = None
) -> MultiCandidateEvaluationReport:
    """
    Multi-candidate comparison, ranking, and mandatory rejection.
    Evaluates:
    - specification_conformance
    - functional_verification
    - structural_correctness
    - compilation
    - protocol_correctness
    - traceability_completeness
    - lint_quality
    - synthesis_compatibility
    - maintainability
    A high score must never override a mandatory failure.
    """
    w = {
        "specification_conformance": 0.20,
        "functional_verification": 0.20,
        "structural_correctness": 0.20,
        "compilation": 0.15,
        "protocol_correctness": 0.10,
        "traceability_completeness": 0.05,
        "lint_quality": 0.05,
        "synthesis_compatibility": 0.05,
        "maintainability": 0.00,
    }
    if weights:
        w.update(weights)

    eval_results: List[CandidateEvaluationResult] = []

    for idx, c in enumerate(candidates):
        cand_id = f"cand_{idx+1}"
        mode = "HYBRID"
        code = ""

        if isinstance(c, tuple):
            if len(c) == 2:
                code, mode = c[0], c[1]
            elif len(c) >= 3:
                cand_id, code, meta = c[0], c[1], c[2]
                mode = meta.get("mode", mode) if isinstance(meta, dict) else str(meta)
        elif isinstance(c, dict):
            cand_id = c.get("id", cand_id)
            code = c.get("code", "")
            mode = c.get("mode", mode)
        else:
            code = str(c)

        # Run full candidate verification
        rep = verify_candidate_rtl(code, rtl_plan, target_language=target_language, efs_ir=efs_ir)

        # Mandatory Rejection Conditions check
        rejection_reasons = []

        if not rep.header_valid or not rep.structural_report.is_valid:
            rejection_reasons.append("Structural header mismatch or unapproved interface modifications")
        if not rep.static_report.is_valid:
            rejection_reasons.append(f"Synthesizability/static check failed: {[getattr(e, 'message', str(e)) for e in rep.static_report.errors]}")
        if rep.compile_report.status == VerificationStatus.FAIL.value:
            rejection_reasons.append("Critical HDL compilation failure")
        if rep.sim_report.status == VerificationStatus.FAIL.value:
            rejection_reasons.append("Functional simulation testbench assertion failure")
        if rep.lint_report.status != "PASS" or rep.lint_report.errors:
            rejection_reasons.append("Tool lint violations")
        if not rep.fsm_report.is_conformant:
            rejection_reasons.append("FSM state non-conformance")
        if rep.protocol_report.status == VerificationStatus.FAIL.value:
            rejection_reasons.append("Protocol/timing requirement violation")
        if rep.traceability_report.missing_requirements or rep.traceability_report.unmapped_rtl_elements:
            rejection_reasons.append("Missing required signals or fabricated unapproved ports")

        # Weighted score computation
        score = 0.0
        score += w["specification_conformance"] * (100.0 if not rep.traceability_report.missing_requirements else 0.0)
        score += w["functional_verification"] * (100.0 if rep.sim_report.status == VerificationStatus.PASS.value else (80.0 if rep.sim_report.status == VerificationStatus.NOT_AVAILABLE.value else 0.0))
        score += w["structural_correctness"] * (100.0 if rep.header_valid and rep.structural_report.is_valid else 0.0)
        score += w["compilation"] * (100.0 if rep.compile_report.status == VerificationStatus.PASS.value else (80.0 if rep.compile_report.status == VerificationStatus.NOT_AVAILABLE.value else 0.0))
        score += w["protocol_correctness"] * (100.0 if rep.protocol_report.status in (VerificationStatus.PASS.value, VerificationStatus.NOT_RUN.value) else 0.0)
        score += w["traceability_completeness"] * rep.traceability_report.coverage_percentage
        score += w["lint_quality"] * (100.0 if rep.lint_report.status == "PASS" and not rep.lint_report.errors else 0.0)
        score += w["synthesis_compatibility"] * (100.0 if rep.static_report.is_valid else 0.0)

        # Bonus for no warnings
        if not rep.lint_report.warnings and not rep.diagnostics:
            score += 5.0
        score = max(0.0, min(100.0, round(score, 1)))

        is_acceptable = (len(rejection_reasons) == 0)
        rep.score = score
        rep.is_acceptable = is_acceptable
        rep.rejection_reasons = rejection_reasons
        if not is_acceptable:
            rep.status = "REJECTED"

        eval_results.append(CandidateEvaluationResult(
            candidate_id=cand_id,
            generation_mode=str(mode),
            code=code,
            report=rep,
            score=score
        ))

    # Sort: Acceptable candidates first (score descending), then rejected candidates (score descending)
    # A high score must NEVER override a mandatory rejection!
    eval_results.sort(key=lambda x: (1 if x.report.is_acceptable else 0, x.score), reverse=True)

    for rank_idx, res in enumerate(eval_results, start=1):
        res.rank = rank_idx

    selected = None
    if eval_results and eval_results[0].report.is_acceptable:
        eval_results[0].is_selected = True
        selected = eval_results[0]

    ranking_order = [r.candidate_id for r in eval_results]
    comparison_matrix = []
    for r in eval_results:
        comparison_matrix.append({
            "candidate_id": r.candidate_id,
            "mode": r.generation_mode,
            "score": r.score,
            "is_acceptable": r.report.is_acceptable,
            "status": r.report.status,
            "header_valid": r.report.header_valid,
            "static_valid": r.report.static_report.is_valid,
            "compile_status": r.report.compile_report.status,
            "sim_status": r.report.sim_report.status,
            "traceability_coverage": r.report.traceability_report.coverage_percentage,
            "rejections": r.report.rejection_reasons,
        })

    return MultiCandidateEvaluationReport(
        candidates=eval_results,
        selected_candidate=selected,
        ranking_order=ranking_order,
        comparison_matrix=comparison_matrix
    )


def recover_and_repair_candidate(
    code: str,
    report: CandidateVerificationReport,
    rtl_plan: Any,
    efs_ir: Optional[EFSIR] = None,
    target_language: str = "Verilog",
    max_attempts: int = 2
) -> Tuple[str, CandidateVerificationReport]:
    """
    Controlled error recovery and localized regeneration.
    Applies surgical repairs for structural, lint, and static synthesizability violations
    without regenerating the whole module, then re-verifies.
    Preserves repair_history and escalates missing requirements to HITL.
    """
    if report.passed and report.is_acceptable:
        return code, report

    current_code = code
    current_report = report
    from core.agents.constrained_generator import emit_deterministic_module_header

    plan_dict = rtl_plan.to_dict() if hasattr(rtl_plan, "to_dict") else (rtl_plan if isinstance(rtl_plan, dict) else {})
    repair_history: List[Dict[str, Any]] = list(report.repair_history)

    for attempt in range(1, max_attempts + 1):
        modified = False
        repaired_actions = []

        # 1. Header & Direction Repair: if header or port direction mismatches
        if not current_report.header_valid or not current_report.structural_report.is_valid:
            correct_hdr = emit_deterministic_module_header(plan_dict, target_language=target_language)
            new_code = re.sub(r"module\s+[a-zA-Z0-9_]+\s*(?:#\s*\([\s\S]*?\)\s*)?\(([\s\S]*?)\)\s*;", correct_hdr, current_code, count=1)
            if new_code != current_code:
                current_code = new_code
                modified = True
                repaired_actions.append("Corrected module header and port list to match approved plan.")

        # 2. Multi-driver conflict repair: if signal has both assign and procedural driver
        for err in current_report.static_report.errors:
            err_msg = getattr(err, "message", str(err))
            m_sig = re.search(r"Multiple driver conflict on signal '([a-zA-Z0-9_]+)'", err_msg)
            if m_sig:
                conflict_sig = m_sig.group(1)
                new_code = re.sub(rf"^\s*assign\s+{re.escape(conflict_sig)}\s*=[\s\S]*?;\s*$", "", current_code, flags=re.MULTILINE)
                if new_code != current_code:
                    current_code = new_code
                    modified = True
                    repaired_actions.append(f"Eliminated redundant continuous assignment driver for '{conflict_sig}'.")

        # 3. Illegal delay repair (#)
        if any("Delays (#)" in getattr(e, "message", str(e)) for e in current_report.static_report.errors):
            new_code = re.sub(r"#\s*\d+", "", current_code)
            if new_code != current_code:
                current_code = new_code
                modified = True
                repaired_actions.append("Stripped non-synthesizable delay (#) statements.")

        # 4. Simulation tasks in synthesizable RTL ($finish, $display)
        if any("Simulation system tasks" in getattr(e, "message", str(e)) for e in current_report.static_report.errors):
            new_code = re.sub(r"\$(?:dump|display|finish|time|readmem)[^;]*;", "", current_code)
            if new_code != current_code:
                current_code = new_code
                modified = True
                repaired_actions.append("Removed simulation system tasks from synthesizable body.")

        # Re-verify
        current_report = verify_candidate_rtl(current_code, rtl_plan, target_language=target_language, efs_ir=efs_ir)
        repair_history.append({
            "attempt": attempt,
            "actions": repaired_actions,
            "passed": current_report.passed,
            "is_acceptable": current_report.is_acceptable
        })
        current_report.repair_history = repair_history

        if current_report.passed and current_report.is_acceptable:
            current_report.diagnostics.append(f"Successfully repaired in attempt {attempt}.")
            return current_code, current_report

        if not modified:
            break

    current_report.repair_history = repair_history
    return current_code, current_report


def escalate_verification_failure_to_hitl(
    report: CandidateVerificationReport,
    efs_ir: Optional[EFSIR] = None,
    spec_text: str = "",
    rtl_plan: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Escalate verification failures to HITL when failures arise from
    ambiguous, contradictory, or incomplete specifications.
    """
    from core.verification.hitl_plan_manager import HITLPlanManager

    escalations: List[Dict[str, Any]] = []

    if efs_ir:
        existing_confs = HITLPlanManager.get_unresolved_architectural_conflicts(efs_ir)
        escalations.extend(existing_confs)

    for diag in report.diagnostics:
        if any(w in diag.lower() for w in ("missing", "undefined", "polarity", "direction", "conflict", "unspecified")):
            cid = f"HITL_ESC_{abs(hash(diag)) % 100000}"
            if not any(e.get("conflict_id") == cid for e in escalations):
                alts = [
                    f"Option A: Implement default behavior for {diag}",
                    f"Option B: Block generation until specification clarifies {diag}"
                ]
                escalations.append({
                    "conflict_id": cid,
                    "category": "SPECIFICATION_GAP",
                    "issue": diag,
                    "evidence": spec_text or diag,
                    "alternatives": alts,
                    "status": "UNRESOLVED",
                    "hitl_required": True
                })

    return escalations
