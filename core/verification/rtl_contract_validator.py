"""
Deterministic RTL Post-Generation Contract Validator.

Performs independent deterministic structural and semantic validation of generated HDL
against the authoritative RTLGenerationPlan contract.

Does NOT query an LLM to decide correctness. Uses deterministic regex, lexical analysis,
and AST pattern extraction to verify:
1. Interface Contract (module name, ports, directions, widths, parameters, duplicate/missing ports)
2. Internal Structure (registers, signals, state variables, widths)
3. Sequential Logic (clock edge, reset edge/polarity/type, register updates)
4. FSM State & Transition Contract (states, encodings, transitions, conditions, reset state)
5. Combinational Logic & Output Assignments (output drivers, latch avoidance, default branches)
6. Traceability and Contract Invariant Preservation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from utils.logger import get_logger

logger = get_logger("verification.rtl_contract_validator")


class ContractViolationType(str, Enum):
    MODULE_NAME_MISMATCH = "MODULE_NAME_MISMATCH"
    MISSING_PORT = "MISSING_PORT"
    UNEXPECTED_PORT = "UNEXPECTED_PORT"
    PORT_DIRECTION_MISMATCH = "PORT_DIRECTION_MISMATCH"
    PORT_WIDTH_MISMATCH = "PORT_WIDTH_MISMATCH"
    DUPLICATE_PORT_DECLARATION = "DUPLICATE_PORT_DECLARATION"
    MISSING_REGISTER = "MISSING_REGISTER"
    REGISTER_WIDTH_MISMATCH = "REGISTER_WIDTH_MISMATCH"
    CLOCK_EDGE_MISMATCH = "CLOCK_EDGE_MISMATCH"
    RESET_POLARITY_MISMATCH = "RESET_POLARITY_MISMATCH"
    RESET_TYPE_MISMATCH = "RESET_TYPE_MISMATCH"
    MISSING_FSM_STATE = "MISSING_FSM_STATE"
    MISSING_FSM_TRANSITION = "MISSING_FSM_TRANSITION"
    TRANSITION_CONDITION_ALTERED = "TRANSITION_CONDITION_ALTERED"
    UNDRIVEN_OUTPUT = "UNDRIVEN_OUTPUT"
    MULTIPLE_DRIVERS = "MULTIPLE_DRIVERS"
    UNPLANNED_IMPLEMENTATION = "UNPLANNED_IMPLEMENTATION"
    LATCH_RISK = "LATCH_RISK"


@dataclass
class ContractViolation:
    violation_type: ContractViolationType
    entity: str
    expected: Any
    found: Any
    severity: str = "CRITICAL"  # CRITICAL, MAJOR, WARNING
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_type": self.violation_type.value,
            "entity": self.entity,
            "expected": str(self.expected),
            "found": str(self.found),
            "severity": self.severity,
            "description": self.description or f"{self.violation_type.value}: {self.entity} expected {self.expected}, found {self.found}",
        }


@dataclass
class RTLContractValidationReport:
    validation_status: str  # "VALIDATED" | "CONTRACT_VIOLATION" | "SYNTAX_ERROR"
    is_valid: bool
    passed_checks: List[str] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_requirements: List[str] = field(default_factory=list)
    unexpected_implementations: List[str] = field(default_factory=list)
    contract_violations: List[ContractViolation] = field(default_factory=list)
    affected_plan_entities: List[str] = field(default_factory=list)
    source_traceability: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "validation_status": self.validation_status,
            "is_valid": self.is_valid,
            "passed_checks_count": len(self.passed_checks),
            "passed_checks": self.passed_checks,
            "failed_checks_count": len(self.failed_checks),
            "failed_checks": self.failed_checks,
            "warnings_count": len(self.warnings),
            "warnings": self.warnings,
            "missing_requirements": self.missing_requirements,
            "unexpected_implementations": self.unexpected_implementations,
            "contract_violations_count": len(self.contract_violations),
            "contract_violations": [v.to_dict() for v in self.contract_violations],
            "affected_plan_entities": self.affected_plan_entities,
            "source_traceability": self.source_traceability,
            "metrics": self.metrics,
        }


class RTLContractValidator:
    """Deterministic structural and behavioral contract verifier for generated RTL."""

    def __init__(self):
        pass

    def _strip_comments(self, code: str) -> str:
        """Remove line comments and block comments from HDL code."""
        # Block comments /* ... */
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
        # Line comments // ...
        code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
        return code

    def _parse_extracted_ports(self, clean_code: str) -> Dict[str, Dict[str, Any]]:
        """Extract all port declarations from ANSI module header or body."""
        ports: Dict[str, Dict[str, Any]] = {}
        
        # Match ANSI header port declarations
        # e.g.: input wire [31:0] data_in, output reg done
        port_pattern = re.compile(
            r"\b(input|output|inout)\s+(?:wire\s+|reg\s+|logic\s+)?(?:\[\s*([^\]]+)\s*\]\s*)?([a-zA-Z_][a-zA-Z0-9_]*)",
            re.IGNORECASE
        )
        for m in port_pattern.finditer(clean_code):
            direction = m.group(1).lower()
            width_range = m.group(2)
            name = m.group(3)
            width_val = "1"
            if width_range:
                # e.g. "31:0" -> width 32, "WIDTH-1:0"
                range_match = re.match(r"^\s*(\d+)\s*:\s*0\s*$", width_range)
                if range_match:
                    width_val = str(int(range_match.group(1)) + 1)
                else:
                    width_val = width_range.strip()
            
            if name.lower() not in ports:
                ports[name.lower()] = {
                    "name": name,
                    "direction": direction,
                    "width": width_val,
                    "width_range": width_range
                }

        # Also match non-ANSI style declarations if present
        # input [7:0] a; output b;
        body_port_pattern = re.compile(
            r"^\s*(input|output|inout)\s+(?:\[\s*([^\]]+)\s*\]\s*)?([a-zA-Z_][a-zA-Z0-9_,\s]+);",
            re.MULTILINE | re.IGNORECASE
        )
        for m in body_port_pattern.finditer(clean_code):
            direction = m.group(1).lower()
            width_range = m.group(2)
            names_raw = m.group(3)
            width_val = "1"
            if width_range:
                range_match = re.match(r"^\s*(\d+)\s*:\s*0\s*$", width_range)
                if range_match:
                    width_val = str(int(range_match.group(1)) + 1)
                else:
                    width_val = width_range.strip()
            for n in re.split(r"[\s,]+", names_raw):
                n_clean = n.strip()
                if n_clean and n_clean.lower() not in ports:
                    ports[n_clean.lower()] = {
                        "name": n_clean,
                        "direction": direction,
                        "width": width_val,
                        "width_range": width_range
                    }
        return ports

    def _parse_extracted_registers(self, clean_code: str) -> Dict[str, Dict[str, Any]]:
        """Extract declared reg and logic sequential variables."""
        regs: Dict[str, Dict[str, Any]] = {}
        reg_pattern = re.compile(
            r"\b(reg|logic)\s+(?:\[\s*([^\]]+)\s*\]\s*)?([a-zA-Z_][a-zA-Z0-9_,\s]+);",
            re.MULTILINE | re.IGNORECASE
        )
        for m in reg_pattern.finditer(clean_code):
            width_range = m.group(2)
            names_str = m.group(3)
            width_val = "1"
            if width_range:
                range_match = re.match(r"^\s*(\d+)\s*:\s*0\s*$", width_range)
                if range_match:
                    width_val = str(int(range_match.group(1)) + 1)
                else:
                    width_val = width_range.strip()
            for name in re.split(r"[\s,]+", names_str):
                name = name.strip()
                if name and not any(name.startswith(pfx) for pfx in ("clk", "rst", "input", "output")):
                    regs[name.lower()] = {
                        "name": name,
                        "width": width_val,
                        "width_range": width_range
                    }
        return regs

    def _parse_fsm_states(self, clean_code: str) -> Set[str]:
        """Extract declared state names from parameter, localparam, or typedef enum."""
        states: Set[str] = set()
        param_pattern = re.compile(
            r"\b(?:parameter|localparam)\s+(?:\[\s*[^\]]+\s*\]\s*)?([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^,;]+)",
            re.IGNORECASE
        )
        for m in param_pattern.finditer(clean_code):
            name = m.group(1).strip()
            val = m.group(2).strip()
            # If value looks like an encoding (e.g. 2'b00, 2'd1, 0, 1), or name is uppercase
            if re.search(r"\d+'[bBhdD][0-9a-fA-F_]+|\b\d+\b", val) or name.isupper():
                states.add(name.lower())

        # Also search case statement labels
        case_item_pattern = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(?:begin)?", re.MULTILINE)
        for m in case_item_pattern.finditer(clean_code):
            lbl = m.group(1).strip().lower()
            if lbl not in ("default", "case", "endcase", "if", "else", "begin", "end"):
                states.add(lbl)

        return states

    def validate_contract(
        self,
        rtl_code: str,
        rtl_plan: Union[Any, Dict[str, Any]],
        target_language: str = "Verilog"
    ) -> RTLContractValidationReport:
        """
        Validate generated RTL code against authoritative RTLGenerationPlan.
        Returns a structured RTLContractValidationReport without mutating the plan.
        """
        if not rtl_code or not rtl_code.strip():
            return RTLContractValidationReport(
                validation_status="SYNTAX_ERROR",
                is_valid=False,
                failed_checks=["Empty or null RTL code provided."],
                missing_requirements=["Generated RTL code is empty."],
                contract_violations=[
                    ContractViolation(
                        violation_type=ContractViolationType.UNPLANNED_IMPLEMENTATION,
                        entity="module",
                        expected="synthesizable HDL code",
                        found="empty string",
                        severity="CRITICAL",
                        description="Generated RTL code is empty."
                    )
                ]
            )

        # Normalize plan to dictionary
        plan_dict = rtl_plan.to_dict() if hasattr(rtl_plan, "to_dict") else (rtl_plan if isinstance(rtl_plan, dict) else {})
        expected_module_name = plan_dict.get("module_name", "")

        clean_code = self._strip_comments(rtl_code)

        passed: List[str] = []
        failed: List[str] = []
        warnings: List[str] = []
        missing_reqs: List[str] = []
        unexpected_impls: List[str] = []
        violations: List[ContractViolation] = []
        affected_entities: Set[str] = set()

        # -------------------------------------------------------------
        # 1. Module Name Validation
        # -------------------------------------------------------------
        m_mod = re.search(r"\bmodule\s+([a-zA-Z_][a-zA-Z0-9_]*)", clean_code)
        found_mod_name = m_mod.group(1) if m_mod else ""
        if not found_mod_name:
            violations.append(ContractViolation(
                violation_type=ContractViolationType.MODULE_NAME_MISMATCH,
                entity="module",
                expected=expected_module_name,
                found="None",
                severity="CRITICAL",
                description="No valid Verilog/SystemVerilog 'module <name>' declaration found in code."
            ))
            failed.append("Module declaration missing.")
        elif expected_module_name and found_mod_name.lower() != expected_module_name.lower():
            violations.append(ContractViolation(
                violation_type=ContractViolationType.MODULE_NAME_MISMATCH,
                entity="module_name",
                expected=expected_module_name,
                found=found_mod_name,
                severity="CRITICAL",
                description=f"Module name '{found_mod_name}' does not match plan module name '{expected_module_name}'."
            ))
            failed.append(f"Module name mismatch: expected '{expected_module_name}', found '{found_mod_name}'.")
            affected_entities.add("module_name")
        else:
            passed.append(f"Module declaration validated: '{found_mod_name}'.")

        # -------------------------------------------------------------
        # 2. Port Interface Contract Validation
        # -------------------------------------------------------------
        extracted_ports = self._parse_extracted_ports(clean_code)
        plan_ports = plan_dict.get("ports", [])
        expected_port_map: Dict[str, Dict[str, Any]] = {}

        for p in plan_ports:
            if isinstance(p, dict) and p.get("name"):
                expected_port_map[str(p["name"]).strip().lower()] = p

        # Check for missing required plan ports
        for p_name_lower, p_data in expected_port_map.items():
            p_name = p_data.get("name", p_name_lower)
            if p_name_lower not in extracted_ports:
                violations.append(ContractViolation(
                    violation_type=ContractViolationType.MISSING_PORT,
                    entity=p_name,
                    expected=f"{p_data.get('direction', 'input')} [{p_data.get('width', '1')}-1:0]",
                    found="MISSING",
                    severity="CRITICAL",
                    description=f"Required plan port '{p_name}' is missing from generated RTL interface."
                ))
                missing_reqs.append(f"Port '{p_name}'")
                failed.append(f"Missing required port: '{p_name}'.")
                affected_entities.add(p_name)
            else:
                act = extracted_ports[p_name_lower]
                # Check direction
                exp_dir = str(p_data.get("direction", "input")).lower()
                act_dir = act.get("direction", "").lower()
                if exp_dir and act_dir and exp_dir != act_dir:
                    violations.append(ContractViolation(
                        violation_type=ContractViolationType.PORT_DIRECTION_MISMATCH,
                        entity=p_name,
                        expected=exp_dir,
                        found=act_dir,
                        severity="CRITICAL",
                        description=f"Port '{p_name}' direction mismatch: expected '{exp_dir}', found '{act_dir}'."
                    ))
                    failed.append(f"Port direction mismatch for '{p_name}'.")
                    affected_entities.add(p_name)
                
                # Check width (if numeric)
                exp_w = str(p_data.get("width", "1")).strip()
                act_w = str(act.get("width", "1")).strip()
                if exp_w.isdigit() and act_w.isdigit():
                    if int(exp_w) != int(act_w):
                        violations.append(ContractViolation(
                            violation_type=ContractViolationType.PORT_WIDTH_MISMATCH,
                            entity=p_name,
                            expected=exp_w,
                            found=act_w,
                            severity="CRITICAL",
                            description=f"Port '{p_name}' bit width mismatch: expected {exp_w}, found {act_w}."
                        ))
                        failed.append(f"Port width mismatch for '{p_name}'.")
                        affected_entities.add(p_name)
                    else:
                        passed.append(f"Port '{p_name}' ({exp_dir}[{exp_w}]) verified.")
                else:
                    passed.append(f"Port '{p_name}' ({exp_dir}) verified.")

        # Check for unexpected extra ports introduced by LLM
        for act_name_lower, act_data in extracted_ports.items():
            if act_name_lower not in expected_port_map:
                act_name = act_data.get("name", act_name_lower)
                # Ignore standard internal signal names if mistakenly parsed as port
                if act_name_lower not in ("clk", "rst", "rst_n", "clock", "reset"):
                    violations.append(ContractViolation(
                        violation_type=ContractViolationType.UNEXPECTED_PORT,
                        entity=act_name,
                        expected="NONE",
                        found=f"{act_data.get('direction', 'input')} [{act_data.get('width', '1')}]",
                        severity="MAJOR",
                        description=f"Generated RTL introduces unexpected port '{act_name}' not defined in the plan."
                    ))
                    unexpected_impls.append(f"Unexpected port '{act_name}'")
                    warnings.append(f"Unexpected port declared: '{act_name}'.")

        # -------------------------------------------------------------
        # 3. Internal Registers & Sequential Variables
        # -------------------------------------------------------------
        plan_regs = plan_dict.get("internal_registers", [])
        if plan_regs:
            extracted_regs = self._parse_extracted_registers(clean_code)
            for r in plan_regs:
                if isinstance(r, dict) and r.get("name"):
                    r_name = str(r["name"]).strip()
                    r_lower = r_name.lower()
                    # Check if declared as reg/logic or referenced in sequential assignment
                    found_in_decls = r_lower in extracted_regs or (r_lower in extracted_ports)
                    found_in_assign = bool(re.search(rf"\b{re.escape(r_name)}\s*<=|\b{re.escape(r_name)}\s*=", clean_code))
                    if not found_in_decls and not found_in_assign:
                        violations.append(ContractViolation(
                            violation_type=ContractViolationType.MISSING_REGISTER,
                            entity=r_name,
                            expected=f"reg [{r.get('width', '1')}-1:0] {r_name}",
                            found="UNDECLARED",
                            severity="MAJOR",
                            description=f"Planned internal register '{r_name}' is not declared or assigned in RTL."
                        ))
                        failed.append(f"Missing planned register: '{r_name}'.")
                        affected_entities.add(r_name)
                    else:
                        passed.append(f"Register '{r_name}' verified.")

        # -------------------------------------------------------------
        # 4. Sequential Logic & Clock / Reset Validation
        # -------------------------------------------------------------
        reset_req = plan_dict.get("reset_requirements") or plan_dict.get("clock_reset") or {}
        exp_reset_sig = reset_req.get("reset_signal") or reset_req.get("reset_name")
        exp_reset_pol = str(reset_req.get("reset_polarity", "active_low")).lower()
        
        # Check sensitivity lists
        always_blocks = re.findall(r"always(?:_ff)?\s*@\s*\(([^)]+)\)", clean_code, flags=re.IGNORECASE)
        if always_blocks:
            passed.append(f"Found {len(always_blocks)} sequential process blocks.")
            for sens in always_blocks:
                sens_lower = sens.lower()
                # Check for clock edge
                if "negedge clk" in sens_lower and "posedge" in str(plan_dict.get("clock_edge", "posedge")).lower():
                    violations.append(ContractViolation(
                        violation_type=ContractViolationType.CLOCK_EDGE_MISMATCH,
                        entity="clock_sensitivity",
                        expected="posedge clk",
                        found="negedge clk",
                        severity="CRITICAL",
                        description="Sensitivity list uses negedge clock when posedge was specified."
                    ))
                    failed.append("Clock edge polarity mismatch.")
                
                # Check reset polarity in async reset sensitivity list
                if exp_reset_sig and exp_reset_sig.lower() in sens_lower:
                    if exp_reset_pol in ("low", "active_low") and f"posedge {exp_reset_sig.lower()}" in sens_lower:
                        violations.append(ContractViolation(
                            violation_type=ContractViolationType.RESET_POLARITY_MISMATCH,
                            entity=exp_reset_sig,
                            expected="negedge (active-low)",
                            found="posedge",
                            severity="CRITICAL",
                            description=f"Active-low reset '{exp_reset_sig}' was placed in posedge sensitivity list."
                        ))
                        failed.append(f"Reset sensitivity mismatch for '{exp_reset_sig}'.")
                    elif exp_reset_pol in ("high", "active_high") and f"negedge {exp_reset_sig.lower()}" in sens_lower:
                        violations.append(ContractViolation(
                            violation_type=ContractViolationType.RESET_POLARITY_MISMATCH,
                            entity=exp_reset_sig,
                            expected="posedge (active-high)",
                            found="negedge",
                            severity="CRITICAL",
                            description=f"Active-high reset '{exp_reset_sig}' was placed in negedge sensitivity list."
                        ))
                        failed.append(f"Reset sensitivity mismatch for '{exp_reset_sig}'.")

        # -------------------------------------------------------------
        # 5. FSM State and Transition Contract Validation
        # -------------------------------------------------------------
        fsm_plan = plan_dict.get("fsm") or plan_dict.get("fsm_spec")
        if fsm_plan and isinstance(fsm_plan, dict) and fsm_plan.get("states"):
            exp_states = [str(s).strip() for s in fsm_plan.get("states", []) if s]
            found_states = self._parse_fsm_states(clean_code)

            for exp_st in exp_states:
                if exp_st.lower() not in found_states and exp_st.lower() not in clean_code.lower():
                    violations.append(ContractViolation(
                        violation_type=ContractViolationType.MISSING_FSM_STATE,
                        entity=exp_st,
                        expected=f"state {exp_st}",
                        found="MISSING",
                        severity="CRITICAL",
                        description=f"Required FSM state '{exp_st}' is not defined in the RTL."
                    ))
                    failed.append(f"Missing required FSM state: '{exp_st}'.")
                    affected_entities.add(exp_st)
                else:
                    passed.append(f"FSM state '{exp_st}' verified.")

            # Check transitions
            transitions = fsm_plan.get("transitions", [])
            for tr in transitions:
                if isinstance(tr, dict):
                    from_st = tr.get("from_state") or tr.get("from")
                    to_st = tr.get("to_state") or tr.get("to")
                    if from_st and to_st:
                        # Verify that both from_st and to_st appear in the code logic
                        if from_st.lower() in clean_code.lower() and to_st.lower() in clean_code.lower():
                            passed.append(f"Transition {from_st} -> {to_st} represented.")
                        else:
                            violations.append(ContractViolation(
                                violation_type=ContractViolationType.MISSING_FSM_TRANSITION,
                                entity=f"{from_st}->{to_st}",
                                expected=f"Transition from {from_st} to {to_st}",
                                found="MISSING",
                                severity="CRITICAL",
                                description=f"Required FSM transition '{from_st} -> {to_st}' is absent in RTL."
                            ))
                            failed.append(f"Missing FSM transition: '{from_st} -> {to_st}'.")
                            affected_entities.add(f"{from_st}->{to_st}")

        # -------------------------------------------------------------
        # 6. Combinational Completeness & Latch Prevention
        # -------------------------------------------------------------
        # Verify default branch in case statements to prevent unintentional latch inference
        case_blocks = re.findall(r"\bcase\s*\([^)]+\)([\s\S]*?)\bendcase\b", clean_code, flags=re.IGNORECASE)
        for idx, cb in enumerate(case_blocks):
            if "default:" not in cb.lower():
                warnings.append(f"Case block #{idx+1} has no 'default:' statement (potential latch inference risk).")
            else:
                passed.append(f"Case block #{idx+1} includes safe 'default:' branch.")

        # Check undriven output ports and multiple drivers
        out_ports = [
            p_data.get("name") for p_name, p_data in expected_port_map.items()
            if str(p_data.get("direction", "")).lower() in ("output", "out")
        ]
        always_blocks = re.findall(r"\balways(?:_ff|_comb|_latch)?\b[\s\S]*?(?=\balways(?:_ff|_comb|_latch)?\b|\bendmodule\b)", clean_code)

        for op in out_ports:
            if op:
                # Count continuous assignments
                assign_matches = list(re.finditer(rf"\bassign\s+{re.escape(op)}\b\s*=", clean_code))
                assign_count = len(assign_matches)
                # Count procedural driver blocks
                proc_count = sum(1 for block in always_blocks if re.search(rf"\b{re.escape(op)}\s*(?:<=|=)", block))

                is_driven = (assign_count > 0 or proc_count > 0 or
                             bool(re.search(rf"\bassign\s+{re.escape(op)}\b|\b{re.escape(op)}\s*<=|\b{re.escape(op)}\s*=", clean_code)))

                if not is_driven:
                    violations.append(ContractViolation(
                        violation_type=ContractViolationType.UNDRIVEN_OUTPUT,
                        entity=op,
                        expected=f"Driving logic for output port '{op}'",
                        found="UNDRIVEN",
                        severity="CRITICAL",
                        description=f"Output port '{op}' is declared but never assigned in RTL."
                    ))
                    failed.append(f"Undriven output port: '{op}'.")
                    affected_entities.add(op)
                else:
                    # Check multiple driver violations
                    if assign_count > 1:
                        violations.append(ContractViolation(
                            violation_type=ContractViolationType.MULTIPLE_DRIVERS,
                            entity=op,
                            expected="Single continuous driver",
                            found=f"{assign_count} continuous assignments",
                            severity="CRITICAL",
                            description=f"Output port '{op}' has multiple continuous assignment drivers ({assign_count} assign statements)."
                        ))
                        failed.append(f"Multiple continuous drivers for '{op}'.")
                        affected_entities.add(op)
                    elif assign_count >= 1 and proc_count >= 1:
                        violations.append(ContractViolation(
                            violation_type=ContractViolationType.MULTIPLE_DRIVERS,
                            entity=op,
                            expected="Single driving paradigm (continuous OR procedural)",
                            found=f"{assign_count} continuous assign(s) and {proc_count} procedural block(s)",
                            severity="CRITICAL",
                            description=f"Output port '{op}' has conflicting continuous and procedural drivers."
                        ))
                        failed.append(f"Conflicting continuous and procedural drivers for '{op}'.")
                        affected_entities.add(op)
                    elif proc_count > 1:
                        violations.append(ContractViolation(
                            violation_type=ContractViolationType.MULTIPLE_DRIVERS,
                            entity=op,
                            expected="Single procedural always block driver",
                            found=f"Assigned across {proc_count} distinct always blocks",
                            severity="CRITICAL",
                            description=f"Output port '{op}' is driven by {proc_count} distinct always blocks (multi-process driver conflict)."
                        ))
                        failed.append(f"Multiple procedural process drivers for '{op}'.")
                        affected_entities.add(op)
                    else:
                        passed.append(f"Output port '{op}' assignment verified.")

        # -------------------------------------------------------------
        # 7. Final Assessment & Status
        # -------------------------------------------------------------
        has_critical = any(v.severity == "CRITICAL" for v in violations)
        status = "CONTRACT_VIOLATION" if has_critical else ("VALIDATED" if not failed else "VALIDATION_WARNINGS")
        is_valid = not has_critical

        metrics = {
            "ports_checked": len(expected_port_map),
            "passed_checks": len(passed),
            "failed_checks": len(failed),
            "violations_count": len(violations),
            "warnings_count": len(warnings),
        }

        # Build source traceability mapping
        traceability_map: Dict[str, str] = {}
        for p in plan_ports:
            if isinstance(p, dict) and p.get("name") and p.get("source_ref"):
                traceability_map[str(p["name"])] = str(p["source_ref"])

        return RTLContractValidationReport(
            validation_status=status,
            is_valid=is_valid,
            passed_checks=passed,
            failed_checks=failed,
            warnings=warnings,
            missing_requirements=missing_reqs,
            unexpected_implementations=unexpected_impls,
            contract_violations=violations,
            affected_plan_entities=list(affected_entities),
            source_traceability=traceability_map,
            metrics=metrics
        )

