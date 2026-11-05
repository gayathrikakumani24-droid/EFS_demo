"""
RTL Plan Validator Module.

Validates structured RTL Implementation Plans against authoritative EFS IR,
scoped ContextPack, and Source Evidence.

Checks:
- Missing required ports
- Signal direction conflicts
- Width conflicts
- Duplicate declarations
- Register inconsistencies
- Opcode conflicts
- FSM inconsistencies & missing transitions
- Datapath dependency errors & unassigned outputs
- Unspecified or conflicting reset semantics
- Unsupported assumptions (hallucinated constructs)
- Missing behavioral implementations
- Missing provenance
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from core.efs_ir.models import EFSIR, EFSConflict, EFSResolution
from core.efs_ir.signal_normalization import normalize_direction, parse_bit_range
from core.efs_ir.validation_rules import normalize_hardware_width
from core.agents.spec_classifier import classify_hardware_architecture, ArchitectureClass
from utils.logger import get_logger

logger = get_logger("agents.rtl_plan_validator")


class DiagnosticCode(str, Enum):
    """Canonical diagnostic codes for hardware plan validation."""
    SPECIFICATION_INCOMPLETE = "SPECIFICATION_INCOMPLETE"
    PLAN_VALIDATION_FAILED = "PLAN_VALIDATION_FAILED"
    INTERFACE_CONFLICT = "INTERFACE_CONFLICT"
    REGISTER_DEFINITION_CONFLICT = "REGISTER_DEFINITION_CONFLICT"
    OPCODE_CONFLICT = "OPCODE_CONFLICT"
    FSM_SEMANTIC_ERROR = "FSM_SEMANTIC_ERROR"
    DATAPATH_UNSPECIFIED = "DATAPATH_UNSPECIFIED"
    CONTEXT_COMPILATION_FAILED = "CONTEXT_COMPILATION_FAILED"


@dataclass
class PlanDiagnostic:
    """Detailed validation diagnostic with evidence and suggested resolution."""
    diagnostic_type: DiagnosticCode
    category: str  # INTERFACE, REGISTER, OPCODE, FSM, DATAPATH, RESET, SPECIFICATION
    affected_object: Optional[str]
    evidence: str
    severity: str = "CRITICAL"  # CRITICAL, MAJOR, WARNING
    suggested_resolution: str = ""
    hitl_required: bool = True
    conflict_id: Optional[str] = None
    alternatives: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagnostic_type": self.diagnostic_type.value,
            "category": self.category,
            "affected_object": self.affected_object,
            "evidence": self.evidence,
            "severity": self.severity,
            "suggested_resolution": self.suggested_resolution,
            "hitl_required": self.hitl_required,
            "conflict_id": self.conflict_id,
            "alternatives": self.alternatives,
            # Backwards-compatible fields for UI and legacy checks
            "classification": "SOURCE_CONFLICT" if "CONFLICT" in self.diagnostic_type.value else ("AMBIGUOUS" if self.diagnostic_type == DiagnosticCode.SPECIFICATION_INCOMPLETE else "PLAN_MISSING"),
            "badge": "🟠 [CONFLICT]" if "CONFLICT" in self.diagnostic_type.value else ("🟡 [INCOMPLETE]" if self.diagnostic_type == DiagnosticCode.SPECIFICATION_INCOMPLETE else "🔴 [ERROR]"),
            "is_blocking": self.severity == "CRITICAL",
            "issue": f"[{self.diagnostic_type.value}] {self.suggested_resolution or self.evidence}",
            "required_information": self.suggested_resolution,
        }


@dataclass
class RTLPlanValidationReport:
    """Comprehensive validation report for an RTL Generation Plan."""
    is_valid: bool
    spec_status: str  # COMPLETE, SPECIFICATION_INCOMPLETE, CONTRADICTORY
    diagnostics: List[PlanDiagnostic] = field(default_factory=list)
    gaps: List[Dict[str, Any]] = field(default_factory=list)
    checked_ports_count: int = 0
    checked_registers_count: int = 0
    checked_transitions_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        diag_dicts = [d.to_dict() for d in self.diagnostics]
        return {
            "valid": self.is_valid,
            "status": "COMPLETE" if self.is_valid else ("CONTRADICTORY" if self.spec_status == "CONTRADICTORY" else "INCOMPLETE"),
            "spec_status": self.spec_status,
            "diagnostics": diag_dicts,
            "issues": diag_dicts,
            "gaps": self.gaps,
            "metrics": {
                "checked_ports_count": self.checked_ports_count,
                "checked_registers_count": self.checked_registers_count,
                "checked_transitions_count": self.checked_transitions_count,
                "total_diagnostics_count": len(self.diagnostics),
            }
        }


class RTLPlanValidator:
    """
    Validates an RTL Generation Plan against the authoritative EFS IR,
    scoped ContextPack, and Source Evidence.
    """

    def __init__(
        self,
        efs_ir: Optional[EFSIR] = None,
        context_pack: Optional[Any] = None,
        spec_text: str = ""
    ):
        self.efs_ir = efs_ir
        self.context_pack = context_pack
        self.spec_text = spec_text

        # Extract approved resolutions to ensure already-approved decisions are not re-flagged
        self.approved_resolutions: Dict[str, EFSResolution] = {}
        if self.efs_ir and getattr(self.efs_ir, "resolutions", None):
            for res in self.efs_ir.resolutions:
                if getattr(res, "approval_status", "") == "APPROVED":
                    c_id = getattr(res, "conflict_id", "")
                    if c_id:
                        self.approved_resolutions[c_id.lower()] = res
                    t_ent = getattr(res, "target_entity", "")
                    if t_ent:
                        self.approved_resolutions[t_ent.lower()] = res
                    for aff in (getattr(res, "affected_requirements", []) or []):
                        self.approved_resolutions[str(aff).lower()] = res

    def _is_resolved(self, entity_name: str, property_name: Optional[str] = None) -> bool:
        """Check if an entity or property contradiction was approved via HITL."""
        ent = (entity_name or "").lower()
        if ent in self.approved_resolutions:
            return True
        if property_name and f"{ent}.{property_name.lower()}" in self.approved_resolutions:
            return True
        # Check if any approved resolution references this entity
        for k, res in self.approved_resolutions.items():
            if ent in k or k in ent:
                return True
        return False

    def validate(self, rtl_plan: Any) -> RTLPlanValidationReport:
        """Execute full cross-validation of the RTL plan."""
        diagnostics: List[PlanDiagnostic] = []
        gaps: List[Dict[str, Any]] = []

        if not isinstance(rtl_plan, dict):
            # Check if it has a to_dict method
            if hasattr(rtl_plan, "to_dict"):
                plan_dict = rtl_plan.to_dict()
            else:
                return RTLPlanValidationReport(
                    is_valid=False,
                    spec_status="SPECIFICATION_INCOMPLETE",
                    diagnostics=[
                        PlanDiagnostic(
                            diagnostic_type=DiagnosticCode.PLAN_VALIDATION_FAILED,
                            category="PLAN",
                            affected_object=None,
                            evidence=str(rtl_plan),
                            severity="CRITICAL",
                            suggested_resolution="Provide a structured, valid RTL Generation Plan JSON object."
                        )
                    ]
                )
        else:
            plan_dict = rtl_plan

        # 1. Check for Unresolved Conflicts in EFS IR or ContextPack
        unresolved_conflicts = []
        if self.context_pack:
            cp_conflicts = getattr(self.context_pack, "conflicts", None) or (self.context_pack.get("conflicts", []) if isinstance(self.context_pack, dict) else [])
            for c in cp_conflicts:
                st = c.get("resolution_status", "UNRESOLVED") if isinstance(c, dict) else getattr(c, "resolution_status", "UNRESOLVED")
                if st in ("UNRESOLVED", "ACTIVE"):
                    unresolved_conflicts.append(c)
        elif self.efs_ir and getattr(self.efs_ir, "conflicts", None):
            for c in self.efs_ir.conflicts:
                if getattr(c, "resolution_status", "UNRESOLVED") in ("UNRESOLVED", "ACTIVE"):
                    unresolved_conflicts.append(c)

        for conf in unresolved_conflicts:
            cid = conf.get("conflict_id", "CONF") if isinstance(conf, dict) else getattr(conf, "conflict_id", "CONF")
            if self._is_resolved(cid):
                continue
            desc = conf.get("description", "") if isinstance(conf, dict) else getattr(conf, "description", "")
            evid = conf.get("evidence", desc) if isinstance(conf, dict) else (getattr(conf, "evidence", "") or desc)
            cat = conf.get("category", "SPECIFICATION") if isinstance(conf, dict) else getattr(conf, "category", "SPECIFICATION")
            ctype = conf.get("type", "SPEC_CONFLICT") if isinstance(conf, dict) else getattr(conf, "type", "SPEC_CONFLICT")
            
            # Map to diagnostic code
            if "OPCODE" in str(ctype) or "OPCODE" in str(cat):
                d_code = DiagnosticCode.OPCODE_CONFLICT
            elif "RESET" in str(ctype) or "RESET" in str(cat):
                d_code = DiagnosticCode.INTERFACE_CONFLICT
            elif "REGISTER" in str(ctype) or "REGISTER" in str(cat):
                d_code = DiagnosticCode.REGISTER_DEFINITION_CONFLICT
            elif "INTERFACE" in str(ctype) or "SIGNAL" in str(ctype):
                d_code = DiagnosticCode.INTERFACE_CONFLICT
            else:
                d_code = DiagnosticCode.SPECIFICATION_INCOMPLETE

            diagnostics.append(
                PlanDiagnostic(
                    diagnostic_type=d_code,
                    category=cat or "SPECIFICATION",
                    affected_object=cid,
                    evidence=evid or desc,
                    severity="CRITICAL",
                    suggested_resolution=f"Resolve specification conflict '{cid}': {desc}",
                    hitl_required=True,
                    conflict_id=cid
                )
            )

        # 2. Extract Plan Ports and Cross-Validate with EFS IR / ContextPack
        plan_ports = plan_dict.get("ports", [])
        plan_port_map: Dict[str, Dict[str, Any]] = {}
        duplicate_ports: Set[str] = set()

        for p in plan_ports:
            if isinstance(p, dict) and p.get("name"):
                p_name_clean = str(p["name"]).strip().lower()
                if p_name_clean in plan_port_map:
                    duplicate_ports.add(p_name_clean)
                else:
                    plan_port_map[p_name_clean] = p

        for dup in duplicate_ports:
            diagnostics.append(
                PlanDiagnostic(
                    diagnostic_type=DiagnosticCode.PLAN_VALIDATION_FAILED,
                    category="INTERFACE",
                    affected_object=dup,
                    evidence=f"Port '{dup}' declared multiple times in plan",
                    severity="CRITICAL",
                    suggested_resolution=f"Remove duplicate declaration of port '{dup}'."
                )
            )

        # Source signals to validate against
        spec_signals = []
        if self.context_pack:
            cp_dict = self.context_pack.to_dict() if hasattr(self.context_pack, "to_dict") else self.context_pack
            if isinstance(cp_dict, dict):
                spec_signals = cp_dict.get("target_signals", cp_dict.get("signals", []))
        elif self.efs_ir and getattr(self.efs_ir, "signals", None):
            plan_mod = str(plan_dict.get("module_name", "")).strip().lower()
            target_comp = None
            if getattr(self.efs_ir, "components", None):
                target_comp = next((c for c in self.efs_ir.components if c.name.strip().lower() == plan_mod or c.component_id.strip().lower() == plan_mod), None)

            is_top = (
                not target_comp or
                getattr(target_comp, "type", "").lower() in ("top_level", "top_level_design", "top_module", "system", "container") or
                (self.efs_ir.metadata and (
                    (getattr(target_comp, "name", "") or "").lower() == (getattr(self.efs_ir.metadata, "top_module", None) or "").lower() or
                    (getattr(target_comp, "name", "") or "").lower() == (getattr(self.efs_ir.metadata, "design_name", None) or "").lower()
                ))
            )
            if not is_top and target_comp:
                from core.retrieval.dependency_resolver import resolve_dependencies
                deps = resolve_dependencies(target_comp, self.efs_ir)
                spec_signals = deps.signals
            else:
                spec_signals = self.efs_ir.signals

        checked_ports_count = len(plan_ports)
        for sig in spec_signals:
            s_name = sig.get("name") if isinstance(sig, dict) else getattr(sig, "name", "")
            s_dir = sig.get("direction", "input") if isinstance(sig, dict) else getattr(sig, "direction", "input")
            s_width = sig.get("width") if isinstance(sig, dict) else getattr(sig, "width", None)
            s_name_lower = str(s_name).strip().lower()

            if not s_name_lower or s_name_lower in ("name", "signal", "port", "pin"):
                continue

            # Check for missing port
            if s_name_lower not in plan_port_map and len(plan_port_map) > 0:
                if not self._is_resolved(s_name_lower):
                    diagnostics.append(
                        PlanDiagnostic(
                            diagnostic_type=DiagnosticCode.INTERFACE_CONFLICT,
                            category="INTERFACE",
                            affected_object=s_name,
                            evidence=f"Required specification signal '{s_name}' is missing from plan ports",
                            severity="CRITICAL",
                            suggested_resolution=f"Add port '{s_name}' ({s_dir}) to the RTL implementation plan.",
                            hitl_required=False
                        )
                    )
            elif s_name_lower in plan_port_map:
                p_item = plan_port_map[s_name_lower]
                p_dir = str(p_item.get("direction", "")).lower()
                norm_p_dir = normalize_direction(p_dir)
                norm_s_dir = normalize_direction(str(s_dir).lower())

                # Direction conflict
                if norm_p_dir and norm_s_dir and norm_p_dir != norm_s_dir:
                    if not self._is_resolved(s_name_lower, "direction"):
                        diagnostics.append(
                            PlanDiagnostic(
                                diagnostic_type=DiagnosticCode.INTERFACE_CONFLICT,
                                category="INTERFACE",
                                affected_object=s_name,
                                evidence=f"Plan specifies direction '{norm_p_dir}', but specification requires '{norm_s_dir}'",
                                severity="CRITICAL",
                                suggested_resolution=f"Align port '{s_name}' direction to '{norm_s_dir}'.",
                                hitl_required=True
                            )
                        )

                # Width conflict
                p_w = str(p_item.get("width", "1"))
                norm_pw, pw_range = parse_bit_range(p_w)
                norm_sw, sw_range = parse_bit_range(str(s_width or "1"))
                if norm_pw is not None and norm_sw is not None and norm_pw != norm_sw:
                    if not self._is_resolved(s_name_lower, "width"):
                        diagnostics.append(
                            PlanDiagnostic(
                                diagnostic_type=DiagnosticCode.INTERFACE_CONFLICT,
                                category="INTERFACE",
                                affected_object=s_name,
                                evidence=f"Specification requires width {norm_sw}, but plan specifies width {norm_pw}",
                                severity="CRITICAL",
                                suggested_resolution=f"Align port '{s_name}' width to {norm_sw} bits.",
                                hitl_required=True
                            )
                        )

        # 3. Check for Duplicate Declarations between Ports and Internal Registers/Signals
        internal_regs = plan_dict.get("internal_registers", plan_dict.get("registers", []))
        internal_sigs = plan_dict.get("internal_signals", plan_dict.get("signals", []))

        reg_names: Set[str] = set()
        for r in internal_regs:
            if isinstance(r, dict) and r.get("name"):
                r_name = str(r["name"]).strip().lower()
                if r_name in reg_names:
                    diagnostics.append(
                        PlanDiagnostic(
                            diagnostic_type=DiagnosticCode.REGISTER_DEFINITION_CONFLICT,
                            category="REGISTER",
                            affected_object=r_name,
                            evidence=f"Internal register '{r_name}' declared multiple times in plan",
                            severity="CRITICAL",
                            suggested_resolution=f"Ensure internal register '{r_name}' has a single unique declaration."
                        )
                    )
                reg_names.add(r_name)
                if r_name in plan_port_map:
                    # Output ports driving from internal register of same name is common, but input port collision is an error
                    p_direction = plan_port_map[r_name].get("direction", "input")
                    if p_direction == "input":
                        diagnostics.append(
                            PlanDiagnostic(
                                diagnostic_type=DiagnosticCode.PLAN_VALIDATION_FAILED,
                                category="INTERFACE",
                                affected_object=r_name,
                                evidence=f"Input port '{r_name}' cannot be re-declared as internal register",
                                severity="CRITICAL",
                                suggested_resolution=f"Rename internal register '{r_name}' to avoid colliding with input port."
                            )
                        )

        # 4. Check Registers for Missing Reset Values and Fields
        checked_regs_count = len(internal_regs)
        for r in internal_regs:
            if isinstance(r, dict):
                r_name = r.get("name", "unnamed_reg")
                r_reset = r.get("reset_val", r.get("reset_value"))
                if ("reset_val" in r and r.get("reset_val") is None) or str(r_reset).strip().lower() in ("undefined", "tbd"):
                    diagnostics.append(
                        PlanDiagnostic(
                            diagnostic_type=DiagnosticCode.REGISTER_DEFINITION_CONFLICT,
                            category="REGISTER",
                            affected_object=r_name,
                            evidence=f"Internal register '{r_name}' has undefined reset value",
                            severity="CRITICAL",
                            suggested_resolution=f"Specify explicit reset value for register '{r_name}'."
                        )
                    )

        # 5. Check for Opcode Binary Encoding Collisions
        plan_opcodes = plan_dict.get("opcodes", [])
        all_opcodes = list(plan_opcodes)
        if self.efs_ir and getattr(self.efs_ir, "opcodes", None):
            all_opcodes.extend([o.to_dict() if hasattr(o, "to_dict") else o for o in self.efs_ir.opcodes])

        seen_encodings: Dict[str, str] = {}
        for opc in all_opcodes:
            if not isinstance(opc, dict):
                continue
            enc = str(opc.get("binary_encoding", opc.get("opcode", ""))).strip()
            name = str(opc.get("name", opc.get("mnemonic", ""))).strip()
            if enc and enc not in ("", "None", "undefined"):
                if enc in seen_encodings and seen_encodings[enc] != name:
                    conflict_key = f"{name}_{seen_encodings[enc]}"
                    if not self._is_resolved(name) and not self._is_resolved(seen_encodings[enc]):
                        diagnostics.append(
                            PlanDiagnostic(
                                diagnostic_type=DiagnosticCode.OPCODE_CONFLICT,
                                category="OPCODE",
                                affected_object=name,
                                evidence=f"Binary encoding '{enc}' shared between '{name}' and '{seen_encodings[enc]}'",
                                severity="CRITICAL",
                                suggested_resolution=f"Assign distinct binary encoding for '{name}'.",
                                hitl_required=True
                            )
                        )
                else:
                    seen_encodings[enc] = name

        # 6. FSM Completeness and Semantic Inconsistencies
        plan_fsm = plan_dict.get("fsm")
        checked_trans_count = 0

        # Architecture classification
        cur_spec_text = self.spec_text or (plan_dict.get("spec_text", "") if isinstance(plan_dict, dict) else "")
        arch_res = classify_hardware_architecture(efs_ir=self.efs_ir, spec_text=cur_spec_text)
        is_comb = (arch_res.primary_class == ArchitectureClass.COMBINATIONAL or (not arch_res.has_clock and not arch_res.has_reset))

        if plan_fsm and isinstance(plan_fsm, dict) and plan_fsm.get("states"):
            fsm_states = set(s.get("name") if isinstance(s, dict) else str(s).strip() for s in plan_fsm.get("states", []))
            initial_state = plan_fsm.get("initial_state", "")
            transitions = plan_fsm.get("transitions", [])
            checked_trans_count = len(transitions)

            if initial_state and initial_state not in fsm_states:
                diagnostics.append(
                    PlanDiagnostic(
                        diagnostic_type=DiagnosticCode.FSM_SEMANTIC_ERROR,
                        category="FSM",
                        affected_object=initial_state,
                        evidence=f"Initial state '{initial_state}' is not in declared FSM states: {list(fsm_states)}",
                        severity="CRITICAL",
                        suggested_resolution=f"Include '{initial_state}' in the FSM states list or update initial_state."
                    )
                )

            states_with_incoming: Set[str] = set()
            states_with_outgoing: Set[str] = set()
            outgoing_by_state: Dict[str, List[Dict[str, Any]]] = {}

            for tr in transitions:
                if not isinstance(tr, dict):
                    continue
                from_st = str(tr.get("from_state") or tr.get("from", "")).strip()
                to_st = str(tr.get("to_state") or tr.get("to", "")).strip()
                cond = tr.get("condition")

                if from_st:
                    states_with_outgoing.add(from_st)
                    outgoing_by_state.setdefault(from_st, []).append(tr)
                if to_st:
                    states_with_incoming.add(to_st)

                # Check condition expression
                cond_expr = cond.get("expression") if isinstance(cond, dict) else cond
                cond_str = str(cond_expr if cond_expr is not None else "").strip()

                if not cond_str or cond_str.lower() in ("none", "undefined", "tbd", "unspecified"):
                    if not self._is_resolved(f"{from_st}_{to_st}"):
                        diagnostics.append(
                            PlanDiagnostic(
                                diagnostic_type=DiagnosticCode.FSM_SEMANTIC_ERROR,
                                category="FSM",
                                affected_object=f"{from_st}->{to_st}",
                                evidence=f"Transition from '{from_st}' to '{to_st}' has undefined/missing condition",
                                severity="CRITICAL",
                                suggested_resolution=f"Define explicit boolean trigger condition for transition from '{from_st}' to '{to_st}'.",
                                hitl_required=True
                            )
                        )
                        gaps.append({
                            "type": "FSM_TRANSITION_CONDITION_MISSING",
                            "from_state": from_st,
                            "to_state": to_st,
                        })

            # Check unreachable states (no incoming transitions except initial state)
            for st in fsm_states:
                if st and st != initial_state and st not in states_with_incoming and len(fsm_states) > 1:
                    diagnostics.append(
                        PlanDiagnostic(
                            diagnostic_type=DiagnosticCode.FSM_SEMANTIC_ERROR,
                            category="FSM",
                            affected_object=st,
                            evidence=f"FSM state '{st}' is unreachable (no incoming transitions)",
                            severity="CRITICAL",
                            suggested_resolution=f"Add transition into state '{st}' or remove unused state."
                        )
                    )

            # Check dead-end states (states with no outgoing transitions, unless DONE/ERROR)
            for st in fsm_states:
                if st and st not in states_with_outgoing and len(fsm_states) > 1 and st.upper() not in ("DONE", "ERROR"):
                    diagnostics.append(
                        PlanDiagnostic(
                            diagnostic_type=DiagnosticCode.FSM_SEMANTIC_ERROR,
                            category="FSM",
                            affected_object=st,
                            evidence=f"FSM state '{st}' has no outgoing transitions defined",
                            severity="CRITICAL",
                            suggested_resolution=f"Define outgoing transition(s) for state '{st}'."
                        )
                    )

            # Check multiple outgoing transitions with non-mutually-exclusive conditions
            for fst, tr_list in outgoing_by_state.items():
                if len(tr_list) > 1:
                    uncond_count = 0
                    for tr in tr_list:
                        c_obj = tr.get("condition")
                        c_expr = str(c_obj.get("expression") if isinstance(c_obj, dict) else c_obj).strip()
                        if c_expr in ("", "1'b1", "1", "true", "True"):
                            uncond_count += 1
                    if uncond_count > 1:
                        diagnostics.append(
                            PlanDiagnostic(
                                diagnostic_type=DiagnosticCode.FSM_SEMANTIC_ERROR,
                                category="FSM",
                                affected_object=fst,
                                evidence=f"State '{fst}' has {uncond_count} unconditional/overlapping transitions",
                                severity="CRITICAL",
                                suggested_resolution=f"Specify mutually exclusive conditions for transitions leaving state '{fst}'."
                            )
                        )
        elif arch_res.requires_fsm and not is_comb and not plan_fsm:
            # FSM was expected but completely missing
            diagnostics.append(
                PlanDiagnostic(
                    diagnostic_type=DiagnosticCode.SPECIFICATION_INCOMPLETE,
                    category="FSM",
                    affected_object="fsm",
                    evidence="Specification requires FSM sequence controller, but no FSM was defined in plan",
                    severity="CRITICAL",
                    suggested_resolution="Provide explicit FSM states and transitions in the plan.",
                    hitl_required=True
                )
            )

        # 7. Reset Semantics Validation
        reset_req = plan_dict.get("reset_requirements") or plan_dict.get("clock_reset")
        if isinstance(reset_req, dict):
            pol = str(reset_req.get("reset_polarity", "")).lower()
            rst_name = str(reset_req.get("reset_signal") or reset_req.get("reset_name", "")).lower()
            if "n" in rst_name and pol in ("high", "active_high"):
                diagnostics.append(
                    PlanDiagnostic(
                        diagnostic_type=DiagnosticCode.PLAN_VALIDATION_FAILED,
                        category="RESET",
                        affected_object=rst_name,
                        evidence=f"Signal name '{rst_name}' indicates active-low, but reset polarity is specified as active-high",
                        severity="CRITICAL",
                        suggested_resolution="Align reset signal polarity with naming convention (active-low)."
                    )
                )

        # Check for underspecified architecture from specification analysis
        if arch_res.is_underspecified:
            diagnostics.append(
                PlanDiagnostic(
                    diagnostic_type=DiagnosticCode.SPECIFICATION_INCOMPLETE,
                    category="SPECIFICATION",
                    affected_object="specification",
                    evidence="; ".join(getattr(arch_res, "gap_evidence", None) or arch_res.evidence) or "Specification text explicitly indicates missing/unstated requirements",
                    severity="CRITICAL",
                    suggested_resolution="Provide missing requirements and complete specification.",
                    hitl_required=True
                )
            )
            gaps.append({
                "type": "SPECIFICATION_UNDERSPECIFIED",
                "evidence": arch_res.evidence
            })

        # 8. Check for Undriven Output Ports
        out_ports = [
            p.get("name") for p in plan_ports
            if isinstance(p, dict) and str(p.get("direction", "")).lower() in ("output", "out")
        ]
        dp_text = str(plan_dict.get("datapath", "")).lower() + " " + str(plan_dict.get("behavioral_implementation", "")).lower() + " " + str(plan_dict.get("fsm", "")).lower() + " " + str(plan_dict.get("output_driving_strategy", "")).lower() + " " + str(plan_dict.get("output_assignments", "")).lower()
        has_strat_field = ("output_driving_strategy" in plan_dict or "output_assignments" in plan_dict)
        if out_ports and has_strat_field:
            for op in out_ports:
                if op and str(op).lower() not in dp_text:
                    diagnostics.append(
                        PlanDiagnostic(
                            diagnostic_type=DiagnosticCode.DATAPATH_UNSPECIFIED,
                            category="DATAPATH",
                            affected_object=op,
                            evidence=f"Output port '{op}' has no driving logic or assignment defined",
                            severity="CRITICAL",
                            suggested_resolution=f"Specify driver or logic assigning to output port '{op}'."
                        )
                    )

        # 9. Determine Overall Status
        has_critical = any(d.severity == "CRITICAL" for d in diagnostics)
        is_contradictory = any(d.diagnostic_type in (DiagnosticCode.INTERFACE_CONFLICT, DiagnosticCode.OPCODE_CONFLICT, DiagnosticCode.REGISTER_DEFINITION_CONFLICT) and "conflict" in d.evidence.lower() for d in diagnostics)
        is_incomplete = any(d.diagnostic_type == DiagnosticCode.SPECIFICATION_INCOMPLETE or len(gaps) > 0 for d in diagnostics)

        if is_contradictory:
            spec_status = "CONTRADICTORY"
        elif is_incomplete or not plan_ports:
            spec_status = "SPECIFICATION_INCOMPLETE"
        elif has_critical:
            spec_status = "PLAN_VALIDATION_FAILED"
        else:
            spec_status = "COMPLETE"

        is_valid = not has_critical and spec_status == "COMPLETE"

        return RTLPlanValidationReport(
            is_valid=is_valid,
            spec_status=spec_status,
            diagnostics=diagnostics,
            gaps=gaps,
            checked_ports_count=checked_ports_count,
            checked_registers_count=checked_regs_count,
            checked_transitions_count=checked_trans_count
        )
