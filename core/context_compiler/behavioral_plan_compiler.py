"""
Behavioral Plan Compiler module.

Compiles behavioral hardware semantics from canonical EFS IR, ContextResolver,
and GenerationContext into complete, structured RTL Generation Plan elements:
- PlanSequentialProcess & PlanRegisterUpdate
- PlanCombinationalProcess & PlanCombinationalAssignment
- PlanExpression (machine-readable structured AST)
- PlanOrigin & DerivationMetadata (provenance)
- PlanUnresolvedLogic (explicit specification gaps)

Constraint Guarantees:
- Generic and architecture-independent (no hardcoded module or signal names).
- Never fabricates ungrounded register update equations.
- Preserves explicit provenance (INHERITED vs. DERIVED vs. IMPLEMENTATION_DECISION).
- Integrates seamlessly with plan_rtl() and RTLGenerationPlan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from core.efs_ir.models import (
    EFSIR, EFSComponent, EFSSignal, EFSRegister, EFSFSM, EFSOpcode, EFSInstruction
)
from core.efs_ir.rtl_plan import (
    PlanSequentialProcess,
    PlanRegisterUpdate,
    PlanCombinationalProcess,
    PlanCombinationalAssignment,
    PlanExpression,
    PlanUnresolvedLogic,
    PlanOrigin,
    DerivationMetadata,
    PlanPort,
    PlanRegister,
    PlanParameter,
    PlanOpcode,
    PlanProtocolRule,
    PlanProtocolContract,
)
from utils.logger import get_logger

logger = get_logger("context_compiler.behavioral_plan_compiler")


@dataclass
class NormalizedProtocolRule:
    """Canonical internal normalized representation of a protocol rule."""
    rule_id: str
    name: str
    rule_type: str
    description: str
    interface: str
    protocol: str
    component: str
    signals: List[str]
    raw_signals: List[Any]
    condition: Optional[str] = None
    handshake_condition: Optional[str] = None
    reset_requirement: Optional[str] = None
    stability_constraint: Optional[str] = None
    timing_constraint: Optional[str] = None
    is_synthesizable_behavior: bool = True
    verification_only: bool = False
    source_evidence: str = ""
    chunk_id: Optional[str] = None
    citation: Optional[str] = None
    raw_rule: Dict[str, Any] = field(default_factory=dict)


def normalize_protocol_rule(rule: Any) -> Optional[NormalizedProtocolRule]:
    """
    Centralized normalization mechanism handling:
    - Dictionary-based rules
    - Pydantic/dataclass/model-based rules (e.g. EFSProtocolRule)
    - Nested 'details' dictionaries merged with top-level fields
    - Missing or optional attributes
    - Signals represented by IDs, names, or structured objects (EFSSignal)
    - Preserves all source fields without mutating input objects.
    """
    if rule is None:
        return None

    if hasattr(rule, "to_dict") and callable(rule.to_dict):
        raw = rule.to_dict()
    elif isinstance(rule, dict):
        raw = dict(rule)
    elif hasattr(rule, "__dict__"):
        raw = dict(rule.__dict__)
    else:
        return None

    if not isinstance(raw, dict):
        return None

    # Merge nested details dictionary (if any)
    details = raw.get("details", {}) if isinstance(raw.get("details"), dict) else {}

    # 1. Rule ID
    rule_id = str(raw.get("rule_id") or raw.get("id") or details.get("rule_id") or details.get("id") or "").strip()

    # 2. Rule Name
    name = str(raw.get("name") or details.get("name") or raw.get("section") or details.get("section") or rule_id or "PROTOCOL_RULE").strip()

    # 3. Description / Text
    description = str(
        raw.get("description") or details.get("description") or
        raw.get("text") or details.get("text") or
        raw.get("rule_statement") or details.get("rule_statement") or
        raw.get("original_text") or details.get("original_text") or
        raw.get("property_statement") or details.get("property_statement") or ""
    ).strip()

    # If completely empty rule (no id, no name, no description, no signals), discard as malformed
    has_signals = bool(raw.get("signals") or details.get("signals") or raw.get("applicable_signals") or details.get("applicable_signals"))
    if not rule_id and not description and not has_signals:
        return None

    # 4. Interface, Protocol, Component
    interface = str(
        raw.get("interface") or details.get("interface") or
        raw.get("interface_name") or details.get("interface_name") or
        raw.get("interface_id") or details.get("interface_id") or ""
    ).strip()

    protocol = str(
        raw.get("protocol") or details.get("protocol") or
        raw.get("protocol_name") or details.get("protocol_name") or ""
    ).strip()

    component = str(
        raw.get("component") or details.get("component") or
        raw.get("affected_component") or details.get("affected_component") or
        raw.get("target_component") or details.get("target_component") or ""
    ).strip()

    # 5. Rule Type
    raw_type = str(raw.get("type") or raw.get("rule_type") or details.get("type") or details.get("rule_type") or "").strip()

    # 6. Signals Normalization (handles strings, dicts, EFSSignal objects)
    raw_signals_list = raw.get("signals") or details.get("signals") or raw.get("applicable_signals") or details.get("applicable_signals") or []
    if not isinstance(raw_signals_list, list):
        raw_signals_list = [raw_signals_list]

    normalized_signals: List[str] = []
    for sig in raw_signals_list:
        if isinstance(sig, str):
            s_name = sig.strip()
        elif isinstance(sig, dict):
            s_name = str(sig.get("name") or sig.get("signal_id") or sig.get("id") or "").strip()
        elif hasattr(sig, "name") and sig.name:
            s_name = str(sig.name).strip()
        elif hasattr(sig, "signal_id") and sig.signal_id:
            s_name = str(sig.signal_id).strip()
        else:
            s_name = str(sig).strip()
        if s_name and s_name.lower() not in [s.lower() for s in normalized_signals]:
            normalized_signals.append(s_name)

    # 7. Conditions & Constraints
    condition = raw.get("condition") or details.get("condition")
    if condition is not None:
        condition = str(condition).strip()

    handshake_condition = (
        raw.get("handshake_condition") or details.get("handshake_condition") or
        condition
    )
    if handshake_condition is not None:
        handshake_condition = str(handshake_condition).strip()

    reset_requirement = (
        raw.get("reset_requirement") or details.get("reset_requirement") or
        raw.get("reset_behavior") or details.get("reset_behavior")
    )
    if reset_requirement is not None:
        reset_requirement = str(reset_requirement).strip()

    stability_constraint = raw.get("stability_constraint") or details.get("stability_constraint")
    if stability_constraint is not None:
        stability_constraint = str(stability_constraint).strip()

    timing_constraint = raw.get("timing_constraint") or details.get("timing_constraint")
    if timing_constraint is not None:
        timing_constraint = str(timing_constraint).strip()

    # 8. Verification-Only & Synthesizability
    desc_lower = description.lower()
    type_lower = raw_type.lower()
    is_v_explicit = (
        bool(raw.get("verification_only", False)) or
        bool(details.get("verification_only", False)) or
        raw.get("is_synthesizable_behavior") is False or
        details.get("is_synthesizable_behavior") is False
    )
    is_v_text = any(kw in desc_lower for kw in (
        "must never", "shall not", "illegal", "assertion", "property check",
        "property assertion", "formal check", "invariant"
    ))
    verification_only = is_v_explicit or is_v_text

    if raw.get("is_synthesizable_behavior") is not None:
        is_synthesizable = bool(raw.get("is_synthesizable_behavior"))
    elif details.get("is_synthesizable_behavior") is not None:
        is_synthesizable = bool(details.get("is_synthesizable_behavior"))
    elif verification_only or type_lower in ("verification_only", "timing", "ordering"):
        is_synthesizable = False
    else:
        is_synthesizable = True

    # 9. Source Evidence, Chunk ID, Citation
    citation = str(raw.get("citation") or details.get("citation") or "")
    chunk_id = str(raw.get("chunk_id") or details.get("chunk_id") or "")
    source_evidence = str(
        raw.get("source_evidence") or details.get("source_evidence") or
        citation or chunk_id or ""
    )
    if not source_evidence and raw.get("traceability"):
        tr = raw.get("traceability")
        if isinstance(tr, dict) and tr.get("source_chunks"):
            source_evidence = str(tr["source_chunks"][0])
        elif hasattr(tr, "source_chunks") and tr.source_chunks:
            source_evidence = str(tr.source_chunks[0])

    return NormalizedProtocolRule(
        rule_id=rule_id,
        name=name,
        rule_type=raw_type,
        description=description,
        interface=interface,
        protocol=protocol,
        component=component,
        signals=normalized_signals,
        raw_signals=raw_signals_list,
        condition=condition,
        handshake_condition=handshake_condition,
        reset_requirement=reset_requirement,
        stability_constraint=stability_constraint,
        timing_constraint=timing_constraint,
        is_synthesizable_behavior=is_synthesizable,
        verification_only=verification_only,
        source_evidence=source_evidence,
        chunk_id=chunk_id if chunk_id else None,
        citation=citation if citation else None,
        raw_rule=raw
    )


class BehavioralPlanCompiler:
    """Compiles EFS IR behavioral semantics into structured sequential and combinational plan directives."""

    @classmethod
    def compile_and_populate(
        cls,
        plan_dict: Dict[str, Any],
        efs_ir: Optional[EFSIR] = None,
        target_comp: Optional[EFSComponent] = None,
        compiled_rtl: Optional[Dict[str, Any]] = None,
        design_spec: str = "",
        context_pack: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Populate plan_dict with structured sequential_logic, combinational_logic,
        unresolved_logic, and accurate entity provenance.
        """
        logger.info(f"Compiling behavioral plan directives for module '{plan_dict.get('module_name', 'target_module')}'...")

        if compiled_rtl is None:
            compiled_rtl = {}

        module_name = plan_dict.get("module_name", "hardware_module")
        ports = plan_dict.get("ports", [])
        registers = plan_dict.get("internal_registers", [])
        fsm_data = plan_dict.get("fsm") or compiled_rtl.get("fsm") or {}
        opcodes = plan_dict.get("opcodes") or compiled_rtl.get("opcodes") or []
        output_assigns = dict(compiled_rtl.get("output_assignments", {}) or plan_dict.get("output_assignments", {}) or {})

        # 1. Compile Provenance for Ports, Registers, and Parameters
        cls._assign_provenance(plan_dict, efs_ir, compiled_rtl, design_spec)

        # 2. Compile Protocol Contract (component-relevant protocol rules & verification constraints)
        proto_contract, proto_unresolved = cls._compile_protocol_contract(
            module_name=module_name,
            ports=ports,
            target_comp=target_comp,
            compiled_rtl=compiled_rtl,
            context_pack=context_pack,
            efs_ir=efs_ir
        )

        # 3. Sequential Logic Compilation
        seq_processes, seq_unresolved = cls._compile_sequential_logic(
            module_name=module_name,
            plan_dict=plan_dict,
            registers=registers,
            fsm_data=fsm_data,
            efs_ir=efs_ir,
            compiled_rtl=compiled_rtl,
            target_comp=target_comp,
            protocol_contract=proto_contract
        )

        # 4. Combinational Logic Compilation
        comb_processes, comb_unresolved = cls._compile_combinational_logic(
            module_name=module_name,
            plan_dict=plan_dict,
            ports=ports,
            registers=registers,
            fsm_data=fsm_data,
            opcodes=opcodes,
            output_assigns=output_assigns,
            efs_ir=efs_ir,
            compiled_rtl=compiled_rtl,
            design_spec=design_spec,
            protocol_contract=proto_contract
        )

        # 5. Consolidate Unresolved Logic
        existing_unresolved = [
            PlanUnresolvedLogic.from_dict(u) if isinstance(u, dict) else u
            for u in plan_dict.get("unresolved_logic", [])
        ]
        all_unresolved = existing_unresolved + seq_unresolved + comb_unresolved + proto_unresolved
        # Deduplicate unresolved logic by (affected_component, affected_entity)
        dedup_unresolved: List[PlanUnresolvedLogic] = []
        seen_unres: Set[Tuple[str, str]] = set()
        for u in all_unresolved:
            key = (str(u.affected_component or ""), str(u.affected_entity or ""))
            if key not in seen_unres:
                seen_unres.add(key)
                dedup_unresolved.append(u)

        # 6. Populate plan_dict
        plan_dict["sequential_logic"] = [sp.to_dict() for sp in seq_processes]
        plan_dict["combinational_logic"] = [cp.to_dict() for cp in comb_processes]
        plan_dict["unresolved_logic"] = [u.to_dict() for u in dedup_unresolved]
        if proto_contract:
            plan_dict["protocol_contract"] = proto_contract.to_dict()

        logger.info(
            f"Behavioral plan compilation complete: "
            f"{len(seq_processes)} sequential processes, "
            f"{len(comb_processes)} combinational processes, "
            f"{len(dedup_unresolved)} unresolved logic directives."
        )

        return plan_dict

    @classmethod
    def _assign_provenance(
        cls,
        plan_dict: Dict[str, Any],
        efs_ir: Optional[EFSIR],
        compiled_rtl: Dict[str, Any],
        design_spec: str
    ) -> None:
        """Assign explicit PlanOrigin (INHERITED, DERIVED, IMPLEMENTATION_DECISION) across plan entities."""
        ir_sig_names = {s.name.lower() for s in (getattr(efs_ir, "signals", []) or [])} if efs_ir else set()
        ir_reg_names = {r.name.lower() for r in (getattr(efs_ir, "registers", []) or [])} if efs_ir else set()
        ir_param_names = {p.name.lower() for p in (getattr(efs_ir, "parameters", []) or [])} if efs_ir else set()

        spec_lower = design_spec.lower()

        # Ports
        for p in plan_dict.get("ports", []):
            if isinstance(p, dict):
                p_name = str(p.get("name", "")).lower()
                if p_name in ir_sig_names or (p_name and p_name in spec_lower):
                    p["origin"] = PlanOrigin.INHERITED.value
                elif p.get("semantic_role") in ("clock", "reset") or any(k in p_name for k in ("clk", "rst")):
                    p["origin"] = PlanOrigin.DERIVED.value
                    p["derivation"] = DerivationMetadata(
                        description="Derived from module clock/reset domain contract",
                        input_entities=["clock_domain", "reset_domain"],
                        rule="DOMAIN_INTERFACE_ASSOCIATION"
                    ).to_dict()
                elif p.get("derivation") or p.get("source_entity") or p.get("derived"):
                    p["origin"] = PlanOrigin.DERIVED.value
                    if not p.get("derivation"):
                        p["derivation"] = DerivationMetadata(
                            description="Derived from interface or architectural boundary requirements",
                            input_entities=[],
                            rule="INTERFACE_SYNTHESIS"
                        ).to_dict()
                else:
                    p["origin"] = PlanOrigin.IMPLEMENTATION_DECISION.value

        # Registers
        for r in plan_dict.get("internal_registers", []):
            if isinstance(r, dict):
                r_name = str(r.get("name", "")).lower()
                if r_name in ir_reg_names or (r_name and r_name in spec_lower):
                    r["origin"] = PlanOrigin.INHERITED.value
                elif r_name in ("state", "state_reg", "current_state", "timer_cnt"):
                    r["origin"] = PlanOrigin.DERIVED.value
                    r["derivation"] = DerivationMetadata(
                        description=f"Internal sequential state register derived for sequencing",
                        input_entities=["fsm_states"],
                        rule="STATE_REGISTER_SYNTHESIS"
                    ).to_dict()
                else:
                    r["origin"] = PlanOrigin.IMPLEMENTATION_DECISION.value

        # Parameters
        for pm in plan_dict.get("parameters", []):
            if isinstance(pm, dict):
                pm_name = str(pm.get("name", "")).lower()
                if pm_name in ir_param_names or (pm_name and pm_name in spec_lower):
                    pm["origin"] = PlanOrigin.INHERITED.value
                elif pm.get("derivation") or pm.get("derived"):
                    pm["origin"] = PlanOrigin.DERIVED.value
                else:
                    pm["origin"] = PlanOrigin.IMPLEMENTATION_DECISION.value

    @classmethod
    def _compile_sequential_logic(
        cls,
        module_name: str,
        plan_dict: Dict[str, Any],
        registers: List[Dict[str, Any]],
        fsm_data: Dict[str, Any],
        efs_ir: Optional[EFSIR],
        compiled_rtl: Dict[str, Any],
        target_comp: Optional[EFSComponent],
        protocol_contract: Optional[PlanProtocolContract] = None
    ) -> Tuple[List[PlanSequentialProcess], List[PlanUnresolvedLogic]]:
        """Compile sequential logic always_ff processes from registers, FSM states, and behavioral rules."""
        seq_processes: List[PlanSequentialProcess] = []
        unresolved_list: List[PlanUnresolvedLogic] = []

        # If architecture is purely combinational, do not compile sequential processes
        arch_class = str(plan_dict.get("architecture_class", "")).upper()
        if "COMBINATIONAL" in arch_class and not registers and not fsm_data.get("states"):
            return seq_processes, unresolved_list

        # Resolve Clock and Reset Parameters
        clock_reset = compiled_rtl.get("clock_reset", [])
        reset_req = plan_dict.get("reset_requirements") or {}

        clk_sig = "clk"
        clk_edge = "POSEDGE"
        rst_sig = None
        rst_type = "ASYNC"
        rst_lvl = "LOW"

        # Detect clock signal
        for cr in clock_reset:
            if cr.get("type") == "clock":
                clk_sig = cr.get("name", "clk")
                clk_edge = "NEGEDGE" if "fall" in str(cr.get("edge", "")).lower() else "POSEDGE"
                break
        if clk_sig == "clk":
            for p in plan_dict.get("ports", []):
                if isinstance(p, dict) and (p.get("semantic_role") == "clock" or str(p.get("name", "")).lower() in ("clk", "aclk", "clock")):
                    clk_sig = p.get("name", "clk")
                    break

        # Detect reset signal
        for cr in clock_reset:
            if cr.get("type") == "reset":
                rst_sig = cr.get("name")
                rst_lvl = "HIGH" if "high" in str(cr.get("polarity", "")).lower() else "LOW"
                break
        if not rst_sig and isinstance(reset_req, dict):
            rst_sig = reset_req.get("reset_signal") or reset_req.get("reset_name")
            rst_lvl = "HIGH" if reset_req.get("reset_polarity") == "active_high" else "LOW"
            rst_type = "SYNC" if reset_req.get("reset_type") == "sync" else "ASYNC"
        if not rst_sig:
            for p in plan_dict.get("ports", []):
                if isinstance(p, dict) and (p.get("semantic_role") == "reset" or any(k in str(p.get("name", "")).lower() for k in ("rst", "aresetn", "reset"))):
                    rst_sig = p.get("name")
                    rst_lvl = "HIGH" if not any(k in rst_sig.lower() for k in ("_n", "rst_n", "reset_n", "low")) else "LOW"
                    break

        reset_actions: Dict[str, str] = {}
        register_updates: List[PlanRegisterUpdate] = []

        # 1. State Machine Sequential Updates
        fsm_states = fsm_data.get("states", [])
        if fsm_states:
            initial_state = fsm_data.get("initial_state") or fsm_states[0]
            state_reg_name = "current_state"
            if any(r.get("name") == "state" for r in registers if isinstance(r, dict)):
                state_reg_name = "state"
            elif any(r.get("name") == "state_reg" for r in registers if isinstance(r, dict)):
                state_reg_name = "state_reg"

            reset_actions[state_reg_name] = str(initial_state)
            register_updates.append(PlanRegisterUpdate(
                target=state_reg_name,
                condition="1'b1",
                enable="1'b1",
                next_value=PlanExpression(
                    expression_id=f"expr_{state_reg_name}_next",
                    expression_type="IDENTIFIER",
                    operands=["next_state"]
                ),
                hold_behavior="PRESERVE",
                assignment_priority=0,
                origin=PlanOrigin.DERIVED,
                derivation=DerivationMetadata(
                    description="FSM state register synchronous update from next-state logic",
                    input_entities=["next_state"],
                    rule="FSM_STATE_REGISTER_UPDATE"
                ),
                source_references=["FSM_SPECIFICATION"]
            ))

        # 2. Internal Registers Updates
        # Scan functional behavior from EFS IR
        functional_behaviors = getattr(efs_ir, "functional_behavior", []) or []
        datapath_ops = compiled_rtl.get("datapath", []) or []

        for reg in registers:
            if not isinstance(reg, dict):
                continue
            r_name = str(reg.get("name", "")).strip()
            if not r_name:
                continue

            # Skip state register if already handled above
            if r_name in reset_actions:
                continue

            r_width = str(reg.get("width", "1")).strip()
            r_reset = str(reg.get("reset_val") or reg.get("reset_value") or "0").strip()

            # Format reset expression appropriately
            if r_reset.isdigit():
                w_int = int(r_width) if r_width.isdigit() else 1
                r_reset_expr = f"{w_int}'d{r_reset}"
            elif re.match(r"^\d+'[bBhdD]", r_reset):
                r_reset_expr = r_reset
            else:
                r_reset_expr = f"{r_width}'d0" if r_width.isdigit() else "0"

            reset_actions[r_name] = r_reset_expr

            # Search for explicit update rules for this register
            update_found = False

            # Check functional behavior for matching assignment
            for fb in functional_behaviors:
                desc = str(getattr(fb, "description", "")).lower()
                action = str(getattr(fb, "action", "") or getattr(fb, "expression", "")).strip()
                if r_name.lower() in desc or r_name.lower() in action.lower():
                    # Check for explicit condition or enable
                    cond_expr = "1'b1"
                    enable_sig = None
                    raw_cond_part = ""
                    if "when " in desc:
                        raw_cond_part = desc.split("when ", 1)[1].strip()
                    elif "if " in desc:
                        raw_cond_part = desc.split("if ", 1)[1].strip()

                    if raw_cond_part:
                        if ":" in raw_cond_part:
                            cond_expr = raw_cond_part.split(":", 1)[0].strip()
                            if not action:
                                action = raw_cond_part.split(":", 1)[1].strip()
                        elif "<=" in raw_cond_part:
                            cond_expr = raw_cond_part.split("<=")[0].strip()
                        elif "," in raw_cond_part and not ("{" in raw_cond_part and "}" not in raw_cond_part.split(",")[0]):
                            cond_expr = raw_cond_part.split(",")[0].strip()
                        else:
                            cond_expr = raw_cond_part.strip()

                    # Sanitize condition expression (remove colon, assignments, unclosed braces)
                    if ":" in cond_expr:
                        cond_expr = cond_expr.split(":")[0].strip()
                    if "<=" in cond_expr:
                        cond_expr = cond_expr.split("<=")[0].strip()
                    cond_expr = cond_expr.replace("{", "").replace("}", "").strip()
                    if not cond_expr or cond_expr in ("1", "true", "1'b1"):
                        cond_expr = "1'b1"

                    # Extract next value
                    if not action and ":" in desc:
                        action = desc.split(":", 1)[1].strip()

                    if "<=" in action:
                        raw_next = action.split("<=")[1].strip().rstrip(";")
                    elif "=" in action:
                        raw_next = action.split("=")[1].strip().rstrip(";")
                    else:
                        raw_next = action or r_name

                    parsed_expr = PlanExpression.parse_expression(raw_next, expr_id=f"expr_{r_name}_update")
                    register_updates.append(PlanRegisterUpdate(
                        target=r_name,
                        condition=cond_expr,
                        enable=enable_sig,
                        next_value=parsed_expr,
                        hold_behavior="PRESERVE",
                        origin=PlanOrigin.INHERITED,
                        derivation=DerivationMetadata(
                            description=f"Register update for '{r_name}' extracted from functional behavior specification",
                            input_entities=parsed_expr.operands,
                            rule="SPECIFICATION_REGISTER_BEHAVIOR"
                        ),
                        source_references=getattr(fb, "source_requirements", []) or []
                    ))
                    update_found = True
                    break

            # If register represents timer/counter referenced in FSM or transitions
            if not update_found and "timer" in r_name.lower() or "cnt" in r_name.lower() or "counter" in r_name.lower():
                register_updates.append(PlanRegisterUpdate(
                    target=r_name,
                    condition="1'b1",
                    enable="1'b1",
                    next_value=PlanExpression(
                        expression_id=f"expr_{r_name}_inc",
                        expression_type="ARITHMETIC",
                        operator="+",
                        operands=[r_name, f"{r_width}'d1" if r_width.isdigit() else "1"]
                    ),
                    hold_behavior="PRESERVE",
                    origin=PlanOrigin.DERIVED,
                    derivation=DerivationMetadata(
                        description=f"Cycle counter increment logic for state dwell timer",
                        input_entities=[r_name],
                        rule="STATE_TIMER_INCREMENT"
                    ),
                    source_references=["FSM_CYCLE_COUNT"]
                ))
                update_found = True

            # Check if a protocol rule provides an explicit handshake transfer condition for this register
            if not update_found and protocol_contract:
                for rule in protocol_contract.rules:
                    if rule.handshake_condition and rule.is_synthesizable_behavior:
                        is_buffer_reg = any(k in r_name.lower() for k in ("data", "buffer", "payload", "reg", "fifo", "word", "item", "byte"))
                        if is_buffer_reg or r_name in rule.applicable_signals:
                            in_ports = [p.get("name") for p in plan_dict.get("ports", []) if isinstance(p, dict) and p.get("direction") == "input"]
                            matched_in = next((p for p in in_ports if any(k in p.lower() for k in ("data", "payload", "din", "wdata", "rx", "in"))), None)
                            load_source = matched_in if matched_in else r_name
                            parsed_load = PlanExpression.parse_expression(load_source, expr_id=f"expr_{r_name}_proto_load")

                            register_updates.append(PlanRegisterUpdate(
                                target=r_name,
                                condition=rule.handshake_condition,
                                enable=rule.handshake_condition,
                                next_value=parsed_load,
                                hold_behavior="PRESERVE",
                                origin=PlanOrigin.DERIVED,
                                derivation=DerivationMetadata(
                                    description=f"Register update enable derived from protocol rule '{rule.rule_id}'",
                                    input_entities=rule.applicable_signals + ([load_source] if load_source != r_name else []),
                                    rule="PROTOCOL_HANDSHAKE_TRANSFER"
                                ),
                                source_references=[rule.rule_id]
                            ))
                            update_found = True
                            break

            # Check if human-approved resolution defines update behavior for this register
            if not update_found and efs_ir and hasattr(efs_ir, "resolutions"):
                for res in getattr(efs_ir, "resolutions", []):
                    if getattr(res, "approval_status", "") == "APPROVED":
                        tent = str(getattr(res, "target_entity", "")).strip().lower()
                        affs = [str(a).strip().lower() for a in getattr(res, "affected_requirements", [])]
                        if tent == r_name.lower() or r_name.lower() in affs:
                            val = str(getattr(res, "resolved_value", "") or getattr(res, "proposed_change", "")).strip()
                            if val:
                                cond_expr = "1'b1"
                                raw_next = val
                                if "if (" in val:
                                    m = re.search(r"if\s*\(([^)]+)\)\s*(?:[A-Za-z0-9_]+\s*<=\s*([^;]+))?", val)
                                    if m:
                                        cond_expr = m.group(1).strip()
                                        if m.group(2):
                                            raw_next = m.group(2).strip()
                                elif "<=" in val:
                                    raw_next = val.split("<=")[1].strip().rstrip(";")
                                parsed_expr = PlanExpression.parse_expression(raw_next, expr_id=f"expr_{r_name}_approved")
                                register_updates.append(PlanRegisterUpdate(
                                    target=r_name,
                                    condition=cond_expr,
                                    next_value=parsed_expr,
                                    hold_behavior="PRESERVE",
                                    origin=PlanOrigin.INHERITED,
                                    derivation=DerivationMetadata(
                                        description=f"Human-approved register update for '{r_name}'",
                                        input_entities=[r_name],
                                        rule="HUMAN_APPROVED_REGISTER_UPDATE"
                                    )
                                ))
                                update_found = True
                                break

            # If register has no explicit update behavior in specification
            if not update_found:
                # Do NOT fabricate an arbitrary equation. Mark as unresolved logic!
                unresolved_list.append(PlanUnresolvedLogic(
                    affected_component=module_name,
                    affected_entity=r_name,
                    description=f"Register '{r_name}' is declared in the specification, but no update condition or next-value expression is defined.",
                    missing_information=f"Update condition, enable signal, or load equation for register '{r_name}'",
                    reason="Specification declared register without defining behavioral update transfers.",
                    severity="WARNING",
                    blocks_generation=False,
                    requires_human_decision=False
                ))
                # Add default preserve hold update
                register_updates.append(PlanRegisterUpdate(
                    target=r_name,
                    condition="1'b0",
                    enable="1'b0",
                    next_value=PlanExpression(
                        expression_id=f"expr_{r_name}_hold",
                        expression_type="IDENTIFIER",
                        operands=[r_name]
                    ),
                    hold_behavior="PRESERVE",
                    origin=PlanOrigin.IMPLEMENTATION_DECISION,
                    derivation=DerivationMetadata(
                        description="Default hold behavior for unassigned register",
                        input_entities=[r_name],
                        rule="DEFAULT_HOLD"
                    )
                ))

        # Check protocol contract for reset requirements on output ports / registers
        if protocol_contract:
            for rule in protocol_contract.rules:
                if (rule.rule_type == "RESET" or rule.reset_requirement) and rule.is_synthesizable_behavior:
                    for sig in rule.applicable_signals:
                        p_entry = next((p for p in plan_dict.get("ports", []) if isinstance(p, dict) and p.get("name") == sig), None)
                        if p_entry and p_entry.get("direction") == "output" and sig not in reset_actions:
                            reset_actions[sig] = "1'b0"

        if register_updates or reset_actions:
            proc_id = f"SEQ_{module_name.upper()}_PROC"
            seq_proc = PlanSequentialProcess(
                process_id=proc_id,
                process_type="ALWAYS_FF",
                clock_signal=clk_sig,
                clock_edge=clk_edge,
                reset_signal=rst_sig,
                reset_type=rst_type,
                reset_active_level=rst_lvl,
                reset_priority="HIGH",
                reset_actions=reset_actions,
                register_updates=register_updates,
                fsm_state_updates={"state_reg": "current_state"} if fsm_states else {},
                origin=PlanOrigin.DERIVED,
                derivation=DerivationMetadata(
                    description="Cohesive sequential process compiled from clock, reset, and register update rules",
                    input_entities=[clk_sig] + ([rst_sig] if rst_sig else []),
                    rule="SEQUENTIAL_PROCESS_COMPILATION"
                )
            )
            seq_processes.append(seq_proc)

        return seq_processes, unresolved_list

    @classmethod
    def _compile_combinational_logic(
        cls,
        module_name: str,
        plan_dict: Dict[str, Any],
        ports: List[Dict[str, Any]],
        registers: List[Dict[str, Any]],
        fsm_data: Dict[str, Any],
        opcodes: List[Any],
        output_assigns: Dict[str, str],
        efs_ir: Optional[EFSIR],
        compiled_rtl: Dict[str, Any],
        design_spec: str = "",
        protocol_contract: Optional[PlanProtocolContract] = None
    ) -> Tuple[List[PlanCombinationalProcess], List[PlanUnresolvedLogic]]:
        """Compile combinational assignments, continuous assignments, and next-state logic."""
        comb_processes: List[PlanCombinationalProcess] = []
        unresolved_list: List[PlanUnresolvedLogic] = []

        all_assignments: List[PlanCombinationalAssignment] = []
        input_deps: Set[str] = set()
        output_targets: Set[str] = set()

        reg_names = {str(r.get("name", "")).strip().lower(): str(r.get("name", "")).strip() for r in registers if isinstance(r, dict)}

        # 1. Output Ports Combinational Driving Assignments
        for port in ports:
            if not isinstance(port, dict):
                continue
            p_name = str(port.get("name", "")).strip()
            p_dir = str(port.get("direction", "")).lower()
            if p_dir not in ("output", "out") or not p_name:
                continue

            p_lower = p_name.lower()

            # Check if human-approved resolution or functional behavior defines driving logic for this port
            approved_driver_expr = None
            if efs_ir and hasattr(efs_ir, "resolutions"):
                for res in getattr(efs_ir, "resolutions", []):
                    if getattr(res, "approval_status", "") == "APPROVED":
                        tent = str(getattr(res, "target_entity", "")).strip().lower()
                        affs = [str(a).strip().lower() for a in getattr(res, "affected_requirements", [])]
                        if tent == p_lower or p_lower in affs or f"port:{p_lower}" in affs:
                            val = str(getattr(res, "resolved_value", "") or getattr(res, "proposed_change", "")).strip()
                            if val:
                                approved_driver_expr = val
                                break
            if not approved_driver_expr and efs_ir and hasattr(efs_ir, "functional_behavior"):
                for fb in getattr(efs_ir, "functional_behavior", []):
                    subj = str(getattr(fb, "subject", "")).strip().lower()
                    act = str(getattr(fb, "action", "") or getattr(fb, "expression", "")).strip()
                    if subj == p_lower or f"drive_{p_lower}" == getattr(fb, "name", "").lower():
                        if "=" in act:
                            approved_driver_expr = act.split("=")[1].strip().rstrip(";")
                        else:
                            approved_driver_expr = act
                        break

            if approved_driver_expr:
                clean_expr = approved_driver_expr
                if clean_expr.startswith("assign "):
                    m_as = re.search(r"assign\s+[A-Za-z0-9_]+\s*=\s*([^;]+);?", clean_expr)
                    if m_as:
                        clean_expr = m_as.group(1).strip()
                    else:
                        clean_expr = clean_expr.replace("assign ", "").strip()
                clean_expr = clean_expr.rstrip(";")
                p_width = str(port.get("width", "1")).strip()
                if clean_expr in ("'0", "0") and p_width.isdigit() and int(p_width) > 1:
                    clean_expr = f"{p_width}'d0"
                elif clean_expr in ("'1", "1") and p_width.isdigit() and int(p_width) > 1:
                    clean_expr = f"{{{p_width}{{1'b1}}}}"
                parsed_expr = PlanExpression.parse_expression(clean_expr, expr_id=f"expr_{p_name}_approved")
                deps = [op for op in parsed_expr.operands if isinstance(op, str) and not re.match(r"^\d+'[bBhdD]|\b\d+\b", op)]
                input_deps.update(deps)

                all_assignments.append(PlanCombinationalAssignment(
                    target=p_name,
                    expression=parsed_expr,
                    expression_raw=clean_expr,
                    origin=PlanOrigin.INHERITED,
                    derivation=DerivationMetadata(
                        description=f"Human-approved driving assignment for output port '{p_name}'",
                        input_entities=deps,
                        rule="HUMAN_APPROVED_OUTPUT_DRIVER"
                    ),
                    source_references=["HUMAN_APPROVED_OUTPUT_DRIVER"]
                ))
                output_targets.add(p_name)
            # Check if explicit output assignment exists
            elif p_name in output_assigns:
                raw_expr = output_assigns[p_name]
                parsed_expr = PlanExpression.parse_expression(raw_expr, expr_id=f"expr_{p_name}_assign")
                deps = [op for op in parsed_expr.operands if isinstance(op, str) and not re.match(r"^\d+'[bBhdD]|\b\d+\b", op)]
                input_deps.update(deps)

                all_assignments.append(PlanCombinationalAssignment(
                    target=p_name,
                    expression=parsed_expr,
                    expression_raw=raw_expr,
                    origin=PlanOrigin.INHERITED,
                    derivation=DerivationMetadata(
                        description=f"Continuous assignment for output port '{p_name}' from specification",
                        input_entities=deps,
                        rule="SPECIFICATION_OUTPUT_ASSIGNMENT"
                    ),
                    source_references=["SPECIFICATION_OUTPUT_ASSIGNMENT"]
                ))
                output_targets.add(p_name)
            # Check if matching register exists (e.g. 'count' driven by 'count_reg' or 'state_out' by 'state')
            elif f"{p_lower}_reg" in reg_names:
                matching_reg = reg_names[f"{p_lower}_reg"]
                expr_obj = PlanExpression(
                    expression_id=f"expr_{p_name}_reg_driver",
                    expression_type="IDENTIFIER",
                    operands=[matching_reg]
                )
                input_deps.add(matching_reg)
                all_assignments.append(PlanCombinationalAssignment(
                    target=p_name,
                    expression=expr_obj,
                    expression_raw=matching_reg,
                    origin=PlanOrigin.DERIVED,
                    derivation=DerivationMetadata(
                        description=f"Output port '{p_name}' continuously driven by internal register '{matching_reg}'",
                        input_entities=[matching_reg],
                        rule="REGISTERED_OUTPUT_PORT_DRIVER"
                    )
                ))
                output_targets.add(p_name)
            # Check if instruction formats or opcodes define slicing for this port
            elif efs_ir and hasattr(efs_ir, "instruction_formats") and efs_ir.instruction_formats and any(
                fld_name and fld_name.lower() == p_lower
                for fmt in efs_ir.instruction_formats
                for fld in getattr(fmt, "fields", [])
                for fld_name in [getattr(fld, "name", "")]
            ):
                slice_found = False
                for fmt in efs_ir.instruction_formats:
                    for fld in getattr(fmt, "fields", []):
                        fld_name = getattr(fld, "name", "")
                        if fld_name and fld_name.lower() == p_lower:
                            # Bit range slice from instruction input
                            msb = getattr(fld, "msb", None)
                            lsb = getattr(fld, "lsb", None)
                            inst_port = next((p.get("name") for p in ports if isinstance(p, dict) and p.get("direction") == "input" and any(k in p.get("name", "").lower() for k in ("instruction", "instr", "word", "cmd"))), "instruction")
                            if msb is not None and lsb is not None:
                                slice_expr = PlanExpression(
                                    expression_id=f"expr_{p_name}_slice",
                                    expression_type="SLICE",
                                    operator="SLICE",
                                    operands=[inst_port, int(msb), int(lsb)]
                                )
                                raw_slice = f"{inst_port}[{msb}:{lsb}]"
                                input_deps.add(inst_port)
                                all_assignments.append(PlanCombinationalAssignment(
                                    target=p_name,
                                    expression=slice_expr,
                                    expression_raw=raw_slice,
                                    origin=PlanOrigin.DERIVED,
                                    derivation=DerivationMetadata(
                                        description=f"Bitfield slice extraction for port '{p_name}' from '{inst_port}'",
                                        input_entities=[inst_port],
                                        rule="INSTRUCTION_FORMAT_SLICE"
                                    ),
                                    source_references=getattr(fld, "source_requirements", []) or []
                                ))
                                output_targets.add(p_name)
                                slice_found = True
                                break
                    if slice_found:
                        break
            # Check if protocol contract provides synthesizable transaction acceptance or transfer flags
            elif protocol_contract and any(
                rule.handshake_condition and rule.is_synthesizable_behavior and (
                    p_name in rule.applicable_signals or
                    any(k in p_lower for k in ("accept", "transfer_active", "hs_done", "handshake_done", "valid_transfer", "commit", "tx_ack"))
                )
                for rule in protocol_contract.rules
            ):
                matched_rule = next(
                    rule for rule in protocol_contract.rules
                    if rule.handshake_condition and rule.is_synthesizable_behavior and (
                        p_name in rule.applicable_signals or
                        any(k in p_lower for k in ("accept", "transfer_active", "hs_done", "handshake_done", "valid_transfer", "commit", "tx_ack"))
                    )
                )
                parsed_hs = PlanExpression.parse_expression(matched_rule.handshake_condition, expr_id=f"expr_{p_name}_hs")
                all_assignments.append(PlanCombinationalAssignment(
                    target=p_name,
                    expression=parsed_hs,
                    expression_raw=matched_rule.handshake_condition,
                    origin=PlanOrigin.DERIVED,
                    derivation=DerivationMetadata(
                        description=f"Transaction acceptance flag derived from protocol rule '{matched_rule.rule_id}'",
                        input_entities=matched_rule.applicable_signals,
                        rule="PROTOCOL_TRANSACTION_ACCEPTANCE"
                    ),
                    source_references=[matched_rule.rule_id]
                ))
                input_deps.update([op for op in parsed_hs.operands if isinstance(op, str) and not re.match(r"^\d+'", op) and not op.isdigit()])
                output_targets.add(p_name)
            else:
                # Output port has no driving expression
                unresolved_list.append(PlanUnresolvedLogic(
                    affected_component=module_name,
                    affected_entity=p_name,
                    description=f"Output port '{p_name}' is declared in the interface, but no driving continuous expression or register assignment is specified.",
                    missing_information=f"Driving equation or signal driver for output port '{p_name}'",
                    reason="Interface declaration omitted driving logic.",
                    severity="BLOCKING",
                    blocks_generation=True,
                    requires_human_decision=True
                ))

        # 2. Opcode Decoding Combinational Logic
        if opcodes:
            # Check if there is an opcode port/signal and valid_opcode output
            opc_port_name = next((p.get("name") for p in ports if isinstance(p, dict) and any(k == p.get("name", "").lower() for k in ("opcode", "opc", "op"))), None)
            valid_opc_port = next((p.get("name") for p in ports if isinstance(p, dict) and any(k == p.get("name", "").lower() for k in ("valid_opcode", "opcode_valid", "valid_op", "valid"))), None)

            if opc_port_name and valid_opc_port:
                input_deps.add(opc_port_name)
                output_targets.add(valid_opc_port)
                comparisons: List[PlanExpression] = []
                comp_strs: List[str] = []

                for op in opcodes:
                    enc = str(op.get("binary_encoding", "") if isinstance(op, dict) else getattr(op, "binary_encoding", "")).strip()
                    if enc and enc not in ("None", "undefined"):
                        comp_expr = PlanExpression(
                            expression_id=f"expr_cmp_{enc}",
                            expression_type="BOOLEAN",
                            operator="==",
                            operands=[opc_port_name, enc]
                        )
                        comparisons.append(comp_expr)
                        comp_strs.append(f"({opc_port_name} == {enc})")

                if comparisons:
                    comb_valid_expr = PlanExpression(
                        expression_id="expr_valid_opcode_or",
                        expression_type="BOOLEAN",
                        operator="||" if len(comparisons) <= 3 else "|",
                        operands=comparisons
                    )
                    all_assignments.append(PlanCombinationalAssignment(
                        target=valid_opc_port,
                        expression=comb_valid_expr,
                        expression_raw=" || ".join(comp_strs),
                        origin=PlanOrigin.DERIVED,
                        derivation=DerivationMetadata(
                            description="Combinational opcode validity decode from registered opcode table",
                            input_entities=[opc_port_name],
                            rule="OPCODE_SET_VALIDITY_DECODE"
                        ),
                        source_references=["OPCODE_TABLE"]
                    ))

        # 3. FSM Next-State Combinational Logic
        fsm_states = fsm_data.get("states", [])
        fsm_transitions = fsm_data.get("transitions", [])
        if fsm_states and fsm_transitions:
            state_reg_name = "current_state"
            if any(r.get("name") == "state" for r in registers if isinstance(r, dict)):
                state_reg_name = "state"
            elif any(r.get("name") == "state_reg" for r in registers if isinstance(r, dict)):
                state_reg_name = "state_reg"

            input_deps.add(state_reg_name)
            output_targets.add("next_state")

            # Map transitions per source state
            transitions_by_state: Dict[str, List[Dict[str, Any]]] = {}
            for tr in fsm_transitions:
                from_st = tr.get("from_state") or tr.get("from") or ""
                if from_st:
                    transitions_by_state.setdefault(str(from_st), []).append(tr)

            for st, tr_list in transitions_by_state.items():
                for tr in tr_list:
                    to_st = tr.get("to_state") or tr.get("to") or ""
                    cond = tr.get("condition", {})
                    cond_expr = cond.get("expression") if isinstance(cond, dict) else str(cond)

                    if not cond_expr or cond_expr.lower() in ("none", "undefined", "tbd"):
                        # Ambiguous or missing FSM transition condition!
                        unresolved_list.append(PlanUnresolvedLogic(
                            affected_component=module_name,
                            affected_entity=f"transition:{st}->{to_st}",
                            description=f"FSM transition from '{st}' to '{to_st}' has an incomplete or unspecified transition condition.",
                            missing_information="Boolean transition condition expression or signal guard",
                            reason="FSM specification contains state transition without guard expression.",
                            severity="BLOCKING",
                            blocks_generation=True,
                            requires_human_decision=True
                        ))
                    else:
                        parsed_cond = PlanExpression.parse_expression(cond_expr, expr_id=f"expr_{st}_{to_st}_cond")
                        all_assignments.append(PlanCombinationalAssignment(
                            target="next_state",
                            expression=PlanExpression(
                                expression_id=f"expr_next_state_{st}_{to_st}",
                                expression_type="IDENTIFIER",
                                operands=[to_st]
                            ),
                            expression_raw=str(to_st),
                            condition=f"({state_reg_name} == {st}) && ({cond_expr})",
                            default_assignment=state_reg_name,
                            origin=PlanOrigin.INHERITED,
                            derivation=DerivationMetadata(
                                description=f"Next-state transition logic from '{st}' to '{to_st}'",
                                input_entities=[state_reg_name],
                                rule="FSM_TRANSITION_SPECIFICATION"
                            ),
                            source_references=["FSM_TRANSITION_TABLE"]
                        ))
                        input_deps.add(state_reg_name)
                        output_targets.add("next_state")

        # Check if protocol contract provides synthesizable transaction acceptance or transfer flags
        if protocol_contract:
            for rule in protocol_contract.rules:
                if rule.handshake_condition and rule.is_synthesizable_behavior:
                    for p in ports:
                        if isinstance(p, dict) and p.get("direction") == "output":
                            p_name = str(p.get("name", ""))
                            p_lower = p_name.lower()
                            if p_name not in output_targets and any(k in p_lower for k in ("accept", "transfer_active", "hs_done", "handshake_done", "valid_transfer", "commit", "tx_ack")):
                                parsed_hs = PlanExpression.parse_expression(rule.handshake_condition, expr_id=f"expr_{p_name}_hs")
                                all_assignments.append(PlanCombinationalAssignment(
                                    target=p_name,
                                    expression=parsed_hs,
                                    expression_raw=rule.handshake_condition,
                                    origin=PlanOrigin.DERIVED,
                                    derivation=DerivationMetadata(
                                        description=f"Transaction acceptance flag derived from protocol rule '{rule.rule_id}'",
                                        input_entities=rule.applicable_signals,
                                        rule="PROTOCOL_TRANSACTION_ACCEPTANCE"
                                    ),
                                    source_references=[rule.rule_id]
                                ))
                                input_deps.update([op for op in parsed_hs.operands if isinstance(op, str) and not re.match(r"^\d+'", op) and not op.isdigit()])
                                output_targets.add(p_name)

        # Package into PlanCombinationalProcess
        if all_assignments:
            comb_proc = PlanCombinationalProcess(
                process_id=f"COMB_{module_name.upper()}_DIRECTIVES",
                process_type="ALWAYS_COMB",
                input_dependencies=list(input_deps),
                output_targets=list(output_targets),
                assignments=all_assignments,
                origin=PlanOrigin.DERIVED,
                derivation=DerivationMetadata(
                    description="Cohesive combinational logic directives compiled from outputs, opcodes, and FSM next-state rules",
                    input_entities=list(input_deps),
                    rule="COMBINATIONAL_DIRECTIVES_COMPILATION"
                )
            )
            comb_processes.append(comb_proc)

        return comb_processes, unresolved_list

    @classmethod
    def _compile_protocol_contract(
        cls,
        module_name: str,
        ports: List[Dict[str, Any]],
        target_comp: Optional[EFSComponent],
        compiled_rtl: Dict[str, Any],
        context_pack: Optional[Any],
        efs_ir: Optional[EFSIR],
    ) -> Tuple[Optional[PlanProtocolContract], List[PlanUnresolvedLogic]]:
        """
        Compile target component-relevant protocol rules into a structured PlanProtocolContract.
        Distinguishes synthesizable behavior (handshakes, acceptance, reset) from
        verification-only assertions and constraints.
        """
        unresolved_list: List[PlanUnresolvedLogic] = []

        # 1. Identify Target Component Interfaces and Ports
        comp_interfaces: Set[str] = set()
        if target_comp and getattr(target_comp, "interfaces", None):
            for if_ref in target_comp.interfaces:
                comp_interfaces.add(str(if_ref).strip().lower())
        for p in ports:
            if isinstance(p, dict) and p.get("interface"):
                comp_interfaces.add(str(p["interface"]).strip().lower())
        if target_comp and getattr(target_comp, "protocol", None):
            comp_interfaces.add(str(target_comp.protocol).strip().lower())

        # Expand interface aliases and protocols from efs_ir.interfaces
        if efs_ir and getattr(efs_ir, "interfaces", None) and comp_interfaces:
            for iface in efs_ir.interfaces:
                if_id = str(getattr(iface, "interface_id", None) or (iface.get("interface_id") if isinstance(iface, dict) else "") or "").strip().lower()
                if_name = str(getattr(iface, "name", None) or (iface.get("name") if isinstance(iface, dict) else "") or "").strip().lower()
                if_proto = str(getattr(iface, "protocol", None) or (iface.get("protocol") if isinstance(iface, dict) else "") or "").strip().lower()
                if any(ci in (if_id, if_name, if_proto) for ci in list(comp_interfaces) if ci):
                    if if_id:
                        comp_interfaces.add(if_id)
                    if if_name:
                        comp_interfaces.add(if_name)
                    if if_proto:
                        comp_interfaces.add(if_proto)

        port_map: Dict[str, Dict[str, Any]] = {}
        for p in ports:
            if isinstance(p, dict) and p.get("name"):
                port_map[str(p["name"]).strip().lower()] = p

        comp_ids: Set[str] = {module_name.lower()}
        if target_comp:
            if getattr(target_comp, "component_id", None):
                comp_ids.add(str(target_comp.component_id).strip().lower())
            if getattr(target_comp, "name", None):
                comp_ids.add(str(target_comp.name).strip().lower())

        # 2. Gather All Candidate Protocol Rules Across All 3 Input Sources
        raw_candidates: List[Any] = []

        # Source 1: compiled_rtl["protocol_rules"]
        if compiled_rtl and compiled_rtl.get("protocol_rules"):
            raw_candidates.extend(compiled_rtl["protocol_rules"])

        # Source 2: context_pack.protocol_rules
        if context_pack:
            cp_rules = (
                getattr(context_pack, "protocol_rules", None) or
                (context_pack.get("protocol_rules", []) if isinstance(context_pack, dict) else []) or
                (context_pack.get("relevant_protocol_rules", []) if isinstance(context_pack, dict) else [])
            )
            raw_candidates.extend(cp_rules)

        # Source 3: efs_ir.protocol_rules
        if efs_ir and getattr(efs_ir, "protocol_rules", None):
            raw_candidates.extend(efs_ir.protocol_rules)

        if not raw_candidates:
            return None, unresolved_list

        # 3. Centralized Protocol Rule Normalization
        normalized_rules: List[NormalizedProtocolRule] = []
        for r in raw_candidates:
            norm_r = normalize_protocol_rule(r)
            if norm_r:
                normalized_rules.append(norm_r)

        if not normalized_rules:
            return None, unresolved_list

        # 4. Deterministic Deduplication (merging richer details)
        dedup_map: Dict[str, NormalizedProtocolRule] = {}
        for nr in normalized_rules:
            sig_key = tuple(sorted(s.lower() for s in nr.signals))
            key = nr.rule_id if nr.rule_id else f"{nr.interface}_{nr.description[:40]}_{sig_key}"
            if key not in dedup_map:
                dedup_map[key] = nr
            else:
                existing = dedup_map[key]
                if not existing.handshake_condition and nr.handshake_condition:
                    existing.handshake_condition = nr.handshake_condition
                if not existing.source_evidence and nr.source_evidence:
                    existing.source_evidence = nr.source_evidence
                if not existing.interface and nr.interface:
                    existing.interface = nr.interface
                if not existing.protocol and nr.protocol:
                    existing.protocol = nr.protocol
                if not existing.timing_constraint and nr.timing_constraint:
                    existing.timing_constraint = nr.timing_constraint
                if not existing.reset_requirement and nr.reset_requirement:
                    existing.reset_requirement = nr.reset_requirement
                if not existing.stability_constraint and nr.stability_constraint:
                    existing.stability_constraint = nr.stability_constraint

        dedup_candidates = list(dedup_map.values())

        # 5. Component Relevance Filtering
        # A component only inherits protocol rules if:
        # a) Rule does not explicitly target an unrelated component, AND
        # b) Rule interface or protocol matches component interfaces, OR
        # c) Rule signals intersect with component ports, OR
        # d) Port name or component interface name appears in rule description.
        # Rules without explicit interface association are evaluated on signal and port dependencies.
        # Irrelevant rules targeting unrelated components/interfaces are strictly EXCLUDED.
        relevant_rules: List[NormalizedProtocolRule] = []
        for nr in dedup_candidates:
            # Component isolation
            if nr.component and nr.component.lower() not in comp_ids:
                continue

            r_iface_lower = nr.interface.lower()
            r_proto_lower = nr.protocol.lower()
            r_desc_lower = nr.description.lower()

            matched_iface = bool(comp_interfaces and (
                (r_iface_lower and r_iface_lower in comp_interfaces) or
                (r_proto_lower and r_proto_lower in comp_interfaces) or
                any(ci in r_desc_lower for ci in comp_interfaces if len(ci) >= 3)
            ))

            matched_signal = any(s.lower() in port_map for s in nr.signals)

            matched_port_name = any(
                p_name in r_desc_lower for p_name in port_map
                if len(p_name) >= 3 and p_name not in ("clk", "clock", "rst", "reset", "rst_n")
            )

            if comp_interfaces:
                is_rel = matched_iface or matched_signal or matched_port_name
            else:
                is_rel = matched_signal or matched_port_name

            if is_rel:
                relevant_rules.append(nr)

        if not relevant_rules:
            return None, unresolved_list

        # 6. Resolve Authoritative Protocol Name
        proto_name = "GENERIC"
        if target_comp and getattr(target_comp, "protocol", None):
            proto_name = target_comp.protocol
        else:
            found_proto = None
            if efs_ir and getattr(efs_ir, "interfaces", None) and comp_interfaces:
                for iface in efs_ir.interfaces:
                    if_id = str(getattr(iface, "interface_id", None) or (iface.get("interface_id") if isinstance(iface, dict) else "") or "").lower()
                    if_name = str(getattr(iface, "name", None) or (iface.get("name") if isinstance(iface, dict) else "") or "").lower()
                    if_p = getattr(iface, "protocol", None) or (iface.get("protocol") if isinstance(iface, dict) else "")
                    if if_p and (if_id in comp_interfaces or if_name in comp_interfaces):
                        found_proto = str(if_p)
                        break
            if found_proto:
                proto_name = found_proto
            else:
                for nr in relevant_rules:
                    if nr.protocol:
                        proto_name = nr.protocol
                        break

        # 7. Construct PlanProtocolRule and PlanProtocolContract
        structured_rules: List[PlanProtocolRule] = []
        verification_assertions: List[Dict[str, Any]] = []
        source_evidence_ids: List[str] = []

        for idx, nr in enumerate(relevant_rules):
            r_id = nr.rule_id or f"prule_{idx+1}"
            r_name = nr.name or f"RULE_{idx+1}"
            r_desc = nr.description
            r_desc_lower = r_desc.lower()
            raw_type = nr.rule_type.lower()

            # Matched signals for this rule
            app_signals = [s for s in nr.signals if s.lower() in port_map]
            if not app_signals:
                app_signals = [p["name"] for p_lower, p in port_map.items() if p_lower in r_desc_lower]

            # Matched interfaces
            app_interfaces = [i for i in comp_interfaces if i in r_desc_lower or i == nr.interface.lower()]
            if not app_interfaces and nr.interface:
                app_interfaces = [nr.interface]

            # Determine Rule Type & Synthesizability
            if raw_type == "stability" or ("stability" in r_desc_lower and raw_type not in ("handshake", "reset")):
                r_type = "STABILITY"
                is_synth = not nr.verification_only
            elif raw_type == "reset" or ("reset" in r_desc_lower and raw_type not in ("handshake", "stability")):
                r_type = "RESET"
                is_synth = True
            elif raw_type in ("acceptance", "transaction_acceptance"):
                r_type = "TRANSACTION_ACCEPTANCE"
                is_synth = not nr.verification_only
            elif raw_type == "handshake" or any(kw in r_desc_lower for kw in ("handshake", "valid/ready", "req/ack", "strobe transfer")):
                r_type = "HANDSHAKE"
                is_synth = not nr.verification_only
            elif nr.verification_only:
                r_type = "VERIFICATION_ONLY"
                is_synth = False
            elif raw_type == "timing" or bool(re.search(r"\b(?:setup\s+time|hold\s+time|propagation\s+delay|\d+\s*(?:ns|ps|us))\b", r_desc_lower)):
                r_type = "TIMING"
                is_synth = False
            elif raw_type in ("ordering", "transaction_ordering") or "order" in r_desc_lower:
                r_type = "ORDERING"
                is_synth = False
            else:
                r_type = (raw_type or "PROTOCOL_CONTRACT").upper()
                is_synth = not nr.verification_only

            # Extract Handshake and Acceptance Conditions
            handshake_cond = nr.handshake_condition or nr.condition
            if not handshake_cond and r_type in ("HANDSHAKE", "TRANSACTION_ACCEPTANCE", "STABILITY"):
                v_sig = next((s for s in app_signals if "valid" in s.lower()), None) or next((p["name"] for p_lower, p in port_map.items() if "valid" in p_lower), None)
                r_sig = next((s for s in app_signals if "ready" in s.lower()), None) or next((p["name"] for p_lower, p in port_map.items() if "ready" in p_lower), None)
                req_sig = next((s for s in app_signals if "req" in s.lower()), None) or next((p["name"] for p_lower, p in port_map.items() if "req" in p_lower), None)
                ack_sig = next((s for s in app_signals if "ack" in s.lower()), None) or next((p["name"] for p_lower, p in port_map.items() if "ack" in p_lower), None)

                if v_sig and r_sig:
                    handshake_cond = f"{v_sig} && {r_sig}"
                elif req_sig and ack_sig:
                    handshake_cond = f"{req_sig} && {ack_sig}"

            # Handshake Gap Check (Unresolved Logic)
            if r_type == "HANDSHAKE":
                rule_sigs = [s.lower() for s in nr.signals]
                cond_lower = (handshake_cond or "").lower()
                expects_ready = any("ready" in s for s in rule_sigs) or ("ready" in r_desc_lower) or ("ready" in cond_lower)
                expects_valid = any("valid" in s for s in rule_sigs) or ("valid" in r_desc_lower) or ("valid" in cond_lower)
                expects_req = any("req" in s for s in rule_sigs) or ("req" in r_desc_lower) or ("req" in cond_lower)
                expects_ack = any("ack" in s for s in rule_sigs) or ("ack" in r_desc_lower) or ("ack" in cond_lower)

                has_ready = any("ready" in p_lower for p_lower in port_map)
                has_valid = any("valid" in p_lower for p_lower in port_map)
                has_req = any("req" in p_lower for p_lower in port_map)
                has_ack = any("ack" in p_lower for p_lower in port_map)

                if expects_valid and expects_ready:
                    if has_valid and not has_ready:
                        missing_sig = next((s for s in nr.signals if "ready" in s.lower()), "s_ready")
                        unresolved_list.append(PlanUnresolvedLogic(
                            affected_component=module_name,
                            affected_entity=f"protocol_rule:{r_id}",
                            description=f"Protocol rule '{r_name}' requires valid/ready handshake, but paired port '{missing_sig}' is not declared in module interface.",
                            missing_information=f"Missing complementary handshake port '{missing_sig}'",
                            reason="Protocol handshake requires both request/valid and acknowledge/ready signals.",
                            severity="WARNING",
                            blocks_generation=False
                        ))
                    elif has_ready and not has_valid:
                        missing_sig = next((s for s in nr.signals if "valid" in s.lower()), "s_valid")
                        unresolved_list.append(PlanUnresolvedLogic(
                            affected_component=module_name,
                            affected_entity=f"protocol_rule:{r_id}",
                            description=f"Protocol rule '{r_name}' requires valid/ready handshake, but paired port '{missing_sig}' is not declared in module interface.",
                            missing_information=f"Missing complementary handshake port '{missing_sig}'",
                            reason="Protocol handshake requires both request/valid and acknowledge/ready signals.",
                            severity="WARNING",
                            blocks_generation=False
                        ))

                if expects_req and expects_ack:
                    if has_req and not has_ack:
                        missing_sig = next((s for s in nr.signals if "ack" in s.lower()), "ack")
                        unresolved_list.append(PlanUnresolvedLogic(
                            affected_component=module_name,
                            affected_entity=f"protocol_rule:{r_id}",
                            description=f"Protocol rule '{r_name}' requires req/ack handshake, but paired port '{missing_sig}' is not declared in module interface.",
                            missing_information=f"Missing complementary handshake port '{missing_sig}'",
                            reason="Protocol handshake requires both req and ack signals.",
                            severity="WARNING",
                            blocks_generation=False
                        ))
                    elif has_ack and not has_req:
                        missing_sig = next((s for s in nr.signals if "req" in s.lower()), "req")
                        unresolved_list.append(PlanUnresolvedLogic(
                            affected_component=module_name,
                            affected_entity=f"protocol_rule:{r_id}",
                            description=f"Protocol rule '{r_name}' requires req/ack handshake, but paired port '{missing_sig}' is not declared in module interface.",
                            missing_information=f"Missing complementary handshake port '{missing_sig}'",
                            reason="Protocol handshake requires both req and ack signals.",
                            severity="WARNING",
                            blocks_generation=False
                        ))

            stability_str = nr.stability_constraint or (r_desc if r_type == "STABILITY" else None)
            reset_req_str = nr.reset_requirement or (r_desc if r_type == "RESET" else None)
            timing_str = nr.timing_constraint or (r_desc if r_type == "TIMING" else None)

            if nr.source_evidence and nr.source_evidence not in source_evidence_ids:
                source_evidence_ids.append(nr.source_evidence)

            plan_rule = PlanProtocolRule(
                rule_id=r_id,
                name=r_name,
                rule_type=r_type,
                description=r_desc,
                applicable_interfaces=app_interfaces,
                applicable_signals=app_signals,
                handshake_condition=handshake_cond,
                acceptance_condition=handshake_cond if r_type == "TRANSACTION_ACCEPTANCE" else None,
                stability_constraint=stability_str,
                reset_requirement=reset_req_str,
                timing_constraint=timing_str,
                is_synthesizable_behavior=is_synth,
                source_references=[r_id] + ([nr.chunk_id] if nr.chunk_id else []),
                source_evidence=nr.source_evidence,
                origin=PlanOrigin.INHERITED,
                derivation=DerivationMetadata(
                    description=f"Protocol rule '{r_id}' inherited from protocol specification",
                    input_entities=app_signals,
                    rule="PROTOCOL_RULE_PROJECTION"
                )
            )
            structured_rules.append(plan_rule)

            if not is_synth:
                verification_assertions.append({
                    "rule_id": r_id,
                    "name": r_name,
                    "assertion_type": r_type,
                    "property_statement": r_desc,
                    "signals": app_signals,
                    "source_evidence": nr.source_evidence
                })

        contract = PlanProtocolContract(
            protocol_name=proto_name,
            protocol_version="1.0",
            applicable_interfaces=list(comp_interfaces),
            rules=structured_rules,
            verification_assertions=verification_assertions,
            source_evidence_ids=source_evidence_ids,
            origin=PlanOrigin.INHERITED,
            derivation=DerivationMetadata(
                description=f"Scoped protocol contract for component '{module_name}'",
                input_entities=list(comp_interfaces),
                rule="COMPONENT_PROTOCOL_PROJECTION"
            )
        )
        return contract, unresolved_list
