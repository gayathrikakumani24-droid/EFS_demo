"""
EFS IR Validator.

Performs deterministic checking of the EFS IR model to find structural issues,
dangling references, invalid definitions, and missing traceability.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from core.efs_ir.models import EFSIR, EFSValidationResult, EFSValidationSummary
from core.efs_ir.validation_rules import is_explicit_address, normalize_hardware_offset


def make_validation_issue(
    category: str,
    message: str,
    object_id: str = "",
    severity: str = "WARNING",
    classification: Optional[str] = None,
    evidence: str = "",
    required_information: str = ""
) -> Dict[str, Any]:
    """Helper to create a fully schema-compliant, provenance-backed validation issue dictionary."""
    if not classification:
        if category in ("Grounding", "Traceability"):
            classification = "GROUNDING_GAP"
        elif severity in ("ERROR", "CRITICAL"):
            classification = "SOURCE_MISSING"
        else:
            classification = "MISSING"
            
    issue_text = message or f"Validation finding in {category} for object '{object_id}'"
    evidence_text = evidence or (f"Object ID: {object_id}" if object_id else "N/A")
    req_info = required_information or (f"Inspect {category} specification for object '{object_id}'." if object_id else "Review design specification.")

    return {
        "classification": classification,
        "category": category,
        "severity": severity,
        "issue": issue_text,
        "message": issue_text,
        "object_id": object_id,
        "evidence": evidence_text,
        "user_spec_evidence": evidence_text,
        "required_information": req_info,
        "is_blocking": severity in ("ERROR", "CRITICAL")
    }


def validate_efs_ir(ir: EFSIR) -> List[Dict[str, Any]]:
    """
    Validate the EFS IR structure deterministically.
    
    Returns a list of structured, schema-compliant validation issues.
    """
    issues = []
    
    # Track defined IDs for duplicates checking
    defined_ids = set()
    
    # Helper to check for duplicate IDs
    def check_id(obj_id: str, category: str):
        if not obj_id:
            return
        if obj_id in defined_ids:
            issues.append(make_validation_issue(
                category=category,
                message=f"Duplicate object ID detected: '{obj_id}'",
                object_id=obj_id,
                severity="ERROR",
                classification="EFSIR_EXTRACTION_GAP"
            ))
        defined_ids.add(obj_id)

    # 1. Gather all object ID lookups
    component_ids = {c.component_id for c in ir.components}
    interface_ids = {i.interface_id for i in ir.interfaces}
    signal_ids = {s.signal_id for s in ir.signals}
    signal_names = {s.name for s in ir.signals}
    register_ids = {r.register_id for r in ir.registers}
    fsm_ids = {f.fsm_id for f in ir.fsms}
    transaction_ids = {t.transaction_id for t in ir.transactions}

    # 2. Check metadata
    if not ir.metadata.design_id:
        issues.append(make_validation_issue(
            category="Metadata",
            message="Design metadata is missing a unique design_id",
            severity="ERROR",
            classification="EFSIR_EXTRACTION_GAP"
        ))
    if not getattr(ir.metadata, "canonical_id", None):
        issues.append(make_validation_issue(
            category="Metadata",
            message="Design metadata is missing a canonical_id",
            severity="ERROR",
            classification="EFSIR_EXTRACTION_GAP"
        ))
    if not getattr(ir.metadata, "display_name", None):
        issues.append(make_validation_issue(
            category="Metadata",
            message="Design metadata is missing a display_name",
            severity="WARNING",
            classification="EFSIR_EXTRACTION_GAP"
        ))
    if not getattr(ir.metadata, "generation_run_id", None):
        issues.append(make_validation_issue(
            category="Metadata",
            message="Design metadata is missing a generation_run_id",
            severity="WARNING",
            classification="EFSIR_EXTRACTION_GAP"
        ))

    # 3. Validate components
    for comp in ir.components:
        check_id(comp.component_id, "Components")
        if not comp.name:
            issues.append(make_validation_issue(
                category="Components",
                message=f"Component '{comp.component_id}' has an empty name",
                object_id=comp.component_id,
                severity="WARNING",
                classification="EFSIR_EXTRACTION_GAP"
            ))
        
        # Verify interfaces list
        for if_id in comp.interfaces:
            if if_id not in interface_ids:
                issues.append(make_validation_issue(
                    category="Components",
                    message=f"Component '{comp.name}' references non-existent interface '{if_id}'",
                    object_id=comp.component_id,
                    severity="ERROR",
                    classification="EFSIR_EXTRACTION_GAP"
                ))
        

    # 4. Validate interfaces
    for iface in ir.interfaces:
        check_id(iface.interface_id, "Interfaces")
        if not iface.name:
            issues.append(make_validation_issue(
                category="Interfaces",
                message=f"Interface '{iface.interface_id}' has an empty name",
                object_id=iface.interface_id,
                severity="WARNING",
                classification="EFSIR_EXTRACTION_GAP"
            ))
        
        # Generic role validation
        if iface.role and iface.role.lower() not in ("master", "slave", "initiator", "target", "controller", "endpoint", "requester", "completer", "transmitter", "receiver"):
            issues.append(make_validation_issue(
                category="Interfaces",
                message=f"Interface '{iface.name}' uses non-standard generic role '{iface.role}'",
                object_id=iface.interface_id,
                severity="WARNING",
                classification="SOURCE_AMBIGUOUS"
            ))
            
        # Verify component connections
        if iface.source_component and iface.source_component not in component_ids:
            issues.append(make_validation_issue(
                category="Interfaces",
                message=f"Interface '{iface.name}' references non-existent source component '{iface.source_component}'",
                object_id=iface.interface_id,
                severity="ERROR",
                classification="EFSIR_EXTRACTION_GAP"
            ))
        if iface.destination_component and iface.destination_component not in component_ids:
            issues.append(make_validation_issue(
                category="Interfaces",
                message=f"Interface '{iface.name}' references non-existent destination component '{iface.destination_component}'",
                object_id=iface.interface_id,
                severity="ERROR",
                classification="EFSIR_EXTRACTION_GAP"
            ))

        # Verify signals in interface
        for sig_id in iface.signals:
            if sig_id not in signal_ids:
                issues.append(make_validation_issue(
                    category="Interfaces",
                    message=f"Interface '{iface.name}' references non-existent signal '{sig_id}'",
                    object_id=iface.interface_id,
                    severity="ERROR",
                    classification="EFSIR_EXTRACTION_GAP"
                ))

    # 5. Validate signals
    for sig in ir.signals:
        check_id(sig.signal_id, "Signals")
        if not sig.name:
            issues.append(make_validation_issue(
                category="Signals",
                message=f"Signal '{sig.signal_id}' is missing a name",
                object_id=sig.signal_id,
                severity="ERROR",
                classification="EFSIR_EXTRACTION_GAP"
            ))
        
        # Signal direction validation
        if sig.direction not in ("input", "output", "inout"):
            issues.append(make_validation_issue(
                category="Signals",
                message=f"Signal '{sig.name}' has invalid direction '{sig.direction}'. Must be input, output, or inout.",
                object_id=sig.signal_id,
                severity="ERROR",
                classification="SOURCE_AMBIGUOUS"
            ))
            
        # Signal width validation
        if not sig.width:
            issues.append(make_validation_issue(
                category="Signals",
                message=f"Signal '{sig.name}' has no defined width. Defaulting to 1.",
                object_id=sig.signal_id,
                severity="WARNING",
                classification="SOURCE_AMBIGUOUS"
            ))
        elif not re.match(r"^(\d+|\[\d+:\d+\]|\w+-\d+|[a-zA-Z_0-9]+)$", str(sig.width).strip()):
            issues.append(make_validation_issue(
                category="Signals",
                message=f"Signal '{sig.name}' width expression '{sig.width}' has unusual formatting",
                object_id=sig.signal_id,
                severity="WARNING",
                classification="SOURCE_AMBIGUOUS"
            ))

        # Verify interface pointer
        if sig.interface and sig.interface not in interface_ids:
            issues.append(make_validation_issue(
                category="Signals",
                message=f"Signal '{sig.name}' references non-existent interface '{sig.interface}'",
                object_id=sig.signal_id,
                severity="ERROR",
                classification="EFSIR_EXTRACTION_GAP"
            ))
        
        # Verify owner/consumer pointers
        if sig.owner and sig.owner not in component_ids:
            issues.append(make_validation_issue(
                category="Signals",
                message=f"Signal '{sig.name}' owner component '{sig.owner}' does not exist",
                object_id=sig.signal_id,
                severity="ERROR",
                classification="EFSIR_EXTRACTION_GAP"
            ))
        if sig.consumer and sig.consumer not in component_ids:
            issues.append(make_validation_issue(
                category="Signals",
                message=f"Signal '{sig.name}' consumer component '{sig.consumer}' does not exist",
                object_id=sig.signal_id,
                severity="ERROR",
                classification="EFSIR_EXTRACTION_GAP"
            ))

    # 5b. Validate Clocks and Resets references
    for clk in ir.clocks:
        if not clk.signal_id:
            issues.append(make_validation_issue(
                category="Clocks",
                message="Clock entry is missing a signal_id",
                severity="ERROR",
                classification="EFSIR_EXTRACTION_GAP"
            ))
        elif clk.signal_id not in signal_ids:
            issues.append(make_validation_issue(
                category="Clocks",
                message=f"Clock references non-existent signal ID '{clk.signal_id}'",
                object_id=clk.signal_id,
                severity="ERROR",
                classification="EFSIR_EXTRACTION_GAP",
                evidence=f"Clock signal '{clk.name or clk.signal_id}' ({clk.signal_id}) is not present in design signals collection",
                required_information="Ensure clock signal is declared in signals list."
            ))
        else:
            target_sig = next((s for s in ir.signals if s.signal_id == clk.signal_id), None)
            if target_sig and target_sig.semantic_role and target_sig.semantic_role.lower() not in ("clock", "clk", "gated_clock", "system_clock"):
                issues.append(make_validation_issue(
                    category="Clocks",
                    message=f"Clock signal '{target_sig.name}' has non-clock semantic role '{target_sig.semantic_role}'",
                    object_id=clk.signal_id,
                    severity="WARNING",
                    classification="SOURCE_AMBIGUOUS"
                ))

    for rst in ir.resets:
        if not rst.signal_id:
            issues.append(make_validation_issue(
                category="Resets",
                message="Reset entry is missing a signal_id",
                severity="ERROR",
                classification="EFSIR_EXTRACTION_GAP"
            ))
        elif rst.signal_id not in signal_ids:
            issues.append(make_validation_issue(
                category="Resets",
                message=f"Reset references non-existent signal ID '{rst.signal_id}'",
                object_id=rst.signal_id,
                severity="ERROR",
                classification="EFSIR_EXTRACTION_GAP",
                evidence=f"Reset signal '{rst.name or rst.signal_id}' ({rst.signal_id}) is not present in design signals collection",
                required_information="Ensure reset signal is declared in signals list."
            ))
        else:
            target_sig = next((s for s in ir.signals if s.signal_id == rst.signal_id), None)
            if target_sig and target_sig.semantic_role and target_sig.semantic_role.lower() not in ("reset", "rst", "preset", "clear", "async_reset", "sync_reset"):
                issues.append(make_validation_issue(
                    category="Resets",
                    message=f"Reset signal '{target_sig.name}' has non-reset semantic role '{target_sig.semantic_role}'",
                    object_id=rst.signal_id,
                    severity="WARNING",
                    classification="SOURCE_AMBIGUOUS"
                ))

    # 6. Validate registers
    reg_offsets: Dict[Any, Any] = {}
    for reg in ir.registers:
        check_id(reg.register_id, "Registers")
        if not reg.name:
            issues.append(make_validation_issue(
                category="Registers",
                message=f"Register '{reg.register_id}' has an empty name",
                object_id=reg.register_id,
                severity="ERROR",
                classification="EFSIR_EXTRACTION_GAP"
            ))
        if not is_explicit_address(reg.offset):
            issues.append(make_validation_issue(
                category="Registers",
                message=f"Register '{reg.name}' has no defined address offset",
                object_id=reg.register_id,
                severity="WARNING",
                classification="SOURCE_AMBIGUOUS"
            ))
        else:
            norm_offset = normalize_hardware_offset(reg.offset)
            try:
                norm_val = int(norm_offset, 16) if norm_offset.startswith("0x") else int(norm_offset)
            except Exception:
                norm_val = norm_offset

            if norm_val in reg_offsets:
                prev_reg = reg_offsets[norm_val]
                is_intentional = any(
                    term in (reg.description or "").lower() or term in (prev_reg.description or "").lower()
                    for term in ("alias", "shadow", "mirror", "banked", "alternate view", "register view")
                )
                if not is_intentional:
                    issues.append(make_validation_issue(
                        category="Registers",
                        message=f"Duplicate register address offset detected: '{reg.offset}' shared by '{prev_reg.name}' and '{reg.name}'",
                        object_id=reg.register_id,
                        severity="ERROR",
                        classification="SOURCE_CONFLICT",
                        evidence=f"Registers '{prev_reg.name}' ({prev_reg.register_id}) and '{reg.name}' ({reg.register_id}) both assigned offset '{reg.offset}'",
                        required_information="Resolve address collision by assigning unique offsets."
                    ))
            else:
                reg_offsets[norm_val] = reg
            
        # Validate fields
        field_bits = set()
        for field in reg.fields:
            if not field.name:
                issues.append(make_validation_issue(
                    category="Registers",
                    message=f"Register '{reg.name}' field has no name",
                    object_id=reg.register_id,
                    severity="WARNING",
                    classification="EFSIR_EXTRACTION_GAP"
                ))
            
            # Check bit bounds overlap
            for bit in range(field.lsb, field.msb + 1):
                if bit in field_bits:
                    issues.append(make_validation_issue(
                        category="Registers",
                        message=f"Register '{reg.name}' has overlapping fields on bit {bit} (field '{field.name}')",
                        object_id=reg.register_id,
                        severity="ERROR",
                        classification="SOURCE_CONFLICT"
                    ))
                field_bits.add(bit)
                
            if field.msb >= reg.width:
                issues.append(make_validation_issue(
                    category="Registers",
                    message=f"Register '{reg.name}' field '{field.name}' MSB {field.msb} exceeds register width {reg.width}",
                    object_id=reg.register_id,
                    severity="ERROR",
                    classification="SOURCE_CONFLICT"
                ))

    # 7. Validate FSMs
    for fsm in ir.fsms:
        check_id(fsm.fsm_id, "State Machines")
        if not fsm.states:
            issues.append(make_validation_issue(
                category="State Machines",
                message=f"FSM '{fsm.name or fsm.fsm_id}' has no defined states",
                object_id=fsm.fsm_id,
                severity="ERROR",
                classification="SOURCE_MISSING"
            ))
            
        state_names = {s.name for s in fsm.states}
        if fsm.initial_state and fsm.initial_state not in state_names:
            issues.append(make_validation_issue(
                category="State Machines",
                message=f"FSM '{fsm.name}' initial state '{fsm.initial_state}' is not in the FSM's state list",
                object_id=fsm.fsm_id,
                severity="ERROR",
                classification="EFSIR_EXTRACTION_GAP"
            ))
            
        for trans in fsm.transitions:
            check_id(trans.transition_id, "Transitions")
            if trans.source_state not in state_names and trans.source_state not in ("*", "ANY", "ALL", "default"):
                issues.append(make_validation_issue(
                    category="State Machines",
                    message=f"FSM '{fsm.name}' transition references undefined source state '{trans.source_state}'",
                    object_id=trans.transition_id,
                    severity="ERROR",
                    classification="EFSIR_EXTRACTION_GAP"
                ))
            if trans.target_state not in state_names:
                issues.append(make_validation_issue(
                    category="State Machines",
                    message=f"FSM '{fsm.name}' transition references undefined target state '{trans.target_state}'",
                    object_id=trans.transition_id,
                    severity="ERROR",
                    classification="EFSIR_EXTRACTION_GAP"
                ))

    # 8. Validate Constraints
    for const in ir.constraints:
        check_id(const.constraint_id, "Constraints")
        if not const.expected_behavior:
            issues.append(make_validation_issue(
                category="Constraints",
                message=f"Constraint '{const.constraint_id}' expected_behavior is empty",
                object_id=const.constraint_id,
                severity="WARNING",
                classification="EFSIR_EXTRACTION_GAP"
            ))
        
        # Verify references in source/target objects
        for obj_id in const.source_objects + const.target_objects:
            if obj_id not in signal_ids and obj_id not in component_ids and obj_id not in interface_ids and obj_id not in signal_names:
                issues.append(make_validation_issue(
                    category="Constraints",
                    message=f"Constraint '{const.constraint_id}' references object name or ID '{obj_id}' not explicitly declared in Signals, Interfaces, or Components",
                    object_id=const.constraint_id,
                    severity="WARNING",
                    classification="SOURCE_AMBIGUOUS"
                ))

    # 9. Validate Instructions
    for instr in getattr(ir, "instructions", []):
        check_id(instr.instruction_id, "Instructions")
        if not instr.name:
            issues.append(make_validation_issue(
                category="Instructions",
                message=f"Instruction '{instr.instruction_id}' is missing a name",
                object_id=instr.instruction_id,
                severity="WARNING",
                classification="EFSIR_EXTRACTION_GAP"
            ))
        try:
            width_val = int(instr.width)
        except (ValueError, TypeError):
            width_val = None
        if not width_val or width_val <= 0:
            issues.append(make_validation_issue(
                category="Instructions",
                message=f"Instruction '{instr.name or instr.instruction_id}' has invalid width {instr.width}",
                object_id=instr.instruction_id,
                severity="ERROR",
                classification="SOURCE_CONFLICT"
            ))
        else:
            instr.width = width_val
            for f in instr.fields:
                try:
                    msb_val = int(f.msb)
                except (ValueError, TypeError):
                    msb_val = None
                if msb_val is None or msb_val >= instr.width:
                    issues.append(make_validation_issue(
                        category="Instructions",
                        message=f"Instruction '{instr.name}' field '{f.name}' MSB {f.msb} exceeds instruction width {instr.width}",
                        object_id=instr.instruction_id,
                        severity="ERROR",
                        classification="SOURCE_CONFLICT"
                    ))
                else:
                    f.msb = msb_val

    # 10. Validate Opcodes
    seen_opcodes: Dict[str, Any] = {}
    for opc in getattr(ir, "opcodes", []):
        check_id(opc.opcode_id, "Opcodes")
        if not opc.mnemonic:
            issues.append(make_validation_issue(
                category="Opcodes",
                message=f"Opcode '{opc.opcode_id}' is missing a mnemonic",
                object_id=opc.opcode_id,
                severity="ERROR",
                classification="EFSIR_EXTRACTION_GAP"
            ))
        enc = (opc.binary_encoding or "").strip().lower()
        if enc:
            if enc in seen_opcodes:
                prev_opc = seen_opcodes[enc]
                if prev_opc.mnemonic != opc.mnemonic:
                    issues.append(make_validation_issue(
                        category="Opcodes",
                        message=f"Duplicate opcode assignment detected: Encoding '{enc}' is shared by '{prev_opc.mnemonic}' and '{opc.mnemonic}'",
                        object_id=opc.opcode_id,
                        severity="ERROR",
                        classification="SOURCE_CONFLICT",
                        evidence=f"Opcode encoding '{enc}' assigned to both '{prev_opc.mnemonic}' ({prev_opc.opcode_id}) and '{opc.mnemonic}' ({opc.opcode_id})",
                        required_information="Resolve opcode collision by assigning unique binary encodings."
                    ))
            else:
                seen_opcodes[enc] = opc

    # 10b. Validate Parameters
    for param in getattr(ir, "parameters", []):
        check_id(param.parameter_id, "Parameters")
        if not param.name:
            issues.append(make_validation_issue(
                category="Parameters",
                message=f"Parameter '{param.parameter_id}' is missing a name",
                object_id=param.parameter_id,
                severity="WARNING",
                classification="EFSIR_EXTRACTION_GAP"
            ))

    # 10c. Validate Functional Behaviors
    for fb in getattr(ir, "functional_behavior", []):
        check_id(fb.behavior_id, "FunctionalBehavior")
        if not fb.name and not fb.description:
            issues.append(make_validation_issue(
                category="FunctionalBehavior",
                message=f"Functional behavior '{fb.behavior_id}' has no name or description",
                object_id=fb.behavior_id,
                severity="WARNING",
                classification="EFSIR_EXTRACTION_GAP"
            ))

    # 11. Validate Conflicts and Resolutions
    known_entities = (
        signal_names | {s.signal_id for s in ir.signals} |
        {r.name for r in ir.registers} | register_ids |
        {c.name for c in ir.components} | component_ids |
        {i.name for i in ir.interfaces} | interface_ids |
        {ins.name for ins in getattr(ir, "instructions", [])} |
        {ins.mnemonic for ins in getattr(ir, "instructions", [])} |
        {ins.instruction_id for ins in getattr(ir, "instructions", [])} |
        {op.mnemonic for op in getattr(ir, "opcodes", [])} |
        {op.opcode_id for op in getattr(ir, "opcodes", [])} |
        {p.name for p in getattr(ir, "parameters", [])} |
        {p.parameter_id for p in getattr(ir, "parameters", [])} |
        {fb.name for fb in getattr(ir, "functional_behavior", [])} |
        {fb.behavior_id for fb in getattr(ir, "functional_behavior", [])} |
        {f.name for f in getattr(ir, "instruction_formats", [])} |
        {f.format_id for f in getattr(ir, "instruction_formats", [])}
    )
    for conf in getattr(ir, "conflicts", []):
        check_id(conf.conflict_id, "Conflicts")
        # Check affected entities resolve to existing entities
        for ent in (conf.affected_entities or conf.entities):
            if ent and ent not in known_entities and not any(k.lower() == ent.lower() for k in known_entities):
                issues.append(make_validation_issue(
                    category="Conflicts",
                    message=f"Conflict '{conf.conflict_id}' references unknown entity '{ent}'",
                    object_id=conf.conflict_id,
                    severity="WARNING",
                    classification="EFSIR_EXTRACTION_GAP"
                ))
        if conf.resolution_status != "RESOLVED":
            issues.append(make_validation_issue(
                category="Conflicts",
                message=f"Unresolved specification conflict [{conf.type}]: {conf.evidence or conf.description}",
                object_id=conf.conflict_id,
                severity=conf.severity,
                classification="SOURCE_CONFLICT",
                evidence=conf.evidence or conf.description,
                required_information=conf.explanation or conf.description
            ))

    for res in getattr(ir, "resolutions", []):
        check_id(res.resolution_id, "Resolutions")
        if res.conflict_id and not any(c.conflict_id == res.conflict_id for c in getattr(ir, "conflicts", [])):
            issues.append(make_validation_issue(
                category="Resolutions",
                message=f"Resolution '{res.resolution_id}' references unknown conflict '{res.conflict_id}'",
                object_id=res.resolution_id,
                severity="WARNING",
                classification="EFSIR_EXTRACTION_GAP"
            ))
        if (
            res.target_entity
            and res.target_entity.lower() not in ("target entity", "target_entity", "requirement", "specification", "design", "global", "")
            and res.target_entity not in known_entities
            and not any(k.lower() == res.target_entity.lower() for k in known_entities)
        ):
            issues.append(make_validation_issue(
                category="Resolutions",
                message=f"Resolution '{res.resolution_id}' targets unknown entity '{res.target_entity}'",
                object_id=res.resolution_id,
                severity="WARNING",
                classification="SOURCE_AMBIGUOUS"
            ))

    # 12. Grounding & Hallucination Audit (Metadata Grounding Gap checks)
    for comp in ir.components:
        trace = getattr(comp, "traceability", None)
        if not trace or (not trace.chunk_id and not trace.doc_id and (trace.confidence is not None and trace.confidence < 0.5)):
            conf_display = f"{trace.confidence:.2f}" if (trace and trace.confidence is not None) else "N/A"
            issues.append(make_validation_issue(
                category="Grounding",
                message=f"Component '{comp.name}' is ungrounded (missing specification traceability)",
                object_id=comp.component_id,
                severity="WARNING",
                classification="GROUNDING_GAP",
                evidence=f"Component ID: {comp.component_id} | Traceability: chunk_id={getattr(trace, 'chunk_id', None)}, doc_id={getattr(trace, 'doc_id', None)}, confidence={conf_display}",
                required_information=f"Verify source document origin for component '{comp.name}'."
            ))
            
    for sig in ir.signals:
        trace = getattr(sig, "traceability", None)
        if not trace or (not trace.chunk_id and not trace.doc_id and (trace.confidence is not None and trace.confidence < 0.5)):
            conf_display = f"{trace.confidence:.2f}" if (trace and trace.confidence is not None) else "N/A"
            issues.append(make_validation_issue(
                category="Grounding",
                message=f"Signal '{sig.name}' is ungrounded (missing specification traceability)",
                object_id=sig.signal_id,
                severity="WARNING",
                classification="GROUNDING_GAP",
                evidence=f"Signal ID: {sig.signal_id} | Traceability: chunk_id={getattr(trace, 'chunk_id', None)}, doc_id={getattr(trace, 'doc_id', None)}, confidence={conf_display}",
                required_information=f"Verify source document origin for signal '{sig.name}'."
            ))

    # Sanity Check & Deduplication Pass: Ensure zero malformed issues
    sanitized_issues = []
    seen_issue_keys = set()
    
    for iss in issues:
        # Sanity check: Ensure non-empty message/issue
        if not iss.get("issue") and not iss.get("message"):
            iss = make_validation_issue(
                category=iss.get("category", "System"),
                message=f"VALIDATION_INTERNAL_ERROR: Malformed validation issue generated for object '{iss.get('object_id', 'unknown')}'",
                object_id=iss.get("object_id", ""),
                severity="WARNING",
                classification="VALIDATION_INTERNAL_ERROR"
            )
            
        key = (iss.get("classification"), iss.get("category"), iss.get("issue"), iss.get("object_id"))
        if key not in seen_issue_keys:
            sanitized_issues.append(iss)
            seen_issue_keys.add(key)

    # Format structured findings and attach EFSValidationResult to ir
    structured_findings = []
    crit_count = 0
    warn_count = 0
    info_count = 0

    for idx, iss in enumerate(sanitized_issues):
        sev = iss.get("severity", "WARNING").upper()
        if sev in ("ERROR", "CRITICAL"):
            crit_count += 1
        elif sev == "WARNING":
            warn_count += 1
        else:
            info_count += 1

        finding = {
            "finding_id": f"VAL_{idx+1:03d}",
            "category": iss.get("category", "General"),
            "severity": sev,
            "message": iss.get("message") or iss.get("issue", ""),
            "object_id": iss.get("object_id", ""),
            "evidence": iss.get("evidence", ""),
            "remediation_action": iss.get("required_information", ""),
            "classification": iss.get("classification", "EFSIR_EXTRACTION_GAP"),
            "is_blocking": iss.get("is_blocking", sev in ("ERROR", "CRITICAL")),
            "issue": iss.get("message") or iss.get("issue", "")
        }
        structured_findings.append(finding)

    val_status = "PASSED"
    if crit_count > 0:
        val_status = "FAILED"
    elif warn_count > 0:
        val_status = "WARNINGS"

    ir.validation = EFSValidationResult(
        status=val_status,
        summary=EFSValidationSummary(
            total=len(structured_findings),
            critical=crit_count,
            warning=warn_count,
            info=info_count
        ),
        findings=structured_findings,
        design_id=ir.metadata.design_id,
        canonical_id=ir.metadata.canonical_id,
        generation_run_id=ir.metadata.generation_run_id,
        document_id=ir.metadata.document_id,
        ingestion_id=ir.metadata.ingestion_id
    )

    return structured_findings


def get_efs_ir_quality(ir: EFSIR, validation_issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate and return a detailed, non-arbitrary EFS IR quality scorecard."""
    num_comp = len(ir.components)
    num_if = len(ir.interfaces)
    num_sig = len(ir.signals)
    num_reg = len(ir.registers)
    num_instr = len(getattr(ir, "instructions", []))
    num_opc = len(getattr(ir, "opcodes", []))
    num_fsm = len(ir.fsms)
    num_const = len(ir.constraints)
    num_timing = len(ir.timing_rules)
    num_proto = len(ir.protocol_rules)
    num_flow = len(ir.flows)
    
    num_conflicts = len(getattr(ir, "conflicts", []))
    num_critical_conflicts = len([c for c in getattr(ir, "conflicts", []) if c.severity == "CRITICAL" and c.resolution_status == "UNRESOLVED"])
    
    # Calculate traceability coverage
    total_traceable = 0
    covered_traceable = 0
    
    all_traceable_lists = [
        ir.components, ir.interfaces, ir.signals, ir.registers,
        getattr(ir, "instructions", []), getattr(ir, "opcodes", []),
        ir.fsms, ir.constraints, ir.timing_rules, ir.protocol_rules, ir.flows
    ]
    
    for lst in all_traceable_lists:
        for obj in lst:
            total_traceable += 1
            trace = getattr(obj, "traceability", None)
            if trace and (trace.chunk_id or trace.original_text or (trace.confidence is not None and trace.confidence > 0.5)):
                covered_traceable += 1
                
    traceability_coverage = (covered_traceable / total_traceable * 100) if total_traceable > 0 else 100.0
    
    # Check for validation blocking errors
    errors = [i for i in validation_issues if i["severity"] == "ERROR"]
    
    status = "READY"
    reasons = []
    
    if num_critical_conflicts > 0:
        status = "BLOCKED"
        reasons.append(f"{num_critical_conflicts} unresolved critical specification conflicts.")
    if errors:
        status = "BLOCKED"
        reasons.append(f"{len(errors)} critical validation errors.")
        
    return {
        "components": num_comp,
        "interfaces": num_if,
        "signals": num_sig,
        "registers": num_reg,
        "instructions": num_instr,
        "opcodes": num_opc,
        "fsms": num_fsm,
        "constraints": num_const,
        "timing_rules": num_timing,
        "protocol_rules": num_proto,
        "flows": num_flow,
        "traceability_coverage": traceability_coverage,
        "conflicts": num_conflicts,
        "critical_conflicts": num_critical_conflicts,
        "status": status,
        "reasons": reasons
    }

