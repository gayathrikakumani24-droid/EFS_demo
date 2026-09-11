"""
EFS IR Validator.

Performs deterministic checking of the EFS IR model to find structural issues,
dangling references, invalid definitions, and missing traceability.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List
from core.efs_ir.models import EFSIR


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

    # 6. Validate registers
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
        if not reg.offset:
            issues.append(make_validation_issue(
                category="Registers",
                message=f"Register '{reg.name}' has no defined address offset",
                object_id=reg.register_id,
                severity="WARNING",
                classification="SOURCE_AMBIGUOUS"
            ))
            
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

    # 11. Report Conflicts as validation issues
    for conf in getattr(ir, "conflicts", []):
        check_id(conf.conflict_id, "Conflicts")
        if conf.resolution_status == "UNRESOLVED":
            issues.append(make_validation_issue(
                category="Conflicts",
                message=f"Unresolved specification conflict [{conf.type}]: {conf.evidence or conf.description}",
                object_id=conf.conflict_id,
                severity=conf.severity,
                classification="SOURCE_CONFLICT",
                evidence=conf.evidence or conf.description,
                required_information=conf.explanation or conf.description
            ))

    # 12. Grounding & Hallucination Audit (Metadata Grounding Gap checks)
    for comp in ir.components:
        trace = getattr(comp, "traceability", None)
        if not trace or (not trace.chunk_id and not trace.doc_id and trace.confidence < 0.5):
            issues.append(make_validation_issue(
                category="Grounding",
                message=f"Component '{comp.name}' is ungrounded (missing specification traceability)",
                object_id=comp.component_id,
                severity="WARNING",
                classification="GROUNDING_GAP",
                evidence=f"Component ID: {comp.component_id} | Traceability: chunk_id={getattr(trace, 'chunk_id', None)}, doc_id={getattr(trace, 'doc_id', None)}, confidence={getattr(trace, 'confidence', 0.0)}",
                required_information=f"Verify source document origin for component '{comp.name}'."
            ))
            
    for sig in ir.signals:
        trace = getattr(sig, "traceability", None)
        if not trace or (not trace.chunk_id and not trace.doc_id and trace.confidence < 0.5):
            issues.append(make_validation_issue(
                category="Grounding",
                message=f"Signal '{sig.name}' is ungrounded (missing specification traceability)",
                object_id=sig.signal_id,
                severity="WARNING",
                classification="GROUNDING_GAP",
                evidence=f"Signal ID: {sig.signal_id} | Traceability: chunk_id={getattr(trace, 'chunk_id', None)}, doc_id={getattr(trace, 'doc_id', None)}, confidence={getattr(trace, 'confidence', 0.0)}",
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

    return sanitized_issues


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
            if trace and (trace.chunk_id or trace.original_text or trace.confidence > 0.5):
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

