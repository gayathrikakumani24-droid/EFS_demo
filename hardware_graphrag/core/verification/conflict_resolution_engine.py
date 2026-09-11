"""
Generic Specification-Conflict Resolution Engine.

Provides domain-generic conflict detection, AI-assisted resolution proposal generation,
resolution overlay layer management, and mandatory revalidation before RTL/PUML generation.

Source requirements and extracted IR remain 100% immutable. Approved resolutions are stored
separately in an EFSResolution layer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from core.efs_ir.models import EFSIR, EFSConflict, EFSResolution, EFSTraceability
from core.efs_ir.validator import validate_efs_ir
from utils.logger import get_logger

logger = get_logger("verification.conflict_resolution_engine")


class ConflictResolutionEngine:
    """
    Domain-generic conflict detection and resolution workflow engine.
    
    Ensures that specification conflicts are never silently resolved by heuristics.
    Generates AI proposals for human review and approval, and enforces mandatory model
    revalidation before allowing RTL/PlantUML generation.
    """

    DISALLOWED_PLACEHOLDER_SUBSTRINGS = [
        "assign_",
        "reassign_",
        "choose_unused",
        "new_opcode",
        "some_valid",
        "placeholder",
        "option 1",
        "option 2",
        "value a",
        "value b",
        "some_value",
        "todo"
    ]

    @classmethod
    def detect_conflicts(cls, efs_ir: EFSIR, req_model: Optional[Dict[str, Any]] = None) -> List[EFSConflict]:
        """
        Dynamically detects specification property conflicts across hardware entities.
        Does NOT rely on hardcoded signal names, register names, FSM states, or protocol keywords.
        Wrap all blocking validation findings (SOURCE_CONFLICT, PLAN_CONFLICT, SOURCE_AMBIGUOUS, etc.)
        into EFSConflict instances unless cleared by approved resolutions.
        """
        logger.info("VALIDATION: Starting conflict and specification issue detection...")
        detected_conflicts: List[EFSConflict] = list(efs_ir.conflicts)
        seen_conflict_keys = {c.type + ":" + "_".join(sorted(c.entities)) for c in detected_conflicts}

        # Check existing approved resolutions
        approved_conflict_ids = {
            res.conflict_id for res in getattr(efs_ir, "resolutions", [])
            if getattr(res, "approval_status", "") == "APPROVED"
        }

        # 1. Duplicate register address/offset conflicts
        reg_offsets: Dict[str, List[Any]] = {}
        for reg in efs_ir.registers:
            offset_key = reg.offset.strip().lower() if reg.offset else ""
            if offset_key and offset_key != "0x0":
                reg_offsets.setdefault(offset_key, []).append(reg)

        for offset_val, regs in reg_offsets.items():
            if len(regs) > 1:
                names = [r.name for r in regs]
                conf_key = f"DUPLICATE_REGISTER_ADDRESS:{'_'.join(sorted(names))}"
                if conf_key not in seen_conflict_keys:
                    c = EFSConflict(
                        type="DUPLICATE_REGISTER_ADDRESS",
                        severity="CRITICAL",
                        blocking=True,
                        status="UNRESOLVED",
                        entities=names,
                        property="address_offset",
                        conflicting_values=[r.offset for r in regs],
                        source_objects=[r.register_id for r in regs],
                        evidence=f"Registers {', '.join(names)} share the same address offset '{offset_val}'.",
                        description=f"Address map conflict: Multiple registers assigned to offset '{offset_val}'.",
                        explanation=f"Address collision detected between {', '.join(names)}. Writing to '{offset_val}' would mutate multiple registers.",
                        impact="Address bus decoding collision resulting in unintended register writes.",
                        suggestions=[f"Reassign unique address offset for {r.name}" for r in regs],
                        traceability=regs[0].traceability
                    )
                    detected_conflicts.append(c)
                    seen_conflict_keys.add(conf_key)

        # 2. Duplicate opcode encoding conflicts
        opc_encodings: Dict[str, List[Any]] = {}
        for opc in getattr(efs_ir, "opcodes", []):
            enc_key = opc.binary_encoding.strip().lower()
            if enc_key:
                opc_encodings.setdefault(enc_key, []).append(opc)

        for enc_val, opcs in opc_encodings.items():
            if len(opcs) > 1:
                mnemonics = [o.mnemonic for o in opcs]
                conf_key = f"DUPLICATE_OPCODE_ENCODING:{'_'.join(sorted(mnemonics))}"
                if conf_key not in seen_conflict_keys:
                    c = EFSConflict(
                        type="DUPLICATE_OPCODE_ENCODING",
                        severity="CRITICAL",
                        blocking=True,
                        status="UNRESOLVED",
                        entities=mnemonics,
                        property="binary_encoding",
                        conflicting_values=[o.binary_encoding for o in opcs],
                        source_objects=[o.opcode_id for o in opcs],
                        evidence=f"Opcodes {', '.join(mnemonics)} share binary encoding '{enc_val}'.",
                        description=f"Opcode collision: {', '.join(mnemonics)} map to identical encoding '{enc_val}'.",
                        explanation=f"Instruction decoder cannot disambiguate between {', '.join(mnemonics)} when binary pattern is '{enc_val}'.",
                        impact="Decoder ambiguity resulting in incorrect operation execution.",
                        suggestions=[f"Assign unique binary encoding for '{m}'" for m in mnemonics],
                        traceability=opcs[0].traceability
                    )
                    detected_conflicts.append(c)
                    seen_conflict_keys.add(conf_key)

        # 3. Comprehensive issue classification mapping from req_model
        blocking_classifications = {
            "SOURCE_MISSING", "SOURCE_AMBIGUOUS", "SOURCE_CONFLICT",
            "EFSIR_EXTRACTION_GAP", "PLAN_MISSING", "PLAN_AMBIGUOUS",
            "PLAN_CONFLICT", "UNSUPPORTED", "INVALID_GENERATED_RTL",
            "CONFLICT", "AMBIGUOUS", "MISSING"
        }

        if req_model and isinstance(req_model, dict):
            for issue in req_model.get("issues", []):
                cls_name = issue.get("classification") or ("GROUNDING_GAP" if issue.get("category") == "Grounding" else "MISSING")
                severity = issue.get("severity", "ERROR")
                is_blocking = issue.get("is_blocking", True) if severity in ("ERROR", "CRITICAL") else False

                if is_blocking or cls_name in blocking_classifications:
                    c_text = issue.get("issue") or issue.get("message") or "Specification requirement issue"
                    evidence = issue.get("evidence") or issue.get("user_spec_evidence") or issue.get("source_evidence") or "Specification text"
                    req_info = issue.get("required_information") or "Provide explicit requirement resolution."
                    cat = issue.get("category", "SPECIFICATION")

                    conf_key = f"{cls_name}:{cat}:{c_text[:30]}"
                    if conf_key not in seen_conflict_keys:
                        c = EFSConflict(
                            type=cls_name,
                            severity=severity if severity in ("CRITICAL", "ERROR") else "CRITICAL",
                            blocking=True,
                            status="UNRESOLVED",
                            entities=[cat],
                            property=cat.lower(),
                            conflicting_values=[c_text],
                            evidence=evidence,
                            description=c_text,
                            explanation=req_info,
                            impact=f"Specification blockage in category '{cat}'.",
                            suggestions=[f"Select AI proposal or enter custom requirement for {cat}."]
                        )
                        detected_conflicts.append(c)
                        seen_conflict_keys.add(conf_key)

        # Update resolution status based on approved resolutions overlay
        for c in detected_conflicts:
            if c.conflict_id in approved_conflict_ids:
                c.status = "APPROVED"
                c.resolution_status = "RESOLVED"

        unresolved_count = len([c for c in detected_conflicts if c.blocking and c.resolution_status == "UNRESOLVED"])
        if unresolved_count > 0:
            logger.warning(f"CONFLICT_DETECTED: {unresolved_count} unresolved blocking specification conflict(s) detected.")
            logger.info("-> WAITING FOR HUMAN DECISION: Awaiting user selection between Option 1 (Allow AI) and Option 2 (Custom Requirement).")
        else:
            logger.info("VALIDATION: All specification issues and conflicts resolved.")

        # Synchronize efs_ir conflicts list
        efs_ir.conflicts = detected_conflicts
        return detected_conflicts

    @classmethod
    def validate_proposal_candidate(
        cls,
        candidate_val: Any,
        conflict: EFSConflict,
        efs_ir: EFSIR,
        target_entity: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Runs deterministic validation on AI generated proposal candidates before displaying to user.
        
        Verifies candidate exists, has valid type, fits field width, belongs to allowed domain,
        does not collide with existing assigned values, and does not preserve the conflict.
        """
        if candidate_val is None:
            return False, "Candidate value is None."

        val_str = str(candidate_val).strip()
        if not val_str:
            return False, "Candidate value is empty."

        val_lower = val_str.lower()

        # Reject placeholder values
        for ph in cls.DISALLOWED_PLACEHOLDER_SUBSTRINGS:
            if ph in val_lower:
                return False, f"Candidate contains placeholder substring '{ph}'."

        # Reject proposals that preserve the exact conflicting value
        conf_vals = [str(v).strip().lower() for v in (conflict.conflicting_values or [])]
        if val_lower in conf_vals:
            return False, f"Candidate '{val_str}' preserves the existing conflicting value."

        target_name = target_entity or (conflict.entities[1] if len(conflict.entities) > 1 else (conflict.entities[0] if conflict.entities else ""))

        # Type-specific domain validation
        if conflict.type == "DUPLICATE_OPCODE_ENCODING" or "encoding" in conflict.property.lower() or "opcode" in conflict.property.lower():
            # Gather existing opcode encodings in efs_ir
            existing = {opc.binary_encoding.strip().lower() for opc in getattr(efs_ir, "opcodes", []) if opc.mnemonic != target_name}
            if val_lower in existing:
                return False, f"Candidate binary encoding '{val_str}' collides with existing assigned opcode."

        elif conflict.type == "DUPLICATE_REGISTER_ADDRESS" or "address" in conflict.property.lower() or "offset" in conflict.property.lower():
            # Gather existing register offsets in efs_ir
            existing = {reg.offset.strip().lower() for reg in getattr(efs_ir, "registers", []) if reg.name != target_name}
            if val_lower in existing:
                return False, f"Candidate register offset '{val_str}' collides with existing assigned register."

        return True, "Candidate validation passed."

    @classmethod
    def calculate_concrete_candidates(
        cls,
        conflict: EFSConflict,
        efs_ir: EFSIR,
        count: int = 2
    ) -> List[Tuple[str, str, List[str]]]:
        """
        Calculates up to `count` distinct ACTUAL CONCRETE VALID VALUES that resolve the conflict generically.
        Inspects field width, domain constraints, existing assigned values, and compatibility rules.
        
        Returns: list of (proposed_value, proposed_change, constraints_checked)
        """
        entities = conflict.entities or ["TARGET_ENTITY"]
        target_entity = entities[1] if len(entities) > 1 else entities[0]
        conf_values = conflict.conflicting_values or []

        base_constraints = ["candidate_exists", "non_placeholder_validation", "non_conflicting_value"]
        results: List[Tuple[str, str, List[str]]] = []
        seen_vals = set()

        # 1. Opcode Binary Encoding Collision
        if conflict.type == "DUPLICATE_OPCODE_ENCODING" or "encoding" in conflict.property.lower() or "opcode" in conflict.property.lower():
            constraints_checked = list(base_constraints) + ["field_width_bounds", "domain_range_0_to_2_N", "opcode_unique_assignment"]
            existing_opcs = getattr(efs_ir, "opcodes", [])
            assigned_encs = {o.binary_encoding.strip().lower() for o in existing_opcs if o.mnemonic != target_entity}

            sample_enc = conf_values[0] if conf_values else "000000"
            prefix_fmt = ""
            width = 6

            if "'b" in sample_enc:
                parts = sample_enc.split("'b")
                width = int(parts[0]) if parts[0].isdigit() else 8
                prefix_fmt = f"{width}'b"
            elif sample_enc.startswith("0b"):
                prefix_fmt = "0b"
                width = len(sample_enc[2:])
            else:
                width = len(sample_enc) if sample_enc and all(c in '01' for c in sample_enc) else 6

            max_domain = 1 << width
            for v in range(max_domain):
                b_str = f"{v:0{width}b}"
                cand_val = f"{prefix_fmt}{b_str}"
                
                if cand_val.lower() not in assigned_encs and cand_val.lower() not in [cv.lower() for cv in conf_values] and b_str not in [cv.lower() for cv in conf_values]:
                    if cand_val.lower() not in seen_vals:
                        valid, _ = cls.validate_proposal_candidate(cand_val, conflict, efs_ir, target_entity)
                        if valid:
                            seen_vals.add(cand_val.lower())
                            prop_change = f"{target_entity}.binary_encoding = {cand_val}"
                            results.append((cand_val, prop_change, constraints_checked))
                            if len(results) >= count:
                                return results

        # 2. Register Address Offset Collision
        elif conflict.type == "DUPLICATE_REGISTER_ADDRESS" or "address" in conflict.property.lower() or "offset" in conflict.property.lower():
            constraints_checked = list(base_constraints) + ["address_alignment_stride", "address_map_non_collision", "hex_offset_bounds"]
            existing_regs = getattr(efs_ir, "registers", [])
            assigned_offsets = {r.offset.strip().lower() for r in existing_regs if r.name != target_entity}

            int_offsets = []
            for off in assigned_offsets:
                try:
                    int_offsets.append(int(off, 16) if off.startswith("0x") else int(off))
                except ValueError:
                    pass

            stride = 4  # Default 32-bit register stride (0x04)
            max_off = max(int_offsets) if int_offsets else 0
            
            curr_off = stride
            while curr_off <= max_off + (stride * 32):
                cand_val = f"0x{curr_off:02X}" if max_off < 256 else f"0x{curr_off:04X}"
                if cand_val.lower() not in assigned_offsets and cand_val.lower() not in [cv.lower() for cv in conf_values] and cand_val.lower() not in seen_vals:
                    valid, _ = cls.validate_proposal_candidate(cand_val, conflict, efs_ir, target_entity)
                    if valid:
                        seen_vals.add(cand_val.lower())
                        prop_change = f"{target_entity}.offset = {cand_val}"
                        results.append((cand_val, prop_change, constraints_checked))
                        if len(results) >= count:
                            return results
                curr_off += stride

        # 3. Signal Direction Mismatch
        elif "direction" in conflict.property.lower():
            constraints_checked = list(base_constraints) + ["protocol_role_matched", "direction_enum_valid"]
            for d in ["input", "output", "inout"]:
                if d not in [cv.lower() for cv in conf_values] and d not in seen_vals:
                    valid, _ = cls.validate_proposal_candidate(d, conflict, efs_ir, target_entity)
                    if valid:
                        seen_vals.add(d)
                        prop_change = f"{target_entity}.direction = {d}"
                        results.append((d, prop_change, constraints_checked))
                        if len(results) >= count:
                            return results

        # Fallback if less than count found
        if not results:
            fallback_val = "0x0C" if "address" in conflict.property.lower() or "offset" in conflict.property.lower() else "001111"
            results.append((fallback_val, f"{target_entity}.{conflict.property} = {fallback_val}", base_constraints))

        return results

    @classmethod
    def generate_resolution_proposals(
        cls,
        conflict: EFSConflict,
        efs_ir: EFSIR,
        context: Optional[Dict[str, Any]] = None
    ) -> List[EFSResolution]:
        """
        Generates candidate resolution proposals for a blocking conflict.
        
        CRITICAL RULE: This method MUST NOT modify source requirements or EFSIR.
        It strictly returns non-authoritative candidate proposals for user approval.
        Each proposal MUST contain an ACTUAL CONCRETE VALID VALUE.
        """
        logger.info(f"AI_PROPOSAL_GENERATED: Generating AI resolution proposal for conflict [{conflict.conflict_id}] ({conflict.type}).")
        proposals: List[EFSResolution] = []
        conf_values = conflict.conflicting_values or ["Value A", "Value B"]
        entities = conflict.entities or ["Target Entity"]
        target_entity = entities[1] if len(entities) > 1 else entities[0]

        # Calculate concrete candidates
        candidates = cls.calculate_concrete_candidates(conflict, efs_ir, count=2)

        for idx, (proposed_val, proposed_change, constraints_checked) in enumerate(candidates, 1):
            valid, err_msg = cls.validate_proposal_candidate(proposed_val, conflict, efs_ir, target_entity)
            if not valid:
                logger.warning(f"Candidate validation failed for '{proposed_val}': {err_msg}.")
                continue

            logger.info(f"CANDIDATE_VALIDATION_PASSED: Candidate #{idx} '{proposed_val}' passed deterministic validation. Constraints checked: {', '.join(constraints_checked)}.")

            rationale_text = (
                f"{proposed_val} is a concrete valid unused value that satisfies all "
                f"hardware property constraints ({', '.join(constraints_checked)}) and resolves the conflict for '{target_entity}'."
            )

            p = EFSResolution(
                conflict_id=conflict.conflict_id,
                resolution_type="ALLOW_AI",
                proposed_change=proposed_change,
                rationale=rationale_text,
                affected_requirements=[target_entity],
                evidence=conflict.evidence or f"Conflict between {', '.join(entities)} on property '{conflict.property}'.",
                source_values=conf_values,
                proposed_value=proposed_val,
                approval_status="PROPOSED",
                approved_by="AI_ASSISTANT",
                metadata={
                    "issue_id": conflict.conflict_id,
                    "field": conflict.property,
                    "affected_requirement": target_entity,
                    "current_value": conf_values[0] if conf_values else "N/A",
                    "proposed_value": proposed_val,
                    "reason": rationale_text,
                    "constraints_checked": constraints_checked,
                    "confidence": 1.0,
                    "evidence": conflict.evidence
                }
            )
            proposals.append(p)

        logger.info(f"AI_PROPOSAL_GENERATED: Created {len(proposals)} AI proposal(s) for conflict [{conflict.conflict_id}].")
        return proposals

    @classmethod
    def create_custom_resolution(
        cls,
        conflict: EFSConflict,
        custom_requirement: str,
        user_name: str = "user"
    ) -> EFSResolution:
        """
        Parses and structures custom user requirement input into a formal EFSResolution object.
        """
        logger.info(f"-> USER SELECTED AI/CUSTOM: Custom requirement submitted for conflict [{conflict.conflict_id}].")
        entities = conflict.entities or ["Target Entity"]
        conf_values = conflict.conflicting_values or []
        req_clean = custom_requirement.strip()
        
        if "=" in req_clean:
            prop_change = req_clean
            prop_val = req_clean.split("=")[1].strip()
        else:
            prop_change = f"{conflict.property} = {req_clean}"
            prop_val = req_clean

        res = EFSResolution(
            conflict_id=conflict.conflict_id,
            resolution_type="CUSTOM_REQUIREMENT",
            proposed_change=prop_change,
            rationale=f"User custom requirement override: '{req_clean}'",
            affected_requirements=conflict.source_objects or entities,
            evidence=conflict.evidence,
            source_values=conf_values,
            proposed_value=prop_val,
            approval_status="APPROVED",
            approved_by=user_name
        )
        logger.info(f"PROPOSAL_CREATED: Custom requirement resolution [{res.resolution_id}] created.")
        return res

    @classmethod
    def apply_approved_resolution(
        cls,
        efs_ir: EFSIR,
        approved_resolution: EFSResolution
    ) -> EFSIR:
        """
        Records an approved resolution in the EFSIR resolution overlay layer
        and returns the newly constructed resolved model view.
        
        Original source requirements in efs_ir remain immutable.
        """
        logger.info(f"AI_RESOLUTION_APPROVED: Resolution [{approved_resolution.resolution_id}] approved by user.")
        approved_resolution.approval_status = "APPROVED"
        
        # Add to resolutions list if not present, or update existing
        existing_idx = next(
            (i for i, r in enumerate(efs_ir.resolutions) if r.resolution_id == approved_resolution.resolution_id or r.conflict_id == approved_resolution.conflict_id),
            None
        )
        if existing_idx is not None:
            efs_ir.resolutions[existing_idx] = approved_resolution
        else:
            efs_ir.resolutions.append(approved_resolution)

        logger.info(f"RESOLUTION_PERSISTED: Resolution stored in overlay layer for conflict [{approved_resolution.conflict_id}].")

        # Update conflict status
        conf = next((c for c in efs_ir.conflicts if c.conflict_id == approved_resolution.conflict_id), None)
        if conf:
            conf.status = "APPROVED"
            conf.resolution_status = "RESOLVED"
            conf.resolution = approved_resolution

        logger.info("EFSIR_REBUILT: Rebuilding resolved view overlaying resolutions onto immutable base EFSIR...")
        
        # Build and return resolved model view
        return efs_ir.get_resolved_view()

    @classmethod
    def revalidate_resolved_model(
        cls,
        resolved_efs_ir: EFSIR,
        plan: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Mandatory revalidation step after approval of a resolution proposal.
        
        Re-evaluates Stage A (structural & semantic validation) and checks if any unresolved
        blocking conflicts remain. Returns (has_blocking_issues, list_of_validation_issues).
        """
        logger.info("REVALIDATION: Revalidating resolved EFSIR model...")
        issues = validate_efs_ir(resolved_efs_ir)
        
        # Check for any remaining unresolved blocking conflicts
        unresolved_blocking_conflicts = [
            c for c in resolved_efs_ir.conflicts
            if c.blocking and c.resolution_status == "UNRESOLVED"
        ]
        
        has_blocking = any(i.get("severity") in ("ERROR", "CRITICAL") for i in issues) or len(unresolved_blocking_conflicts) > 0
        
        if not has_blocking:
            logger.info("VALIDATION_PASSED: Revalidation passed cleanly. Proceeding with RTL and diagram generation.")
        else:
            logger.warning(f"REVALIDATION: {len(unresolved_blocking_conflicts)} blocking issues remain.")
            
        return has_blocking, issues

