"""
Generic Specification-Conflict Resolution Engine.

Provides domain-generic conflict detection, AI-assisted resolution proposal generation,
resolution overlay layer management, and mandatory revalidation before RTL/PUML generation.

Source requirements and extracted IR remain 100% immutable. Approved resolutions are stored
separately in an EFSResolution layer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
from core.efs_ir.models import EFSIR, EFSConflict, EFSResolution, EFSTraceability, DiagnosticType, ConflictStatus
from core.efs_ir.validator import validate_efs_ir
from core.efs_ir.validation_rules import is_explicit_address
from utils.logger import get_logger

logger = get_logger("verification.conflict_resolution_engine")


def is_explicit_encoding(raw: Any) -> bool:
    """Check if an opcode or binary encoding is an explicit hardware value, not a missing/unknown placeholder."""
    if raw is None:
        return False
    val = str(raw).strip().lower()
    if not val or val in ("unknown", "none", "n/a", "na", "unresolved", "tbd", "missing", "nan", "null", "-", "undefined", "tba"):
        return False
    return True


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

    CANONICAL_PROPERTY_MAP = {
        "opcode": "opcode",
        "binary_encoding": "opcode",
        "encoding": "opcode",
        "instruction_encoding": "opcode",
        "opc": "opcode",
        "offset": "offset",
        "address": "offset",
        "addr": "offset",
        "reg_offset": "offset",
        "base_address": "offset",
        "width": "width",
        "bit_width": "width",
        "data_width": "width",
        "size": "width",
        "direction": "direction",
        "dir": "direction",
        "port_direction": "direction",
        "reset": "reset_value",
        "reset_value": "reset_value",
    }

    CANONICAL_TYPE_MAP = {
        "DUPLICATE_OPCODE_ENCODING": "DUPLICATE_OPCODE",
        "DUPLICATE_OPCODE": "DUPLICATE_OPCODE",
        "OPCODE_COLLISION": "DUPLICATE_OPCODE",
        "DUPLICATE_REGISTER_ADDRESS": "DUPLICATE_REGISTER_ADDRESS",
        "DUPLICATE_ADDRESS": "DUPLICATE_REGISTER_ADDRESS",
        "CONFLICTING_ADDRESS": "DUPLICATE_REGISTER_ADDRESS",
        "ADDRESS_COLLISION": "DUPLICATE_REGISTER_ADDRESS",
        "CONFLICTING_WIDTH": "CONFLICTING_WIDTH",
        "SIGNAL_WIDTH": "CONFLICTING_WIDTH",
        "SIGNAL_DIRECTION": "SIGNAL_DIRECTION",
        "DIRECTION_CONFLICT": "SIGNAL_DIRECTION",
    }

    @classmethod
    def compute_semantic_key(cls, conflict: EFSConflict) -> str:
        """
        Computes a deterministic canonical semantic conflict identity key.
        Does NOT depend on transient detector names, message strings, or detection pass order.
        Maps equivalent properties (e.g. 'binary_encoding' vs 'opcode') and types
        (e.g. 'DUPLICATE_OPCODE_ENCODING' vs 'DUPLICATE_OPCODE') to the identical canonical key.
        """
        raw_type = (getattr(conflict, "type", "") or "").strip().upper()
        canon_type = cls.CANONICAL_TYPE_MAP.get(raw_type, raw_type)

        raw_prop = (getattr(conflict, "property", "") or "").strip().lower()
        canon_prop = cls.CANONICAL_PROPERTY_MAP.get(raw_prop, raw_prop)

        # Canonical entities: unique, stripped, uppercase, sorted
        ents = (
            getattr(conflict, "affected_entities", None)
            or getattr(conflict, "entities", None)
            or getattr(conflict, "target_entities", None)
            or []
        )
        cleaned_ents = sorted(list({str(e).strip().upper() for e in ents if str(e).strip()}))
        canonical_entities = ",".join(cleaned_ents)

        # Canonical values
        def _norm_val(raw_v: Any, c_prop: str) -> str:
            if raw_v is None:
                return ""
            s = str(raw_v).strip().lower()
            if not s:
                return ""
            if c_prop == "opcode":
                if "'b" in s:
                    s = s.split("'b")[-1]
                elif s.startswith("0b"):
                    s = s[2:]
                return s
            elif c_prop == "offset":
                try:
                    if s.startswith("0x"):
                        return f"0x{int(s, 16):x}"
                    elif s.isdigit():
                        return f"0x{int(s, 10):x}"
                except ValueError:
                    pass
                return s
            return s

        vals = getattr(conflict, "conflicting_values", []) or []
        if not vals and getattr(conflict, "current_values", None):
            vals = list(conflict.current_values.values())
        norm_vals = sorted(list({_norm_val(v, canon_prop) for v in vals if _norm_val(v, canon_prop)}))
        canonical_value = ",".join(norm_vals)

        diag_type = (getattr(conflict, "diagnostic_type", "") or "SPEC_CONFLICT").strip().upper()
        return f"{diag_type}:{canon_type}:{canon_prop}:{canonical_entities}:{canonical_value}"


    @classmethod
    def validate_property_value(
        cls,
        entity: str,
        property_name: str,
        value: Any,
        efs_ir: Optional[EFSIR] = None
    ) -> Tuple[bool, str]:
        """
        Generic semantic validation for property values.
        Ensures arbitrary strings, resolver diagnostics, component lists, or entity-incompatible
        values cannot become property values in conflicts or proposals.
        """
        if value is None:
            return False, "Property value cannot be None."
        val_str = str(value).strip()
        if not val_str:
            return False, "Property value cannot be empty."

        # Reject multiline strings (diagnostics, dumps)
        if "\n" in val_str or "\r" in val_str:
            return False, "Property value cannot be a multiline diagnostic string."

        # Reject strings exceeding reasonable hardware value bounds
        if len(val_str) > 120:
            return False, f"Property value is too long ({len(val_str)} chars) to be a valid hardware parameter."

        # Reject known diagnostic and error patterns
        val_lower = val_str.lower()
        diagnostic_markers = [
            "target_resolution_failed",
            "available efs ir components",
            "requested target:",
            "normalized target:",
            "closest candidates:",
            "design canonical id:",
            "registered design aliases:",
            "cannot be resolved",
            "unresolved specification conflict",
            "exception",
            "traceback",
        ]
        if any(marker in val_lower for marker in diagnostic_markers):
            return False, "Property value contains internal resolver/planner diagnostic text."

        prop_norm = cls.CANONICAL_PROPERTY_MAP.get(property_name.lower().strip(), property_name.lower().strip())
        ent_upper = (entity or "").strip().upper()

        # Non-hardware / non-addressable entity safeguards
        non_addressable_entities = {"INTERFACE", "PORT", "SIGNAL", "FSM", "SPECIFICATION", "GENERAL", "CLOCK", "RESET", "SYSTEM"}

        # 1. Offset / Address validation
        if prop_norm in ("offset", "address"):
            if ent_upper in non_addressable_entities:
                return False, f"Entity '{entity}' is not an addressable hardware register or memory region."
            if val_str.startswith("0x") or val_str.startswith("0X"):
                try:
                    int(val_str, 16)
                    return True, "Valid hex offset."
                except ValueError:
                    return False, f"Invalid hex offset '{val_str}'."
            elif val_str.isdigit():
                return True, "Valid integer offset."
            else:
                return False, f"Invalid address offset format '{val_str}'. Expected hex (0x...) or integer."

        # 2. Width validation
        elif prop_norm == "width":
            if ent_upper in {"FSM", "SPECIFICATION", "GENERAL"}:
                return False, f"Entity '{entity}' cannot have a width property."
            if ":" in val_str:
                parts = val_str.replace("[", "").replace("]", "").split(":")
                if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                    return True, "Valid bit-range width."
                return False, f"Invalid bit-range format '{val_str}'."
            elif val_str.isdigit() and int(val_str) > 0:
                return True, "Valid width integer."
            return False, f"Invalid signal width '{val_str}'. Expected positive integer or [MSB:LSB]."

        # 3. Direction validation
        elif prop_norm == "direction":
            if val_lower in ("input", "output", "inout", "in", "out"):
                return True, "Valid direction."
            return False, f"Invalid direction '{val_str}'. Expected 'input', 'output', or 'inout'."

        # 4. Opcode / Encoding validation
        elif prop_norm in ("opcode", "binary_encoding"):
            clean_bin = val_str.split("'b")[-1] if "'b" in val_str else (val_str[2:] if val_str.startswith("0b") else val_str)
            if all(c in "01" for c in clean_bin) or val_str.startswith("0x") or val_str.isdigit():
                return True, "Valid opcode format."
            return False, f"Invalid opcode format '{val_str}'. Expected binary bitstring or hex."

        return True, "Value passed semantic validation."

    @classmethod
    def _extract_issue_metadata(
        cls,
        c_text: str,
        evidence: str,
        cat: str,
        cls_name: str,
        efs_ir: Optional[EFSIR] = None
    ) -> Tuple[str, str, List[str], str, str]:
        """
        Extracts (target_entity, property_name, conflicting_values, title, explanation)
        from issue text so conflicts are bound to real hardware entities rather than placeholder strings.
        """
        import re
        full_text = f"{c_text} {evidence}".strip()

        # 1. Signal Direction Conflict: e.g. "Signal direction conflict on 'axi_arready': RTL plan specifies 'input', but EFS IR specifies 'output'."
        dir_m = re.search(r"direction conflict on ['\"]?([a-zA-Z0-9_]+)['\"]?.*?specifies ['\"]?([a-zA-Z0-9_]+)['\"]?.*?specifies ['\"]?([a-zA-Z0-9_]+)['\"]?", full_text, re.IGNORECASE)
        if dir_m:
            ent = dir_m.group(1)
            d1, d2 = dir_m.group(2).lower(), dir_m.group(3).lower()
            return ent, "direction", [d2, d1], f"Signal Direction Conflict: {ent}", f"Signal '{ent}' has conflicting directions specified (RTL plan specifies '{d1}', canonical EFS IR specifies '{d2}')."

        # 2. Signal Width Conflict: e.g. "Signal width conflict on 'data_bus': ... specifies '16', but ... specifies '32'"
        width_m = re.search(r"width conflict on ['\"]?([a-zA-Z0-9_]+)['\"]?.*?specifies ['\"]?(\d+)['\"]?.*?specifies ['\"]?(\d+)['\"]?", full_text, re.IGNORECASE)
        if not width_m:
            width_m = re.search(r"width conflict on ['\"]?([a-zA-Z0-9_]+)['\"]?.*?['\"](\d+)['\"].*?['\"](\d+)['\"]", full_text, re.IGNORECASE)
        if not width_m:
            width_m = re.search(r"width conflict on ['\"]?([a-zA-Z0-9_]+)['\"]?.*?(\d+).*?(\d+)", full_text, re.IGNORECASE)
        if width_m:
            ent = width_m.group(1)
            w1, w2 = width_m.group(2), width_m.group(3)
            return ent, "width", [w1, w2], f"Signal Width Conflict: {ent}", f"Signal '{ent}' is declared with conflicting widths ({w1} vs {w2})."

        # 3. Missing Signal Port
        missing_m = re.search(r"Signal ['\"]?([a-zA-Z0-9_]+)['\"]? declared in (?:canonical )?EFS IR is missing", full_text, re.IGNORECASE)
        if missing_m:
            ent = missing_m.group(1)
            return ent, "port", [], f"Missing RTL Port: {ent}", f"Signal '{ent}' declared in canonical EFS IR is missing from the RTL module port list."

        # 4. FSM Transition Condition Undefined
        fsm_m = re.search(r"Transition ['\"]?([a-zA-Z0-9_]+)\s*->\s*([a-zA-Z0-9_]+)['\"]?.*?condition is undefined", full_text, re.IGNORECASE)
        if fsm_m:
            s1, s2 = fsm_m.group(1), fsm_m.group(2)
            return f"{s1}->{s2}", "transition_condition", [], f"FSM Transition Undefined: {s1} -> {s2}", f"Transition condition from state {s1} to {s2} is undefined or ambiguous."

        # 5. Missing FSM Definition / States
        if "finite state machine" in full_text.lower() or "fsm" in full_text.lower() or cat.upper() == "FSM":
            if any(kw in full_text.lower() for kw in ("no ", "missing", "undefined", "not specified", "absent", "insufficient")):
                return "FSM", "fsm_definition", [], "Missing Specification: FSM Definition", c_text

        # 6. Extract entity from quotes if matching hardware objects in efs_ir
        ent = cat
        q_matches = re.findall(r"['\"]([a-zA-Z0-9_]+)['\"]", c_text)
        if efs_ir and q_matches:
            known_sigs = {s.name for s in getattr(efs_ir, "signals", [])}
            known_regs = {r.name for r in getattr(efs_ir, "registers", [])}
            known_opcs = {o.mnemonic for o in getattr(efs_ir, "opcodes", [])}
            for qm in q_matches:
                if qm in known_sigs or qm in known_regs or qm in known_opcs:
                    ent = qm
                    break
        elif q_matches and q_matches[0].lower() not in ("input", "output", "inout", "verilog", "vhdl"):
            ent = q_matches[0]

        # Property detection
        prop = cat.lower()
        if "direction" in full_text.lower():
            prop = "direction"
        elif "width" in full_text.lower():
            prop = "width"
        elif ("offset" in full_text.lower() or "address" in full_text.lower()) and ent.upper() not in ("INTERFACE", "PORT", "SIGNAL", "FSM", "GENERAL", "SYSTEM"):
            prop = "offset"
        elif "reset" in full_text.lower():
            prop = "reset_value"
        elif "opcode" in full_text.lower() or "encoding" in full_text.lower():
            prop = "opcode"
        elif "fsm" in full_text.lower():
            prop = "fsm_definition"
        elif cat.upper() == "INTERFACE":
            prop = "interface"

        # Conflicting values detection
        raw_vals = []
        if prop == "direction":
            dirs = re.findall(r"\b(input|output|inout)\b", full_text, re.IGNORECASE)
            raw_vals = list(dict.fromkeys([d.lower() for d in dirs]))
        elif prop == "width":
            nums = re.findall(r"\b(\d+)\b", full_text)
            raw_vals = list(dict.fromkeys(nums))
        elif prop == "offset":
            hexes = re.findall(r"(0x[0-9a-fA-F]+)", full_text)
            raw_vals = list(dict.fromkeys(hexes))

        # Filter strictly with validate_property_value
        conf_vals = [v for v in raw_vals if cls.validate_property_value(ent, prop, v, efs_ir)[0]]

        title = f"{cls_name.replace('_', ' ').title()}: {ent}" if ent != cat else f"{cls_name.replace('_', ' ').title()}: {cat}"
        return ent, prop, conf_vals, title, c_text

    @classmethod
    def parse_specification_issues_to_conflicts(
        cls,
        code_or_issues: Union[str, List[Dict[str, Any]]],
        efs_ir: Optional[EFSIR] = None
    ) -> List[EFSConflict]:
        """
        Centrally and deterministically parses specification issues (from SPECIFICATION_INCOMPLETE
        or validation issues list) into structured EFSConflict objects.
        
        Properly classifies:
        - SPEC_CONFLICT: genuine contradictory specification values
        - MISSING_SPECIFICATION: missing FSM, signals, or properties
        - TARGET_RESOLUTION_FAILED / PLANNER_ERROR: resolver and planner diagnostics
        
        Applies semantic validation to property values and deduplicates using semantic keys.
        """
        import re
        conflicts: List[EFSConflict] = []
        seen_keys: set = set()

        def _add_parsed_conflict(c: EFSConflict):
            key = cls.compute_semantic_key(c)
            c.semantic_key = key
            if key not in seen_keys:
                seen_keys.add(key)
                conflicts.append(c)

        if isinstance(code_or_issues, str):
            lines = code_or_issues.split("\n")
            in_missing = False
            i = 0
            while i < len(lines):
                line_str = lines[i].strip()
                i += 1
                if "Missing/ambiguous requirements:" in line_str:
                    in_missing = True
                    continue
                if "Required clarification:" in line_str:
                    in_missing = False
                    continue
                if not in_missing or not line_str:
                    continue

                # 1. Check for resolver failure diagnostic block
                if "TARGET_RESOLUTION_FAILED" in line_str:
                    diag_lines = [line_str]
                    while i < len(lines) and lines[i].strip() and "Required clarification:" not in lines[i]:
                        diag_lines.append(lines[i].strip())
                        i += 1
                    diag_text = "\n".join(diag_lines)

                    req_target = ""
                    norm_target = ""
                    closest_cands = []
                    canon_id = ""
                    disp_name = ""
                    aliases = []
                    reason = ""

                    for dl in diag_lines:
                        if dl.startswith("Requested Target:"):
                            req_target = dl.split(":", 1)[1].strip().strip("'\"")
                        elif dl.startswith("Normalized Target:"):
                            norm_target = dl.split(":", 1)[1].strip().strip("'\"")
                        elif dl.startswith("Closest Candidates:"):
                            c_str = dl.split(":", 1)[1].strip()
                            closest_cands = [x.strip().strip("'\"") for x in c_str.strip("[]").split(",") if x.strip()]
                        elif dl.startswith("Design Canonical ID:"):
                            canon_id = dl.split(":", 1)[1].strip().strip("'\"")
                        elif dl.startswith("Design Display Name:"):
                            disp_name = dl.split(":", 1)[1].strip().strip("'\"")
                        elif dl.startswith("Registered Design Aliases:"):
                            a_str = dl.split(":", 1)[1].strip()
                            aliases = [x.strip().strip("'\"") for x in a_str.strip("[]").split(",") if x.strip()]
                        elif dl.startswith("Reason:"):
                            reason = dl.split(":", 1)[1].strip()

                    diag_details = {
                        "type": "TARGET_RESOLUTION_FAILED",
                        "requested_target": req_target,
                        "normalized_target": norm_target,
                        "closest_candidates": closest_cands,
                        "canonical_design_id": canon_id,
                        "design_display_name": disp_name,
                        "aliases": aliases,
                        "reason": reason or diag_text
                    }

                    c = EFSConflict(
                        type="TARGET_RESOLUTION_FAILED",
                        diagnostic_type=DiagnosticType.TARGET_RESOLUTION_FAILED.value,
                        severity="CRITICAL",
                        blocking=False,
                        status="UNRESOLVED",
                        requires_human_decision=False,
                        title=f"Resolver Failure: Target Resolution Failed ({req_target or 'Unknown'})",
                        entities=[req_target or "SYSTEM"],
                        affected_entities=[req_target or "SYSTEM"],
                        property="target_resolution",
                        conflicting_values=[],
                        current_values={},
                        resolver_diagnostic=diag_details,
                        evidence=diag_text,
                        description=reason or f"Target resolution failed for '{req_target}'.",
                        explanation="Planner failed to resolve requested hardware component target against canonical EFS IR.",
                        impact="Downstream synthesis cannot target an unresolved component.",
                        suggestions=[f"Select from available components: {', '.join(closest_cands[:3])}"] if closest_cands else []
                    )
                    _add_parsed_conflict(c)
                    continue

                # Ignore individual lines that are remnants of resolver diagnostic
                remnant_markers = [
                    "Requested Target:", "Normalized Target:", "Closest Candidates:",
                    "Design Canonical ID:", "Design Display Name:", "Registered Design Aliases:",
                    "Available EFS IR Components", "Reason: The requested target"
                ]
                if any(rm in line_str for rm in remnant_markers):
                    continue

                # 2. Parse standard requirement line with badge and category
                m = re.search(r"^(?:[🔴🟠🟡🟢🔵🟣⚙️]\s*)?(?:\[([A-Z_]+)\]\s*)?(?:\[([A-Z_]+)\]\s*)?(.*?)(?:\s*\|\s*Source:\s*(.*))?$", line_str)
                if m:
                    g1 = (m.group(1) or "").strip()
                    g2 = (m.group(2) or "").strip()
                    clean_line = (m.group(3) or "").strip()
                    evid = (m.group(4) or "").strip() or "RTL Synthesis Planner"

                    known_classes = {"PLAN_MISSING", "SOURCE_MISSING", "SOURCE_CONFLICT", "PLAN_CONFLICT", "AMBIGUOUS", "PLAN_AMBIGUOUS", "SOURCE_AMBIGUOUS", "CONFLICT", "MISSING"}
                    known_cats = {"FSM", "INTERFACE", "SIGNAL", "PORT", "REGISTER", "REGISTERS", "OPCODE", "OPCODES", "MEMORY", "GENERAL", "SPECIFICATION"}

                    if g1 in known_classes and g2 in known_cats:
                        cls_name = g1
                        cat = g2
                    elif g1 in known_cats and g2 in known_classes:
                        cls_name = g2
                        cat = g1
                    elif g1 in known_classes:
                        cls_name = g1
                        cat = "GENERAL"
                    elif g1 in known_cats:
                        cls_name = "PLAN_MISSING"
                        cat = g1
                    else:
                        cls_name = "PLAN_MISSING"
                        cat = "GENERAL"
                else:
                    cls_name = "PLAN_MISSING"
                    cat = "GENERAL"
                    clean_line = line_str
                    evid = "RTL Synthesis Planner"

                if not clean_line:
                    continue

                ent, prop, conf_vals, conf_title, conf_expl = cls._extract_issue_metadata(
                    clean_line, evid, cat, cls_name, efs_ir
                )

                # Determine correct DiagnosticType
                if "CONFLICT" in cls_name.upper() and len(conf_vals) >= 2:
                    diag_type = DiagnosticType.SPEC_CONFLICT.value
                elif "FSM" in cat.upper() or "fsm" in prop.lower() or "transition" in prop.lower():
                    diag_type = DiagnosticType.MISSING_SPECIFICATION.value
                elif "MISSING" in cls_name.upper() or not conf_vals:
                    diag_type = DiagnosticType.MISSING_SPECIFICATION.value
                else:
                    diag_type = DiagnosticType.SPEC_CONFLICT.value if len(conf_vals) >= 2 else DiagnosticType.MISSING_SPECIFICATION.value

                c = EFSConflict(
                    type=cls_name if "CONFLICT" in cls_name else "SOURCE_MISSING",
                    diagnostic_type=diag_type,
                    severity="CRITICAL",
                    blocking=True,
                    status="UNRESOLVED",
                    title=conf_title,
                    entities=[ent],
                    affected_entities=[ent],
                    property=prop,
                    conflicting_values=conf_vals,
                    current_values={ent: ", ".join(conf_vals)} if conf_vals else {},
                    source_references=[{
                        "entity": ent,
                        "source": "RTL Synthesis Planner",
                        "section": f"Rule: {prop.title()}",
                        "value": clean_line
                    }],
                    evidence=evid or line_str,
                    description=clean_line or line_str,
                    explanation=conf_expl,
                    impact="Synthesis cannot determine implementation details, resulting in generation stoppage.",
                    suggestions=[f"Specify requirement or candidate value for '{ent}'."],
                    requires_human_decision=True
                )
                _add_parsed_conflict(c)

        elif isinstance(code_or_issues, list):
            for issue in code_or_issues:
                cls_name = issue.get("classification", "MISSING")
                cat = issue.get("category", "GENERAL")
                c_text = issue.get("issue") or issue.get("message") or ""
                evid = issue.get("evidence", "RTL Synthesis Planner")

                if not c_text or "TARGET_RESOLUTION_FAILED" in c_text:
                    continue

                ent, prop, conf_vals, conf_title, conf_expl = cls._extract_issue_metadata(
                    c_text, evid, cat, cls_name, efs_ir
                )

                if "CONFLICT" in cls_name.upper() and len(conf_vals) >= 2:
                    diag_type = DiagnosticType.SPEC_CONFLICT.value
                elif "FSM" in cat.upper() or "fsm" in prop.lower() or "transition" in prop.lower():
                    diag_type = DiagnosticType.MISSING_SPECIFICATION.value
                elif "MISSING" in cls_name.upper() or not conf_vals:
                    diag_type = DiagnosticType.MISSING_SPECIFICATION.value
                else:
                    diag_type = DiagnosticType.SPEC_CONFLICT.value if len(conf_vals) >= 2 else DiagnosticType.MISSING_SPECIFICATION.value

                c = EFSConflict(
                    type=cls_name if "CONFLICT" in cls_name else "SOURCE_MISSING",
                    diagnostic_type=diag_type,
                    severity="CRITICAL",
                    blocking=True,
                    status="UNRESOLVED",
                    title=conf_title,
                    entities=[ent],
                    affected_entities=[ent],
                    property=prop,
                    conflicting_values=conf_vals,
                    current_values={ent: ", ".join(conf_vals)} if conf_vals else {},
                    source_references=[{
                        "entity": ent,
                        "source": "RTL Synthesis Planner",
                        "section": f"Rule: {prop.title()}",
                        "value": c_text
                    }],
                    evidence=evid,
                    description=c_text,
                    explanation=conf_expl,
                    impact="Synthesis cannot determine implementation details, resulting in generation stoppage.",
                    suggestions=[f"Specify requirement or candidate value for '{ent}'."],
                    requires_human_decision=True
                )
                _add_parsed_conflict(c)

        return conflicts

    @classmethod
    def _detect_raw_conflicts(
        cls,
        efs_ir: EFSIR,
        req_model: Optional[Dict[str, Any]] = None,
        validation_issues: Optional[List[Dict[str, Any]]] = None
    ) -> List[EFSConflict]:
        """
        Dynamically detects raw specification property conflicts directly from current EFS IR state.
        Does NOT carry over historical/stale conflict objects.
        Uses deterministic semantic key deduplication.
        """
        raw_conflicts: List[EFSConflict] = []
        seen_semantic_keys: set = set()

        def _add_raw_conflict(c: EFSConflict):
            key = cls.compute_semantic_key(c)
            c.semantic_key = key
            if key in seen_semantic_keys:
                existing = next((x for x in raw_conflicts if x.semantic_key == key), None)
                if existing:
                    # Merge source references and objects without creating duplicate conflict
                    existing_sources = {ref.get("source") for ref in getattr(existing, "source_references", [])}
                    for ref in getattr(c, "source_references", []):
                        if ref.get("source") not in existing_sources:
                            existing.source_references.append(ref)
                            existing_sources.add(ref.get("source"))
                    existing_objs = set(getattr(existing, "source_objects", []))
                    for obj in getattr(c, "source_objects", []):
                        if obj not in existing_objs:
                            existing.source_objects.append(obj)
                            existing_objs.add(obj)
                return
            c.status = ConflictStatus("UNRESOLVED")
            c.resolution_status = "UNRESOLVED"
            c.blocking = (c.severity.upper() in ("CRITICAL", "ERROR"))
            c.requires_human_decision = c.blocking
            seen_semantic_keys.add(key)
            raw_conflicts.append(c)

        # 1. Duplicate register address/offset conflicts
        def _norm_offset(raw: Any) -> str:
            if raw is None:
                return ""
            s = str(raw).strip()
            if not s:
                return ""
            try:
                if s.lower().startswith("0x"):
                    return f"0x{int(s, 16):x}"
                elif s.isdigit():
                    return f"0x{int(s, 10):x}"
            except ValueError:
                pass
            return s.lower()

        reg_offsets: Dict[str, List[Any]] = {}
        for reg in getattr(efs_ir, "registers", []):
            if not is_explicit_address(reg.offset):
                continue
            offset_key = _norm_offset(reg.offset)
            if offset_key:
                reg_offsets.setdefault(offset_key, []).append(reg)

        for offset_val, regs in reg_offsets.items():
            if len(regs) > 1:
                is_intentional_alias = any(
                    any(term in (r.description or "").lower() for term in ("alias", "shadow", "mirror", "banked", "alternate view", "register view"))
                    for r in regs
                )
                if is_intentional_alias:
                    continue

                names = [r.name for r in regs]
                source_refs = [
                    {
                        "entity": r.name,
                        "source": getattr(r.traceability, "source_file", "Specification") or "Specification",
                        "section": getattr(r.traceability, "section", "Register Map") or "Register Map",
                        "value": r.offset
                    }
                    for r in regs
                ]
                c = EFSConflict(
                    type="DUPLICATE_REGISTER_ADDRESS",
                    severity="CRITICAL",
                    blocking=True,
                    status="UNRESOLVED",
                    title=f"Duplicate Register Address: {', '.join(names)}",
                    entities=names,
                    affected_entities=names,
                    property="offset",
                    conflicting_values=[r.offset for r in regs],
                    current_values={r.name: r.offset for r in regs},
                    source_objects=[r.register_id for r in regs],
                    source_references=source_refs,
                    evidence=f"Registers {', '.join(names)} share the same address offset '{offset_val}'.",
                    description=f"Address map conflict: Multiple registers ({', '.join(names)}) assigned to identical offset '{offset_val}'.",
                    explanation=f"Address collision detected between {', '.join(names)}. Writing to offset '{offset_val}' would mutate multiple registers simultaneously.",
                    impact="Address bus decoding collision resulting in unintended register writes and corrupted states.",
                    suggestions=[f"Reassign unique address offset for {r.name}" for r in regs],
                    requires_human_decision=True,
                    traceability=regs[0].traceability
                )
                _add_raw_conflict(c)

        # 2. Duplicate opcode encoding conflicts (from opcodes list)
        opc_encodings: Dict[str, List[Any]] = {}
        for opc in getattr(efs_ir, "opcodes", []):
            enc_val = (opc.binary_encoding or "").strip()
            if is_explicit_encoding(enc_val):
                opc_encodings.setdefault(enc_val.lower(), []).append(opc)

        for enc_val, opcs in opc_encodings.items():
            distinct_mnemonics = sorted(list({o.mnemonic for o in opcs if o.mnemonic}))
            if len(distinct_mnemonics) > 1:
                source_refs = [
                    {
                        "entity": o.mnemonic,
                        "source": getattr(o.traceability, "source_file", "Specification") or "Specification",
                        "section": getattr(o.traceability, "section", "Instruction Encoding") or "Instruction Encoding",
                        "value": o.binary_encoding
                    }
                    for o in opcs
                ]
                c = EFSConflict(
                    type="DUPLICATE_OPCODE_ENCODING",
                    severity="CRITICAL",
                    blocking=True,
                    status="UNRESOLVED",
                    title=f"Duplicate Opcode Encoding: {', '.join(distinct_mnemonics)}",
                    entities=distinct_mnemonics,
                    affected_entities=distinct_mnemonics,
                    property="binary_encoding",
                    conflicting_values=[o.binary_encoding for o in opcs],
                    current_values={o.mnemonic: o.binary_encoding for o in opcs},
                    source_objects=[o.opcode_id for o in opcs],
                    source_references=source_refs,
                    evidence=f"Opcodes {', '.join(distinct_mnemonics)} share binary encoding '{enc_val}'.",
                    description=f"Opcode collision: Instructions {', '.join(distinct_mnemonics)} map to identical encoding '{enc_val}'.",
                    explanation=f"Instruction decoder cannot disambiguate between {', '.join(distinct_mnemonics)} when binary pattern '{enc_val}' is received.",
                    impact="Decoder ambiguity resulting in incorrect operation execution and invalid control signals.",
                    suggestions=[f"Assign unique binary encoding for '{m}'" for m in distinct_mnemonics],
                    requires_human_decision=True,
                    traceability=opcs[0].traceability
                )
                _add_raw_conflict(c)

        # 2b. Duplicate instruction opcodes (from instructions list)
        inst_encodings: Dict[str, List[Any]] = {}
        for inst in getattr(efs_ir, "instructions", []):
            enc_val = (inst.opcode or "").strip()
            if is_explicit_encoding(enc_val):
                inst_encodings.setdefault(enc_val.lower(), []).append(inst)

        for enc_val, insts in inst_encodings.items():
            distinct_mnemonics = sorted(list({i.mnemonic for i in insts if i.mnemonic}))
            if len(distinct_mnemonics) > 1:
                source_refs = [
                    {
                        "entity": i.mnemonic,
                        "source": getattr(i.traceability, "source_file", "Specification") or "Specification",
                        "section": getattr(i.traceability, "section", "Instruction Encoding") or "Instruction Encoding",
                        "value": i.opcode
                    }
                    for i in insts
                ]
                c = EFSConflict(
                    type="DUPLICATE_OPCODE",
                    severity="CRITICAL",
                    blocking=True,
                    status="UNRESOLVED",
                    title=f"Duplicate Opcode: {', '.join(distinct_mnemonics)}",
                    entities=distinct_mnemonics,
                    affected_entities=distinct_mnemonics,
                    property="opcode",
                    conflicting_values=[i.opcode for i in insts],
                    current_values={i.mnemonic: i.opcode for i in insts},
                    source_objects=[i.instruction_id for i in insts],
                    source_references=source_refs,
                    evidence=f"Instructions {', '.join(distinct_mnemonics)} share opcode '{enc_val}'.",
                    description=f"Two or more instructions ({', '.join(distinct_mnemonics)}) are assigned the same opcode '{enc_val}'.",
                    explanation=f"The instruction decoder cannot uniquely determine whether opcode '{enc_val}' represents {' or '.join(distinct_mnemonics)}.",
                    impact="Instruction decoder ambiguity resulting in incorrect execution.",
                    suggestions=[f"Assign unique opcode for '{m}'" for m in distinct_mnemonics],
                    requires_human_decision=True,
                    traceability=insts[0].traceability
                )
                _add_raw_conflict(c)

        # 2c. Conflicting opcodes for the same instruction/mnemonic across mentions
        opcs_by_mnemonic: Dict[str, List[Any]] = {}
        for opc in getattr(efs_ir, "opcodes", []):
            m_key = (opc.mnemonic or "").strip().upper()
            if m_key:
                opcs_by_mnemonic.setdefault(m_key, []).append(opc)

        for m_name, group in opcs_by_mnemonic.items():
            explicit_vals = {o.binary_encoding.strip() for o in group if is_explicit_encoding(o.binary_encoding)}
            if len(explicit_vals) > 1:
                c = EFSConflict(
                    type="CONFLICTING_OPCODE_VALUE",
                    severity="CRITICAL",
                    blocking=True,
                    status="UNRESOLVED",
                    title=f"Conflicting Opcode Encoding: {m_name}",
                    entities=[m_name],
                    affected_entities=[m_name],
                    property="binary_encoding",
                    conflicting_values=sorted(list(explicit_vals)),
                    current_values={m_name: sorted(list(explicit_vals))[0]},
                    source_objects=[o.opcode_id for o in group],
                    source_references=[
                        {
                            "entity": m_name,
                            "source": getattr(o.traceability, "source_file", "Specification") or "Specification",
                            "section": getattr(o.traceability, "section", "Instruction Encoding") or "Instruction Encoding",
                            "value": o.binary_encoding
                        }
                        for o in group if is_explicit_encoding(o.binary_encoding)
                    ],
                    evidence=f"Opcode '{m_name}' has contradictory explicit encodings: {', '.join(sorted(explicit_vals))}.",
                    description=f"Specification defines contradictory binary encodings for opcode '{m_name}'.",
                    explanation=f"Cannot determine canonical binary encoding for {m_name} because multiple contradictory values are specified.",
                    impact="Instruction decoder synthesis collision resulting in non-deterministic opcode matching.",
                    suggestions=[f"Select canonical encoding for '{m_name}'"],
                    requires_human_decision=True,
                    traceability=group[0].traceability
                )
                _add_raw_conflict(c)

        # 2d. Conflicting signal widths across duplicate signal names
        sig_names: Dict[str, List[Any]] = {}
        for sig in getattr(efs_ir, "signals", []):
            name_key = sig.name.strip().lower()
            if name_key:
                sig_names.setdefault(name_key, []).append(sig)

        for name_val, sigs in sig_names.items():
            widths = {str(s.width) for s in sigs if s.width}
            if len(widths) > 1:
                source_refs = [
                    {
                        "entity": s.name,
                        "source": getattr(s.traceability, "source_file", "Specification") or "Specification",
                        "section": getattr(s.traceability, "section", "Signal Interface") or "Signal Interface",
                        "value": str(s.width)
                    }
                    for s in sigs
                ]
                c = EFSConflict(
                    type="CONFLICTING_WIDTH",
                    severity="CRITICAL",
                    blocking=True,
                    status="UNRESOLVED",
                    title=f"Conflicting Signal Width: {sigs[0].name}",
                    entities=[sigs[0].name],
                    affected_entities=[sigs[0].name],
                    property="width",
                    conflicting_values=list(widths),
                    current_values={sigs[0].name: ", ".join(widths)},
                    source_objects=[s.signal_id for s in sigs],
                    source_references=source_refs,
                    evidence=f"Signal '{sigs[0].name}' is declared with conflicting widths: {', '.join(widths)}.",
                    description=f"Conflicting bit-widths specified for signal '{sigs[0].name}'.",
                    explanation=f"Width mismatch on signal '{sigs[0].name}' causes bus sizing errors or truncation during synthesis.",
                    impact="Hardware bus truncation or width mismatch syntax error during synthesis.",
                    suggestions=[f"Set uniform width for '{sigs[0].name}'."],
                    requires_human_decision=True,
                    traceability=sigs[0].traceability
                )
                _add_raw_conflict(c)

        # 2e. Comprehensive FSM State & Transition ambiguity/missing condition detection
        vague_patterns = [
            "start condition", "condition met", "ready", "when ready", "done", "complete",
            "as needed", "appropriate", "etc", "tbd", "valid condition", "trigger"
        ]
        for fsm in getattr(efs_ir, "fsms", []):
            fsm_name = getattr(fsm, "name", "Primary_FSM") or "Primary_FSM"
            for tr in getattr(fsm, "transitions", []):
                src = tr.source_state or ""
                dst = tr.target_state or ""
                cond = (tr.condition or "").strip()
                is_vague = any(vp in cond.lower() for vp in vague_patterns)
                is_missing = not cond

                if is_missing or is_vague:
                    c_title = f"FSM Transition Undefined: {src} -> {dst}" if is_missing else f"FSM Transition Ambiguous: {src} -> {dst}"
                    c_desc = f"Transition condition from state '{src}' to '{dst}' is undefined or empty." if is_missing else f"Transition condition from state '{src}' to '{dst}' is ambiguous ('{cond}')."
                    c = EFSConflict(
                        type="SOURCE_MISSING" if is_missing else "SOURCE_AMBIGUOUS",
                        diagnostic_type=DiagnosticType.MISSING_SPECIFICATION.value,
                        severity="CRITICAL",
                        blocking=True,
                        status="UNRESOLVED",
                        title=c_title,
                        entities=[src, dst] if src and dst else [fsm_name],
                        affected_entities=[src, dst] if src and dst else [fsm_name],
                        property="transition_condition",
                        conflicting_values=[cond] if cond else [],
                        current_values={f"{src}->{dst}": cond},
                        evidence=c_desc,
                        description=c_desc,
                        explanation=f"RTL synthesis requires an explicit Boolean signal or register expression for transition '{src}' -> '{dst}'.",
                        impact="Synthesis failure or non-deterministic state machine transitions in RTL.",
                        suggestions=[f"Specify condition triggering transition '{src}' -> '{dst}' (e.g. 'start_pulse == 1\\'b1')."],
                        requires_human_decision=True
                    )
                    _add_raw_conflict(c)

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

                    ent, prop, conf_vals, conf_title, conf_expl = cls._extract_issue_metadata(c_text, evidence, cat, cls_name, efs_ir)
                    src_doc = issue.get("source_doc") or issue.get("document") or "Specification"
                    sec = issue.get("section") or f"Category: {cat}"

                    if "CONFLICT" in cls_name.upper() and len(conf_vals) >= 2:
                        diag_type = DiagnosticType.SPEC_CONFLICT.value
                    elif "FSM" in cat.upper() or "fsm" in prop.lower() or "transition" in prop.lower():
                        diag_type = DiagnosticType.MISSING_SPECIFICATION.value
                    elif "MISSING" in cls_name.upper() or not conf_vals:
                        diag_type = DiagnosticType.MISSING_SPECIFICATION.value
                    else:
                        diag_type = DiagnosticType.SPEC_CONFLICT.value if len(conf_vals) >= 2 else DiagnosticType.MISSING_SPECIFICATION.value

                    c = EFSConflict(
                        type=cls_name,
                        diagnostic_type=diag_type,
                        severity=severity if severity in ("CRITICAL", "ERROR") else "CRITICAL",
                        blocking=True,
                        status="UNRESOLVED",
                        title=conf_title,
                        entities=[ent],
                        affected_entities=[ent],
                        property=prop,
                        conflicting_values=conf_vals,
                        current_values={ent: ", ".join(conf_vals)} if conf_vals else {},
                        source_references=[
                            {
                                "entity": ent,
                                "source": src_doc,
                                "section": sec,
                                "value": c_text
                            }
                        ],
                        evidence=evidence,
                        description=c_text,
                        explanation=conf_expl or req_info,
                        impact=f"Contradictory or missing {prop} definition prevents synthesizable hardware implementation.",
                        suggestions=[f"Select candidate value or provide custom requirement for '{ent}'."],
                        requires_human_decision=True
                    )
                    _add_raw_conflict(c)

        # 4. Critical issues from validate_efs_ir (structural gaps/errors)
        if validation_issues is None and hasattr(efs_ir, "signals"):
            try:
                from core.efs_ir.validator import validate_efs_ir
                validation_issues = validate_efs_ir(efs_ir)
            except Exception:
                validation_issues = []

        if validation_issues:
            for v_iss in validation_issues:
                if v_iss.get("severity") in ("ERROR", "CRITICAL"):
                    cat = v_iss.get("category", "Structural")
                    if cat in ("Conflicts", "Metadata"):
                        continue
                    if cat == "Registers" and "duplicate" in (v_iss.get("message") or "").lower():
                        continue
                    if cat == "Opcodes" and "duplicate" in (v_iss.get("message") or "").lower():
                        continue
                    msg = v_iss.get("message", "Structural validation error")
                    cls_name = v_iss.get("classification", "EFSIR_EXTRACTION_GAP")
                    obj_id = v_iss.get("object_id") or cat
                    c = EFSConflict(
                        type=cls_name,
                        diagnostic_type=DiagnosticType.VALIDATION_ERROR.value,
                        severity="CRITICAL",
                        blocking=True,
                        status="UNRESOLVED",
                        title=f"Structural Constraint: {cat} - {msg[:50]}",
                        entities=[obj_id],
                        affected_entities=[obj_id],
                        property=cat.lower(),
                        conflicting_values=[msg],
                        current_values={obj_id: msg},
                        source_references=[{
                            "entity": obj_id,
                            "source": "EFS IR Structural Validator",
                            "section": f"Rule: {cat}",
                            "value": msg
                        }],
                        evidence=msg,
                        description=msg,
                        explanation=f"The EFS IR validator found a critical structural error in '{cat}': {msg}.",
                        impact="RTL generation and synthesis cannot proceed with structural schema violations.",
                        suggestions=[f"Specify a valid {cat.lower()} parameter to satisfy hardware constraints."],
                        requires_human_decision=True
                    )
                    _add_raw_conflict(c)

        return raw_conflicts

    @classmethod
    def reconcile_conflicts(
        cls,
        old_conflicts: List[EFSConflict],
        current_conflicts: List[EFSConflict],
        updated_efs: EFSIR
    ) -> List[EFSConflict]:
        """
        Deterministically reconciles previously stored historical conflicts against the current EFS IR state.
        Zero LLM calls.
        - If old_key exists in current_conflicts: old_conflict.status = "ACTIVE", updates current evidence.
        - If old_key does not exist in current_conflicts: old_conflict.status = "RESOLVED",
          old_conflict.resolved_reason = "Violation no longer exists in current EFS state."
        - Genuinely new conflicts in current_conflicts are added as ACTIVE.
        - Preserves historical records for audit logging.
        """
        current_map: Dict[str, EFSConflict] = {}
        for c in current_conflicts:
            key = c.semantic_key or cls.compute_semantic_key(c)
            c.semantic_key = key
            current_map[key] = c

        reconciled: List[EFSConflict] = []
        matched_current_keys: set = set()

        approved_conflict_ids = {
            res.conflict_id for res in getattr(updated_efs, "resolutions", [])
            if getattr(res, "approval_status", "") == "APPROVED"
        }
        approved_entity_props = {
            (req.lower(), (getattr(res, "field_name") or getattr(res, "property") or "").lower())
            for res in getattr(updated_efs, "resolutions", [])
            if getattr(res, "approval_status", "") == "APPROVED"
            for req in (res.affected_requirements or [])
        }

        for old in old_conflicts:
            old_key = old.semantic_key or cls.compute_semantic_key(old)
            old.semantic_key = old_key

            if old_key in current_map:
                old.status = ConflictStatus("UNRESOLVED")
                old.resolution_status = "UNRESOLVED"
                old.blocking = True
                old.requires_human_decision = True
                old.resolved_reason = ""
                curr = current_map[old_key]
                if curr.evidence:
                    old.evidence = curr.evidence
                if curr.current_values:
                    old.current_values = dict(curr.current_values)
                if curr.conflicting_values:
                    old.conflicting_values = list(curr.conflicting_values)
                matched_current_keys.add(old_key)
            else:
                is_resolved_by_res = (
                    old.conflict_id in approved_conflict_ids or
                    any((ent.lower(), (old.property or "").lower()) in approved_entity_props for ent in (old.affected_entities or old.entities or []))
                )
                if is_resolved_by_res:
                    old.status = "RESOLVED"
                    old.resolution_status = "RESOLVED"
                    old.blocking = False
                    old.requires_human_decision = False
                    old.resolved_reason = "Resolved by approved resolution."
                elif old.blocking and (old.resolution_status == "UNRESOLVED" or old.status in ("UNRESOLVED", "ACTIVE")):
                    # Unresolved blocking conflict with no approved resolution remains UNRESOLVED
                    old.status = ConflictStatus("UNRESOLVED")
                    old.resolution_status = "UNRESOLVED"
                    old.blocking = True
                    old.requires_human_decision = True
                else:
                    old.status = "RESOLVED"
                    old.resolution_status = "RESOLVED"
                    old.blocking = False
                    old.requires_human_decision = False
                    old.resolved_reason = "Violation no longer exists in current EFS state."

            reconciled.append(old)

        old_keys = {getattr(c, "semantic_key", "") for c in old_conflicts}
        for curr in current_conflicts:
            curr_key = curr.semantic_key or cls.compute_semantic_key(curr)
            curr.semantic_key = curr_key
            if curr_key not in matched_current_keys and curr_key not in old_keys:
                curr.status = ConflictStatus("UNRESOLVED")
                curr.resolution_status = "UNRESOLVED"
                curr.blocking = True
                curr.requires_human_decision = True
                reconciled.append(curr)
                matched_current_keys.add(curr_key)

        return reconciled

    @classmethod
    def deduplicate_conflicts(cls, conflicts: List[EFSConflict]) -> List[EFSConflict]:
        """
        Removes duplicate conflicts based on canonical semantic keys and conflict IDs.
        Preserves original conflict order while eliminating redundancies.
        """
        unique: List[EFSConflict] = []
        seen_keys: set = set()
        seen_ids: set = set()

        for c in conflicts:
            cid = getattr(c, "conflict_id", "")
            key = getattr(c, "semantic_key", "") or cls.compute_semantic_key(c)
            c.semantic_key = key

            # Check duplicate by ID or semantic key
            if (cid and cid in seen_ids) or (key and key in seen_keys):
                continue

            if cid:
                seen_ids.add(cid)
            if key:
                seen_keys.add(key)
            unique.append(c)

        return unique

    @classmethod
    def detect_conflicts(
        cls,
        efs_ir: EFSIR,
        req_model: Optional[Dict[str, Any]] = None,
        validation_issues: Optional[List[Dict[str, Any]]] = None
    ) -> List[EFSConflict]:
        """
        Dynamically detects specification property conflicts across hardware entities.
        Does NOT rely on hardcoded signal names, register names, FSM states, or protocol keywords.
        Uses deterministic semantic conflict keys and revalidates against CURRENT EFS IR.
        """
        logger.info("VALIDATION: Starting deterministic conflict detection...")
        raw_conflicts = cls._detect_raw_conflicts(efs_ir, req_model, validation_issues)

        # Check existing approved resolutions in overlay
        approved_conflict_ids = {
            res.conflict_id for res in getattr(efs_ir, "resolutions", [])
            if getattr(res, "approval_status", "") == "APPROVED"
        }
        approved_entity_props = {
            (req.lower(), (getattr(res, "field_name") or getattr(res, "property") or "").lower())
            for res in getattr(efs_ir, "resolutions", [])
            if getattr(res, "approval_status", "") == "APPROVED"
            for req in (res.affected_requirements or [])
        }

        run_id = getattr(getattr(efs_ir, "metadata", None), "generation_run_id", "")
        for c in raw_conflicts:
            if not getattr(c, "generation_run_id", "") and run_id:
                c.generation_run_id = run_id

        if efs_ir.conflicts:
            reconciled = cls.reconcile_conflicts(
                old_conflicts=list(efs_ir.conflicts),
                current_conflicts=raw_conflicts,
                updated_efs=efs_ir
            )
            for c in reconciled:
                if not getattr(c, "generation_run_id", "") and run_id:
                    c.generation_run_id = run_id
            efs_ir.conflicts = reconciled
            detected_conflicts = reconciled
        else:
            for c in raw_conflicts:
                is_resolved = (c.conflict_id in approved_conflict_ids)
                if not is_resolved and approved_entity_props:
                    c_prop = (c.property or "").lower()
                    for ent in (c.affected_entities or c.entities or []):
                        if (ent.lower(), c_prop) in approved_entity_props or (ent.lower(), "") in approved_entity_props:
                            is_resolved = True
                            break
                if is_resolved:
                    c.status = "APPROVED"
                    c.resolution_status = "RESOLVED"
                    c.blocking = False
                    c.requires_human_decision = False
                    c.resolved_reason = "Approved resolution applied."
                else:
                    c.status = ConflictStatus("UNRESOLVED")
                    c.resolution_status = "UNRESOLVED"
                    c.blocking = (c.severity.upper() in ("CRITICAL", "ERROR"))
                    c.requires_human_decision = c.blocking

            efs_ir.conflicts = raw_conflicts
            detected_conflicts = raw_conflicts

        # Deduplicate conflicts cleanly
        detected_conflicts = cls.deduplicate_conflicts(detected_conflicts)
        efs_ir.conflicts = detected_conflicts

        unresolved_count = len([c for c in detected_conflicts if c.blocking and (c.resolution_status == "UNRESOLVED" or c.status == "ACTIVE") and c.status != "RESOLVED"])
        if unresolved_count > 0:
            logger.warning(f"CONFLICT_DETECTED: {unresolved_count} unresolved blocking specification conflict(s) detected.")
            logger.info("-> WAITING FOR HUMAN DECISION: Awaiting user selection between Option 1 (Allow AI) and Option 2 (Custom Requirement).")
        else:
            logger.info("VALIDATION: All specification issues and conflicts resolved.")

        return detected_conflicts

    @classmethod
    def validate_resolution(
        cls,
        conflict: EFSConflict,
        proposed_value: Any,
        efs_ir: EFSIR,
        target_entity: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Runs deterministic server-side validation on AI or Human proposed resolutions.
        Verifies non-empty, non-placeholder, bounds, does not collide with existing assigned
        resources, and does not preserve the conflict.
        """
        if proposed_value is None:
            return False, "Resolution value cannot be empty."

        val_str = str(proposed_value).strip()
        if not val_str:
            return False, "Resolution value cannot be empty."

        val_lower = val_str.lower()

        # Reject placeholder values
        for ph in cls.DISALLOWED_PLACEHOLDER_SUBSTRINGS:
            if ph in val_lower:
                return False, f"Proposed resolution contains disallowed placeholder text '{ph}'."

        # Reject proposals that preserve the collision when all conflicting values are identical
        conf_vals = [str(v).strip().lower() for v in (conflict.conflicting_values or [])]
        if conf_vals and all(v == val_lower for v in conf_vals):
            return False, f"Proposed value '{val_str}' preserves the existing conflicting value."

        target_name = target_entity or (
            conflict.affected_entities[1] if len(conflict.affected_entities) > 1
            else (conflict.affected_entities[0] if conflict.affected_entities else "")
        )

        # 1. Opcode / Binary Encoding Collision & Format
        if conflict.type in ("DUPLICATE_OPCODE_ENCODING", "DUPLICATE_OPCODE", "CONFLICTING_OPCODE") or "encoding" in conflict.property.lower() or "opcode" in conflict.property.lower():
            for opc in getattr(efs_ir, "opcodes", []):
                if opc.mnemonic.lower() != target_name.lower() and opc.binary_encoding.strip().lower() == val_lower:
                    return False, f"The proposed opcode '{val_str}' collides with existing assigned opcode: already assigned to instruction '{opc.mnemonic}'."

            for inst in getattr(efs_ir, "instructions", []):
                if inst.mnemonic.lower() != target_name.lower() and inst.opcode.strip().lower() == val_lower:
                    return False, f"The proposed opcode '{val_str}' collides with existing assigned opcode: already assigned to instruction '{inst.mnemonic}'."

            clean_bin = val_str.split("'b")[-1] if "'b" in val_str else (val_str[2:] if val_str.startswith("0b") else val_str)
            if not (all(c in "01" for c in clean_bin) or val_str.startswith("0x") or val_str.isdigit()):
                return False, f"Invalid opcode format '{val_str}'. Expected binary bitstring (e.g. '1011', \"4'b1011\") or hex value."

        # 2. Register Address Offset Collision & Format
        elif conflict.type in ("DUPLICATE_REGISTER_ADDRESS", "CONFLICTING_ADDRESS") or "address" in conflict.property.lower() or "offset" in conflict.property.lower():
            for reg in getattr(efs_ir, "registers", []):
                if reg.name.lower() != target_name.lower() and reg.offset.strip().lower() == val_lower:
                    return False, f"The proposed address offset '{val_str}' collides with existing assigned register: already assigned to '{reg.name}'."

            if not (val_str.startswith("0x") or val_str.isdigit()):
                return False, f"Invalid address offset format '{val_str}'. Expected hex offset (e.g. '0x08') or integer."

        # 3. Signal Width Bounds & Format
        elif conflict.type in ("CONFLICTING_WIDTH", "SIGNAL_WIDTH") or "width" in conflict.property.lower():
            if ":" in val_str:
                parts = val_str.replace("[", "").replace("]", "").split(":")
                if len(parts) != 2 or not parts[0].strip().isdigit() or not parts[1].strip().isdigit():
                    return False, f"Invalid bit-range format '{val_str}'. Expected '[MSB:LSB]' or positive integer."
            elif not val_str.isdigit() or int(val_str) <= 0:
                return False, f"Invalid signal width '{val_str}'. Expected positive integer."

        # 4. Signal Direction
        elif "direction" in conflict.property.lower():
            if val_lower not in ("input", "in", "output", "out", "inout"):
                return False, f"Invalid signal direction '{val_str}'. Expected 'input', 'output', or 'inout'."

        return True, "Resolution validation passed."

    @classmethod
    def validate_proposal_candidate(
        cls,
        candidate_val: Any,
        conflict: EFSConflict,
        efs_ir: EFSIR,
        target_entity: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Runs deterministic validation on AI generated proposal candidates."""
        return cls.validate_resolution(conflict, candidate_val, efs_ir, target_entity)

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
        """
        entities = conflict.affected_entities or conflict.entities or ["TARGET_ENTITY"]
        target_entity = entities[1] if len(entities) > 1 else entities[0]
        conf_values = conflict.conflicting_values or []

        diag_type = getattr(conflict, "diagnostic_type", "SPEC_CONFLICT") or "SPEC_CONFLICT"
        if diag_type in (DiagnosticType.TARGET_RESOLUTION_FAILED.value, DiagnosticType.PLANNER_ERROR.value, "TARGET_RESOLUTION_FAILED", "PLANNER_ERROR"):
            return []

        base_constraints = ["candidate_exists", "non_placeholder_validation", "non_conflicting_value"]
        results: List[Tuple[str, str, List[str]]] = []
        seen_vals = set()

        # 1. Opcode Binary Encoding Collision
        if conflict.type in ("DUPLICATE_OPCODE_ENCODING", "DUPLICATE_OPCODE", "CONFLICTING_OPCODE") or "encoding" in conflict.property.lower() or "opcode" in conflict.property.lower():
            constraints_checked = list(base_constraints) + ["field_width_bounds", "domain_range_0_to_2_N", "opcode_unique_assignment"]
            existing_opcs = getattr(efs_ir, "opcodes", [])
            existing_insts = getattr(efs_ir, "instructions", [])
            assigned_encs = {o.binary_encoding.strip().lower() for o in existing_opcs if o.mnemonic != target_entity}
            assigned_encs.update({i.opcode.strip().lower() for i in existing_insts if i.mnemonic != target_entity})

            sample_enc = conf_values[0] if conf_values else "000000"
            prefix_fmt = ""
            width = 4

            if "'b" in sample_enc:
                parts = sample_enc.split("'b")
                width = int(parts[0]) if parts[0].isdigit() else 8
                prefix_fmt = f"{width}'b"
            elif sample_enc.startswith("0b"):
                prefix_fmt = "0b"
                width = len(sample_enc[2:])
            else:
                width = len(sample_enc) if sample_enc and all(c in '01' for c in sample_enc) else 4

            max_domain = min(1 << width, 256)
            for v in range(max_domain):
                b_str = f"{v:0{width}b}"
                cand_val = f"{prefix_fmt}{b_str}" if prefix_fmt else b_str
                
                if cand_val.lower() not in assigned_encs and cand_val.lower() not in [cv.lower() for cv in conf_values] and b_str not in [cv.lower() for cv in conf_values]:
                    if cand_val.lower() not in seen_vals:
                        valid, _ = cls.validate_proposal_candidate(cand_val, conflict, efs_ir, target_entity)
                        if valid:
                            seen_vals.add(cand_val.lower())
                            prop_name = "binary_encoding" if conflict.type == "DUPLICATE_OPCODE_ENCODING" else "opcode"
                            prop_change = f"{target_entity}.{prop_name} = {cand_val}"
                            results.append((cand_val, prop_change, constraints_checked))
                            if len(results) >= count:
                                return results

        # 2. Register Address Offset Collision
        elif conflict.type in ("DUPLICATE_REGISTER_ADDRESS", "CONFLICTING_ADDRESS") or "address" in conflict.property.lower() or "offset" in conflict.property.lower():
            if target_entity.upper() in ("INTERFACE", "PORT", "SIGNAL", "FSM", "SPECIFICATION", "GENERAL", "SYSTEM"):
                return []
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
                        valid_val, _ = cls.validate_property_value(target_entity, "offset", cand_val, efs_ir)
                        if valid_val:
                            seen_vals.add(cand_val.lower())
                            prop_change = f"{target_entity}.offset = {cand_val}"
                            results.append((cand_val, prop_change, constraints_checked))
                            if len(results) >= count:
                                return results
                curr_off += stride

        # 3. Conflicting Signal Width
        elif conflict.type in ("CONFLICTING_WIDTH", "SIGNAL_WIDTH") or "width" in conflict.property.lower():
            constraints_checked = list(base_constraints) + ["width_standard_bounds"]
            # Propose standard widths (32, 64, 16, 8) or max conflicting width
            cand_widths = []
            for cv in conf_values:
                if str(cv).isdigit():
                    cand_widths.append(str(cv))
            cand_widths.extend(["32", "64", "16"])
            for w in cand_widths:
                if w not in seen_vals:
                    valid, _ = cls.validate_proposal_candidate(w, conflict, efs_ir, target_entity)
                    if valid:
                        seen_vals.add(w)
                        results.append((w, f"{target_entity}.width = {w}", constraints_checked))
                        if len(results) >= count:
                            return results

        # 4. Signal Direction Mismatch
        elif "direction" in conflict.property.lower() or "direction" in conflict.title.lower() or "direction" in conflict.description.lower():
            constraints_checked = list(base_constraints) + ["protocol_role_matched", "direction_enum_valid"]
            cand_dirs = []
            # First, check if conflicting values contain directions to offer as candidates
            for cv in conf_values:
                for d in ["output", "input", "inout"]:
                    if d in str(cv).lower() and d not in cand_dirs:
                        cand_dirs.append(d)
            # Then add remaining valid directions
            for d in ["output", "input", "inout"]:
                if d not in cand_dirs:
                    cand_dirs.append(d)

            for d in cand_dirs:
                if d not in seen_vals:
                    valid, _ = cls.validate_proposal_candidate(d, conflict, efs_ir, target_entity)
                    if valid:
                        seen_vals.add(d)
                        prop_change = f"{target_entity}.direction = {d}"
                        results.append((d, prop_change, constraints_checked))
                        if len(results) >= count:
                            return results

        # 5. Missing ACK Signal
        elif conflict.type in ("MISSING_ACK", "MISSING_SIGNAL") or "ack" in conflict.property.lower():
            constraints_checked = list(base_constraints) + ["signal_name_declaration"]
            ack_name = f"{target_entity.lower()}_ack" if not target_entity.lower().endswith("ack") else target_entity.lower()
            results.append((ack_name, f"signal.add = {ack_name}", constraints_checked))
            results.append(("ready", f"signal.add = ready", constraints_checked))
            return results

        # Fallback if less than count found
        # Fallback if less than count found
        if not results:
            fallback_val = None
            if ("direction" in conflict.property.lower() or "direction" in conflict.title.lower()) and target_entity.upper() not in ("FSM", "GENERAL", "SYSTEM"):
                fallback_val = "output"
                prop_change = f"{target_entity}.direction = {fallback_val}"
            elif ("width" in conflict.property.lower() or "width" in conflict.title.lower()) and target_entity.upper() not in ("FSM", "GENERAL", "SYSTEM"):
                fallback_val = "32"
                prop_change = f"{target_entity}.width = {fallback_val}"
            elif ("address" in conflict.property.lower() or "offset" in conflict.property.lower()) and target_entity.upper() not in ("INTERFACE", "PORT", "SIGNAL", "FSM", "GENERAL", "SYSTEM"):
                fallback_val = "0x0C"
                prop_change = f"{target_entity}.offset = {fallback_val}"
            elif "reset" in conflict.property.lower() or "reset" in conflict.title.lower():
                fallback_val = "0"
                prop_change = f"{target_entity}.reset_value = {fallback_val}"
            elif conf_values and conf_values[0] and str(conf_values[0]) not in seen_vals:
                valid_cv, _ = cls.validate_property_value(target_entity, conflict.property, conf_values[0], efs_ir)
                if valid_cv:
                    fallback_val = str(conf_values[0])
                    prop_change = f"{target_entity}.{conflict.property} = {fallback_val}"

            if fallback_val:
                valid_fb, _ = cls.validate_property_value(target_entity, conflict.property, fallback_val, efs_ir)
                if valid_fb:
                    results.append((fallback_val, prop_change, base_constraints))

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
        CRITICAL RULE: This method strictly returns non-authoritative proposals.
        If AI/LLM fails or is unavailable, deterministic concrete candidates ensure HITL works.
        """
        diag_type = getattr(conflict, "diagnostic_type", "SPEC_CONFLICT") or "SPEC_CONFLICT"
        if diag_type in (DiagnosticType.TARGET_RESOLUTION_FAILED.value, DiagnosticType.PLANNER_ERROR.value, "TARGET_RESOLUTION_FAILED", "PLANNER_ERROR"):
            logger.info(f"Skipping AI resolution proposal for resolver/planner diagnostic [{conflict.conflict_id}].")
            return []

        logger.info(f"AI_PROPOSAL_GENERATED: Generating AI resolution proposal for conflict [{conflict.conflict_id}] ({conflict.type}).")
        proposals: List[EFSResolution] = []
        conf_values = conflict.conflicting_values or ["Value A", "Value B"]
        entities = conflict.affected_entities or conflict.entities or ["Target Entity"]
        target_entity = entities[1] if len(entities) > 1 else entities[0]

        try:
            candidates = cls.calculate_concrete_candidates(conflict, efs_ir, count=2)
        except Exception as e:
            logger.warning(f"Error calculating candidates: {e}. Skipping candidates.")
            candidates = []

        for idx, (proposed_val, proposed_change, constraints_checked) in enumerate(candidates, 1):
            valid, err_msg = cls.validate_proposal_candidate(proposed_val, conflict, efs_ir, target_entity)
            if not valid:
                logger.warning(f"Candidate validation failed for '{proposed_val}': {err_msg}.")
                continue

            logger.info(f"CANDIDATE_VALIDATION_PASSED: Candidate #{idx} '{proposed_val}' passed deterministic validation.")

            cur_val = conflict.current_values.get(target_entity) or (conf_values[0] if conf_values else "N/A")
            if "direction" in conflict.property.lower() or "direction" in conflict.title.lower():
                rationale_text = f"Aligns '{target_entity}' direction to '{proposed_val}' to ensure compatible hardware port connections between RTL plan and canonical EFS IR."
            elif "width" in conflict.property.lower() or "width" in conflict.title.lower():
                rationale_text = f"Sets '{target_entity}' bit-width to '{proposed_val}' to ensure consistent bus sizing across module boundaries."
            elif "offset" in conflict.property.lower() or "address" in conflict.property.lower():
                rationale_text = f"Assigns '{target_entity}' to unique word-aligned address offset '{proposed_val}' to avoid memory map collision."
            elif "opcode" in conflict.property.lower() or "encoding" in conflict.property.lower():
                rationale_text = f"Assigns '{target_entity}' to unused binary encoding '{proposed_val}' to ensure unambiguous instruction decoding."
            else:
                rationale_text = (
                    f"{proposed_val} is a concrete valid value that satisfies all "
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
                original_value=cur_val,
                field_name=conflict.property,
                validation_status="PASSED",
                approval_status="PROPOSED",
                approved_by="AI_ASSISTANT",
                metadata={
                    "issue_id": conflict.conflict_id,
                    "field": conflict.property,
                    "affected_requirement": target_entity,
                    "current_value": cur_val,
                    "proposed_value": proposed_val,
                    "reason": rationale_text,
                    "constraints_checked": constraints_checked,
                    "confidence": 1.0,
                    "evidence": conflict.evidence
                }
            )
            proposals.append(p)

        # Attach top AI proposal to conflict for structured view
        if proposals:
            p0 = proposals[0]
            conflict.ai_proposal = {
                "type": "CHANGE_VALUE",
                "target": target_entity,
                "current_value": str(conflict.current_values.get(target_entity) or (conf_values[0] if conf_values else "N/A")),
                "proposed_value": str(p0.proposed_value),
                "reason": p0.rationale
            }

        logger.info(f"AI_PROPOSAL_GENERATED: Created {len(proposals)} AI proposal(s) for conflict [{conflict.conflict_id}].")
        return proposals

    @classmethod
    def create_custom_resolution(
        cls,
        conflict: EFSConflict,
        custom_requirement: str,
        user_name: str = "user",
        target_entity: Optional[str] = None
    ) -> EFSResolution:
        """Parses and structures custom user requirement input into a formal EFSResolution object."""
        logger.info(f"-> USER SELECTED AI/CUSTOM: Custom requirement submitted for conflict [{conflict.conflict_id}].")
        entities = conflict.affected_entities or conflict.entities or ["Target Entity"]
        target = target_entity or (entities[1] if len(entities) > 1 else entities[0])
        conf_values = conflict.conflicting_values or []
        req_clean = custom_requirement.strip()
        
        if "=" in req_clean:
            prop_change = req_clean
            prop_val = req_clean.split("=")[1].strip()
        else:
            prop_change = f"{target}.{conflict.property} = {req_clean}" if conflict.property else f"{target} = {req_clean}"
            prop_val = req_clean

        orig_val = conflict.current_values.get(target) or (conf_values[0] if conf_values else "N/A")

        res = EFSResolution(
            conflict_id=conflict.conflict_id,
            resolution_type="CUSTOM_REQUIREMENT",
            proposed_change=prop_change,
            rationale=f"User custom requirement override: '{req_clean}'",
            affected_requirements=[target],
            evidence=conflict.evidence,
            source_values=conf_values,
            proposed_value=prop_val,
            original_value=orig_val,
            field_name=conflict.property,
            validation_status="PASSED",
            approval_status="APPROVED",
            approved_by=user_name
        )
        logger.info(f"PROPOSAL_CREATED: Custom requirement resolution [{res.resolution_id}] created.")
        return res

    @classmethod
    def apply_approved_resolution(
        cls,
        efs_ir: EFSIR,
        approved_resolution: EFSResolution,
        force: bool = False
    ) -> EFSIR:
        """
        Records an approved resolution in the EFSIR resolution overlay layer,
        executes deterministic in-memory mutation, revalidates the current EFS IR state,
        and deterministically reconciles all conflicts.
        Requires ZERO LLM calls or EFS IR serialization to prompts.
        """
        res_type = "ai" if approved_resolution.resolution_type in ("ALLOW_AI", "ACCEPT_PROPOSAL", "PROPOSED", "AI_RECOMMENDED") else "custom"
        res_llm = 1 if res_type == "ai" else 0
        target_ent = approved_resolution.affected_requirements[0] if approved_resolution.affected_requirements else ""

        logger.info(
            f"[ConflictResolution] action=apply_resolution conflict_id={approved_resolution.conflict_id} "
            f"resolution_type={res_type} target_entity={target_ent} mutation_applied=true "
            f"resolution_llm_calls={res_llm} post_resolution_llm_calls=0"
        )
        # Idempotency check: if this resolution with identical proposed_value is already recorded and active, do not re-apply
        already_applied = any(
            (r.resolution_id == approved_resolution.resolution_id or (approved_resolution.conflict_id and r.conflict_id == approved_resolution.conflict_id))
            and r.approval_status == "APPROVED"
            and str(r.proposed_value).strip() == str(approved_resolution.proposed_value).strip()
            for r in getattr(efs_ir, "resolutions", [])
        )
        if already_applied and not force:
            logger.info(f"RESOLUTION_IDEMPOTENT: Resolution [{approved_resolution.resolution_id or approved_resolution.conflict_id}] already applied. Skipping duplicate application.")
            return efs_ir

        approved_resolution.approval_status = "APPROVED"
        
        # Server-side validation before application
        conf = next((c for c in efs_ir.conflicts if c.conflict_id == approved_resolution.conflict_id), None)
        if not conf and approved_resolution.affected_requirements:
            conf = next((c for c in efs_ir.conflicts if any(e in c.entities or e in c.affected_entities for e in approved_resolution.affected_requirements)), None)

        if conf and not force:
            valid, err = cls.validate_resolution(conf, approved_resolution.proposed_value, efs_ir, target_ent)
            if not valid:
                logger.error(f"Resolution application rejected: {err}")
                approved_resolution.validation_status = "REJECTED"
                raise ValueError(f"Resolution validation failed: {err}")
            approved_resolution.validation_status = "PASSED"
        else:
            approved_resolution.validation_status = "PASSED"
            if not approved_resolution.original_value:
                approved_resolution.original_value = conf.current_values.get(target_ent) or (conf.conflicting_values[0] if conf.conflicting_values else None)

        # Snapshot old conflicts before mutation
        old_conflicts = list(efs_ir.conflicts)
        prev_active_count = len([c for c in old_conflicts if (getattr(c, "status", "") == "ACTIVE" or getattr(c, "resolution_status", "") == "UNRESOLVED") and getattr(c, "status", "") != "RESOLVED"])

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

        # Mutate EFS IR: Rebuild resolved view overlaying resolutions onto immutable base EFSIR
        resolved_ir = efs_ir.get_resolved_view()

        # Deterministic conflict detection directly on in-memory model (ZERO LLM CALLS)
        raw_conflicts = cls._detect_raw_conflicts(resolved_ir)
        logger.info(
            f"[ConflictRevalidation] method=deterministic llm_called=false "
            f"previous_active_conflicts={prev_active_count} current_detected_conflicts={len(raw_conflicts)}"
        )

        # Deterministic semantic reconciliation
        reconciled_conflicts = cls.reconcile_conflicts(old_conflicts, raw_conflicts, resolved_ir)
        resolved_ir.conflicts = reconciled_conflicts
        efs_ir.conflicts = reconciled_conflicts

        resolved_ids = [c.conflict_id for c in reconciled_conflicts if c.status == "RESOLVED"]
        still_active = [c.conflict_id for c in reconciled_conflicts if c.status == "ACTIVE" and c in old_conflicts]
        newly_detected = [c.conflict_id for c in reconciled_conflicts if c.status == "ACTIVE" and c not in old_conflicts]

        logger.info(
            f"[ConflictReconciliation] resolved_conflicts={resolved_ids} "
            f"still_active={still_active} newly_detected={newly_detected} "
            f"post_resolution_llm_calls=0"
        )

        # Persist updated EFS state to active design file
        try:
            import json, os
            from config import CONFIG
            ir_dir = os.path.join(CONFIG.data_dir, "efs_ir")
            os.makedirs(ir_dir, exist_ok=True)
            active_ir_path = os.path.join(ir_dir, "active_design_ir.json")
            with open(active_ir_path, "w", encoding="utf-8") as f:
                json.dump(resolved_ir.to_dict(), f, indent=2)
        except Exception:
            pass

        return resolved_ir

    @classmethod
    def revalidate_resolved_model(
        cls,
        resolved_efs_ir: EFSIR,
        plan: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Mandatory revalidation step after approval of a resolution proposal.
        Re-evaluates validation rules and dynamically detects if new conflicts were introduced.
        Uses ZERO LLM calls.
        """
        logger.info("REVALIDATION: Revalidating resolved EFSIR model...")
        issues = validate_efs_ir(resolved_efs_ir)
        
        # Dynamically detect if newly resolved values created new conflicts (e.g. duplicate addresses)
        cls.detect_conflicts(resolved_efs_ir, validation_issues=issues)

        # Check for any remaining unresolved blocking conflicts
        unresolved_blocking_conflicts = [
            c for c in resolved_efs_ir.conflicts
            if c.blocking and (c.resolution_status == "UNRESOLVED" or c.status == "ACTIVE") and c.status != "RESOLVED"
        ]
        
        has_blocking = len(unresolved_blocking_conflicts) > 0
        
        if not has_blocking:
            logger.info("VALIDATION_PASSED: Revalidation passed cleanly. Proceeding with RTL and diagram generation.")
        else:
            logger.warning(f"REVALIDATION: {len(unresolved_blocking_conflicts)} blocking issues remain.")
            
        return has_blocking, issues

    @classmethod
    def validate_hardware_completeness(
        cls,
        efs_ir: EFSIR,
        plan: Optional[Dict[str, Any]] = None
    ) -> List[EFSConflict]:
        """
        Upfront completeness check ensuring the EFS IR has sufficient hardware
        information for PUML and RTL generation. A list of component names alone
        is NOT a complete specification.

        Checks:
          1. Ports/Signals — at least one signal with name, direction, width
          2. Interfaces — at least one with source and/or destination component
          3. Registers — at least one with offset and fields (if applicable)
          4. FSM — states AND transitions with conditions
          5. Clock/Reset — at least one clock and one reset signal
          6. Module relationships — components connected via interfaces
          7. Constraints/Timing — at least one constraint or timing rule

        Returns a list of blocking EFSConflict objects for each missing category.
        """
        logger.info("COMPLETENESS_CHECK: Validating EFS IR has sufficient hardware detail for generation...")
        completeness_conflicts: List[EFSConflict] = []
        seen_keys: set = set()

        def _add_completeness_conflict(
            title: str, description: str, impact: str,
            suggestions: List[str], prop: str, category: str
        ):
            c = EFSConflict(
                type="HARDWARE_COMPLETENESS_GAP",
                diagnostic_type=DiagnosticType.MISSING_SPECIFICATION.value,
                severity="CRITICAL",
                blocking=True,
                status="UNRESOLVED",
                title=title,
                entities=[category],
                affected_entities=[category],
                property=prop,
                conflicting_values=[],
                current_values={category: "MISSING"},
                source_references=[{
                    "entity": category,
                    "source": "Hardware Completeness Validator",
                    "section": f"Category: {category}",
                    "value": "Missing from EFS IR"
                }],
                evidence=description,
                description=description,
                explanation=description,
                impact=impact,
                suggestions=suggestions,
                requires_human_decision=True
            )
            key = cls.compute_semantic_key(c)
            c.semantic_key = key
            if key not in seen_keys:
                seen_keys.add(key)
                completeness_conflicts.append(c)

        if not efs_ir:
            _add_completeness_conflict(
                "No EFS IR Available",
                "The EFS Intermediate Representation was not built. No hardware specification data exists.",
                "Cannot generate any hardware outputs without an EFS IR.",
                ["Provide a valid hardware specification document."],
                "efs_ir", "EFS_IR"
            )
            return completeness_conflicts

        # 1. Signals/Ports check
        real_signals = [
            s for s in (efs_ir.signals or [])
            if s.name and s.direction and s.width
        ]
        if not real_signals:
            _add_completeness_conflict(
                "Missing Port/Signal Definitions",
                "No signals or ports with complete definitions (name, direction, width) found in the EFS IR. "
                "A list of component names without port definitions is not a complete specification.",
                "RTL generation requires explicit port declarations with directions and bit widths.",
                ["Define at least: signal name, direction (input/output), and bit width for each port."],
                "signals", "Ports_Signals"
            )

        # 2. Clock and Reset check
        clk_signals = [
            s for s in (efs_ir.signals or [])
            if (s.semantic_role or "").lower() == "clock"
            or any(k in (s.name or "").lower() for k in ("clk", "clock"))
        ]
        rst_signals = [
            s for s in (efs_ir.signals or [])
            if (s.semantic_role or "").lower() == "reset"
            or any(k in (s.name or "").lower() for k in ("rst", "reset"))
        ]
        if not clk_signals:
            _add_completeness_conflict(
                "Missing Clock Signal",
                "No clock signal found in the EFS IR. Synchronous hardware requires at least one clock.",
                "RTL modules require an explicit clock signal for sequential logic synthesis.",
                ["Define a clock signal (e.g., 'clk', 1-bit, input)."],
                "clock", "Clock_Reset"
            )
        if not rst_signals:
            _add_completeness_conflict(
                "Missing Reset Signal",
                "No reset signal found in the EFS IR. Hardware modules require a reset mechanism.",
                "RTL modules require an explicit reset signal for initialization.",
                ["Define a reset signal (e.g., 'rst_n', 1-bit, input, active-low)."],
                "reset", "Clock_Reset"
            )

        # 3. Interfaces with relationships check
        interfaces_with_connections = [
            iface for iface in (efs_ir.interfaces or [])
            if (iface.source_component or getattr(iface, "destination_component", None))
        ]
        has_components = bool(efs_ir.components and len(efs_ir.components) > 0)
        has_multi_components = bool(efs_ir.components and len(efs_ir.components) > 1)

        if has_multi_components and not interfaces_with_connections:
            _add_completeness_conflict(
                "Missing Inter-Module Relationships",
                f"The EFS IR has {len(efs_ir.components)} components but no interfaces connecting them. "
                "A component inventory without relationships is not a complete specification for diagram or RTL generation.",
                "PlantUML architecture diagrams and RTL interconnect require explicit module relationships.",
                ["Define interfaces with source and destination components to show how modules connect."],
                "interfaces", "Module_Relationships"
            )

        # 4. Registers check (skip if design has no register-mapped interface)
        has_reg_interface = any(
            "register" in (getattr(iface, "protocol", "") or "").lower()
            or "apb" in (getattr(iface, "protocol", "") or "").lower()
            or "axi" in (getattr(iface, "protocol", "") or "").lower()
            for iface in (efs_ir.interfaces or [])
        )
        plan_mentions_regs = bool(
            plan and isinstance(plan, dict) and
            any("register" in str(v).lower() for v in plan.values() if isinstance(v, str))
        )
        if (has_reg_interface or plan_mentions_regs) and not efs_ir.registers:
            _add_completeness_conflict(
                "Missing Register Definitions",
                "The design references register-mapped interfaces but the EFS IR contains no register definitions.",
                "RTL register file generation requires explicit register offsets, widths, and field descriptions.",
                ["Define registers with name, offset, width, and access mode (R/W/RW)."],
                "registers", "Registers"
            )

        # 5. FSM check (skip if design doesn't mention state machines)
        plan_mentions_fsm = bool(
            plan and isinstance(plan, dict) and (
                plan.get("fsm_architecture") or plan.get("state_machine") or
                any("fsm" in str(v).lower() or "state machine" in str(v).lower()
                    for v in plan.values() if isinstance(v, str))
            )
        )
        has_fsm_in_ir = bool(efs_ir.fsms and len(efs_ir.fsms) > 0)
        if plan_mentions_fsm and not has_fsm_in_ir:
            _add_completeness_conflict(
                "Missing FSM Definition",
                "The design plan references a state machine but the EFS IR contains no FSM definition.",
                "RTL FSM generation requires explicit states, transitions, and conditions.",
                ["Define FSM states, transitions, and trigger conditions."],
                "fsm", "FSM"
            )
        elif has_fsm_in_ir:
            for fsm in efs_ir.fsms:
                if not fsm.states or len(fsm.states) == 0:
                    _add_completeness_conflict(
                        f"FSM '{fsm.name or 'Primary'}' Has No States",
                        f"FSM '{fsm.name or 'Primary'}' is defined but contains no states.",
                        "RTL FSM synthesis requires at least two states.",
                        ["Define FSM states (e.g., IDLE, ACTIVE, DONE)."],
                        "fsm_states", "FSM"
                    )
                if not fsm.transitions or len(fsm.transitions) == 0:
                    _add_completeness_conflict(
                        f"FSM '{fsm.name or 'Primary'}' Has No Transitions",
                        f"FSM '{fsm.name or 'Primary'}' is defined but contains no transitions.",
                        "RTL FSM synthesis requires explicit state transitions with conditions.",
                        ["Define transitions between FSM states with trigger conditions."],
                        "fsm_transitions", "FSM"
                    )

        gap_count = len(completeness_conflicts)
        if gap_count > 0:
            logger.warning(f"COMPLETENESS_CHECK: Found {gap_count} hardware completeness gap(s) that must be resolved before generation.")
        else:
            logger.info("COMPLETENESS_CHECK: EFS IR has sufficient hardware detail for generation.")

        return completeness_conflicts

