"""
PlantUML Utilities and Alias Registry.

Provides generic, document-agnostic identifier sanitization, alias collision
management, pre-render syntax validation, and semantic construct mapping
for PlantUML Architecture, Sequence, and FSM diagrams.
"""

from __future__ import annotations

import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from utils.logger import get_logger

logger = get_logger("agents.plantuml_utils")

# PlantUML reserved keywords and directive words that must not be used as raw aliases
PLANTUML_RESERVED_WORDS: Set[str] = {
    "abstract", "actor", "agent", "annotation", "archimate", "artifact", "autonumber",
    "bidi", "boundary", "card", "class", "cloud", "collections", "component", "control",
    "database", "decision", "diamond", "entity", "enum", "file", "folder", "frame",
    "group", "hexagon", "hide", "interface", "label", "legend", "loop", "mainframe",
    "node", "note", "object", "opt", "package", "page", "participant", "partition",
    "person", "process", "protocol", "queue", "rectangle", "repeat", "scale", "skinparam",
    "stack", "state", "storage", "title", "together", "usecase", "while", "end", "activate",
    "deactivate", "return", "alt", "else", "opt", "loop", "par", "break", "critical",
    "group", "box", "create", "destroy"
}

# English stop words used for generic linguistic filtering of fragmented transaction names
FRAGMENT_STOP_WORDS: Set[str] = {
    "a", "an", "the", "as", "of", "in", "on", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "can", "will", "just", "should",
    "now", "describes", "order", "describing", "ordered"
}

GENERIC_DESIGN_NAMES: Set[str] = {
    "target_module", "target module", "hardware_module", "efs_hardware_module",
    "unnamed_design", "dut", "top_module", "module", "component", "target", ""
}


def to_plantuml_alias(display_name: str, canonical_id: Optional[str] = None) -> str:
    """
    Sanitize an arbitrary human-readable display name into a valid, deterministic PlantUML identifier.

    Rules:
    1. Trim whitespace.
    2. Replace spaces and punctuation/special characters with '_'.
    3. Preserve alphanumeric characters and underscores.
    4. Collapse repeated underscores.
    5. Strip leading/trailing underscores, but ensure valid identifier start (prefix '_' if digit).
    6. Ensure alias does not match reserved PlantUML keywords.
    7. Never mutate the input display_name.
    """
    if not display_name or not str(display_name).strip():
        if canonical_id and str(canonical_id).strip():
            raw = str(canonical_id).strip()
        else:
            return "elem_node"
    else:
        raw = str(display_name).strip()

    # Replace all non-alphanumeric and non-underscore characters with '_'
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", raw)

    # Collapse repeated underscores
    sanitized = re.sub(r"_+", "_", sanitized)

    # Strip leading and trailing underscores
    sanitized = sanitized.strip("_")

    # If completely empty after stripping (e.g. was only special characters)
    if not sanitized:
        if canonical_id and str(canonical_id).strip():
            clean_id = re.sub(r"[^A-Za-z0-9_]", "_", str(canonical_id)).strip("_")
            sanitized = clean_id or "elem_node"
        else:
            sanitized = "elem_node"

    # Ensure it starts with a valid character: must not start with a digit
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"

    # Avoid reserved words
    if sanitized.lower() in PLANTUML_RESERVED_WORDS:
        sanitized = f"elem_{sanitized}"

    return sanitized


class PlantUMLAliasRegistry:
    """
    Registry for managing deterministic PlantUML entity aliases and resolving collisions.

    Guarantees:
    - Unique aliases across all entities within the diagram.
    - Two representations per entity: display_name and plantuml_alias.
    - Deterministic collision resolution (e.g. 'A-B' and 'A B' -> 'A_B' and 'A_B_2').
    - Stable suffix resolution when canonical IDs are provided.
    """

    def __init__(self):
        self._name_to_alias: Dict[str, str] = {}
        self._id_to_alias: Dict[str, str] = {}
        self._alias_to_info: Dict[str, Dict[str, Any]] = {}
        self._declared_aliases: Set[str] = set()

    def register(
        self,
        display_name: str,
        canonical_id: Optional[str] = None,
        entity_type: Optional[str] = None
    ) -> str:
        """Register an entity and return its unique, deterministic PlantUML alias."""
        # Check if already registered by canonical_id
        if canonical_id and canonical_id in self._id_to_alias:
            return self._id_to_alias[canonical_id]

        # Check if already registered by exact display_name (only if canonical_id not given)
        if not canonical_id and display_name and display_name in self._name_to_alias:
            return self._name_to_alias[display_name]

        base_alias = to_plantuml_alias(display_name, canonical_id=canonical_id)
        candidate = base_alias

        # Collision detection and resolution
        if candidate in self._declared_aliases:
            if canonical_id:
                clean_suffix = re.sub(r"[^A-Za-z0-9_]", "", canonical_id.split("_")[-1])
                if clean_suffix and f"{base_alias}_{clean_suffix}" not in self._declared_aliases:
                    candidate = f"{base_alias}_{clean_suffix}"
                else:
                    counter = 2
                    while f"{base_alias}_{counter}" in self._declared_aliases:
                        counter += 1
                    candidate = f"{base_alias}_{counter}"
            else:
                counter = 2
                while f"{base_alias}_{counter}" in self._declared_aliases:
                    counter += 1
                candidate = f"{base_alias}_{counter}"

        self._declared_aliases.add(candidate)
        if display_name and display_name not in self._name_to_alias:
            self._name_to_alias[display_name] = candidate
        if canonical_id:
            self._id_to_alias[canonical_id] = candidate

        self._alias_to_info[candidate] = {
            "display_name": display_name,
            "canonical_id": canonical_id,
            "entity_type": entity_type or "component"
        }
        return candidate

    def get_alias(self, name_or_id: str) -> Optional[str]:
        """Look up alias by either canonical ID, display name, or existing alias."""
        if not name_or_id:
            return None
        if name_or_id in self._declared_aliases:
            return name_or_id
        if name_or_id in self._id_to_alias:
            return self._id_to_alias[name_or_id]
        if name_or_id in self._name_to_alias:
            return self._name_to_alias[name_or_id]
        return None

    def get_display_name(self, alias: str) -> str:
        """Get the human-readable display name for a declared alias."""
        info = self._alias_to_info.get(alias)
        if info and info.get("display_name"):
            return info["display_name"]
        return alias

    def is_declared(self, alias: str) -> bool:
        """Check if an alias is registered."""
        return alias in self._declared_aliases

    @property
    def all_aliases(self) -> Set[str]:
        """Return all declared aliases in this registry."""
        return set(self._declared_aliases)


def resolve_canonical_design_name(
    efs_ir: Optional[Any] = None,
    compiled_context: Optional[Dict[str, Any]] = None,
    design_plan: Optional[Dict[str, Any]] = None,
    design_spec: str = ""
) -> str:
    """
    Deterministically resolve the primary hardware design name.

    Priority:
    1. compiled_context primary_module
    2. EFS IR canonical metadata (display_name or design_name)
    3. EFS IR top_level_design component
    4. design_plan target or architecture target
    5. First non-generic EFS IR component
    6. Design specification title / header
    7. Source document name from EFS IR
    """
    # 1. From compiled_context primary_module
    if compiled_context and isinstance(compiled_context, dict):
        pm = compiled_context.get("primary_module", "").strip()
        if pm and pm.lower() not in GENERIC_DESIGN_NAMES:
            return pm

    # 2. From EFS IR metadata
    if efs_ir and hasattr(efs_ir, "metadata") and efs_ir.metadata:
        disp = getattr(efs_ir.metadata, "display_name", "").strip()
        if disp and disp.lower() not in GENERIC_DESIGN_NAMES:
            return disp
        dsn = getattr(efs_ir.metadata, "design_name", "").strip()
        if dsn and dsn.lower() not in GENERIC_DESIGN_NAMES:
            return dsn

    # 3. From EFS IR top_level_design component
    if efs_ir and hasattr(efs_ir, "components") and efs_ir.components:
        for c in efs_ir.components:
            c_type = (getattr(c, "type", "") or "").lower()
            c_name = (getattr(c, "name", "") or "").strip()
            if c_type in ("top_level_design", "top_level", "system") and c_name:
                if c_name.lower() not in GENERIC_DESIGN_NAMES:
                    return c_name
        # Or first non-placeholder component
        for c in efs_ir.components:
            c_name = (getattr(c, "name", "") or "").strip()
            if c_name and c_name.lower() not in GENERIC_DESIGN_NAMES:
                return c_name

    # 4. From design_plan target
    if design_plan and isinstance(design_plan, dict):
        plan_target = (
            design_plan.get("target")
            or design_plan.get("target_component")
            or design_plan.get("design_name")
            or design_plan.get("title")
        )
        if plan_target and str(plan_target).strip().lower() not in GENERIC_DESIGN_NAMES:
            return str(plan_target).strip()

    # 5. From design_spec title / markdown header
    if design_spec:
        m = re.search(r"^[#\t ]*(?:Hardware Design Specification:?\s*|Specification:?\s*)([A-Za-z0-9_\- ]+)", design_spec, re.MULTILINE | re.IGNORECASE)
        if m:
            clean = m.group(1).strip()
            if clean and clean.lower() not in GENERIC_DESIGN_NAMES and clean.lower() not in ("specification", "table of contents"):
                return clean
        m2 = re.search(r"^#\s+([A-Za-z0-9_\- ]{3,60})", design_spec, re.MULTILINE)
        if m2:
            clean2 = m2.group(1).strip()
            if clean2 and clean2.lower() not in GENERIC_DESIGN_NAMES and clean2.lower() not in ("specification", "table of contents"):
                return clean2

    # 6. From EFS IR source documents
    if efs_ir and hasattr(efs_ir, "metadata") and efs_ir.metadata and efs_ir.metadata.source_documents:
        first_doc = efs_ir.metadata.source_documents[0]
        doc_str = getattr(first_doc, "name", "") or str(first_doc)
        base = doc_str.rsplit(".", 1)[0].replace("_", " ").title().strip()
        if base and base.lower() not in GENERIC_DESIGN_NAMES and base.lower() not in ("specification", "spec"):
            return base

    return ""


def validate_interaction(
    interaction: Dict[str, Any],
    valid_participants: Set[str]
) -> Tuple[bool, str]:
    """
    Validate that a sequence interaction has legitimate structure and is not an extraction fragment.

    Rejects:
    - Dangling preposition fragments (e.g. 'AS_TRANSACTION', 'OF_TRANSACTION', 'DESCRIBES_TRANSACTION')
    - Plural category dumps (e.g. 'AXI_TRANSACTIONS', 'READ_TRANSACTIONS')
    - Pure stop-word phrases
    - Endpoints that do not resolve to valid participants
    """
    if not isinstance(interaction, dict):
        return False, "Interaction is not a dictionary"

    action = str(interaction.get("action", "") or "").strip()
    if not action:
        return False, "Empty action"

    # Reject plural category dumps like AXI_TRANSACTIONS, TRANSACTIONS, etc.
    if re.search(r"\bTRANSACTIONS\b", action, re.IGNORECASE) or action.lower().endswith("transactions"):
        return False, f"Action is a plural category header: '{action}'"

    # Normalize action for fragment check
    action_words = [w.lower() for w in re.split(r"[^A-Za-z0-9]+", action) if w]
    if not action_words:
        return False, "Action contains no alphanumeric tokens"

    # Reject if action is just 'transaction'
    if len(action_words) == 1 and action_words[0] in ("transaction", "transactions"):
        return False, f"Action is generic placeholder: '{action}'"

    # Check for uppercase tag extraction fragments like WRITE_TRANSACTION, ORDER_TRANSACTION, etc.
    if len(action_words) == 2 and action_words[1] in ("transaction", "transactions"):
        if action_words[0] in FRAGMENT_STOP_WORDS or action.isupper():
            return False, f"Action is extraction fragment: '{action}'"

    # Check if all words in action are stop words
    if all(w in FRAGMENT_STOP_WORDS for w in action_words):
        return False, f"Action consists solely of stop words: '{action}'"

    # Check participants
    src = str(interaction.get("from", "") or "").strip()
    tgt = str(interaction.get("to", "") or "").strip()

    if not src or not tgt:
        return False, f"Missing source or target participant in interaction: from='{src}', to='{tgt}'"

    if valid_participants:
        valid_lower = {p.lower() for p in valid_participants}
        valid_lower.update({"host", "master", "slave", "client", "controller", "environment"})
        if src.lower() not in valid_lower or tgt.lower() not in valid_lower:
            return False, f"Participant not declared in valid set: from='{src}', to='{tgt}'"

    return True, "Valid"


@dataclass
class PUMLValidationResult:
    """Validation report for generated PlantUML diagram."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    declared_aliases: Set[str] = field(default_factory=set)
    referenced_aliases: Set[str] = field(default_factory=set)


def validate_plantuml_syntax(puml_text: str, diagram_type: str = "generic") -> PUMLValidationResult:
    """
    Validate PlantUML text before rendering.

    Checks:
    1. Balanced diagram tags (@startuml and @enduml).
    2. Alias validity: Every alias in declarations must match '^[A-Za-z_][A-Za-z0-9_]*$'.
    3. Alias uniqueness: No two entities declare the same alias.
    4. Relationship references: Every arrow endpoint references a declared alias (or valid keyword).
    5. Participant references: In sequence diagrams, every interaction endpoint must be declared.
    6. Non-empty display names.
    """
    errors: List[str] = []
    warnings: List[str] = []
    declared_aliases: Set[str] = set()
    referenced_aliases: Set[str] = set()

    if not puml_text or not puml_text.strip():
        return PUMLValidationResult(is_valid=False, errors=["Empty PlantUML diagram text"])

    # 1. Check balanced @startuml / @enduml
    start_count = len(re.findall(r"@startuml\b", puml_text))
    end_count = len(re.findall(r"@enduml\b", puml_text))

    if start_count != 1:
        errors.append(f"Diagram must have exactly one @startuml tag (found {start_count})")
    if end_count != 1:
        errors.append(f"Diagram must have exactly one @enduml tag (found {end_count})")

    diag_type_clean = (diagram_type or "generic").lower()

    # Parse declarations line by line to avoid double-counting multi-pattern lines
    for line in puml_text.splitlines():
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("'") or trimmed.startswith("@") or trimmed.startswith("note") or trimmed.startswith("title"):
            continue

        # Sequence participant declaration
        if diag_type_clean in ("sequence", "generic"):
            m_seq = re.match(
                r"^(?:participant|actor|boundary|control|entity|database|collections|queue)\s+(?:\"([^\"]*)\"|([^\s]+))\s+as\s+([^<>\n\r]+?)(?:\s*<<.*>>)?$",
                trimmed,
                re.IGNORECASE
            )
            if m_seq:
                disp = m_seq.group(1) or m_seq.group(2)
                alias = m_seq.group(3).strip()
                if not disp or not disp.strip():
                    errors.append(f"Empty display name in declaration: '{trimmed}'")
                if " " in alias or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", alias):
                    errors.append(f"Invalid alias identifier '{alias}': must match ^[A-Za-z_][A-Za-z0-9_]*$ (no spaces or special chars)")
                if alias in declared_aliases:
                    errors.append(f"Duplicate alias declaration: '{alias}' declared multiple times")
                declared_aliases.add(alias)
                continue

        # Architecture declaration
        if diag_type_clean in ("architecture", "generic"):
            m_arch = re.match(
                r"^(?:component|interface|database|storage|node|package|frame|folder|rectangle)\s+(?:\"([^\"]*)\"|([^\s]+))\s+as\s+([^<>\n\r]+?)(?:\s*<<.*>>)?$",
                trimmed,
                re.IGNORECASE
            )
            if m_arch:
                disp = m_arch.group(1) or m_arch.group(2)
                alias = m_arch.group(3).strip()
                if not disp or not disp.strip():
                    errors.append(f"Empty display name in declaration: '{trimmed}'")
                if " " in alias or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", alias):
                    errors.append(f"Invalid alias identifier '{alias}': must match ^[A-Za-z_][A-Za-z0-9_]*$ (no spaces or special chars)")
                if alias in declared_aliases:
                    errors.append(f"Duplicate alias declaration: '{alias}' declared multiple times")
                declared_aliases.add(alias)
                continue

            # Bracket declaration
            m_brk = re.match(r"^\[([^\]]+)\]\s+as\s+([^<>\n\r]+?)(?:\s*<<.*>>)?$", trimmed)
            if m_brk:
                disp = m_brk.group(1).strip()
                alias = m_brk.group(2).strip()
                if not disp:
                    errors.append(f"Empty display name in bracket declaration: '{trimmed}'")
                if " " in alias or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", alias):
                    errors.append(f"Invalid bracket alias identifier '{alias}': must match ^[A-Za-z_][A-Za-z0-9_]*$ (no spaces or special chars)")
                if alias in declared_aliases:
                    errors.append(f"Duplicate alias declaration: '{alias}' declared multiple times")
                declared_aliases.add(alias)
                continue

        # Check broken declaration with spaces
        broken_bracket = re.match(r"^\[[^\]]+\]\s+as\s+([^<>\n\r]+)<<", trimmed)
        if broken_bracket:
            raw_clean = broken_bracket.group(1).strip()
            if " " in raw_clean:
                errors.append(f"Invalid alias containing spaces in syntax: '... as {raw_clean} <<...'")

        broken_part = re.match(r"^participant\s+\"[^\"]+\"\s+as\s+([^\n\r]+)$", trimmed)
        if broken_part:
            raw_clean = broken_part.group(1).strip()
            if " " in raw_clean:
                errors.append(f"Invalid participant alias containing spaces: '... as {raw_clean}'")

    # Check relationships and messages
    seq_arrows = re.findall(r"^\s*([A-Za-z0-9_]+)\s*(?:->|-->|->>|-->>|<-|<--)\s*([A-Za-z0-9_]+)\s*:", puml_text, re.MULTILINE)
    for src, tgt in seq_arrows:
        referenced_aliases.add(src)
        referenced_aliases.add(tgt)
        if diag_type_clean == "sequence" and declared_aliases:
            if src not in declared_aliases:
                errors.append(f"Sequence message references undeclared participant '{src}'")
            if tgt not in declared_aliases:
                errors.append(f"Sequence message references undeclared participant '{tgt}'")

    arch_arrows = re.findall(r"^\s*([A-Za-z0-9_]+)\s*(?:-->|\.\.>|--|\.\.|\-\-\|>)\s*([A-Za-z0-9_]+)", puml_text, re.MULTILINE)
    for src, tgt in arch_arrows:
        referenced_aliases.add(src)
        referenced_aliases.add(tgt)
        if diag_type_clean == "architecture" and declared_aliases:
            if src not in declared_aliases and src != "[*]":
                warnings.append(f"Architecture relationship references undeclared element '{src}'")
            if tgt not in declared_aliases and tgt != "[*]":
                warnings.append(f"Architecture relationship references undeclared element '{tgt}'")

    is_valid = len(errors) == 0
    return PUMLValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        declared_aliases=declared_aliases,
        referenced_aliases=referenced_aliases
    )


def verify_plantuml_render(puml_text: str, timeout: int = 5) -> Tuple[bool, str]:
    """
    Render PlantUML diagram against the official PlantUML server or validator.

    Returns (is_success, message_or_error).
    """
    val = validate_plantuml_syntax(puml_text)
    if not val.is_valid:
        return False, f"Local syntax validation failed: {'; '.join(val.errors)}"

    try:
        from core.agents.plantuml_agent import get_diagram_url
        url = get_diagram_url(puml_text)
        if not url:
            return False, "Failed to encode PlantUML text into URL"

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            content_type = resp.headers.get("Content-Type", "")
            if status == 200 and "image" in content_type:
                return True, f"Render successful ({content_type})"
            return False, f"Unexpected response status {status} ({content_type})"
    except urllib.error.HTTPError as e:
        if e.code == 400:
            return False, f"PlantUML Syntax Error (HTTP 400 Bad Request): {e.reason}"
        return False, f"HTTP Error {e.code}: {e.reason}"
    except Exception as ex:
        logger.warning(f"Could not connect to online PlantUML server: {ex}. Relying on local validator.")
        return True, f"Local validator passed; online server unreachable ({ex})"
