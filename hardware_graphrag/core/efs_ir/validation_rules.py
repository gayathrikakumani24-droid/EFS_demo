"""
Reusable Entity Validation & Safety Rules for EFS IR.

Enforces structural and semantic validation to ensure:
1. False hardware entities are never created from non-hardware text or malformed tables.
2. Missing values are preserved explicitly as None (uncertainty preserved).
3. Explicit values (including 0x0) are preserved accurately.
4. Duplicate-address validation only operates on valid, explicit addresses.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple


# Common English grammatical words, verbs, structural labels that cannot be register names
# without supporting hardware evidence (e.g., explicit hex offset, RW access, register fields).
NON_HARDWARE_WORDS: Set[str] = {
    # Verbs and prepositions
    "contains", "containing", "contain", "includes", "including", "include",
    "consists", "consisting", "provides", "providing", "shows", "defined",
    "of", "in", "to", "for", "with", "by", "from", "at", "on", "as",
    "and", "or", "nor", "but", "so", "yet", "if", "then", "else",
    "the", "a", "an", "this", "that", "these", "those", "each", "all",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    # Document and structural meta-words
    "table", "figure", "section", "page", "chapter", "row", "column",
    "note", "notes", "comment", "comments", "description", "descriptions",
    "name", "names", "title", "type", "types", "value", "values",
    "source", "destination", "dimension", "dimensions", "address", "addresses",
    "offset", "offsets", "index", "indices", "item", "items", "entry", "entries",
    "example", "examples", "summary", "overview", "specification", "spec",
    "parameter", "parameters", "property", "properties", "attribute", "attributes",
    "total", "count", "size", "length", "width", "height", "level", "levels",
    "step", "steps", "mode", "operation", "reserved", "none", "null", "tbd", "na", "n/a",
}

# Standard hardware access modes
VALID_ACCESS_MODES: Set[str] = {
    "RW", "RO", "WO", "R/W", "R/O", "W/O", "W1C", "W1S", "W0C", "RC", "RS", "WC", "WS",
    "RWH", "ROH", "W1T", "R", "W", "READ-ONLY", "READ-WRITE", "WRITE-ONLY", "READ/WRITE"
}


def is_missing_value(val: Any) -> bool:
    """Check if a value represents missing/unassigned specification data."""
    if val is None:
        return True
    if isinstance(val, str):
        clean = val.strip().lower()
        return clean in {"", "none", "null", "nil", "tbd", "unassigned", "n/a", "na", "-", "--"}
    return False


def is_explicit_value(val: Any) -> bool:
    """Check if a value is explicitly provided (not missing or placeholder)."""
    return not is_missing_value(val)


def normalize_hardware_offset(raw: Any) -> Optional[str]:
    """
    Normalize a raw offset or address value into a canonical hex string (e.g. '0x0', '0x10').
    Returns None if the value is missing, empty, non-numeric text, or unassigned.
    Explicit 0 / '0x0' / '0' is preserved as '0x0'.
    """
    if is_missing_value(raw):
        return None

    if isinstance(raw, int):
        return f"0x{raw:X}"

    if isinstance(raw, str):
        clean = raw.strip()
        if not clean:
            return None

        # Check for hex representation e.g. 0x04, 0X1A, 'h04
        if clean.lower().startswith("0x"):
            hex_part = clean[2:].replace("_", "").rstrip("hH")
            try:
                int(hex_part, 16)
                return clean
            except ValueError:
                return None
        elif clean.startswith("'h") or clean.startswith("16'h") or clean.startswith("32'h"):
            hex_part = clean.split("'h")[-1].replace("_", "")
            try:
                val = int(hex_part, 16)
                return f"0x{val:X}"
            except ValueError:
                return None

        # Pure decimal integer
        clean_dec = clean.replace("_", "")
        if clean_dec.isdigit():
            val = int(clean_dec, 10)
            return f"0x{val:X}"

        # Pure hex digits without 0x prefix if length <= 8 and all valid hex digits,
        # but only if not a standard English word
        if re.fullmatch(r"[0-9A-Fa-f]{1,8}", clean) and not clean.lower() in NON_HARDWARE_WORDS:
            try:
                val = int(clean, 16)
                return f"0x{val:X}"
            except ValueError:
                pass

    return None


def is_explicit_address(val: Any) -> bool:
    """
    Check whether a register offset/address is an explicit valid address.
    Explicit 0x0 returns True. None or missing returns False.
    """
    norm = normalize_hardware_offset(val)
    return norm is not None


def normalize_hardware_width(raw: Any) -> Optional[int]:
    """
    Normalize width to positive integer. Returns None if missing or invalid.
    Never silently defaults to 1 or 32 when source is missing.
    """
    if is_missing_value(raw):
        return None
    if isinstance(raw, int) and raw > 0:
        return raw
    if isinstance(raw, str):
        # Handle bit range e.g. [31:0] -> 32
        m = re.match(r"\[?\s*(\d+)\s*:\s*(\d+)\s*\]?", raw.strip())
        if m:
            msb, lsb = int(m.group(1)), int(m.group(2))
            return abs(msb - lsb) + 1
        clean = raw.strip().replace("bits", "").replace("bit", "").strip()
        if clean.isdigit():
            val = int(clean)
            return val if val > 0 else None
    return None


def is_valid_register_candidate(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    context: Optional[str] = None,
) -> bool:
    """
    Determine if a candidate name is a plausible hardware register entity.
    Rejects:
    - Pure numbers (e.g. "1", "2")
    - Punctuation / whitespace
    - Long sentences or multi-word phrases (> 4 words)
    - Common English non-hardware tokens UNLESS supported by strong hardware evidence
      (such as explicit hex address or RW access or bitfield list).
    """
    if not name or not isinstance(name, str):
        return False

    clean_name = name.strip()
    if not clean_name:
        return False

    # Bare numbers e.g. "1", "2", "10" are never valid register names
    if clean_name.isdigit():
        return False

    # Pure symbols/punctuation
    if not re.search(r"[A-Za-z]", clean_name):
        return False

    # Multi-word prose (e.g. table header fragments or sentences)
    words = clean_name.split()
    if len(words) > 4:
        return False

    # Check for strong hardware evidence in attributes
    attrs = attributes or {}
    has_hex_offset = is_explicit_address(attrs.get("offset") or attrs.get("address"))
    access_type = str(attrs.get("access_type") or attrs.get("access") or "").strip().upper()
    has_access = access_type in VALID_ACCESS_MODES
    has_fields = bool(attrs.get("fields"))
    has_reset = is_explicit_value(attrs.get("reset_value") or attrs.get("reset"))

    # Strong hardware evidence allows somewhat generic names (e.g. STATUS, CONTROL, SOURCE_ADDR)
    if has_hex_offset and (has_access or has_fields or has_reset):
        return True

    # If the token is a single English non-hardware word (e.g. "contains", "Source", "Dimension", "Addresses")
    # without hex offset and without access type, it is NOT a register.
    lower_name = clean_name.lower()
    if lower_name in NON_HARDWARE_WORDS:
        # If it doesn't even have a hex offset, reject immediately
        if not has_hex_offset:
            return False

    # Hardware naming conventions: typically UPPERCASE, snake_case, or contains hardware register tokens
    # e.g. REG, CTRL, STATUS, CFG, CONFIG, DATA, ADDR, IER, ISR, CR, SR, etc.
    reg_keyword_pattern = r"(?i)(reg|ctrl|control|status|cfg|config|addr|address|data|ier|isr|icr|id|cnt|count|val|timer|irq|mask|en|enable|buf|fifo|div)"
    if re.search(reg_keyword_pattern, clean_name):
        return True

    # Common C/hardware identifier: [A-Za-z_][A-Za-z0-9_]*
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", clean_name):
        # All-caps or snake_case with hardware evidence
        if clean_name.isupper() or "_" in clean_name:
            if has_hex_offset or has_access or has_fields:
                return True
            # Even without offset, standard all-caps register abbreviations are plausible
            if len(clean_name) <= 16 and not lower_name in NON_HARDWARE_WORDS:
                return True

    # If context explicitly identifies it as a register
    if context and ("register" in context.lower() or "reg" in context.lower()):
        if has_hex_offset or has_access:
            return True

    return False


def has_sufficient_register_evidence(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    headers: Optional[List[str]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Score the confidence and evidence for a register candidate.
    Returns (is_sufficient, evidence_dict).
    """
    attrs = attributes or {}
    hdrs = [h.lower() for h in (headers or [])]

    evidence: Dict[str, Any] = {
        "explicit_register_name": False,
        "address_present": False,
        "register_map_context": False,
        "access_type_present": False,
        "fields_present": False,
        "reset_value_present": False,
    }

    # Address
    offset_val = attrs.get("offset") or attrs.get("address")
    if is_explicit_address(offset_val):
        evidence["address_present"] = True

    # Access
    access = str(attrs.get("access_type") or attrs.get("access") or "").strip().upper()
    if access in VALID_ACCESS_MODES:
        evidence["access_type_present"] = True

    # Fields
    if bool(attrs.get("fields")):
        evidence["fields_present"] = True

    # Reset
    if is_explicit_value(attrs.get("reset_value") or attrs.get("reset")):
        evidence["reset_value_present"] = True

    # Context / Headers
    if any("register" in h or "reg name" in h or "reg_name" in h for h in hdrs):
        evidence["register_map_context"] = True

    # Name
    clean_name = name.strip() if name else ""
    if is_valid_register_candidate(clean_name, attributes):
        evidence["explicit_register_name"] = True

    # A candidate needs at least valid name + (address OR access OR register_map_context OR fields)
    is_sufficient = evidence["explicit_register_name"] and (
        evidence["address_present"] or evidence["access_type_present"] or
        evidence["register_map_context"] or evidence["fields_present"]
    )

    return is_sufficient, evidence


def is_valid_signal_candidate(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Validate whether a token is a plausible hardware signal/port.
    Rejects bare numbers, stop words, punctuation.
    """
    if not name or not isinstance(name, str):
        return False
    clean = name.strip()
    if not clean or clean.isdigit():
        return False
    if not re.search(r"[A-Za-z]", clean):
        return False
    if clean.lower() in NON_HARDWARE_WORDS:
        attrs = attributes or {}
        has_dir = str(attrs.get("direction", "")).lower() in {"in", "out", "inout", "input", "output"}
        has_width = is_explicit_value(attrs.get("width"))
        return has_dir and has_width
    return True
