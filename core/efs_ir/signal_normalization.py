"""
Generic Hardware Signal Metadata Normalization and Conflict Resolution.

Protocol-agnostic and design-agnostic normalization for arbitrary signals, ports, and interfaces.
Enforces evidence-based authority hierarchy:
  EXPLICIT_HDL (authority 4): Explicit HDL declaration (Verilog/SystemVerilog/VHDL port syntax)
  EXPLICIT_SPEC (authority 3): Explicit structured specification attributes (tables, bullet lists)
  INFERRED_ROLE (authority 2): Protocol/architectural role inference when explicitly configured
  INFERRED_CONVENTION (authority 1): Naming convention inference (low authority)
  DEFAULT / UNKNOWN (authority 0): Unspecified or fallback values

Provides reusable utilities for:
- Normalizing direction representations consistently.
- Parsing explicit HDL port declarations and bit ranges generically.
- Preserving widths and ranges from authoritative source evidence.
- Merging duplicate records using explicit precedence rules.
- Preserving provenance for every resolved field.
- Distinguishing explicitly specified values from inferred, defaulted, or unknown values.
- Comparing port metadata semantically without representational bias.
"""

from __future__ import annotations

import re
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple


class EvidenceAuthority(IntEnum):
    """Authority level for signal metadata evidence."""
    UNKNOWN = 0
    DEFAULT = 1
    INFERRED_CONVENTION = 2
    INFERRED_ROLE = 3
    EXPLICIT_SPEC = 4
    EXPLICIT_HDL = 5


# Direction canonical strings
CANONICAL_DIRECTIONS: Set[str] = {"input", "output", "inout"}

# Exact direction aliases
_DIRECTION_MAP: Dict[str, str] = {
    "in": "input",
    "input": "input",
    "in_port": "input",
    "ip": "input",
    "i": "input",
    "out": "output",
    "output": "output",
    "out_port": "output",
    "op": "output",
    "o": "output",
    "inout": "inout",
    "in_out": "inout",
    "bidir": "inout",
    "bidirectional": "inout",
    "bdir": "inout",
    "io": "inout",
}


def normalize_direction(raw: Any) -> Optional[str]:
    """
    Normalize an arbitrary direction representation to canonical 'input', 'output', or 'inout'.
    Returns None if missing, empty, or unparseable.
    
    Protocol-agnostic and design-agnostic: never infers direction from signal names.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        raw = str(raw)

    clean = raw.strip().lower()
    if not clean or clean in ("none", "null", "undefined", "unspecified", "tbd", "n/a", "-", "--", "dir", "direction"):
        return None

    # Check exact dictionary match
    if clean in _DIRECTION_MAP:
        return _DIRECTION_MAP[clean]

    # Clean punctuation / whitespace
    clean_alpha = re.sub(r"[^a-z/_]", "", clean)
    if clean_alpha in _DIRECTION_MAP:
        return _DIRECTION_MAP[clean_alpha]

    # Handle combined/compound patterns
    if ("in" in clean_alpha and "out" in clean_alpha) or "bidir" in clean_alpha:
        return "inout"
    if "out" in clean_alpha:
        return "output"
    if "in" in clean_alpha:
        return "input"

    return None


def parse_bit_range(val: Any) -> Tuple[Optional[int], Optional[str]]:
    """
    Parse a width, bit range, or dimension string generically.
    Returns (integer_width, range_str).
    
    Examples:
      - 32 -> (32, "[31:0]")
      - "32" -> (32, "[31:0]")
      - "32 bits" -> (32, "[31:0]")
      - "[31:0]" -> (32, "[31:0]")
      - "[0:31]" -> (32, "[0:31]")
      - "[511:0]" -> (512, "[511:0]")
      - "1" -> (1, "")
      - "[DATA_WIDTH-1:0]" -> (None, "[DATA_WIDTH-1:0]")
      - "ADDR_WIDTH" -> (None, "[ADDR_WIDTH-1:0]")
      - None / "" -> (None, None)
    """
    if val is None:
        return None, None

    if isinstance(val, int):
        if val <= 0:
            return None, None
        if val == 1:
            return 1, ""
        return val, f"[{val - 1}:0]"

    s = str(val).strip()
    if not s or s.lower() in ("none", "null", "undefined", "unspecified", "tbd", "n/a", "-", "--", "width", "size"):
        return None, None

    # Match bit range pattern e.g. [31:0] or 31:0 or [DATA_WIDTH-1:0]
    m_range = re.match(r"^\[?\s*([^:\]\s]+)\s*:\s*([^:\]\s]+)\s*\]?$", s)
    if m_range:
        msb_str = m_range.group(1).strip()
        lsb_str = m_range.group(2).strip()
        if msb_str.isdigit() and lsb_str.isdigit():
            msb = int(msb_str)
            lsb = int(lsb_str)
            width = abs(msb - lsb) + 1
            return width, f"[{msb}:{lsb}]"
        else:
            # Symbolic range
            return None, f"[{msb_str}:{lsb_str}]"

    # Match integer width e.g. "32", "512", "32 bits", "1 bit"
    clean_num = re.sub(r"\s*(?:bits?|bit)\s*$", "", s, flags=re.IGNORECASE).strip()
    if clean_num.isdigit():
        w = int(clean_num)
        if w <= 0:
            return None, None
        if w == 1:
            return 1, ""
        return w, f"[{w - 1}:0]"

    # Symbolic parameter name e.g. "DATA_WIDTH"
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", s):
        return None, f"[{s}-1:0]"

    return None, None


# Generic HDL port regex:
# Matches:
#   [optional line number/bullet prefix]
#   input|output|inout
#   [optional wire|reg|logic]
#   [optional [MSB:LSB] range]
#   signal_name
#   [optional trailing comma/semicolon/comment]
_HDL_PORT_REGEX = re.compile(
    r"(?:^|[^\w])"
    r"(?:\[\d+\]\s*|\d+:\s*|[-*]\s*)?"
    r"\b(input|output|inout)\s+"
    r"(?:wire|reg|logic)?\s*"
    r"(?:\[\s*([^:\]\s]+)\s*:\s*([^:\]\s]+)\s*\])?\s*"
    r"([a-zA-Z_][a-zA-Z0-9_$]*)\b"
    r"(?:\s*[,;])?",
    re.IGNORECASE
)


def parse_hdl_port_declaration(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse an explicit HDL port declaration from arbitrary text, code lines, or descriptions.
    
    Protocol-agnostic and design-agnostic.
    Handles:
      - 'output wire [31:0] axi_araddr,'
      - '[242] output wire [31:0] axi_araddr,'
      - '- input wire [511:0] axi_rdata,'
      - 'input wire axi_arready,'
      - 'inout [7:0] data_bus;'
      - 'output logic [DATA_WIDTH-1:0] result'
      - 'input logic clk'
    """
    if not text or not isinstance(text, str):
        return None

    clean = text.strip()
    m = _HDL_PORT_REGEX.search(clean)
    if not m:
        return None

    raw_dir = m.group(1)
    msb_str = m.group(2)
    lsb_str = m.group(3)
    sig_name = m.group(4)

    norm_dir = normalize_direction(raw_dir)
    if not norm_dir:
        return None

    if msb_str is not None and lsb_str is not None:
        msb_str = msb_str.strip()
        lsb_str = lsb_str.strip()
        if msb_str.isdigit() and lsb_str.isdigit():
            width = abs(int(msb_str) - int(lsb_str)) + 1
            range_str = f"[{msb_str}:{lsb_str}]"
        else:
            width = None
            range_str = f"[{msb_str}:{lsb_str}]"
    else:
        width = 1
        range_str = ""

    return {
        "name": sig_name,
        "direction": norm_dir,
        "width": width,
        "width_str": str(width) if width is not None else (range_str or "1"),
        "range": range_str,
        "authority": EvidenceAuthority.EXPLICIT_HDL,
        "source_evidence": clean
    }


def parse_markdown_bullet_port(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse structured markdown bullet port declarations:
    e.g. '- axi_araddr: output, 32 bits, Address of first transfer'
         '* data_out : out, [31:0], Output data'
    """
    if not text or not isinstance(text, str):
        return None

    m = re.match(
        r"^[-*]\s*([a-zA-Z0-9_$]+)\s*:\s*(input|output|inout|in|out|io|bidir)\s*,\s*(\[[^\]]+\]|\d+|\w+)\s*(?:bits?|bit)?(?:\s*,\s*(.*))?$",
        text.strip(),
        re.IGNORECASE
    )
    if not m:
        return None

    name = m.group(1).strip()
    raw_dir = m.group(2).strip()
    raw_w = m.group(3).strip()
    desc = (m.group(4) or "").strip()

    norm_dir = normalize_direction(raw_dir)
    if not norm_dir:
        return None

    w_int, r_str = parse_bit_range(raw_w)
    return {
        "name": name,
        "direction": norm_dir,
        "width": w_int,
        "width_str": str(w_int) if w_int is not None else (r_str or raw_w),
        "range": r_str,
        "description": desc,
        "authority": EvidenceAuthority.EXPLICIT_SPEC,
        "source_evidence": text.strip()
    }


class CandidateMetadata:
    """Represents a candidate attribute value with explicit authority and provenance."""
    def __init__(
        self,
        value: Any,
        authority: EvidenceAuthority,
        source_description: str,
        raw_value: Any = None
    ):
        self.value = value
        self.raw_value = raw_value if raw_value is not None else value
        self.authority = authority
        self.source_description = source_description

    def __repr__(self) -> str:
        return f"Candidate(val={self.value!r}, auth={self.authority.name}, src={self.source_description!r})"


def resolve_property_by_evidence(
    candidates: List[CandidateMetadata],
    property_name: str,
    entity_name: str = ""
) -> Dict[str, Any]:
    """
    Generic conflict-resolution mechanism based on source authority, explicitness, and provenance.
    
    Rules:
    1. Filter out empty/unknown candidates.
    2. Group candidates by normalized canonical value.
    3. Identify maximum authority level.
    4. If all candidates at max authority agree -> WINNER resolved with provenance.
    5. If equally authoritative candidates disagree -> UNRESOLVED conflict flagged.
    6. If no candidates exist -> UNKNOWN returned without fabricated evidence.
    """
    valid_cands = [c for c in candidates if c.value is not None and c.authority > EvidenceAuthority.UNKNOWN]
    if not valid_cands:
        return {
            "resolved_value": None,
            "authority": EvidenceAuthority.UNKNOWN,
            "source": "none",
            "is_conflicted": False,
            "evidence": "",
            "candidates": []
        }

    # Find highest authority level
    max_auth = max(c.authority for c in valid_cands)
    top_cands = [c for c in valid_cands if c.authority == max_auth]

    # Group unique top-level values
    distinct_vals: Dict[Any, List[CandidateMetadata]] = {}
    for c in top_cands:
        distinct_vals.setdefault(c.value, []).append(c)

    if len(distinct_vals) == 1:
        # Unanimous agreement at highest authority
        winner_val = next(iter(distinct_vals.keys()))
        winner_cands = distinct_vals[winner_val]
        src_desc = "; ".join(c.source_description for c in winner_cands)
        return {
            "resolved_value": winner_val,
            "authority": max_auth,
            "source": max_auth.name.lower(),
            "is_conflicted": False,
            "evidence": src_desc,
            "candidates": [repr(c) for c in valid_cands]
        }
    else:
        # Genuine conflict between equally authoritative sources!
        conflict_details = []
        for val, c_list in distinct_vals.items():
            sources = ", ".join(c.source_description for c in c_list)
            conflict_details.append(f"value '{val}' from [{sources}]")
        conflict_msg = (
            f"Conflicting {property_name} for '{entity_name}': "
            f"Equally authoritative sources ({max_auth.name}) disagree: "
            + "; versus; ".join(conflict_details)
        )
        return {
            "resolved_value": None,
            "authority": max_auth,
            "source": f"conflict_{max_auth.name.lower()}",
            "is_conflicted": True,
            "conflict_message": conflict_msg,
            "evidence": "; ".join(c.source_description for c in top_cands),
            "candidates": [repr(c) for c in valid_cands]
        }


def extract_signal_candidates_from_record(
    record: Any,
    signal_name: str = "",
    spec_text: str = ""
) -> Tuple[List[CandidateMetadata], List[CandidateMetadata]]:
    """
    Extract direction and width candidate metadata from arbitrary signal records,
    dictionaries, attributes, and descriptions.
    
    Returns (direction_candidates, width_candidates).
    """
    dir_cands: List[CandidateMetadata] = []
    width_cands: List[CandidateMetadata] = []

    # Unwrap record
    r_dict: Dict[str, Any] = {}
    attrs_dict: Dict[str, Any] = {}
    desc = ""
    name = signal_name

    if isinstance(record, dict):
        r_dict = record
        if isinstance(record.get("attributes"), dict):
            attrs_dict = record["attributes"]
        desc = str(record.get("description") or attrs_dict.get("description") or "")
        name = name or str(record.get("name") or attrs_dict.get("name") or "")
    elif hasattr(record, "__dict__"):
        r_dict = getattr(record, "__dict__", {})
        if hasattr(record, "attributes") and isinstance(record.attributes, dict):
            attrs_dict = record.attributes
        desc = str(getattr(record, "description", "") or attrs_dict.get("description", ""))
        name = name or str(getattr(record, "name", "") or attrs_dict.get("name", ""))

    # 1. Check description or source_text for explicit HDL port declaration
    text_to_scan = [desc]
    if r_dict.get("source_text"):
        text_to_scan.append(str(r_dict["source_text"]))
    if attrs_dict.get("source_text"):
        text_to_scan.append(str(attrs_dict["source_text"]))
    if r_dict.get("mention"):
        text_to_scan.append(str(r_dict["mention"]))

    for txt in text_to_scan:
        if not txt:
            continue
        hdl_info = parse_hdl_port_declaration(txt)
        if hdl_info and (not name or hdl_info["name"].lower() == name.lower()):
            dir_cands.append(CandidateMetadata(
                value=hdl_info["direction"],
                authority=EvidenceAuthority.EXPLICIT_HDL,
                source_description=f"Explicit HDL declaration in description: '{hdl_info['source_evidence']}'"
            ))
            if hdl_info["width"] is not None:
                width_cands.append(CandidateMetadata(
                    value=hdl_info["width"],
                    raw_value=hdl_info["width_str"],
                    authority=EvidenceAuthority.EXPLICIT_HDL,
                    source_description=f"Explicit HDL range/width in description: '{hdl_info['source_evidence']}'"
                ))

        bullet_info = parse_markdown_bullet_port(txt)
        if bullet_info and (not name or bullet_info["name"].lower() == name.lower()):
            dir_cands.append(CandidateMetadata(
                value=bullet_info["direction"],
                authority=EvidenceAuthority.EXPLICIT_SPEC,
                source_description=f"Markdown bullet port specification: '{bullet_info['source_evidence']}'"
            ))
            if bullet_info["width"] is not None:
                width_cands.append(CandidateMetadata(
                    value=bullet_info["width"],
                    raw_value=bullet_info["width_str"],
                    authority=EvidenceAuthority.EXPLICIT_SPEC,
                    source_description=f"Markdown bullet width: '{bullet_info['source_evidence']}'"
                ))

    # 2. Check explicit attributes (from table extraction or structured IR)
    raw_dir = r_dict.get("direction") or attrs_dict.get("direction")
    norm_dir = normalize_direction(raw_dir)
    if norm_dir:
        dir_cands.append(CandidateMetadata(
            value=norm_dir,
            raw_value=raw_dir,
            authority=EvidenceAuthority.EXPLICIT_SPEC,
            source_description=f"Explicit specification attribute direction='{raw_dir}'"
        ))

    raw_width = r_dict.get("width") or attrs_dict.get("width")
    w_int, r_str = parse_bit_range(raw_width)
    if w_int is not None:
        width_cands.append(CandidateMetadata(
            value=w_int,
            raw_value=raw_width,
            authority=EvidenceAuthority.EXPLICIT_SPEC,
            source_description=f"Explicit specification attribute width='{raw_width}'"
        ))
    elif r_str:
        width_cands.append(CandidateMetadata(
            value=r_str,
            raw_value=raw_width,
            authority=EvidenceAuthority.EXPLICIT_SPEC,
            source_description=f"Explicit specification symbolic range='{r_str}'"
        ))

    # 3. Check design_spec lines for explicit HDL port declaration matching signal name
    if spec_text and name:
        pattern = rf"(?:^|[^\w])(?:\[\d+\]\s*|\d+:\s*|[-*]\s*)?\b(input|output|inout)\s+(?:wire|reg|logic)?\s*(?:\[\s*([^:\]\s]+)\s*:\s*([^:\]\s]+)\s*\])?\s*{re.escape(name)}\b(?:\s*[,;])?"
        m = re.search(pattern, spec_text, re.IGNORECASE)
        if m:
            matched_line = m.group(0).strip()
            spec_hdl = parse_hdl_port_declaration(matched_line)
            if spec_hdl:
                dir_cands.append(CandidateMetadata(
                    value=spec_hdl["direction"],
                    authority=EvidenceAuthority.EXPLICIT_HDL,
                    source_description=f"Explicit HDL declaration in design specification: '{matched_line}'"
                ))
                if spec_hdl["width"] is not None:
                    width_cands.append(CandidateMetadata(
                        value=spec_hdl["width"],
                        raw_value=spec_hdl["width_str"],
                        authority=EvidenceAuthority.EXPLICIT_HDL,
                        source_description=f"Explicit HDL declaration in design specification: '{matched_line}'"
                    ))

    return dir_cands, width_cands


def resolve_signal_metadata(
    candidates_or_records: List[Any],
    signal_name: str = "",
    spec_text: str = "",
    role_inferred_dir: Optional[str] = None,
    role_inferred_role: Optional[str] = None
) -> Dict[str, Any]:
    """
    Resolve complete metadata (direction, width, role, conflicts) for a signal across all sources.
    
    Protocol-agnostic and design-agnostic.
    Returns:
    {
        "name": str,
        "direction": str ("input"|"output"|"inout"),
        "direction_source": str,
        "width": str (e.g. "32", "512", "1", "[31:0]"),
        "width_int": Optional[int],
        "width_source": str,
        "semantic_role": str,
        "is_conflicted": bool,
        "conflicts": List[Dict[str, Any]],
        "traceability_method": str
    }
    """
    all_dir_cands: List[CandidateMetadata] = []
    all_width_cands: List[CandidateMetadata] = []

    for item in candidates_or_records:
        if isinstance(item, CandidateMetadata):
            all_dir_cands.append(item)
        else:
            d_cands, w_cands = extract_signal_candidates_from_record(item, signal_name=signal_name, spec_text=spec_text)
            all_dir_cands.extend(d_cands)
            all_width_cands.extend(w_cands)

    # Add role-inferred direction candidate if provided (low authority: INFERRED_ROLE)
    if role_inferred_dir:
        norm_role_dir = normalize_direction(role_inferred_dir)
        if norm_role_dir:
            all_dir_cands.append(CandidateMetadata(
                value=norm_role_dir,
                authority=EvidenceAuthority.INFERRED_ROLE,
                source_description="Inferred from established protocol role contract"
            ))

    # Resolve direction
    dir_res = resolve_property_by_evidence(all_dir_cands, property_name="direction", entity_name=signal_name)
    # Resolve width
    width_res = resolve_property_by_evidence(all_width_cands, property_name="width", entity_name=signal_name)

    conflicts: List[Dict[str, Any]] = []
    if dir_res["is_conflicted"]:
        conflicts.append({
            "category": "INTERFACE",
            "property": "direction",
            "issue": dir_res["conflict_message"],
            "evidence": dir_res["evidence"]
        })
    if width_res["is_conflicted"]:
        conflicts.append({
            "category": "INTERFACE",
            "property": "width",
            "issue": width_res["conflict_message"],
            "evidence": width_res["evidence"]
        })

    # Final resolved direction
    final_dir = dir_res["resolved_value"]
    if not final_dir:
        # Fallback to input if unspecified, but record provenance as default
        final_dir = "input"
        dir_source = "unspecified_default"
    else:
        dir_source = dir_res["source"]

    # Final resolved width
    final_w_int = width_res["resolved_value"] if isinstance(width_res["resolved_value"], int) else None
    if final_w_int is not None:
        final_w_str = str(final_w_int)
        width_source = width_res["source"]
    elif isinstance(width_res["resolved_value"], str):
        final_w_str = width_res["resolved_value"]
        width_source = width_res["source"]
    else:
        # Width unspecified: do not fabricate width without noting it as default
        final_w_str = "1"
        final_w_int = 1
        width_source = "unspecified_default"

    # Semantic role
    sem_role = role_inferred_role or "data"

    is_conflicted = bool(conflicts)
    traceability_method = "resolved_evidence" if (not is_conflicted and dir_res["authority"] >= EvidenceAuthority.EXPLICIT_SPEC) else (
        "unresolved_conflict" if is_conflicted else "default_inferred"
    )

    return {
        "name": signal_name,
        "direction": final_dir,
        "direction_source": dir_source,
        "direction_authority": dir_res["authority"].value,
        "width": final_w_str,
        "width_int": final_w_int,
        "width_source": width_source,
        "width_authority": width_res["authority"].value,
        "semantic_role": sem_role,
        "is_conflicted": is_conflicted,
        "conflicts": conflicts,
        "evidence_direction": dir_res["evidence"],
        "evidence_width": width_res["evidence"],
        "traceability_method": traceability_method
    }


def compare_signal_metadata(
    sig_a: Dict[str, Any],
    sig_b: Dict[str, Any]
) -> Tuple[bool, Optional[str]]:
    """
    Compare two signal port specifications semantically.
    Ignores representational differences that normalize to the same meaning (e.g. 'in' vs 'input', '32' vs '[31:0]').
    Detects genuine direction and width conflicts.
    
    Returns (is_compatible, conflict_reason_if_incompatible).
    """
    name_a = sig_a.get("name", "")
    name_b = sig_b.get("name", "")
    if name_a and name_b and name_a.lower() != name_b.lower():
        return False, f"Signal names differ: '{name_a}' vs '{name_b}'"

    # Direction comparison
    dir_a = normalize_direction(sig_a.get("direction"))
    dir_b = normalize_direction(sig_b.get("direction"))
    if dir_a and dir_b and dir_a != dir_b:
        return False, f"Direction conflict on '{name_a or name_b}': '{dir_a}' vs '{dir_b}'"

    # Width comparison
    w_a_raw = sig_a.get("width")
    w_b_raw = sig_b.get("width")
    wa_int, wa_range = parse_bit_range(w_a_raw)
    wb_int, wb_range = parse_bit_range(w_b_raw)

    if wa_int is not None and wb_int is not None:
        if wa_int != wb_int:
            return False, f"Width conflict on '{name_a or name_b}': width {wa_int} vs {wb_int}"
    elif wa_range and wb_range:
        if wa_range.lower().replace(" ", "") != wb_range.lower().replace(" ", ""):
            return False, f"Bit range conflict on '{name_a or name_b}': '{wa_range}' vs '{wb_range}'"

    return True, None
