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
    known_components: Optional[List[str]] = None,
    known_interfaces: Optional[List[str]] = None,
) -> bool:
    """
    Validate whether a candidate is a plausible physical/logical hardware signal.
    Disqualifies:
    - Empty, numeric, or punctuation-only strings
    - Common non-hardware English stop words (unless supported by explicit direction & width)
    - Structural table/header words ('name', 'signal', 'port', 'pin', 'width', 'direction', etc.)
    - Known component or module names (narrative/module entities)
    - Known interface or bus names
    - Entities whose type is 'component', 'module', 'interface', 'channel', 'subsystem', 'block', 'unit', 'register', 'opcode', etc.
    - Entities with interconnect role or narrative relationship labels
    - Multi-word prose without hardware signal indicators
    """
    if not name or not isinstance(name, str):
        return False
    clean = name.strip("` \t\r\n'\"")
    if not clean or clean.isdigit():
        return False
    if not re.search(r"[A-Za-z]", clean):
        return False

    clean_lower = clean.lower()

    # Reserved header / structural words
    if clean_lower in (
        "name", "signal", "port", "pin", "width", "direction",
        "signal name", "port name", "signal_name", "port_name",
        "description", "protocol/description", "type", "clock/reset",
        "interface", "interfaces", "module", "modules", "component", "components",
        "value", "values"
    ):
        return False

    attrs = attributes or {}

    # Disqualify if entity type is component, module, interface, register, opcode, etc.
    ent_type = str(attrs.get("type") or attrs.get("entity_type") or "").strip().lower()
    if ent_type in (
        "component", "module", "subsystem", "block", "unit", "controller",
        "top_level_design", "interface", "channel", "bus", "relationship",
        "module_relationship", "interface_relationship", "register", "reg",
        "opcode", "instruction", "instruction_format", "memory_region"
    ):
        return False

    # Disqualify narrative interconnect roles
    role = str(attrs.get("role") or "").strip().lower()
    if role in ("interconnect", "relationship", "communication"):
        return False

    # Disqualify if matching known components
    if known_components:
        comp_set = {str(c).strip().lower() for c in known_components if c}
        if clean_lower in comp_set:
            return False

    # Disqualify if matching known interfaces
    if known_interfaces:
        iface_set = {str(i).strip().lower() for i in known_interfaces if i}
        if clean_lower in iface_set:
            return False

    # Disqualify narrative relationship labels without explicit signal evidence
    # e.g. "X_Y_interface" or "X_connects_to_Y"
    if ("_interface" in clean_lower or "_connects_to_" in clean_lower or "_sends_to_" in clean_lower or "_returns_to_" in clean_lower):
        if ent_type not in ("signal", "port"):
            return False

    # Disqualify multi-word prose (> 3 words) without signal evidence
    words = clean.split()
    if len(words) > 3 and ent_type not in ("signal", "port"):
        return False

    if clean_lower in NON_HARDWARE_WORDS:
        has_dir = str(attrs.get("direction", "")).lower() in {"in", "out", "inout", "input", "output"}
        has_width = is_explicit_value(attrs.get("width"))
        return has_dir and has_width

    return True


COMMON_FSM_STATES: Set[str] = {
    "idle", "counting", "done", "reset", "wait", "waiting", "active", "init", "initial",
    "ready", "load", "loading", "compute", "computing", "writeback", "stall", "stalled",
    "error", "finish", "finished", "stop", "stopped", "pause", "paused", "exec", "execute",
    "fetch", "decode", "st_idle", "st_counting", "st_done", "s_idle", "s_counting", "s_done",
    "state_idle", "state_counting", "state_done"
}

HARDWARE_COMPONENT_KEYWORDS: Set[str] = {
    "controller", "unit", "engine", "core", "subsystem", "module", "block",
    "processor", "generator", "synchronizer", "slice", "datapath", "system",
    "accelerator", "bridge", "interconnect", "arbiter", "adapter", "crossbar",
    # Multi-word architectural component terms
    "scheduler", "network", "fabric", "hierarchy", "elements", "element",
    "interface", "memory", "cache", "switch", "router", "bus", "fifo", "queue",
    "decoder", "encoder", "divider", "distributor", "dispatcher", "fetcher",
    "register", "buffer", "pipeline", "multiplexer", "mux", "demux", "noc"
}

PHYSICAL_SIGNAL_KEYWORDS: Set[str] = {
    "clk", "clock", "clk_i", "clk_in", "sys_clk", "rst", "rst_n", "reset", "reset_n",
    "rst_b", "arst", "arst_n", "start", "busy", "done", "valid", "ready", "enable",
    "en", "err", "error", "ack", "req", "count", "counter", "data", "addr", "wdata",
    "rdata", "we", "re", "strb", "wstrb", "resp", "bresp", "rresp", "last", "rlast", "wlast"
}


def is_fsm_state_name(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    known_states: Optional[Set[str]] = None,
) -> bool:
    """
    Check if a candidate identifier represents an FSM state rather than a hardware component.
    FSM states must never be classified as EFSComponent or EFSInterface endpoints.
    """
    if not name or not isinstance(name, str):
        return False
    clean = name.strip("` \t\r\n'\"")
    clean_lower = clean.lower()

    if not clean or clean.isdigit():
        return False

    # Never treat actual component types as states
    if any(k in clean_lower for k in HARDWARE_COMPONENT_KEYWORDS):
        return False

    if known_states and (clean_lower in {s.lower() for s in known_states} or clean.upper() in known_states):
        return True

    attrs = attributes or {}
    c_type = str(attrs.get("type") or attrs.get("candidate_type") or "").strip().lower()
    if c_type in ("state", "fsm_state", "fsm_states"):
        return True

    desc = str(attrs.get("description") or "").lower()
    if "fsm state" in desc or "state machine state" in desc or "state:" in desc or "state " in desc:
        return True

    # Standard FSM state tokens
    if clean_lower in COMMON_FSM_STATES:
        return True

    # State prefix patterns like ST_IDLE, S_COUNTING
    if re.fullmatch(r"(?:st|s|state)_[a-z0-9_]+", clean_lower):
        return True

    # Uppercase state name matching e.g. IDLE, COUNTING, DONE, EXEC_OP
    if clean.isupper() and len(clean.split()) == 1 and clean_lower in COMMON_FSM_STATES:
        return True

    return False


def is_fsm_transition_or_condition(text: str) -> bool:
    """
    Check if a string represents an FSM condition or timing expression rather than a hardware entity.
    """
    if not text or not isinstance(text, str):
        return False
    clean = text.strip()
    clean_lower = clean.lower()

    # Comparison / boolean operators e.g. start == 1, count == 10, count >= 10
    if re.search(r"(?:==|!=|<=|>=|<|>|&&|\|\||:=)", clean):
        return True

    # Timing / clock cycle phrases e.g. next clock cycle, after 1 cycle
    if re.search(r"\b(?:next\s+clock\s+cycle|clock\s+cycle|one\s+cycle|\d+\s+cycles?|posedge\b|negedge\b)\b", clean_lower):
        return True

    # Assignment or condition expressions e.g. "start is 1", "when start asserted"
    if clean_lower.startswith("when ") or clean_lower.startswith("after ") or clean_lower.startswith("on "):
        return True

    return False


def is_physical_signal_name(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    known_signals: Optional[Set[str]] = None,
) -> bool:
    """
    Check if a candidate name is a physical signal, port, clock, or reset.
    """
    if not name or not isinstance(name, str):
        return False
    clean = name.strip("` \t\r\n'\"")
    clean_lower = clean.lower()

    if not clean:
        return False

    # Disqualify internal generated IDs from being physical signal names
    if re.match(r"^SIG_[0-9a-fA-F]{8}$", clean, re.IGNORECASE) or clean.startswith("SIG_"):
        return False

    attrs = attributes or {}
    c_type = str(attrs.get("type") or attrs.get("candidate_type") or "").strip().lower()
    if c_type in ("signal", "port", "clock", "reset"):
        return True

    # Bit slicing e.g. count[7:0], data[31:0]
    if re.search(r"\[\s*\d+\s*:\s*\d+\s*\]", clean):
        return True

    # Explicit direction attribute
    dir_val = str(attrs.get("direction", "")).lower()
    if dir_val in ("input", "output", "inout", "in", "out"):
        return True

    # Check evidence / description / mention for hardware signal phrasing
    desc_text = " ".join([
        str(attrs.get("description", "")),
        str(attrs.get("mention", "")),
        str(attrs.get("original_text", "")),
        str(attrs.get("statement", ""))
    ]).lower()
    if any(k in desc_text for k in ("output signal", "input signal", "control signal", "status signal", "port controlling", "output port", "input port")):
        return True

    if known_signals and clean_lower in {s.lower() for s in known_signals}:
        return True

    # Strip bit slice for base name check
    base_name = re.sub(r"\[.*?\]", "", clean_lower).strip()
    if base_name in PHYSICAL_SIGNAL_KEYWORDS:
        return True

    # Common signal prefix/suffix patterns across domains (e.g. clk, rst, data, valid, ready, light, led, pwm, trig)
    if re.search(r"_(?:clk|rst|rst_n|valid|ready|en|enable|req|ack|data|addr|out|in|light|led|pin|trigger|trig|pwm|val|err|error|int|irq|tx|rx|busy|done|start|strobe|strb|sel)$", clean_lower):
        return True

    return False


DISQUALIFIED_COMPONENT_NAMES: Set[str] = {
    "module name", "module_name", "design name", "design_name",
    "component name", "component_name", "block name", "block_name",
    "unit name", "unit_name", "top module", "top_module",
    "target module", "target_module", "target component", "target_component",
    "specification title", "project name", "submodule name", "unnamed_submodule"
}


def is_hardware_component_candidate(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    known_components: Optional[Set[str]] = None,
) -> bool:
    """
    Determine if a candidate identifier represents a legitimate hardware component/module.
    Disqualifies: signals, clocks, resets, FSM states, transition conditions, opcodes,
    register fields, timing expressions, natural language actions/descriptions,
    and meta-label placeholders (e.g. 'Module Name').
    """
    if not name or not isinstance(name, str):
        return False
    clean = name.strip("` \t\r\n'\"")
    clean_lower = clean.lower()

    if not clean or clean.isdigit():
        return False

    # Disqualify meta-label table headers and placeholders (e.g. 'Module Name')
    if clean_lower in DISQUALIFIED_COMPONENT_NAMES:
        return False

    attrs = attributes or {}

    # Disqualify FSM states and state machine representations
    if is_fsm_state_name(clean, attrs) or clean_lower in ("fsm", "state_machine", "finite_state_machine", "finite state machine"):
        return False

    # Disqualify conditions and timing expressions
    if is_fsm_transition_or_condition(clean):
        return False

    # Disqualify physical signals, ports, clocks, resets
    if is_physical_signal_name(clean, attrs):
        return False

    # Disqualify bit slicing
    if re.search(r"\[.*?\]", clean):
        return False

    # Disqualify common non-hardware stop words
    if clean_lower in NON_HARDWARE_WORDS:
        return False

    # Disqualify prose phrases (> 4 words) unless explicit component keyword
    words = clean.split()
    if len(words) > 4 and not any(k in clean_lower for k in HARDWARE_COMPONENT_KEYWORDS):
        return False

    # Disqualify opcodes and registers
    c_type = str(attrs.get("type") or attrs.get("candidate_type") or "").strip().lower()
    if c_type in ("opcode", "instruction", "instruction_format", "register_field", "field"):
        return False

    clean_no_parens = re.sub(r"\s*\(.*?\)", "", clean_lower).strip()
    if known_components and (clean_lower in {c.lower() for c in known_components} or clean_no_parens in {c.lower() for c in known_components}):
        return True

    # Component keyword presence
    if any(k in clean_lower for k in HARDWARE_COMPONENT_KEYWORDS) or any(k in clean_no_parens for k in HARDWARE_COMPONENT_KEYWORDS):
        return True

    # Attributes check
    if c_type in ("component", "module", "subsystem", "block", "unit", "architectural_component", "core_component", "processing_element"):
        return True

    # Fallback: snake_case or PascalCase identifier with length >= 3
    clean_ident = re.sub(r"\s*\(.*?\)", "", clean).replace(" ", "_")
    if len(clean_ident) >= 3 and re.match(r"^[A-Za-z][A-Za-z0-9_]*$", clean_ident):
        return True

    return False


def is_valid_interface_candidate(
    source: str,
    target: str,
    attributes: Optional[Dict[str, Any]] = None,
    known_components: Optional[Set[str]] = None,
    known_signals: Optional[Set[str]] = None,
    known_states: Optional[Set[str]] = None,
) -> Tuple[bool, str]:
    """
    Hard Semantic Boundary for EFSInterface construction.
    Only construct EFSInterface when ALL of the following are true:
    1. Source resolves to a valid hardware component/entity.
    2. Destination resolves to a valid hardware component/entity (or memory).
    3. Neither endpoint is a signal, clock, reset, FSM state, condition, opcode, or action phrase.
    4. The relationship is not merely signal ownership (e.g. clk -> controller, controller -> busy).
    5. The relationship is not an FSM transition (e.g. IDLE -> COUNTING).
    6. The relationship is not a control-flow condition or timing expression.
    7. The relationship is not a component description.
    8. The relationship is not a register/instruction relationship.
    """
    if not source or not target:
        return False, "Missing source or target entity."

    s_clean = source.strip("` \t\r\n'\"")
    t_clean = target.strip("` \t\r\n'\"")

    if not s_clean or not t_clean:
        return False, "Empty source or target entity."

    attrs = attributes or {}
    if s_clean.lower() == t_clean.lower():
        is_peer = attrs.get("is_peer_to_peer") or attrs.get("relation") == "communicates_with" or "mpe" in s_clean.lower() or "element" in s_clean.lower() or "core" in s_clean.lower()
        if not is_peer:
            return False, "Source and destination are identical."

    # Check for FSM transition endpoints
    if is_fsm_state_name(s_clean, attributes, known_states) or is_fsm_state_name(t_clean, attributes, known_states):
        return False, f"FSM state transition ({s_clean} -> {t_clean}) must be preserved in EFSFSM.transitions, not EFSInterface."

    # Check for FSM conditions / timing expressions
    if is_fsm_transition_or_condition(s_clean) or is_fsm_transition_or_condition(t_clean):
        return False, f"Control condition / timing expression ({s_clean} -> {t_clean}) is not an interface."

    # Check for signal ownership / port associations
    s_is_sig = is_physical_signal_name(s_clean, attributes, known_signals)
    t_is_sig = is_physical_signal_name(t_clean, attributes, known_signals)

    if s_is_sig and t_is_sig:
        return False, f"Signal-to-signal association ({s_clean} -> {t_clean}) is not an interface."

    if s_is_sig or t_is_sig:
        sig_name = s_clean if s_is_sig else t_clean
        comp_name = t_clean if s_is_sig else s_clean
        dir_str = "input" if s_is_sig else "output"
        return False, f"Signal/port association ({sig_name} is {dir_str} of {comp_name}) must remain EFSSignal, not EFSInterface."

    # Check component descriptions and containment
    attrs = attributes or {}
    rel = str(attrs.get("relation") or "").lower()
    if rel in ("contains", "has", "includes", "defines", "instantiates", "manages", "orchestrates", "controls", "description", "is_description"):
        return False, f"Structural containment or hierarchy relationship ('{s_clean} {rel} {t_clean}') is not an interface."

    # Word-count guard: reject prose targets (> 5 tokens) UNLESS they are registered known components
    # or contain hardware architecture keywords. This allows multi-word names like
    # "Matrix Processing Elements (MPEs)" or "Shared Memory Hierarchy" to pass.
    t_words = t_clean.split()
    s_words = s_clean.split()
    _known_comp_lower = {c.lower() for c in (known_components or set())}
    t_in_known = t_clean.lower() in _known_comp_lower or re.sub(r"\s*\(.*?\)", "", t_clean).strip().lower() in _known_comp_lower
    s_in_known = s_clean.lower() in _known_comp_lower or re.sub(r"\s*\(.*?\)", "", s_clean).strip().lower() in _known_comp_lower
    t_has_hw_kw = any(k in t_clean.lower() for k in HARDWARE_COMPONENT_KEYWORDS)
    s_has_hw_kw = any(k in s_clean.lower() for k in HARDWARE_COMPONENT_KEYWORDS)
    if len(t_words) > 5 and not t_in_known and not t_has_hw_kw:
        return False, f"Target '{t_clean}' appears to be a prose description, not a hardware component entity."
    if len(s_words) > 5 and not s_in_known and not s_has_hw_kw:
        return False, f"Source '{s_clean}' appears to be a prose description, not a hardware component entity."

    if s_clean.lower() in ("fsm", "state_machine", "finite_state_machine") or t_clean.lower() in ("fsm", "state_machine", "finite_state_machine"):
        return False, f"FSM ({s_clean} -> {t_clean}) is an internal state machine representation, not an inter-module interface."

    # Source and Destination must be valid hardware component candidates
    if not is_hardware_component_candidate(s_clean, attributes, known_components):
        return False, f"Source '{s_clean}' is not a valid hardware component entity."

    # Destination may be a hardware component or memory region
    t_is_comp = is_hardware_component_candidate(t_clean, attributes, known_components)
    t_is_mem = any(k in t_clean.lower() for k in ("memory", "sram", "dram", "bram", "rom", "ram", "storage", "buffer", "cache"))
    if not (t_is_comp or t_is_mem):
        return False, f"Destination '{t_clean}' is not a valid hardware component or memory entity."

    return True, ""


def is_timed_condition(condition_text: str) -> Tuple[bool, Optional[int]]:
    """
    Check if a condition string expresses a timing or cycle count constraint.
    E.g. '5 clock cycles', 'after 10 cycles', 'remains on for 3 clock cycles', 'counter reaches 4'.
    Returns (True, cycle_count) or (False, None).
    """
    if not condition_text or not isinstance(condition_text, str):
        return False, None
    c_lower = condition_text.strip().lower()
    # Direct cycle count regex e.g. '5 clock cycles', '10 cycles'
    m = re.search(r"\b(\d+)\s*(?:clock\s+)?cycles?\b", c_lower)
    if m:
        return True, int(m.group(1))
    m2 = re.search(r"(?:counter|timer|count)\s*(?:>=|==|reaches|=)\s*(\d+)", c_lower)
    if m2:
        return True, int(m2.group(1)) + 1
    if any(k in c_lower for k in ("timer_done", "timeout", "cycle count reached", "delay elapsed")):
        return True, None
    return False, None


# Re-export generic signal normalization and resolution utilities
from core.efs_ir.signal_normalization import (
    normalize_direction,
    parse_bit_range,
    parse_hdl_port_declaration,
    parse_markdown_bullet_port,
    resolve_signal_metadata,
    compare_signal_metadata,
    EvidenceAuthority,
)

