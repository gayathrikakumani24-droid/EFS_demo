"""
Deterministic Heuristic Extractor module (Step 3 & Step 4).

Extracts hardware conditions, timing rules, edge events, reset rules, valid/ready handshakes,
and register definitions deterministically without using an LLM.
Calculates transparent confidence scores for every extraction path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from config import CONFIG
from utils.models import new_id


@dataclass
class HeuristicExtractionResult:
    requirements: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    timing: List[Dict[str, Any]] = field(default_factory=list)
    method: str = "heuristic"
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requirements": self.requirements,
            "entities": self.entities,
            "relationships": self.relationships,
            "conditions": self.conditions,
            "events": self.events,
            "timing": self.timing,
            "method": self.method,
            "confidence": self.confidence,
        }


# ------------------------------------------------------------------
# Regex Patterns for Deterministic Heuristic Extraction
# ------------------------------------------------------------------

# 1. Handshake Pattern: "If VALID and READY are high..." or "When AWVALID and AWREADY..."
_HANDSHAKE_RE = re.compile(
    r"\b(?:if|when|whenever)\s+([A-Z0-9_]+VALID)\s+(?:and|&{1,2})\s+([A-Z0-9_]+READY)\s+(?:are|is)?\s*(?:high|asserted|1)\s*,?\s*(?:a|an)?\s*(.+?)(?:\.|$)",
    re.IGNORECASE
)

# 2. Timing Pattern: "must be returned within 16 cycles" or "latency of 4 cycles"
_TIMING_RE = re.compile(
    r"\b([A-Za-z0-9_ ]{3,30})\s+(?:must|shall|should)\s+be\s+(?:returned|asserted|completed|processed|issued)\s+within\s+(\d+)\s*(cycles?|ns|us|ms|clock cycles?)\b",
    re.IGNORECASE
)

# 3. Clock Edge Event Pattern: "On the (rising|falling) edge of CLK, the register captures DATA."
_EDGE_EVENT_RE = re.compile(
    r"\b(?:on|upon)\s+the\s+(rising|falling|pos|neg)\s*edge\s+of\s+([A-Z0-9_]+)\s*,?\s*(?:the\s+)?(.+?)(?:\.|$)",
    re.IGNORECASE
)

# 4. Reset Rule Pattern: "When RESETn is low, all registers are cleared."
_RESET_RE = re.compile(
    r"\b(?:when|whenever|upon)\s+([A-Z0-9_]*RESET[A-Z0-9_]*)\s+is\s+(low|high|asserted|deasserted|0|1)\s*,?\s*(?:the\s+)?(.+?)(?:\.|$)",
    re.IGNORECASE
)

# 5. Simple Conditional Behavior: "If <cond>, then <action>."
_CONDITIONAL_ACTION_RE = re.compile(
    r"\bif\s+([A-Z0-9_\s==&\|!]+?)\s*,?\s*(?:then\s+)?(?:the\s+)?([A-Z0-9_]+)\s+(?:shall|must|will|is)\s+(asserted|deasserted|set|cleared|updated|transmitted)\b",
    re.IGNORECASE
)

# 6. Explicit Register Format: "Register REG_NAME at offset 0x04 (R/W, 32 bits)"
_REG_DECL_RE = re.compile(
    r"\b(?:register|reg)\s+([A-Z0-9_]+)\s*(?:at|offset)?\s*(0x[0-9A-Fa-f]+)?\s*\(?\s*(R/W|RO|WO|RW)?\s*,?\s*(\d+)?\s*(?:bits?|b)?\)?",
    re.IGNORECASE
)

# 7. Action Signal Assertion Pattern: "shall assert ACK when REQ is high"
_ASSERT_WHEN_RE = re.compile(
    r"\b(?:the\s+system\s+)?(?:shall|must|will)?\s*(assert|deassert|set|clear|drive)\s+([A-Z0-9_]+)\s+(?:when|if|whenever)\s+(.+?)(?:\.|$)",
    re.IGNORECASE
)

# 8. RTL Module & Port Declarations
_RTL_MODULE_RE = re.compile(r"\bmodule\s+([A-Za-z0-9_]+)\s*(?:#\s*\((.*?)\)\s*)?\((.*?)\);", re.DOTALL)
_RTL_PORT_RE = re.compile(
    r"\b(input|output|inout)\s+(?:wire|reg|logic)?\s*(?:\[\s*([^:]+?)\s*:\s*([^\]]+?)\s*\])?\s*([A-Za-z0-9_]+)\b(?:\s*,?\s*//\s*([^\n\r]*))?",
    re.IGNORECASE
)
_RTL_PARAM_RE = re.compile(r"\bparameter\s+([A-Za-z0-9_]+)\s*=\s*([^;,\n]+)", re.IGNORECASE)

# 9. Instruction Format & Field Bit Allocation
_INST_FORMAT_HEADER_RE = re.compile(r"(?:instruction\s+format\s*(?:\((\d+)\s*bits?\))?|each\s+instruction\s+is\s+a\s+(\d+)-bit\s+word)", re.IGNORECASE)
_INST_FIELD_RE = re.compile(r"^\s*[-*]?\s*([A-Za-z0-9_\s/\-]+?)\s*\(\s*(\d+)\s*bits?\s*\)\s*:\s*(.*)$", re.IGNORECASE)

# 10. Opcode & Instruction Lines
_OPCODE_LINE_RE = re.compile(r"^\s*([01]{4,32}|0x[0-9A-Fa-f]+|\b\d{1,4}\b)\s*:\s*([A-Za-z0-9_]+)\s*(?:\((.*?)\)|[-–—]\s*(.*?))?$", re.IGNORECASE)
_MNEMONIC_LINE_RE = re.compile(r"^\s*[-*]?\s*\b([A-Za-z0-9_]{2,20})\b\s*(?:\[[^\]]+\])?\s*:\s*(?:opcode\s*(0x[0-9A-Fa-f]+|[01]{4,32}|\d+)\s*,?\s*)?(.*)$", re.IGNORECASE)

# 11. Architectural Components
_COMPONENT_DEF_RE = re.compile(r"^\s*(?:[-*]|\d+\s+)?([A-Za-z0-9_\(\)][A-Za-z0-9_\s\(\)\-]{2,50}?)\s*:\s*(.{10,250})$")

# 12. Performance Goals
_PERF_METRIC_RE = re.compile(r"\b(maximum\s+clock\s+frequency|clock\s+frequency|frequency|throughput|power\s+consumption|power|bandwidth|capacity|matrix multiplies throughput)\b\s*[:=]\s*([^\n\r]+)", re.IGNORECASE)

# 13. Timing Complexity
_TIMING_COMPLEXITY_RE = re.compile(r"^\s*(?:[-*]|\d+\s+)?([A-Za-z0-9_/\s\-]+?)\s*:\s*(O\([^\)]+\)\s*cycles?|within\s+\d+\s*cycles?|\d+\s*cycles?)\b", re.IGNORECASE)

# 14. Memory Architecture & Hierarchy
_MEMORY_LINE_RE = re.compile(r"^\s*(?:[-*]|\d+\s+)?([A-Za-z0-9_\s]{2,35}?(?:Cache|SRAM|DRAM|Buffer|Memory|Hierarchy))\s*:\s*(.*)$", re.IGNORECASE)

# 15. External Interfaces
_INTERFACE_LINE_RE = re.compile(r"^\s*(?:[-*]|\d+\s+)?([A-Za-z0-9_\s]{3,35}?(?:Interface|Controller|Signals?))\s*:\s*(.*)$", re.IGNORECASE)

# 16. Error Handling
_ERROR_LINE_RE = re.compile(r"^\s*[-*]?\s*([A-Za-z0-9_]{3,25}_ERR|[A-Za-z0-9_]{3,25}_ERROR|(?:Detection|reporting|monitoring|checking|overflow|underflow|singular)[A-Za-z0-9_\s]*?)\s*(?::\s*(.*))?$", re.IGNORECASE)


def extract_heuristically(text: str, chunk_id: str = "", doc_id: str = "") -> HeuristicExtractionResult:
    """
    Apply deterministic extraction rules to text.
    Computes a transparent confidence score between 0.0 and 1.0.
    """
    if not text or not text.strip():
        return HeuristicExtractionResult(confidence=0.0)

    requirements: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    relationships: List[Dict[str, Any]] = []
    conditions: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    timing: List[Dict[str, Any]] = []

    pattern_hits = 0

    # 1. Valid/Ready Handshake
    for m in _HANDSHAKE_RE.finditer(text):
        valid_sig, ready_sig, action = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        pattern_hits += 1
        cond_str = f"{valid_sig} && {ready_sig}"
        conditions.append({
            "condition": cond_str,
            "action": action,
            "type": "HANDSHAKE"
        })
        entities.extend([
            {"name": valid_sig, "type": "signal", "description": "Handshake valid signal"},
            {"name": ready_sig, "type": "signal", "description": "Handshake ready signal"},
        ])
        relationships.append({
            "source": valid_sig,
            "relation": "HANDSHAKES_WITH",
            "target": ready_sig,
        })
        requirements.append({
            "req_id": new_id("req"),
            "statement": f"Transfer occurs when {cond_str} is high: {action}",
            "type": "BEHAVIOR",
            "condition": cond_str,
            "action": action,
            "confidence": 0.95,
            "chunk_id": chunk_id,
            "doc_id": doc_id,
        })

    # 2. Timing Constraint
    for m in _TIMING_RE.finditer(text):
        target_obj, val_str, unit = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        pattern_hits += 1
        val_int = int(val_str)
        timing_item = {
            "constraint": target_obj,
            "operator": "<=",
            "value": val_int,
            "unit": unit,
        }
        timing.append(timing_item)
        requirements.append({
            "req_id": new_id("req"),
            "statement": f"{target_obj} must be completed within {val_int} {unit}",
            "type": "TIMING",
            "timing": f"<={val_int} {unit}",
            "value": val_int,
            "unit": unit,
            "confidence": 0.95,
            "chunk_id": chunk_id,
            "doc_id": doc_id,
        })

    # 3. Clock Edge Events
    for m in _EDGE_EVENT_RE.finditer(text):
        edge_type, clk_sig, action = m.group(1).strip().lower(), m.group(2).strip(), m.group(3).strip()
        pattern_hits += 1
        edge_name = "rising_edge" if "ris" in edge_type or "pos" in edge_type else "falling_edge"
        event_str = f"{edge_name}({clk_sig})"
        events.append({
            "event": event_str,
            "action": action
        })
        entities.append({"name": clk_sig, "type": "clock", "description": "Clock signal"})
        requirements.append({
            "req_id": new_id("req"),
            "statement": f"On {event_str}: {action}",
            "type": "EVENT",
            "event": event_str,
            "action": action,
            "confidence": 0.90,
            "chunk_id": chunk_id,
            "doc_id": doc_id,
        })

    # 4. Reset Rules
    for m in _RESET_RE.finditer(text):
        rst_sig, rst_val, action = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        pattern_hits += 1
        cond_str = f"{rst_sig} == {rst_val}"
        conditions.append({
            "condition": cond_str,
            "action": action,
            "type": "RESET"
        })
        if rst_sig:
            entities.append({"name": rst_sig, "type": "reset", "description": "Reset signal"})
        requirements.append({
            "req_id": new_id("req"),
            "statement": f"When {cond_str}: {action}",
            "type": "BEHAVIOR",
            "condition": cond_str,
            "action": action,
            "confidence": 0.90,
            "chunk_id": chunk_id,
            "doc_id": doc_id,
        })

    # 5. Conditional Action
    for m in _CONDITIONAL_ACTION_RE.finditer(text):
        cond, target_sig, action = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        pattern_hits += 1
        entities.append({"name": target_sig, "type": "signal", "description": "Target signal"})
        requirements.append({
            "req_id": new_id("req"),
            "statement": f"If {cond}, then {target_sig} is {action}",
            "type": "CONSTRAINT",
            "condition": cond,
            "subject": target_sig,
            "action": action,
            "confidence": 0.85,
            "chunk_id": chunk_id,
            "doc_id": doc_id,
        })

    # 6. Action Signal Assertion Pattern
    for m in _ASSERT_WHEN_RE.finditer(text):
        act_verb, sig_name, cond_expr = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        pattern_hits += 1
        entities.append({"name": sig_name, "type": "signal", "description": "Signal entity"})
        requirements.append({
            "req_id": new_id("req"),
            "statement": f"Assert {sig_name} when {cond_expr}",
            "type": "BEHAVIOR",
            "condition": cond_expr,
            "subject": sig_name,
            "action": act_verb.upper(),
            "confidence": 0.90,
            "chunk_id": chunk_id,
            "doc_id": doc_id,
        })

    # 6b. Functional / Algorithmic Behavior ("When <cond>, <subj> computes/executes <action>")
    for m in re.finditer(r"\b(?:when|upon|on)\s+([^,\n]+),\s*(?:the\s+)?([A-Za-z0-9_\s]{2,30}?)\s+(computes?|executes?|performs?|generates?|processes?)\s+([^.\n]+)", text, re.IGNORECASE):
        cond = m.group(1).strip()
        subj = m.group(2).strip()
        verb = m.group(3).strip()
        result = m.group(4).strip()
        pattern_hits += 1
        requirements.append({
            "req_id": new_id("req"),
            "statement": f"When {cond}, {subj} {verb} {result}",
            "type": "BEHAVIOR",
            "subject": subj,
            "condition": cond,
            "action": f"{verb} {result}",
            "confidence": 0.90,
            "chunk_id": chunk_id,
            "doc_id": doc_id,
        })

    # 7. Explicit Register Declarations
    for m in _REG_DECL_RE.finditer(text):
        reg_name, offset, access, width = m.group(1), m.group(2), m.group(3), m.group(4)
        pattern_hits += 1
        attrs = {}
        if offset: attrs["offset"] = offset
        if access: attrs["access"] = access
        if width: attrs["width"] = int(width)
        entities.append({
            "name": reg_name,
            "type": "register",
            "attributes": attrs
        })

    # 7b. Register Ranges (e.g. Matrix Registers: M0-M7 (8 registers in total), General Purpose Registers: R0-R31)
    for m in re.finditer(r"([A-Za-z0-9_]+)\s*Registers?\s*:\s*([A-Za-z0-9_]+)\s*-\s*([A-Za-z0-9_]+)", text, re.I):
        cat_name = m.group(1).strip()
        start_name = m.group(2).strip()
        end_name = m.group(3).strip()
        m_start = re.match(r"^([A-Za-z]+)(\d+)$", start_name)
        m_end = re.match(r"^([A-Za-z]+)(\d+)$", end_name)
        if m_start and m_end and m_start.group(1).upper() == m_end.group(1).upper():
            prefix = m_start.group(1).upper()
            s_idx = int(m_start.group(2))
            e_idx = int(m_end.group(2))
            for r_num in range(s_idx, e_idx + 1):
                r_id = f"{prefix}{r_num}"
                pattern_hits += 1
                entities.append({
                    "name": r_id,
                    "type": "register",
                    "description": f"{cat_name} Register {r_id}",
                    "attributes": {"access": "RW", "category": f"{cat_name} Registers"}
                })
                requirements.append({
                    "req_id": new_id("req"),
                    "statement": f"Register {r_id} ({cat_name} Register range {start_name}-{end_name})",
                    "type": "REGISTER",
                    "subject": r_id,
                    "action": "READ_WRITE",
                    "confidence": 0.95,
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                })
            # Also preserve the named register range category itself
            group_name = f"{cat_name} Registers {start_name}-{end_name}"
            pattern_hits += 1
            entities.append({
                "name": group_name,
                "type": "register",
                "description": f"{cat_name} Registers {start_name}-{end_name} ({e_idx - s_idx + 1} registers in total)",
                "attributes": {"access": "RW", "category": f"{cat_name} Registers", "range": f"{start_name}-{end_name}"}
            })
            requirements.append({
                "req_id": new_id("req"),
                "statement": f"{cat_name} Registers: {start_name}-{end_name}",
                "type": "REGISTER",
                "subject": group_name,
                "action": "READ_WRITE",
                "confidence": 0.95,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
            })

    # 7c. Register Categories without immediate addresses (e.g. Configuration Registers: Control matrix engine parameters)
    for m in re.finditer(r"^[-*]?\s*([A-Za-z0-9_\s]{2,30}\s*Registers?)\s*:\s*(.+)$", text, re.M | re.I):
        cat_name = m.group(1).strip()
        cat_desc = m.group(2).strip()
        if cat_name.lower() not in ("register addresses", "registers", "external interfaces"):
            pattern_hits += 1
            entities.append({
                "name": cat_name,
                "type": "register",
                "description": cat_desc,
                "attributes": {"access": "RW", "description": cat_desc}
            })
            requirements.append({
                "req_id": new_id("req"),
                "statement": f"{cat_name}: {cat_desc}",
                "type": "REGISTER",
                "subject": cat_name,
                "action": "READ_WRITE",
                "confidence": 0.90,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
            })

    # 8. RTL Module & Port Declarations
    for mod_m in _RTL_MODULE_RE.finditer(text):
        mod_name = mod_m.group(1).strip()
        params_text = mod_m.group(2) or ""
        ports_text = mod_m.group(3) or ""
        pattern_hits += 2

        entities.append({
            "name": mod_name,
            "type": "component",
            "description": f"RTL module declaration {mod_name}"
        })

        # Check for interface groupings in comments inside ports list
        current_iface_name = None
        iface_signals: Dict[str, List[str]] = {}

        for line in ports_text.split('\n'):
            line_s = line.strip()
            if line_s.startswith("//") and ("interface" in line_s.lower() or "bus" in line_s.lower()):
                current_iface_name = line_s.lstrip("/").strip()
                iface_signals.setdefault(current_iface_name, [])
                continue

            port_m = _RTL_PORT_RE.search(line_s)
            if port_m:
                direction = port_m.group(1).lower()
                msb_s = port_m.group(2)
                lsb_s = port_m.group(3)
                port_name = port_m.group(4).strip()
                comment = (port_m.group(5) or "").strip()

                try:
                    width = (int(msb_s) - int(lsb_s) + 1) if (msb_s and lsb_s) else ("1" if not msb_s else msb_s)
                except Exception:
                    width = f"[{msb_s}:{lsb_s}]" if (msb_s and lsb_s) else "1"

                pn_l = port_name.lower()
                role = "data"
                if pn_l in ("clk", "clock") or "clk" in pn_l:
                    role = "clock"
                elif "rst" in pn_l or "reset" in pn_l:
                    role = "reset"
                elif pn_l in ("start", "busy", "done", "error"):
                    role = "control"
                elif "addr" in pn_l:
                    role = "address"
                elif "valid" in pn_l or "ready" in pn_l:
                    role = "handshake"

                sig_type = "clock" if role == "clock" else ("reset" if role == "reset" else "signal")

                entities.append({
                    "name": port_name,
                    "type": sig_type,
                    "description": comment or f"RTL {direction} port {port_name}",
                    "attributes": {
                        "direction": direction,
                        "width": str(width),
                        "semantic_role": role,
                        "owner": mod_name
                    }
                })

                if current_iface_name:
                    iface_signals[current_iface_name].append(port_name)

                requirements.append({
                    "req_id": new_id("req"),
                    "statement": f"RTL port {port_name} ({direction}, width {width}): {comment}",
                    "type": "INTERFACE",
                    "subject": port_name,
                    "direction": direction,
                    "width": str(width),
                    "semantic_role": role,
                    "action": "TRANSMIT",
                    "confidence": 0.95,
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                })

        for if_name, sig_list in iface_signals.items():
            if sig_list:
                entities.append({
                    "name": if_name,
                    "type": "interface",
                    "description": f"Grouped interface ports: {', '.join(sig_list)}",
                    "attributes": {
                        "signals": sig_list,
                        "protocol": "axi4" if "axi" in if_name.lower() else "generic",
                        "owner": mod_name
                    }
                })

    # 9. RTL Parameters
    for param_m in _RTL_PARAM_RE.finditer(text):
        p_name = param_m.group(1).strip()
        p_val = param_m.group(2).strip()
        pattern_hits += 1
        entities.append({
            "name": p_name,
            "type": "parameter",
            "description": f"Hardware parameter {p_name} = {p_val}",
            "attributes": {"value": p_val}
        })

    # 10. Instruction Format & Bitfields
    fmt_fields: List[Dict[str, Any]] = []
    total_inst_width = 32
    hdr_m = _INST_FORMAT_HEADER_RE.search(text)
    if hdr_m:
        w_val = hdr_m.group(1) or hdr_m.group(2)
        if w_val:
            total_inst_width = int(w_val)

    for line in text.split('\n'):
        fld_m = _INST_FIELD_RE.match(line.strip())
        if fld_m:
            fld_name = fld_m.group(1).strip()
            fld_w = int(fld_m.group(2))
            fld_desc = fld_m.group(3).strip()
            fmt_fields.append({
                "name": fld_name,
                "width": fld_w,
                "description": fld_desc
            })

    if fmt_fields:
        pattern_hits += 2
        sum_w = sum(f["width"] for f in fmt_fields)
        fmt_width = total_inst_width if total_inst_width else (sum_w or 32)
        entities.append({
            "name": f"{fmt_width}-Bit Instruction Format",
            "type": "instruction_format",
            "description": f"Instruction format ({fmt_width} bits)",
            "attributes": {
                "total_width": fmt_width,
                "fields": fmt_fields
            }
        })

    # Inline format with bit ranges: Format R-Type: opcode[31:26], rs[25:21]...
    inline_fmt_m = re.search(r"(?:format\s+([A-Za-z0-9_\-]+)\s*:\s*)(.+)", text, re.IGNORECASE)
    if inline_fmt_m:
        fmt_name = inline_fmt_m.group(1).strip()
        fields_str = inline_fmt_m.group(2).strip()
        parsed_fields = []
        for fld in re.finditer(r"([A-Za-z0-9_]+)\s*\[\s*(\d+)\s*:\s*(\d+)\s*\]", fields_str):
            fname = fld.group(1).strip()
            msb_i = int(fld.group(2))
            lsb_i = int(fld.group(3))
            w = abs(msb_i - lsb_i) + 1
            parsed_fields.append({"name": fname, "width": w, "description": f"Field {fname} [{msb_i}:{lsb_i}]"})
        if parsed_fields:
            pattern_hits += 2
            entities.append({
                "name": f"{fmt_name} Format",
                "type": "instruction_format",
                "description": f"Instruction format {fmt_name}",
                "attributes": {
                    "total_width": 32,
                    "fields": parsed_fields
                }
            })

    # 11. Opcodes & Instructions
    for line in text.split('\n'):
        l_str = line.strip()
        op_m = _OPCODE_LINE_RE.match(l_str)
        if op_m:
            op_code_str = op_m.group(1).strip()
            mnemonic = op_m.group(2).strip().upper()
            desc = (op_m.group(3) or op_m.group(4) or "").strip()
            pattern_hits += 1

            entities.append({
                "name": mnemonic,
                "type": "instruction",
                "description": desc or f"Instruction {mnemonic}",
                "attributes": {
                    "opcode": op_code_str,
                    "mnemonic": mnemonic,
                    "description": desc
                }
            })
            entities.append({
                "name": f"{mnemonic}_OPCODE",
                "type": "opcode",
                "description": f"Opcode mapping for {mnemonic}",
                "attributes": {
                    "opcode_hex": op_code_str,
                    "binary_encoding": op_code_str,
                    "instruction": mnemonic,
                    "value": op_code_str
                }
            })
            requirements.append({
                "req_id": new_id("req"),
                "statement": f"Instruction {mnemonic} (opcode {op_code_str}): {desc}",
                "type": "INSTRUCTION",
                "subject": mnemonic,
                "action": "EXECUTE",
                "confidence": 0.95,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
            })

    # 12. Architectural Components from text descriptions
    if any(k in text.lower() for k in ("component", "architecture", "consists of", "core components", "mpe contains", "processor", "scheduler", "network", "elements", "hierarchy")):
        for line in text.split('\n'):
            comp_m = _COMPONENT_DEF_RE.match(line.strip())
            if comp_m:
                c_name = comp_m.group(1).strip()
                c_desc = comp_m.group(2).strip()
                c_name_l = c_name.lower()
                if not any(k in c_name_l for k in ("note", "rule", "section", "table", "figure", "page", "format", "instruction format", "example", "dimension", "dimensions")):
                    pattern_hits += 1
                    entities.append({
                        "name": c_name,
                        "type": "component",
                        "description": c_desc
                    })

    # 13. Performance Targets
    for perf_m in _PERF_METRIC_RE.finditer(text):
        metric_name = perf_m.group(1).strip()
        target_val = perf_m.group(2).strip()
        pattern_hits += 1
        entities.append({
            "name": metric_name,
            "type": "performance_requirement",
            "description": f"{metric_name}: {target_val}",
            "attributes": {
                "metric": metric_name,
                "target_value": target_val,
                "value": target_val
            }
        })
        requirements.append({
            "req_id": new_id("req"),
            "statement": f"Performance requirement {metric_name}: {target_val}",
            "type": "PERFORMANCE",
            "subject": metric_name,
            "action": "MEET_TARGET",
            "confidence": 0.95,
            "chunk_id": chunk_id,
            "doc_id": doc_id,
        })

    # 14. Timing Complexity
    for line in text.split('\n'):
        tc_m = _TIMING_COMPLEXITY_RE.match(line.strip())
        if tc_m:
            op_target = tc_m.group(1).strip()
            timing_expr = tc_m.group(2).strip()
            pattern_hits += 1
            timing.append({
                "constraint": op_target,
                "operator": "<=",
                "value": timing_expr,
                "unit": "cycles"
            })
            requirements.append({
                "req_id": new_id("req"),
                "statement": f"Execution complexity for {op_target}: {timing_expr}",
                "type": "TIMING",
                "timing": timing_expr,
                "confidence": 0.95,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
            })

    # 15. Memory Architecture
    if any(k in text.lower() for k in ("memory", "cache", "sram", "dram", "hierarchy", "buffer")):
        for line in text.split('\n'):
            mem_m = _MEMORY_LINE_RE.match(line.strip())
            if mem_m:
                m_name = mem_m.group(1).strip()
                m_desc = mem_m.group(2).strip()
                pattern_hits += 1
                addr_m = re.search(r"0x[0-9A-Fa-f]+", m_desc)
                size_m = re.search(r"\b\d+\s*(?:KB|MB|GB|TB|bytes|bits)\b", m_desc, re.IGNORECASE)
                entities.append({
                    "name": m_name,
                    "type": "memory",
                    "description": m_desc,
                    "attributes": {
                        "base_address": addr_m.group(0) if addr_m else "",
                        "size": size_m.group(0) if size_m else "",
                        "type": "cache" if "cache" in m_name.lower() else "sram"
                    }
                })
        for pm in re.finditer(r"\b(\d+\s*(?:KB|MB|GB|TB))\s+([A-Za-z0-9_\s]{2,25}?(?:Cache|SRAM|DRAM|Buffer|Memory))\b", text, re.IGNORECASE):
            sz = pm.group(1).strip()
            m_name = pm.group(2).strip()
            pattern_hits += 1
            entities.append({
                "name": m_name,
                "type": "memory",
                "description": f"Memory {m_name} ({sz})",
                "attributes": {
                    "size": sz,
                    "type": "cache" if "cache" in m_name.lower() else "buffer"
                }
            })

    # 16. External Interfaces
    if "interface" in text.lower():
        for line in text.split('\n'):
            if_m = _INTERFACE_LINE_RE.match(line.strip())
            if if_m:
                if_name = if_m.group(1).strip()
                if_desc = if_m.group(2).strip()
                if if_name.lower() not in ("user interface", "gui"):
                    pattern_hits += 1
                    entities.append({
                        "name": if_name,
                        "type": "interface",
                        "description": if_desc or f"Interface {if_name}",
                        "attributes": {
                            "protocol": "axi4" if "axi" in if_name.lower() else "generic"
                        }
                    })

    # 17. Error Handling
    if any(k in text.lower() for k in ("error", "fault", "exception")):
        for line in text.split('\n'):
            err_m = _ERROR_LINE_RE.match(line.strip())
            if err_m:
                err_token = err_m.group(1).strip()
                err_rest = (err_m.group(2) or "").strip()
                pattern_hits += 1
                entities.append({
                    "name": err_token,
                    "type": "error",
                    "description": err_rest or line.strip()
                })
                requirements.append({
                    "req_id": new_id("req"),
                    "statement": f"Error condition: {line.strip()}",
                    "type": "ERROR",
                    "subject": err_token,
                    "action": "HANDLE_ERROR",
                    "confidence": 0.90,
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                })

    # 18. Functional Behaviors & Execution Strategies
    if any(k in text.lower() for k in ("tiling", "partitioning", "scheduler", "load balancing", "collective operations")):
        for line in text.split('\n'):
            l_strip = line.strip()
            if any(k in l_strip.lower() for k in ("splits", "distributes", "optimizes", "stealing", "interleaves", "coherence")):
                pattern_hits += 1
                requirements.append({
                    "req_id": new_id("req"),
                    "statement": l_strip,
                    "type": "BEHAVIOR",
                    "subject": l_strip.split(":")[0].strip() if ":" in l_strip else "System",
                    "action": "EXECUTE",
                    "confidence": 0.90,
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                })

    # 19. Natural-Language Port & Signal Declarations (e.g. Inputs: / Outputs: lists or in-line bullet points)
    _RESERVED_SIG_WORDS = {"note", "rule", "section", "table", "figure", "page", "reset", "operation", "the", "for", "and", "inputs", "outputs", "input", "output"}
    current_port_direction = None
    for line in text.split('\n'):
        l_s = line.strip()
        if not l_s:
            continue
        # Section headers
        if re.match(r"^[-*#]*\s*(?:inputs?|input\s+signals?|input\s+ports?)\s*[:]?\s*$", l_s, re.IGNORECASE):
            current_port_direction = "input"
            continue
        elif re.match(r"^[-*#]*\s*(?:outputs?|output\s+signals?|output\s+ports?)\s*[:]?\s*$", l_s, re.IGNORECASE):
            current_port_direction = "output"
            continue
        elif re.match(r"^[-*#]*\s*(?:inouts?|bidirectional\s+ports?)\s*[:]?\s*$", l_s, re.IGNORECASE):
            current_port_direction = "inout"
            continue
        elif re.match(r"^[-*#]*\s*(?:operations?|fsm|states?|transitions?|features?|requirements?)\s*[:]?\s*$", l_s, re.IGNORECASE):
            current_port_direction = None
            continue

        sig_m = re.match(r"^[-*]?\s*([A-Za-z0-9_]+)\s*(?:\[([^\]]+)\])?\s*:\s*(.*)$", l_s)
        if sig_m:
            cand_name = sig_m.group(1).strip()
            bracket_w = sig_m.group(2)
            rest_desc = sig_m.group(3).strip()
            cand_l = cand_name.lower()

            if cand_l not in _RESERVED_SIG_WORDS:
                dir_val = None
                rest_l = rest_desc.lower()
                if "inout" in rest_l or "bidirectional" in rest_l:
                    dir_val = "inout"
                elif re.search(r"\boutput\b|\boutputs\b|\bout\b|\bdrives?\b|\blamp\b|\bled\b|\btransmit\b", rest_l):
                    dir_val = "output"
                elif re.search(r"\binput\b|\binputs\b|\bin\b|\breceives?\b|\bclock\b|\breset\b", rest_l):
                    dir_val = "input"
                elif current_port_direction:
                    dir_val = current_port_direction

                if dir_val:
                    width_val = "1"
                    w_m = re.search(r"(\d+)\s*bits?", rest_desc, re.IGNORECASE)
                    if w_m:
                        width_val = w_m.group(1)
                    elif bracket_w:
                        width_val = bracket_w

                    role_val = "data"
                    if "clk" in cand_l or "clock" in cand_l:
                        role_val = "clock"
                    elif "rst" in cand_l or "reset" in cand_l:
                        role_val = "reset"
                    elif any(k in cand_l for k in ("valid", "ready", "start", "busy", "done", "enable")):
                        role_val = "control"

                    pattern_hits += 1
                    entities.append({
                        "name": cand_name,
                        "type": "clock" if role_val == "clock" else ("reset" if role_val == "reset" else "signal"),
                        "description": rest_desc or f"{dir_val} port {cand_name}",
                        "attributes": {
                            "direction": dir_val,
                            "width": width_val,
                            "semantic_role": role_val,
                            "source_quote": l_s
                        }
                    })
                    requirements.append({
                        "req_id": new_id("req"),
                        "statement": f"Signal '{cand_name}' direction={dir_val}, width={width_val}: {rest_desc}",
                        "type": "INTERFACE",
                        "subject": cand_name,
                        "action": "TRANSMIT" if dir_val == "output" else "RECEIVE",
                        "confidence": 0.95,
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                    })

    # 20. Natural-Language FSM States & Transitions
    for st_dur_m in re.finditer(r"\b([A-Za-z0-9_]{2,20})\s+(?:state\s+)?lasts\s+(\d+)\s+cycles?\b", text, re.IGNORECASE):
        st_name = st_dur_m.group(1).strip()
        dur_val = int(st_dur_m.group(2).strip())
        if st_name.lower() not in ("each", "the", "every", "all", "this"):
            pattern_hits += 1
            entities.append({
                "name": st_name,
                "type": "state",
                "description": f"FSM state {st_name} (duration: {dur_val} cycles)",
                "attributes": {
                    "duration_cycles": dur_val,
                    "source_quote": st_dur_m.group(0)
                }
            })
            requirements.append({
                "req_id": new_id("req"),
                "statement": f"State '{st_name}' active duration is {dur_val} clock cycles.",
                "type": "TIMING",
                "subject": st_name,
                "state": st_name,
                "timing": f"{dur_val} cycles",
                "value": dur_val,
                "confidence": 0.95,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
            })

    for trans_m in re.finditer(r"(?:transitions?\s+from\s+([A-Za-z0-9_]+)\s+to\s+([A-Za-z0-9_]+)|([A-Za-z0-9_]+)\s*(?:state)?\s*(?:then\s+)?transitions?\s+to\s+([A-Za-z0-9_]+)|([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+))", text, re.IGNORECASE):
        src_st = trans_m.group(1) or trans_m.group(3) or trans_m.group(5)
        dst_st = trans_m.group(2) or trans_m.group(4) or trans_m.group(6)
        if src_st and dst_st:
            src_st = src_st.strip()
            dst_st = dst_st.strip()
            if src_st.lower() not in ("reset", "it", "this") and dst_st.lower() not in ("reset", "it", "this"):
                pattern_hits += 1
                requirements.append({
                    "req_id": new_id("req"),
                    "statement": f"State transition from '{src_st}' to '{dst_st}'",
                    "type": "STATE",
                    "subject": src_st,
                    "state": src_st,
                    "next_state": dst_st,
                    "action": "TRANSITION",
                    "confidence": 0.95,
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                })

    for rst_st_m in re.finditer(r"(?:reset\s+(?:goes\s+to|enters|forces)\s+(?:state\s+)?([A-Za-z0-9_]+)|reset(?:_n)?\s*(?:==?|<=?)\s*['01b]+\s*->\s*([A-Za-z0-9_]+))", text, re.IGNORECASE):
        rst_st = (rst_st_m.group(1) or rst_st_m.group(2)).strip()
        if rst_st and rst_st.lower() not in ("state", "the", "a"):
            pattern_hits += 1
            requirements.append({
                "req_id": new_id("req"),
                "statement": f"Reset initializes state machine to '{rst_st}'",
                "type": "STATE",
                "subject": "reset",
                "next_state": rst_st,
                "action": "INITIALIZE",
                "confidence": 0.95,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
            })

    # Deduplicate entities by (type, name) while preserving ALL distinct opcodes/instructions
    dedup_entities: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for e in entities:
        k = (e.get("type", "").lower(), e.get("name", "").lower())
        if k not in dedup_entities:
            dedup_entities[k] = e
    entities = list(dedup_entities.values())

    # Confidence calculation
    high_thresh = getattr(CONFIG, "heuristic_high_confidence", 0.90)
    med_thresh = getattr(CONFIG, "heuristic_medium_confidence", 0.70)

    if pattern_hits >= 2 and (requirements or entities):
        confidence = high_thresh
    elif pattern_hits == 1 and (requirements or entities):
        confidence = med_thresh
    elif entities or requirements:
        confidence = 0.75
    else:
        confidence = 0.0

    return HeuristicExtractionResult(
        requirements=requirements,
        entities=entities,
        relationships=relationships,
        conditions=conditions,
        events=events,
        timing=timing,
        method="heuristic",
        confidence=confidence,
    )
