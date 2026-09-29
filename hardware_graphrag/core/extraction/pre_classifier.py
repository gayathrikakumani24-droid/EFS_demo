"""
Deterministic Chunk Pre-Classifier module (Step 2).

Pre-classifies document blocks/chunks before extraction to determine relevance,
identify semantic categories, detect signal/register tokens, and estimate complexity.
Uses strict regex and normalized keyword rules with ZERO LLM dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple


@dataclass
class ClassificationResult:
    relevant: bool
    categories: List[str] = field(default_factory=list)
    signals_detected: List[str] = field(default_factory=list)
    registers_detected: List[str] = field(default_factory=list)
    complexity: str = "low"  # low | medium | high
    has_unresolved_references: bool = False
    has_tables: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relevant": self.relevant,
            "categories": self.categories,
            "signals_detected": self.signals_detected,
            "registers_detected": self.registers_detected,
            "complexity": self.complexity,
            "has_unresolved_references": self.has_unresolved_references,
            "has_tables": self.has_tables,
        }


# Pattern rules for categories
_CATEGORY_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("handshake", re.compile(r"\b([A-Z0-9_]*VALID\s+(?:and|&{1,2}|with)\s+[A-Z0-9_]*READY|valid\s*[/&]\s*ready|handshake|req\s*[/&]\s*ack|request\s*[/&]\s*grant|valid/ready)\b", re.IGNORECASE)),
    ("condition", re.compile(r"\b(if|when|whenever|provided that|only if|in case|where|supposing|unless)\b", re.IGNORECASE)),
    ("timing", re.compile(r"\b(cycles?|ns|us|ms|clock cycles?|timing|setup time|hold time|latency|t_su|t_h|timeout|propagation|o\([a-z0-9_\s\*\+\-×]+\))\b", re.IGNORECASE)),
    ("event", re.compile(r"\b(rising edge|falling edge|posedge|negedge|trigger|upon|on assertion|on deassertion|edge of)\b", re.IGNORECASE)),
    ("reset", re.compile(r"\b(reset|rst|rst_n|resetn|power-on|por|initialization|boot|default value)\b", re.IGNORECASE)),
    ("state_transition", re.compile(r"\b(state|fsm|transition|idle|active|wait|busy|done|next_state|current_state|state machine)\b", re.IGNORECASE)),
    ("signal", re.compile(r"\b([A-Z0-9_]{2,}(?:_VALID|_READY|_RESET|_CLK|_ENABLE|_SEL|_ACK|_REQ|_ADDR|_DATA|_RESP|_STRB|_LAST|_ID|_BURST|_LEN|_SIZE|_LOCK|_CACHE|_PROT|_bus|data_bus|bus))\b|bus|port|pin|bits|wire|logic", re.IGNORECASE)),
    ("register", re.compile(r"\b(\w+\s+register|bit\s*\d+(?:[:\-]\d+)?|read-only|write-only|read/write|r/w|ro|wo|offset\s+0x[0-9a-f]+)\b", re.IGNORECASE)),
    ("transaction", re.compile(r"\b(transaction|burst|transfer|read operation|write operation|packet|message|payload)\b", re.IGNORECASE)),
    ("rule", re.compile(r"\b(must|shall|should|must not|shall not|required|prohibited)\b", re.IGNORECASE)),
    ("instruction", re.compile(r"\b(opcode|instruction|mnemonic|syntax|encoding|instruction format|dest_reg|src_reg|operands?)\b", re.IGNORECASE)),
    ("component", re.compile(r"\b(consists of|contains|key components|subsystem|core components|processing elements?|execution units?|controller|decoder|alu|fpu|register file|scheduler|engine|accelerator)\b", re.IGNORECASE)),
    ("memory", re.compile(r"\b(sram|dram|cache|memory hierarchy|scratchpad|buffer|tiling|descriptor|coherence|mesi|banked|prefetching|memory interface)\b", re.IGNORECASE)),
    ("performance", re.compile(r"\b(frequency|ghz|mhz|throughput|tflops|mflops|elements per second|power consumption|scaling|efficiency|bandwidth)\b", re.IGNORECASE)),
    ("error", re.compile(r"\b(error handling|overflow|underflow|singular|exception|fault|error reporting)\b", re.IGNORECASE)),
    ("rtl", re.compile(r"\b(module\s+[a-zA-Z0-9_]+\s*\(|input\s+(?:wire|logic|reg)?|output\s+(?:wire|logic|reg)?|parameter\s+[a-zA-Z0-9_]+)\b", re.IGNORECASE)),
]

# Pattern for signal detection
_SIGNAL_RE = re.compile(r"\b([A-Z0-9_]{2,}(?:VALID|READY|RESET|CLK|ENABLE|SEL|ACK|REQ|ADDR|DATA|RESP|STRB|LAST|ID|BURST|LEN|SIZE|LOCK|CACHE|PROT|bus|n)?)\b", re.IGNORECASE)

# Pattern for register detection
_REGISTER_RE = re.compile(r"\b([A-Z0-9_]{3,}_REG|REG_[A-Z0-9_]+|[A-Z][a-zA-Z0-9_]+Register)\b")

# Reference ambiguity patterns (pronouns or cross-references needing LLM escalation)
_UNRESOLVED_REF_RE = re.compile(r"\b(this signal|the previous transaction|the aforementioned|it|these signals|the latter|the former|see section|as described above)\b", re.IGNORECASE)


def classify_chunk(text: str, heading: str = "", is_table: bool = False) -> ClassificationResult:
    """
    Classify a chunk of text or table data.
    """
    if not text or not text.strip():
        return ClassificationResult(relevant=False, complexity="low")

    norm_text = text.strip()
    categories: Set[str] = set()

    for cat, pat in _CATEGORY_PATTERNS:
        if pat.search(norm_text) or (heading and pat.search(heading)):
            categories.add(cat)

    # Table detection
    has_table = is_table or "|" in norm_text

    # Signal & Register extraction
    signals = sorted(set(_SIGNAL_RE.findall(norm_text)))[:10]
    registers = sorted(set(_REGISTER_RE.findall(norm_text)))[:10]

    # Reference resolution need
    has_unresolved_refs = bool(_UNRESOLVED_REF_RE.search(norm_text))

    # Relevance threshold
    relevant = bool(
        categories or signals or registers or has_table or
        any(kw in norm_text.lower() for kw in ["bus", "port", "pin", "bit", "width", "interface", "register", "signal", "opcode", "instruction", "module", "component", "memory", "performance", "error"]) or
        (heading and any(kw in heading.lower() for kw in ["signal", "register", "interface", "protocol", "timing", "reset", "state", "fsm", "processing", "unit", "hardware", "opcode", "instruction", "format", "syntax", "architecture", "memory", "performance", "error", "component"]))
    )

    # Complexity estimation
    word_count = len(norm_text.split())
    if word_count > 150 or len(categories) >= 4 or (has_unresolved_refs and word_count > 60):
        complexity = "high"
    elif word_count > 60 or len(categories) >= 2 or has_table:
        complexity = "medium"
    else:
        complexity = "low"

    return ClassificationResult(
        relevant=relevant,
        categories=sorted(categories),
        signals_detected=signals,
        registers_detected=registers,
        complexity=complexity,
        has_unresolved_references=has_unresolved_refs,
        has_tables=has_table,
    )
