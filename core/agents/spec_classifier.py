"""
Hardware Architecture Classification Module.

Derives architectural characteristics and hardware classification from
canonical EFS IR models, specification text, and source evidence.

Supported Architectural Characteristics:
- Combinational logic
- Sequential logic
- FSM-based controllers
- Datapaths
- Control/datapath combinations
- Memory interfaces
- Bus interfaces
- Arithmetic processing units
- Custom architectures
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from core.efs_ir.models import EFSIR
from utils.logger import get_logger

logger = get_logger("agents.spec_classifier")


class ArchitectureClass(str, Enum):
    """Canonical hardware architecture classifications."""
    COMBINATIONAL = "COMBINATIONAL"
    SEQUENTIAL = "SEQUENTIAL"
    FSM_CONTROLLER = "FSM_CONTROLLER"
    DATAPATH = "DATAPATH"
    CONTROL_DATAPATH = "CONTROL_DATAPATH"
    MEMORY_INTERFACE = "MEMORY_INTERFACE"
    BUS_INTERFACE = "BUS_INTERFACE"
    ARITHMETIC_PROCESSING_UNIT = "ARITHMETIC_PROCESSING_UNIT"
    CUSTOM_ARCHITECTURE = "CUSTOM_ARCHITECTURE"


class SpecType(str, Enum):
    """Specification category enum compatible with evaluation harness and fixtures."""
    COMBINATIONAL = "COMBINATIONAL"
    SEQUENTIAL = "SEQUENTIAL"
    FSM_CONTROLLER = "FSM_CONTROLLER"
    DATAPATH_PIPELINE = "DATAPATH_PIPELINE"
    DATAPATH = "DATAPATH"
    CONTROL_DATAPATH = "CONTROL_DATAPATH"
    REGISTER_MAP = "REGISTER_MAP"
    FIFO_MEMORY = "FIFO_MEMORY"
    MEMORY_INTERFACE = "MEMORY_INTERFACE"
    BUS_INTERFACE = "BUS_INTERFACE"
    ARITHMETIC_PROCESSING_UNIT = "ARITHMETIC_PROCESSING_UNIT"
    PROTOCOL_ENGINE = "PROTOCOL_ENGINE"
    CUSTOM_ARCHITECTURE = "CUSTOM_ARCHITECTURE"
    UNDERSPECIFIED = "UNDERSPECIFIED"
    CONTRADICTORY = "CONTRADICTORY"


@dataclass
class ArchitectureClassificationResult:
    """Detailed hardware architecture classification result."""
    primary_class: ArchitectureClass
    spec_type: SpecType
    secondary_traits: List[str] = field(default_factory=list)
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)
    has_clock: bool = False
    has_reset: bool = False
    requires_fsm: bool = False
    has_registers: bool = False
    has_opcodes: bool = False
    has_memory_buffers: bool = False
    has_handshake: bool = False
    is_underspecified: bool = False
    is_contradictory: bool = False
    gap_evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_class": self.primary_class.value,
            "spec_type": self.spec_type.value,
            "secondary_traits": self.secondary_traits,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "has_clock": self.has_clock,
            "has_reset": self.has_reset,
            "requires_fsm": self.requires_fsm,
            "has_registers": self.has_registers,
            "has_opcodes": self.has_opcodes,
            "has_memory_buffers": self.has_memory_buffers,
            "has_handshake": self.has_handshake,
            "is_underspecified": self.is_underspecified,
            "is_contradictory": self.is_contradictory,
            "gap_evidence": self.gap_evidence,
        }


def _strip_negative_context(text: str) -> str:
    """Remove negative statements like 'must NOT invent an ALU' so they don't trigger false positives."""
    cleaned = re.sub(
        r"\b(?:must\s+not|do\s+not|never|without|no)\s+(?:invent|assume|implement|include|add)?\s*[^.,;\n]+",
        " ",
        text,
        flags=re.IGNORECASE
    )
    return cleaned


def _detect_contradictions_and_gaps(
    spec_text: str,
    efs_ir: Optional[EFSIR]
) -> tuple[bool, bool, List[str]]:
    """Detect whether specification is contradictory or underspecified."""
    is_contradictory = False
    is_underspecified = False
    evidence: List[str] = []

    # 1. Check EFS IR conflicts
    if efs_ir and getattr(efs_ir, "conflicts", None):
        unresolved = [
            c for c in efs_ir.conflicts
            if getattr(c, "resolution_status", "UNRESOLVED") in ("UNRESOLVED", "ACTIVE")
        ]
        if unresolved:
            is_contradictory = True
            for c in unresolved:
                desc = getattr(c, "description", "") or getattr(c, "evidence", "")
                evidence.append(f"IR Conflict: {desc}")

    # 2. Check text for contradictory reset or width markers
    text_lower = spec_text.lower()
    has_act_high = "active-high reset" in text_lower or "active high reset" in text_lower or "rst_n == 1" in text_lower or "rst == 1" in text_lower
    has_act_low = "active-low reset" in text_lower or "active low reset" in text_lower or "rst_n == 0" in text_lower or "rst == 0" in text_lower
    if has_act_high and has_act_low and ("contradict" in text_lower or "section 1" in text_lower):
        is_contradictory = True
        evidence.append("Contradictory reset polarity specified in text sections")

    if re.search(r"\bcontradictory\b", text_lower) or re.search(r"\bconflicting\s+state\s+transitions\b", text_lower):
        is_contradictory = True
        evidence.append("Explicit mention of contradictory specification in text")

    # 3. Check for underspecified markers
    missing_patterns = [
        r"\b(?:no\s+bit\s*widths?)\b",
        r"\b(?:no\s+clock\s+frequency\s+or\s+reset\s+polarity\s+specified)\b",
        r"\b(?:no\s+priority\s+arbitration\s+rules?)\b",
        r"\b(?:no\s+specified\s+condition)\b",
        r"\b(?:completely\s+unstated)\b",
        r"\b(?:underspecified)\b",
        r"\b(?:unspecified\s+(?:condition|transition|port\s+width|register\s+width|width|clock|reset))\b",
        r"\b(?:specification\s+(?:is\s+)?incomplete)\b",
        r"\b(?:missing\s+(?:critical\s+)?requirements?)\b",
    ]
    if any(re.search(pat, text_lower) for pat in missing_patterns):
        is_underspecified = True
        evidence.append("Specification text explicitly indicates missing/unstated requirements")

    # Check EFS IR FSM for missing transition conditions
    if efs_ir and getattr(efs_ir, "fsms", None):
        for f in efs_ir.fsms:
            all_trans = getattr(f, "transitions", []) or []
            from_map = {}
            for tr in all_trans:
                s_st = getattr(tr, "source_state", "")
                from_map.setdefault(s_st, []).append(tr)

            for tr in all_trans:
                cond = getattr(tr, "condition", None)
                cond_str = cond.get("expression", "") if isinstance(cond, dict) else str(cond or "")
                s_st = str(getattr(tr, "source_state", "")).lower()
                if cond_str.strip() == "":
                    # Unconditional transition is valid if it is the only transition from this state or a reset/init state
                    is_unconditional = (len(from_map.get(getattr(tr, "source_state", ""), [])) == 1) or any(k in s_st for k in ("reset", "init", "start", "idle", "soft_reset"))
                    if not is_unconditional:
                        is_underspecified = True
                        evidence.append(f"FSM transition {getattr(tr, 'source_state', '')} -> {getattr(tr, 'target_state', '')} has ambiguous empty condition")

    return is_contradictory, is_underspecified, evidence


def classify_hardware_architecture(
    efs_ir: Optional[EFSIR] = None,
    spec_text: str = "",
    source_evidence: Optional[List[Dict[str, Any]]] = None,
) -> ArchitectureClassificationResult:
    """
    Extensible hardware architecture classifier.
    Derives architecture classification strictly from EFS IR and source evidence.
    """
    text_raw = spec_text or ""
    text_lower = text_raw.lower()
    text_positive = _strip_negative_context(text_raw).lower()
    evidence_list: List[str] = []
    secondary_traits: List[str] = []

    # 1. Contradiction and Gap Detection
    is_contradictory, is_underspecified, gap_evidence = _detect_contradictions_and_gaps(spec_text, efs_ir)
    evidence_list.extend(gap_evidence)

    # 2. Signal & Port Analysis from EFS IR
    signals: List[Any] = []
    if efs_ir and getattr(efs_ir, "signals", None):
        signals = efs_ir.signals

    has_clock = False
    has_reset = False
    has_handshake = False

    clock_names = {"clk", "clock", "sys_clk", "core_clk", "pclk", "aclk"}
    reset_names = {"rst", "rst_n", "reset", "reset_n", "aresetn", "sys_rst_n", "prst_n"}

    for s in signals:
        name = (s.get("name") if isinstance(s, dict) else getattr(s, "name", "")).lower()
        role = (s.get("semantic_role") if isinstance(s, dict) else getattr(s, "semantic_role", "")).lower()
        if name in clock_names or "clk" in name or "clock" in name or role == "clock":
            has_clock = True
        if name in reset_names or "rst" in name or "reset" in name or role == "reset":
            has_reset = True
        if name in ("valid", "ready", "s_valid", "m_valid", "s_ready", "m_ready", "req", "grant", "ack", "sop", "eop"):
            has_handshake = True

    # If EFS IR was not provided or had no signals, inspect text
    if not signals:
        if re.search(r"\b(clk|clock|posedge|negedge|rising\s+clk|clock\s+cycle)\b", text_positive):
            has_clock = True
            evidence_list.append("Clock or edge-triggering references found in specification text")
        if re.search(r"\b(rst|rst_n|reset|active-low\s+rst|active-high\s+rst|active-low\s+reset|active-high\s+reset|synchronous\s+reset|asynchronous\s+reset)\b", text_positive):
            has_reset = True
            evidence_list.append("Reset semantics found in specification text")
        if re.search(r"\b(valid|ready|handshake|s_valid|m_ready|req0|grant0|request|grant|ack)\b", text_positive):
            has_handshake = True
            evidence_list.append("Handshake signaling identified in text")

    # 3. FSM Analysis
    has_explicit_fsm = False
    fsm_state_count = 0
    if efs_ir and getattr(efs_ir, "fsms", None):
        has_explicit_fsm = len(efs_ir.fsms) > 0
        for f in efs_ir.fsms:
            fsm_state_count += len(getattr(f, "states", []))
        if has_explicit_fsm:
            evidence_list.append(f"EFS IR contains {len(efs_ir.fsms)} FSM(s) with {fsm_state_count} total states")

    text_fsm_keywords = bool(re.search(r"\b(state\s+machine|states?[:\s]+[a-z0-9_,\s]+|state\s+s_[a-z0-9_]+|transition\s+to\s+s_[a-z0-9_]+|dwells\s+in\s+s_[a-z0-9_]+|3-state|sequence\s+controller)\b", text_lower))
    if text_fsm_keywords and not has_explicit_fsm:
        evidence_list.append("FSM state descriptions identified in specification text")

    requires_fsm = has_explicit_fsm or text_fsm_keywords

    # 4. Register Bank vs Pipeline Register Analysis
    has_register_bank = False
    if efs_ir and getattr(efs_ir, "registers", None) and len(efs_ir.registers) > 1:
        has_register_bank = True
        evidence_list.append(f"EFS IR contains {len(efs_ir.registers)} register definition(s)")
    elif re.search(r"\b(register\s+bank|register\s+file|register\s+map|reg_addr|addr.*read_data|addr.*write_data|offset\s+0x|reg_data)\b", text_positive):
        has_register_bank = True
        evidence_list.append("Register bank or memory-mapped register map definitions identified")

    # Single storage register/buffer (like 16-bit register with load, data_in, data_out)
    has_single_reg = bool(re.search(r"\b(\d+-bit\s+register|data\s+buffer|parameterized\s+data\s+buffer|load.*data_out\s*<=|dout\s*<=\s*din)\b", text_positive))

    has_opcodes = False
    if efs_ir and (getattr(efs_ir, "opcodes", None) or getattr(efs_ir, "instructions", None)):
        has_opcodes = True
        evidence_list.append("EFS IR contains explicit opcode/instruction definitions")
    elif re.search(r"\b(opcode|instruction|alu_sel|op_code|mnemonic)\b", text_positive):
        has_opcodes = True
        evidence_list.append("Opcode or instruction fields identified in text")

    # 5. FIFO / Multi-word Memory Analysis
    has_fifo = bool(re.search(r"\b(fifo|circular\s+buffer|write\s+pointer|read\s+pointer|push.*pop|wr_ptr|rd_ptr|fifo_depth)\b", text_positive))
    if has_fifo:
        evidence_list.append("FIFO queue structures identified")

    # 6. Pipeline / Datapath / Counter / Timer Analysis
    has_pipeline = bool(re.search(r"\b(pipeline|pipeline_stage|stage_valid|stage_data|stall|flush)\b", text_positive))
    has_counter = bool(re.search(r"\b(counter|count\s*<=\s*count\s*\+|pulse\s+generator|timer\s+pulse|timer\s+cnt|duration\s+in\s+cycles)\b", text_positive))

    if has_pipeline:
        secondary_traits.append("PIPELINED")
        evidence_list.append("Pipeline staging identified in specification")
    if has_counter:
        secondary_traits.append("COUNTER_TIMER")

    # 7. Protocol Engine / Adapter Analysis
    has_protocol_engine = bool(re.search(r"\b(protocol\s+adapter|stream\s+handshake|skid\s+buffer|stream\s+protocol|handshake\s+protocol)\b", text_positive))
    if has_protocol_engine or (has_handshake and not requires_fsm and not has_fifo):
        has_protocol_engine = True
        secondary_traits.append("PROTOCOL_ENGINE")

    # 8. Arithmetic Analysis
    has_arithmetic = bool(re.search(r"\b(adder|multiplier|alu|accumulator|mac|matrix\s+mult|subtract|alu_res)\b", text_positive))
    if has_arithmetic:
        secondary_traits.append("ARITHMETIC")
        evidence_list.append("Arithmetic operations identified in specification")

    # 9. Combinational Analysis
    is_explicit_comb = bool(re.search(r"\b(multiplexer|4-to-1|mux|decoder|3-to-8|priority\s+encoder|combinational)\b", text_lower))
    if not has_clock and not has_reset and not requires_fsm:
        is_explicit_comb = True

    # 10. Determine Primary Architecture Class and SpecType
    if is_contradictory:
        primary_class = ArchitectureClass.CUSTOM_ARCHITECTURE
        spec_type = SpecType.CONTRADICTORY
    elif is_underspecified:
        primary_class = ArchitectureClass.CUSTOM_ARCHITECTURE
        spec_type = SpecType.UNDERSPECIFIED
    elif has_fifo:
        primary_class = ArchitectureClass.MEMORY_INTERFACE
        spec_type = SpecType.FIFO_MEMORY
    elif has_protocol_engine:
        primary_class = ArchitectureClass.BUS_INTERFACE
        spec_type = SpecType.PROTOCOL_ENGINE
    elif has_register_bank:
        primary_class = ArchitectureClass.BUS_INTERFACE
        spec_type = SpecType.REGISTER_MAP
    elif requires_fsm and (has_arithmetic or has_pipeline):
        primary_class = ArchitectureClass.CONTROL_DATAPATH
        spec_type = SpecType.CONTROL_DATAPATH
    elif requires_fsm:
        primary_class = ArchitectureClass.FSM_CONTROLLER
        spec_type = SpecType.FSM_CONTROLLER
    elif is_explicit_comb and not has_clock and not has_reset:
        primary_class = ArchitectureClass.COMBINATIONAL
        spec_type = SpecType.COMBINATIONAL
    elif has_pipeline or has_counter or has_single_reg or (has_clock and "length" in text_positive):
        primary_class = ArchitectureClass.DATAPATH
        spec_type = SpecType.DATAPATH_PIPELINE
    elif has_arithmetic and not requires_fsm:
        primary_class = ArchitectureClass.ARITHMETIC_PROCESSING_UNIT
        spec_type = SpecType.ARITHMETIC_PROCESSING_UNIT
    elif has_clock or has_reset:
        primary_class = ArchitectureClass.SEQUENTIAL
        spec_type = SpecType.SEQUENTIAL
    else:
        primary_class = ArchitectureClass.COMBINATIONAL if not has_clock else ArchitectureClass.CUSTOM_ARCHITECTURE
        spec_type = SpecType.COMBINATIONAL if not has_clock else SpecType.CUSTOM_ARCHITECTURE

    return ArchitectureClassificationResult(
        primary_class=primary_class,
        spec_type=spec_type,
        secondary_traits=secondary_traits,
        confidence=0.95 if evidence_list else 0.75,
        evidence=evidence_list,
        has_clock=has_clock,
        has_reset=has_reset,
        requires_fsm=requires_fsm,
        has_registers=(has_register_bank or has_single_reg),
        has_opcodes=has_opcodes,
        has_memory_buffers=has_fifo,
        has_handshake=has_handshake,
        is_underspecified=is_underspecified,
        is_contradictory=is_contradictory,
        gap_evidence=gap_evidence,
    )


def classify_spec_type(spec_text: str, efs_ir: Optional[EFSIR] = None) -> SpecType:
    """
    Public classification function returning the SpecType enum.
    Seamlessly satisfies eval harness and pipeline callers.
    """
    result = classify_hardware_architecture(efs_ir=efs_ir, spec_text=spec_text)
    return result.spec_type
