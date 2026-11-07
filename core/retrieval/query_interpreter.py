"""
Query Interpreter module.

Interprets natural-language user queries to resolve:
1. Target Component / Subsystem (e.g., "Instruction Decoder", "Matrix Compute Unit", "ALU", "FIFO")
2. Requested Scope ("COMPONENT", "SUBMODULE", "INTERFACE", "FULL_SYSTEM")
3. Target Artifact ("RTL", "TESTBENCH", "SVA", "PLANTUML", "ARCHITECTURE_DOC")
4. Target Language ("Verilog", "SystemVerilog", "VHDL", "PlantUML")
5. Referenced / Implied Protocols (e.g., ["AXI4", "APB", "AHB"])
6. Relevant Operation Modes / Transactions (e.g., ["decode", "read", "write", "burst"])
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from core.retrieval.context_pack import TaskIntent
from core.retrieval.query_analyzer import analyze_query, classify_query_category
from utils.logger import get_logger

logger = get_logger("retrieval.query_interpreter")


@dataclass
class ScopedQueryIntent:
    target_component: str = ""
    requested_scope: str = "COMPONENT"  # "COMPONENT" | "SUBMODULE" | "INTERFACE" | "FULL_SYSTEM"
    target_artifact: str = "RTL"        # "RTL" | "TESTBENCH" | "SVA" | "PLANTUML" | "ARCHITECTURE_DOC"
    target_language: str = "Verilog"    # "Verilog" | "SystemVerilog" | "VHDL" | "PlantUML"
    referenced_protocols: List[str] = field(default_factory=list)
    relevant_modes: List[str] = field(default_factory=list)
    query_category: str = "fact"
    raw_query: str = ""

    def to_task_intent(self) -> TaskIntent:
        """Convert to legacy TaskIntent for backwards compatibility."""
        task_map = {
            "RTL": f"{self.target_language.upper()}_GENERATION",
            "TESTBENCH": "TESTBENCH_GENERATION",
            "SVA": "SVA_GENERATION",
            "PLANTUML": "PLANTUML_GENERATION",
            "ARCHITECTURE_DOC": "ARCHITECTURE_DOC",
        }
        task_name = task_map.get(self.target_artifact, "RTL_GENERATION")
        if self.target_artifact == "RTL" and self.target_language == "Verilog":
            task_name = "RTL_GENERATION"

        primary_proto = self.referenced_protocols[0] if self.referenced_protocols else None
        return TaskIntent(
            task=task_name,
            target=self.target_component,
            interface=primary_proto,
            language=self.target_language,
            query_category=self.query_category,
            raw_query=self.raw_query,
        )


def interpret_query(query: str, default_design_name: str = "") -> ScopedQueryIntent:
    """
    Parse natural language query into a ScopedQueryIntent.
    """
    raw_q = (query or "").strip()
    if not raw_q and default_design_name:
        raw_q = f"Generate synthesizable Verilog RTL for {default_design_name}."

    # Use existing query analyzer for base intent
    base_intent = analyze_query(raw_q)
    target_comp = base_intent.target.strip()
    clean_target = re.sub(r"\s+(?:handling|implementing|with|supporting|using|attached\s+to|connected\s+to)\s+.*$", "", target_comp, flags=re.IGNORECASE).strip()
    if clean_target:
        target_comp = clean_target
    if not target_comp and default_design_name:
        target_comp = default_design_name

    q_lower = raw_q.lower()

    # 1. Determine Scope
    scope = "COMPONENT"
    if any(k in q_lower for k in ["submodule", "internal block", "sub-block", "unit within"]):
        scope = "SUBMODULE"
    elif any(k in q_lower for k in ["interface only", "bus interface", "port interface", "channel interface"]):
        scope = "INTERFACE"
    elif any(k in q_lower for k in ["full system", "complete system", "entire design", "top level", "all modules"]):
        scope = "FULL_SYSTEM"
    elif target_comp.lower() in ("top", "toplevel", "full", "system", "all", "completedesign"):
        scope = "FULL_SYSTEM"

    # 2. Determine Artifact
    artifact = "RTL"
    if "sva" in q_lower or "assertion" in q_lower:
        artifact = "SVA"
    elif "testbench" in q_lower or "tb" in q_lower or "uvm" in q_lower:
        artifact = "TESTBENCH"
    elif "plantuml" in q_lower or "puml" in q_lower or "diagram" in q_lower or "architecture doc" in q_lower:
        artifact = "PLANTUML"
    elif "doc" in q_lower or "specification document" in q_lower:
        artifact = "ARCHITECTURE_DOC"

    # 3. Determine Language
    lang = base_intent.language or "Verilog"
    if "systemverilog" in q_lower or "sv rtl" in q_lower or artifact in ("SVA", "TESTBENCH"):
        lang = "SystemVerilog"
    elif "vhdl" in q_lower:
        lang = "VHDL"
    elif artifact == "PLANTUML":
        lang = "PlantUML"

    # 4. Extract Protocols
    protocols: List[str] = []
    if base_intent.interface:
        protocols.append(base_intent.interface)

    found_protos = re.findall(
        r"\b(AXI\d?|AHB|APB|PCIe|UART|SPI|I2C|Wishbone|Avalon)\b",
        raw_q,
        re.IGNORECASE
    )
    for p in found_protos:
        p_up = p.upper()
        if p_up not in protocols:
            protocols.append(p_up)

    # 5. Extract Modes / Transactions
    modes: List[str] = []
    mode_candidates = ["decode", "execute", "read", "write", "burst", "handshake", "idle", "busy", "reset"]
    for m in mode_candidates:
        if m in q_lower:
            modes.append(m)

    cat = classify_query_category(raw_q)

    logger.info(
        f"Interpreted Query -> Component: '{target_comp}', Scope: '{scope}', "
        f"Artifact: '{artifact}', Language: '{lang}', Protocols: {protocols}"
    )

    return ScopedQueryIntent(
        target_component=target_comp,
        requested_scope=scope,
        target_artifact=artifact,
        target_language=lang,
        referenced_protocols=protocols,
        relevant_modes=modes,
        query_category=cat,
        raw_query=raw_q,
    )
