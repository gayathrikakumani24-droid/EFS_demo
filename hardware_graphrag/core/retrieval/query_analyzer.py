"""
Query Analyzer module.

Analyzes natural language user generation requests and parses them into structured intent
specifying task type, target component, protocol interface, and target language.
"""

from __future__ import annotations

import re
from typing import Optional
from core.retrieval.context_pack import TaskIntent
from utils.logger import get_logger

from utils.logger import get_logger, truncate_for_log

logger = get_logger("retrieval.query_analyzer")

# Supported Task Types
SUPPORTED_TASKS = [
    "RTL_GENERATION",
    "SYSTEMVERILOG_GENERATION",
    "VHDL_GENERATION",
    "UVM_GENERATION",
    "SVA_GENERATION",
    "PLANTUML_GENERATION",
    "TESTBENCH_GENERATION",
    "FSM_GENERATION"
]


def classify_query_category(query: str) -> str:
    """Classify user query into lightweight retrieval categories: fact | relationship | multi-hop | comparison | full-flow."""
    q_lower = query.lower().strip()
    if any(k in q_lower for k in ["compare", "difference", "vs", "versus"]):
        return "comparison"
    elif any(k in q_lower for k in ["full flow", "complete flow", "end to end", "sequence", "architecture"]):
        return "full-flow"
    elif any(k in q_lower for k in ["how does", "why does", "impact", "depend", "connect"]):
        return "multi-hop"
    elif any(k in q_lower for k in ["relationship", "channel", "handshake", "interface"]):
        return "relationship"
    else:
        return "fact"


def _analyze_query_rule_based(query: str) -> TaskIntent:
    """Fallback rule-based parser for user queries."""
    q_clean = query.strip()

    # If query contains multiple lines or is a document, isolate the first line or specific query phrase
    if "\n" in q_clean or len(q_clean) > 150:
        lines = [line.strip() for line in q_clean.split("\n") if line.strip()]
        # Check if first line or any line starts with a generation command
        cmd_line = next((l for l in lines if re.match(r"^(?:generate|build|create|make|draw|show)\b", l, re.IGNORECASE)), "")
        q_clean = cmd_line if cmd_line else lines[0] if lines else ""

    q_lower = q_clean.lower()

    cat = classify_query_category(query)
    task = "RTL_GENERATION"
    language = "Verilog"

    if "sva" in q_lower or "assertion" in q_lower:
        task = "SVA_GENERATION"
        language = "SystemVerilog"
    elif "uvm" in q_lower or "uvm testbench" in q_lower:
        task = "UVM_GENERATION"
        language = "SystemVerilog"
    elif "vhdl" in q_lower:
        task = "VHDL_GENERATION"
        language = "VHDL"
    elif "systemverilog" in q_lower or "sv rtl" in q_lower:
        task = "SYSTEMVERILOG_GENERATION"
        language = "SystemVerilog"
    elif "plantuml" in q_lower or "puml" in q_lower or "diagram" in q_lower or "architecture" in q_lower:
        task = "PLANTUML_GENERATION"
        language = "PlantUML"
    elif "testbench" in q_lower or "tb" in q_lower:
        task = "TESTBENCH_GENERATION"
        language = "SystemVerilog"
    elif "fsm" in q_lower or "state machine" in q_lower:
        task = "FSM_GENERATION"
        language = "Verilog"

    interface: Optional[str] = None
    interface_matches = re.findall(r"\b(AXI\d?|AHB|APB|PCIe|UART|SPI|I2C|[A-Za-z0-9_]+(?:_interface|_bus|_channel|_ctrl))\b", q_clean, re.IGNORECASE)
    if interface_matches:
        interface = interface_matches[0].upper()

    target = ""
    clean_q = re.sub(r"^(?:generate|build|create|make|draw|show|display|render)\s+", "", q_clean, flags=re.IGNORECASE).strip()
    clean_q = re.sub(
        r"^(?:synthesizable\s+)?(?:plantuml|puml|verilog|systemverilog|vhdl|uvm|sva|efs\s+ir|efs|rtl|code|design|spec|assertions?|testbench|tb|diagrams?|architecture|fsm)\s*",
        "", clean_q, flags=re.IGNORECASE
    ).strip()
    clean_q = re.sub(r"^(?:rtl|design|code|testbench|assertions?|diagrams?|architecture|plantuml|puml)\s*", "", clean_q, flags=re.IGNORECASE).strip()
    clean_q = re.sub(r"^(?:for|of|on|about)\s+", "", clean_q, flags=re.IGNORECASE).strip()
    clean_q = re.sub(r"^(?:the\s+)?(?:[a-z0-9_-]+\s+interface\s+of\s+)", "", clean_q, flags=re.IGNORECASE).strip()
    clean_q = re.sub(r"^(?:the|a|an)\s+", "", clean_q, flags=re.IGNORECASE).strip()
    clean_q = re.sub(r"[\.\;\,]+$", "", clean_q).strip()
    clean_q = re.sub(r"\s+(rtl|module|component|block|unit|spec|specification|architecture|diagram|code)$", "", clean_q, flags=re.IGNORECASE).strip()

    generic_terms = {
        "efs", "efs ir", "efsir", "design", "rtl", "code", "system", "all", "top", "main",
        "generateefs", "generate efs", "generate efs ir", "plantuml", "puml", "diagram",
        "architecture", "structure", "overview"
    }

    if clean_q and clean_q.lower() not in generic_terms:
        words = clean_q.split()
        filtered_words = [w for w in words if w.lower() not in ("for", "the", "a", "an", "of", "in", "to", "architecture", "diagram", "code")]
        if filtered_words and len(filtered_words) <= 5 and len(" ".join(filtered_words)) <= 60:
            if len(filtered_words) <= 3 and all(w.isalnum() or "/" in w for w in filtered_words):
                target = "".join([w[0].upper() + w[1:] if "/" not in w else w.upper() for w in filtered_words])
            else:
                target = " ".join(filtered_words)

    if target and (target.lower() in generic_terms or len(target) > 60):
        target = ""

    return TaskIntent(
        task=task,
        target=target,
        interface=interface,
        language=language,
        raw_query=query,
        query_category=cat
    )


def analyze_query(query: str) -> TaskIntent:
    """
    Parse a user query into a structured TaskIntent.
    Uses LLM semantic parsing if available, with robust rule-based fallback.
    """
    if not query or not query.strip():
        return TaskIntent(task="RTL_GENERATION", target="", interface=None, language="Verilog", raw_query=query)

    # Attempt fast LLM structured extraction if client is available
    try:
        from core.extraction.llm_client import get_llm_client
        llm = get_llm_client()
        if llm and llm.available:
            system_prompt = (
                "You are an expert NLP parser for hardware design specifications.\n"
                "Extract structured intent from the user query. Output strictly JSON:\n"
                "{\n"
                '  "task": "RTL_GENERATION" | "PLANTUML_GENERATION" | "SVA_GENERATION" | "UVM_GENERATION" | "VHDL_GENERATION" | "SYSTEMVERILOG_GENERATION" | "TESTBENCH_GENERATION" | "FSM_GENERATION",\n'
                '  "target": "target component name or empty string if top-level/generic design",\n'
                '  "interface": "discovered interface name or null",\n'
                '  "language": "Verilog" | "SystemVerilog" | "VHDL" | "PlantUML"\n'
                "}"
            )
            user_prompt = f"Query: \"{truncate_for_log(query, 120)}\""
            res = llm.complete_json(system_prompt, user_prompt)
            if res and isinstance(res, dict) and "task" in res:
                task = str(res.get("task", "RTL_GENERATION")).upper()
                if task not in SUPPORTED_TASKS:
                    task = "RTL_GENERATION"
                target = str(res.get("target") or "").strip()
                if target.lower() in ("null", "none", "system", "top", "all", "design", "efs", "plantuml", "architecture") or len(target) > 60:
                    target = ""
                elif target and " " in target and len(target.split()) <= 3:
                    words = target.split()
                    if all(w.isalnum() for w in words):
                        target = "".join([w.capitalize() for w in words])
                interface = res.get("interface")
                if interface:
                    interface = str(interface).strip()
                    if interface.lower() in ("null", "none", ""):
                        interface = None

                lang_raw = str(res.get("language", "Verilog")).strip()
                if "systemverilog" in lang_raw.lower() or lang_raw.lower() == "sv":
                    language = "SystemVerilog"
                elif "verilog" in lang_raw.lower():
                    language = "Verilog"
                elif "vhdl" in lang_raw.lower():
                    language = "VHDL"
                elif "plantuml" in lang_raw.lower() or "puml" in lang_raw.lower():
                    language = "PlantUML"
                else:
                    language = lang_raw.capitalize()

                intent = TaskIntent(
                    task=task,
                    target=target,
                    interface=interface,
                    language=language,
                    raw_query=query
                )
                logger.info(f"LLM Query parsed: intent task={intent.task}, target='{intent.target}', interface='{intent.interface}', language='{intent.language}', query_len={len(query)}")
                return intent
    except Exception as e:
        logger.debug(f"LLM query parsing skipped/failed: {e}. Falling back to rule-based parser.")

    intent = _analyze_query_rule_based(query)
    logger.info(f"Rule-based Query parsed: intent task={intent.task}, target='{intent.target}', interface='{intent.interface}', language='{intent.language}', query_len={len(query)}")
    return intent



