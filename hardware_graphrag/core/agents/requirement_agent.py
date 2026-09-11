"""
Requirement Understanding Agent.

Parses natural language requirements into a structured, machine-readable
Requirement Model containing interfaces, registers, state machines, and constraints.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from config import CONFIG
from core.extraction.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("agents.requirement")

SYSTEM_PROMPT = """## Specification → EFS JSON IR Extraction Prompt

You are a Specification Extraction Engine.

Your task is to convert an arbitrary engineering/functional specification document into a structured JSON Intermediate Representation (IR) that can be consumed by downstream RTL generation, verification, and documentation agents.

The specification may describe any kind of digital hardware system. It may contain modules, components, interfaces, signals, registers, instructions, state machines, protocols, timing requirements, algorithms, memory behavior, error conditions, performance requirements, or other engineering information.

### PRIMARY OBJECTIVE

Extract information from the specification faithfully and completely.
The JSON IR must represent what the specification actually says.
Do NOT design the hardware yourself.
Do NOT infer missing architecture.
Do NOT invent signals, registers, states, opcodes, protocols, timing, widths, reset values, or behaviors.
Do NOT silently resolve contradictions.
If the specification is ambiguous, incomplete, or contradictory, explicitly represent that condition in the JSON.

1. SOURCE-GROUNDED EXTRACTION
Preserve source section, location, excerpt, and confidence score (0.0 to 1.0) for extracted objects.

2. EXTRACTION VS INFERENCE
Set "extraction_type": "explicit" or "inferred". Do not convert optional statements ("may", "typically", "can") into mandatory requirements.

3. PRESERVE CONTRADICTIONS
Record conflicting statements in "contradictions": [{ "topic": "string", "statements": [...], "severity": "low|medium|high|critical", "resolution": null }].

4. DO NOT INVENT DEFAULTS
Never automatically create default signals/states (clk, rst, IDLE, RUN, DONE, req, ack, data_in, data_out, 32, 64, 0) unless explicitly present in source text. Represent unspecified fields as null.

5. CAPTURE THE COMPLETE SYSTEM
Extract metadata, components, interfaces, signals, clocks, resets, parameters, registers, instructions, fsms, transactions, functional_behavior, memory, protocol_rules, timing_requirements, performance_requirements, errors, constraints, examples, requirements, ambiguities, contradictions, and extraction_quality.
Do not accidentally treat table headers as hardware objects.

Return ONLY a valid JSON object matching the full EFS IR Extraction schema. No Markdown fences, no conversational prose, no comments outside JSON.
"""


def analyze_requirements(requirement_text: str) -> Dict[str, Any]:
    """
    Parse requirement text by converting it into a canonical DocumentIR
    and processing it through GenericRequirementExtractor.
    Strictly avoids sending uncapped raw text directly to LLM.
    """
    logger.info("Analyzing requirement specification via Document IR & Generic Extractor...")
    from core.requirement_ir.models import DocumentIR, DocumentBlock, SourceLocation
    from core.extraction.requirement_extractor import GenericRequirementExtractor
    from core.compiler.efs_compiler import RequirementToEFSCompiler

    # Split text into structured DocumentBlocks by section/paragraph
    raw_blocks = [p.strip() for p in requirement_text.split("\n\n") if p.strip()]
    if not raw_blocks:
        raw_blocks = [requirement_text] if requirement_text.strip() else []

    doc_blocks = []
    for i, b_text in enumerate(raw_blocks):
        doc_blocks.append(DocumentBlock(
            block_id=f"REQ_BLK_{i+1:03d}",
            type="requirement_text",
            section_id="SEC_REQ",
            section_title="Specification Core Requirements",
            page=1,
            text=b_text,
            source_location=SourceLocation(
                document_id="text_spec",
                section="Specification Core Requirements",
                section_id="SEC_REQ",
                page=1,
                block_id=f"REQ_BLK_{i+1:03d}"
            )
        ))

    doc_ir = DocumentIR(
        document_id="text_spec",
        document_version="1.0",
        source_type="text",
        sections=[{"section_id": "SEC_REQ", "title": "Specification Core Requirements"}],
        blocks=doc_blocks
    )

    try:
        extractor = GenericRequirementExtractor(doc_ir)
        req_ir = extractor.extract_requirement_ir()
        compiler = RequirementToEFSCompiler(req_ir)
        efs_ir = compiler.compile()
        full_result = req_ir.to_full_combined_dict() if hasattr(req_ir, "to_full_combined_dict") else efs_ir.to_dict()
        if full_result and (full_result.get("interfaces") or full_result.get("requirements")):
            return full_result
    except Exception as e:
        logger.warning(f"Generic extraction failed for requirement text: {e}")

    # Rule-based fallback
    return get_heuristic_fallback(requirement_text)


# Alias for backward compatibility
parse_requirement_specification = analyze_requirements


def get_heuristic_fallback(text: str) -> Dict[str, Any]:
    """Extremely simple heuristic fallback when LLM is unavailable."""
    import re
    lines = text.split("\n")
    functional = []
    interfaces = []
    
    # Extract submodules from section headers and text
    submodules = []
    header_matches = re.findall(r"##\s*\d*\.?\s*([A-Za-z0-9_\s]+?)(?:\s+Submodule|\s+Block|\s+Module|\n|$)", text, re.IGNORECASE)
    for m in header_matches:
        clean = m.strip()
        words = [w.capitalize() for w in clean.split() if w.lower() not in ("system", "overview", "architecture", "hardware")]
        if words:
            submodules.append({"name": "".join(words), "purpose": f"{clean} submodule"})

    module_matches = re.findall(r"\b([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*(?:\s+Adder|\s+Subtractor|\s+Multiplier|\s+Controller|\s+Engine|\s+Unit|\s+File|\s+DMA|\s+Scheduler))\b", text)
    for m in module_matches:
        name = "".join([w.capitalize() for w in m.strip().split()])
        if name and not any(s["name"] == name for s in submodules):
            submodules.append({"name": name, "purpose": f"{m.strip()} hardware module"})

    # Try to extract bullet points or simple keywords
    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            continue
        # Check for bullet point interface signals (e.g. - instr_in: input, 32 bits, Instruction word)
        bullet_match = re.match(r"^[-*]\s*([a-zA-Z0-9_`]+)\s*:\s*(input|output|inout|in|out)\s*,\s*(\d+|\d+:\d+|\w+)\s*(?:bits?|bit)?(?:\s*,\s*(.*))?", line_strip, re.IGNORECASE)
        if bullet_match:
            clean_name = bullet_match.group(1).strip("` \t\r\n'\"")
            interfaces.append({
                "name": clean_name,
                "direction": bullet_match.group(2).lower(),
                "width": bullet_match.group(3),
                "description": bullet_match.group(4) or ""
            })
            continue

        if line_strip.startswith(("-", "*", "1.", "2.", "3.")):
            functional.append(line_strip.lstrip("-*123456789. "))

    return {
        "functional_requirements": functional or ["Design a hardware block matching specified interface characteristics."],
        "submodules": submodules,
        "interfaces": interfaces or [
            {"name": "clk", "direction": "input", "width": "1", "description": "System clock"},
            {"name": "rst_n", "direction": "input", "width": "1", "description": "System reset"},
        ],
        "protocol_references": ["General Interface"],
        "timing_constraints": ["Ensure setup/hold times are met for target frequency"],
        "clocks_resets": [
            {"name": "clk", "type": "clock", "active_level": "rising", "description": "Main clock"},
            {"name": "rst_n", "type": "reset", "active_level": "low", "description": "Main reset"}
        ],
        "registers": [],
        "memories": [],
        "fsm_info": {
            "states": [],
            "transitions": []
        }
    }

