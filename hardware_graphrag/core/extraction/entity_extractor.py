"""
Step 3: Entity Extraction.

For every chunk, extract structured entities of the types defined in
utils.models.ENTITY_TYPES. Primary path uses an LLM prompted to return
strict JSON; if the LLM is unavailable or fails, a rule-based fallback
extractor uses hardware-domain regex/keyword heuristics so the pipeline
still produces usable (if lower-recall) entities offline.
"""

from __future__ import annotations

import re
from typing import List

from config import CONFIG
from core.extraction.llm_client import get_llm_client
from utils.models import Chunk, Entity, ENTITY_TYPES, new_id
from utils.logger import get_logger

logger = get_logger("extraction.entities")

_SYSTEM_PROMPT = f"""You are an expert in digital hardware protocols, RTL design, verification, and knowledge-graph construction.
Your task is to analyze the given text chunk of a hardware protocol specification document and convert its relevant information into structured protocol entities suitable for Neo4j and GraphRAG.

Valid entity types:
{", ".join(ENTITY_TYPES)}

Categories of interest:
1. Protocol-Level: Protocol, Interface, Channel, Component, Master, Slave, Initiator, Target, Register, Field
2. Transactions & Steps: Transaction (Read, Write, Burst, Handshake, Reset), TransactionStep (ordered sequence step)
3. Signals & Timing: Signal (width, direction, valid/ready), State, Event, Response, Error, ClockDomain, TimingConstraint, TimingRule
4. Rules & Conditions: ProtocolRule (normative MUST, SHALL, SHOULD, REQUIRED, PROHIBITED), Condition (VALID/READY handshakes, preconditions), Constraint

Return ONLY a JSON object of the form:
{{"entities": [{{"name": "<canonical name>", "type": "<one of valid types>", "text_span": "<verbatim snippet from input>"}}]}}

Rules:
- Only extract entities explicitly supported by the text.
- Preserve exact signal names, channel names, and transaction names.
- Do not invent entities. If nothing qualifies, return {{"entities": []}}.
"""

# ------------------------------------------------------------------
# Heuristic fallback (no LLM required)
# ------------------------------------------------------------------

_HEURISTIC_PATTERNS = [
    (re.compile(r"\b([A-Z]{2,6}[- ]?(?:Bus|Protocol))\b"), "Protocol"),
    (re.compile(r"\b(AXI4?|AHB|APB|PCIe|I2C|SPI|UART|USB\d?|SATA|DDR\d?)\b"), "Protocol"),
    (re.compile(r"\b(\w+\s?Interface)\b", re.IGNORECASE), "Interface"),
    (re.compile(r"\b(Write Address Channel|Read Address Channel|Write Data Channel|Read Data Channel|Write Response Channel|WA Channel|RA Channel|WD Channel|RD Channel|WR Channel|B Channel)\b", re.IGNORECASE), "Channel"),
    (re.compile(r"\b([A-Z][A-Z0-9_]{2,}(?:VALID|READY|RESET|CLK|ENABLE|SEL|ACK|REQ|ADDR|DATA|RESP|STRB|LAST|ID|BURST|LEN|SIZE|LOCK|CACHE|PROT))\b"), "Signal"),
    (re.compile(r"\b(\w+\s?Register)\b", re.IGNORECASE), "Register"),
    (re.compile(r"\b(bit\s*\d+(?:[:\-]\d+)?)\b", re.IGNORECASE), "Field"),
    (re.compile(r"\b(Read Transaction|Write Transaction|Burst Transaction|\w+ Transaction|\w+ Transfer)\b", re.IGNORECASE), "Transaction"),
    (re.compile(r"\b(STEP_\d+|\bstep \d+\b)\b", re.IGNORECASE), "TransactionStep"),
    (re.compile(r"\b(Master|Initiator)\b", re.IGNORECASE), "Master"),
    (re.compile(r"\b(Slave|Target)\b", re.IGNORECASE), "Slave"),
    (re.compile(r"\b(MUST|SHALL|SHOULD|MUST NOT|SHALL NOT|REQUIRED|PROHIBITED)\b[^\.\n]{5,100}", re.IGNORECASE), "ProtocolRule"),
    (re.compile(r"\b(VALID\s*==?\s*1\s*&&\s*READY\s*==?\s*1|VALID\s*and\s*READY)\b", re.IGNORECASE), "Condition"),
    (re.compile(r"\b(setup time|hold time|t_su|t_h|propagation delay|latency of \w+)\b", re.IGNORECASE), "TimingConstraint"),
    (re.compile(r"\b(\w+\s?clock domain|CLK_\w+)\b", re.IGNORECASE), "ClockDomain"),
    (re.compile(r"\b(memory region|address space|\w+ RAM|\w+ ROM)\b", re.IGNORECASE), "MemoryRegion"),
    (re.compile(r"\b(0x[0-9A-Fa-f]{2,})\b"), "Address"),
    (re.compile(r"\b(\w+\s?interrupt|IRQ\d*)\b", re.IGNORECASE), "Interrupt"),
    (re.compile(r"\b(IDLE|ACTIVE|RESET|WAIT|BUSY|DONE)\s+state\b", re.IGNORECASE), "State"),
    (re.compile(r"\b([A-Z_]{3,}CMD|command \w+)\b", re.IGNORECASE), "Command"),
    (re.compile(r"\b(\w+\s?descriptor|\w+\s?FIFO|\w+\s?buffer)\b", re.IGNORECASE), "DataStructure"),
    (re.compile(r"\b(supports? [\w\- ]{3,40})\b", re.IGNORECASE), "Feature"),
]


def _heuristic_extract(chunk: Chunk) -> List[Entity]:
    found: dict = {}
    for pattern, etype in _HEURISTIC_PATTERNS:
        for m in pattern.finditer(chunk.text):
            raw_name = m.group(1).strip()
            key = (raw_name.lower(), etype)
            if key not in found:
                found[key] = raw_name
    entities = []
    for (name_lower, etype), raw_name in found.items():
        entities.append(
            Entity(
                entity_id=new_id("ent"),
                name=raw_name,
                raw_name=raw_name,
                entity_type=etype,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                page=chunk.page,
                chapter=chunk.chapter,
                section=chunk.section,
                original_text=chunk.text[:300],
            )
        )
    return entities


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def extract_entities(chunk: Chunk) -> List[Entity]:
    """Extract entities from a single chunk, preferring the LLM path."""
    llm = get_llm_client()
    if llm.available:
        user_prompt = (
            f"Chapter: {chunk.chapter}\nSection: {chunk.section}\nHeading: {chunk.heading}\n\n"
            f"Text:\n{chunk.text}"
        )
        result = llm.complete_json(_SYSTEM_PROMPT, user_prompt, model=CONFIG.llm.extraction_model)
        if result and isinstance(result.get("entities"), list):
            entities = []
            for item in result["entities"]:
                if not isinstance(item, dict):
                    continue
                etype = (item.get("type") or "").strip()
                if etype not in ENTITY_TYPES:
                    continue
                name = (item.get("name") or "").strip()
                if not name:
                    continue
                entities.append(
                    Entity(
                        entity_id=new_id("ent"),
                        name=name,
                        raw_name=name,
                        entity_type=etype,
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        page=chunk.page,
                        chapter=chunk.chapter,
                        section=chunk.section,
                        original_text=item.get("text_span", chunk.text[:300]) if isinstance(item.get("text_span"), str) else chunk.text[:300],
                    )
                )
            if entities:
                return entities
        logger.debug(f"LLM produced no usable entities for chunk {chunk.chunk_id}; falling back to heuristics.")

    return _heuristic_extract(chunk)


def extract_entities_batch(chunks: List[Chunk]) -> List[Entity]:
    all_entities: List[Entity] = []
    for chunk in chunks:
        all_entities.extend(extract_entities(chunk))
    logger.info(f"Extracted {len(all_entities)} raw entities from {len(chunks)} chunks.")
    return all_entities
