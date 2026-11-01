"""
Semantic Requirement Chunker (Sections 11 & 12).

Groups atomic requirements by coherent logical requirement groups (transactions,
interfaces, FSMs, registers, timing) rather than arbitrary token counts.
"""

from __future__ import annotations

from typing import Dict, List
from core.requirement_ir.models import (
    AtomicRequirement, SemanticChunk, DiscoveredEntity, new_req_id
)
from utils.logger import get_logger

logger = get_logger("requirement_ir.chunker")


def build_semantic_chunks(
    requirements: List[AtomicRequirement],
    entities: List[DiscoveredEntity]
) -> List[SemanticChunk]:
    """Group atomic requirements into logical semantic units."""
    type_groups: Dict[str, List[AtomicRequirement]] = {}
    for req in requirements:
        cat = _determine_chunk_category(req)
        type_groups.setdefault(cat, []).append(req)

    chunks: List[SemanticChunk] = []

    for cat, reqs in type_groups.items():
        chunk_id = new_req_id("chk")

        # Collect entities participating in these requirements
        participating_entities = set()
        source_blocks = set()
        summaries = []

        for r in reqs:
            participating_entities.update(r.entities)
            if r.subject:
                participating_entities.add(r.subject)
            if r.source.block_id:
                source_blocks.add(r.source.block_id)
            summaries.append(r.statement)

        chunk_content = "\n".join([f"- [{r.requirement_id}] {r.statement}" for r in reqs])
        summary_text = f"Semantic group for {cat} containing {len(reqs)} requirements."

        chunk = SemanticChunk(
            chunk_id=chunk_id,
            chunk_type=cat,
            requirement_ids=[r.requirement_id for r in reqs],
            entity_ids=sorted(list(participating_entities)),
            dependencies=[],
            source_blocks=sorted(list(source_blocks)),
            summary=summary_text,
            content=chunk_content
        )
        chunks.append(chunk)

    return chunks


def _determine_chunk_category(req: AtomicRequirement) -> str:
    """Map requirement attributes to a generic chunk type."""
    t = req.type.upper()
    stmt = req.statement.lower()

    if t in ("INTERFACE", "PORT", "SIGNAL") or "interface" in stmt or "port" in stmt or "pin" in stmt:
        return "INTERFACE"
    elif t in ("TRANSACTION", "TRANSFER", "HANDSHAKE") or "transaction" in stmt or "handshake" in stmt:
        return "TRANSACTION_FLOW"
    elif t in ("STATE", "TRANSITION", "FSM") or "state" in stmt or "fsm" in stmt or "transition" in stmt:
        return "STATE_MACHINE"
    elif t in ("REGISTER", "FIELD") or "register" in stmt or "offset" in stmt or "bitfield" in stmt:
        return "REGISTER"
    elif t in ("TIMING", "CLOCK", "RESET") or "cycle" in stmt or "latency" in stmt or "clock" in stmt or "reset" in stmt:
        return "TIMING"
    elif t in ("ERROR", "EXCEPTION", "INTERRUPT") or "error" in stmt or "fault" in stmt or "interrupt" in stmt:
        return "ERROR_HANDLING"
    elif t in ("MEMORY", "BUFFER", "FIFO") or "memory" in stmt or "fifo" in stmt or "buffer" in stmt:
        return "MEMORY"
    elif t in ("PROTOCOL_RULE", "CONSTRAINT") or "must" in stmt or "shall" in stmt:
        return "PROTOCOL_RULE"
    else:
        return "GENERAL"
