"""
Chunk Enricher: Assembles enriched JSON for chunks containing mapped entities,
signals, registers, and interfaces from the ingestion registry and active EFS IR.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from config import CONFIG
from core.registry import get_registry, Registry
from core.efs_ir.models import EFSIR
from utils.models import Chunk, Entity
from utils.logger import get_logger

logger = get_logger("chunk_enricher")


def _load_active_efs_ir() -> Optional[EFSIR]:
    """Load active EFS IR if present."""
    ir_path = os.path.join(CONFIG.data_dir, "efs_ir", "active_design_ir.json")
    if os.path.exists(ir_path):
        try:
            with open(ir_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return EFSIR.from_dict(data)
        except Exception as e:
            logger.warning(f"Could not load active EFS IR: {e}")
    return None


def enrich_chunk(chunk: Chunk, registry: Optional[Registry] = None, efs_ir: Optional[EFSIR] = None) -> Dict[str, Any]:
    """
    Build an enriched dictionary representation of a chunk including:
    - Raw chunk metadata & text
    - Extracted entities mapped to this chunk
    - Signals mapped to this chunk
    - Registers mapped to this chunk
    - Interfaces mapped to this chunk
    """
    if registry is None:
        registry = get_registry()
    if efs_ir is None:
        efs_ir = _load_active_efs_ir()

    # 1. Gather all entities for this chunk
    doc_entities: List[Entity] = registry.entities.get(chunk.doc_id, [])
    chunk_entities = [e for e in doc_entities if e.chunk_id == chunk.chunk_id]
    
    # Fallback match by exact text presence if empty
    if not chunk_entities and doc_entities:
        c_text = chunk.text.lower()
        chunk_entities = [e for e in doc_entities if e.name.lower() in c_text or any(a.lower() in c_text for a in e.aliases)]

    entities_json = [e.to_dict() for e in chunk_entities]
    entity_names = {e.name.lower() for e in chunk_entities}

    # 2. Extract Signals
    signals_json: List[Dict[str, Any]] = []
    seen_sig_names = set()

    # Check EFS IR signals first
    if efs_ir and efs_ir.signals:
        for s in efs_ir.signals:
            matched = False
            if s.traceability and s.traceability.chunk_id == chunk.chunk_id:
                matched = True
            elif s.name.lower() in entity_names or (s.traceability and s.traceability.original_text and s.traceability.original_text in chunk.text):
                matched = True
            
            if matched and s.name not in seen_sig_names:
                seen_sig_names.add(s.name)
                signals_json.append(s.to_dict())

    # Complement with entity-extracted signals
    for e in chunk_entities:
        if e.entity_type.lower() == "signal" and e.name not in seen_sig_names:
            seen_sig_names.add(e.name)
            signals_json.append({
                "signal_id": f"SIG_{e.entity_id}",
                "name": e.name,
                "raw_name": e.raw_name,
                "width": 1,
                "direction": "unspecified",
                "description": e.original_text or f"Signal mention: {e.name}",
                "clock_domain": None,
                "aliases": e.aliases,
            })

    # 3. Extract Registers
    registers_json: List[Dict[str, Any]] = []
    seen_reg_names = set()

    if efs_ir and efs_ir.registers:
        for r in efs_ir.registers:
            matched = False
            if r.traceability and r.traceability.chunk_id == chunk.chunk_id:
                matched = True
            elif r.name.lower() in entity_names or (r.traceability and r.traceability.original_text and r.traceability.original_text in chunk.text):
                matched = True
            
            if matched and r.name not in seen_reg_names:
                seen_reg_names.add(r.name)
                registers_json.append(r.to_dict())

    for e in chunk_entities:
        if e.entity_type.lower() in ("register", "field") and e.name not in seen_reg_names:
            seen_reg_names.add(e.name)
            registers_json.append({
                "register_id": f"REG_{e.entity_id}",
                "name": e.name,
                "raw_name": e.raw_name,
                "address_offset": "0x00",
                "size_bits": 32,
                "access_type": "RW",
                "description": e.original_text or f"Register mention: {e.name}",
                "fields": [],
                "aliases": e.aliases,
            })

    # 4. Extract Interfaces
    interfaces_json: List[Dict[str, Any]] = []
    seen_if_names = set()

    if efs_ir and efs_ir.interfaces:
        for i in efs_ir.interfaces:
            matched = False
            if i.traceability and i.traceability.chunk_id == chunk.chunk_id:
                matched = True
            elif i.name.lower() in entity_names or (i.traceability and i.traceability.original_text and i.traceability.original_text in chunk.text):
                matched = True
            
            if matched and i.name not in seen_if_names:
                seen_if_names.add(i.name)
                interfaces_json.append(i.to_dict())

    for e in chunk_entities:
        if e.entity_type.lower() in ("interface", "channel", "protocol") and e.name not in seen_if_names:
            seen_if_names.add(e.name)
            interfaces_json.append({
                "interface_id": f"IF_{e.entity_id}",
                "name": e.name,
                "raw_name": e.raw_name,
                "protocol": "generic",
                "role": "unspecified",
                "description": e.original_text or f"Interface mention: {e.name}",
                "aliases": e.aliases,
            })

    # 5. Return complete enriched dict
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "page": chunk.page,
        "chapter": chunk.chapter,
        "section": chunk.section,
        "subsection": chunk.subsection,
        "heading": chunk.heading,
        "content_type": chunk.content_type,
        "previous_chunk": chunk.previous_chunk,
        "next_chunk": chunk.next_chunk,
        "overlap_prefix": chunk.overlap_prefix,
        "text": chunk.text,
        "entities": entities_json,
        "signals": signals_json,
        "registers": registers_json,
        "interfaces": interfaces_json,
    }


def get_all_enriched_chunks(doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve all chunks enriched with entities, signals, registers, and interfaces."""
    registry = get_registry()
    efs_ir = _load_active_efs_ir()
    
    if doc_id:
        chunks = registry.chunks.get(doc_id, [])
    else:
        chunks = registry.all_chunks()

    return [enrich_chunk(c, registry=registry, efs_ir=efs_ir) for c in chunks]


def export_enriched_chunks_json(output_path: Optional[str] = None, doc_id: Optional[str] = None) -> str:
    """Save enriched chunks JSON to disk."""
    if output_path is None:
        output_path = os.path.join(CONFIG.data_dir, "efs_ir", "enriched_chunks.json")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = get_all_enriched_chunks(doc_id=doc_id)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Exported {len(data)} enriched chunks to {output_path}")
    return output_path
