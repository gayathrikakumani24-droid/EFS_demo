"""
Step 4: Relationship Extraction.

Given a chunk and the entities already extracted from it, identify
semantic relationships between those entities, expressed as structured
(source, relation, target) triples using the relation vocabulary in
utils.models.RELATIONSHIP_TYPES.

Primary path: LLM prompted with the chunk text + candidate entity list,
constrained to only emit relations between entities that were actually
extracted (avoids hallucinated nodes). Fallback: simple co-occurrence +
keyword-cue heuristic (e.g. "handshake" -> HANDSHAKES_WITH, "requires" ->
REQUIRES) between entities that appear in the same chunk.
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import List

from config import CONFIG
from core.extraction.llm_client import get_llm_client
from utils.models import Chunk, Entity, Relationship, RELATIONSHIP_TYPES, new_id
from utils.logger import get_logger

logger = get_logger("extraction.relationships")

_SYSTEM_PROMPT = f"""You are an expert in digital hardware protocols and knowledge-graph construction.
Given a text chunk of a hardware spec and a list of identified entities in it, extract semantic relationships between those entities only.

Valid relationship types:
{", ".join(RELATIONSHIP_TYPES)}

Key relationship categories to extract:
- Channel & Signal: (:Protocol)-[:HAS_CHANNEL]->(:Channel), (:Channel)-[:HAS_SIGNAL]->(:Signal), (:Transaction)-[:USES_CHANNEL]->(:Channel), (:Transaction)-[:USES_SIGNAL]->(:Signal)
- Transaction & Component: (:Transaction)-[:INITIATED_BY]->(:Master), (:Transaction)-[:TARGETS]->(:Slave), (:Transaction)-[:NEXT_STEP]->(:TransactionStep), (:Transaction)-[:FOLLOWED_BY]->(:Transaction)
- Rules & Conditions: (:ProtocolRule)-[:APPLIES_TO]->(:Transaction), (:ProtocolRule)-[:CONSTRAINS]->(:Signal), (:Condition)-[:ENABLES]->(:Transaction), (:Transaction)-[:PRODUCES]->(:Response), (:Transaction)-[:MAY_CAUSE]->(:Error)

Return ONLY a JSON object of the form:
{{"relationships": [{{"source": "<entity name>", "relation": "<one of valid types>", "target": "<entity name>", "evidence": "<short verbatim snippet>"}}]}}

Rules:
- source and target MUST be exact names from the provided entity list.
- Only emit relationships clearly supported by the text chunk.
- If nothing qualifies, return {{"relationships": []}}.
"""

_KEYWORD_CUES = [
    (re.compile(r"\bhas channel\b", re.IGNORECASE), "HAS_CHANNEL"),
    (re.compile(r"\bhas signal\b", re.IGNORECASE), "HAS_SIGNAL"),
    (re.compile(r"\buses channel\b", re.IGNORECASE), "USES_CHANNEL"),
    (re.compile(r"\buses signal\b", re.IGNORECASE), "USES_SIGNAL"),
    (re.compile(r"\binitiated by\b", re.IGNORECASE), "INITIATED_BY"),
    (re.compile(r"\btargets\b", re.IGNORECASE), "TARGETS"),
    (re.compile(r"\bnext step\b", re.IGNORECASE), "NEXT_STEP"),
    (re.compile(r"\bfollowed by\b", re.IGNORECASE), "FOLLOWED_BY"),
    (re.compile(r"\bapplies to\b", re.IGNORECASE), "APPLIES_TO"),
    (re.compile(r"\bconstrains\b", re.IGNORECASE), "CONSTRAINS"),
    (re.compile(r"\benables\b", re.IGNORECASE), "ENABLES"),
    (re.compile(r"\bmay cause\b", re.IGNORECASE), "MAY_CAUSE"),
    (re.compile(r"\bdefines\b", re.IGNORECASE), "DEFINES"),
    (re.compile(r"\bhandshake", re.IGNORECASE), "HANDSHAKES_WITH"),
    (re.compile(r"\brequires?\b", re.IGNORECASE), "REQUIRES"),
    (re.compile(r"\bdepends? on\b", re.IGNORECASE), "DEPENDS_ON"),
    (re.compile(r"\bconnect(?:s|ed)? to\b", re.IGNORECASE), "CONNECTS_TO"),
    (re.compile(r"\buses?\b", re.IGNORECASE), "USES"),
    (re.compile(r"\bconfigures?\b", re.IGNORECASE), "CONFIGURES"),
    (re.compile(r"\breads? from\b", re.IGNORECASE), "READS_FROM"),
    (re.compile(r"\bwrites? to\b", re.IGNORECASE), "WRITES_TO"),
    (re.compile(r"\bgenerates?\b", re.IGNORECASE), "GENERATES"),
    (re.compile(r"\btransitions? to\b", re.IGNORECASE), "TRANSITIONS_TO"),
    (re.compile(r"\basserted before\b", re.IGNORECASE), "ASSERTED_BEFORE"),
    (re.compile(r"\basserted after\b", re.IGNORECASE), "ASSERTED_AFTER"),
    (re.compile(r"\brefers? to|references?\b", re.IGNORECASE), "REFERENCES"),
    (re.compile(r"\bdefined in\b", re.IGNORECASE), "DEFINED_IN"),
    (re.compile(r"\bpart of\b", re.IGNORECASE), "PART_OF"),
    (re.compile(r"\bbelongs to\b", re.IGNORECASE), "BELONGS_TO"),
]


def _heuristic_extract(chunk: Chunk, entities: List[Entity]) -> List[Relationship]:
    if len(entities) < 2:
        return []
    relations: List[Relationship] = []
    text = chunk.text
    for e1, e2 in combinations(entities, 2):
        # only relate entities whose mentions are reasonably close together in the text
        idx1 = text.lower().find(e1.name.lower())
        idx2 = text.lower().find(e2.name.lower())
        if idx1 == -1 or idx2 == -1:
            continue
        window_start, window_end = sorted((idx1, idx2))
        window = text[max(0, window_start - 40): window_end + 40]

        relation_type = None
        for pattern, rel in _KEYWORD_CUES:
            if pattern.search(window):
                relation_type = rel
                break
        if relation_type is None:
            # default weak relation for co-occurring entities within the same subsection
            relation_type = "REFERENCES"

        relations.append(
            Relationship(
                rel_id=new_id("rel"),
                source=e1.name,
                relation=relation_type,
                target=e2.name,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                evidence=window.strip()[:200],
            )
        )
    return relations


def extract_relationships(chunk: Chunk, entities: List[Entity]) -> List[Relationship]:
    """Extract relationships among the given entities, constrained to this chunk's text."""
    if len(entities) < 2:
        return []

    llm = get_llm_client()
    if llm.available:
        entity_names = sorted({e.name for e in entities})
        user_prompt = (
            f"Entities: {entity_names}\n\nText:\n{chunk.text}"
        )
        result = llm.complete_json(_SYSTEM_PROMPT, user_prompt, model=CONFIG.llm.extraction_model)
        if result and isinstance(result.get("relationships"), list):
            valid_names = set(entity_names)
            rels = []
            for item in result["relationships"]:
                if not isinstance(item, dict):
                    continue
                rel_type = (item.get("relation") or "").strip()
                source = (item.get("source") or "").strip()
                target = (item.get("target") or "").strip()
                if rel_type not in RELATIONSHIP_TYPES:
                    continue
                if source not in valid_names or target not in valid_names or source == target:
                    continue
                evidence = item.get("evidence", "")
                if not isinstance(evidence, str):
                    evidence = ""
                rels.append(
                    Relationship(
                        rel_id=new_id("rel"),
                        source=source,
                        relation=rel_type,
                        target=target,
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        evidence=evidence,
                    )
                )
            if rels:
                return rels
        logger.debug(f"LLM produced no usable relationships for chunk {chunk.chunk_id}; falling back to heuristics.")

    return _heuristic_extract(chunk, entities)


def extract_relationships_batch(chunks_with_entities) -> List[Relationship]:
    """chunks_with_entities: List[Tuple[Chunk, List[Entity]]]"""
    all_rels: List[Relationship] = []
    for chunk, entities in chunks_with_entities:
        all_rels.extend(extract_relationships(chunk, entities))
    logger.info(f"Extracted {len(all_rels)} relationships.")
    return all_rels
