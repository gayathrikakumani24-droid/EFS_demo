"""
Step 5: Entity Normalization.

Merges duplicate / near-duplicate entity mentions (e.g. "Write Address",
"Write Address Channel", "WA Channel") into a single canonical name
(e.g. "WRITE_ADDRESS_CHANNEL") before the entities and their relationships
are inserted into Neo4j / the vector store's linked-entity metadata.

Approach:
  1. Deterministic normalization: uppercase + underscore canonical form,
     common hardware abbreviation expansion (WA -> WRITE_ADDRESS, etc).
  2. Fuzzy clustering: entities of the same type whose canonical forms
     are highly similar (token overlap / substring / edit-distance ratio)
     are merged into one cluster, keyed by the longest / most descriptive
     member's canonical name.
  3. All raw_name variants are preserved as `aliases` on the merged Entity.
"""

from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

from utils.models import Entity
from utils.logger import get_logger

logger = get_logger("normalization.entities")

# Common hardware-spec abbreviation expansions, applied before clustering.
_ABBREVIATIONS = {
    r"\bWA\b": "WRITE ADDRESS",
    r"\bRA\b": "READ ADDRESS",
    r"\bWD\b": "WRITE DATA",
    r"\bRD\b": "READ DATA",
    r"\bWR\b": "WRITE RESPONSE",
    r"\bB\b(?=\s*CHANNEL)": "WRITE RESPONSE",
    r"\bCLK\b": "CLOCK",
    r"\bIRQ\b": "INTERRUPT",
    r"\bRST\b": "RESET",
    r"\bADDR\b": "ADDRESS",
    r"\bREG\b": "REGISTER",
}


def _canonicalize(raw_name: str) -> str:
    """Produce a deterministic UPPER_SNAKE_CASE canonical form."""
    name = raw_name.strip().upper()
    for pattern, expansion in _ABBREVIATIONS.items():
        name = re.sub(pattern, expansion, name)
    name = re.sub(r"[^A-Z0-9]+", "_", name).strip("_")
    name = re.sub(r"_+", "_", name)
    return name


# Generic structural suffix tokens that carry little disambiguating meaning on
# their own (e.g. "CHANNEL", "REGISTER"). These are stripped before comparing
# two canonical names so that e.g. "WRITE_ADDRESS_CHANNEL" and "WRITE_ADDRESS"
# are recognized as the same entity, while "WRITE_ADDRESS_CHANNEL" and
# "WRITE_RESPONSE_CHANNEL" are correctly kept distinct (they share the
# generic "CHANNEL" suffix but differ in their core, meaning-bearing tokens).
_GENERIC_SUFFIX_TOKENS = {
    "CHANNEL", "REGISTER", "INTERFACE", "SIGNAL", "FIELD", "PROTOCOL",
    "TRANSACTION", "STATE", "COMMAND", "STRUCTURE", "REGION", "CONSTRAINT",
    "DOMAIN", "FEATURE",
}


def _core_tokens(name: str) -> Tuple[str, ...]:
    """Strip trailing generic suffix tokens (e.g. 'CHANNEL') to get the core, meaning-bearing tokens."""
    tokens = [t for t in name.split("_") if t]
    while len(tokens) > 1 and tokens[-1] in _GENERIC_SUFFIX_TOKENS:
        tokens = tokens[:-1]
    return tuple(tokens)


def _similarity(a: str, b: str) -> float:
    """High-confidence textual similarity, used only to catch near-identical
    spelling variants / typos of what is otherwise the same core name."""
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _cluster_canonical_names(names: List[str], fuzzy_threshold: float = 0.92) -> Dict[str, str]:
    """
    Greedy clustering: for a list of canonical name candidates (same entity
    type), group names that refer to the same underlying concept and map
    each to the representative (longest) name in its cluster.

    Two names are merged if either:
      (a) their core tokens (after stripping generic structural suffixes
          like "CHANNEL"/"REGISTER") are identical or one is a token-prefix
          of the other (handles "WRITE_ADDRESS" vs "WRITE_ADDRESS_CHANNEL"), or
      (b) their full canonical strings are near-identical (>= fuzzy_threshold),
          which only catches spelling variants, not conceptually different names.

    This intentionally does NOT merge purely on generic-suffix overlap, so
    "WRITE_ADDRESS_CHANNEL" and "WRITE_RESPONSE_CHANNEL" stay separate.
    """
    unique_names = sorted(set(names), key=len, reverse=True)
    clusters: List[List[str]] = []

    def cores_compatible(core_a: Tuple[str, ...], core_b: Tuple[str, ...]) -> bool:
        if not core_a or not core_b:
            return False
        shorter, longer = (core_a, core_b) if len(core_a) <= len(core_b) else (core_b, core_a)
        return longer[: len(shorter)] == shorter

    for name in unique_names:
        core = _core_tokens(name)
        placed = False
        for cluster in clusters:
            for member in cluster:
                if cores_compatible(core, _core_tokens(member)) or _similarity(name, member) >= fuzzy_threshold:
                    cluster.append(name)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            clusters.append([name])

    mapping: Dict[str, str] = {}
    for cluster in clusters:
        # representative = longest name (usually the most descriptive / least abbreviated)
        representative = max(cluster, key=len)
        for member in cluster:
            mapping[member] = representative
    return mapping


def normalize_entities(entities: List[Entity]) -> List[Entity]:
    """
    Normalize + merge duplicate entities.

    Returns a new list of Entity objects (one per merged cluster, per doc),
    each carrying `aliases` listing every raw_name variant that was merged.
    Individual per-chunk Entity records are preserved as separate objects
    (for graph provenance) but all share the same `.name` after this call
    so downstream code can group/merge by `.name`.
    """
    if not entities:
        return []

    # Step A: deterministic canonicalization, grouped by entity_type
    by_type: Dict[str, List[str]] = defaultdict(list)
    canon_by_entity: Dict[str, str] = {}
    for e in entities:
        canon = _canonicalize(e.raw_name)
        canon_by_entity[e.entity_id] = canon
        by_type[e.entity_type].append(canon)

    # Step B: fuzzy cluster within each type
    representative_by_canon: Dict[Tuple[str, str], str] = {}
    for etype, names in by_type.items():
        mapping = _cluster_canonical_names(names)
        for name, rep in mapping.items():
            representative_by_canon[(etype, name)] = rep

    # Step C: assign final normalized name + collect aliases
    aliases_by_final_name: Dict[str, set] = defaultdict(set)
    normalized: List[Entity] = []
    for e in entities:
        canon = canon_by_entity[e.entity_id]
        final_name = representative_by_canon.get((e.entity_type, canon), canon)
        aliases_by_final_name[final_name].add(e.raw_name)
        e.name = final_name
        normalized.append(e)

    for e in normalized:
        e.aliases = sorted(a for a in aliases_by_final_name[e.name] if a.upper().replace(" ", "_") != e.name)

    n_unique = len({e.name for e in normalized})
    logger.info(f"Normalized {len(entities)} raw entity mentions into {n_unique} unique canonical entities.")
    return normalized
