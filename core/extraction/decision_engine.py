"""
Centralized Escalation Decision Engine module (Step 5).

Determines whether a chunk requires LLM escalation or if deterministic
heuristic extraction results are sufficient based on confidence, complexity,
unresolved pronoun/cross references, and semantic ambiguity.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional
from config import CONFIG
from core.extraction.pre_classifier import ClassificationResult
from core.extraction.heuristic_extractor import HeuristicExtractionResult
from utils.logger import get_logger

logger = get_logger("extraction.decision_engine")

_PRONOUN_REF_RE = re.compile(
    r"\b(it|this|that|these|those|the former|the latter|aforementioned|described above|see section)\b",
    re.IGNORECASE
)

_AMBIGUOUS_LANG_RE = re.compile(
    r"\b(depending on|typically|generally|optionally|may|might|subject to|otherwise|unless specified|implementation dependent)\b",
    re.IGNORECASE
)


def needs_llm(
    text: str,
    classification: ClassificationResult,
    heuristic_res: HeuristicExtractionResult,
    enable_llm_escalation: Optional[bool] = None
) -> bool:
    """
    Decide if chunk requires LLM escalation.
    
    Returns True if LLM is required, False if heuristic output is sufficient.
    """
    if enable_llm_escalation is None:
        enable_llm_escalation = getattr(CONFIG, "enable_llm_escalation", True)

    if not enable_llm_escalation:
        logger.debug("LLM escalation is explicitly disabled in config.")
        return False

    # 1. Non-relevant chunks never need LLM
    if not classification.relevant:
        return False

    high_thresh = getattr(CONFIG, "heuristic_high_confidence", 0.90)

    # 2. High-confidence heuristic extraction with low/medium complexity & no unresolved refs -> No LLM needed
    if (heuristic_res.confidence >= high_thresh and
        not classification.has_unresolved_references and
        classification.complexity != "high"):
        logger.debug(f"Accepted heuristic extraction (conf={heuristic_res.confidence:.2f}, complexity={classification.complexity}). Skipping LLM.")
        return False

    # 3. Escalation triggers:
    # Trigger A: Unresolved pronouns or cross-references ("it", "the previous transaction")
    if classification.has_unresolved_references or _PRONOUN_REF_RE.search(text):
        logger.debug("Escalating to LLM: Unresolved references or pronouns detected.")
        return True

    # Trigger B: High complexity or multi-clause ambiguous language
    if classification.complexity == "high" or _AMBIGUOUS_LANG_RE.search(text):
        logger.debug("Escalating to LLM: High chunk complexity or ambiguous conditional language.")
        return True

    # Trigger C: State machine semantics or complex transitions that heuristics missed
    if "state_transition" in classification.categories and not heuristic_res.requirements:
        logger.debug("Escalating to LLM: State machine semantics detected without explicit heuristic requirement.")
        return True

    # Trigger D: Low confidence heuristic extraction
    med_thresh = getattr(CONFIG, "heuristic_medium_confidence", 0.70)
    if heuristic_res.confidence < med_thresh:
        logger.debug(f"Escalating to LLM: Low heuristic confidence ({heuristic_res.confidence:.2f} < {med_thresh}).")
        return True

    return False
