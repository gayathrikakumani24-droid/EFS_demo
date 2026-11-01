"""
Centralized Target Normalization Utility.

Provides a unified normalization function for hardware target lookup,
alias generation, canonicalization, knowledge graph matching, and target resolution.
"""

from __future__ import annotations

import re


def normalize_target_name(name: str) -> str:
    """
    Normalize target/component names deterministically.
    
    Converts to lowercase and strips all non-alphanumeric characters
    (spaces, underscores, hyphens, punctuation).
    
    Examples:
        normalize_target_name("Instruction Decoder") -> "instructiondecoder"
        normalize_target_name("InstructionDecoder") -> "instructiondecoder"
        normalize_target_name("instruction_decoder") -> "instructiondecoder"
        normalize_target_name("instruction-decoder") -> "instructiondecoder"
        normalize_target_name("InstrucionDecoder") -> "instruciondecoder"
    """
    if not name:
        return ""
    return re.sub(r"[^a-zA-Z0-9]", "", str(name)).lower()
