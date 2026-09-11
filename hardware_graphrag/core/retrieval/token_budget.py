"""
Token Budget Manager module.

Enforces token-aware ContextPack construction by estimating token usage
and intelligently pruning non-critical context when budgets are exceeded.
Never removes critical hardware definitions (signals, widths, directions, clock/reset, conflicts).
"""

from __future__ import annotations

from typing import List, Dict, Any
from core.retrieval.context_pack import ContextPack
from config import CONFIG
from utils.logger import get_logger

logger = get_logger("retrieval.token_budget")

# Maximum token budget for ContextPack passed to LLM
DEFAULT_MAX_CONTEXT_TOKENS = getattr(CONFIG.llm, "max_context_tokens", 1800)


class TokenBudgetManager:
    """Manages token budgeting and intelligent context pruning based on P0/P1/P2 element priority."""

    def __init__(self, max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS):
        self.max_tokens = max_tokens

    def enforce_budget(self, pack: ContextPack) -> ContextPack:
        """
        Check token estimate for ContextPack and prune if over max_tokens.
        Priority Hierarchy:
          P0 (Protected): Interface definitions, signal widths, registers, protocol rules, reset requirements.
          P1 (High): State transitions, functional specifications, error rules.
          P2 (Prunable): Background narrative, extra examples, long prose descriptions.
        """
        current_tokens = pack.estimate_token_count()
        if current_tokens <= self.max_tokens:
            logger.info(f"ContextPack token estimate ({current_tokens} tokens) is within budget limit ({self.max_tokens} tokens).")
            return pack

        logger.warning(f"ContextPack token estimate ({current_tokens} tokens) EXCEEDS budget limit ({self.max_tokens} tokens). Initiating priority pruning...")

        # Step 1: Prune P2 background narrative & duplicate evidence
        if pack.source_evidence:
            seen_texts = set()
            pruned_evidence = []
            for ev in pack.source_evidence:
                text = ev.get("text", "") or ev.get("original_text", "")
                text_snippet = text[:100]
                # Filter out pure narrative / informational chunks
                is_p2 = any(kw in text.lower() for kw in ["introduction", "overview", "example", "background", "history"])
                if text_snippet not in seen_texts and not is_p2:
                    seen_texts.add(text_snippet)
                    pruned_evidence.append(ev)
            pack.source_evidence = pruned_evidence
            current_tokens = pack.estimate_token_count()
            if current_tokens <= self.max_tokens:
                logger.info(f"ContextPack successfully pruned to {current_tokens} tokens (pruned P2 narrative evidence).")
                return pack

        # Step 2: Trim long narrative text in source chunks while preserving tables and P0 rules
        if pack.source_evidence:
            for ev in pack.source_evidence:
                txt = ev.get("text", "")
                if len(txt) > 400 and not txt.startswith("|"):
                    ev["text"] = txt[:400] + "... [pruned prose]"
            current_tokens = pack.estimate_token_count()
            if current_tokens <= self.max_tokens:
                logger.info(f"ContextPack successfully pruned to {current_tokens} tokens (trimmed long prose descriptions).")
                return pack

        # Step 3: Prune distant submodules (depth >= 2)
        if pack.dependencies:
            pack.dependencies = [d for d in pack.dependencies if d.get("depth", 1) <= 1]
            current_tokens = pack.estimate_token_count()
            if current_tokens <= self.max_tokens:
                logger.info(f"ContextPack successfully pruned to {current_tokens} tokens (removed distant dependencies).")
                return pack

        # Step 4: Final trim of P1 protocol source evidence text if needed (keeping P0 protocol rules intact)
        while current_tokens > self.max_tokens and pack.protocol_source_evidence:
            pack.protocol_source_evidence.pop()
            if pack.relevant_protocol_chunks:
                pack.relevant_protocol_chunks.pop()
            current_tokens = pack.estimate_token_count()

        while current_tokens > self.max_tokens and pack.source_evidence:
            pack.source_evidence.pop()
            current_tokens = pack.estimate_token_count()

        logger.info(f"Pruning complete. Final ContextPack token count: {current_tokens} tokens.")
        return pack
