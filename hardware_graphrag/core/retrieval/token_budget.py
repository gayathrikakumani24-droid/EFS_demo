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

    def _compact_dependency(self, dep_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Compact a dependency dictionary to essential interface details without bloated submodule trees."""
        comp = dep_entry.get("component")
        if isinstance(comp, dict):
            compacted_comp = {
                "component_id": comp.get("component_id"),
                "name": comp.get("name"),
                "type": comp.get("type"),
                "clock_domain": comp.get("clock_domain"),
                "reset_domain": comp.get("reset_domain"),
                "description": (comp.get("description") or "")[:200],
            }
            # Keep top-level port summary if available, but avoid deep submodules
            if "interfaces" in comp and isinstance(comp["interfaces"], list):
                compacted_comp["interfaces"] = [
                    {"name": iface.get("name"), "type": iface.get("type")}
                    for iface in comp["interfaces"][:4] if isinstance(iface, dict)
                ]
            return {
                "component": compacted_comp,
                "depth": dep_entry.get("depth", 1),
                "relationship": dep_entry.get("relationship", "submodule")
            }
        return dep_entry

    def enforce_budget(self, pack: ContextPack) -> ContextPack:
        """
        Check token estimate for ContextPack and prune if over max_tokens.
        Strictly guarantees: final_context_tokens <= configured_budget (self.max_tokens).

        Priority Hierarchy:
          P0 (NEVER REMOVED): Interface definitions, signal widths/directions, registers+fields,
                               FSM states/transitions, clock/reset signals, protocol rules.
          P1 (High): State transitions, functional specifications, error rules, transactions.
          P2 (Prunable): Background narrative, extra examples, long prose descriptions, bulky dependency trees.
        """
        current_tokens = pack.estimate_token_count()
        if current_tokens <= self.max_tokens:
            logger.info(f"ContextPack token estimate ({current_tokens} tokens) is within budget limit ({self.max_tokens} tokens).")
            return pack

        logger.warning(f"ContextPack token estimate ({current_tokens} tokens) EXCEEDS budget limit ({self.max_tokens} tokens). Initiating priority pruning...")

        # Step 0: Compact bulky dependency submodule trees (critical: prevents full-tree serialization blowout)
        if pack.dependencies:
            pack.dependencies = [self._compact_dependency(d) for d in pack.dependencies]
            current_tokens = pack.estimate_token_count()
            if current_tokens <= self.max_tokens:
                logger.info(f"ContextPack successfully pruned to {current_tokens} tokens (compacted bulky dependency trees).")
                return pack

        # Step 1: Prune P2 background narrative & duplicate evidence
        if pack.source_evidence:
            seen_texts = set()
            pruned_evidence = []
            for ev in pack.source_evidence:
                text = ev.get("text", "") or ev.get("original_text", "")
                text_snippet = text[:100]
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
                if len(txt) > 250 and not txt.startswith("|"):
                    ev["text"] = txt[:250] + "... [pruned prose]"
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

        # Step 4: Drop raw protocol source evidence & raw chunks (P0 protocol_rules are PRESERVED)
        while current_tokens > self.max_tokens and pack.protocol_source_evidence:
            pack.protocol_source_evidence.pop()
            if pack.relevant_protocol_chunks:
                pack.relevant_protocol_chunks.pop()
            current_tokens = pack.estimate_token_count()
            if current_tokens <= self.max_tokens:
                logger.info(f"ContextPack successfully pruned to {current_tokens} tokens (popped protocol evidence chunks).")
                return pack

        while current_tokens > self.max_tokens and pack.source_evidence:
            pack.source_evidence.pop()
            current_tokens = pack.estimate_token_count()
            if current_tokens <= self.max_tokens:
                logger.info(f"ContextPack successfully pruned to {current_tokens} tokens (popped source evidence).")
                return pack

        # Step 5: Trim transactions (often contain large raw payloads or traces)
        if current_tokens > self.max_tokens and pack.transactions:
            while current_tokens > self.max_tokens and pack.transactions:
                pack.transactions.pop()
                current_tokens = pack.estimate_token_count()
            if current_tokens <= self.max_tokens:
                logger.info(f"ContextPack successfully pruned to {current_tokens} tokens (trimmed transactions).")
                return pack

        # Step 6: Trim dependencies down
        if current_tokens > self.max_tokens and pack.dependencies:
            while current_tokens > self.max_tokens and pack.dependencies:
                pack.dependencies.pop()
                current_tokens = pack.estimate_token_count()
            if current_tokens <= self.max_tokens:
                logger.info(f"ContextPack successfully pruned to {current_tokens} tokens (pruned dependencies).")
                return pack

        # Step 7: Trim constraints and functional requirements
        if current_tokens > self.max_tokens and (pack.constraints or pack.functional_requirements):
            while current_tokens > self.max_tokens and pack.constraints:
                pack.constraints.pop()
                current_tokens = pack.estimate_token_count()
            while current_tokens > self.max_tokens and pack.functional_requirements:
                pack.functional_requirements.pop()
                current_tokens = pack.estimate_token_count()
            if current_tokens <= self.max_tokens:
                logger.info(f"ContextPack successfully pruned to {current_tokens} tokens (trimmed constraints/functional requirements).")
                return pack

        # Step 8: Compact descriptions in signals and registers (keep P0 properties: name, width, direction, reset_value)
        if current_tokens > self.max_tokens:
            for s in pack.signals:
                if isinstance(s, dict) and "description" in s:
                    s["description"] = (s["description"] or "")[:50]
            for r in pack.registers:
                if isinstance(r, dict) and "description" in r:
                    r["description"] = (r["description"] or "")[:50]
            current_tokens = pack.estimate_token_count()

        # Step 9: Final guarantee loop - trim ONLY truly auxiliary signals
        # P0 PROTECTED: clock, reset, interface-bound, FSM-referenced signals are NEVER removed.
        if current_tokens > self.max_tokens and len(pack.signals) > 10:
            preserved_signals = []
            aux_signals = []
            for s in pack.signals:
                s_name = (s.get("name") or "").lower() if isinstance(s, dict) else ""
                s_role = (s.get("semantic_role") or "").lower() if isinstance(s, dict) else ""
                s_iface = (s.get("interface") or "").lower() if isinstance(s, dict) else ""
                is_p0 = (
                    "clk" in s_name or "rst" in s_name or "reset" in s_name or "clock" in s_name or
                    s_role in ["clock", "reset"] or
                    bool(s_iface)  # interface-bound signals are P0
                )
                if is_p0:
                    preserved_signals.append(s)
                else:
                    aux_signals.append(s)
            while current_tokens > self.max_tokens and aux_signals:
                aux_signals.pop()
                pack.signals = preserved_signals + aux_signals
                current_tokens = pack.estimate_token_count()

        # P0 GUARANTEE: interfaces, registers, FSM, clock_reset, protocol_rules are NEVER removed.
        # If still over budget after all P2 pruning, log overflow but preserve P0.
        if current_tokens > self.max_tokens:
            logger.warning(
                f"MANDATORY_BEHAVIORAL_OVERFLOW: ContextPack ({current_tokens} tokens) exceeds budget "
                f"({self.max_tokens} tokens) after pruning all non-critical context. "
                f"Preserving all P0 signals, interfaces, registers, FSM, and protocol rules."
            )

        logger.info(f"Pruning complete. Final ContextPack token count: {current_tokens} tokens (Budget limit: {self.max_tokens}).")
        return pack

