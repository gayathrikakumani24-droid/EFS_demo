"""
Token Budget Controller module for Compiled Contexts.

Enforces configurable token budgets using semantic, priority-based element pruning.
Crucially: NEVER arbitrarily slices JSON text mid-stream.
Preserves required behavioral semantics (ports, registers, FSM transitions, handshake signals).
Prunes lower-priority narrative, verbose comments, and distant source evidence first.
Logs an explicit overflow warning if required behavioral context exceeds budget.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from core.context_compiler.base import estimate_tokens
from utils.logger import get_logger

logger = get_logger("context_compiler.budget")


class TokenBudgetController:
    """Manages token budgeting and priority-based element pruning for compiled contexts."""

    @staticmethod
    def enforce_rtl_budget(
        context: Dict[str, Any],
        max_tokens: int = 8000
    ) -> Tuple[Dict[str, Any], bool, Optional[str]]:
        """
        Enforce token budget on an RTL context using strict priority:
          P0 (Non-negotiable behavioral & semantic core):
             - module identity & parameters
             - ports (names, directions, widths)
             - clock / reset (polarity, behavior)
             - directly connected signals
             - registers (widths, reset values, fields)
             - FSM states, transitions, conditions, actions, alternate branches
             - opcodes, instruction fields & formats (when relevant by dependency)
             - relevant transactions, ordered steps, preconditions, handshakes
             - commands, datapath operations, memory operations
             - ordering constraints, timeouts, out-of-order behavior
          P1 (Protocol & Requirements):
             - directly relevant protocol rules
             - functional requirements, flows and branches
          P2 (Prunable source context):
             - raw source text excerpts
             - verbose descriptions / commentary
             - source citations / reference lists
             - secondary dependencies / submodules

        Returns:
            (pruned_context, is_exceeded, overflow_warning)
        """
        current_tokens = estimate_tokens(context)
        if current_tokens <= max_tokens:
            return context, False, None

        logger.info(f"RTL Context ({current_tokens} tokens) exceeds budget ({max_tokens} tokens). Starting priority pruning...")

        # Step 1: Trim verbose descriptions inside ports and registers
        if "ports" in context:
            for p in context["ports"]:
                if "description" in p and len(str(p["description"])) > 60:
                    p["description"] = str(p["description"])[:57] + "..."

        if "registers" in context:
            for r in context["registers"]:
                if "description" in r and len(str(r["description"])) > 60:
                    r["description"] = str(r["description"])[:57] + "..."

        current_tokens = estimate_tokens(context)
        if current_tokens <= max_tokens:
            logger.info(f"Pruned verbose port/register descriptions. Tokens now: {current_tokens}")
            return context, False, None

        # Step 2: Prune P2 source_evidence & raw source text
        if "source_refs" in context and context["source_refs"]:
            # Keep only the ref IDs, drop raw text snippets
            context["source_refs"] = context["source_refs"][:10]

        if "source_evidence" in context and context["source_evidence"]:
            context["source_evidence"] = []

        current_tokens = estimate_tokens(context)
        if current_tokens <= max_tokens:
            logger.info(f"Pruned source evidence. Tokens now: {current_tokens}")
            return context, False, None

        # Step 3: Prune lower priority design plan / notes
        if "implementation_notes" in context:
            context.pop("implementation_notes", None)

        if "design_plan" in context and isinstance(context["design_plan"], dict):
            # Keep only architecture overview and interface guidelines
            dp = context["design_plan"]
            context["design_plan"] = {
                k: v for k, v in dp.items()
                if k in ("target", "architecture_overview", "interface_strategy")
            }

        current_tokens = estimate_tokens(context)
        if current_tokens <= max_tokens:
            logger.info(f"Pruned design plan notes. Tokens now: {current_tokens}")
            return context, False, None

        # Step 4: Prune secondary submodules (keep only direct target)
        if "connected_submodules" in context and len(context["connected_submodules"]) > 3:
            context["connected_submodules"] = context["connected_submodules"][:3]

        current_tokens = estimate_tokens(context)
        if current_tokens <= max_tokens:
            logger.info(f"Pruned submodules. Tokens now: {current_tokens}")
            return context, False, None

        # Step 5: Trim protocol rules to top directly relevant rules
        if "protocol_rules" in context and len(context["protocol_rules"]) > 10:
            context["protocol_rules"] = context["protocol_rules"][:10]

        current_tokens = estimate_tokens(context)
        if current_tokens <= max_tokens:
            logger.info(f"Pruned protocol rules. Tokens now: {current_tokens}")
            return context, False, None

        # Critical Grounding Safeguard: DO NOT remove P0 behavioral constructs.
        # NEVER REMOVED: ports, signals, registers, FSM transitions, datapath,
        #                 clock_reset, opcodes, handshakes_and_constraints.
        # Instead, record an explicit overflow warning.
        overflow_msg = (
            f"MANDATORY_BEHAVIORAL_OVERFLOW: Scoped RTL context ({current_tokens} tokens) "
            f"exceeds budget limit ({max_tokens} tokens) after pruning all non-critical context. "
            f"Preserving all P0 ports, signals, registers, FSM, clock/reset, and protocol semantics to prevent hallucination."
        )
        logger.warning(overflow_msg)
        return context, True, overflow_msg

    @staticmethod
    def enforce_diagram_budget(
        context: Dict[str, Any],
        max_tokens: int = 4000
    ) -> Tuple[Dict[str, Any], bool, Optional[str]]:
        """
        Enforce token budget on Diagram context:
        - Keeps essential architecture components, interfaces, ports, registers
        - Keeps FSM states & transitions
        - Caps sequence transactions to representative flow (max 10-15 interactions)
        - Drops verbose textual descriptions
        """
        current_tokens = estimate_tokens(context)
        if current_tokens <= max_tokens:
            return context, False, None

        logger.info(f"Diagram Context ({current_tokens} tokens) exceeds budget ({max_tokens} tokens). Starting pruning...")

        # Step 1: Cap sequence diagram interactions if oversized
        seq_ctx = context.get("sequence", {})
        if isinstance(seq_ctx, dict):
            interactions = seq_ctx.get("transactions", seq_ctx.get("interactions", []))
            if len(interactions) > 12:
                seq_ctx["transactions"] = interactions[:12]

        current_tokens = estimate_tokens(context)
        if current_tokens <= max_tokens:
            return context, False, None

        # Step 2: Prune long component descriptions
        arch_ctx = context.get("architecture", {})
        if isinstance(arch_ctx, dict):
            for c in arch_ctx.get("components", []):
                if isinstance(c, dict) and "description" in c and len(str(c["description"])) > 40:
                    c["description"] = str(c["description"])[:37] + "..."

        current_tokens = estimate_tokens(context)
        if current_tokens <= max_tokens:
            return context, False, None

        # Step 3: Limit architecture signals to top interface & clock/reset signals
        if isinstance(arch_ctx, dict):
            sigs = arch_ctx.get("signals", [])
            if len(sigs) > 25:
                # Prioritize clock, reset, and interface signals
                p0_sigs = [s for s in sigs if s.get("semantic_role") in ("clock", "reset") or s.get("interface")]
                other_sigs = [s for s in sigs if s not in p0_sigs]
                arch_ctx["signals"] = p0_sigs + other_sigs[:max(0, 25 - len(p0_sigs))]

        current_tokens = estimate_tokens(context)
        is_exceeded = current_tokens > max_tokens
        warning = f"Diagram context ({current_tokens} tokens) exceeds budget ({max_tokens} tokens)" if is_exceeded else None
        return context, is_exceeded, warning
