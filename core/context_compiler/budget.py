"""
Token Budget Controller module for Compiled Contexts.

Enforces configurable token budgets using semantic, priority-based element pruning.
Crucially: NEVER arbitrarily slices JSON text mid-stream.
Preserves required behavioral semantics (ports, registers, FSM transitions, handshake signals).
Prunes lower-priority narrative, verbose comments, and distant source evidence first.
Logs an explicit overflow warning if required behavioral context exceeds budget.
Records all reduction decisions in an explicit context_reduction_manifest.
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
        context.setdefault("context_reduction_manifest", [])
        manifest = context["context_reduction_manifest"]

        current_tokens = estimate_tokens(context)
        if current_tokens <= max_tokens:
            return context, False, None

        logger.info(f"RTL Context ({current_tokens} tokens) exceeds budget ({max_tokens} tokens). Starting priority pruning...")

        # Step 1: Trim verbose descriptions inside ports and registers
        if "ports" in context:
            for p in context["ports"]:
                if "description" in p and len(str(p["description"])) > 60:
                    orig_desc = str(p["description"])
                    p["description"] = orig_desc[:57] + "..."
                    manifest.append({
                        "category": "port_description_trimmed",
                        "target": p.get("name", "unknown_port"),
                        "reason": "Token budget reduction",
                        "original_length": len(orig_desc),
                        "preserved": p["description"]
                    })

        if "registers" in context:
            for r in context["registers"]:
                if "description" in r and len(str(r["description"])) > 60:
                    orig_desc = str(r["description"])
                    r["description"] = orig_desc[:57] + "..."
                    manifest.append({
                        "category": "register_description_trimmed",
                        "target": r.get("name", "unknown_reg"),
                        "reason": "Token budget reduction",
                        "original_length": len(orig_desc),
                        "preserved": r["description"]
                    })

        current_tokens = estimate_tokens(context)
        if current_tokens <= max_tokens:
            logger.info(f"Pruned verbose port/register descriptions. Tokens now: {current_tokens}")
            return context, False, None

        # Step 2: Prune P2 source_evidence & raw source text
        if "source_refs" in context and context["source_refs"]:
            if len(context["source_refs"]) > 10:
                dropped = context["source_refs"][10:]
                context["source_refs"] = context["source_refs"][:10]
                manifest.append({
                    "category": "source_refs_pruned",
                    "dropped_count": len(dropped),
                    "dropped_items": dropped,
                    "reason": "Lower priority P2 source references trimmed to fit budget"
                })

        if "source_evidence" in context and context["source_evidence"]:
            ev_count = len(context["source_evidence"])
            context["source_evidence"] = []
            manifest.append({
                "category": "source_evidence_pruned",
                "dropped_count": ev_count,
                "reason": "Raw source evidence text pruned; behavioral models preserved"
            })

        current_tokens = estimate_tokens(context)
        if current_tokens <= max_tokens:
            logger.info(f"Pruned source evidence. Tokens now: {current_tokens}")
            return context, False, None

        # Step 3: Prune lower priority design plan / notes
        if "implementation_notes" in context:
            notes = context.pop("implementation_notes", None)
            manifest.append({
                "category": "implementation_notes_removed",
                "reason": "Informational design notes removed"
            })

        if "design_plan" in context and isinstance(context["design_plan"], dict):
            dp = context["design_plan"]
            dropped_keys = [k for k in dp.keys() if k not in ("target", "architecture_overview", "interface_strategy")]
            context["design_plan"] = {
                k: v for k, v in dp.items()
                if k in ("target", "architecture_overview", "interface_strategy")
            }
            if dropped_keys:
                manifest.append({
                    "category": "design_plan_keys_pruned",
                    "dropped_keys": dropped_keys,
                    "reason": "Non-structural design plan sections trimmed"
                })

        current_tokens = estimate_tokens(context)
        if current_tokens <= max_tokens:
            logger.info(f"Pruned design plan notes. Tokens now: {current_tokens}")
            return context, False, None

        # Step 4: Prune secondary submodules (keep only direct target)
        if "connected_submodules" in context and len(context["connected_submodules"]) > 3:
            dropped_subs = context["connected_submodules"][3:]
            context["connected_submodules"] = context["connected_submodules"][:3]
            manifest.append({
                "category": "submodules_pruned",
                "dropped_count": len(dropped_subs),
                "reason": "Secondary submodules trimmed; direct target preserved"
            })

        current_tokens = estimate_tokens(context)
        if current_tokens <= max_tokens:
            logger.info(f"Pruned submodules. Tokens now: {current_tokens}")
            return context, False, None

        # Critical Grounding Safeguard: DO NOT remove P0 behavioral constructs.
        # NEVER REMOVED: interface, ports, signals, registers, FSM transitions,
        #                 clock_reset, and protocol rules.
        # These fields are required for RTL generation and must not be removed during token pruning.
        overflow_msg = (
            f"MANDATORY_BEHAVIORAL_OVERFLOW: Scoped RTL context ({current_tokens} tokens) "
            f"exceeds budget limit ({max_tokens} tokens) after pruning non-critical narrative context. "
            f"Preserving all interface, signal, FSM, register, clock/reset, and protocol semantics to prevent hallucination."
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
