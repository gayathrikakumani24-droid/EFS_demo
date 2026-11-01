"""
Telemetry and Langfuse Instrumentation module for Compiled Contexts.

Records artifact generation observability metrics:
- artifact_type
- compiler_used
- original_context_tokens
- compiled_context_tokens
- final_input_tokens
- output_tokens
- context_reduction_ratio

Gracefully integrates with Langfuse when configured, and always emits structured logs.
Does NOT log proprietary full specifications unnecessarily.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional
from core.context_compiler.base import CompilationMetrics
from utils.logger import get_logger
import langfuse
logger = get_logger("context_compiler.telemetry")

_LANGFUSE_CLIENT = None
_LANGFUSE_CHECKED = False

def _get_langfuse_client():
    global _LANGFUSE_CLIENT, _LANGFUSE_CHECKED
    if _LANGFUSE_CHECKED:
        return _LANGFUSE_CLIENT
    _LANGFUSE_CHECKED = True
    try:
        # Check if API keys are set
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            _LANGFUSE_CLIENT = langfuse.Langfuse()
            logger.info("Langfuse tracing client initialized.")
    except Exception as e:
        logger.debug(f"Langfuse not initialized: {e}")
    return _LANGFUSE_CLIENT


def record_generation_telemetry(
    metrics: CompilationMetrics,
    generation_name: Optional[str] = None,
    model: Optional[str] = None
) -> None:
    """
    Log and trace context compilation and generation metrics.
    """
    metrics.calculate_reduction()
    name = generation_name or f"efs_generate_{metrics.artifact_type}"

    # 1. Structured logging
    logger.info(
        f"[ContextCompiler Telemetry] {name} | Compiler: {metrics.compiler_used} | "
        f"Original: {metrics.original_context_tokens} tokens -> Compiled: {metrics.compiled_context_tokens} tokens | "
        f"Reduction: {metrics.context_reduction_ratio}% | Final Input: {metrics.final_input_tokens} | "
        f"Output: {metrics.output_tokens} | Budget Exceeded: {metrics.budget_exceeded}"
    )

    if metrics.overflow_warning:
        logger.warning(f"[ContextCompiler Telemetry Warning] {metrics.overflow_warning}")

    # 2. Langfuse trace if active
    lf = _get_langfuse_client()
    if lf:
        try:
            lf.generation(
                name=name,
                model=model or "unknown",
                metadata={
                    "artifact_type": metrics.artifact_type,
                    "compiler_used": metrics.compiler_used,
                    "original_context_tokens": metrics.original_context_tokens,
                    "compiled_context_tokens": metrics.compiled_context_tokens,
                    "reduction_ratio_pct": metrics.context_reduction_ratio,
                    "budget_exceeded": metrics.budget_exceeded,
                    **metrics.extra_details
                },
                usage={
                    "input": metrics.final_input_tokens or metrics.compiled_context_tokens,
                    "output": metrics.output_tokens,
                    "total": (metrics.final_input_tokens or metrics.compiled_context_tokens) + metrics.output_tokens
                }
            )
        except Exception as e:
            logger.debug(f"Failed to record Langfuse telemetry: {e}")
