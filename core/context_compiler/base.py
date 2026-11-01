"""
Base Context Compiler module.

Defines the abstract base class, compilation metrics, dispatch interface,
and token estimation utilities for the Artifact-Specific Context Compilation Layer.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type
from core.efs_ir.models import EFSIR
from utils.logger import get_logger

logger = get_logger("context_compiler.base")


def estimate_tokens(content: Any) -> int:
    """
    Estimate token count for string, dict, or object representation.
    Uses ~4 characters per token heuristic standard across the project.
    """
    if content is None:
        return 0
    if isinstance(content, str):
        text = content
    elif hasattr(content, "to_dict"):
        text = json.dumps(content.to_dict(), default=str)
    elif isinstance(content, (dict, list)):
        text = json.dumps(content, default=str)
    else:
        text = str(content)
    return max(1, len(text) // 4) if text else 0


@dataclass
class CompilationMetrics:
    """Observability metrics recorded for an artifact-specific context compilation run."""
    artifact_type: str
    compiler_used: str
    original_context_tokens: int = 0
    compiled_context_tokens: int = 0
    system_prompt_tokens: int = 0
    final_input_tokens: int = 0
    output_tokens: int = 0
    context_reduction_ratio: float = 0.0
    budget_exceeded: bool = False
    overflow_warning: Optional[str] = None
    extra_details: Dict[str, Any] = field(default_factory=dict)

    def calculate_reduction(self) -> None:
        if self.original_context_tokens > 0:
            reduction = max(0.0, 1.0 - (self.compiled_context_tokens / self.original_context_tokens))
            self.context_reduction_ratio = round(reduction * 100, 2)
        else:
            self.context_reduction_ratio = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "compiler_used": self.compiler_used,
            "original_context_tokens": self.original_context_tokens,
            "compiled_context_tokens": self.compiled_context_tokens,
            "system_prompt_tokens": self.system_prompt_tokens,
            "final_input_tokens": self.final_input_tokens,
            "output_tokens": self.output_tokens,
            "context_reduction_ratio": f"{self.context_reduction_ratio}%",
            "budget_exceeded": self.budget_exceeded,
            "overflow_warning": self.overflow_warning,
            "extra_details": self.extra_details
        }


class BaseContextCompiler(ABC):
    """Abstract base class for all artifact-specific context compilers."""

    artifact_types: List[str] = []

    @abstractmethod
    def compile(
        self,
        efs_ir: Optional[EFSIR] = None,
        design_plan: Optional[Dict[str, Any]] = None,
        protocol_contract: Optional[Any] = None,
        retrieval_context: Optional[Any] = None,
        token_budget: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compile an artifact-specific compact context dictionary from canonical EFS IR and inputs.
        """
        pass


_COMPILER_REGISTRY: Dict[str, Type[BaseContextCompiler]] = {}


def register_compiler(compiler_cls: Type[BaseContextCompiler]) -> Type[BaseContextCompiler]:
    """Decorator to register an artifact-specific context compiler."""
    for art in compiler_cls.artifact_types:
        _COMPILER_REGISTRY[art.lower()] = compiler_cls
    return compiler_cls


def get_compiler(artifact_type: str) -> Optional[BaseContextCompiler]:
    """Retrieve an instantiated compiler for the given artifact type."""
    art = artifact_type.lower().strip()
    compiler_cls = _COMPILER_REGISTRY.get(art)
    if compiler_cls:
        return compiler_cls()
    return None


def compile_context(
    artifact_type: str,
    efs_ir: Optional[EFSIR] = None,
    design_plan: Optional[Dict[str, Any]] = None,
    protocol_contract: Optional[Any] = None,
    retrieval_context: Optional[Any] = None,
    token_budget: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main dispatch interface for the Artifact-Specific Context Compilation Layer.

    Args:
        artifact_type: Target artifact type ('diagram', 'plantuml', 'rtl', 'verilog',
                       'systemverilog', 'vhdl', 'tb', 'testbench', 'uvm', 'sva')
        efs_ir: Canonical authoritative EFS IR
        design_plan: Optional full design plan dictionary
        protocol_contract: Optional protocol contract or rules list
        retrieval_context: Optional vector/graph retrieval context
        token_budget: Optional custom token budget override

    Returns:
        Compact, grounded, artifact-specific context dictionary with '_metrics' attached.
    """
    compiler = get_compiler(artifact_type)
    if not compiler:
        logger.warning(f"No specific compiler registered for artifact '{artifact_type}'. Using fallback RTL compiler.")
        compiler = get_compiler("rtl")
        if not compiler:
            raise ValueError(f"Unknown artifact type '{artifact_type}' and default compiler unavailable.")

    return compiler.compile(
        efs_ir=efs_ir,
        design_plan=design_plan,
        protocol_contract=protocol_contract,
        retrieval_context=retrieval_context,
        token_budget=token_budget,
        **kwargs
    )
