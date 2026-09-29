"""
Artifact-Specific Context Compiler Package.

Compiles canonical EFS IR and design specifications into compact, rich,
grounded contexts specifically tailored for:
- PlantUML Diagrams (Architecture, FSM, Sequence)
- Synthesizable RTL (SystemVerilog, Verilog, VHDL)
- Verification Collateral (Testbench, UVM, SVA assertions)
"""

from __future__ import annotations

from core.context_compiler.base import (
    BaseContextCompiler,
    CompilationMetrics,
    compile_context,
    estimate_tokens,
    get_compiler,
    register_compiler,
)
from core.context_compiler.resolver import ContextResolver
from core.context_compiler.validator import GroundingValidationResult, GroundingValidator
from core.context_compiler.budget import TokenBudgetController
from core.context_compiler.telemetry import record_generation_telemetry
from core.context_compiler.diagram_context import DiagramContextCompiler
from core.context_compiler.rtl_context import RTLContextCompiler
from core.context_compiler.verification_context import (
    TBContextCompiler,
    UVMContextCompiler,
    SVAContextCompiler,
)
from core.context_compiler.semantic_selector import (
    SemanticDependencies,
    SemanticDependencySelector,
)
from core.context_compiler.generation_context import GenerationContext

__all__ = [
    "compile_context",
    "BaseContextCompiler",
    "GenerationContext",
    "DiagramContextCompiler",
    "RTLContextCompiler",
    "TBContextCompiler",
    "UVMContextCompiler",
    "SVAContextCompiler",
    "ContextResolver",
    "SemanticDependencySelector",
    "SemanticDependencies",
    "GroundingValidator",
    "GroundingValidationResult",
    "TokenBudgetController",
    "CompilationMetrics",
    "estimate_tokens",
    "get_compiler",
    "register_compiler",
    "record_generation_telemetry",
]
