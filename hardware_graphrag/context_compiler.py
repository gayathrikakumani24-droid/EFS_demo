"""
Context Compiler top-level entrypoint and compatibility shim.
Re-exports the core context compilation layer.
"""

from core.context_compiler import (
    compile_context,
    BaseContextCompiler,
    DiagramContextCompiler,
    RTLContextCompiler,
    TBContextCompiler,
    UVMContextCompiler,
    SVAContextCompiler,
    ContextResolver,
    GroundingValidator,
    GroundingValidationResult,
    TokenBudgetController,
    CompilationMetrics,
    estimate_tokens,
    get_compiler,
    register_compiler,
    record_generation_telemetry,
)

__all__ = [
    "compile_context",
    "BaseContextCompiler",
    "DiagramContextCompiler",
    "RTLContextCompiler",
    "TBContextCompiler",
    "UVMContextCompiler",
    "SVAContextCompiler",
    "ContextResolver",
    "GroundingValidator",
    "GroundingValidationResult",
    "TokenBudgetController",
    "CompilationMetrics",
    "estimate_tokens",
    "get_compiler",
    "register_compiler",
    "record_generation_telemetry",
]
