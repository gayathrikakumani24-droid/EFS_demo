"""
Verification Collateral Context Compilers module.

Provides dedicated compilers for:
- TBContextCompiler (Testbench stimulus and driver/monitor interface)
- UVMContextCompiler (UVM agent, sequence item, driver, monitor, scoreboard)
- SVAContextCompiler (SystemVerilog Assertions, protocol invariants, timing rules)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from config import CONFIG
from core.efs_ir.models import EFSIR
from core.context_compiler.base import (
    BaseContextCompiler, CompilationMetrics, estimate_tokens, register_compiler
)
from core.context_compiler.resolver import ContextResolver
from core.context_compiler.rtl_context import RTLContextCompiler
from utils.logger import get_logger

logger = get_logger("context_compiler.verification")


@register_compiler
class TBContextCompiler(BaseContextCompiler):
    """Compiles testbench-specific context (ports, clock/reset stimulus, transaction sequences)."""

    artifact_types = ["tb", "testbench"]

    def compile(
        self,
        efs_ir: Optional[EFSIR] = None,
        design_plan: Optional[Dict[str, Any]] = None,
        protocol_contract: Optional[Any] = None,
        retrieval_context: Optional[Any] = None,
        token_budget: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        max_budget = token_budget or getattr(getattr(CONFIG, "context_compiler", None), "tb_token_budget", 8000)
        metrics = CompilationMetrics(artifact_type="testbench", compiler_used="TBContextCompiler")

        rtl_comp = RTLContextCompiler()
        rtl_ctx = rtl_comp.compile(
            efs_ir=efs_ir,
            design_plan=design_plan,
            protocol_contract=protocol_contract,
            token_budget=max_budget,
            **kwargs
        )

        mod = rtl_ctx.get("module", {})
        ports = rtl_ctx.get("ports", [])
        clk_rst = rtl_ctx.get("clock_reset", [])

        tb_context = {
            "dut_module": mod.get("name", "dut"),
            "ports": ports,
            "clock_reset": clk_rst,
            "registers": rtl_ctx.get("registers", []),
            "fsm_states": rtl_ctx.get("fsm", {}).get("states", []),
            "parameters": rtl_ctx.get("parameters", []),
            "stimulus_transactions": rtl_ctx.get("relevant_transactions", []) or [
                {
                    "operation": "reset_sequence",
                    "description": "Assert reset, wait 5 cycles, deassert reset"
                },
                {
                    "operation": "basic_transaction",
                    "description": "Drive interface inputs and verify responses"
                }
            ],
            "protocol_rules": rtl_ctx.get("protocol_rules", [])[:5]
        }
        if rtl_ctx.get("opcodes"):
            tb_context["opcodes"] = rtl_ctx["opcodes"]
        if rtl_ctx.get("instructions"):
            tb_context["instructions"] = rtl_ctx["instructions"]
        if rtl_ctx.get("commands"):
            tb_context["commands"] = rtl_ctx["commands"]
        if rtl_ctx.get("ordering_and_timeouts"):
            tb_context["ordering_and_timeouts"] = rtl_ctx["ordering_and_timeouts"]

        metrics.original_context_tokens = rtl_ctx.get("_metrics", {}).get("original_context_tokens", 1)
        metrics.compiled_context_tokens = estimate_tokens(tb_context)
        metrics.calculate_reduction()
        tb_context["_metrics"] = metrics.to_dict()
        return tb_context


@register_compiler
class UVMContextCompiler(BaseContextCompiler):
    """Compiles UVM-specific verification context."""

    artifact_types = ["uvm", "uvm_testbench"]

    def compile(
        self,
        efs_ir: Optional[EFSIR] = None,
        design_plan: Optional[Dict[str, Any]] = None,
        protocol_contract: Optional[Any] = None,
        retrieval_context: Optional[Any] = None,
        token_budget: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        max_budget = token_budget or getattr(getattr(CONFIG, "context_compiler", None), "uvm_token_budget", 8000)
        metrics = CompilationMetrics(artifact_type="uvm", compiler_used="UVMContextCompiler")

        rtl_comp = RTLContextCompiler()
        rtl_ctx = rtl_comp.compile(
            efs_ir=efs_ir,
            design_plan=design_plan,
            protocol_contract=protocol_contract,
            token_budget=max_budget,
            **kwargs
        )

        uvm_context = {
            "dut_module": rtl_ctx.get("module", {}).get("name", "dut"),
            "interfaces": [
                {
                    "name": p.get("interface") or "native_if",
                    "signals": [p.get("name")]
                }
                for p in rtl_ctx.get("ports", []) if p.get("interface")
            ],
            "signals": rtl_ctx.get("ports", []),
            "registers": rtl_ctx.get("registers", []),
            "parameters": rtl_ctx.get("parameters", []),
            "protocol_rules": rtl_ctx.get("protocol_rules", [])[:8],
            "uvm_components_required": ["sequence_item", "driver", "monitor", "agent", "env", "test"]
        }
        if rtl_ctx.get("opcodes"):
            uvm_context["opcodes"] = rtl_ctx["opcodes"]
        if rtl_ctx.get("instructions"):
            uvm_context["instructions"] = rtl_ctx["instructions"]
        if rtl_ctx.get("instruction_formats"):
            uvm_context["instruction_formats"] = rtl_ctx["instruction_formats"]
        if rtl_ctx.get("relevant_transactions"):
            uvm_context["transactions"] = rtl_ctx["relevant_transactions"]
        if rtl_ctx.get("handshakes_and_constraints"):
            uvm_context["handshakes_and_constraints"] = rtl_ctx["handshakes_and_constraints"]
        if rtl_ctx.get("ordering_and_timeouts"):
            uvm_context["ordering_and_timeouts"] = rtl_ctx["ordering_and_timeouts"]

        metrics.original_context_tokens = rtl_ctx.get("_metrics", {}).get("original_context_tokens", 1)
        metrics.compiled_context_tokens = estimate_tokens(uvm_context)
        metrics.calculate_reduction()
        uvm_context["_metrics"] = metrics.to_dict()
        return uvm_context


@register_compiler
class SVAContextCompiler(BaseContextCompiler):
    """Compiles SVA (SystemVerilog Assertions) specific context."""

    artifact_types = ["sva", "assertion", "assertions", "checker"]

    def compile(
        self,
        efs_ir: Optional[EFSIR] = None,
        design_plan: Optional[Dict[str, Any]] = None,
        protocol_contract: Optional[Any] = None,
        retrieval_context: Optional[Any] = None,
        token_budget: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        max_budget = token_budget or getattr(getattr(CONFIG, "context_compiler", None), "sva_token_budget", 6000)
        metrics = CompilationMetrics(artifact_type="sva", compiler_used="SVAContextCompiler")

        rtl_comp = RTLContextCompiler()
        rtl_ctx = rtl_comp.compile(
            efs_ir=efs_ir,
            design_plan=design_plan,
            protocol_contract=protocol_contract,
            token_budget=max_budget,
            **kwargs
        )

        resolver = ContextResolver(efs_ir)
        timing_rules = []
        if efs_ir and efs_ir.timing_rules:
            for tr in efs_ir.timing_rules:
                timing_rules.append({
                    "rule": resolver.scrub_internal_ids(getattr(tr, "name", "") or getattr(tr, "description", "")),
                    "cycles": getattr(tr, "max_cycles", None) or getattr(tr, "cycles", None)
                })

        sva_context = {
            "target_module": rtl_ctx.get("module", {}).get("name", "dut"),
            "clock_reset": rtl_ctx.get("clock_reset", []),
            "signals": rtl_ctx.get("ports", []),
            "fsm_transitions": rtl_ctx.get("fsm", {}).get("transitions", []),
            "timing_rules": timing_rules,
            "protocol_rules": rtl_ctx.get("protocol_rules", []),
            "properties_to_assert": [
                "Reset deassertion state check",
                "Handshake stability (valid without ready must hold data)",
                "FSM valid state transition sequence",
                "No unexpected deadlock in RUN state"
            ]
        }
        if rtl_ctx.get("handshakes_and_constraints"):
            sva_context["handshakes_and_constraints"] = rtl_ctx["handshakes_and_constraints"]
        if rtl_ctx.get("ordering_and_timeouts"):
            sva_context["ordering_and_timeouts"] = rtl_ctx["ordering_and_timeouts"]
        if rtl_ctx.get("opcodes"):
            sva_context["opcodes"] = rtl_ctx["opcodes"]
            sva_context["properties_to_assert"].append("Instruction opcode valid decoding check")

        metrics.original_context_tokens = rtl_ctx.get("_metrics", {}).get("original_context_tokens", 1)
        metrics.compiled_context_tokens = estimate_tokens(sva_context)
        metrics.calculate_reduction()
        sva_context["_metrics"] = metrics.to_dict()
        return sva_context
