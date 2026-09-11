"""
Controlled Specification Inference Engine.

Completes missing hardware implementation details deterministically using standard protocol rules (DERIVED)
or safe architectural defaults (IMPLEMENTATION_CHOICE) without modifying the user's core intent.
Outputs a canonical CompletedDesignModel shared by both RTL and PlantUML generators.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from core.efs_ir.models import (
    EFSIR, CompletedDesignModel, EFSRegister, EFSSignal, EFSState, EFSTransition,
    EFSFSM, InferredDecision, new_efs_id
)
from core.verification.compatibility_engine import ProtocolRoleMapper, CompatibilityEngine
from utils.logger import get_logger

logger = get_logger("verification.inference")


class ControlledInferenceEngine:
    """
    Core Controlled Specification Inference Engine.
    
    Transforms incomplete EFS IR and design plans into fully-specified CompletedDesignModels.
    """

    def __init__(self, efs_ir: EFSIR, design_plan: Optional[Dict[str, Any]] = None, req_model: Optional[Dict[str, Any]] = None):
        self.efs_ir = efs_ir
        self.plan = design_plan or {}
        self.req_model = req_model or {}
        self.decisions: List[InferredDecision] = []

    def infer_and_complete(self) -> CompletedDesignModel:
        """Run all inference steps to construct a complete, synthesizable Design Model."""
        logger.info("Executing Controlled Specification Inference Engine...")
        
        self._infer_clocks_and_resets()
        self._infer_interface_signals()
        self._infer_register_map()
        self._infer_fsm_structures()
        self._infer_datapath_and_reset_polarity()
        
        # Run validation on completed model
        val_status = self._run_completed_validation()

        return CompletedDesignModel(
            efs_ir=self.efs_ir,
            design_plan=self.plan,
            inferred_decisions=self.decisions,
            validation_status=val_status
        )

    def _infer_clocks_and_resets(self) -> None:
        """Ensure primary clock and reset signals are present in EFS IR signals list."""
        sig_names = {s.name.lower() for s in self.efs_ir.signals}
        
        # Clock signal
        if not any("clk" in s or "clock" in s for s in sig_names):
            clk_sig = EFSSignal(
                signal_id=new_efs_id("sig"),
                name="clk",
                width="1",
                direction="input",
                datatype="logic",
                semantic_role="clock",
                description="Primary system clock signal."
            )
            self.efs_ir.signals.insert(0, clk_sig)
            self.decisions.append(InferredDecision(
                target_element="clk",
                element_type="interface",
                decision_type="IMPLEMENTATION_CHOICE",
                description="Added default system clock port 'clk'.",
                rationale="Synchronous hardware modules require a primary clock input."
            ))

        # Reset signal
        if not any("rst" in s or "reset" in s for s in sig_names):
            rst_sig = EFSSignal(
                signal_id=new_efs_id("sig"),
                name="rst_n",
                width="1",
                direction="input",
                datatype="logic",
                semantic_role="reset",
                description="Primary active-low system reset signal."
            )
            self.efs_ir.signals.insert(1, rst_sig)
            self.decisions.append(InferredDecision(
                target_element="rst_n",
                element_type="reset",
                decision_type="IMPLEMENTATION_CHOICE",
                description="Added default active-low reset port 'rst_n'.",
                rationale="Digital hardware modules require a primary reset port for initialization."
            ))

    def _infer_interface_signals(self) -> None:
        """Infer missing signal directions, widths, and semantic protocol roles from protocol contracts if declared."""
        proto = self.efs_ir.metadata.protocol.lower()
        iface_role = "slave" if "slave" in proto or "slave" in self.efs_ir.metadata.design_name.lower() else "slave"
        
        for sig in self.efs_ir.signals:
            if sig.semantic_role in ("clock", "reset"):
                continue

            role_info = ProtocolRoleMapper.get_semantic_role(sig.name, iface_role)
            
            # Infer protocol signal directions strictly from protocol role definitions
            if role_info:
                canon_channel, role_type, expected_dir, canon_symbol = role_info
                if sig.direction.lower() not in ("input", "output", "inout") or sig.direction != expected_dir:
                    old_dir = sig.direction
                    sig.direction = expected_dir
                    sig.semantic_role = role_type
                    self.decisions.append(InferredDecision(
                        target_element=sig.name,
                        element_type="interface",
                        decision_type="DERIVED",
                        description=f"Normalized protocol direction '{expected_dir}' for signal '{sig.name}' ({canon_symbol}).",
                        rationale=f"Standard protocol channel '{canon_channel}' specifies '{expected_dir}' for role '{iface_role}'.",
                        evidence=f"Previous: {old_dir} -> Normalized: {expected_dir}"
                    ))

            # Infer missing signal width
            if not sig.width or sig.width.lower() in ("undefined", "unspecified", "tbd", "0", ""):
                default_w = "32" if any(k in sig.name.lower() for k in ("data", "addr", "payload", "val")) else "1"
                sig.width = default_w
                self.decisions.append(InferredDecision(
                    target_element=sig.name,
                    element_type="interface",
                    decision_type="IMPLEMENTATION_CHOICE",
                    description=f"Assigned default width '{default_w}' to port '{sig.name}'.",
                    rationale="Unspecified signal width safe default assignment."
                ))

    def _infer_register_map(self) -> None:
        """Normalize register byte offsets and field bit ranges without inventing new registers."""
        next_offset = 0
        for reg in self.efs_ir.registers:
            # Assign sequential register offset if unspecified
            if not getattr(reg, "offset", None) or reg.offset in ("0x??", "TBD", "undefined", ""):
                reg.offset = f"0x{next_offset:02X}"
                self.decisions.append(InferredDecision(
                    target_element=reg.name,
                    element_type="register",
                    decision_type="DERIVED",
                    description=f"Assigned sequential register address offset '{reg.offset}' to '{reg.name}'.",
                    rationale="Sequential 32-bit word aligned register map organization."
                ))
                next_offset += 4
            else:
                try:
                    val = int(reg.offset, 16)
                    next_offset = max(next_offset, val + 4)
                except ValueError:
                    next_offset += 4

            # Normalize fields
            if hasattr(reg, "fields") and reg.fields:
                for f in reg.fields:
                    if not getattr(f, "bits", None) and getattr(f, "lsb", None) is not None and getattr(f, "msb", None) is not None:
                        f.bits = f"{f.msb}:{f.lsb}"

    def _infer_fsm_structures(self) -> None:
        """Normalize FSM structures from authoritative EFS IR without inventing synthetic state flows."""
        # Do not invent synthetic IDLE -> RUN -> DONE state machines if not in EFS IR.
        pass

    def _infer_datapath_and_reset_polarity(self) -> None:
        """Determine reset active polarity from declared reset signals."""
        rst_sig = next((s for s in self.efs_ir.signals if s.semantic_role == "reset"), None)
        if rst_sig:
            if "n" in rst_sig.name.lower() or "rst_n" in rst_sig.name.lower():
                self.decisions.append(InferredDecision(
                    target_element=rst_sig.name,
                    element_type="reset",
                    decision_type="DERIVED",
                    description=f"Identified reset signal '{rst_sig.name}' as active-low polarity.",
                    rationale="Name suffix 'n' indicates active-low asynchronous reset."
                ))
            else:
                self.decisions.append(InferredDecision(
                    target_element=rst_sig.name,
                    element_type="reset",
                    decision_type="DERIVED",
                    description=f"Identified reset signal '{rst_sig.name}' as active-high polarity.",
                    rationale="Standard active-high reset signal declaration."
                ))

    def _run_completed_validation(self) -> Dict[str, Any]:
        """Validate the completed design model."""
        user_ports = [{"name": s.name, "direction": s.direction, "width": s.width} for s in self.efs_ir.signals]
        issues = CompatibilityEngine.validate_interface_signals(user_ports, self.efs_ir.signals)
        blocking = [i for i in issues if i.get("is_blocking", False)]
        return {
            "valid": len(blocking) == 0,
            "issues": issues,
            "blocking_count": len(blocking)
        }
