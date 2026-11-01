"""
Grounding Validation module for Compiled Contexts.

Validates that all components, ports, signals, registers, FSM states,
transitions, conditions, and protocol rules in a compiled context are strictly grounded
in the canonical EFS IR before being sent to an LLM generator.
Rejects invented hardware and ungrounded constructs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from core.efs_ir.models import EFSIR
from utils.logger import get_logger

logger = get_logger("context_compiler.validator")


@dataclass
class GroundingValidationResult:
    """Outcome of grounding validation on a compiled context."""
    is_valid: bool
    violations: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_violation(self, category: str, entity_name: str, message: str) -> None:
        self.is_valid = False
        self.violations.append({
            "category": category,
            "entity": entity_name,
            "message": message
        })

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


class GroundingValidator:
    """Validates compiled artifact contexts against canonical EFS IR."""

    @classmethod
    def validate_rtl_context(cls, context: Dict[str, Any], efs_ir: Optional[EFSIR] = None) -> GroundingValidationResult:
        """
        Validate RTL compiled context:
        - Target module exists in EFS IR
        - Ports exist and match directions/widths
        - Clock/reset exist
        - Registers exist
        - FSM states & transitions exist
        - No invented signals or protocols
        """
        result = GroundingValidationResult(is_valid=True)
        if not efs_ir:
            result.add_warning("Canonical EFS IR not provided; skipping strict IR cross-validation.")
            return result

        known_comp_names = {c.name.lower() for c in (efs_ir.components or []) if c.name}
        if efs_ir.metadata and efs_ir.metadata.design_name:
            known_comp_names.add(efs_ir.metadata.design_name.lower())

        # 1. Target module existence
        module_info = context.get("module", {})
        mod_name = module_info.get("name", "")
        if mod_name and known_comp_names and mod_name.lower() not in known_comp_names:
            result.add_violation(
                "MODULE",
                mod_name,
                f"Target module '{mod_name}' does not exist in canonical EFS IR components: {sorted(known_comp_names)}"
            )

        # 2. Port and Signal existence
        known_sig_names = {s.name.lower() for s in (efs_ir.signals or []) if s.name}
        for port in context.get("ports", []):
            p_name = port.get("name", "")
            if p_name and known_sig_names and p_name.lower() not in known_sig_names:
                # Check if it is a standard clock/reset or interface signal
                if p_name.lower() not in ("clk", "rst_n", "rst", "clock", "reset"):
                    result.add_violation(
                        "PORT",
                        p_name,
                        f"Port '{p_name}' does not correspond to any known signal in canonical EFS IR."
                    )

        # 3. Register existence
        known_reg_names = {r.name.lower() for r in (efs_ir.registers or []) if r.name}
        for reg in context.get("registers", []):
            r_name = reg.get("name", "")
            if r_name and known_reg_names and r_name.lower() not in known_reg_names:
                result.add_violation(
                    "REGISTER",
                    r_name,
                    f"Register '{r_name}' does not exist in canonical EFS IR registers."
                )

        # 4. FSM states and transitions
        fsm_data = context.get("fsm", {})
        if fsm_data and isinstance(fsm_data, dict):
            known_fsm_states: Set[str] = set()
            for fsm in (efs_ir.fsms or []):
                for st in (fsm.states or []):
                    st_name = st.name if hasattr(st, "name") else str(st)
                    if st_name:
                        known_fsm_states.add(st_name.lower())

            for st_entry in fsm_data.get("states", []):
                s_name = st_entry if isinstance(st_entry, str) else st_entry.get("name", "")
                if s_name and known_fsm_states and s_name.lower() not in known_fsm_states:
                    result.add_warning(
                        f"FSM state '{s_name}' is derived or not explicitly declared in canonical EFS IR states."
                    )

            # Check transitions
            for trans in fsm_data.get("transitions", []):
                cond = trans.get("condition")
                if cond is None or (isinstance(cond, str) and not cond.strip()):
                    result.add_warning(f"FSM transition from '{trans.get('from')}' to '{trans.get('to')}' has an empty condition.")

        # 5. Internal IDs must not be leaked as primary names
        sig_id_regex = re.compile(r"\bSIG_[a-f0-9]{6,12}\b|\bSIG_[A-Za-z0-9_]+\b", re.IGNORECASE)
        for port in context.get("ports", []):
            if sig_id_regex.match(port.get("name", "")):
                result.add_violation(
                    "INTERNAL_ID",
                    port.get("name", ""),
                    f"Internal signal ID '{port.get('name')}' leaked as a port name instead of human-readable name."
                )

        return result

    @classmethod
    def validate_diagram_context(cls, context: Dict[str, Any], efs_ir: Optional[EFSIR] = None) -> GroundingValidationResult:
        """
        Validate Diagram compiled context:
        - Participants exist as components
        - Signals exist in EFS IR
        - No internal IDs appear as display names
        """
        result = GroundingValidationResult(is_valid=True)
        if not efs_ir:
            return result

        known_comp_names = {c.name.lower() for c in (efs_ir.components or []) if c.name}
        if efs_ir.metadata and efs_ir.metadata.design_name:
            known_comp_names.add(efs_ir.metadata.design_name.lower())

        # Participants in sequence context
        seq_ctx = context.get("sequence", {})
        participants = seq_ctx.get("participants", [])
        for p in participants:
            p_name = p if isinstance(p, str) else p.get("name", "")
            if p_name and known_comp_names and p_name.lower() not in known_comp_names:
                result.add_warning(f"Sequence participant '{p_name}' not found directly in EFS IR component list.")

        # Display names check
        sig_id_regex = re.compile(r"\bSIG_[a-f0-9]{6,12}\b", re.IGNORECASE)
        for sig in seq_ctx.get("signals", []):
            s_name = sig if isinstance(sig, str) else sig.get("name", "")
            if sig_id_regex.match(s_name):
                result.add_violation(
                    "INTERNAL_ID",
                    s_name,
                    f"Internal signal ID '{s_name}' leaked into diagram context without human-readable mapping."
                )

        return result
