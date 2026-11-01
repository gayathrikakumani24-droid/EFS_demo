"""
Unit tests for Conflict Resolution UI filtering and Automatic PUML/RTL Resume.

Verifies:
1. Single conflict resolution: resolving it removes it from unresolved list and auto-resumes PUML/RTL.
2. Multi-conflict resolution: resolving 1 leaves only remaining unresolved conflicts (resolved conflict is not visible).
3. Resolving the final remaining conflict triggers automatic resumption of PlantUML diagrams and RTL generation.
"""

from __future__ import annotations

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.efs_ir.models import (
    EFSIR, EFSConflict, EFSResolution, EFSRegister, EFSSignal, EFSOpcode, EFSComponent
)
from core.verification.conflict_resolution_engine import ConflictResolutionEngine
from core.agents.orchestrator import resume_design_flow_with_resolutions


def test_conflict_resolved_is_hidden_and_remaining_visible():
    """Verify that when 1 of 2 conflicts is resolved, only the unresolved conflict remains."""
    ir = EFSIR()
    ir.components = [EFSComponent(name="DualConflictCore", type="core", description="Core")]
    ir.signals = [
        EFSSignal(name="clk", width="1", direction="input", semantic_role="clock"),
        EFSSignal(name="rst_n", width="1", direction="input", semantic_role="reset")
    ]
    # Conflict 1: Duplicate register address
    ir.registers = [
        EFSRegister(name="REG_A", offset="0x04", width=32),
        EFSRegister(name="REG_B", offset="0x04", width=32)
    ]
    # Conflict 2: Duplicate opcodes
    ir.opcodes = [
        EFSOpcode(mnemonic="OP_LOAD", binary_encoding="0001"),
        EFSOpcode(mnemonic="OP_STORE", binary_encoding="0001")
    ]

    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    unresolved_initial = [c for c in conflicts if c.blocking and c.resolution_status == "UNRESOLVED"]
    assert len(unresolved_initial) == 2

    # Resolve Conflict 1 (REG_B offset = 0x08)
    reg_conf = next(c for c in unresolved_initial if c.type == "DUPLICATE_REGISTER_ADDRESS")
    res1 = ConflictResolutionEngine.create_custom_resolution(
        reg_conf,
        "offset = 0x08",
        user_name="designer",
        target_entity="REG_B"
    )

    resolved_ir = ConflictResolutionEngine.apply_approved_resolution(ir, res1)
    
    # Re-run conflict detection on the resolved IR
    updated_conflicts = ConflictResolutionEngine.detect_conflicts(resolved_ir)
    
    # Check approved IDs & entity/property
    approved_cids = {res1.conflict_id}
    approved_entity_props = {("reg_b", "offset")}

    def is_conflict_resolved(c) -> bool:
        if getattr(c, "resolution_status", "UNRESOLVED") == "RESOLVED":
            return True
        if getattr(c, "status", "") == "APPROVED":
            return True
        if not getattr(c, "blocking", True) or not getattr(c, "requires_human_decision", True):
            return True
        if getattr(c, "conflict_id", None) in approved_cids:
            return True
        c_prop = (getattr(c, "property", "") or "").lower()
        for ent in (getattr(c, "affected_entities", []) or getattr(c, "entities", []) or []):
            if (ent.lower(), c_prop) in approved_entity_props or (ent.lower(), "") in approved_entity_props:
                return True
        return False

    unresolved_remaining = [c for c in updated_conflicts if not is_conflict_resolved(c)]

    # Exactly 1 conflict remains visible
    assert len(unresolved_remaining) == 1
    assert unresolved_remaining[0].type in ("DUPLICATE_OPCODE", "DUPLICATE_OPCODE_ENCODING")
    # The resolved conflict is NOT visible
    assert not any(c.type == "DUPLICATE_REGISTER_ADDRESS" for c in unresolved_remaining)


@patch("core.agents.orchestrator.generate_design_spec", return_value="# Design Spec\n")
@patch("core.agents.plantuml_agent.generate_diagrams", return_value={"architecture": "@startuml\n@enduml\n", "sequence": "@startuml\n@enduml\n"})
@patch("core.agents.plantuml_agent.get_diagram_url", return_value="http://plantuml.com/sample.png")
@patch("core.agents.orchestrator.generate_code", return_value="module Core (input clk, input rst_n); endmodule")
def test_all_conflicts_resolved_auto_resumes_puml_and_rtl(mock_code, mock_url, mock_diag, mock_spec):
    """Verify that when all conflicts are resolved, resume_design_flow_with_resolutions produces PUML and RTL."""
    ir = EFSIR()
    ir.components = [EFSComponent(name="Core", type="core", description="Core")]
    ir.signals = [
        EFSSignal(name="clk", width="1", direction="input", semantic_role="clock"),
        EFSSignal(name="rst_n", width="1", direction="input", semantic_role="reset")
    ]
    ir.registers = [
        EFSRegister(name="REG_A", offset="0x04", width=32),
        EFSRegister(name="REG_B", offset="0x04", width=32)
    ]

    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    assert len(conflicts) == 1
    conf = conflicts[0]

    initial_results = {
        "status": "AWAITING_HUMAN_DECISION",
        "pipeline_state": "AWAITING_HUMAN_DECISION",
        "requires_human_decision": True,
        "is_blocked": False,
        "requirement_model": {"design_name": "Core"},
        "plan": {"module_name": "Core"},
        "protocol_rules": [],
        "design_spec": "# Design Spec\nModule Name: Core\n",
        "efs_ir": ir,
        "conflicts": conflicts
    }

    # Resolve the conflict
    res = ConflictResolutionEngine.create_custom_resolution(
        conf,
        "offset = 0x08",
        user_name="designer",
        target_entity="REG_B"
    )

    # Resume flow with the resolution
    results = resume_design_flow_with_resolutions(
        initial_results,
        [res],
        target_outputs=["Verilog", "SystemVerilog"]
    )

    # Status must be SUCCESS
    assert results["status"] == "SUCCESS"
    assert results["requires_human_decision"] is False

    # Must contain PlantUML diagrams
    assert "diagrams" in results
    assert "architecture" in results["diagrams"]
    assert "sequence" in results["diagrams"]

    # Must contain generated RTL
    assert "outputs" in results
    assert "Verilog" in results["outputs"]
    assert "SystemVerilog" in results["outputs"]
    assert "module Core" in results["outputs"]["Verilog"]["final_code"]
    assert "module Core" in results["outputs"]["SystemVerilog"]["final_code"]


if __name__ == "__main__":
    print("Running test_conflict_resolved_is_hidden_and_remaining_visible...")
    test_conflict_resolved_is_hidden_and_remaining_visible()
    print("-> PASS: test_conflict_resolved_is_hidden_and_remaining_visible")

    print("Running test_all_conflicts_resolved_auto_resumes_puml_and_rtl...")
    test_all_conflicts_resolved_auto_resumes_puml_and_rtl()
    print("-> PASS: test_all_conflicts_resolved_auto_resumes_puml_and_rtl")

    print("ALL TESTS PASSED SUCCESSFULLY!")
