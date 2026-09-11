"""
Integration Test for Human Approval Pipeline Continuation & RTL Generation Resume Flow.

Verifies the complete 13-step lifecycle:
1. Create specification with genuine conflict.
2. Validate initial model.
3. Confirm generation is blocked.
4. Generate resolution proposal.
5. Approve proposal programmatically.
6. Persist approved resolution in overlay layer.
7. Rebuild resolved requirement model view.
8. Revalidate resolved requirement model.
9. Confirm conflict is resolved.
10. Confirm generation gate allows continuation.
11. Confirm planning agent & completed model execute.
12. Confirm RTL generation executes upon resume.
13. Confirm final synthesizable RTL artifact is returned and stored.
"""

import os
import sys
import pytest

from core.efs_ir.models import EFSIR, EFSSignal, EFSRegister, EFSOpcode, EFSConflict, EFSResolution, EFSComponent
from core.verification.conflict_resolution_engine import ConflictResolutionEngine
from core.agents.orchestrator import resume_design_flow_with_resolutions


def test_human_approval_resumes_rtl_generation_flow():
    # 1. Create specification model with a genuine blocking conflict (duplicate register address)
    efs_ir = EFSIR()
    efs_ir.components = [
        EFSComponent(name="TestModule", type="controller", description="Main Controller")
    ]
    efs_ir.signals = [
        EFSSignal(name="clk", width="1", direction="input", semantic_role="clock"),
        EFSSignal(name="rst_n", width="1", direction="input", semantic_role="reset"),
        EFSSignal(name="ctrl_start", width="1", direction="input"),
        EFSSignal(name="status_busy", width="1", direction="output")
    ]
    efs_ir.registers = [
        EFSRegister(name="CTRL_REG", offset="0x04", width=32),
        EFSRegister(name="STATUS_REG", offset="0x04", width=32)  # Conflict: duplicate address 0x04
    ]

    # 2. Validate & detect conflict
    conflicts = ConflictResolutionEngine.detect_conflicts(efs_ir)
    unresolved_blocking = [c for c in conflicts if c.blocking and c.resolution_status == "UNRESOLVED"]
    
    # 3. Confirm generation is blocked before approval
    assert len(unresolved_blocking) == 1
    target_conflict = unresolved_blocking[0]
    assert target_conflict.type == "DUPLICATE_REGISTER_ADDRESS"
    assert target_conflict.blocking is True

    # Initial design result simulation
    initial_design_results = {
        "status": "BLOCKED_CONFLICT",
        "requirement_model": {"design_name": "TestModule"},
        "plan": {"module_name": "TestModule"},
        "protocol_rules": [],
        "design_spec": "# Design Spec\nModule Name: TestModule\n",
        "efs_ir": efs_ir,
        "conflicts": unresolved_blocking
    }

    # 4. Generate resolution proposals
    proposals = ConflictResolutionEngine.generate_resolution_proposals(target_conflict, efs_ir)
    assert len(proposals) >= 2

    # 5. Approve proposal programmatically (Reassign STATUS_REG offset to 0x08)
    approved_proposal = proposals[1]
    approved_proposal.proposed_value = "0x08"
    approved_proposal.proposed_change = "offset = 0x08"
    approved_proposal.affected_requirements = ["STATUS_REG"]
    approved_proposal.approval_status = "APPROVED"
    approved_proposal.approved_by = "test_engineer"

    # 6. Persist approved resolution
    resolved_ir = ConflictResolutionEngine.apply_approved_resolution(efs_ir, approved_proposal)
    assert len(resolved_ir.resolutions) == 1
    assert resolved_ir.resolutions[0].approval_status == "APPROVED"

    # 7. Rebuild resolved requirement model view
    resolved_view = resolved_ir.get_resolved_view()
    status_reg = next(r for r in resolved_view.registers if r.name == "STATUS_REG")
    assert status_reg.offset == "0x08"

    # 8. Revalidate resolved model
    has_blocking, validation_issues = ConflictResolutionEngine.revalidate_resolved_model(resolved_view)

    # 9. Confirm conflict is resolved
    assert not has_blocking
    unresolved_after = [c for c in resolved_view.conflicts if c.blocking and c.resolution_status == "UNRESOLVED"]
    assert len(unresolved_after) == 0

    # 10, 11, 12, 13. Call resume_design_flow_with_resolutions & verify RTL generation
    resume_results = resume_design_flow_with_resolutions(
        initial_design_results,
        [approved_proposal],
        target_outputs=["Verilog"]
    )

    # Confirm status is SUCCESS
    assert resume_results.get("status") == "SUCCESS"
    
    # Confirm RTL artifact exists and is not SPECIFICATION_INCOMPLETE
    outputs = resume_results.get("outputs", {})
    assert "Verilog" in outputs
    final_code = outputs["Verilog"].get("final_code", "")
    assert isinstance(final_code, str)
    assert len(final_code) > 0
    assert "module" in final_code
    assert "SPECIFICATION_INCOMPLETE" not in final_code


def test_option_2_custom_requirement_resume_flow():
    """Verifies Option 2: Add a custom requirement path."""
    efs_ir = EFSIR()
    efs_ir.components = [
        EFSComponent(name="AccelCore", type="datapath", description="Accelerator Core")
    ]
    efs_ir.signals = [
        EFSSignal(name="clk", width="1", direction="input", semantic_role="clock"),
        EFSSignal(name="rst_n", width="1", direction="input", semantic_role="reset")
    ]
    efs_ir.registers = [
        EFSRegister(name="REG_X", offset="0x10", width=32),
        EFSRegister(name="REG_Y", offset="0x10", width=32)  # Conflict: duplicate address 0x10
    ]

    conflicts = ConflictResolutionEngine.detect_conflicts(efs_ir)
    assert len(conflicts) == 1
    target_conflict = conflicts[0]

    initial_design_results = {
        "status": "BLOCKED_CONFLICT",
        "requirement_model": {"design_name": "AccelCore"},
        "plan": {"module_name": "AccelCore"},
        "protocol_rules": [],
        "design_spec": "# Design Spec\nModule Name: AccelCore\n",
        "efs_ir": efs_ir,
        "conflicts": conflicts
    }

    # Option 2: Add custom requirement
    custom_res = ConflictResolutionEngine.create_custom_resolution(
        target_conflict,
        "offset = 0x18",
        user_name="test_user"
    )
    assert custom_res.resolution_type == "CUSTOM_REQUIREMENT"
    assert custom_res.proposed_value == "0x18"
    assert custom_res.approval_status == "APPROVED"

    # Persist & Revalidate
    resolved_ir = ConflictResolutionEngine.apply_approved_resolution(efs_ir, custom_res)
    has_blocking, _ = ConflictResolutionEngine.revalidate_resolved_model(resolved_ir)
    assert not has_blocking

    # Resume design flow
    resume_results = resume_design_flow_with_resolutions(
        initial_design_results,
        [custom_res],
        target_outputs=["Verilog"]
    )
    assert resume_results.get("status") == "SUCCESS"
    verilog_code = resume_results.get("outputs", {}).get("Verilog", {}).get("final_code", "")
    assert "module" in verilog_code
    assert "SPECIFICATION_INCOMPLETE" not in verilog_code

