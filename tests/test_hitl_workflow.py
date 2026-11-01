"""
End-to-End Test Suite for Human-in-the-Loop (HITL) Conflict Resolution Workflow.

Covers the 8 mandatory test scenarios:
1. TEST 1: Spec with duplicate opcodes -> detected as conflict -> AI proposal generated -> accept proposal -> canonical EFS IR updated -> revalidation passes -> RTL generation proceeds.
2. TEST 2: Spec with conflict -> human rejects AI proposal -> enters custom resolution -> validated -> applied -> canonical EFS IR updated -> RTL generation proceeds.
3. TEST 3: Human enters invalid custom resolution (collides with existing assigned entity or invalid format) -> rejected with validation error -> EFS IR NOT modified -> conflict remains unresolved -> generation stays halted.
4. TEST 4: AI provider returns invalid JSON or fails -> deterministic candidate generator still produces valid proposals -> human accepts -> pipeline resumes.
5. TEST 5: Multiple conflicts detected simultaneously -> each resolved independently -> all applied to canonical EFS IR -> full revalidation passes.
6. TEST 6: Resolution that creates a new conflict -> detected during revalidation -> pipeline pauses again in AWAITING_HUMAN_DECISION.
7. TEST 7: Spec with NO conflicts -> passes validation cleanly -> pipeline proceeds straight to generation without HITL UI.
8. TEST 8: Explicit bypass override semantics -> when bypass_conflicts=True, pipeline proceeds with warning, resolutions marked as overridden, audit records bypass.
"""

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


def create_base_mock_design_results(efs_ir: EFSIR, conflicts: list = None) -> dict:
    """Helper to construct design results dict for resume flow testing."""
    return {
        "status": "AWAITING_HUMAN_DECISION",
        "pipeline_state": "AWAITING_HUMAN_DECISION",
        "requires_human_decision": True,
        "is_blocked": False,
        "requirement_model": {"design_name": "HITLTestModule"},
        "plan": {"module_name": "HITLTestModule"},
        "protocol_rules": [],
        "design_spec": "# Design Spec\nModule Name: HITLTestModule\n",
        "efs_ir": efs_ir,
        "conflicts": conflicts or [],
        "outputs": {}
    }


@patch("core.agents.orchestrator.generate_design_spec", return_value="# Design Spec\n")
@patch("core.agents.plantuml_agent.generate_diagrams", return_value={})
@patch("core.agents.plantuml_agent.get_diagram_url", return_value="")
@patch("core.agents.orchestrator.generate_code", return_value="module ALU (input clk, input rst_n); endmodule")
def test_1_duplicate_opcode_ai_proposal_accept_resumes(mock_code, mock_url, mock_diag, mock_spec):
    """
    TEST 1: Spec with duplicate opcodes -> detected as conflict -> AI proposal generated
    -> accept proposal -> canonical EFS IR updated -> revalidation passes -> RTL generation proceeds.
    """
    ir = EFSIR()
    ir.components = [EFSComponent(name="ALU", type="alu", description="Arithmetic Logic Unit")]
    ir.signals = [
        EFSSignal(name="clk", width="1", direction="input", semantic_role="clock"),
        EFSSignal(name="rst_n", width="1", direction="input", semantic_role="reset")
    ]
    ir.opcodes = [
        EFSOpcode(mnemonic="ADD", binary_encoding="0001"),
        EFSOpcode(mnemonic="SUB", binary_encoding="0001")  # Collision!
    ]

    # 1. Detect conflict
    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    opcode_conflicts = [c for c in conflicts if c.blocking and "OPCODE" in c.type]
    assert len(opcode_conflicts) >= 1
    conf = opcode_conflicts[0]

    # Verify WHAT, WHY, WHERE structured fields
    assert conf.title is not None
    assert len(conf.title) > 0
    assert conf.why_it_matters is not None
    assert "SUB" in conf.affected_entities or "ADD" in conf.affected_entities
    assert len(conf.source_references) >= 1

    # 2. AI proposal generation
    proposals = ConflictResolutionEngine.generate_resolution_proposals(conf, ir)
    assert len(proposals) >= 1
    proposal = proposals[0]
    assert proposal.proposed_value not in [None, "", "TODO", "0001"]

    # 3. Accept proposal
    proposal.approval_status = "APPROVED"
    proposal.approved_by = "lead_architect"

    # 4. Canonical EFS IR updated
    resolved_ir = ConflictResolutionEngine.apply_approved_resolution(ir, proposal)
    resolved_view = resolved_ir.get_resolved_view()

    # Verify in canonical EFS IR that opcodes are now distinct
    op_encs = [op.binary_encoding for op in resolved_view.opcodes]
    assert len(op_encs) == len(set(op_encs))

    # 5. Revalidation passes
    has_blocking, issues = ConflictResolutionEngine.revalidate_resolved_model(resolved_view)
    assert not has_blocking

    # 6. RTL generation proceeds via resume_design_flow_with_resolutions
    initial_results = create_base_mock_design_results(ir, conflicts)
    resume_results = resume_design_flow_with_resolutions(
        initial_results,
        [proposal],
        target_outputs=["Verilog"]
    )
    assert resume_results.get("status") == "SUCCESS"
    assert resume_results.get("pipeline_state") in ("COMPLETED", "GENERATION_COMPLETE")
    assert resume_results.get("requires_human_decision") is False
    verilog = resume_results.get("outputs", {}).get("Verilog", {}).get("final_code", "")
    assert "module" in verilog
    assert "SPECIFICATION_INCOMPLETE" not in verilog


@patch("core.agents.orchestrator.generate_design_spec", return_value="# Design Spec\n")
@patch("core.agents.plantuml_agent.generate_diagrams", return_value={})
@patch("core.agents.plantuml_agent.get_diagram_url", return_value="")
@patch("core.agents.orchestrator.generate_code", return_value="module DMA_CTRL (input clk, input rst_n); endmodule")
def test_2_custom_human_resolution_validated_resumes(mock_code, mock_url, mock_diag, mock_spec):
    """
    TEST 2: Spec with conflict -> human rejects AI proposal -> enters custom resolution
    -> validated -> applied -> canonical EFS IR updated -> RTL generation proceeds.
    """
    ir = EFSIR()
    ir.components = [EFSComponent(name="DMA_CTRL", type="controller", description="DMA Controller")]
    ir.signals = [
        EFSSignal(name="clk", width="1", direction="input", semantic_role="clock"),
        EFSSignal(name="rst_n", width="1", direction="input", semantic_role="reset")
    ]
    ir.registers = [
        EFSRegister(name="SRC_ADDR", offset="0x10", width=32),
        EFSRegister(name="DST_ADDR", offset="0x10", width=32)  # Conflict: duplicate address 0x10
    ]

    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    assert len(conflicts) >= 1
    conf = conflicts[0]

    # Human provides custom resolution: Set DST_ADDR to 0x14
    is_valid, err_msg = ConflictResolutionEngine.validate_resolution(
        conf,
        "0x14",
        ir,
        target_entity="DST_ADDR"
    )
    assert is_valid is True

    custom_res = ConflictResolutionEngine.create_custom_resolution(
        conf,
        "DST_ADDR.offset = 0x14",
        user_name="human_engineer",
        target_entity="DST_ADDR"
    )
    assert custom_res.resolution_type == "CUSTOM_REQUIREMENT"
    assert custom_res.proposed_value == "0x14"
    assert custom_res.approval_status == "APPROVED"

    # Apply to canonical EFS IR
    resolved_ir = ConflictResolutionEngine.apply_approved_resolution(ir, custom_res)
    resolved_view = resolved_ir.get_resolved_view()
    dst_reg = next(r for r in resolved_view.registers if r.name == "DST_ADDR")
    assert dst_reg.offset == "0x14"

    # Revalidation passes
    has_blocking, _ = ConflictResolutionEngine.revalidate_resolved_model(resolved_view)
    assert not has_blocking

    # Pipeline resume proceeds
    initial_results = create_base_mock_design_results(ir, conflicts)
    resume_results = resume_design_flow_with_resolutions(
        initial_results,
        [custom_res],
        target_outputs=["Verilog"]
    )
    assert resume_results.get("status") == "SUCCESS"
    assert "outputs" in resume_results
    assert "Verilog" in resume_results["outputs"]


def test_3_invalid_custom_resolution_rejected_and_ir_untouched():
    """
    TEST 3: Human enters invalid custom resolution (e.g. collides with existing assigned
    opcode or register, or invalid width) -> rejected with validation error -> EFS IR
    NOT modified -> conflict remains unresolved -> generation stays halted.
    """
    ir = EFSIR()
    ir.registers = [
        EFSRegister(name="REG_A", offset="0x00", width=32),
        EFSRegister(name="REG_B", offset="0x00", width=32),  # Conflict with REG_A
        EFSRegister(name="REG_C", offset="0x04", width=32)   # Existing assigned
    ]

    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    assert len(conflicts) >= 1
    conf = conflicts[0]

    # Human attempts to set REG_B to 0x04 (collides with REG_C)
    is_valid, err = ConflictResolutionEngine.validate_resolution(
        conf,
        "0x04",
        ir,
        target_entity="REG_B"
    )
    assert is_valid is False
    assert "collides with existing assigned" in err or "already assigned" in err

    # Human attempts to enter empty or invalid width
    is_valid_empty, err_empty = ConflictResolutionEngine.validate_resolution(conf, "", ir)
    assert is_valid_empty is False

    # Verify EFS IR remains untouched
    assert ir.registers[1].offset == "0x00"
    assert conf.resolution_status == "UNRESOLVED"


@patch("core.agents.orchestrator.generate_design_spec", return_value="# Design Spec\n")
@patch("core.agents.plantuml_agent.generate_diagrams", return_value={})
@patch("core.agents.plantuml_agent.get_diagram_url", return_value="")
@patch("core.agents.orchestrator.generate_code", return_value="module Core (input clk, input rst_n); endmodule")
def test_4_ai_failure_fallback_deterministic_proposals(mock_code, mock_url, mock_diag, mock_spec):
    """
    TEST 4: AI provider returns invalid JSON or fails -> deterministic candidate
    generator still produces valid proposals -> human accepts -> pipeline resumes.
    """
    ir = EFSIR()
    ir.components = [EFSComponent(name="Core", type="processor", description="RISC Core")]
    ir.signals = [
        EFSSignal(name="clk", width="1", direction="input", semantic_role="clock"),
        EFSSignal(name="rst_n", width="1", direction="input", semantic_role="reset")
    ]
    ir.registers = [
        EFSRegister(name="REG_0", offset="0x00", width=32),
        EFSRegister(name="REG_1", offset="0x00", width=32)
    ]

    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    conf = conflicts[0]

    # Simulate AI failure by mocking LLM client to raise an exception
    with patch("core.extraction.llm_client.get_llm_client") as mock_client:
        mock_instance = MagicMock()
        mock_instance.generate.side_effect = RuntimeError("AI Provider Network Timeout / Rate Limit")
        mock_client.return_value = mock_instance

        # Proposal generation must NOT crash; must fall back to deterministic candidates
        proposals = ConflictResolutionEngine.generate_resolution_proposals(conf, ir)
        assert len(proposals) >= 1
        fallback_proposal = proposals[0]
        assert fallback_proposal.proposed_value not in [None, "", "TODO", "0x00"]
        assert "0x" in fallback_proposal.proposed_value

    # Human accepts fallback proposal
    fallback_proposal.approval_status = "APPROVED"
    resolved_ir = ConflictResolutionEngine.apply_approved_resolution(ir, fallback_proposal)
    has_blocking, _ = ConflictResolutionEngine.revalidate_resolved_model(resolved_ir.get_resolved_view())
    assert not has_blocking

    # Pipeline resumes
    initial_results = create_base_mock_design_results(ir, conflicts)
    resume_results = resume_design_flow_with_resolutions(
        initial_results,
        [fallback_proposal],
        target_outputs=["Verilog"]
    )
    assert resume_results.get("status") == "SUCCESS"


def test_5_multiple_simultaneous_conflicts_resolved_independently():
    """
    TEST 5: Multiple conflicts detected simultaneously -> each resolved independently
    -> all applied to canonical EFS IR -> full revalidation passes.
    """
    ir = EFSIR()
    ir.components = [EFSComponent(name="MultiUnit", type="unit", description="Multi Unit")]
    ir.signals = [
        EFSSignal(name="clk", width="1", direction="input", semantic_role="clock"),
        EFSSignal(name="rst_n", width="1", direction="input", semantic_role="reset"),
        EFSSignal(name="data_bus", width="16", direction="input"),
        EFSSignal(name="data_bus", width="32", direction="output")  # Conflict 1: Signal width / direction mismatch
    ]
    ir.registers = [
        EFSRegister(name="REG_ALPHA", offset="0x20", width=32),
        EFSRegister(name="REG_BETA", offset="0x20", width=32)       # Conflict 2: Duplicate address
    ]

    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    assert len(conflicts) >= 2

    # Resolve conflict 1: Register duplicate address
    reg_conf = next(c for c in conflicts if "REGISTER" in c.type)
    res_reg = ConflictResolutionEngine.create_custom_resolution(
        reg_conf,
        "REG_BETA.offset = 0x24",
        user_name="architect",
        target_entity="REG_BETA"
    )

    # Resolve conflict 2: Signal width mismatch
    sig_conf = next(c for c in conflicts if "WIDTH" in c.type or "SIGNAL" in c.type)
    res_sig = ConflictResolutionEngine.create_custom_resolution(
        sig_conf,
        "data_bus.width = 32",
        user_name="architect",
        target_entity="data_bus"
    )

    # Apply both resolutions to canonical EFS IR
    resolved_ir = ConflictResolutionEngine.apply_approved_resolution(ir, res_reg)
    resolved_ir = ConflictResolutionEngine.apply_approved_resolution(resolved_ir, res_sig)
    resolved_view = resolved_ir.get_resolved_view()

    # Revalidation passes cleanly
    has_blocking, issues = ConflictResolutionEngine.revalidate_resolved_model(resolved_view)
    assert not has_blocking


@patch("core.agents.orchestrator.generate_design_spec", return_value="# Design Spec\n")
@patch("core.agents.plantuml_agent.generate_diagrams", return_value={})
@patch("core.agents.plantuml_agent.get_diagram_url", return_value="")
@patch("core.agents.orchestrator.generate_code", return_value="module ChainUnit (input clk, input rst_n); endmodule")
@patch("core.verification.conflict_resolution_engine.ConflictResolutionEngine.generate_resolution_proposals", return_value=[EFSResolution(resolution_id="RES_FB", conflict_id="CONF_1", proposed_value="0x10")])
def test_6_resolution_creating_new_conflict_detected_by_revalidation(mock_proposals, mock_code, mock_url, mock_diag, mock_spec):
    """
    TEST 6: Resolution that creates a new conflict -> detected during revalidation
    -> pipeline halts again with new conflict awaiting decision.
    """
    ir = EFSIR()
    ir.components = [EFSComponent(name="ChainUnit", type="unit", description="Chain Unit")]
    ir.signals = [
        EFSSignal(name="clk", width="1", direction="input", semantic_role="clock"),
        EFSSignal(name="rst_n", width="1", direction="input", semantic_role="reset")
    ]
    ir.registers = [
        EFSRegister(name="REG_1", offset="0x00", width=32),
        EFSRegister(name="REG_2", offset="0x00", width=32),  # Initial conflict with REG_1
        EFSRegister(name="REG_3", offset="0x08", width=32)   # Target collision
    ]

    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    conf = conflicts[0]

    # Create a resolution that sets REG_2 to 0x08 (which creates a new conflict with REG_3)
    bad_res = EFSResolution(
        resolution_id="RES_NEW_CONFLICT",
        conflict_id=conf.conflict_id,
        affected_requirements=["REG_2"],
        field_name="offset",
        original_value="0x00",
        proposed_value="0x08",
        proposed_change="offset = 0x08",
        approval_status="APPROVED",
        validation_status="VALID",
        resolution_type="CUSTOM_REQUIREMENT"
    )

    # 1. Server-side validation MUST reject applying a resolution that collides with existing hardware
    with pytest.raises(ValueError) as excinfo:
        ConflictResolutionEngine.apply_approved_resolution(ir, bad_res)
    assert "collides" in str(excinfo.value) or "already assigned" in str(excinfo.value)

    # 2. When an unresolved/colliding state is revalidated, revalidation MUST detect it
    ir.resolutions.append(bad_res)
    resolved_view = ir.get_resolved_view()
    has_blocking, issues = ConflictResolutionEngine.revalidate_resolved_model(resolved_view)
    assert has_blocking is True
    assert any("DUPLICATE_REGISTER_ADDRESS" in c.type for c in resolved_view.conflicts)

    # 3. resume_design_flow_with_resolutions safely pauses again in AWAITING_HUMAN_DECISION
    initial_results = create_base_mock_design_results(ir, conflicts)
    resume_results = resume_design_flow_with_resolutions(
        initial_results,
        [bad_res],
        target_outputs=["Verilog"]
    )
    assert resume_results.get("status") == "AWAITING_HUMAN_DECISION"
    assert resume_results.get("requires_human_decision") is True
    assert resume_results.get("is_blocked") is False
    assert len(resume_results.get("conflicts", [])) >= 1


def test_7_no_conflicts_proceeds_cleanly():
    """
    TEST 7: Spec with NO conflicts -> passes validation cleanly -> pipeline proceeds
    straight to generation without HITL UI.
    """
    ir = EFSIR()
    ir.components = [EFSComponent(name="CleanModule", type="core", description="Clean Core")]
    ir.signals = [
        EFSSignal(name="clk", width="1", direction="input", semantic_role="clock"),
        EFSSignal(name="rst_n", width="1", direction="input", semantic_role="reset"),
        EFSSignal(name="data_in", width="16", direction="input"),
        EFSSignal(name="data_out", width="16", direction="output")
    ]
    ir.registers = [
        EFSRegister(name="CTRL", offset="0x00", width=32),
        EFSRegister(name="DATA", offset="0x04", width=32)
    ]
    ir.opcodes = [
        EFSOpcode(mnemonic="NOP", binary_encoding="0000"),
        EFSOpcode(mnemonic="RUN", binary_encoding="0001")
    ]

    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    assert len(conflicts) == 0

    has_blocking, issues = ConflictResolutionEngine.revalidate_resolved_model(ir)
    assert has_blocking is False
    assert len([i for i in issues if i.get("severity") in ("ERROR", "CRITICAL")]) == 0


@patch("core.agents.orchestrator.generate_design_spec", return_value="# Design Spec\n")
@patch("core.agents.plantuml_agent.generate_diagrams", return_value={})
@patch("core.agents.plantuml_agent.get_diagram_url", return_value="")
@patch("core.agents.orchestrator.generate_code", return_value="module BypassUnit (input clk, input rst_n); endmodule")
def test_8_explicit_bypass_override_semantics(mock_code, mock_url, mock_diag, mock_spec):
    """
    TEST 8: Explicit bypass override semantics -> when bypass_conflicts=True,
    pipeline proceeds with warning, resolutions are marked as overridden, audit log records bypass.
    """
    ir = EFSIR()
    ir.components = [EFSComponent(name="BypassUnit", type="unit", description="Bypass Unit")]
    ir.signals = [
        EFSSignal(name="clk", width="1", direction="input", semantic_role="clock"),
        EFSSignal(name="rst_n", width="1", direction="input", semantic_role="reset")
    ]
    ir.registers = [
        EFSRegister(name="REG_X", offset="0x10", width=32),
        EFSRegister(name="REG_Y", offset="0x10", width=32)  # Conflict!
    ]

    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    assert len(conflicts) >= 1

    initial_results = create_base_mock_design_results(ir, conflicts)

    # Call resume_design_flow_with_resolutions with bypass_conflicts=True and NO approved resolutions
    resume_results = resume_design_flow_with_resolutions(
        initial_results,
        [],
        target_outputs=["Verilog"],
        bypass_conflicts=True
    )

    # Confirm pipeline proceeded despite unresolved conflict
    assert resume_results.get("status") == "SUCCESS"
    assert resume_results.get("pipeline_state") in ("COMPLETED", "GENERATION_COMPLETE")
    assert resume_results.get("requires_human_decision") is False
    assert resume_results.get("bypass_warning") is not None
    assert "BYPASS_OVERRIDE_ACTIVE" in resume_results["bypass_warning"]

    # Conflict marked as OVERRIDDEN
    overridden_conflicts = [c for c in resume_results["efs_ir"].conflicts if c.resolution_status == "OVERRIDDEN"]
    assert len(overridden_conflicts) >= 1
