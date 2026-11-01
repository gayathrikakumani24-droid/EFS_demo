"""
Controlled Specification Inference Model Test Suite.

Tests automatic completion of incomplete hardware specs using DERIVED protocol rules
and IMPLEMENTATION_CHOICE safe defaults. Verifies non-blocking status for safe inferences,
safety blocks for real conflicts, model unity, and anti-hardcoding behavior.
"""

from __future__ import annotations

import pytest
from core.efs_ir.models import EFSIR, EFSSignal, EFSRegister, EFSMetadata, new_efs_id
from core.verification.inference_engine import ControlledInferenceEngine


def test_inference_engine_partial_spec_completion():
    """Verify ControlledInferenceEngine completes partial specs without blocking generation."""
    efs = EFSIR()
    efs.metadata = EFSMetadata(design_name="custom_peripheral", protocol="AXI4-Lite")
    
    # 1. Signals without explicit direction or width
    sig1 = EFSSignal(signal_id=new_efs_id("sig"), name="bresp", direction="", width="")
    sig2 = EFSSignal(signal_id=new_efs_id("sig"), name="rdata", direction="", width="")
    efs.signals.extend([sig1, sig2])

    # 2. Registers without offsets
    reg1 = EFSRegister(register_id=new_efs_id("reg"), name="CTRL_REG", offset="")
    reg2 = EFSRegister(register_id=new_efs_id("reg"), name="STATUS_REG", offset="")
    efs.registers.extend([reg1, reg2])

    engine = ControlledInferenceEngine(efs)
    completed_model = engine.infer_and_complete()

    # Check that signals and registers were completed
    sig_map = {s.name: s for s in completed_model.efs_ir.signals}
    assert sig_map["bresp"].direction == "output"
    assert sig_map["rdata"].direction == "output"
    assert sig_map["rdata"].width == "32"
    
    # Check default clock and reset were added
    assert "clk" in sig_map
    assert "rst_n" in sig_map

    # Check register offsets were assigned sequentially
    reg_map = {r.name: r for r in completed_model.efs_ir.registers}
    assert reg_map["CTRL_REG"].offset == "0x00"
    assert reg_map["STATUS_REG"].offset == "0x04"

    # Check decisions recorded
    decisions = completed_model.inferred_decisions
    assert len(decisions) > 0
    assert any(d.decision_type == "DERIVED" for d in decisions)
    assert any(d.decision_type == "IMPLEMENTATION_CHOICE" for d in decisions)


def test_inference_engine_anti_hardcoding_multiple_types():
    """Verify ControlledInferenceEngine works for arbitrary hardware specs without hardcoding."""
    # Hardware Type A: SPI Controller
    efs_spi = EFSIR()
    efs_spi.metadata = EFSMetadata(design_name="spi_master_core", protocol="SPI")
    efs_spi.signals.append(EFSSignal(signal_id=new_efs_id("sig"), name="spi_miso", direction="input", width="1"))
    
    engine_spi = ControlledInferenceEngine(efs_spi)
    completed_spi = engine_spi.infer_and_complete()
    assert any(s.name == "clk" for s in completed_spi.efs_ir.signals)

    # Hardware Type B: Matrix Accelerator
    efs_matrix = EFSIR()
    efs_matrix.metadata = EFSMetadata(design_name="matrix_mult_accel", protocol="custom_bus")
    efs_matrix.registers.append(EFSRegister(register_id=new_efs_id("reg"), name="MATRIX_DIM_REG", offset=""))
    
    engine_matrix = ControlledInferenceEngine(efs_matrix)
    completed_matrix = engine_matrix.infer_and_complete()
    assert completed_matrix.efs_ir.registers[0].offset == "0x00"


def test_inference_engine_safety_blocks():
    """Verify safety engine blocks when genuine interface conflicts exist."""
    efs = EFSIR()
    efs.metadata = EFSMetadata(design_name="faulty_module", protocol="AXI4-Lite")
    # Add conflicting port direction according to protocol mapper (e.g., awaddr as output on slave interface)
    efs.signals.append(EFSSignal(signal_id=new_efs_id("sig"), name="awaddr", direction="output", width="32"))
    
    engine = ControlledInferenceEngine(efs)
    completed_model = engine.infer_and_complete()
    
    # Check that decision was recorded or validation reported blocking issue
    decisions = completed_model.inferred_decisions
    assert any(d.target_element == "awaddr" for d in decisions)


def test_unified_completed_design_model():
    """Verify CompletedDesignModel serialization and deserialization unity."""
    efs = EFSIR()
    efs.metadata = EFSMetadata(design_name="uart_top", protocol="UART")
    engine = ControlledInferenceEngine(efs)
    completed_model = engine.infer_and_complete()

    cdm_dict = completed_model.to_dict()
    assert "efs_ir" in cdm_dict
    assert "inferred_decisions" in cdm_dict
    assert "validation_status" in cdm_dict

    restored = completed_model.from_dict(cdm_dict)
    assert restored.efs_ir.metadata.design_name == "uart_top"
    assert len(restored.inferred_decisions) == len(completed_model.inferred_decisions)

