"""
Unit Test Suite for Target Resolver and Design Model Pipeline.

Tests multi-tier target resolution:
1. Exact & normalized design target resolution
2. Alias & camelCase design name resolution
3. Primitive submodule component resolution
4. Structured TARGET_NOT_FOUND diagnostics for unknown targets
5. Anti-hardcoding test with arbitrary custom hardware design title
"""

import pytest
from core.efs_ir.models import EFSIR, EFSMetadata, EFSComponent
from core.efs_ir.builder import EFSIRBuilder
from core.retrieval.target_resolver import resolve_target


@pytest.fixture
def sample_design_efs_ir():
    spec_text = """
    # Hardware Design Specification: Simple Register-Controlled Data Engine

    ## Architecture
    Submodules: AXI4Lite_Slave, Register_Bank, Processing_Engine, Control_FSM

    ## Interface
    - aclk: input, 1 bit, Clock
    - aresetn: input, 1 bit, Reset
    - s_axi_awvalid: input, 1 bit, AWVALID
    - s_axi_awready: output, 1 bit, AWREADY

    ## Registers
    - CONTROL: 32 bits, Offset 0x00
    - STATUS: 32 bits, Offset 0x04
    - DATA: 32 bits, Offset 0x08
    """
    req_model = {
        "title": "Simple Register-Controlled Data Engine",
        "interfaces": [
            {"name": "aclk", "direction": "input", "width": "1"},
            {"name": "aresetn", "direction": "input", "width": "1"},
            {"name": "s_axi_awvalid", "direction": "input", "width": "1"},
            {"name": "s_axi_awready", "direction": "output", "width": "1"}
        ]
    }
    plan = {
        "system_architecture": "Simple Register-Controlled Data Engine",
        "submodules": [
            {"name": "AXI4Lite_Slave", "purpose": "AXI Protocol Handshake"},
            {"name": "Register_Bank", "purpose": "Register Storage"},
            {"name": "Processing_Engine", "purpose": "Datapath Computation"}
        ]
    }
    builder = EFSIRBuilder(req_model=req_model, plan=plan, design_spec=spec_text)
    return builder.build()


def test_1_resolve_user_spec_display_name(sample_design_efs_ir):
    """TEST 1 — User spec display name 'Simple Register-Controlled Data Engine' resolves successfully."""
    res = resolve_target("Simple Register-Controlled Data Engine", sample_design_efs_ir)
    assert res.is_success() is True
    assert res.target_component is not None
    assert "Simple" in res.target_component.name or "Register" in res.target_component.name


def test_2_resolve_pascal_case_design_target(sample_design_efs_ir):
    """TEST 2 — Input target 'SimpleRegisterControlledDataEngine' resolves via normalized/alias match."""
    res = resolve_target("SimpleRegisterControlledDataEngine", sample_design_efs_ir)
    assert res.is_success() is True
    assert res.target_component is not None


def test_3_resolve_canonical_snake_case_target(sample_design_efs_ir):
    """TEST 3 — Input target 'simple_register_controlled_data_engine' resolves via canonical ID."""
    res = resolve_target("simple_register_controlled_data_engine", sample_design_efs_ir)
    assert res.is_success() is True
    assert res.target_component is not None


def test_4_resolve_alias_registered_target(sample_design_efs_ir):
    """TEST 4 — Input target 'RegisteredControlledDataEngine' resolves via registered design alias."""
    res = resolve_target("RegisteredControlledDataEngine", sample_design_efs_ir)
    assert res.is_success() is True
    assert res.target_component is not None


def test_5_unknown_target_returns_structured_diagnostic(sample_design_efs_ir):
    """TEST 5 — Completely unknown target returns TARGET_NOT_FOUND with structured diagnostic error."""
    res = resolve_target("CompletelyUnknownHardwareBlock", sample_design_efs_ir)
    assert res.is_success() is False
    assert res.status == "TARGET_NOT_FOUND"
    assert "TARGET_RESOLUTION_FAILED" in res.error_message
    assert "CompletelyUnknownHardwareBlock" in res.error_message
    assert "Available EFS IR Components" in res.error_message


def test_6_resolve_primitive_submodule_component(sample_design_efs_ir):
    """TEST 6 — Primitive submodule component 'Register_Bank' resolves to specific internal component."""
    res = resolve_target("Register_Bank", sample_design_efs_ir)
    assert res.is_success() is True
    assert res.target_component.name == "Register_Bank"


def test_7_composite_design_retrieves_top_level_target(sample_design_efs_ir):
    """TEST 7 — Composite design resolution retrieves top-level design component."""
    res = resolve_target("Simple Register-Controlled Data Engine", sample_design_efs_ir)
    assert res.is_success() is True
    assert res.target_component.type == "top_level_design" or "top" in res.target_component.type.lower() or res.match_method in ("design_alias", "design_substring", "primary_component_default")


def test_8_anti_hardcoding_custom_hardware_design():
    """TEST 8 — Anti-Hardcoding: Arbitrary custom design title 'Configurable Counter Engine' resolves dynamically."""
    spec_text = """
    # Hardware Design Specification: Configurable Counter Engine

    ## Interface
    - clk: input, 1 bit, Clock
    - rst_n: input, 1 bit, Reset
    """
    req_model = {"title": "Configurable Counter Engine"}
    plan = {
        "system_architecture": "Configurable Counter Engine",
        "submodules": [{"name": "Counter_Core", "purpose": "Counting Logic"}]
    }
    builder = EFSIRBuilder(req_model=req_model, plan=plan, design_spec=spec_text)
    efs_ir = builder.build()

    res1 = resolve_target("Configurable Counter Engine", efs_ir)
    assert res1.is_success() is True

    res2 = resolve_target("ConfigurableCounterEngine", efs_ir)
    assert res2.is_success() is True

    res3 = resolve_target("configurable_counter_engine", efs_ir)
    assert res3.is_success() is True

    # Ensure it does NOT resolve to Simple Register-Controlled Data Engine
    assert "Counter" in res1.target_component.name
