"""
Validation Pipeline Comprehensive Test Suite.

Tests 7-tier validation classification, interface role semantic normalization,
generic design support, negative test cases, and anti-hardcoding behavior.
"""

from __future__ import annotations

import pytest
from core.verification.compatibility_engine import ProtocolRoleMapper, CompatibilityEngine
from core.agents.rtl_agent import validate_rtl_plan, plan_rtl
from core.efs_ir.builder import EFSIRBuilder


def test_protocol_role_mapper_axi4_lite():
    """Verify standard AXI4-Lite port mapping and slave output/input direction deduction."""
    # Write response (BRESP/bresp) must be output for a slave
    role_bresp = ProtocolRoleMapper.get_semantic_role("bresp", interface_role="slave")
    assert role_bresp is not None
    assert role_bresp[0] == "write_response"
    assert role_bresp[2] == "output"
    assert role_bresp[3] == "BRESP"

    # Read data (RDATA/read_data) must be output for a slave
    role_rdata = ProtocolRoleMapper.get_semantic_role("read_data", interface_role="slave")
    assert role_rdata is not None
    assert role_rdata[0] == "read_response"
    assert role_rdata[2] == "output"
    assert role_rdata[3] == "RDATA"

    # Write address (AWADDR) must be input for a slave
    role_awaddr = ProtocolRoleMapper.get_semantic_role("awaddr", interface_role="slave")
    assert role_awaddr is not None
    assert role_awaddr[2] == "input"
    assert role_awaddr[3] == "AWADDR"


def test_efs_ir_builder_axi_slave_directions():
    """Verify EFS IR Builder automatically sets correct slave directions for AXI signals."""
    req_model = {
        "title": "Matrix Compute Unit",
        "protocol_references": ["AXI4-Lite"],
        "clocks_resets": [{"name": "aclk", "type": "clock"}, {"name": "aresetn", "type": "reset"}],
        "interfaces": [
            {"name": "awaddr", "width": "32"},  # Missing raw direction, should infer input
            {"name": "bresp", "width": "2"},    # Missing raw direction, should infer output
            {"name": "read_data", "width": "32"} # Missing raw direction, should infer output
        ]
    }
    builder = EFSIRBuilder(requirement_model=req_model, design_spec="Implement AXI4-Lite slave compute module.")
    efs_ir = builder.build()

    sig_map = {s.name: s for s in efs_ir.signals}
    assert sig_map["awaddr"].direction == "input"
    assert sig_map["bresp"].direction == "output"
    assert sig_map["read_data"].direction == "output"


def test_compatibility_engine_conflict_detection():
    """Verify CompatibilityEngine flags interface direction mismatches as CONFLICT issues."""
    user_ports = [
        {"name": "bresp", "direction": "input", "width": "2"} # User spec mistakenly marks bresp as input
    ]
    
    # EFS IR has bresp correctly as output
    builder = EFSIRBuilder(
        requirement_model={
            "clocks_resets": [{"name": "clk", "type": "clock"}],
            "interfaces": [{"name": "bresp", "direction": "output", "width": "2"}]
        },
        design_spec="AXI4-Lite slave module"
    )
    efs_ir = builder.build()

    issues = CompatibilityEngine.validate_interface_signals(user_ports, efs_ir.signals, interface_role="slave")
    assert len(issues) > 0
    assert any(i["classification"] == "CONFLICT" for i in issues)
    assert any("direction conflict" in i["issue"].lower() for i in issues)


def test_negative_cases():
    """Verify negative test cases trigger expected validation categories."""
    # 1. Undefined FSM transition condition -> AMBIGUOUS
    rtl_plan_ambiguous = {
        "module_name": "test_mod",
        "ports": [{"name": "clk", "direction": "input"}],
        "fsm": {
            "states": ["IDLE", "RUN"],
            "is_explicit": True,
            "transitions": [
                {"from": "IDLE", "to": "RUN", "condition": "control start condition met"}
            ]
        }
    }
    val_ambiguous = validate_rtl_plan(rtl_plan_ambiguous)
    assert not val_ambiguous["valid"]
    assert any(i["classification"] in ("AMBIGUOUS", "PLAN_AMBIGUOUS", "SOURCE_AMBIGUOUS") for i in val_ambiguous["issues"])

    # 2. Missing condition -> MISSING
    rtl_plan_missing = {
        "module_name": "test_mod",
        "ports": [{"name": "clk", "direction": "input"}],
        "fsm": {
            "states": ["IDLE", "RUN"],
            "is_explicit": True,
            "transitions": [
                {"from": "IDLE", "to": "RUN", "condition": None}
            ]
        }
    }
    val_missing = validate_rtl_plan(rtl_plan_missing)
    assert not val_missing["valid"]
    assert any(i["classification"] in ("MISSING", "PLAN_MISSING", "SOURCE_MISSING") for i in val_missing["issues"])


def test_anti_hardcoding_unrelated_designs():
    """Verify validator works for two completely unrelated designs without sample-specific hardcoding."""
    # Design 1: UART Transmitter
    plan1 = {
        "module_name": "uart_tx_engine",
        "ports": [
            {"name": "clk", "direction": "input", "width": "1"},
            {"name": "tx_data", "direction": "input", "width": "8"},
            {"name": "tx_start", "direction": "input", "width": "1"},
            {"name": "tx_out", "direction": "output", "width": "1"}
        ],
        "fsm": {
            "states": ["TX_IDLE", "TX_START", "TX_DATA", "TX_STOP"],
            "is_explicit": True,
            "transitions": [
                {"from": "TX_IDLE", "to": "TX_START", "condition": "tx_start == 1'b1"},
                {"from": "TX_START", "to": "TX_DATA", "condition": "baud_cnt == 4'd15"}
            ]
        }
    }
    val1 = validate_rtl_plan(plan1)
    assert val1["valid"] is True

    # Design 2: DMA Stream Controller
    plan2 = {
        "module_name": "dma_stream_controller",
        "ports": [
            {"name": "sys_clk", "direction": "input", "width": "1"},
            {"name": "dma_req", "direction": "input", "width": "1"},
            {"name": "stream_data", "direction": "output", "width": "64"}
        ],
        "fsm": {
            "states": ["DMA_INIT", "DMA_TRANSFER", "DMA_DONE"],
            "is_explicit": True,
            "transitions": [
                {"from": "DMA_INIT", "to": "DMA_TRANSFER", "condition": "dma_req == 1'b1"}
            ]
        }
    }
    val2 = validate_rtl_plan(plan2)
    assert val2["valid"] is True
