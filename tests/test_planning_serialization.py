"""
Tests ensuring that RequirementIR and nested objects are properly sanitized and serialized
without raising TypeError: Object of type RequirementIR is not JSON serializable.
"""

import sys
import os
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.requirement_ir.models import RequirementIR, AtomicRequirement, DiscoveredEntity
from core.agents.planning_agent import create_plan
from core.agents.spec_agent import generate_design_spec
from core.agents.requirement_agent import analyze_requirements


@patch("core.agents.planning_agent.get_llm_client")
def test_create_plan_with_requirement_ir_object(mock_llm):
    """Test create_plan when requirement_model is a RequirementIR object directly."""
    mock_client = mock_llm.return_value
    mock_client.available = False

    req_ir = RequirementIR(
        document_id="doc_test",
        requirements=[
            AtomicRequirement(
                requirement_id="REQ_001",
                statement="The module shall implement a 32-bit adder.",
                type="BEHAVIOR"
            )
        ],
        entities=[
            DiscoveredEntity(
                name="Adder32",
                type="module",
                description="32-bit hardware adder"
            )
        ]
    )

    # Calling create_plan with a RequirementIR object must not raise TypeError
    plan = create_plan(req_ir)
    assert plan is not None
    assert isinstance(plan, dict)
    assert "submodules" in plan


@patch("core.agents.planning_agent.get_llm_client")
def test_create_plan_with_nested_raw_req_ir(mock_llm):
    """Test create_plan when requirement_model dict contains a RequirementIR object."""
    mock_client = mock_llm.return_value
    mock_client.available = False

    req_ir = RequirementIR(document_id="doc_test")
    req_dict = {
        "module_name": "TestModule",
        "interfaces": [{"name": "clk", "direction": "input", "width": 1}],
        "_raw_req_ir": req_ir,
        "submodules": [{"name": "Sub1", "purpose": "Testing"}]
    }

    plan = create_plan(req_dict)
    assert plan is not None
    assert isinstance(plan, dict)


@patch("core.agents.spec_agent.get_llm_client")
def test_generate_design_spec_with_requirement_ir_object(mock_llm):
    """Test generate_design_spec when requirement_model is a RequirementIR object."""
    mock_client = mock_llm.return_value
    mock_client.available = False

    req_ir = RequirementIR(
        document_id="doc_test",
        requirements=[
            AtomicRequirement(
                requirement_id="REQ_001",
                statement="The module shall implement a 32-bit adder.",
                type="BEHAVIOR"
            )
        ]
    )
    plan = {"submodules": [{"name": "AdderCore", "purpose": "Arithmetic"}]}
    rules = []

    spec = generate_design_spec(req_ir, plan, rules)
    assert spec is not None
    assert isinstance(spec, str)
    assert len(spec) > 0


def test_analyze_requirements_with_requirement_ir_object():
    """Test analyze_requirements when input is already a RequirementIR object."""
    req_ir = RequirementIR(
        document_id="doc_test",
        requirements=[
            AtomicRequirement(requirement_id="REQ_001", statement="Test statement", type="BEHAVIOR")
        ]
    )
    res = analyze_requirements(req_ir)
    assert isinstance(res, dict)
    import json
    # Must be 100% JSON serializable
    json_str = json.dumps(res, indent=2)
    assert "REQ_001" in json_str


def test_get_heuristic_spec_with_missing_keys():
    """Test get_heuristic_spec when interfaces or registers lack 'width', 'direction', etc."""
    from core.agents.spec_agent import get_heuristic_spec

    # Model with missing 'width', 'direction', 'description', 'offset', 'access'
    req_model = {
        "interfaces": [
            {"name": "req_val"}  # NO 'width', 'direction', or 'description'
        ],
        "registers": [
            {"name": "CTRL"}      # NO 'offset', 'access', or 'description'
        ],
        "protocol_references": [] # empty list
    }
    plan = {"submodules": []}
    rules = [{"rule": "Check handshake timing"}] # NO 'citation' or 'text'

    spec = get_heuristic_spec(req_model, plan, rules)
    assert spec is not None
    assert "req_val" in spec
    assert "CTRL" in spec
    assert "Check handshake timing" in spec

