"""
Unit test suite for Knowledge Graph generation from User Specifications.

Verifies that signals, interfaces, relationships, and components extracted
from an uploaded User Specification are correctly built and mapped in the Knowledge Graph.
"""

import pytest
from core.efs_ir.models import (
    EFSIR, EFSMetadata, EFSComponent, EFSInterface, EFSSignal, EFSRegister, EFSRegisterField, EFSFSM, EFSState
)
from utils.models import Entity, Relationship
from core.graph.neo4j_builder import build_graph, get_graph_store


@pytest.fixture
def extracted_user_spec_data():
    """Sample extracted components, signals, interfaces, and relationships from a User Spec."""
    # 1. Extracted legacy entities
    entities = [
        Entity(entity_id="ENT_01", name="Custom_Peripheral_Controller", raw_name="Custom Peripheral Controller", entity_type="Component", chunk_id="BLK_1", doc_id="user_spec_01"),
        Entity(entity_id="ENT_02", name="AXI4Lite_Slave_Interface", raw_name="AXI4Lite Slave Interface", entity_type="Interface", chunk_id="BLK_1", doc_id="user_spec_01"),
        Entity(entity_id="ENT_03", name="aclk", raw_name="aclk", entity_type="Signal", chunk_id="BLK_2", doc_id="user_spec_01"),
        Entity(entity_id="ENT_04", name="aresetn", raw_name="aresetn", entity_type="Signal", chunk_id="BLK_2", doc_id="user_spec_01"),
        Entity(entity_id="ENT_05", name="Control_Register", raw_name="Control Register", entity_type="Register", chunk_id="BLK_3", doc_id="user_spec_01"),
    ]

    # 2. Extracted relationships
    relationships = [
        Relationship(rel_id="REL_01", source="Custom_Peripheral_Controller", relation="CONTAINS_INTERFACE", target="AXI4Lite_Slave_Interface", chunk_id="BLK_1", doc_id="user_spec_01"),
        Relationship(rel_id="REL_02", source="AXI4Lite_Slave_Interface", relation="USES_SIGNAL", target="aclk", chunk_id="BLK_2", doc_id="user_spec_01"),
        Relationship(rel_id="REL_03", source="Custom_Peripheral_Controller", relation="HAS_REGISTER", target="Control_Register", chunk_id="BLK_3", doc_id="user_spec_01"),
    ]

    # 3. Canonical EFS IR representation built from User Spec
    efs_ir = EFSIR(
        metadata=EFSMetadata(
            design_id="DSN_USER_01",
            design_name="custom_peripheral_controller",
            canonical_id="customperipheralcontroller",
            display_name="Custom Peripheral Controller",
            aliases=["Custom Peripheral Controller", "customperipheralcontroller"],
            document_id="user_spec_01"
        ),
        components=[
            EFSComponent(component_id="COMP_TOP", name="Custom Peripheral Controller", type="top_level_design", description="Top level controller"),
            EFSComponent(component_id="COMP_REG", name="Register_File", type="register_file", description="Register storage")
        ],
        interfaces=[
            EFSInterface(interface_id="IF_01", name="AXI4Lite_Slave_Interface", protocol="AXI4Lite", role="slave")
        ],
        signals=[
            EFSSignal(signal_id="SIG_01", name="aclk", width="1", direction="input", semantic_role="clock"),
            EFSSignal(signal_id="SIG_02", name="aresetn", width="1", direction="input", semantic_role="reset")
        ],
        registers=[
            EFSRegister(
                register_id="REG_01", name="Control_Register", offset="0x00", width=32, access_type="RW", reset_value="0x00000000",
                fields=[EFSRegisterField(name="ENABLE", msb=0, lsb=0, access="RW", reset_value="0")]
            )
        ]
    )

    return entities, relationships, efs_ir


def test_build_graph_from_user_spec(extracted_user_spec_data):
    """Verify that build_graph upserts extracted components, signals, interfaces, registers, and relationships."""
    entities, relationships, efs_ir = extracted_user_spec_data

    # Execute graph build
    stats = build_graph(entities, relationships, efs_ir=efs_ir)

    store = get_graph_store()
    all_nodes = store.all_nodes()
    all_edges = store.all_edges()

    node_names = {n["name"] for n in all_nodes}
    node_types = {n.get("type") for n in all_nodes}

    # Verify extracted component nodes exist
    assert "Custom Peripheral Controller" in node_names or "Custom_Peripheral_Controller" in node_names
    assert "AXI4Lite_Slave_Interface" in node_names
    assert "aclk" in node_names
    assert "Control_Register" in node_names

    # Verify entity types extracted from spec exist
    assert "Component" in node_types or "top_level_design" in node_types
    assert "Interface" in node_types
    assert "Signal" in node_types
    assert "Register" in node_types

    # Verify extracted relationships exist
    relations = {e["relation"] for e in all_edges}
    assert any(r in relations for r in ("CONTAINS_INTERFACE", "USES_SIGNAL", "HAS_REGISTER", "HAS_FIELD", "MAPS_TO"))
