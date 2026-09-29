"""
Comprehensive test suite for Neo4j / Knowledge Graph generation from User Spec and EFS IR.

Validates that:
1. All canonical hardware entities (Design, Component, Interface, Signal, Register,
   Field, Transaction, FSM, State, Constraint, TimingConstraint, Flow) are created as nodes.
2. All hardware relationships (CONTAINS_COMPONENT, HAS_INTERFACE, HAS_SIGNAL,
   HAS_REGISTER, HAS_FIELD, INITIATES, TARGETS, INVOLVES_SIGNAL, USES_INTERFACE,
   CLOCKS, RESETS, DRIVES, FEEDS, HAS_STATE, TRANSITIONS_TO, CONSTRAINS,
   HAS_TIMING_RULE, INVOLVES_COMPONENT, INVOLVES_INTERFACE) connect canonical node names.
3. Edge endpoints match node names so that UI graph filtering retains all connections.
"""

import pytest
from core.efs_ir.models import (
    EFSIR,
    EFSMetadata,
    EFSComponent,
    EFSInterface,
    EFSSignal,
    EFSRegister,
    EFSRegisterField,
    EFSTransaction,
    EFSFSM,
    EFSState,
    EFSTransition,
    EFSConstraint,
    EFSTimingRule,
    EFSFlow,
)
from utils.models import Entity, Relationship
from core.graph.neo4j_builder import build_graph, get_graph_store


@pytest.fixture(autouse=True)
def clean_graph_store():
    """Ensure clean graph store for test isolation."""
    store = get_graph_store()
    store.clear()
    yield
    store.clear()


@pytest.fixture
def sample_complete_efs_ir():
    """Build a comprehensive EFS IR fixture containing all hardware constructs."""
    return EFSIR(
        metadata=EFSMetadata(
            design_id="DSN_DMA_001",
            design_name="dma_subsystem",
            display_name="DMA Subsystem Controller",
            top_module="dma_top",
            protocol="AXI4",
            protocol_version="4.0",
            document_id="spec_dma_v1",
            aliases=["DMA Controller", "dma_subsystem"],
        ),
        components=[
            EFSComponent(
                component_id="COMP_CTRL",
                name="DMA_Main_Controller",
                type="controller",
                description="Central DMA state machine and command handler",
                interfaces=["IF_AXI_CFG", "IF_STREAM"],
                clock_domain="aclk",
                reset_domain="aresetn",
            ),
            EFSComponent(
                component_id="COMP_REG",
                name="DMA_Register_File",
                type="register_file",
                description="Control and status registers",
                clock_domain="aclk",
                reset_domain="aresetn",
            ),
            EFSComponent(
                component_id="COMP_FIFO",
                name="Data_FIFO",
                type="fifo",
                description="Internal buffering FIFO",
                clock_domain="aclk",
                reset_domain="aresetn",
            ),
        ],
        interfaces=[
            EFSInterface(
                interface_id="IF_AXI_CFG",
                name="AXI_Config_Slave",
                protocol="AXI4Lite",
                role="slave",
                signals=["SIG_AWADDR", "SIG_AWVALID", "SIG_AWREADY"],
            ),
            EFSInterface(
                interface_id="IF_STREAM",
                name="AXIS_Data_Master",
                protocol="AXIS",
                role="master",
                source_component="COMP_CTRL",
                destination_component="COMP_FIFO",
                signals=["SIG_TDATA", "SIG_TVALID"],
            ),
        ],
        signals=[
            EFSSignal(signal_id="SIG_CLK", name="aclk", width="1", direction="input", semantic_role="clock"),
            EFSSignal(signal_id="SIG_RST", name="aresetn", width="1", direction="input", semantic_role="reset"),
            EFSSignal(signal_id="SIG_AWADDR", name="s_axi_awaddr", width="32", direction="input", interface="IF_AXI_CFG", semantic_role="address"),
            EFSSignal(signal_id="SIG_AWVALID", name="s_axi_awvalid", width="1", direction="input", interface="IF_AXI_CFG", semantic_role="handshake"),
            EFSSignal(signal_id="SIG_AWREADY", name="s_axi_awready", width="1", direction="output", interface="IF_AXI_CFG", owner="COMP_CTRL", semantic_role="handshake"),
            EFSSignal(signal_id="SIG_TDATA", name="m_axis_tdata", width="64", direction="output", interface="IF_STREAM", owner="COMP_CTRL", consumer="COMP_FIFO", semantic_role="data"),
            EFSSignal(signal_id="SIG_TVALID", name="m_axis_tvalid", width="1", direction="output", interface="IF_STREAM", owner="COMP_CTRL", consumer="COMP_FIFO", semantic_role="control"),
        ],
        clocks=[
            EFSSignal(signal_id="SIG_CLK", name="aclk", width="1", direction="input", semantic_role="clock"),
        ],
        resets=[
            EFSSignal(signal_id="SIG_RST", name="aresetn", width="1", direction="input", semantic_role="reset"),
        ],
        registers=[
            EFSRegister(
                register_id="REG_CTRL",
                name="DMA_CONTROL",
                address="0x40000000",
                offset="0x00",
                width=32,
                access_type="RW",
                reset_value="0x00000000",
                owner="COMP_REG",
                description="DMA Control Register",
                fields=[
                    EFSRegisterField(name="START", msb=0, lsb=0, access="RW", reset_value="0", description="Start transfer"),
                    EFSRegisterField(name="IE", msb=1, lsb=1, access="RW", reset_value="0", description="Interrupt Enable"),
                    EFSRegisterField(name="BURST_LEN", msb=7, lsb=4, access="RW", reset_value="0", description="Burst Length"),
                ],
            ),
            EFSRegister(
                register_id="REG_STATUS",
                name="DMA_STATUS",
                address="0x40000004",
                offset="0x04",
                width=32,
                access_type="RO",
                reset_value="0x00000002",
                owner="COMP_REG",
                description="DMA Status Register",
                fields=[
                    EFSRegisterField(name="BUSY", msb=0, lsb=0, access="RO", reset_value="0", description="Engine busy"),
                    EFSRegisterField(name="DONE", msb=1, lsb=1, access="RO", reset_value="1", description="Transfer complete"),
                ],
            ),
        ],
        transactions=[
            EFSTransaction(
                transaction_id="TX_BURST_READ",
                name="DMA_Burst_Read_Tx",
                type="burst_read",
                protocol="AXI4",
                initiator="COMP_CTRL",
                target="COMP_FIFO",
                participating_interfaces=["IF_STREAM"],
                signals=["SIG_TDATA", "SIG_TVALID"],
                description="Burst read transaction transferring packet payload",
            ),
        ],
        fsms=[
            EFSFSM(
                fsm_id="FSM_DMA_CTRL",
                name="DMA_Control_FSM",
                initial_state="IDLE",
                encoding="onehot",
                states=[
                    EFSState(state_id="ST_IDLE", name="IDLE"),
                    EFSState(state_id="ST_FETCH", name="FETCH"),
                    EFSState(state_id="ST_TRANSFER", name="TRANSFER"),
                    EFSState(state_id="ST_DONE", name="DONE"),
                ],
                transitions=[
                    EFSTransition(transition_id="TR_1", source_state="IDLE", target_state="FETCH", condition="start_pulse == 1"),
                    EFSTransition(transition_id="TR_2", source_state="FETCH", target_state="TRANSFER", condition="descriptor_valid == 1"),
                    EFSTransition(transition_id="TR_3", source_state="TRANSFER", target_state="DONE", condition="byte_count == 0"),
                    EFSTransition(transition_id="TR_4", source_state="DONE", target_state="IDLE", condition="clear_status == 1"),
                ],
            ),
        ],
        constraints=[
            EFSConstraint(
                constraint_id="CONST_HANDSHAKE",
                type="handshake",
                source_objects=["s_axi_awvalid"],
                target_objects=["s_axi_awready"],
                condition="s_axi_awvalid == 1",
                expected_behavior="s_axi_awready must assert within 16 cycles",
                severity="CRITICAL",
            ),
        ],
        timing_rules=[
            EFSTimingRule(
                rule_id="TIME_LATENCY_01",
                type="latency",
                minimum_delay="1 cycle",
                maximum_delay="4 cycles",
                cycle_relationship="m_axis_tvalid within 2 cycles of descriptor_valid",
            ),
        ],
        flows=[
            EFSFlow(
                flow_id="FLOW_TRANSFER",
                name="DMA_Block_Transfer_Flow",
                components=["COMP_CTRL", "COMP_FIFO"],
                interfaces=["IF_STREAM"],
                conditions="START == 1 and FIFO_EMPTY == 1",
            ),
        ],
    )


def test_neo4j_hardware_graph_node_types_and_counts(sample_complete_efs_ir):
    """Verify that all hardware constructs exist as nodes in the graph store."""
    entities = [
        Entity(entity_id="E1", name="DMA Subsystem Controller", raw_name="DMA Subsystem Controller", entity_type="Design", chunk_id="C1", doc_id="spec_dma_v1"),
        Entity(entity_id="E2", name="DMA_Main_Controller", raw_name="DMA Controller", entity_type="Component", chunk_id="C1", doc_id="spec_dma_v1"),
    ]
    relationships = []

    stats = build_graph(entities, relationships, efs_ir=sample_complete_efs_ir)
    store = get_graph_store()

    all_nodes = store.all_nodes()
    node_names = {n["name"] for n in all_nodes}
    node_types = {n.get("type") for n in all_nodes}

    # Verify key node types
    assert "Design" in node_types
    assert "Component" in node_types
    assert "Interface" in node_types
    assert "Signal" in node_types
    assert "Register" in node_types
    assert "Field" in node_types
    assert "Transaction" in node_types
    assert "FSM" in node_types
    assert "State" in node_types
    assert "Constraint" in node_types
    assert "TimingConstraint" in node_types
    assert "Flow" in node_types

    # Verify specific entity names
    assert "DMA Subsystem Controller" in node_names
    assert "DMA_Main_Controller" in node_names
    assert "DMA_Register_File" in node_names
    assert "Data_FIFO" in node_names
    assert "AXI_Config_Slave" in node_names
    assert "AXIS_Data_Master" in node_names
    assert "aclk" in node_names
    assert "aresetn" in node_names
    assert "s_axi_awaddr" in node_names
    assert "m_axis_tdata" in node_names
    assert "DMA_CONTROL" in node_names
    assert "DMA_STATUS" in node_names
    assert "DMA_Burst_Read_Tx" in node_names
    assert "DMA_Control_FSM" in node_names
    assert "DMA_Block_Transfer_Flow" in node_names


def test_neo4j_hardware_graph_semantic_relationships(sample_complete_efs_ir):
    """Verify that all hardware relationships connect canonical nodes with correct relations."""
    build_graph([], [], efs_ir=sample_complete_efs_ir)
    store = get_graph_store()

    all_edges = store.all_edges()
    edge_triples = {(e["source"], e["relation"], e["target"]) for e in all_edges}

    # 1. Design contains components
    assert ("DMA Subsystem Controller", "CONTAINS_COMPONENT", "DMA_Main_Controller") in edge_triples
    assert ("DMA Subsystem Controller", "CONTAINS_COMPONENT", "DMA_Register_File") in edge_triples
    assert ("DMA Subsystem Controller", "CONTAINS_COMPONENT", "Data_FIFO") in edge_triples

    # 2. Components have interfaces
    assert ("DMA_Main_Controller", "HAS_INTERFACE", "AXI_Config_Slave") in edge_triples
    assert ("DMA_Main_Controller", "HAS_INTERFACE", "AXIS_Data_Master") in edge_triples

    # 3. Interface exposures / connections
    assert ("DMA_Main_Controller", "EXPOSES_INTERFACE", "AXIS_Data_Master") in edge_triples
    assert ("AXIS_Data_Master", "CONNECTS_TO", "Data_FIFO") in edge_triples

    # 4. Interface has signals
    assert ("AXI_Config_Slave", "HAS_SIGNAL", "s_axi_awaddr") in edge_triples
    assert ("AXI_Config_Slave", "HAS_SIGNAL", "s_axi_awvalid") in edge_triples
    assert ("AXIS_Data_Master", "HAS_SIGNAL", "m_axis_tdata") in edge_triples

    # 5. Clock and Reset wiring
    assert ("aclk", "CLOCKS", "DMA_Main_Controller") in edge_triples
    assert ("aresetn", "RESETS", "DMA_Main_Controller") in edge_triples

    # 6. Drivers and Consumers
    assert ("DMA_Main_Controller", "DRIVES", "s_axi_awready") in edge_triples
    assert ("DMA_Main_Controller", "DRIVES", "m_axis_tdata") in edge_triples
    assert ("m_axis_tdata", "FEEDS", "Data_FIFO") in edge_triples

    # 7. Registers and Fields
    assert ("DMA_Register_File", "HAS_REGISTER", "DMA_CONTROL") in edge_triples
    assert ("DMA_Register_File", "HAS_REGISTER", "DMA_STATUS") in edge_triples
    assert any(src == "DMA_CONTROL" and rel == "HAS_FIELD" for src, rel, tgt in edge_triples)

    # 8. Transactions
    assert ("DMA_Main_Controller", "INITIATES", "DMA_Burst_Read_Tx") in edge_triples
    assert ("DMA_Burst_Read_Tx", "TARGETS", "Data_FIFO") in edge_triples
    assert ("DMA_Burst_Read_Tx", "INVOLVES_SIGNAL", "m_axis_tdata") in edge_triples
    assert ("DMA_Burst_Read_Tx", "USES_INTERFACE", "AXIS_Data_Master") in edge_triples

    # 9. FSM and States
    assert ("DMA_Control_FSM", "HAS_STATE", "DMA_Control_FSM_IDLE") in edge_triples
    assert ("DMA_Control_FSM_IDLE", "TRANSITIONS_TO", "DMA_Control_FSM_FETCH") in edge_triples

    # 10. Constraints and Timing
    assert any(rel == "CONSTRAINS" and tgt == "s_axi_awvalid" for src, rel, tgt in edge_triples)
    assert any(src == "DMA Subsystem Controller" and rel == "HAS_TIMING_RULE" for src, rel, tgt in edge_triples)

    # 11. Flow
    assert ("DMA_Block_Transfer_Flow", "INVOLVES_COMPONENT", "DMA_Main_Controller") in edge_triples
    assert ("DMA_Block_Transfer_Flow", "INVOLVES_INTERFACE", "AXIS_Data_Master") in edge_triples


def test_ui_graph_view_integrity(sample_complete_efs_ir):
    """Verify that every edge in the graph connects two nodes that exist in all_nodes (no orphan edges)."""
    build_graph([], [], efs_ir=sample_complete_efs_ir)
    store = get_graph_store()

    all_nodes = store.all_nodes()
    all_edges = store.all_edges()

    node_names = {n["name"] for n in all_nodes}

    # Verify that ui/graph_view filtering `e["source"] in node_names and e["target"] in node_names`
    # keeps 100% of the canonical hardware edges
    dropped_edges = [
        e for e in all_edges
        if e["source"] not in node_names or e["target"] not in node_names
    ]

    assert len(dropped_edges) == 0, f"Found {len(dropped_edges)} orphan edges with unmatched node names: {dropped_edges}"
