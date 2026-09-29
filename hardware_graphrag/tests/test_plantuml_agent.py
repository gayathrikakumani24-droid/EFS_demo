"""Tests for PlantUML Agent diagram generation, alias sanitization, and syntax validation."""

import re
import pytest
from core.efs_ir.models import (
    EFSIR, EFSSignal, EFSComponent, EFSInterface, EFSTransition, EFSFSM, EFSMetadata
)
from core.agents.plantuml_agent import (
    resolve_diagram_ids_to_names,
    get_heuristic_diagrams,
    generate_diagrams,
    repair_plantuml_syntax
)
from core.agents.plantuml_utils import (
    to_plantuml_alias,
    PlantUMLAliasRegistry,
    resolve_canonical_design_name,
    validate_plantuml_syntax,
    validate_interaction,
    verify_plantuml_render
)


# --------------------------------------------------------------------------
# 1. Alias Sanitizer Tests
# --------------------------------------------------------------------------

def test_alias_sanitizer_arbitrary_names():
    """Verify alias sanitizer produces valid PlantUML identifiers from diverse names."""
    test_cases = [
        ("AXI4 Slave Interface", "AXI4_Slave_Interface"),
        ("Matrix Processing Elements (MPEs)", "Matrix_Processing_Elements_MPEs"),
        ("64-bit FPU", "_64_bit_FPU"),
        ("Work-Stealing Scheduler", "Work_Stealing_Scheduler"),
        ("foo/bar", "foo_bar"),
        ("component-A", "component_A"),
        ("component A", "component_A"),
        ("   DMA  Controller  ", "DMA_Controller"),
        ("128_bit_ALU", "_128_bit_ALU"),
        ("PCIe-Gen4x8", "PCIe_Gen4x8"),
        ("component@bus#0", "component_bus_0"),
        ("state", "elem_state"),  # Reserved word handling
        ("interface", "elem_interface"),  # Reserved word handling
    ]

    identifier_regex = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    for raw_name, expected in test_cases:
        alias = to_plantuml_alias(raw_name)
        assert alias == expected, f"Failed for '{raw_name}': got '{alias}', expected '{expected}'"
        assert identifier_regex.match(alias), f"Alias '{alias}' is not a valid identifier for '{raw_name}'"
        assert " " not in alias, f"Alias '{alias}' contains spaces!"


def test_alias_registry_collision_resolution():
    """Verify registry resolves collisions deterministically without mutating display names."""
    registry = PlantUMLAliasRegistry()

    # Two distinct names that sanitize to 'A_B'
    alias1 = registry.register("A-B")
    alias2 = registry.register("A B")
    alias3 = registry.register("A_B")

    assert alias1 == "A_B"
    assert alias2 == "A_B_2"
    assert alias3 == "A_B_3"

    # Display names are preserved
    assert registry.get_display_name(alias1) == "A-B"
    assert registry.get_display_name(alias2) == "A B"
    assert registry.get_display_name(alias3) == "A_B"

    # Idempotence: re-registering returns existing alias
    assert registry.register("A-B") == "A_B"
    assert registry.register("A B") == "A_B_2"


def test_alias_registry_with_canonical_id():
    """Verify registry utilizes canonical IDs for stable disambiguation."""
    registry = PlantUMLAliasRegistry()

    alias1 = registry.register("Port", canonical_id="IF_in_01")
    alias2 = registry.register("Port", canonical_id="IF_out_02")

    assert alias1 != alias2
    assert "Port" in alias1 and "Port" in alias2
    assert registry.get_alias("IF_in_01") == alias1
    assert registry.get_alias("IF_out_02") == alias2


# --------------------------------------------------------------------------
# 2. Synthetic Hardware Specification Tests
# --------------------------------------------------------------------------

def test_synthetic_hardware_entities():
    """Test arbitrary synthetic hardware entity names compile to valid PUML."""
    synthetic_entities = [
        ("DMA Controller", "controller"),
        ("Memory/Cache Bridge", "bridge"),
        ("Core-0", "processor"),
        ("L2 Shared Cache", "memory"),
        ("Control & Status Register Block", "register_block")
    ]

    registry = PlantUMLAliasRegistry()
    aliases = {}
    for name, etype in synthetic_entities:
        alias = registry.register(name, entity_type=etype)
        aliases[name] = alias
        assert re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", alias)
        assert " " not in alias

    # All aliases must be unique
    assert len(set(aliases.values())) == len(synthetic_entities)


# --------------------------------------------------------------------------
# 3. Architecture PUML Generation & Evidence-Based Grounding
# --------------------------------------------------------------------------

def test_architecture_generation_valid_aliases_and_types():
    """Verify Architecture PUML separates display name from alias and uses correct semantic types."""
    comps = [
        EFSComponent(component_id="COMP_01", name="AXI4 Slave Interface", type="interface"),
        EFSComponent(component_id="COMP_02", name="L2 Shared Cache", type="memory"),
        EFSComponent(component_id="COMP_03", name="Work-Stealing Scheduler", type="controller"),
    ]
    iface = EFSInterface(
        interface_id="IF_01",
        name="AXI4_Bus",
        source_component="COMP_01",
        destination_component="COMP_03"
    )
    efs_ir = EFSIR(components=comps, interfaces=[iface])
    plan = {"target": "Neural_Processing_Core"}

    diagrams = get_heuristic_diagrams(plan, efs_ir=efs_ir)
    arch = diagrams["architecture"]

    # Must contain valid declarations with quotes and safe aliases
    assert 'interface "AXI4 Slave Interface" as AXI4_Slave_Interface' in arch
    assert 'database "L2 Shared Cache" as L2_Shared_Cache' in arch
    assert 'component "Work-Stealing Scheduler" as Work_Stealing_Scheduler' in arch

    # Must NEVER contain spaces in 'as ...' on the same line
    assert not re.search(r"as[ \t]+[A-Za-z0-9_]+[ \t]+[A-Za-z0-9_]+", arch)

    # Relationships must connect safe aliases
    assert "AXI4_Slave_Interface ..> AXI4_Bus" in arch
    assert "AXI4_Bus ..> Work_Stealing_Scheduler" in arch

    # Local syntax validation must pass
    val = validate_plantuml_syntax(arch, diagram_type="architecture")
    assert val.is_valid, f"Architecture validation failed: {val.errors}"


def test_architecture_no_hallucinated_connections():
    """Verify when connections are empty, no fake relationship arrows are created."""
    comps = [
        EFSComponent(component_id="COMP_01", name="Module Alpha", type="component"),
        EFSComponent(component_id="COMP_02", name="Module Beta", type="component"),
    ]
    efs_ir = EFSIR(components=comps, interfaces=[])  # No interfaces / connections
    plan = {"target": "Independent_Subsystem"}

    diagrams = get_heuristic_diagrams(plan, efs_ir=efs_ir)
    arch = diagrams["architecture"]

    # Confirm components are present
    assert 'component "Module Alpha" as Module_Alpha' in arch
    assert 'component "Module Beta" as Module_Beta' in arch

    # Must NOT hallucinate arrows between Alpha and Beta
    assert "Module_Alpha --> Module_Beta" not in arch
    assert "-->" not in arch and ".." not in arch or "note" in arch
    assert "Explicit inter-module topology not specified" in arch


# --------------------------------------------------------------------------
# 4. Sequence PUML Generation & Interaction Validation
# --------------------------------------------------------------------------

def test_sequence_generation_valid_aliases():
    """Verify Sequence PUML uses safe participant aliases and safe message arrows."""
    comps = [
        EFSComponent(component_id="COMP_01", name="DMA Controller", type="controller"),
        EFSComponent(component_id="COMP_02", name="Memory/Cache Bridge", type="bridge"),
    ]
    sigs = [
        EFSSignal(signal_id="SIG_01", name="dma_req", direction="input"),
        EFSSignal(signal_id="SIG_02", name="dma_ack", direction="output"),
    ]
    efs_ir = EFSIR(components=comps, signals=sigs)
    plan = {"target": "DMA_Subsystem"}

    diagrams = get_heuristic_diagrams(plan, efs_ir=efs_ir)
    seq = diagrams["sequence"]

    # Verify participant declarations
    assert 'participant "DMA Controller" as DMA_Controller' in seq
    assert 'participant "Memory/Cache Bridge" as Memory_Cache_Bridge' in seq

    # Verify message arrows use safe aliases
    assert "Host -> DMA_Controller : Assert dma_req" in seq
    assert "DMA_Controller -> Memory_Cache_Bridge : Transfer dma_ack" in seq

    # No spaces in aliases on arrows
    assert "DMA Controller ->" not in seq
    assert "-> Memory/Cache Bridge" not in seq

    # Local syntax validation must pass
    val = validate_plantuml_syntax(seq, diagram_type="sequence")
    assert val.is_valid, f"Sequence validation failed: {val.errors}"


def test_validate_interaction_rejects_fragments():
    """Verify linguistic fragment actions are rejected generically."""
    valid_participants = {"Master", "Slave", "Host", "Controller"}

    fragment_actions = [
        "AXI_TRANSACTIONS",
        "AS_TRANSACTION",
        "DESCRIBES_TRANSACTION",
        "ORDER_TRANSACTION",
        "OF_TRANSACTION",
        "WRITE_TRANSACTION",
        "THE_TRANSACTION",
        "transaction",
        "for the",
    ]

    for bad_act in fragment_actions:
        interaction = {"from": "Master", "to": "Slave", "action": bad_act}
        is_valid, reason = validate_interaction(interaction, valid_participants)
        assert not is_valid, f"Action '{bad_act}' should have been rejected as a fragment, but passed!"

    # Legitimate actions must pass
    valid_actions = [
        "Write Address Valid (AWVALID)",
        "Read Data Transfer",
        "Trigger Computation",
        "Assert Interrupt",
        "Fetch Instruction Word"
    ]
    for good_act in valid_actions:
        interaction = {"from": "Master", "to": "Slave", "action": good_act}
        is_valid, reason = validate_interaction(interaction, valid_participants)
        assert is_valid, f"Action '{good_act}' should be valid, but failed: {reason}"


# --------------------------------------------------------------------------
# 5. FSM Context Grounding
# --------------------------------------------------------------------------

def test_fsm_insufficient_context_reports_notice():
    """Verify when no FSM states/transitions exist, the generator reports insufficient context rather than hallucinating."""
    efs_ir = EFSIR(components=[EFSComponent(component_id="COMP_01", name="Compute Core")])
    plan = {"target": "Compute_Core"}

    diagrams = get_heuristic_diagrams(plan, efs_ir=efs_ir)
    fsm = diagrams["fsm"]

    assert "INSUFFICIENT_DIAGRAM_CONTEXT" in fsm
    assert "trigger" not in fsm
    assert "reset / return" not in fsm


def test_fsm_with_transitions_renders_cleanly():
    """Verify when real FSM transitions exist, they render with valid syntax."""
    fsm_obj = EFSFSM(
        fsm_id="FSM_01",
        name="State Controller",
        initial_state="IDLE",
        transitions=[
            EFSTransition(source_state="IDLE", target_state="ACTIVE", condition="start_pulse"),
            EFSTransition(source_state="ACTIVE", target_state="DONE", condition="count_max"),
            EFSTransition(source_state="DONE", target_state="IDLE", condition="ack"),
        ]
    )
    efs_ir = EFSIR(fsms=[fsm_obj])
    plan = {"target": "FSM_Device"}

    diagrams = get_heuristic_diagrams(plan, efs_ir=efs_ir)
    fsm = diagrams["fsm"]

    assert "[*] --> IDLE" in fsm
    assert "IDLE --> ACTIVE : start_pulse" in fsm
    assert "ACTIVE --> DONE : count_max" in fsm
    assert "DONE --> IDLE : ack" in fsm

    val = validate_plantuml_syntax(fsm, diagram_type="fsm")
    assert val.is_valid, f"FSM validation failed: {val.errors}"


# --------------------------------------------------------------------------
# 6. Target Module Resolution Tests
# --------------------------------------------------------------------------

def test_resolve_canonical_design_name():
    """Verify target_module is never selected when a valid design name is available."""
    # From plan target
    name1 = resolve_canonical_design_name(design_plan={"target": "Custom_Systolic_Array"})
    assert name1 == "Custom_Systolic_Array"

    # From EFS IR metadata
    ir2 = EFSIR(metadata=EFSMetadata(display_name="Tensor Engine", design_name="tensor_engine"))
    name2 = resolve_canonical_design_name(efs_ir=ir2)
    assert name2 == "Tensor Engine"

    # From EFS IR components when metadata is generic
    ir3 = EFSIR(
        metadata=EFSMetadata(display_name="target_module"),
        components=[EFSComponent(component_id="COMP_01", name="Neural Processor", type="top_level_design")]
    )
    name3 = resolve_canonical_design_name(efs_ir=ir3)
    assert name3 == "Neural Processor"

    # From spec header
    spec = "# Hardware Design Specification: Cryptographic Co-Processor\nOverview..."
    name4 = resolve_canonical_design_name(design_spec=spec)
    assert name4 == "Cryptographic Co-Processor"


# --------------------------------------------------------------------------
# 7. Syntax Validation and Repair Layer Tests
# --------------------------------------------------------------------------

def test_syntax_validator_detects_broken_syntax():
    """Verify syntax validator flags spaces in aliases and unbalanced tags."""
    broken_puml = """@startuml
participant "AXI4 Slave Interface" as AXI4 Slave Interface
[target_module] as target_module <<top_level_design>>
[L2 Cache] as L2 Cache <<memory>>
Host -> AXI4 Slave Interface : request
@enduml"""

    val = validate_plantuml_syntax(broken_puml)
    assert not val.is_valid
    assert any("Invalid participant alias" in e or "Invalid alias" in e for e in val.errors)


def test_syntax_repair_layer():
    """Verify repair layer sanitizes spaces in aliases and propagates to arrows."""
    broken_puml = """@startuml
participant "AXI4 Slave Interface" as AXI4 Slave Interface
participant "DMA Unit" as DMA Unit
AXI4 Slave Interface -> DMA Unit : start
@enduml"""

    repaired = repair_plantuml_syntax(broken_puml)
    assert 'participant "AXI4 Slave Interface" as AXI4_Slave_Interface' in repaired
    assert 'participant "DMA Unit" as DMA_Unit' in repaired
    assert "AXI4_Slave_Interface -> DMA_Unit : start" in repaired

    val = validate_plantuml_syntax(repaired, diagram_type="sequence")
    assert val.is_valid, f"Repaired PUML failed validation: {val.errors}"


# --------------------------------------------------------------------------
# 8. Legacy Signal ID Replacement Test (Preserve existing functionality)
# --------------------------------------------------------------------------

def test_resolve_diagram_ids_to_names_with_efs_ir():
    sig1 = EFSSignal(signal_id="SIG_53f10a8d", name="inst_req", direction="input")
    sig2 = EFSSignal(signal_id="SIG_cc34814f", name="inst_ack", direction="output")
    sig3 = EFSSignal(signal_id="SIG_fa092020", name="bus_addr", direction="input")
    sig4 = EFSSignal(signal_id="SIG_a750d045", name="bus_data", direction="output")

    comp1 = EFSComponent(component_id="COMP_001", name="Instruction_Decoder")
    comp2 = EFSComponent(component_id="COMP_002", name="External_Bus_Interface")

    efs_ir = EFSIR(
        signals=[sig1, sig2, sig3, sig4],
        components=[comp1, comp2]
    )

    sample_sequence_puml = """@startuml
participant "Instruction Decoder" as COMP_001
participant "External Bus Interface" as COMP_002
COMP_001 -> COMP_002 : SIG_53f10a8d
COMP_002 --> COMP_001 : SIG_cc34814f
COMP_001 -> COMP_002 : SIG_fa092020
COMP_002 --> COMP_001 : SIG_a750d045
@enduml"""

    diagrams = {"sequence": sample_sequence_puml}
    resolved = resolve_diagram_ids_to_names(diagrams, efs_ir=efs_ir)

    res_seq = resolved["sequence"]
    assert "SIG_53f10a8d" not in res_seq
    assert "SIG_cc34814f" not in res_seq
    assert "SIG_fa092020" not in res_seq
    assert "SIG_a750d045" not in res_seq
    assert "COMP_001" not in res_seq
    assert "COMP_002" not in res_seq
    assert "inst_req" in res_seq
    assert "inst_ack" in res_seq
    assert "bus_addr" in res_seq
    assert "bus_data" in res_seq
    assert "Instruction_Decoder" in res_seq
    assert "External_Bus_Interface" in res_seq


# --------------------------------------------------------------------------
# 9. Actual PlantUML Server Rendering Verification (Online / Network Verification)
# --------------------------------------------------------------------------

def test_actual_plantuml_server_rendering():
    """Verify that generated Architecture and Sequence PUML render without syntax errors."""
    comps = [
        EFSComponent(component_id="COMP_01", name="AXI4 Slave Interface", type="interface"),
        EFSComponent(component_id="COMP_02", name="Matrix Processing Elements (MPEs)", type="component"),
        EFSComponent(component_id="COMP_03", name="64-bit FPU", type="component"),
    ]
    sigs = [
        EFSSignal(signal_id="SIG_01", name="matrix_req_valid", direction="input"),
        EFSSignal(signal_id="SIG_02", name="matrix_done", direction="output"),
    ]
    iface = EFSInterface(
        interface_id="IF_01",
        name="AXI4_Bus",
        source_component="COMP_01",
        destination_component="COMP_02"
    )
    efs_ir = EFSIR(components=comps, signals=sigs, interfaces=[iface])
    plan = {"target": "High_Performance_Accelerator"}

    diagrams = get_heuristic_diagrams(plan, efs_ir=efs_ir)

    # Render Architecture diagram
    arch_ok, arch_msg = verify_plantuml_render(diagrams["architecture"])
    assert arch_ok, f"Architecture PUML rendering failed: {arch_msg}\nSource:\n{diagrams['architecture']}"

    # Render Sequence diagram
    seq_ok, seq_msg = verify_plantuml_render(diagrams["sequence"])
    assert seq_ok, f"Sequence PUML rendering failed: {seq_msg}\nSource:\n{diagrams['sequence']}"
