"""Tests for PlantUML Agent diagram post-processing and signal ID resolution."""

import pytest
from core.efs_ir.models import EFSIR, EFSSignal, EFSComponent, EFSInterface
from core.agents.plantuml_agent import resolve_diagram_ids_to_names


def test_resolve_diagram_ids_to_names_with_efs_ir():
    # Setup EFS IR with signals and components
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
participant Instruction_Decoder
participant external_bus_interface
Instruction_Decoder -> external_bus_interface : SIG_53f10a8d
external_bus_interface --> Instruction_Decoder : SIG_cc34814f
Instruction_Decoder -> external_bus_interface : SIG_fa092020
external_bus_interface --> Instruction_Decoder : SIG_a750d045
@enduml"""

    diagrams = {"sequence": sample_sequence_puml}
    resolved = resolve_diagram_ids_to_names(diagrams, efs_ir=efs_ir)

    res_seq = resolved["sequence"]
    assert "SIG_53f10a8d" not in res_seq
    assert "SIG_cc34814f" not in res_seq
    assert "SIG_fa092020" not in res_seq
    assert "SIG_a750d045" not in res_seq
    assert "inst_req" in res_seq
    assert "inst_ack" in res_seq
    assert "bus_addr" in res_seq
    assert "bus_data" in res_seq


def test_resolve_diagram_ids_unmapped_fallback():
    sample_sequence_puml = """@startuml
Instruction_Decoder -> external_bus_interface : SIG_99999999
@enduml"""

    diagrams = {"sequence": sample_sequence_puml}
    resolved = resolve_diagram_ids_to_names(diagrams, efs_ir=None)

    res_seq = resolved["sequence"]
    assert "SIG_99999999" not in res_seq
    assert "sig_transfer_1" in res_seq
