import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import urllib.request
from core.agents.plantuml_agent import generate_diagrams, get_heuristic_diagrams, get_diagram_url
from core.agents.plantuml_utils import validate_plantuml_syntax, verify_plantuml_render
from core.efs_ir.models import EFSIR, EFSComponent, EFSInterface, EFSSignal, EFSFSM, EFSTransition

def run_verification():
    print("=== STARTING PLANTUML PIPELINE VERIFICATION ===")

    # Test 1: Synthetic High-Performance Neural Accelerator
    comps = [
        EFSComponent(component_id="COMP_01", name="AXI4 Slave Interface", type="interface"),
        EFSComponent(component_id="COMP_02", name="Matrix Processing Elements (MPEs)", type="component"),
        EFSComponent(component_id="COMP_03", name="64-bit FPU", type="component"),
        EFSComponent(component_id="COMP_04", name="L2 Shared Cache", type="memory"),
        EFSComponent(component_id="COMP_05", name="Work-Stealing Scheduler", type="controller"),
    ]
    sigs = [
        EFSSignal(signal_id="SIG_01", name="matrix_req_valid", direction="input"),
        EFSSignal(signal_id="SIG_02", name="matrix_done", direction="output"),
    ]
    ifaces = [
        EFSInterface(
            interface_id="IF_01",
            name="AXI4_Bus",
            source_component="COMP_01",
            destination_component="COMP_05"
        ),
        EFSInterface(
            interface_id="IF_02",
            name="Cache_Bus",
            source_component="COMP_05",
            destination_component="COMP_04"
        )
    ]
    efs_ir = EFSIR(components=comps, signals=sigs, interfaces=ifaces)
    plan = {"target": "High_Performance_Accelerator"}

    diagrams = get_heuristic_diagrams(plan, efs_ir=efs_ir)

    print("\n--- 1. ARCHITECTURE PUML ---")
    print(diagrams["architecture"])
    val_arch = validate_plantuml_syntax(diagrams["architecture"], diagram_type="architecture")
    print("Arch Valid:", val_arch.is_valid, "Errors:", val_arch.errors)
    arch_ok, arch_msg = verify_plantuml_render(diagrams["architecture"])
    print("Arch Render:", arch_ok, arch_msg)
    assert val_arch.is_valid and arch_ok

    print("\n--- 2. SEQUENCE PUML ---")
    print(diagrams["sequence"])
    val_seq = validate_plantuml_syntax(diagrams["sequence"], diagram_type="sequence")
    print("Seq Valid:", val_seq.is_valid, "Errors:", val_seq.errors)
    seq_ok, seq_msg = verify_plantuml_render(diagrams["sequence"])
    print("Seq Render:", seq_ok, seq_msg)
    assert val_seq.is_valid and seq_ok

    print("\n--- 3. FSM PUML (Evidence-driven insufficient context) ---")
    print(diagrams["fsm"])
    val_fsm = validate_plantuml_syntax(diagrams["fsm"], diagram_type="fsm")
    print("FSM Valid:", val_fsm.is_valid, "Errors:", val_fsm.errors)
    fsm_ok, fsm_msg = verify_plantuml_render(diagrams["fsm"])
    print("FSM Render:", fsm_ok, fsm_msg)
    assert val_fsm.is_valid and fsm_ok
    assert "INSUFFICIENT_DIAGRAM_CONTEXT" in diagrams["fsm"]

    # Confirm no target_module
    for k, puml in diagrams.items():
        assert "target_module" not in puml.lower(), f"Placeholder 'target_module' found in {k} diagram!"

    print("\n=== ALL E2E DIAGRAM GENERATION & RENDERING CHECKS PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_verification()
