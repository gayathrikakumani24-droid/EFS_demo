import sys
import os
import json

sys.path.insert(0, os.path.abspath("hardware_graphrag"))

from core.efs_ir.models import EFSIR, EFSMetadata, EFSComponent, EFSSignal, EFSRegister
from core.efs_ir.builder import EFSIRBuilder
from core.efs_ir.repair_engine import EFSStructuralRepairEngine
from core.verification.conflict_resolution_engine import ConflictResolutionEngine
from core.agents.rtl_agent import plan_rtl, validate_rtl_plan, generate_code

def main():
    req_model = {
        "design_name": "traffic_light_controller",
        "timing_requirements": [
            "The RED light remains ON for 5 clock cycles.",
            "The GREEN state remains ON for 10 clock cycles.",
            "The YELLOW state remains ON for 3 clock cycles."
        ],
        "fsm_info": {
            "name": "traffic_light_fsm",
            "states": [
                {"name": "RED", "duration_cycles": 5},
                {"name": "GREEN", "duration_cycles": 10},
                {"name": "YELLOW", "duration_cycles": 3}
            ],
            "initial_state": "RED",
            "transitions": [
                {"from": "RED", "to": "GREEN"},
                {"from": "GREEN", "to": "YELLOW"},
                {"from": "YELLOW", "to": "RED"}
            ]
        }
    }

    builder = EFSIRBuilder(req_model=req_model)
    built_ir = builder.build()
    built_ir.signals = [
        EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
        EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset"),
        EFSSignal(name="red_light", direction="output", width="1"),
        EFSSignal(name="green_light", direction="output", width="1"),
        EFSSignal(name="yellow_light", direction="output", width="1"),
    ]

    repaired_ir, log, issues = EFSStructuralRepairEngine.repair_and_validate(built_ir, req_model=req_model)

    print("Repaired FSM transitions:")
    for tr in repaired_ir.fsms[0].transitions:
        print(f"  {tr.source_state} -> {tr.target_state}: {tr.condition} (type={tr.condition_type}, assoc_reg={tr.associated_register})")

    conflicts = ConflictResolutionEngine.detect_conflicts(repaired_ir)
    tc_conflicts = [c for c in conflicts if c.property == "transition_condition"]
    print(f"Detected transition_condition conflicts: {len(tc_conflicts)}")

    # Plan RTL
    plan = plan_rtl(repaired_ir)
    val_report = validate_rtl_plan(plan, repaired_ir)
    print("Plan issues:", val_report.get("issues", []))
    print("Plan FSMs:", json.dumps(plan.get("fsms", plan.get("fsm", {})), indent=2))
    print("Internal registers in plan:", plan.get("internal_registers", []))

    # Generate RTL
    code = generate_code("traffic light controller design", "Verilog", efs_ir=repaired_ir)
    print("\n--- GENERATED RTL ---")
    print(code)

if __name__ == "__main__":
    main()
