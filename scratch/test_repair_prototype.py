import sys
import os
import json
import re
import docx

sys.path.insert(0, os.path.abspath("hardware_graphrag"))

from core.efs_ir.models import (
    EFSIR, EFSComponent, EFSInterface, EFSSignal, EFSRegister,
    EFSFSM, EFSState, EFSTransition, EFSConstraint, EFSTraceability,
    EFSConflict, EFSResolution, new_efs_id
)
from core.efs_ir.validator import validate_efs_ir
from core.verification.conflict_resolution_engine import ConflictResolutionEngine

docx_path = os.path.join("hardware_graphrag", "Traffic_controller_spec.docx")
doc = docx.Document(docx_path)
spec_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

print("Specification Text loaded:")
for line in spec_text.splitlines():
    if line.strip():
        print("  ", line)

# Build an initial IR similar to what EFSIRBuilder produces before repair
initial_ir = EFSIR(
    components=[EFSComponent(component_id="COMP_01", name="traffic_light_controller", type="module")],
    signals=[
        EFSSignal(signal_id="SIG_01", name="clk", direction="input", width="1", semantic_role="clock"),
        EFSSignal(signal_id="SIG_02", name="reset_n", direction="input", width="1", semantic_role="reset"),
        EFSSignal(signal_id="SIG_03", name="red_light", direction="output", width="1"),
        EFSSignal(signal_id="SIG_04", name="green_light", direction="output", width="1"),
        EFSSignal(signal_id="SIG_05", name="yellow_light", direction="output", width="1"),
    ],
    fsms=[
        EFSFSM(
            fsm_id="FSM_01",
            name="traffic_light_controller_fsm",
            initial_state="RED",
            states=[
                EFSState(state_id="ST_01", name="RED", outputs={"red_light": "1'b1", "green_light": "1'b0", "yellow_light": "1'b0"}),
                EFSState(state_id="ST_02", name="GREEN", outputs={"red_light": "1'b0", "green_light": "1'b1", "yellow_light": "1'b0"}),
                EFSState(state_id="ST_03", name="YELLOW", outputs={"red_light": "1'b0", "green_light": "1'b0", "yellow_light": "1'b1"}),
            ],
            transitions=[
                EFSTransition(transition_id="TR_01", source_state="RED", target_state="GREEN", condition="1'b1", condition_type="semantic"),
                EFSTransition(transition_id="TR_02", source_state="GREEN", target_state="YELLOW", condition="1'b1", condition_type="semantic"),
                EFSTransition(transition_id="TR_03", source_state="YELLOW", target_state="RED", condition="1'b1", condition_type="semantic"),
            ]
        )
    ]
)

print("\n--- BEFORE REPAIR ---")
val_issues_before = validate_efs_ir(initial_ir)
conflicts_before = ConflictResolutionEngine.detect_conflicts(initial_ir, {}, validation_issues=val_issues_before)
print(f"Conflicts before repair ({len(conflicts_before)}):")
for c in conflicts_before:
    print(f"  [{c.severity}] {c.conflict_id} | prop={c.property} | title={c.title} | vals={c.conflicting_values}")

print("Transitions before repair:")
for tr in initial_ir.fsms[0].transitions:
    print(f"  {tr.source_state} -> {tr.target_state} | cond: '{tr.condition}' | type: {tr.condition_type}")
