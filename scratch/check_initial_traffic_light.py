import sys
import os
import json

sys.path.insert(0, os.path.abspath('.'))
from core.efs_ir.models import (
    EFSIR, EFSMetadata, EFSComponent, EFSSignal,
    EFSFSM, EFSState, EFSTransition, EFSTimingRule, EFSConflict, EFSTraceability
)
from core.verification.conflict_resolution_engine import ConflictResolutionEngine

ir = EFSIR(
    metadata=EFSMetadata(
        design_id="DSN_traffic_light",
        design_name="traffic_light_controller",
        canonical_id="EFS_traffic_light",
        display_name="Traffic Light Controller",
        top_module="traffic_light_controller"
    ),
    components=[
        EFSComponent(name="traffic_light_controller", type="module")
    ],
    signals=[
        EFSSignal(signal_id="SIG_clk", name="clk", direction="input", width="1", semantic_role="clock"),
        EFSSignal(signal_id="SIG_rst_n", name="reset_n", direction="input", width="1", semantic_role="reset"),
        EFSSignal(signal_id="SIG_red", name="red_light", direction="output", width="1"),
        EFSSignal(signal_id="SIG_green", name="green_light", direction="output", width="1"),
        EFSSignal(signal_id="SIG_yellow", name="yellow_light", direction="output", width="1")
    ],
    clocks=[EFSSignal(signal_id="SIG_clk", name="clk", direction="input", width="1", semantic_role="clock")],
    resets=[EFSSignal(signal_id="SIG_rst_n", name="reset_n", direction="input", width="1", semantic_role="reset")],
    fsms=[
        EFSFSM(
            fsm_id="FSM_traffic_light",
            name="traffic_light_fsm",
            initial_state="RED",
            reset_state="RED",
            reset_behavior="Asynchronous active-low reset to RED on reset_n",
            states=[
                EFSState(state_id="ST_red", name="RED", outputs={"red_light": "1'b1", "green_light": "1'b0", "yellow_light": "1'b0"}, actions=["red_light = 1'b1"]),
                EFSState(state_id="ST_green", name="GREEN", outputs={"red_light": "1'b0", "green_light": "1'b1", "yellow_light": "1'b0"}, actions=["green_light = 1'b1"]),
                EFSState(state_id="ST_yellow", name="YELLOW", outputs={"red_light": "1'b0", "green_light": "1'b0", "yellow_light": "1'b1"}, actions=["yellow_light = 1'b1"])
            ],
            transitions=[
                EFSTransition(transition_id="TRANS_01", source_state="RED", target_state="GREEN", condition="1'b1", condition_type="semantic", duration_cycles=5),
                EFSTransition(transition_id="TRANS_02", source_state="GREEN", target_state="YELLOW", condition="1'b1", condition_type="semantic", duration_cycles=10),
                EFSTransition(transition_id="TRANS_03", source_state="YELLOW", target_state="RED", condition="1'b1", condition_type="semantic", duration_cycles=3)
            ],
            outputs=["green_light", "red_light", "yellow_light"]
        )
    ]
)

conflicts = ConflictResolutionEngine.detect_conflicts(ir)
print("Total detected conflicts:", len(conflicts))
for c in conflicts:
    print("-------------------------------------------")
    print("ID:", c.conflict_id)
    print("Type:", c.type)
    print("Title:", c.title)
    print("Entities:", c.entities)
    print("Property:", c.property)
    print("Current values:", c.current_values)
    print("Description:", c.description)
