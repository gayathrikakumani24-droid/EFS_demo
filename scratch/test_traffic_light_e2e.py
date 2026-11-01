import sys
import os

sys.path.insert(0, os.path.abspath('.'))
from core.efs_ir.models import (
    EFSIR, EFSMetadata, EFSComponent, EFSSignal,
    EFSFSM, EFSState, EFSTransition, EFSRegister, EFSConflict
)
from core.verification.conflict_resolution_engine import ConflictResolutionEngine
from core.efs_ir.validator import validate_efs_ir

# Build IR with timer_cnt and derived signals
ir = EFSIR(
    metadata=EFSMetadata(
        design_id="DSN_traffic_light",
        design_name="traffic_light_controller",
        canonical_id="EFS_traffic_light",
        display_name="Traffic Light Controller",
        top_module="traffic_light_controller"
    ),
    components=[
        EFSComponent(component_id="COMP_tlc", name="traffic_light_controller", type="module")
    ],
    signals=[
        EFSSignal(signal_id="SIG_clk", name="clk", direction="input", width="1", semantic_role="clock"),
        EFSSignal(signal_id="SIG_rst_n", name="reset_n", direction="input", width="1", semantic_role="reset"),
        EFSSignal(signal_id="SIG_red", name="red_light", direction="output", width="1"),
        EFSSignal(signal_id="SIG_green", name="green_light", direction="output", width="1"),
        EFSSignal(signal_id="SIG_yellow", name="yellow_light", direction="output", width="1"),
        EFSSignal(signal_id="SIG_timer_done", name="timer_done", direction="output", width="1", semantic_role="control", description="Terminal count flag asserted when timer reaches duration - 1")
    ],
    clocks=[EFSSignal(signal_id="SIG_clk", name="clk", direction="input", width="1", semantic_role="clock")],
    resets=[EFSSignal(signal_id="SIG_rst_n", name="reset_n", direction="input", width="1", semantic_role="reset")],
    registers=[
        EFSRegister(
            register_id="REG_timer_cnt",
            name="timer_cnt",
            offset="0x00",
            width=4,
            reset_value="0",
            description="4-bit cycle counter for timing state durations. Increments on clk, clears on state change or reset_n assertion."
        )
    ],
    fsms=[
        EFSFSM(
            fsm_id="FSM_traffic_light",
            name="traffic_light_fsm",
            initial_state="RED",
            reset_state="RED",
            reset_behavior="Asynchronous active-low reset to RED on reset_n with timer_cnt cleared to 0",
            states=[
                EFSState(state_id="ST_red", name="RED", outputs={"red_light": "1'b1", "green_light": "1'b0", "yellow_light": "1'b0"}, actions=["red_light = 1'b1"]),
                EFSState(state_id="ST_green", name="GREEN", outputs={"red_light": "1'b0", "green_light": "1'b1", "yellow_light": "1'b0"}, actions=["green_light = 1'b1"]),
                EFSState(state_id="ST_yellow", name="YELLOW", outputs={"red_light": "1'b0", "green_light": "1'b0", "yellow_light": "1'b1"}, actions=["yellow_light = 1'b1"])
            ],
            transitions=[
                EFSTransition(
                    transition_id="TRANS_01",
                    source_state="RED",
                    target_state="GREEN",
                    condition="timer_cnt == 4'd4",
                    condition_type="signal_assertion",
                    associated_register="timer_cnt",
                    duration_cycles=5
                ),
                EFSTransition(
                    transition_id="TRANS_02",
                    source_state="GREEN",
                    target_state="YELLOW",
                    condition="timer_cnt == 4'd9",
                    condition_type="signal_assertion",
                    associated_register="timer_cnt",
                    duration_cycles=10
                ),
                EFSTransition(
                    transition_id="TRANS_03",
                    source_state="YELLOW",
                    target_state="RED",
                    condition="timer_cnt == 4'd2",
                    condition_type="signal_assertion",
                    associated_register="timer_cnt",
                    duration_cycles=3
                )
            ],
            outputs=["green_light", "red_light", "yellow_light"]
        )
    ]
)

# Test with original conflicts in ir.conflicts to verify reconciliation
ir.conflicts = [
    EFSConflict(
        conflict_id="CONF_e6e7868f",
        type="SOURCE_AMBIGUOUS",
        property="transition_condition",
        title="FSM Transition Semantic/Ambiguous: RED -> GREEN",
        entities=["RED", "GREEN"],
        affected_entities=["RED", "GREEN"],
        conflicting_values=["1'b1"],
        current_values={"RED->GREEN": "1'b1"}
    ),
    EFSConflict(
        conflict_id="CONF_48403789",
        type="SOURCE_AMBIGUOUS",
        property="transition_condition",
        title="FSM Transition Semantic/Ambiguous: GREEN -> YELLOW",
        entities=["GREEN", "YELLOW"],
        affected_entities=["GREEN", "YELLOW"],
        conflicting_values=["1'b1"],
        current_values={"GREEN->YELLOW": "1'b1"}
    ),
    EFSConflict(
        conflict_id="CONF_be28b83d",
        type="SOURCE_AMBIGUOUS",
        property="transition_condition",
        title="FSM Transition Semantic/Ambiguous: YELLOW -> RED",
        entities=["YELLOW", "RED"],
        affected_entities=["YELLOW", "RED"],
        conflicting_values=["1'b1"],
        current_values={"YELLOW->RED": "1'b1"}
    )
]

detected = ConflictResolutionEngine.detect_conflicts(ir)
issues = validate_efs_ir(ir)

print("=== RECONCILED CONFLICTS ===")
print("Total conflicts remaining active:", len([c for c in detected if c.status in ("ACTIVE", "UNRESOLVED")]))
for c in detected:
    print(f"  Conflict {c.conflict_id}: status={c.status}, blocking={c.blocking}, title={c.title}")

print("\n=== VALIDATION ISSUES ===")
print("Total validation issues:", len(issues))
for iss in issues:
    print(f"  Issue: [{iss.get('severity')}] {iss.get('message')}")
