import sys
import os

sys.path.insert(0, os.path.abspath('.'))
from core.efs_ir.models import EFSIR, EFSMetadata, EFSFSM, EFSState, EFSTransition, EFSSignal, EFSRegister
from core.verification.conflict_resolution_engine import ConflictResolutionEngine
from core.efs_ir.validator import validate_efs_ir

# Test different condition types and associated_register / associated_signal
cases = [
    ("timer_cnt == 4", "boolean", "timer_cnt", None),
    ("timer_cnt == 4", "signal_assertion", "timer_cnt", None),
    ("timer_cnt == 4", "explicit", "timer_cnt", None),
    ("timer_cnt == 4", "timed", "timer_cnt", None),
    ("timer_done == 1'b1", "boolean", None, "timer_done"),
    ("timer_done == 1'b1", "signal_assertion", None, "timer_done"),
    ("cycle_cnt == 4", "boolean", "cycle_cnt", None),
]

for cond, ctype, assoc_reg, assoc_sig in cases:
    t = EFSTransition(
        source_state="RED",
        target_state="GREEN",
        condition=cond,
        condition_type=ctype,
        associated_register=assoc_reg,
        associated_signal=assoc_sig
    )
    # force ctype in case __post_init__ altered it
    t.condition_type = ctype
    
    ir = EFSIR(
        metadata=EFSMetadata(design_id="DSN_test", design_name="test", top_module="test"),
        signals=[
            EFSSignal(signal_id="s1", name="clk", direction="input", semantic_role="clock"),
            EFSSignal(signal_id="s2", name="reset_n", direction="input", semantic_role="reset"),
            EFSSignal(signal_id="s3", name="red_light", direction="output"),
            EFSSignal(signal_id="s4", name="green_light", direction="output"),
            EFSSignal(signal_id="s5", name="yellow_light", direction="output"),
        ],
        registers=[
            EFSRegister(register_id="r1", name="cycle_cnt", offset="0x00", width=4),
            EFSRegister(register_id="r2", name="timer_cnt", offset="0x04", width=4),
        ],
        clocks=[EFSSignal(signal_id="s1", name="clk", direction="input", semantic_role="clock")],
        resets=[EFSSignal(signal_id="s2", name="reset_n", direction="input", semantic_role="reset")],
        fsms=[
            EFSFSM(
                name="test_fsm",
                initial_state="RED",
                reset_state="RED",
                states=[EFSState(name="RED"), EFSState(name="GREEN"), EFSState(name="YELLOW")],
                transitions=[t]
            )
        ]
    )
    if assoc_sig:
        ir.signals.append(EFSSignal(signal_id="s_done", name=assoc_sig, direction="output", width="1"))
    
    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    issues = validate_efs_ir(ir)
    print(f"Case: cond='{cond}', ctype='{ctype}', assoc_reg='{assoc_reg}', assoc_sig='{assoc_sig}'")
    print(f"  -> Conflicts: {len(conflicts)}, Validation Issues: {len(issues)}")
    if conflicts:
        print("     Conflicts:", [c.title for c in conflicts])
    if issues:
        print("     Issues:", [i["message"] for i in issues])
