import sys
import os

sys.path.insert(0, os.path.abspath('.'))
from core.efs_ir.models import EFSIR, EFSMetadata, EFSFSM, EFSState, EFSTransition, EFSSignal, EFSRegister
from core.verification.conflict_resolution_engine import ConflictResolutionEngine
from core.efs_ir.validator import validate_efs_ir

# Test if duration_cycles=5 on transition triggers validator when ctype='boolean' vs 'signal_assertion' vs 'timed'
for ctype in ["boolean", "signal_assertion", "timed", "explicit"]:
    t = EFSTransition(
        source_state="RED",
        target_state="GREEN",
        condition="timer_cnt == 4",
        condition_type=ctype,
        duration_cycles=5,
        associated_register="timer_cnt"
    )
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
            EFSRegister(register_id="r1", name="timer_cnt", offset="0x00", width=4),
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
    
    conflicts = ConflictResolutionEngine.detect_conflicts(ir)
    issues = validate_efs_ir(ir)
    print(f"ctype='{ctype}', duration_cycles=5:")
    print(f"  -> Conflicts: {len(conflicts)}, Issues: {len(issues)}")
    if conflicts:
        print("     Conflicts:", [c.title for c in conflicts])
    if issues:
        print("     Issues:", [i["message"] for i in issues])
