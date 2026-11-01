import sys
import os

sys.path.insert(0, os.path.abspath('.'))
from core.efs_ir.models import EFSIR, EFSMetadata, EFSFSM, EFSState, EFSTransition, EFSSignal
from core.efs_ir.validator import validate_efs_ir

# Test 1: condition contains 'cycle' and condition_type='boolean'
t1 = EFSTransition(
    source_state="RED",
    target_state="GREEN",
    condition="cycle_counter == 4",
    condition_type="boolean"
)
ir1 = EFSIR(
    metadata=EFSMetadata(design_id="DSN_1", design_name="test", top_module="test"),
    signals=[
        EFSSignal(signal_id="s1", name="clk", direction="input", semantic_role="clock"),
        EFSSignal(signal_id="s2", name="rst_n", direction="input", semantic_role="reset"),
        EFSSignal(signal_id="s3", name="out", direction="output")
    ],
    resets=[EFSSignal(signal_id="s2", name="rst_n", direction="input", semantic_role="reset")],
    fsms=[EFSFSM(name="test_fsm", states=[EFSState(name="RED"), EFSState(name="GREEN")], transitions=[t1])]
)
issues1 = validate_efs_ir(ir1)
print("Issues with 'cycle_counter == 4' and boolean:")
for iss in issues1:
    if "Cycle duration condition" in iss.get("message", ""):
        print(" ->", iss["message"])

# Test 2: condition does NOT contain 'cycle' (e.g. 'timer_cnt == 4' or 'timer == 4' or 'count == 4') and condition_type='boolean', duration_cycles=None
t2 = EFSTransition(
    source_state="RED",
    target_state="GREEN",
    condition="timer_cnt == 4",
    condition_type="boolean",
    duration_cycles=None
)
ir2 = EFSIR(
    metadata=EFSMetadata(design_id="DSN_2", design_name="test", top_module="test"),
    signals=[
        EFSSignal(signal_id="s1", name="clk", direction="input", semantic_role="clock"),
        EFSSignal(signal_id="s2", name="rst_n", direction="input", semantic_role="reset"),
        EFSSignal(signal_id="s3", name="out", direction="output")
    ],
    resets=[EFSSignal(signal_id="s2", name="rst_n", direction="input", semantic_role="reset")],
    fsms=[EFSFSM(name="test_fsm", states=[EFSState(name="RED"), EFSState(name="GREEN")], transitions=[t2])]
)
issues2 = validate_efs_ir(ir2)
print("Issues with 'timer_cnt == 4' and boolean:")
for iss in issues2:
    if "Cycle duration condition" in iss.get("message", ""):
        print(" ->", iss["message"])
