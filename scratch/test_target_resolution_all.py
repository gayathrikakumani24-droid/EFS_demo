import sys
sys.path.insert(0, "hardware_graphrag")
import json
from core.efs_ir.models import EFSIR
from core.retrieval.target_resolver import resolve_target

d = json.load(open("data/efs_ir/active_design_ir.json"))
ir = EFSIR.from_dict(d)

test_cases = [
    ("InstructionDecoder", True, "InstructionDecoder"),
    ("instruction_decoder", True, "instruction_decoder"),
    ("Instruction Decoder", True, "Instruction Decoder"),
    ("CommandDecoder", True, "CommandDecoder"),
    ("Matrix ALU", True, "Matrix ALU"),
    ("Global Scheduler", True, "Global Scheduler"),
    ("CompletelyUnknownUnit", False, None),
    ("CompletelyUnknownHardwareBlock", False, None),
]

all_passed = True
for target_name, expected_success, expected_comp_name in test_cases:
    res = resolve_target(target_name, ir)
    success = res.is_success()
    comp_name = res.target_component.name if res.target_component else None
    passed = (success == expected_success) and (comp_name == expected_comp_name if expected_success else comp_name is None)
    status_icon = "PASS" if passed else "FAIL"
    if not passed:
        all_passed = False
    print(f"[{status_icon}] Query: '{target_name}' -> Success: {success} (Expected: {expected_success}), Comp: {comp_name} (Expected: {expected_comp_name}), Method: {res.match_method}")

if all_passed:
    print("\nALL TARGET RESOLUTION TESTS PASSED!")
else:
    print("\nSOME TESTS FAILED!")
    sys.exit(1)
