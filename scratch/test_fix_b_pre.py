import sys
sys.path.insert(0, "hardware_graphrag")
import json
import re
from core.efs_ir.models import EFSIR, EFSComponent
from core.retrieval.target_resolver import resolve_target

# Load active design IR
d = json.load(open("data/efs_ir/active_design_ir.json"))
ir = EFSIR.from_dict(d)

print("Current resolution for InstructionDecoder:")
res = resolve_target("InstructionDecoder", ir)
print("Status:", res.status, "Target:", res.target_component.name if res.target_component else None)
