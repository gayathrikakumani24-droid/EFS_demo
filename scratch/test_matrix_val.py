import sys
import os
import json

sys.path.insert(0, os.path.abspath("."))
from core.efs_ir.models import EFSIR
from core.agents.rtl_agent import validate_rtl_plan

with open("data/efs_ir/active_design_ir.json", "r", encoding="utf-8") as f:
    ir_dict = json.load(f)

efs_ir = EFSIR.from_dict(ir_dict)

plan = {
    "module_name": "multi_core_matrix_system",
    "ports": [
        {"name": s.name, "direction": s.direction, "width": s.width}
        for s in efs_ir.signals
    ],
    "specification_status": {
        "status": "COMPLETE",
        "blocking_issues": []
    }
}

val = validate_rtl_plan(plan, efs_ir=efs_ir)
print("=== VALIDATION RESULT ===")
print("Valid:", val.get("valid"))
print("Status:", val.get("status"))
print("Issues count:", len(val.get("issues", [])))
for iss in val.get("issues", []):
    print(" - Issue:", iss.get("issue"))
