import sys
import os
import json

sys.path.insert(0, os.path.abspath('.'))
from core.context_compiler.rtl_context import RTLContextCompiler
from core.efs_ir.models import EFSIR

with open("data/efs_ir/traffic_light_updated_ir.json", "r", encoding="utf-8") as f:
    data = json.load(f)

ir = EFSIR.from_dict(data)
compiler = RTLContextCompiler()
bundle = compiler.compile(efs_ir=ir)

print("=== COMPILED RTL CONTEXT BUNDLE ===")
print("Module Name:", bundle.get("module_name"))
print("Ports:", [p["name"] for p in bundle.get("ports", [])])
print("Registers:", [r["name"] for r in bundle.get("registers", [])])
print("FSM Name:", bundle.get("fsm", {}).get("name"))
print("FSM States:", bundle.get("fsm", {}).get("states"))
print("FSM Transitions:")
for tr in bundle.get("fsm", {}).get("transitions", []):
    print(f"  {tr['from']} -> {tr['to']}: {tr['condition']['expression']}")
