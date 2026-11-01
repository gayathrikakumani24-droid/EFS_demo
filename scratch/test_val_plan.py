import sys
import os
import json

sys.path.insert(0, os.path.abspath('hardware_graphrag'))
from core.efs_ir.models import EFSIR
from core.retrieval.context_pack import ContextPack, TaskIntent
from core.retrieval.target_resolver import resolve_target
from core.retrieval.dependency_resolver import resolve_dependencies
from core.agents.rtl_agent import validate_rtl_plan

with open('hardware_graphrag/data/efs_ir/active_design_ir.json', 'r', encoding='utf-8') as f:
    ir_dict = json.load(f)

efs_ir = EFSIR.from_dict(ir_dict)

res_result = resolve_target("AccelerationController", efs_ir)
target_comp = res_result.target_component
dep_res = resolve_dependencies(target_comp, efs_ir, max_depth=1)

print("Target Component:", target_comp.name)
print("Dep Submodules:", [m.name for m in dep_res.direct_submodules])
print("Dep Interfaces:", [i.name for i in dep_res.interfaces])
print("Dep Signals:")
for s in dep_res.signals:
    print(f"  {s.name} ({s.direction}, {s.width} bits, owner={s.owner}, consumer={s.consumer})")
print("Dep FSMs:", [f.name for f in dep_res.fsms])
for f in dep_res.fsms:
    print(f"  Transitions in {f.name}:")
    for t in f.transitions:
        print(f"    {t.source_state} -> {t.target_state} : '{t.condition}'")

intent = TaskIntent(task="RTL_GENERATION", target="AccelerationController", interface="", language="Verilog")
pack = ContextPack(
    task=intent,
    target_component=target_comp.to_dict(),
    interfaces=[i.to_dict() for i in dep_res.interfaces],
    signals=[s.to_dict() for s in dep_res.signals],
    registers=[r.to_dict() for r in dep_res.registers],
    dependencies=[d.to_dict() for d in dep_res.direct_submodules],
    fsm=dep_res.fsms[0].to_dict() if dep_res.fsms else {},
    conflicts=[]
)

# Test 1: Empty plan
val_empty = validate_rtl_plan({}, efs_ir=efs_ir, context_pack=pack)
print("\n=== VAL EMPTY PLAN ===")
print("valid:", val_empty.get("valid"))
print("status:", val_empty.get("status"))
for iss in val_empty.get("issues", []):
    print("Issue:", iss)

# Test 2: Perfectly matched plan according to signals and FSM
matched_plan = {
    "module_name": "AccelerationController",
    "ports": [{"name": s.name, "direction": s.direction, "width": s.width} for s in dep_res.signals],
    "fsm": {
        "states": [st.name for st in dep_res.fsms[0].states] if dep_res.fsms else [],
        "transitions": [{"from": t.source_state, "to": t.target_state, "condition": t.condition} for t in dep_res.fsms[0].transitions] if dep_res.fsms else []
    }
}

val_matched = validate_rtl_plan(matched_plan, efs_ir=efs_ir, context_pack=pack)
print("\n=== VAL MATCHED PLAN (using EFS IR transition conditions) ===")
print("valid:", val_matched.get("valid"))
print("status:", val_matched.get("status"))
for iss in val_matched.get("issues", []):
    print("Issue:", iss)
