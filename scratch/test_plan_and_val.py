import sys
import os
import json

sys.path.insert(0, os.path.abspath('.'))
from core.efs_ir.models import EFSIR
from core.retrieval.context_pack import ContextPack, TaskIntent
from core.retrieval.target_resolver import resolve_target
from core.retrieval.dependency_resolver import resolve_dependencies
from core.agents.rtl_agent import plan_rtl, validate_rtl_plan

with open('data/efs_ir/active_design_ir.json', 'r', encoding='utf-8') as f:
    ir_dict = json.load(f)

efs_ir = EFSIR.from_dict(ir_dict)

res_result = resolve_target("AccelerationController", efs_ir)
target_comp = res_result.target_component
dep_res = resolve_dependencies(target_comp, efs_ir, max_depth=1)

intent = TaskIntent(task="RTL_GENERATION", target="AccelerationController", language="Verilog")
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

print("=== CALLING plan_rtl ===")
plan_res = plan_rtl(design_spec="", target="Verilog", efs_ir=efs_ir, context_pack=pack)

print("\n=== PLAN PORTS ===")
for p in plan_res.get("ports", []):
    print(f"  {p.get('name')}: dir={p.get('direction')}, width={p.get('width')}")

print("\n=== PLAN FSM TRANSITIONS ===")
for tr in plan_res.get("fsm", {}).get("transitions", []):
    print(f"  {tr.get('from')} -> {tr.get('to')}: condition={tr.get('condition')}")

print("\n=== PLAN SPEC STATUS ===")
print(plan_res.get("specification_status"))

print("\n=== CALLING validate_rtl_plan ===")
val_res = validate_rtl_plan(plan_res, efs_ir=efs_ir, context_pack=pack)
print(f"Valid: {val_res.get('valid')}, Status: {val_res.get('status')}")
print(f"Issues count: {len(val_res.get('issues', []))}")
for idx, iss in enumerate(val_res.get("issues", []), 1):
    print(f"  Issue {idx}: [{iss.get('category')}] {iss.get('classification')} - {iss.get('issue')}")
    print(f"           Evidence: {iss.get('evidence')}")
    print(f"           Required: {iss.get('required_information')}")
    print(f"           Blocking: {iss.get('is_blocking')}")
