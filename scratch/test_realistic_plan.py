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

# Realistic LLM plan for AccelerationController
realistic_plan = {
    "module_name": "AccelerationController",
    "ports": [
        {"name": "clk", "direction": "input", "width": "1"},
        {"name": "rst_n", "direction": "input", "width": "1"},
        {"name": "start", "direction": "input", "width": "1"},
        {"name": "busy", "direction": "output", "width": "1"},
        {"name": "done", "direction": "output", "width": "1"},
        {"name": "instruction", "direction": "input", "width": "32"},
        {"name": "mem_addr", "direction": "output", "width": "16"},
        {"name": "mem_wdata", "direction": "output", "width": "32"},
        {"name": "mem_we", "direction": "output", "width": "1"},
        {"name": "mem_rdata", "direction": "input", "width": "32"}
    ],
    "fsm": {
        "states": ["IDLE", "EXECUTE", "DONE"],
        "initial_state": "IDLE",
        "transitions": [
            {"from": "IDLE", "to": "EXECUTE", "condition": {"expression": "start == 1'b1"}},
            {"from": "EXECUTE", "to": "DONE", "condition": {"expression": "execution completes"}},
            {"from": "DONE", "to": "IDLE", "condition": {"expression": "the next clock cycle"}}
        ]
    }
}

val = validate_rtl_plan(realistic_plan, efs_ir=efs_ir, context_pack=pack)
print("=== VALIDATE REALISTIC PLAN ===")
print("valid:", val.get("valid"))
print("status:", val.get("status"))
for iss in val.get("issues", []):
    print("Issue:", iss)

print("\n=== CHECKING SECTION 4: BLOCKING ISSUES IN PLAN SPECIFICATION STATUS ===")
# What if the LLM plan returned specification_status with blocking issues?
realistic_plan_with_issues = dict(realistic_plan)
realistic_plan_with_issues["specification_status"] = {
    "status": "INCOMPLETE",
    "blocking_issues": [
        {
            "category": "FSM",
            "issue": "EXECUTE -> DONE transition condition is ambiguous ('execution completes')",
            "required_information": "Specify explicit signal or counter condition for execution completion",
            "evidence": "EXECUTE -> DONE: execution completes"
        }
    ]
}
val_issues = validate_rtl_plan(realistic_plan_with_issues, efs_ir=efs_ir, context_pack=pack)
print("valid with issues:", val_issues.get("valid"))
for iss in val_issues.get("issues", []):
    print("Issue:", iss)
