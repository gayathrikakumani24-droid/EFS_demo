import sys
import os
import json

sys.path.insert(0, os.path.abspath('.'))
from core.efs_ir.models import EFSIR
from core.efs_ir.validation_rules import is_valid_signal_candidate
from core.retrieval.context_pack import ContextPack, TaskIntent
from core.retrieval.target_resolver import resolve_target
from core.retrieval.dependency_resolver import resolve_dependencies
from core.agents.rtl_agent import validate_rtl_plan

print("==================================================", flush=True)
print("TEST A: Pseudo-Signal Qualification", flush=True)
print("==================================================", flush=True)
known_comps = ["AccelerationController", "LocalMemory", "Specification Core Requirements"]
known_ifaces = ["AccelerationController_LocalMemory_interface", "LocalMemory_AccelerationController_interface", "Memory Interface"]

pseudo_candidates = [
    ("AccelerationController", {"type": "component"}),
    ("AccelerationController_LocalMemory_interface", {"type": "interface", "role": "interconnect"}),
    ("LocalMemory_AccelerationController_interface", {"type": "interface", "role": "interconnect"}),
    ("Memory Interface", {"type": "interface"}),
]

for name, attrs in pseudo_candidates:
    is_valid = is_valid_signal_candidate(name, attrs, known_components=known_comps, known_interfaces=known_ifaces)
    print(f"Candidate: '{name}' -> is_valid_signal={is_valid} (Expected: False)", flush=True)
    assert not is_valid, f"Failed: {name} should NOT be a valid signal!"

print("\n==================================================", flush=True)
print("TEST B: Real Signal Qualification", flush=True)
print("==================================================", flush=True)
real_candidates = [
    ("clk", {"direction": "input", "width": "1"}),
    ("rst_n", {"direction": "input", "width": "1"}),
    ("start", {"direction": "input", "width": "1"}),
    ("busy", {"direction": "output", "width": "1"}),
    ("done", {"direction": "output", "width": "1"}),
    ("instruction", {"direction": "input", "width": "1"}),
    ("mem_addr", {"direction": "output", "width": "unspecified"}),
    ("mem_wdata", {"direction": "output", "width": "unspecified"}),
    ("mem_we", {"direction": "output", "width": "1"}),
    ("mem_rdata", {"direction": "input", "width": "unspecified"}),
]

for name, attrs in real_candidates:
    is_valid = is_valid_signal_candidate(name, attrs, known_components=known_comps, known_interfaces=known_ifaces)
    print(f"Candidate: '{name}' -> is_valid_signal={is_valid} (Expected: True)", flush=True)
    assert is_valid, f"Failed: {name} should be a valid signal!"

print("\n==================================================", flush=True)
print("TEST C: ContextPack Scoping for AccelerationController", flush=True)
print("==================================================", flush=True)
with open('data/efs_ir/active_design_ir.json', 'r', encoding='utf-8') as f:
    ir_dict = json.load(f)

efs_ir = EFSIR.from_dict(ir_dict)

res_result = resolve_target("AccelerationController", efs_ir)
target_comp = res_result.target_component
assert target_comp is not None, "AccelerationController component not found in active IR!"
print(f"Target component: '{target_comp.name}' (ID: {target_comp.component_id})", flush=True)
print(f"Target component interfaces: {target_comp.interfaces}", flush=True)

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

print(f"\nResolved dependencies signals count: {len(dep_res.signals)}", flush=True)
dep_signal_names = [s.name for s in dep_res.signals]
print(f"Signals in resolved dependencies: {dep_signal_names}", flush=True)

print(f"\nContextPack target signals count: {len(pack.signals)}", flush=True)
pack_signal_names = [s.get('name') if isinstance(s, dict) else s.name for s in pack.signals]
print(f"Signals in ContextPack: {pack_signal_names}", flush=True)

expected_signals = ["clk", "rst_n", "start", "busy", "done", "instruction", "mem_addr", "mem_wdata", "mem_we", "mem_rdata"]
for s in expected_signals:
    assert s in pack_signal_names, f"Expected signal '{s}' missing from ContextPack!"
    assert s in dep_signal_names, f"Expected signal '{s}' missing from resolved dependencies!"

for pseudo in ["AccelerationController", "AccelerationController_LocalMemory_interface", "LocalMemory_AccelerationController_interface", "Memory Interface"]:
    assert pseudo not in pack_signal_names, f"Pseudo-signal '{pseudo}' falsely present in ContextPack!"
    assert pseudo not in dep_signal_names, f"Pseudo-signal '{pseudo}' falsely present in resolved dependencies!"

print("\nALL SIGNALS IN CONTEXTPACK VERIFIED CORRECT!", flush=True)

print("\n==================================================", flush=True)
print("TEST D: Width Handling", flush=True)
print("==================================================", flush=True)
for s in efs_ir.signals:
    print(f"  {s.name:15} -> width = {s.width} (status: {'unspecified (uncertainty preserved)' if s.width == 'unspecified' else 'explicit'})", flush=True)

print("\n==================================================", flush=True)
print("TEST E: RTL Plan Validation & FSM Blocker Check", flush=True)
print("==================================================", flush=True)

# Realistic plan representing AccelerationController architecture
realistic_plan = {
    "module_name": "AccelerationController",
    "ports": [
        {"name": "clk", "direction": "input", "width": "1"},
        {"name": "rst_n", "direction": "input", "width": "1"},
        {"name": "start", "direction": "input", "width": "1"},
        {"name": "busy", "direction": "output", "width": "1"},
        {"name": "done", "direction": "output", "width": "1"},
        {"name": "instruction", "direction": "input", "width": "1"},
        {"name": "mem_addr", "direction": "output", "width": "16"},
        {"name": "mem_wdata", "direction": "output", "width": "32"},
        {"name": "mem_we", "direction": "output", "width": "1"},
        {"name": "mem_rdata", "direction": "input", "width": "32"},
    ],
    "fsm": {
        "states": ["IDLE", "EXECUTE", "DONE"],
        "transitions": [
            {
                "from": "IDLE",
                "to": "EXECUTE",
                "condition": "start == 1'b1"
            },
            {
                "from": "EXECUTE",
                "to": "DONE",
                "condition": "execution completes"
            },
            {
                "from": "DONE",
                "to": "IDLE",
                "condition": "1'b1"
            }
        ]
    },
    "specification_status": {
        "status": "READY_FOR_GENERATION",
        "blocking_issues": []
    }
}

val_result = validate_rtl_plan(realistic_plan, efs_ir=efs_ir, context_pack=pack)
print(f"Plan validation status: valid = {val_result.get('valid')}, status = {val_result.get('status')}", flush=True)
print(f"Total issues: {len(val_result.get('issues', []))}", flush=True)
for idx, iss in enumerate(val_result.get("issues", []), 1):
    print(f"  Issue {idx}: [{iss.get('category')}] {iss.get('classification')} - {iss.get('issue')}", flush=True)
    print(f"           Evidence: {iss.get('evidence')}", flush=True)
    print(f"           Required: {iss.get('required_information')}", flush=True)
    print(f"           Blocking: {iss.get('is_blocking')}", flush=True)
