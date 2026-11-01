import sys
import os
import json
import docx

sys.path.insert(0, os.path.abspath("hardware_graphrag"))

from core.agents.orchestrator import run_design_flow

docx_path = os.path.join("hardware_graphrag", "Traffic_controller_spec.docx")
doc = docx.Document(docx_path)
spec_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

print("Running run_design_flow with spec_text:")
print(spec_text)

results = run_design_flow(
    spec_text,
    target_outputs=["Verilog"],
    max_repair_iterations=0,
    compliance_threshold=95,
    bypass_conflicts=False,
    user_query="Generate synthesizable Verilog RTL for traffic_light_controller"
)

print("\n=== RUN_DESIGN_FLOW RESULTS ===")
print("Status:", results.get("status"))
print("Conflicts count:", len(results.get("conflicts", [])))
for c in results.get("conflicts", []):
    print(f"  Conflict: [{c.severity}] {c.conflict_id} | prop={c.property} | title={c.title} | vals={c.conflicting_values}")

efs_ir = results.get("efs_ir")
if efs_ir:
    print("\nFSMs count:", len(efs_ir.fsms))
    for f in efs_ir.fsms:
        print(f"  FSM: {f.name}")
        for st in f.states:
            print(f"    State: {st.name} | duration_cycles={getattr(st, 'duration_cycles', None)} | actions={getattr(st, 'actions', [])}")
        for tr in f.transitions:
            print(f"    Transition: {tr.source_state} -> {tr.target_state} | cond: '{tr.condition}' | type: {getattr(tr, 'condition_type', None)} | reg: {getattr(tr, 'associated_register', None)} | sig: {getattr(tr, 'associated_signal', None)}")

    print("\nSignals count:", len(efs_ir.signals))
    for s in efs_ir.signals:
        print(f"  Signal: {s.name} | dir={s.direction} | width={s.width}")

    print("\nRegisters count:", len(efs_ir.registers))
    for r in efs_ir.registers:
        print(f"  Register: {r.name} | width={r.width}")
