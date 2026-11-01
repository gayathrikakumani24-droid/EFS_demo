import sys
import os
import json
import docx

sys.path.insert(0, os.path.abspath("hardware_graphrag"))

from core.agents.requirement_agent import analyze_requirements
from core.agents.planning_agent import create_plan
from core.agents.retrieval_agent import retrieve_protocol_rules
from core.efs_ir.builder import EFSIRBuilder
from core.efs_ir.repair_engine import EFSStructuralRepairEngine
from core.verification.inference_engine import ControlledInferenceEngine
from core.efs_ir.validator import validate_efs_ir
from core.verification.conflict_resolution_engine import ConflictResolutionEngine

docx_path = os.path.join("hardware_graphrag", "Traffic_controller_spec.docx")
doc = docx.Document(docx_path)
spec_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

print("================ STAGE 1: analyze_requirements ================")
req_model = analyze_requirements(spec_text)
print("req_model keys:", list(req_model.keys()))
print("req_model['states']:", req_model.get("states"))
print("req_model['fsms']:", req_model.get("fsms"))
print("req_model['timing_rules']:", req_model.get("timing_rules"))
print("req_model['timing_requirements']:", req_model.get("timing_requirements"))
print("req_model['constraints']:", req_model.get("constraints"))
print("req_model['requirements'] count:", len(req_model.get("requirements", [])))
for r in req_model.get("requirements", []):
    print("  req:", r.get("id"), r.get("type"), "| stmt:", r.get("statement"), "| timing:", r.get("timing"), "| cond:", r.get("condition"), "| state:", r.get("state"), "| next_state:", r.get("next_state"))

print("\n================ STAGE 2: create_plan ================")
plan = create_plan(req_model)
print("plan keys:", list(plan.keys()))
print("plan fsm architecture:", plan.get("fsm_architecture") or plan.get("state_machine") or plan.get("fsm"))

print("\n================ STAGE 3: EFSIRBuilder (BEFORE repair) ================")
protocol_rules = retrieve_protocol_rules(req_model, plan)
builder = EFSIRBuilder(req_model, plan, protocol_rules, design_spec=spec_text, generation_run_id="TEST_RUN")
# Let's inspect builder.ir right BEFORE EFSStructuralRepairEngine.repair_and_validate is called!
# We can run builder's internal methods up to line 196:
builder._build_metadata()
builder._build_components()
builder._build_interfaces_and_signals()
builder._build_registers()
builder._build_instructions()
builder._build_opcodes()
builder._build_memory_regions()
builder._build_data_structures()
builder._build_errors()
builder._build_performance_requirements()
builder._build_fsms()
builder._build_protocol_rules_and_constraints()
builder._build_flows()
builder._build_transactions()
builder._build_assertion_intent()
builder._build_coverage_intent()
builder._build_conflicts()
builder._build_extraction_quality()

print("Builder FSM count before repair:", len(builder.ir.fsms))
for f in builder.ir.fsms:
    print(f"  FSM: {f.name} (initial: {f.initial_state})")
    for s in f.states:
        print(f"    State '{s.name}': duration_cycles={getattr(s, 'duration_cycles', None)}, actions={s.actions}")
    for t in f.transitions:
        print(f"    Trans: {t.source_state} -> {t.target_state} | cond: '{t.condition}' | type: {getattr(t, 'condition_type', None)}")
print("Builder timing_rules:", len(builder.ir.timing_rules))
for tr in builder.ir.timing_rules:
    print(f"  TimingRule: {tr.rule_id} | {tr.type} | {tr.cycle_relationship}")

print("\n================ STAGE 4: EFSStructuralRepairEngine.repair_and_validate ================")
repaired_ir, repair_log, issues = EFSStructuralRepairEngine.repair_and_validate(
    builder.ir,
    req_model=req_model,
    spec_text=spec_text,
    plan=plan,
    doc_ir=req_model.get("doc_ir")
)
print(f"Repairs applied ({len(repair_log)}):")
for rl in repair_log:
    print("  ", rl)
print(f"Validation issues after repair ({len(issues)}):")
for vi in issues:
    print("  ", vi)
print("FSM after repair:")
for f in repaired_ir.fsms:
    print(f"  FSM: {f.name} (initial: {f.initial_state})")
    for s in f.states:
        print(f"    State '{s.name}': duration_cycles={getattr(s, 'duration_cycles', None)}, actions={s.actions}")
    for t in f.transitions:
        print(f"    Trans: {t.source_state} -> {t.target_state} | cond: '{t.condition}' | type: {getattr(t, 'condition_type', None)} | reg: {getattr(t, 'associated_register', None)} | sig: {getattr(t, 'associated_signal', None)}")

print("\n================ STAGE 5: ControlledInferenceEngine ================")
inference_engine = ControlledInferenceEngine(repaired_ir, plan, req_model)
completed_model = inference_engine.infer_and_complete()
inferred_ir = completed_model.efs_ir
print("FSM after ControlledInferenceEngine:")
for f in inferred_ir.fsms:
    print(f"  FSM: {f.name}")
    for t in f.transitions:
        print(f"    Trans: {t.source_state} -> {t.target_state} | cond: '{t.condition}' | type: {getattr(t, 'condition_type', None)}")

print("\n================ STAGE 6: Conflict Detection ================")
val_issues = validate_efs_ir(inferred_ir)
detected_conflicts = ConflictResolutionEngine.detect_conflicts(inferred_ir, req_model, validation_issues=val_issues)
detected_conflicts = ConflictResolutionEngine.deduplicate_conflicts(detected_conflicts)
print(f"Conflicts detected ({len(detected_conflicts)}):")
for c in detected_conflicts:
    print(f"  [{c.severity}] {c.conflict_id} | prop={c.property} | title={c.title} | vals={c.conflicting_values}")
