import sys
import os
import json
import docx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import CONFIG
from core.pipeline import run_pipeline
from core.efs_ir.models import EFSIR
from core.agents.requirement_agent import analyze_requirements
from core.agents.planning_agent import create_plan
from core.agents.retrieval_agent import retrieve_protocol_rules
from core.efs_ir.builder import EFSIRBuilder
from core.efs_ir.repair_engine import EFSStructuralRepairEngine
from core.verification.inference_engine import ControlledInferenceEngine
from core.verification.conflict_resolution_engine import ConflictResolutionEngine
from core.efs_ir.validator import validate_efs_ir

docx_path = os.path.join(os.path.dirname(__file__), "..", "Traffic_controller_spec.docx")

def print_fsm(label, efs_ir):
    print(f"\n=== FSM STATE at {label} ===")
    if not efs_ir or not getattr(efs_ir, "fsms", None):
        print("  No FSMs present.")
        return
    for fsm in efs_ir.fsms:
        print(f"  FSM: {fsm.name} (initial: {fsm.initial_state})")
        print("  States:")
        for s in fsm.states:
            print(f"    - State '{s.name}': duration_cycles={getattr(s, 'duration_cycles', None)}, actions={getattr(s, 'actions', [])}")
        print("  Transitions:")
        for t in fsm.transitions:
            print(f"    - {t.source_state} -> {t.target_state} | cond: '{t.condition}' | type: {getattr(t, 'condition_type', None)} | reg: {getattr(t, 'associated_register', None)} | sig: {getattr(t, 'associated_signal', None)}")

def print_conflicts(label, conflicts):
    print(f"\n=== CONFLICTS at {label} (count: {len(conflicts)}) ===")
    for c in conflicts:
        print(f"  - [{c.severity}] ID={c.conflict_id} | prop={c.property} | title={c.title} | vals={c.conflicting_values} | status={getattr(c, 'resolution_status', 'UNRESOLVED')}/{getattr(c, 'status', 'N/A')}")

def run_repro():
    print("==================================================")
    print("PATH 1: run_pipeline (Upload Page Flow)")
    print("==================================================")
    pipeline_res = run_pipeline(docx_path, filename="Traffic_controller_spec.docx")
    print_fsm("run_pipeline efs_ir", pipeline_res.efs_ir)
    print_conflicts("run_pipeline conflicts", pipeline_res.efs_ir.conflicts)

    print("\n==================================================")
    print("PATH 2: Hardware Design Flow (Upload docx in Wizard)")
    print("==================================================")
    doc_obj = docx.Document(docx_path)
    spec_text = "\n".join([p.text for p in doc_obj.paragraphs])
    print(f"Spec text extracted from paragraphs ({len(spec_text)} chars):")
    for line in spec_text.splitlines():
        if line.strip():
            print("  ", line)

    print("\n[Stage 1: analyze_requirements]")
    req_model = analyze_requirements(spec_text)
    print(f"req_model states: {req_model.get('states')}")
    print(f"req_model fsms: {req_model.get('fsms')}")
    print(f"req_model timing_rules: {len(req_model.get('timing_rules', []))}")
    for tr in req_model.get('timing_rules', []):
        print(f"  TimingRule: {tr.get('rule_id')} {tr.get('name')} {tr.get('rule_type')} {tr.get('cycle_relationship')}")

    print("\n[Stage 2: create_plan & retrieve_protocol_rules]")
    plan = create_plan(req_model)
    protocol_rules = retrieve_protocol_rules(req_model, plan)

    print("\n[Stage 3: EFSIRBuilder.build()]")
    builder = EFSIRBuilder(req_model, plan, protocol_rules, design_spec=spec_text)
    efs_ir_built = builder.build()
    print_fsm("After EFSIRBuilder.build()", efs_ir_built)

    print("\n[Stage 4: ControlledInferenceEngine.infer_and_complete()]")
    inference_engine = ControlledInferenceEngine(efs_ir_built, plan, req_model)
    completed_model = inference_engine.infer_and_complete()
    efs_ir_inferred = completed_model.efs_ir
    print_fsm("After ControlledInferenceEngine.infer_and_complete()", efs_ir_inferred)

    print("\n[Stage 5: Conflict Detection]")
    val_issues = validate_efs_ir(efs_ir_inferred)
    conflicts = ConflictResolutionEngine.detect_conflicts(efs_ir_inferred, req_model, validation_issues=val_issues)
    print_conflicts("After detect_conflicts in Design Flow", conflicts)

    print("\n[Stage 6: Review Page Validation Issues]")
    print(f"Total validation issues: {len(val_issues)}")
    for issue in val_issues:
        print(f"  - [{issue['severity']}] {issue['category']}: {issue['message']}")

if __name__ == "__main__":
    run_repro()
