import sys
sys.path.insert(0, ".")
import json
from core.efs_ir.models import EFSIR
from core.agents.plantuml_agent import get_heuristic_diagrams, resolve_diagram_ids_to_names, validate_plantuml_syntax

with open("data/efs_ir/active_design_ir.json", "r", encoding="utf-8") as f:
    ir = EFSIR.from_dict(json.load(f))

plan = {"target": "counter_controller"}
diags = get_heuristic_diagrams(plan, efs_ir=ir)
for k, v in diags.items():
    print(f"RAW {k} startuml count:", v.count("@startuml"))

resolved = resolve_diagram_ids_to_names(diags, efs_ir=ir)
for k, v in resolved.items():
    print(f"RESOLVED {k} startuml count:", v.count("@startuml"))
    val = validate_plantuml_syntax(v, diagram_type=k)
    print(f"  Valid: {val.is_valid}, errors: {val.errors}")
