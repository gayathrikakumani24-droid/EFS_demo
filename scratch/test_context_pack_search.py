import sys
sys.path.insert(0, "hardware_graphrag")
import json
from core.efs_ir.models import EFSIR
from core.retrieval.context_compiler import compile_task_context

d = json.load(open("data/efs_ir/active_design_ir.json"))
ir = EFSIR.from_dict(d)

cp, res = compile_task_context("InstructionDecoder", ir)
print("Source evidence count:", len(cp.source_evidence))
for i, se in enumerate(cp.source_evidence):
    txt = se.get("text", "")
    print(f"\n--- Chunk {i} ---")
    print(txt[:250])
