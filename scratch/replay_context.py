import sys
import os
import json

sys.path.insert(0, os.path.abspath('hardware_graphrag'))
from core.efs_ir.models import EFSIR
from core.context_compiler import compile_context
from core.context_compiler.generation_context import GenerationContext
from core.retrieval.context_compiler import compile_task_context

with open('hardware_graphrag/data/efs_ir/active_design_ir.json', 'r', encoding='utf-8') as f:
    ir_dict = json.load(f)

efs_ir = EFSIR.from_dict(ir_dict)

# Compile task context
context_pack, res_result = compile_task_context('', efs_ir, all_protocol_rules=[])
print("Target resolved:", getattr(context_pack, "task", None))
print("ContextPack signals:", len(context_pack.target_signals) if hasattr(context_pack, "target_signals") else "none")
print("ContextPack fsm:", getattr(context_pack, "target_fsm", None))

compiled_rtl = compile_context(
    "rtl",
    efs_ir=efs_ir,
    target_component="Verilog",
    design_spec="",
    context_pack=context_pack
)

print("\n=== COMPILED RTL KEYS ===")
print(list(compiled_rtl.keys()))
print("Module:", compiled_rtl.get("module"))
print("Signals:", [s.get("name") if isinstance(s, dict) else getattr(s, "name", "") for s in compiled_rtl.get("signals", [])])
print("FSM:", compiled_rtl.get("fsm"))
print("Registers:", compiled_rtl.get("registers"))
print("Protocol rules:", len(compiled_rtl.get("protocol_rules", [])))

gen_ctx = GenerationContext.from_compiled_rtl(compiled_rtl, token_budget=8000)
summary = gen_ctx.to_prompt_json(indent=2)
print("\n=== GENERATION CONTEXT PROMPT JSON ===")
print(summary[:1500])
