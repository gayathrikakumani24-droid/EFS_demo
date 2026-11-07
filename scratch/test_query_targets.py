import sys
sys.path.insert(0, 'hardware_graphrag')
import json
from core.retrieval.context_compiler import compile_task_context
from core.efs_ir.models import EFSIR

with open('hardware_graphrag/data/efs_ir/active_design_ir.json', 'r', encoding='utf-8') as f:
    efs_ir = EFSIR.from_dict(json.load(f))

for query in [
    'Generate Verilog RTL for multi_core_matrix_system',
    'Generate Verilog RTL for FSM_Control_Unit',
    'Generate Verilog RTL for Matrix Processing Elements',
    'Generate Verilog RTL'
]:
    pack, res = compile_task_context(query, efs_ir)
    print(f'Query: "{query}" -> Status: {res.status}, Target: {getattr(res.target_component, "name", None)}')
