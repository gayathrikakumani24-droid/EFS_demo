import json

with open("hardware_graphrag/data/efs_ir/active_design_ir.json", "r", encoding="utf-8") as f:
    ir = json.load(f)

for s in ir.get("signals", []):
    print(f"{s.get('name'):25} | dir: {s.get('direction'):8} | desc: {s.get('description'):35} | dir_src: {s.get('direction_source')}")
