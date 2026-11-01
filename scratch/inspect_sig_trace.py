import json

with open('hardware_graphrag/data/efs_ir/active_design_ir.json', 'r', encoding='utf-8') as f:
    ir = json.load(f)

for s in ir.get('signals', []):
    t = s.get('traceability') or {}
    print(f"Name: {s.get('name')}")
    print(f"  Dir: {s.get('direction')}, Width: {s.get('width')}, Owner: {s.get('owner')}, Consumer: {s.get('consumer')}")
    print(f"  Method: {t.get('extraction_method')}")
    print(f"  Orig Text: {t.get('original_text')}")
    print(f"  Desc: {s.get('description')}")
    print("-" * 50)
