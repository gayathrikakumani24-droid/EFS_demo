import json

with open('hardware_graphrag/data/efs_ir/active_design_ir.json', 'r', encoding='utf-8') as f:
    ir = json.load(f)

texts = set()
for s in ir.get('signals', []):
    t = s.get('traceability', {})
    if t.get('original_text'):
        texts.add((t.get('block_id'), t.get('original_text')))

for f in ir.get('fsms', []):
    t = f.get('traceability', {})
    if t.get('original_text'):
        texts.add((t.get('block_id'), t.get('original_text')))
    for tr in f.get('transitions', []):
        t2 = tr.get('traceability', {})
        if t2.get('original_text'):
            texts.add((t2.get('block_id'), t2.get('original_text')))

for iface in ir.get('interfaces', []):
    t = iface.get('traceability', {})
    if t.get('original_text'):
        texts.add((t.get('block_id'), t.get('original_text')))

for m in ir.get('memory_regions', []):
    t = m.get('traceability', {})
    if t.get('original_text'):
        texts.add((t.get('block_id'), t.get('original_text')))

for blk, txt in sorted(texts, key=lambda x: str(x[0])):
    print(f"[{blk}] {txt}")
