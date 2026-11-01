import json

with open('data/efs_ir/active_design_ir.json', 'r', encoding='utf-8') as f:
    ir = json.load(f)

pseudo_names = {
    'memory interface', 
    'accelerationcontroller', 
    'localmemory', 
    'accelerationcontroller_localmemory_interface', 
    'localmemory_accelerationcontroller_interface'
}

removed_ids = set()
valid_signals = []
for s in ir.get('signals', []):
    s_name = s.get('name', '').lower()
    if s_name in pseudo_names:
        removed_ids.add(s.get('signal_id'))
    else:
        if s.get('name') in ('mem_addr', 'mem_wdata', 'mem_rdata') and s.get('width') == '1':
            s['width'] = 'unspecified'
        valid_signals.append(s)

print(f"Original signals count: {len(ir['signals'])}")
print(f"Removed pseudo signals: {removed_ids}")
print(f"Valid signals count: {len(valid_signals)}")
for s in valid_signals:
    print(f"  - {s.get('name')}: dir={s.get('direction')}, width={s.get('width')}, role={s.get('semantic_role')}")

for iface in ir.get('interfaces', []):
    iface['signals'] = [sid for sid in iface.get('signals', []) if sid not in removed_ids]

ir['signals'] = valid_signals
with open('data/efs_ir/active_design_ir.json', 'w', encoding='utf-8') as f:
    json.dump(ir, f, indent=2)

print('active_design_ir.json updated successfully.')
