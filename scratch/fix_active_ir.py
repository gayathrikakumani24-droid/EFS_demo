import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
from core.efs_ir.signal_normalization import parse_hdl_port_declaration


with open('data/efs_ir/active_design_ir.json', 'r', encoding='utf-8') as f:
    ir = json.load(f)

changed = []
for s in ir.get('signals', []):
    desc = s.get('description', '')
    if desc:
        hdl = parse_hdl_port_declaration(desc)
        if hdl:
            old_dir = s.get('direction')
            new_dir = hdl['direction']
            old_w = s.get('width')
            new_w = str(hdl['width']) if hdl['width'] is not None else str(hdl['width_str'])
            if old_dir != new_dir or (new_w and new_w != 'None' and old_w != new_w):
                changed.append((s['name'], f"dir: {old_dir} -> {new_dir}", f"w: {old_w} -> {new_w}"))
                s['direction'] = new_dir
                s['direction_source'] = 'explicit_hdl'
                if new_w and new_w != 'None':
                    s['width'] = new_w
                    s['width_source'] = 'explicit_hdl'

print(f"Fixed {len(changed)} signals:")
for c in changed:
    print(" ", c)

# Remove the spurious INTERFACE resolution and conflict
ir['resolutions'] = [r for r in ir.get('resolutions', []) if r.get('target_entity') != 'INTERFACE' and r.get('conflict_id') != 'CONF_9a5090d9']
ir['conflicts'] = [c for c in ir.get('conflicts', []) if c.get('title') != 'Plan Conflict: INTERFACE' and c.get('conflict_id') != 'CONF_9a5090d9']

with open('data/efs_ir/active_design_ir.json', 'w', encoding='utf-8') as f:
    json.dump(ir, f, indent=2)

print("Saved active_design_ir.json successfully.")
