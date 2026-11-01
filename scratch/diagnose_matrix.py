import json
import os

with open('data/efs_ir/active_design_ir.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

print("=== SIGNALS ===")
for s in d.get('signals', []):
    print(f"  {s.get('signal_id')}: name='{s.get('name')}', dir={s.get('direction')}, width={s.get('width')}, role={s.get('semantic_role')}")

print("\n=== CLOCKS ===")
for c in d.get('clocks', []):
    print(f"  {c.get('name')}")

print("\n=== RESETS ===")
for r in d.get('resets', []):
    print(f"  {r.get('name')}")

print("\n=== FSMS ===")
for fsm in d.get('fsms', []):
    print(f"  FSM: {fsm.get('name')}")
    print(f"  States: {[s.get('name') for s in fsm.get('states', [])]}")
    for t in fsm.get('transitions', []):
        print(f"    {t.get('source_state')} -> {t.get('target_state')}: cond='{t.get('condition')}', type={t.get('condition_type')}, assoc_sig={t.get('associated_signal')}")

print("\n=== OPCODES ===")
for op in d.get('opcodes', []):
    print(f"  {op.get('mnemonic')}: {op.get('binary_encoding')}")

print("\n=== CONFLICTS ===")
for c in d.get('conflicts', []):
    print(f"  [{c.get('conflict_id')}] {c.get('type')}: {c.get('title')}")
