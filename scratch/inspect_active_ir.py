import json
import sys

def main():
    with open('hardware_graphrag/data/efs_ir/active_design_ir.json', 'r', encoding='utf-8') as f:
        ir = json.load(f)

    print('=== ALL SIGNALS ===')
    for s in ir.get('signals', []):
        print(f"{s.get('name')}: dir={s.get('direction')}, width={s.get('width')}, owner={s.get('owner')}, consumer={s.get('consumer')}")

    print('\n=== ALL INTERFACES ===')
    for i in ir.get('interfaces', []):
        print(i)

    print('\n=== MEMORY REGIONS ===')
    print(ir.get('memory_regions'))

    print('\n=== TIMING RULES ===')
    print(ir.get('timing_rules'))

    print('\n=== CONSTRAINTS ===')
    print(ir.get('constraints'))

    print('\n=== PERFORMANCE REQUIREMENTS ===')
    print(ir.get('performance_requirements'))

if __name__ == '__main__':
    main()
