with open('hardware_graphrag/data/logs/app.log', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if '2026-10-01 14:' in line:
        if any(keyword in line for keyword in ['BLOCKED', 'ERROR', 'WARNING', 'validate_rtl_plan', 'issues', 'SPECIFICATION_INCOMPLETE', 'RTL Generation', 'plan_rtl', 'validate', 'Target', 'ContextPack']):
            print(f"L{i}: {line.strip()[:160]}")
