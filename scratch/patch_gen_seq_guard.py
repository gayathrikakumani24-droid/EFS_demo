from pathlib import Path

gen_path = Path("hardware_graphrag/core/agents/constrained_generator.py")
content = gen_path.read_text(encoding="utf-8")

old_code = '''    # Only generate sequential block if clocks/resets exist in ports or reset_requirements
    has_clk = any((p.get("name") if isinstance(p, dict) else getattr(p, "name", "")) == clk_name for p in plan_dict.get("ports", []))
    internal_regs = plan_dict.get("internal_registers", [])

    if has_clk:'''

new_code = '''    # Only generate sequential block if clocks/resets exist in ports or reset_requirements
    has_clk = any((p.get("name") if isinstance(p, dict) else getattr(p, "name", "")) == clk_name for p in plan_dict.get("ports", []))
    internal_regs = plan_dict.get("internal_registers", [])

    if has_clk and (internal_regs or not (plan_dict.get("fsm") or plan_dict.get("fsm_spec"))):'''

if old_code in content:
    content = content.replace(old_code, new_code)
    gen_path.write_text(content, encoding="utf-8")
    print("Updated sequential block guard in constrained_generator.py")
else:
    print("old_code not found")

