file_path = "hardware_graphrag/core/retrieval/target_resolver.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace in 4b
old_sub_comp_4b = '''                sub_comp = EFSComponent(
                    component_id=f"{c.component_id}_{normalize_target_name(p).upper()}",
                    name=p,
                    type="sub_module",
                    description=f"Sub-module '{p}' derived from compound component '{c.name}'",
                    parent_component_id=c.component_id
                )'''

new_sub_comp_4b = '''                sub_comp = EFSComponent(
                    component_id=f"{c.component_id}_{normalize_target_name(p).upper()}",
                    name=p,
                    type="sub_module",
                    description=f"Sub-module '{p}' derived from compound component '{c.name}'",
                    clock_domain=getattr(c, "clock_domain", "clk"),
                    reset_domain=getattr(c, "reset_domain", "rst_n")
                )'''

# Replace in 7b
old_sub_comp_7b = '''            sub_comp = EFSComponent(
                component_id=f"COMP_{target_norm.upper()}",
                name=raw_target,
                type="sub_module",
                description=f"Sub-module derived from design instruction/command specification: {raw_target}",
                parent_component_id=parent_comp.component_id if parent_comp else None
            )'''

new_sub_comp_7b = '''            sub_comp = EFSComponent(
                component_id=f"COMP_{target_norm.upper()}",
                name=raw_target,
                type="sub_module",
                description=f"Sub-module derived from design instruction/command specification: {raw_target}",
                clock_domain=getattr(parent_comp, "clock_domain", "clk") if parent_comp else "clk",
                reset_domain=getattr(parent_comp, "reset_domain", "rst_n") if parent_comp else "rst_n"
            )'''

content = content.replace(old_sub_comp_4b, new_sub_comp_4b)
content = content.replace(old_sub_comp_7b, new_sub_comp_7b)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Updated EFSComponent instantiation in target_resolver.py")
