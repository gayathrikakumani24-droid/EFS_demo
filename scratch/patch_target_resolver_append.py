file_path = "hardware_graphrag/core/retrieval/target_resolver.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Add components.append(sub_comp) in 4b
old_4b = '''                compound_matches.append(sub_comp)
                break'''

new_4b = '''                if not any(comp.component_id == sub_comp.component_id for comp in components):
                    components.append(sub_comp)
                compound_matches.append(sub_comp)
                break'''

# Add components.append(sub_comp) in 7b
old_7b = '''            logger.info(f"Target '{truncate_for_log(raw_target, 60)}' resolved via ISA CONTRACT SUB-MODULE -> {sub_comp.name} (parent: {parent_comp.name if parent_comp else 'None'})")'''

new_7b = '''            if not any(comp.component_id == sub_comp.component_id for comp in components):
                components.append(sub_comp)
            logger.info(f"Target '{truncate_for_log(raw_target, 60)}' resolved via ISA CONTRACT SUB-MODULE -> {sub_comp.name} (parent: {parent_comp.name if parent_comp else 'None'})")'''

content = content.replace(old_4b, new_4b, 1)
content = content.replace(old_7b, new_7b, 1)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Updated target_resolver.py to register derived sub_module in components")
