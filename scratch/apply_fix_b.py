import re

file_path = "hardware_graphrag/core/retrieval/target_resolver.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update 4b Compound matching
old_4b = '''    # 4b. Compound Canonical Name Sub-Part Match (e.g. "Control Processor / Instruction Decoder" matches "instruction_decoder")
    compound_matches = []
    for c in components:
        parts = [p.strip() for p in re.split(r"[/&|]", c.name) if p.strip()]
        if len(parts) > 1 and any(normalize_target_name(p) == target_norm for p in parts):
            compound_matches.append(c)
    if len(compound_matches) == 1:
        logger.info(f"Target '{truncate_for_log(raw_target, 60)}' resolved via COMPOUND PART match -> {compound_matches[0].name}")
        return TargetResolutionResult(status="SUCCESS", target_component=compound_matches[0], match_method="compound_part")'''

new_4b = '''    # 4b. Compound Canonical Name Sub-Part Match (e.g. "Control Processor and Instruction Decoder" matches "instruction_decoder")
    compound_matches = []
    for c in components:
        parts = [p.strip() for p in re.split(r"[/&|+]|\s+(?:and|&|\+)\s+", c.name, flags=re.IGNORECASE) if p.strip()]
        for p in parts:
            if len(parts) > 1 and normalize_target_name(p) == target_norm:
                sub_comp = EFSComponent(
                    component_id=f"{c.component_id}_{normalize_target_name(p).upper()}",
                    name=p,
                    type="sub_module",
                    description=f"Sub-module '{p}' derived from compound component '{c.name}'",
                    parent_component_id=c.component_id
                )
                compound_matches.append(sub_comp)
                break
    if len(compound_matches) == 1:
        logger.info(f"Target '{truncate_for_log(raw_target, 60)}' resolved via COMPOUND PART match -> {compound_matches[0].name}")
        return TargetResolutionResult(status="SUCCESS", target_component=compound_matches[0], match_method="compound_part")'''

# 2. Add 7b & 7c after FSM check
old_7_end = '''                parent_comp_id = getattr(fsm, "component_id", "")
                parent_comp = next((c for c in components if c.component_id == parent_comp_id), get_top_component())
                logger.info(f"Target '{truncate_for_log(raw_target, 60)}' matched EFS IR FSM '{fsm_name}' -> {parent_comp.name}")
                return TargetResolutionResult(status="SUCCESS", target_component=parent_comp, match_method="sub_entity_fsm")'''

new_7_end = '''                parent_comp_id = getattr(fsm, "component_id", "")
                parent_comp = next((c for c in components if c.component_id == parent_comp_id), get_top_component())
                logger.info(f"Target '{truncate_for_log(raw_target, 60)}' matched EFS IR FSM '{fsm_name}' -> {parent_comp.name}")
                return TargetResolutionResult(status="SUCCESS", target_component=parent_comp, match_method="sub_entity_fsm")

    # 7b. Sub-Entity Contract Derivation: ISA / Command Decoder & Parser
    # If the design defines instruction/opcode contracts and the target is a decoder/parser/dispatcher,
    # resolve to a focused sub_module component linked to the parent control unit / top level.
    if any(k in target_norm for k in ("decoder", "parser", "dispatch", "unpack")):
        has_isa = bool(
            getattr(efs_ir, "opcodes", None) or 
            getattr(efs_ir, "instructions", None) or 
            getattr(efs_ir, "instruction_formats", None)
        )
        if has_isa:
            parent_comp = None
            for c in components:
                c_norm = normalize_target_name(c.name)
                if any(k in c_norm for k in ("controlprocessor", "controlunit", "processor", "core", "controller")):
                    parent_comp = c
                    break
            if not parent_comp:
                parent_comp = get_top_component()

            sub_comp = EFSComponent(
                component_id=f"COMP_{target_norm.upper()}",
                name=raw_target,
                type="sub_module",
                description=f"Sub-module derived from design instruction/command specification: {raw_target}",
                parent_component_id=parent_comp.component_id if parent_comp else None
            )
            logger.info(f"Target '{truncate_for_log(raw_target, 60)}' resolved via ISA CONTRACT SUB-MODULE -> {sub_comp.name} (parent: {parent_comp.name if parent_comp else 'None'})")
            return TargetResolutionResult(
                status="SUCCESS",
                target_component=sub_comp,
                match_method="isa_submodule_contract"
            )

    # 7c. Signal / Register Owner Resolution
    for sig in getattr(efs_ir, "signals", []) or []:
        owner = getattr(sig, "owner", "") or ""
        if owner and normalize_target_name(owner) == target_norm:
            owner_comp = next((c for c in components if normalize_target_name(c.name) == target_norm), None)
            if not owner_comp:
                owner_comp = EFSComponent(
                    component_id=f"COMP_{target_norm.upper()}",
                    name=owner,
                    type="sub_module",
                    description=f"Sub-module resolved from signal ownership: {owner}"
                )
            logger.info(f"Target '{truncate_for_log(raw_target, 60)}' resolved via SIGNAL OWNER match -> {owner_comp.name}")
            return TargetResolutionResult(status="SUCCESS", target_component=owner_comp, match_method="signal_owner")'''

# 3. Step 8b: Guard single_focus_design
old_8b = '''        single_focus_design = len(components) <= 8  # Sub-components of one focused design
        if significant_overlap or single_focus_design:'''

new_8b = '''        if significant_overlap:'''

# Normalize line endings for replacement
norm_content = content.replace("\r\n", "\n")
norm_old_4b = old_4b.replace("\r\n", "\n")
norm_old_7 = old_7_end.replace("\r\n", "\n")
norm_old_8b = old_8b.replace("\r\n", "\n")

assert norm_old_4b in norm_content, "Error: old_4b not found in file"
assert norm_old_7 in norm_content, "Error: old_7_end not found in file"
assert norm_old_8b in norm_content, "Error: old_8b not found in file"

norm_content = norm_content.replace(norm_old_4b, new_4b.replace("\r\n", "\n"), 1)
norm_content = norm_content.replace(norm_old_7, new_7_end.replace("\r\n", "\n"), 1)
norm_content = norm_content.replace(norm_old_8b, new_8b.replace("\r\n", "\n"), 1)

# Preserve original line endings if file was CRLF
if "\r\n" in content:
    final_content = norm_content.replace("\n", "\r\n")
else:
    final_content = norm_content

with open(file_path, "w", encoding="utf-8") as f:
    f.write(final_content)

print("Fix B successfully applied to target_resolver.py")
