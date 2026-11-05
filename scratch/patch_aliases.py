with open('hardware_graphrag/core/verification/conflict_resolution_engine.py', 'r', encoding='utf-8') as f:
    text = f.read()

# For opcodes:
old_opc_check = """        for enc_val, opcs in opc_encodings.items():
            distinct_mnemonics = sorted(list({o.mnemonic for o in opcs if o.mnemonic}))
            if len(distinct_mnemonics) > 1:"""

new_opc_check = """        for enc_val, opcs in opc_encodings.items():
            distinct_mnemonics = sorted(list({o.mnemonic for o in opcs if o.mnemonic}))
            if len(distinct_mnemonics) > 1:
                # Check for intentional instruction aliases
                non_alias_opcs = [
                    o for o in opcs if not getattr(o, "is_alias", False) and not getattr(o, "alias_of", None)
                    and not any(k in (getattr(o, "description", "") or "").lower() for k in ("alias of", "pseudo-instruction", "equivalent to"))
                ]
                if len(non_alias_opcs) <= 1 and len(opcs) > 1:
                    logger.info(f"Skipping duplicate opcode conflict for intentional alias sharing encoding '{enc_val}': {distinct_mnemonics}")
                    continue"""

assert old_opc_check in text, "old_opc_check not found"
text = text.replace(old_opc_check, new_opc_check, 1)

# For instructions:
old_inst_check = """        for enc_val, insts in inst_encodings.items():
            distinct_mnemonics = sorted(list({i.mnemonic for i in insts if i.mnemonic}))
            if len(distinct_mnemonics) > 1:"""

new_inst_check = """        for enc_val, insts in inst_encodings.items():
            distinct_mnemonics = sorted(list({i.mnemonic for i in insts if i.mnemonic}))
            if len(distinct_mnemonics) > 1:
                # Check for intentional instruction aliases
                non_alias_insts = [
                    i for i in insts if not getattr(i, "is_alias", False) and not getattr(i, "alias_of", None)
                    and not any(k in (getattr(i, "description", "") or "").lower() for k in ("alias of", "pseudo-instruction", "equivalent to"))
                ]
                if len(non_alias_insts) <= 1 and len(insts) > 1:
                    logger.info(f"Skipping duplicate opcode conflict for intentional alias sharing opcode '{enc_val}': {distinct_mnemonics}")
                    continue"""

assert old_inst_check in text, "old_inst_check not found"
text = text.replace(old_inst_check, new_inst_check, 1)

with open('hardware_graphrag/core/verification/conflict_resolution_engine.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated alias checking in conflict_resolution_engine.py!")

