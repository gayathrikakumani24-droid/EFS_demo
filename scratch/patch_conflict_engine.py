import re

with open('hardware_graphrag/core/verification/conflict_resolution_engine.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update calculate_concrete_candidates signature and body
old_calc_sig = """    def calculate_concrete_candidates(
        cls,
        conflict: EFSConflict,
        efs_ir: EFSIR,
        count: int = 2
    ) -> List[Tuple[str, str, List[str]]]:"""

new_calc_sig = """    def calculate_concrete_candidates(
        cls,
        conflict: EFSConflict,
        efs_ir: EFSIR,
        count: int = 2,
        reserved_pool: Optional[Set[str]] = None
    ) -> List[Tuple[str, str, List[str]]]:"""

assert old_calc_sig in text, "old_calc_sig not found"
text = text.replace(old_calc_sig, new_calc_sig, 1)

old_assigned_encs = """            assigned_encs = {o.binary_encoding.strip().lower() for o in existing_opcs if o.mnemonic != target_entity}
            assigned_encs.update({i.opcode.strip().lower() for i in existing_insts if i.mnemonic != target_entity})"""

new_assigned_encs = """            assigned_encs = {o.binary_encoding.strip().lower() for o in existing_opcs if o.mnemonic != target_entity}
            assigned_encs.update({i.opcode.strip().lower() for i in existing_insts if i.mnemonic != target_entity})
            if reserved_pool:
                assigned_encs.update({r.lower() for r in reserved_pool})"""

assert old_assigned_encs in text, "old_assigned_encs not found"
text = text.replace(old_assigned_encs, new_assigned_encs, 1)

old_seen_cand = """                        if valid:
                            seen_vals.add(cand_val.lower())
                            prop_name = "binary_encoding" if conflict.type == "DUPLICATE_OPCODE_ENCODING" else "opcode"
                            prop_change = f"{target_entity}.{prop_name} = {cand_val}"
                            results.append((cand_val, prop_change, constraints_checked))
                            if len(results) >= count:
                                return results"""

new_seen_cand = """                        if valid:
                            seen_vals.add(cand_val.lower())
                            if reserved_pool is not None:
                                reserved_pool.add(cand_val.lower())
                                reserved_pool.add(b_str.lower())
                            prop_name = "binary_encoding" if conflict.type == "DUPLICATE_OPCODE_ENCODING" else "opcode"
                            prop_change = f"{target_entity}.{prop_name} = {cand_val}"
                            results.append((cand_val, prop_change, constraints_checked))
                            if len(results) >= count:
                                return results"""

assert old_seen_cand in text, "old_seen_cand not found"
text = text.replace(old_seen_cand, new_seen_cand, 1)

# 2. Update generate_resolution_proposals signature and call to calculate_concrete_candidates
old_gen_sig = """    def generate_resolution_proposals(
        cls,
        conflict: EFSConflict,
        efs_ir: EFSIR,
        context: Optional[Dict[str, Any]] = None
    ) -> List[EFSResolution]:"""

new_gen_sig = """    def generate_resolution_proposals(
        cls,
        conflict: EFSConflict,
        efs_ir: EFSIR,
        context: Optional[Dict[str, Any]] = None,
        reserved_pool: Optional[Set[str]] = None
    ) -> List[EFSResolution]:"""

assert old_gen_sig in text, "old_gen_sig not found"
text = text.replace(old_gen_sig, new_gen_sig, 1)

old_call_calc = """            candidates = cls.calculate_concrete_candidates(conflict, efs_ir, count=2)"""
new_call_calc = """            candidates = cls.calculate_concrete_candidates(conflict, efs_ir, count=2, reserved_pool=reserved_pool)"""

assert old_call_calc in text, "old_call_calc not found"
text = text.replace(old_call_calc, new_call_calc, 1)

# 3. Add generate_batch_resolution_proposals
batch_method = """    @classmethod
    def generate_batch_resolution_proposals(
        cls,
        conflicts: List[EFSConflict],
        efs_ir: EFSIR,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[EFSResolution]]:
        \"\"\"
        Generates resolution proposals across a batch of conflicts while tracking
        a shared reservation pool across proposals, ensuring no duplicate candidates
        are proposed across distinct conflicts (e.g. multiple opcode conflicts).
        \"\"\"
        reserved_encodings: Set[str] = set()
        for op in getattr(efs_ir, "opcodes", []):
            if getattr(op, "binary_encoding", None):
                reserved_encodings.add(op.binary_encoding.strip().lower())
        for inst in getattr(efs_ir, "instructions", []):
            if getattr(inst, "opcode", None):
                reserved_encodings.add(inst.opcode.strip().lower())

        batch_proposals: Dict[str, List[EFSResolution]] = {}
        for conf in conflicts:
            props = cls.generate_resolution_proposals(
                conf, efs_ir, context=context, reserved_pool=reserved_encodings
            )
            batch_proposals[conf.conflict_id] = props
        return batch_proposals

"""

# Insert before generate_resolution_proposals
text = text.replace(new_gen_sig, batch_method + new_gen_sig, 1)

with open('hardware_graphrag/core/verification/conflict_resolution_engine.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated conflict_resolution_engine.py successfully!")

