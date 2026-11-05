with open("hardware_graphrag/core/agents/spec_classifier.py", "rb") as f:
    content = f.read()

# 1. Update line 284 in classify_spec_type_deterministic
target1 = b'    if has_csr_kw or has_offsets or (len(reg_list) >= 2 and any(k in spec_lower for k in ("reg", "offset", "address", "addr"))):'
replacement1 = b'    if not has_instructions and (has_csr_kw or has_offsets or (len(reg_list) >= 2 and any(k in spec_lower for k in ("reg", "offset", "address", "addr")))):'

if target1 in content:
    content = content.replace(target1, replacement1, 1)

# 2. Update can_use_library_module
target2 = b'''    if any(k == mod_name or mod_name.endswith(k) for k in ("sync_ff", "synchronizer", "cdc_sync")):
        return "synchronizer"

    if any(k == mod_name or mod_name.endswith(k) for k in ("fifo_sync", "sync_fifo", "fifo")):
        return "fifo_sync"

    if any(k == mod_name or mod_name.endswith(k) for k in ("register_bank", "csr_bank", "reg_bank")):
        return "register_bank"'''

replacement2 = b'''    if any(k == mod_name or mod_name.endswith(f"_{k}") for k in ("sync_ff", "synchronizer", "cdc_sync")):
        return "synchronizer"

    if any(k == mod_name or mod_name.endswith(f"_{k}") for k in ("fifo_sync", "sync_fifo")):
        return "fifo_sync"

    if any(k == mod_name or mod_name.endswith(f"_{k}") for k in ("register_bank", "csr_bank", "reg_bank")):
        return "register_bank"'''

if target2 in content:
    content = content.replace(target2, replacement2, 1)

target3 = b'if ("fifo" in spec_lower or "queue" in spec_lower) and any(p in spec_lower for p in ("almost_full", "almost_empty", "wr_ptr", "rd_ptr")):'
replacement3 = b'if re.search(r"\\b(?:fifo|queue)\\b", spec_lower) and any(p in spec_lower for p in ("almost_full", "almost_empty", "wr_ptr", "rd_ptr")):'

if target3 in content:
    content = content.replace(target3, replacement3, 1)

with open("hardware_graphrag/core/agents/spec_classifier.py", "wb") as f:
    f.write(content)

print("Updated spec_classifier.py successfully")

