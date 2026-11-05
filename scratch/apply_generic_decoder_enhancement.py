import re

rtl_context_path = "hardware_graphrag/core/context_compiler/rtl_context.py"

with open(rtl_context_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace _derive_decoder_bitfield_signals definition
old_fn = '''def _derive_decoder_bitfield_signals(
    design_spec: str,
    efs_ir: Optional[Any] = None,
    target_comp: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Derive scoped sub-module ports for instruction decoders generically from
    specification bitfield tables without hardcoding.
    """
    field_pattern = re.compile(r'^\s*([A-Za-z0-9_\s/]+?)\s*\(\s*(\d+)\s*bits?\s*\)', re.MULTILINE)
    matches = field_pattern.findall(design_spec or "")
    if not matches and efs_ir and hasattr(efs_ir, "instruction_formats"):
        for fmt in getattr(efs_ir, "instruction_formats", []):
            for f in getattr(fmt, "fields", []):
                matches.append((f.name, str(f.width)))

    if not matches:
        return []

    # 1. Collect raw fields
    raw_fields = []
    for raw_name, w_str in matches:
        w = int(w_str)
        name = raw_name.strip()
        clean = re.sub(r'[^a-zA-Z0-9]+', '_', name).strip('_').lower()
        clean = clean.replace('destination_register', 'dest_reg').replace('source_register', 'src_reg')
        clean = clean.replace('first_operand_register', 'src1_reg').replace('second_operand_register', 'src2_reg')
        raw_fields.append((clean, w, name))

    # 2. Filter out composite group headers
    leaf_fields = []
    i = 0
    while i < len(raw_fields):
        name, w, raw = raw_fields[i]
        sub_sum = 0
        j = i + 1
        sub_fields = []
        while j < len(raw_fields) and sub_sum < w:
            sub_sum += raw_fields[j][1]
            sub_fields.append(raw_fields[j])
            j += 1
        if sub_sum == w and len(sub_fields) > 1:
            leaf_fields.extend(sub_fields)
            i = j
        else:
            leaf_fields.append(raw_fields[i])
            i += 1

    total_inst_width = sum(w for _, w, _ in leaf_fields)
    derived: List[Dict[str, Any]] = []

    # Input instruction word
    derived.append({
        "name": "instruction",
        "direction": "input",
        "width": str(total_inst_width if total_inst_width > 0 else 32),
        "semantic_role": "data",
        "description": "Raw instruction word to decode"
    })

    # Start / strobe input
    derived.append({
        "name": "start",
        "direction": "input",
        "width": "1",
        "semantic_role": "control",
        "description": "Decode enable strobe"
    })

    # Output decoded bitfields with big-endian bit slice expressions
    current_msb = total_inst_width - 1
    for name, w, raw in leaf_fields:
        if current_msb >= 0:
            field_msb = current_msb
            field_lsb = max(0, current_msb - w + 1)
            current_msb = field_lsb - 1
            slice_str = f"instruction[{field_msb}]" if w == 1 else f"instruction[{field_msb}:{field_lsb}]"
        else:
            slice_str = f"{w}'d0"

        if name != "reserved" and not any(s["name"] == name for s in derived):
            derived.append({
                "name": name,
                "direction": "output",
                "width": str(w),
                "semantic_role": "control" if "op" in name else "data",
                "description": f"Decoded {raw} field",
                "expression": slice_str
            })

    # Status outputs
    derived.append({
        "name": "done",
        "direction": "output",
        "width": "1",
        "semantic_role": "status",
        "description": "Decode complete flag",
        "expression": "start"
    })
    derived.append({
        "name": "error",
        "direction": "output",
        "width": "1",
        "semantic_role": "error",
        "description": "Illegal instruction or dimension error flag",
        "expression": "1'b0"
    })'''

new_fn = '''def _derive_decoder_bitfield_signals(
    design_spec: str,
    efs_ir: Optional[Any] = None,
    target_comp: Optional[Any] = None,
    context_pack: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Derive scoped sub-module ports for instruction decoders generically from
    specification bitfield tables without hardcoding.
    """
    # 1. Collect all available specification text sources (design_spec, retrieved chunks, EFS descriptions)
    search_texts = [str(design_spec or "")]
    if context_pack:
        for se in getattr(context_pack, "source_evidence", []) or []:
            txt = se.get("text") or se.get("original_text") or ""
            if txt:
                search_texts.append(txt)
        for pe in getattr(context_pack, "protocol_source_evidence", []) or []:
            txt = pe.get("text") or pe.get("original_text") or ""
            if txt:
                search_texts.append(txt)
    if efs_ir and getattr(efs_ir, "metadata", None):
        desc = getattr(efs_ir.metadata, "description", "") or ""
        if desc:
            search_texts.append(desc)

    combined_text = "\\n".join(search_texts)

    field_pattern = re.compile(r'^\\s*([A-Za-z0-9_\\s/]+?)\\s*\\(\\s*(\\d+)\\s*bits?\\s*\\)', re.MULTILINE)
    matches = field_pattern.findall(combined_text)

    # 1b. Support list style: - FieldName: X bits
    if not matches:
        alt_pattern1 = re.compile(r'^\\s*[-*]?\\s*([A-Za-z0-9_\\s/]+?)\\s*[:\\-]\\s*(\\d+)\\s*bits?', re.MULTILINE)
        matches = alt_pattern1.findall(combined_text)

    # 1c. Support markdown table style: | FieldName | X |
    if not matches:
        alt_pattern2 = re.compile(r'\\|\\s*([A-Za-z0-9_\\s/]+?)\\s*\\|\\s*(\\d+)(?:\\s*bits?)?\\s*\\|')
        matches = alt_pattern2.findall(combined_text)

    # 1d. Support EFS IR instruction formats
    if not matches and efs_ir and hasattr(efs_ir, "instruction_formats"):
        for fmt in getattr(efs_ir, "instruction_formats", []):
            for f in getattr(fmt, "fields", []):
                matches.append((f.name, str(f.width)))

    has_opcodes = bool(efs_ir and getattr(efs_ir, "opcodes", None))

    # 1e. Generic Opcode / ISA Fallback when design defines opcodes but no explicit bitfield table
    if not matches and has_opcodes:
        opc_w = 6
        for op in efs_ir.opcodes:
            enc = getattr(op, "binary_encoding", "") or getattr(op, "opcode_hex", "") or ""
            m_enc = re.match(r"^(\\d+)'[bB]", str(enc))
            if m_enc:
                opc_w = int(m_enc.group(1))
                break
        inst_sig = next((s for s in getattr(efs_ir, "signals", []) if "instruction" in getattr(s, "name", "").lower()), None)
        inst_w = int(getattr(inst_sig, "width", 64)) if inst_sig and str(getattr(inst_sig, "width", "")).isdigit() else 64
        matches = [
            ("Opcode", str(opc_w)),
            ("Payload", str(max(1, inst_w - opc_w)))
        ]

    if not matches:
        return []

    # 2. Collect raw fields
    raw_fields = []
    for raw_name, w_str in matches:
        w = int(w_str)
        name = raw_name.strip()
        clean = re.sub(r'[^a-zA-Z0-9]+', '_', name).strip('_').lower()
        clean = clean.replace('destination_register', 'dest_reg').replace('source_register', 'src_reg')
        clean = clean.replace('first_operand_register', 'src1_reg').replace('second_operand_register', 'src2_reg')
        raw_fields.append((clean, w, name))

    # 3. Filter out composite group headers
    leaf_fields = []
    i = 0
    while i < len(raw_fields):
        name, w, raw = raw_fields[i]
        sub_sum = 0
        j = i + 1
        sub_fields = []
        while j < len(raw_fields) and sub_sum < w:
            sub_sum += raw_fields[j][1]
            sub_fields.append(raw_fields[j])
            j += 1
        if sub_sum == w and len(sub_fields) > 1:
            leaf_fields.extend(sub_fields)
            i = j
        else:
            leaf_fields.append(raw_fields[i])
            i += 1

    total_inst_width = sum(w for _, w, _ in leaf_fields)
    derived: List[Dict[str, Any]] = []

    # Input instruction word
    derived.append({
        "name": "instruction",
        "direction": "input",
        "width": str(total_inst_width if total_inst_width > 0 else 64),
        "semantic_role": "data",
        "description": "Raw instruction word to decode"
    })

    # Start / strobe input
    derived.append({
        "name": "start",
        "direction": "input",
        "width": "1",
        "semantic_role": "control",
        "description": "Decode enable strobe"
    })

    # Output decoded bitfields with big-endian bit slice expressions
    current_msb = total_inst_width - 1
    for name, w, raw in leaf_fields:
        if current_msb >= 0:
            field_msb = current_msb
            field_lsb = max(0, current_msb - w + 1)
            current_msb = field_lsb - 1
            slice_str = f"instruction[{field_msb}]" if w == 1 else f"instruction[{field_msb}:{field_lsb}]"
        else:
            slice_str = f"{w}'d0"

        if name not in ("reserved", "payload") and not any(s["name"] == name for s in derived):
            derived.append({
                "name": name,
                "direction": "output",
                "width": str(w),
                "semantic_role": "control" if "op" in name else "data",
                "description": f"Decoded {raw} field",
                "expression": slice_str
            })

    # Add valid_opcode flag if design defines opcodes
    if has_opcodes and not any(s["name"] == "valid_opcode" for s in derived):
        derived.append({
            "name": "valid_opcode",
            "direction": "output",
            "width": "1",
            "semantic_role": "status",
            "description": "Opcode validation flag",
            "expression": None
        })

    # Status outputs
    derived.append({
        "name": "busy",
        "direction": "output",
        "width": "1",
        "semantic_role": "status",
        "description": "Decode busy indicator",
        "expression": "1'b0"
    })
    derived.append({
        "name": "done",
        "direction": "output",
        "width": "1",
        "semantic_role": "status",
        "description": "Decode complete flag",
        "expression": "start && valid_opcode" if has_opcodes else "start"
    })
    derived.append({
        "name": "error",
        "direction": "output",
        "width": "1",
        "semantic_role": "error",
        "description": "Illegal instruction error flag",
        "expression": "start && !valid_opcode" if has_opcodes else "1'b0"
    })'''

# Replace call site
old_call = '''            if not has_decoded_outputs and design_spec:
                derived_ports = _derive_decoder_bitfield_signals(design_spec, efs_ir, target_comp)'''

new_call = '''            if not has_decoded_outputs:
                derived_ports = _derive_decoder_bitfield_signals(design_spec, efs_ir, target_comp, context_pack=context_pack)'''

norm_content = content.replace("\r\n", "\n")
norm_old_fn = old_fn.replace("\r\n", "\n")
norm_old_call = old_call.replace("\r\n", "\n")

assert norm_old_fn in norm_content, "old_fn not found"
assert norm_old_call in norm_content, "old_call not found"

norm_content = norm_content.replace(norm_old_fn, new_fn.replace("\r\n", "\n"), 1)
norm_content = norm_content.replace(norm_old_call, new_call.replace("\r\n", "\n"), 1)

final_content = norm_content.replace("\n", "\r\n") if "\r\n" in content else norm_content

with open(rtl_context_path, "w", encoding="utf-8") as f:
    f.write(final_content)

print("Successfully updated rtl_context.py")

