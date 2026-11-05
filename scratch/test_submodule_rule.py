"""Test the sub-module derivation rule logic."""

import re
import json

def derive_decoder_ports(spec_text: str, efs_signals: list) -> list:
    """Derive scoped sub-module ports for instruction decoders generically."""
    field_pattern = re.compile(r'^\s*([A-Za-z0-9_\s/]+?)\s*\(\s*(\d+)\s*bits?\s*\)', re.MULTILINE)
    matches = field_pattern.findall(spec_text)

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

    # 3. Build sub-module scoped signals
    derived_signals = []
    
    # Clocks and resets
    for s in efs_signals:
        s_name = s.get("name", "").lower()
        if any(k in s_name for k in ("clk", "clock", "rst", "reset")):
            derived_signals.append(s)

    # Input word (instruction / cmd)
    inst_sig = next((s for s in efs_signals if any(k in s.get("name", "").lower() for k in ("instruction", "cmd", "inst", "word"))), None)
    if inst_sig:
        derived_signals.append(inst_sig)
    else:
        derived_signals.append({"name": "instruction", "direction": "input", "width": "64", "semantic_role": "data"})

    # Control strobe
    start_sig = next((s for s in efs_signals if s.get("name", "").lower() in ("start", "valid", "enable", "inst_valid")), None)
    if start_sig:
        derived_signals.append(start_sig)

    # Output decoded bitfields
    for name, w, raw in leaf_fields:
        if name not in [s.get("name") for s in derived_signals] and name != "reserved":
            derived_signals.append({
                "name": name,
                "direction": "output",
                "width": str(w),
                "semantic_role": "control" if "op" in name else "data",
                "description": f"Decoded {raw} field"
            })

    # Status / Handshake outputs
    derived_signals.append({"name": "done", "direction": "output", "width": "1", "semantic_role": "status"})
    derived_signals.append({"name": "error", "direction": "output", "width": "1", "semantic_role": "error"})

    return derived_signals

sample_spec = '''
3.1 Instruction Format
Each instruction is a 64-bit word with the following fields:
   Opcode    (6 bits): Operation to perform
   Destination Register    (4 bits): Target register (16 matrix registers)
   Source Register 1    (4 bits): First operand register
   Source Register 2    (4 bits): Second operand register
   Dimension Control    (10 bits): Matrix dimension information
Rows Source 1 (5 bits): Number of rows in first source matrix
Cols Source 1/Rows Source 2 (5 bits): Number of columns in first source/rows in second source
   Cols Source 2    (5 bits): Number of columns in second source matrix
   Format    (3 bits): Data format control (fixed-point, floating-point, etc.)
   Flags    (4 bits): Special operation modifiers
   Reserved    (24 bits): For future extensions
'''

efs_sigs = [
    {"name": "clk", "direction": "input", "width": "1"},
    {"name": "rst_n", "direction": "input", "width": "1"},
    {"name": "start", "direction": "input", "width": "1"},
    {"name": "done", "direction": "output", "width": "1"},
    {"name": "error", "direction": "output", "width": "1"},
    {"name": "axi_araddr", "direction": "output", "width": "32"},
    {"name": "axi_wdata", "direction": "output", "width": "512"},
    {"name": "instruction", "direction": "input", "width": "64"},
]

derived = derive_decoder_ports(sample_spec, efs_sigs)
print(f"Total derived ports for sub-module: {len(derived)}")
for s in derived:
    print(f"  {s['direction']:6s} {s['name']:28s} [{s['width']}-1:0]")

# Assertions
port_names = [s["name"] for s in derived]
assert "axi_wdata" not in port_names, "ERROR: axi_wdata leaked into decoder!"
assert "axi_araddr" not in port_names, "ERROR: axi_araddr leaked into decoder!"
assert "opcode" in port_names, "ERROR: opcode missing!"
assert "dest_reg" in port_names, "ERROR: dest_reg missing!"
assert "src_reg_1" in port_names, "ERROR: src_reg_1 missing!"
assert "rows_source_1" in port_names, "ERROR: rows_source_1 missing!"
print("\nALL SUB-MODULE DERIVATION CHECKS PASSED!")

