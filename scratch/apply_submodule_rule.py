"""Apply the Sub-Module Contract Derivation Rule to rtl_context.py, dependency_resolver.py, and rtl_agent.py."""

import re

def patch_rtl_context():
    path = "hardware_graphrag/core/context_compiler/rtl_context.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    crlf = "\r\n" in content
    newline = "\r\n" if crlf else "\n"
    text = content.replace("\r\n", "\n")

    helper_code = '''
def _derive_decoder_bitfield_signals(design_spec: str, efs_ir: Optional[Any] = None) -> List[Dict[str, Any]]:
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

    # Output decoded bitfields
    for name, w, raw in leaf_fields:
        if name != "reserved" and not any(s["name"] == name for s in derived):
            derived.append({
                "name": name,
                "direction": "output",
                "width": str(w),
                "semantic_role": "control" if "op" in name else "data",
                "description": f"Decoded {raw} field"
            })

    # Status outputs
    derived.append({"name": "done", "direction": "output", "width": "1", "semantic_role": "status", "description": "Decode complete flag"})
    derived.append({"name": "error", "direction": "output", "width": "1", "semantic_role": "error", "description": "Illegal instruction or dimension error flag"})

    return derived
'''

    # Add helper_code above RTLContextCompiler class
    target_class = "@register_compiler\nclass RTLContextCompiler(BaseContextCompiler):"
    assert target_class in text, "Could not find RTLContextCompiler class definition"
    text = text.replace(target_class, helper_code.strip() + "\n\n\n" + target_class, 1)

    # Replace scoped_signals logic in compile()
    old_scoped = """        if cp_signals:
            scoped_signals = cp_signals
        elif target_comp:
            scoped_signals = resolver.get_component_signals(target_comp)
        else:
            scoped_signals = efs_ir.signals if efs_ir else []"""

    new_scoped = """        # Check if target is top-level or internal sub-module
        is_target_top_level = (
            not target_comp or
            getattr(target_comp, "type", "").lower() in ("top_level", "top_level_design", "top_module", "system", "container") or
            (efs_ir and getattr(efs_ir, "metadata", None) and (
                (getattr(target_comp, "name", "") or "").lower() == (getattr(efs_ir.metadata, "top_module", None) or "").lower() or
                (getattr(target_comp, "name", "") or "").lower() == (getattr(efs_ir.metadata, "design_name", None) or "").lower()
            ))
        )
        is_decoder_comp = any(k in mod_name.lower() or (target_comp and k in getattr(target_comp, "name", "").lower()) for k in ("decoder", "parser", "dispatch", "unpack"))
        is_bus_comp = any(k in mod_name.lower() or (target_comp and k in getattr(target_comp, "name", "").lower()) for k in ("axi", "dma", "bus", "memory", "mem_iface", "bridge"))

        if cp_signals:
            scoped_signals = cp_signals
        elif target_comp:
            scoped_signals = resolver.get_component_signals(target_comp)
        else:
            scoped_signals = efs_ir.signals if efs_ir else []

        if not is_target_top_level and is_decoder_comp:
            # Sub-Module Contract Derivation for instruction/command decoders
            # 1. Filter out protocol bus signals (AXI, AHB, APB, etc.)
            scoped_signals = [s for s in scoped_signals if not any(
                (s.get("name") if isinstance(s, dict) else getattr(s, "name", "")).lower().startswith(pfx)
                for pfx in ("axi_", "ahb_", "apb_", "wb_", "dma_")
            )]

            # 2. Check if decoded output fields are already present
            has_decoded_outputs = any(
                (s.get("direction") if isinstance(s, dict) else getattr(s, "direction", "")) == "output" and
                (s.get("name") if isinstance(s, dict) else getattr(s, "name", "")).lower() not in ("clk", "rst", "done", "error", "busy")
                for s in scoped_signals
            )
            if not has_decoded_outputs and design_spec:
                derived_ports = _derive_decoder_bitfield_signals(design_spec, efs_ir)
                if derived_ports:
                    # Keep clock/reset from scoped_signals
                    clk_rst = [s for s in scoped_signals if any(k in (s.get("name") if isinstance(s, dict) else getattr(s, "name", "")).lower() for k in ("clk", "rst", "clock", "reset"))]
                    scoped_signals = clk_rst + derived_ports
        elif not is_target_top_level and not is_bus_comp:
            # General bus isolation for computational sub-modules
            scoped_signals = [s for s in scoped_signals if not any(
                (s.get("name") if isinstance(s, dict) else getattr(s, "name", "")).lower().startswith(pfx)
                for pfx in ("axi_", "ahb_", "apb_", "wb_", "dma_")
            )]"""

    assert old_scoped in text, "Could not find old_scoped in rtl_context.py"
    text = text.replace(old_scoped, new_scoped, 1)

    final_content = text.replace("\n", newline)
    with open(path, "w", encoding="utf-8") as f:
        f.write(final_content)
    print("Successfully patched rtl_context.py!")


def patch_dependency_resolver():
    path = "hardware_graphrag/core/retrieval/dependency_resolver.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    crlf = "\r\n" in content
    newline = "\r\n" if crlf else "\n"
    text = content.replace("\r\n", "\n")

    old_res = """    comp_name_lower = target_comp.name.lower()
    comp_type_lower = (getattr(target_comp, "type", "") or "").lower()
    is_decoder = "decoder" in comp_name_lower or "decoder" in comp_type_lower"""

    new_res = """    comp_name_lower = target_comp.name.lower()
    comp_type_lower = (getattr(target_comp, "type", "") or "").lower()
    is_decoder = "decoder" in comp_name_lower or "decoder" in comp_type_lower

    # Sub-Module Bus Isolation Rule: Internal compute/decoder sub-blocks must not inherit external protocol buses
    if not is_top_level and not any(k in comp_name_lower for k in ("axi", "dma", "bus", "memory", "mem_iface", "bridge")):
        signals = [s for s in signals if not any(s.name.lower().startswith(pfx) for pfx in ("axi_", "ahb_", "apb_", "wb_", "dma_"))]"""

    assert old_res in text, "Could not find old_res in dependency_resolver.py"
    text = text.replace(old_res, new_res, 1)

    final_content = text.replace("\n", newline)
    with open(path, "w", encoding="utf-8") as f:
        f.write(final_content)
    print("Successfully patched dependency_resolver.py!")


def patch_rtl_agent():
    path = "hardware_graphrag/core/agents/rtl_agent.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    crlf = "\r\n" in content
    newline = "\r\n" if crlf else "\n"
    text = content.replace("\r\n", "\n")

    old_rule = """3. Do NOT invent signal names or expressions for missing behavior. Set "expression": null and "is_fully_defined": false for incomplete conditions."""
    new_rule = """3. Do NOT invent signal names or expressions for missing behavior. Set "expression": null and "is_fully_defined": false for incomplete conditions.
4. For instruction decoders or command parsers (modules with an 'instruction' or 'cmd' input and decoded field outputs):
   - Map each output field directly to the corresponding sliced bits of the input word in datapath continuous assignments or combinational logic.
   - Do NOT invent multi-cycle memory execution states (such as LOAD, STORE, or CONFIG) unless explicitly defined in the module interface."""

    assert old_rule in text, "Could not find old_rule in rtl_agent.py"
    text = text.replace(old_rule, new_rule, 1)

    final_content = text.replace("\n", newline)
    with open(path, "w", encoding="utf-8") as f:
        f.write(final_content)
    print("Successfully patched rtl_agent.py!")


if __name__ == "__main__":
    patch_rtl_context()
    patch_dependency_resolver()
    patch_rtl_agent()

