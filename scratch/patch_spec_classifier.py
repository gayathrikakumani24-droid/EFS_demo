import re

with open('hardware_graphrag/core/agents/spec_classifier.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = """    # 1. FIFO / Memory Check
    fifo_keywords = ["fifo", "queue", "circular buffer", "first-in first-out"]
    has_fifo_kw = any(k in spec_lower for k in fifo_keywords)
    fifo_ports = {"wr_en", "rd_en", "full", "empty", "almost_full", "almost_empty", "wdata", "rdata"}
    fifo_port_matches = len(port_names.intersection(fifo_ports))
    if has_fifo_kw or fifo_port_matches >= 3:
        return SpecType.FIFO_MEMORY, f"Detected FIFO/Memory indicators: keyword match={has_fifo_kw}, port matches={fifo_port_matches}\""""

replacement = """    has_instructions = bool(
        (efs_ir and (getattr(efs_ir, "opcodes", None) or getattr(efs_ir, "instructions", None))) or
        re.search(r"\\b(?:opcode|instruction\\s+decoder|instruction\\s+set|instruction\\s+format|mnemonics?)\\b", spec_lower)
    )

    # 1. FIFO / Memory Check (Guarded against false positives on compute units mentioning queues)
    fifo_keywords = ["fifo", "first-in first-out", "circular buffer"]
    fifo_ports = {"wr_en", "rd_en", "full", "empty", "almost_full", "almost_empty"}
    fifo_port_matches = len(port_names.intersection(fifo_ports))
    is_fifo_named = bool(re.search(r"\\b(?:fifo|async_fifo|sync_fifo)\\b", spec_lower[:300]))
    has_fifo_kw = any(re.search(rf"\\b{re.escape(k)}\\b", spec_lower) for k in fifo_keywords)
    if not has_instructions and (fifo_port_matches >= 2 or (has_fifo_kw and fifo_port_matches >= 1) or is_fifo_named):
        return SpecType.FIFO_MEMORY, f"Detected FIFO/Memory indicators: keyword match={has_fifo_kw}, port matches={fifo_port_matches}\""""

norm_text = text.replace('\r\n', '\n')
norm_target = target.replace('\r\n', '\n')
norm_replacement = replacement.replace('\r\n', '\n')

assert norm_target in norm_text, "Target not found in spec_classifier.py"
new_text = norm_text.replace(norm_target, norm_replacement, 1)

with open('hardware_graphrag/core/agents/spec_classifier.py', 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(new_text)

print("Updated spec_classifier.py successfully!")

