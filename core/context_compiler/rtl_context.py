"""
RTL Context Compiler module.

Compiles a rich, target-scoped context specifically for RTL generation (SystemVerilog, Verilog, VHDL).
Extracts all required behavioral semantics (ports, clocks, resets, registers, FSM transitions,
conditions, datapath operations, interface handshakes, and scoped protocol rules).

CRITICAL RULE:
Token reduction MUST NOT remove behavioral semantics.
REMOVE IRRELEVANCE, NOT REQUIRED BEHAVIOR.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set
from config import CONFIG
from core.efs_ir.models import EFSIR, EFSComponent, EFSSignal, EFSRegister
from core.efs_ir.validation_rules import normalize_hardware_width
from core.context_compiler.base import (
    BaseContextCompiler, CompilationMetrics, estimate_tokens, register_compiler
)
from core.context_compiler.resolver import ContextResolver
from core.context_compiler.validator import GroundingValidator
from core.context_compiler.budget import TokenBudgetController
from core.agents.spec_classifier import classify_hardware_architecture, ArchitectureClass
from utils.logger import get_logger

logger = get_logger("context_compiler.rtl")


def _derive_decoder_bitfield_signals(
    design_spec: str,
    efs_ir: Optional[Any] = None,
    target_comp: Optional[Any] = None,
    context_pack: Optional[Any] = None,
    retrieval_context: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Derive scoped sub-module ports for instruction decoders generically from
    specification bitfield tables without hardcoding.
    """
    text_sources = []
    if design_spec:
        text_sources.append(design_spec)
    if context_pack:
        ev = getattr(context_pack, "source_evidence", None) or (context_pack.get("source_evidence") if isinstance(context_pack, dict) else [])
        for item in (ev or []):
            txt = item.get("content") or item.get("text") if isinstance(item, dict) else getattr(item, "text", getattr(item, "content", ""))
            if txt:
                text_sources.append(txt)
        freq = getattr(context_pack, "functional_requirements", None) or (context_pack.get("functional_requirements") if isinstance(context_pack, dict) else [])
        for item in (freq or []):
            txt = item.get("description") or item.get("text") if isinstance(item, dict) else getattr(item, "description", "")
            if txt:
                text_sources.append(txt)
        task_desc = getattr(getattr(context_pack, "task", None), "description", None) or (context_pack.get("task", {}).get("description") if isinstance(context_pack, dict) else None)
        if task_desc:
            text_sources.append(task_desc)
    if retrieval_context:
        if isinstance(retrieval_context, str):
            text_sources.append(retrieval_context)
        elif isinstance(retrieval_context, list):
            for item in retrieval_context:
                txt = item.get("content") or item.get("text") if isinstance(item, dict) else str(item)
                if txt:
                    text_sources.append(txt)
    spec_text = "\n\n".join(text_sources)

    field_pattern = re.compile(r'^\s*[-*•]?\s*([A-Za-z0-9_\s/]+?)\s*\(\s*(\d+)\s*bits?\s*\)', re.MULTILINE)
    matches = field_pattern.findall(spec_text)

    if not matches:
        pipe_bits_pattern = re.compile(r'\|\s*([A-Za-z0-9_\s/]+?)\s*\|\s*(\d+)\s*(?:bits?)?\s*\|', re.MULTILINE)
        matches = [(m[0].strip(), m[1].strip()) for m in pipe_bits_pattern.findall(spec_text) if m[0].strip().lower() not in ("field", "name", "parameter", "signal")]

    if not matches and efs_ir and hasattr(efs_ir, "instruction_formats"):
        for fmt in getattr(efs_ir, "instruction_formats", []):
            for f in getattr(fmt, "fields", []):
                matches.append((f.name, str(f.width)))

    if not matches and efs_ir and (getattr(efs_ir, "opcodes", None) or getattr(efs_ir, "instructions", None)):
        opc_list = getattr(efs_ir, "opcodes", []) or []
        opc_w = 6
        if opc_list:
            max_enc = max((len(str(getattr(o, "binary_encoding", getattr(o, "encoding", ""))).replace("_", "").strip()) for o in opc_list), default=6)
            if max_enc > 0:
                opc_w = max_enc
        matches = [
            ("opcode", str(opc_w)),
            ("dest_reg", "4"),
            ("src_reg_1", "4"),
            ("src_reg_2", "4"),
        ]

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
    has_opcode_field = False
    for name, w, raw in leaf_fields:
        if "opcode" in name or name == "op":
            has_opcode_field = True
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

    # If opcodes are present in IR or fields, ensure valid_opcode is exposed as an output
    if has_opcode_field or (efs_ir and getattr(efs_ir, "opcodes", None)):
        if not any(s["name"] == "valid_opcode" for s in derived):
            derived.append({
                "name": "valid_opcode",
                "direction": "output",
                "width": "1",
                "semantic_role": "status",
                "description": "Valid decoded opcode indicator"
            })

    # Status outputs
    if not any(s["name"] == "done" for s in derived):
        derived.append({
            "name": "done",
            "direction": "output",
            "width": "1",
            "semantic_role": "status",
            "description": "Decode complete flag",
            "expression": "done_reg"
        })
    if not any(s["name"] == "error" for s in derived):
        derived.append({
            "name": "error",
            "direction": "output",
            "width": "1",
            "semantic_role": "error",
            "description": "Illegal instruction or dimension error flag",
            "expression": "error_reg"
        })

    # Register derived signals into canonical EFS IR for grounding validity
    if efs_ir and hasattr(efs_ir, "signals") and isinstance(efs_ir.signals, list):
        existing_sig_names = {s.name.lower() for s in efs_ir.signals if getattr(s, "name", None)}
        owner_id = getattr(target_comp, "component_id", None) if target_comp else None
        for item in derived:
            s_name = item["name"]
            if s_name.lower() not in existing_sig_names:
                new_sig = EFSSignal(
                    name=s_name,
                    direction=item["direction"],
                    width=item["width"],
                    semantic_role=item.get("semantic_role", "data"),
                    owner=owner_id,
                    description=item.get("description", "")
                )
                efs_ir.signals.append(new_sig)
                existing_sig_names.add(s_name.lower())

    return derived


@register_compiler
class RTLContextCompiler(BaseContextCompiler):
    """Compiles a compact, rich behavioral context tailored strictly to the target RTL module."""

    artifact_types = ["rtl", "verilog", "systemverilog", "vhdl"]

    def compile(
        self,
        efs_ir: Optional[EFSIR] = None,
        design_plan: Optional[Dict[str, Any]] = None,
        protocol_contract: Optional[Any] = None,
        retrieval_context: Optional[Any] = None,
        token_budget: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        max_budget = token_budget or getattr(
            getattr(CONFIG, "context_compiler", None),
            "rtl_token_budget",
            8000
        )

        metrics = CompilationMetrics(
            artifact_type="rtl",
            compiler_used="RTLContextCompiler"
        )

        # 1. Target Component Identification
        target_name = kwargs.get("target_component") or kwargs.get("target") or ""
        context_pack = kwargs.get("context_pack")
        if (not target_name or target_name in ("Verilog", "SystemVerilog", "VHDL")) and context_pack:
            tc = getattr(context_pack, "target_component", {})
            if isinstance(tc, dict) and tc.get("name"):
                target_name = tc.get("name")
            elif hasattr(tc, "name") and getattr(tc, "name"):
                target_name = tc.name
            elif hasattr(context_pack, "task") and getattr(context_pack.task, "target", None):
                target_name = context_pack.task.target

        if not target_name and design_plan and isinstance(design_plan, dict):
            target_name = design_plan.get("target") or design_plan.get("module_name") or ""
        if not target_name and efs_ir and efs_ir.metadata and efs_ir.metadata.design_name:
            target_name = efs_ir.metadata.design_name

        resolver = ContextResolver(efs_ir)

        target_comp: Optional[EFSComponent] = None
        if efs_ir and efs_ir.components:
            def _norm_cname(s: str) -> str:
                return re.sub(r"[\s_-]+", "", s.lower()) if s else ""

            if target_name:
                t_norm = _norm_cname(target_name)
                target_comp = next((c for c in efs_ir.components if _norm_cname(c.name) == t_norm or c.component_id == target_name), None)
                if not target_comp and getattr(efs_ir, "metadata", None):
                    aliases = [_norm_cname(a) for a in getattr(efs_ir.metadata, "aliases", [])]
                    if t_norm in aliases:
                        target_comp = next((c for c in efs_ir.components if _norm_cname(c.name) in aliases), None)

            if not target_comp:
                d_norm = _norm_cname(getattr(efs_ir.metadata, "design_name", "") or getattr(efs_ir.metadata, "top_module", ""))
                target_comp = next((c for c in efs_ir.components if d_norm and _norm_cname(c.name) == d_norm), None)

            if not target_comp:
                # Default to top-level or primary design component
                target_comp = next((c for c in efs_ir.components if c.type in ("top_level", "top_level_design", "top_module", "core_module")), None)

            if not target_comp:
                # Fallback to component with the most owned signals or primary FSM
                target_comp = max(
                    efs_ir.components,
                    key=lambda c: len(resolver.get_component_signals(c)) + (20 if resolver.get_component_fsm(c) else 0)
                )

        raw_name = None
        if target_comp and target_comp.name:
            raw_name = target_comp.name
        elif target_name and target_name.lower() not in ("verilog", "systemverilog", "vhdl", "uvm", "assertions", "custom_module", "target_module", "hardware_module"):
            raw_name = target_name
        else:
            design_spec_arg = kwargs.get("design_spec", "")
            if design_spec_arg:
                m_match = re.search(r"#\s*Hardware Design Specification:\s*([A-Za-z0-9_\-\s]+)", design_spec_arg)
                if not m_match:
                    m_match = re.search(r"module\s+([a-zA-Z0-9_]+)", design_spec_arg, re.IGNORECASE)
                if not m_match:
                    m_match = re.search(r"#\s*([A-Za-z0-9_\-\s]+)", design_spec_arg)
                if m_match:
                    raw_name = m_match.group(1).strip().lower().replace(" ", "_").replace("-", "_")
        clean_name = re.sub(r"[^a-zA-Z0-9_]+", "_", (raw_name or "hardware_module").strip()).strip("_")
        stop_words = {"builds", "upon", "the", "original", "while", "adding", "for", "and", "with", "a", "an", "in", "to", "of", "from"}
        tokens = [t for t in clean_name.split("_") if t and t.lower() not in stop_words]
        if len(clean_name) > 40 or len(clean_name.split("_")) > 5:
            mod_name = "_".join(tokens[:4]) if tokens else "_".join(clean_name.split("_")[:3])
        else:
            mod_name = clean_name
        if not mod_name or mod_name[0].isdigit():
            mod_name = f"mod_{mod_name}" if mod_name else "hardware_module"

        # Measure original context size
        orig_tokens = 0
        if efs_ir:
            orig_tokens += estimate_tokens(efs_ir)
        if design_plan:
            orig_tokens += estimate_tokens(design_plan)
        design_spec = kwargs.get("design_spec", "")
        if design_spec:
            orig_tokens += estimate_tokens(design_spec)
        metrics.original_context_tokens = max(orig_tokens, 1)

        # 2. Extract Ports and Directly Relevant Signals
        ports: List[Dict[str, Any]] = []
        signals: List[Dict[str, Any]] = []
        clock_reset: List[Dict[str, Any]] = []
        seen_port_names: Set[str] = set()

        cp = kwargs.get("context_pack")
        cp_signals = []
        cp_registers = []
        cp_fsm = None
        if cp:
            cp_signals = getattr(cp, "signals", None) or (cp.get("signals") if isinstance(cp, dict) else []) or (getattr(cp, "target_signals", None) or (cp.get("target_signals") if isinstance(cp, dict) else []))
            cp_registers = getattr(cp, "registers", None) or (cp.get("registers") if isinstance(cp, dict) else []) or (getattr(cp, "target_registers", None) or (cp.get("target_registers") if isinstance(cp, dict) else []))
            cp_fsm = getattr(cp, "fsm", None) or (cp.get("fsm") if isinstance(cp, dict) else None) or (getattr(cp, "target_fsm", None) or (cp.get("target_fsm") if isinstance(cp, dict) else None))

        # Check if target is top-level or internal sub-module
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
            if not has_decoded_outputs and (design_spec or cp or retrieval_context or (efs_ir and (getattr(efs_ir, "opcodes", None) or getattr(efs_ir, "instructions", None)))):
                derived_ports = _derive_decoder_bitfield_signals(
                    design_spec=design_spec,
                    efs_ir=efs_ir,
                    target_comp=target_comp,
                    context_pack=cp,
                    retrieval_context=retrieval_context
                )
                if derived_ports:
                    # Keep clock/reset from scoped_signals
                    clk_rst = [s for s in (scoped_signals if any(any(k in (x.get("name") if isinstance(x, dict) else getattr(x, "name", "")).lower() for k in ("clk", "rst", "clock", "reset")) for x in scoped_signals) else (efs_ir.signals if efs_ir else [])) if any(k in (s.get("name") if isinstance(s, dict) else getattr(s, "name", "")).lower() for k in ("clk", "rst", "clock", "reset"))]
                    scoped_signals = clk_rst + derived_ports
        elif not is_target_top_level and not is_bus_comp:
            # General bus isolation for computational sub-modules
            scoped_signals = [s for s in scoped_signals if not any(
                (s.get("name") if isinstance(s, dict) else getattr(s, "name", "")).lower().startswith(pfx)
                for pfx in ("axi_", "ahb_", "apb_", "wb_", "dma_")
            )]

        for s in scoped_signals:
            s_name = s.get("name") if isinstance(s, dict) else getattr(s, "name", None)
            if not s_name:
                s_id = s.get("signal_id") if isinstance(s, dict) else getattr(s, "signal_id", "")
                s_name = resolver.resolve_signal_name(s_id)
            if not s_name or s_name.lower() in seen_port_names:
                continue
            seen_port_names.add(s_name.lower())

            direction = s.get("direction", "input") if isinstance(s, dict) else (s.direction or "input")
            raw_w = s.get("width") if isinstance(s, dict) else getattr(s, "width", None)
            norm_w = normalize_hardware_width(raw_w)
            if norm_w is not None:
                width = str(norm_w)
                width_provenance = "explicit_numeric"
            elif isinstance(raw_w, str) and raw_w.strip() and raw_w.strip().lower() not in ("none", "null", "undefined", "tbd"):
                width = raw_w.strip()
                width_provenance = "explicit_expression"
            else:
                # Trace parameter or protocol width: check if signal name suggests AXI / bus width
                is_axi_addr = any(k in s_name.lower() for k in ("awaddr", "araddr"))
                is_axi_data = any(k in s_name.lower() for k in ("wdata", "rdata"))
                is_axi_resp = any(k in s_name.lower() for k in ("bresp", "rresp"))
                is_axi_strb = any(k in s_name.lower() for k in ("wstrb",))
                if is_axi_addr:
                    width = "32"
                    width_provenance = "derived_protocol_contract"
                elif is_axi_data:
                    width = "32"
                    width_provenance = "derived_protocol_contract"
                elif is_axi_resp:
                    width = "2"
                    width_provenance = "derived_protocol_contract"
                elif is_axi_strb:
                    width = "4"
                    width_provenance = "derived_protocol_contract"
                elif any(b in s_name.lower() for b in ("addr", "data", "result", "matrix", "vector", "payload")):
                    width = "32"
                    width_provenance = "inferred_bus_width"
                else:
                    width = "1"
                    width_provenance = "inferred_scalar"

            role = s.get("semantic_role", "data") if isinstance(s, dict) else (s.semantic_role or "data")
            signedness = bool(s.get("signedness", False) if isinstance(s, dict) else getattr(s, "signedness", False))
            proto_role = s.get("protocol_role") if isinstance(s, dict) else getattr(s, "protocol_role", None)
            s_clk = s.get("clock") if isinstance(s, dict) else getattr(s, "clock", None)
            s_rst = s.get("reset") if isinstance(s, dict) else getattr(s, "reset", None)
            s_desc = s.get("description", "") if isinstance(s, dict) else getattr(s, "description", "")
            s_iface = s.get("interface", "") if isinstance(s, dict) else getattr(s, "interface", "")

            is_clk_rst = role in ("clock", "reset") or any(k in s_name.lower() for k in ("clk", "rst", "clock", "reset"))
            if is_clk_rst:
                is_clk = "clk" in s_name.lower() or "clock" in s_name.lower() or role == "clock"
                pol = "rising" if is_clk else ("active_low" if any(k in s_name.lower() for k in ("_n", "rst_n", "reset_n", "low")) else "active_high")
                clock_reset.append({
                    "name": s_name,
                    "type": "clock" if is_clk else "reset",
                    "direction": direction,
                    "polarity": pol
                })

            port_item = {
                "name": s_name,
                "direction": direction,
                "width": width,
                "width_provenance": width_provenance,
                "signedness": signedness,
                "semantic_role": role,
                "protocol_role": proto_role,
                "clock": s_clk,
                "reset": s_rst,
                "description": (s_desc or "")[:80],
                "interface": resolver.resolve_interface_name(s_iface or "")
            }
            expr = s.get("expression") if isinstance(s, dict) else getattr(s, "expression", None)
            if expr:
                port_item["expression"] = expr
            ports.append(port_item)
            signals.append(port_item)

        # Architecture classification derivation
        arch_res = classify_hardware_architecture(efs_ir=efs_ir, spec_text=kwargs.get("design_spec", ""))
        is_pure_comb = (arch_res.primary_class == ArchitectureClass.COMBINATIONAL or (not arch_res.has_clock and not arch_res.has_reset))

        # If ports list is empty and design is sequential, synthesize standard clock/reset safely
        if not ports:
            if not is_pure_comb:
                ports = [
                    {"name": "clk", "direction": "input", "width": "1", "semantic_role": "clock", "signedness": False},
                    {"name": "rst_n", "direction": "input", "width": "1", "semantic_role": "reset", "signedness": False}
                ]
                clock_reset = [
                    {"name": "clk", "type": "clock", "direction": "input", "polarity": "rising"},
                    {"name": "rst_n", "type": "reset", "direction": "input", "polarity": "active_low"}
                ]
        elif is_pure_comb:
            # For pure combinational circuits, ensure no spurious clock or reset is injected
            clock_reset = []

        # 3. Extract Registers & Fields (Preserve All Explicit Bit Meanings)
        registers: List[Dict[str, Any]] = []
        if cp_registers:
            scoped_registers = cp_registers
        elif target_comp:
            scoped_registers = resolver.get_component_registers(target_comp)
        else:
            scoped_registers = efs_ir.registers if efs_ir else []

        for r in scoped_registers:
            r_name = r.get("name") if isinstance(r, dict) else resolver.id_to_name.get(r.register_id, r.name)
            raw_rw = r.get("width") if isinstance(r, dict) else getattr(r, "width", None)
            norm_rw = normalize_hardware_width(raw_rw)
            r_width = str(norm_rw if norm_rw is not None else (raw_rw or 32))
            r_offset = r.get("offset", "0x0") if isinstance(r, dict) else (r.offset or "0x0")
            raw_reset = r.get("reset_val", r.get("reset_value")) if isinstance(r, dict) else (getattr(r, "reset_val", None) or getattr(r, "reset_value", None))
            raw_flds = r.get("fields", []) if isinstance(r, dict) else (getattr(r, "fields", []) or [])
            if raw_reset is not None and str(raw_reset).strip() not in ("", "None", "undefined"):
                r_reset = str(raw_reset)
                reset_source = "explicit"
            elif raw_flds and any((fld.get("reset_value") if isinstance(fld, dict) else getattr(fld, "reset_value", None)) is not None for fld in raw_flds):
                r_reset = "32'h0"
                reset_source = "derived_from_fields"
            else:
                r_reset = f"{r_width}'h0" if r_width.isdigit() else "32'h0"
                reset_source = "default_zero"
            r_access = r.get("access", "RW") if isinstance(r, dict) else getattr(r, "access", "RW")
            r_desc = r.get("description", "") if isinstance(r, dict) else (r.description or "")

            fields: List[Dict[str, Any]] = []
            for fld in raw_flds:
                f_name = fld.get("name") if isinstance(fld, dict) else fld.name
                msb = fld.get("msb", 0) if isinstance(fld, dict) else (fld.msb if fld.msb is not None else 0)
                lsb = fld.get("lsb", 0) if isinstance(fld, dict) else (fld.lsb if fld.lsb is not None else 0)
                bit_range = f"{msb}:{lsb}" if msb != lsb else str(msb)
                fields.append({
                    "name": f_name,
                    "bits": bit_range,
                    "meaning": (fld.get("meaning") or fld.get("description") or f_name) if isinstance(fld, dict) else (fld.description or fld.name),
                    "reset_value": str(fld.get("reset_value", "0") if isinstance(fld, dict) else getattr(fld, "reset_value", "0")),
                    "source": "EFS"
                })

            registers.append({
                "name": r_name,
                "offset": r_offset,
                "width": r_width,
                "reset_value": r_reset,
                "reset_source": reset_source,
                "access": r_access,
                "description": (r_desc or "")[:80],
                "fields": fields
            })

        from core.agents.rtl_agent import sanitize_hardware_registers
        registers = sanitize_hardware_registers(registers)

        if not is_target_top_level and is_decoder_comp:
            owner_id = getattr(target_comp, "component_id", None) if target_comp else None
            existing_reg_names = {getattr(r, "name", "").lower() for r in (getattr(efs_ir, "registers", []) or [])} if efs_ir else set()
            if any(p.get("name") == "done" for p in ports) and not any(r.get("name") == "done_reg" for r in registers):
                registers.append({
                    "name": "done_reg",
                    "width": "1",
                    "reset_value": "1'b0",
                    "reset_source": "handshake_init",
                    "access": "RO",
                    "description": "Registered decode completion flag",
                    "fields": []
                })
                if efs_ir and hasattr(efs_ir, "registers") and isinstance(efs_ir.registers, list) and "done_reg" not in existing_reg_names:
                    efs_ir.registers.append(EFSRegister(name="done_reg", width="1", reset_value="0", owner=owner_id, description="Registered decode completion flag"))
                    existing_reg_names.add("done_reg")
            if any(p.get("name") == "error" for p in ports) and not any(r.get("name") == "error_reg" for r in registers):
                registers.append({
                    "name": "error_reg",
                    "width": "1",
                    "reset_value": "1'b0",
                    "reset_source": "handshake_init",
                    "access": "RO",
                    "description": "Registered illegal opcode or decode error flag",
                    "fields": []
                })
                if efs_ir and hasattr(efs_ir, "registers") and isinstance(efs_ir.registers, list) and "error_reg" not in existing_reg_names:
                    efs_ir.registers.append(EFSRegister(name="error_reg", width="1", reset_value="0", owner=owner_id, description="Registered illegal opcode or decode error flag"))
                    existing_reg_names.add("error_reg")

        # 4. Extract FSM (States, Transitions, Conditions, Actions, Alternate Branches, Timeouts)
        fsm_data: Dict[str, Any] = {}
        target_fsm = None
        if cp_fsm:
            if isinstance(cp_fsm, dict):
                fsm_data = dict(cp_fsm)
            else:
                target_fsm = cp_fsm
        else:
            target_fsm = resolver.get_component_fsm(target_comp) if target_comp else None

        if not fsm_data and target_fsm and not isinstance(target_fsm, dict):
            states_list = [st.name if hasattr(st, "name") else str(st) for st in (getattr(target_fsm, "states", []) or [])]
            transitions_list: List[Dict[str, Any]] = []
            
            # Count outgoing transitions per source state to detect alternate branches
            out_degrees: Dict[str, int] = {}
            for tr in (getattr(target_fsm, "transitions", []) or []):
                s_st = getattr(tr, "source_state", "")
                out_degrees[s_st] = out_degrees.get(s_st, 0) + 1

            for tr in (getattr(target_fsm, "transitions", []) or []):
                cond = getattr(tr, "condition", None)
                cond_raw = cond.get("expression") if isinstance(cond, dict) else (str(cond) if cond is not None else "")
                cond_clean = resolver.scrub_internal_ids(cond_raw)
                action_clean = resolver.scrub_internal_ids(getattr(tr, "action", "") or "")
                
                is_dwell = bool(re.search(r"\b(?:next\s+clock\s+cycle|after\s+1\s+cycle|single[- ]cycle\s+dwell|unconditional)\b", cond_clean.lower()))
                if is_dwell:
                    norm_expr = "1'b1"
                    cond_source = "explicit_dwell"
                    is_defined = True
                elif cond_clean in ("1'b1", "1", "true", "True"):
                    has_evid = bool(getattr(tr, "evidence", "") or getattr(tr, "source_evidence", ""))
                    norm_expr = "1'b1"
                    cond_source = "explicit_unconditional" if has_evid else "generic_assumed"
                    is_defined = True
                elif cond_clean and cond_clean.lower() not in ("none", "null", "undefined", "tbd", "unspecified", ""):
                    norm_expr = cond_clean
                    cond_source = "explicit_condition"
                    is_defined = True
                else:
                    norm_expr = None
                    cond_source = "missing"
                    is_defined = False
                
                tr_data: Dict[str, Any] = {
                    "from": getattr(tr, "source_state", ""),
                    "to": getattr(tr, "target_state", ""),
                    "condition": {
                        "expression": norm_expr,
                        "signals": [resolver.resolve_signal_name(s) for s in getattr(tr, "signals", [])] if not is_dwell else [],
                        "source": cond_source,
                        "source_evidence": cond_clean,
                        "is_fully_defined": is_defined
                    }
                }
                if action_clean:
                    tr_data["action"] = action_clean
                if getattr(tr, "priority", 0):
                    tr_data["priority"] = tr.priority
                if out_degrees.get(getattr(tr, "source_state", ""), 0) > 1:
                    tr_data["is_alternate_branch"] = True

                transitions_list.append(tr_data)

            fsm_data = {
                "name": getattr(target_fsm, "name", "") or "Primary_FSM",
                "initial_state": getattr(target_fsm, "initial_state", "") or (states_list[0] if states_list else "IDLE"),
                "states": states_list,
                "transitions": transitions_list,
                "has_alternate_branches": any(deg > 1 for deg in out_degrees.values())
            }
        elif not fsm_data and not cp and efs_ir and efs_ir.fsms:
            fallback_fsm = efs_ir.fsms[0]
            st_list = [st.name if hasattr(st, "name") else str(st) for st in (fallback_fsm.states or [])]
            tr_list = []
            for tr in (fallback_fsm.transitions or []):
                cond = tr.condition
                cond_raw = cond.get("expression") if isinstance(cond, dict) else (str(cond) if cond is not None else "")
                cond_clean = resolver.scrub_internal_ids(cond_raw)
                is_dwell = bool(re.search(r"\b(?:next\s+clock\s+cycle|after\s+1\s+cycle|single[- ]cycle\s+dwell|unconditional)\b", cond_clean.lower()))
                if is_dwell:
                    norm_expr = "1'b1"
                    cond_source = "explicit_dwell"
                    is_defined = True
                elif cond_clean in ("1'b1", "1", "true", "True"):
                    has_evid = bool(getattr(tr, "evidence", "") or getattr(tr, "source_evidence", ""))
                    norm_expr = "1'b1"
                    cond_source = "explicit_unconditional" if has_evid else "generic_assumed"
                    is_defined = True
                elif cond_clean and cond_clean.lower() not in ("none", "null", "undefined", "tbd", "unspecified", ""):
                    norm_expr = cond_clean
                    cond_source = "explicit_condition"
                    is_defined = True
                else:
                    norm_expr = None
                    cond_source = "missing"
                    is_defined = False

                tr_list.append({
                    "from": tr.source_state,
                    "to": tr.target_state,
                    "condition": {
                        "expression": norm_expr,
                        "source": cond_source,
                        "source_evidence": cond_clean,
                        "is_fully_defined": is_defined
                    }
                })
            fsm_data = {
                "name": fallback_fsm.name or "Primary_FSM",
                "initial_state": fallback_fsm.initial_state or (st_list[0] if st_list else "IDLE"),
                "states": st_list,
                "transitions": tr_list,
                "has_alternate_branches": False
            }

        # 5. Dependency-Based Semantic Relevance Selection
        # (Preserves opcodes, instructions, commands, transactions, handshakes, memory ops, timeouts, parameters)
        from core.context_compiler.semantic_selector import SemanticDependencySelector
        sem_selector = SemanticDependencySelector(efs_ir, resolver)
        sem_deps = sem_selector.select_dependencies(
            target_comp=target_comp,
            artifact_type="rtl",
            scoped_signals=scoped_signals,
            target_fsm=target_fsm
        )

        # Merge datapath operations from functional behavior
        datapath_operations: List[Dict[str, Any]] = list(sem_deps.datapath_operations)
        if efs_ir and hasattr(efs_ir, "functional_behavior") and efs_ir.functional_behavior:
            for fb in efs_ir.functional_behavior:
                subj = getattr(fb, "subject", "") or getattr(fb, "target_component", "")
                target_match = not subj or subj.lower() == mod_name.lower()
                if target_match:
                    desc = getattr(fb, "description", "")
                    expr = getattr(fb, "action", "") or getattr(fb, "expression", "") or getattr(fb, "behavior", "") or desc
                    dp_item = {
                        "name": getattr(fb, "name", "op"),
                        "description": resolver.scrub_internal_ids(desc),
                        "expression": resolver.scrub_internal_ids(expr)
                    }
                    if not any(d.get("name") == dp_item["name"] for d in datapath_operations):
                        datapath_operations.append(dp_item)

        # 6. Sliced Protocol Rules (strictly relevant to this module's interfaces)
        target_interfaces = set()
        if target_comp:
            for iface_ref in (target_comp.interfaces or []):
                target_interfaces.add(iface_ref.lower())
                i_obj = resolver.interface_by_id.get(iface_ref) or resolver.interface_by_name.get(iface_ref.lower())
                if i_obj:
                    if i_obj.name:
                        target_interfaces.add(i_obj.name.lower())
                    if i_obj.protocol:
                        target_interfaces.add(i_obj.protocol.lower())

        relevant_protocol_rules: List[Dict[str, Any]] = []
        all_rules = []
        if isinstance(protocol_contract, list):
            all_rules.extend(protocol_contract)
        elif isinstance(protocol_contract, dict):
            all_rules.extend(protocol_contract.get("rules", []))
        elif efs_ir and hasattr(efs_ir, "protocol_rules"):
            all_rules.extend([r.to_dict() if hasattr(r, "to_dict") else r for r in (efs_ir.protocol_rules or [])])

        seen_rule_texts = set()
        for rule in all_rules:
            r_name = rule.get("name", "")
            r_text = rule.get("text", "") or rule.get("rule_statement", "") or rule.get("description", "") or r_name
            r_citation = rule.get("citation", "")
            r_iface = (rule.get("interface", "") or rule.get("protocol", "")).lower()

            # Match on interface protocol names, rule names, or port signal stems
            port_stems = []
            for p in ports:
                p_raw = p["name"].lower()
                port_stems.append(p_raw)
                for pfx in ("m_axi_", "s_axi_", "axi_", "m_", "s_", "in_", "out_"):
                    if p_raw.startswith(pfx):
                        port_stems.append(p_raw[len(pfx):])

            is_relevant = (
                not target_interfaces or
                any(ti in r_iface for ti in target_interfaces) or
                any(ti in r_name.lower() for ti in target_interfaces) or
                any(ti in r_text.lower() for ti in target_interfaces) or
                any(stem in r_text.lower() for stem in port_stems if len(stem) > 3)
            )

            if is_relevant and r_text and r_text not in seen_rule_texts:
                seen_rule_texts.add(r_text)
                relevant_protocol_rules.append({
                    "text": resolver.scrub_internal_ids(r_text),
                    "citation": r_citation
                })

        # 7. Sliced Design Plan (Module-specific implementation constraints)
        filtered_plan = {}
        if design_plan and isinstance(design_plan, dict):
            for k in ("architecture_overview", "datapath_strategy", "reset_polarity", "fsm", "parameters"):
                if k in design_plan:
                    filtered_plan[k] = design_plan[k]

        # 8. Source Traceability References
        source_refs: List[str] = []
        if target_comp:
            source_refs.extend(resolver.get_source_refs(target_comp))
        for p in scoped_signals:
            source_refs.extend(resolver.get_source_refs(p))

        # Merge parameters from EFS IR if available
        all_params = list(sem_deps.parameters)
        if efs_ir and hasattr(efs_ir, "parameters") and efs_ir.parameters:
            for param in efs_ir.parameters:
                p_dict = param.to_dict() if hasattr(param, "to_dict") else vars(param)
                if not any(p.get("name") == p_dict.get("name") for p in all_params):
                    all_params.append(p_dict)

        # Collect approved HITL resolutions
        approved_resolutions: List[Dict[str, Any]] = []
        if efs_ir and hasattr(efs_ir, "resolutions"):
            for res in (efs_ir.resolutions or []):
                if getattr(res, "approval_status", "") == "APPROVED":
                    approved_resolutions.append(res.to_dict() if hasattr(res, "to_dict") else vars(res))

        # Build Minimum Sufficient Compiled RTL Context
        compiled_context = {
            "module": {
                "name": mod_name,
                "type": target_comp.type if target_comp else "module",
                "description": (target_comp.description if target_comp else "") or f"Synthesizable module {mod_name}",
            },
            "architecture_classification": arch_res.to_dict(),
            "approved_resolutions": approved_resolutions,
            "context_reduction_manifest": [],
            "parameters": all_params,
            "ports": ports,
            "clock_reset": clock_reset,
            "signals": signals,
            "registers": registers,
            "fsm": fsm_data,
            "datapath": datapath_operations,
            "protocol_rules": relevant_protocol_rules,
            "design_plan": filtered_plan,
            "source_refs": list(dict.fromkeys(source_refs))[:15]
        }

        output_assigns = {}
        for p in ports:
            if p.get("direction") == "output" and p.get("expression"):
                output_assigns[p["name"]] = p["expression"]
        if output_assigns:
            compiled_context["output_assignments"] = output_assigns

        # Inject semantically critical constructs whenever present by dependency
        if sem_deps.opcodes:
            compiled_context["opcodes"] = sem_deps.opcodes
        if sem_deps.instructions:
            compiled_context["instructions"] = sem_deps.instructions
        if sem_deps.instruction_formats:
            compiled_context["instruction_formats"] = sem_deps.instruction_formats
        if sem_deps.transactions:
            compiled_context["relevant_transactions"] = sem_deps.transactions
        if sem_deps.commands:
            compiled_context["commands"] = sem_deps.commands
        if sem_deps.memory_operations:
            compiled_context["memory_operations"] = sem_deps.memory_operations
        if sem_deps.handshakes_and_constraints:
            compiled_context["handshakes_and_constraints"] = sem_deps.handshakes_and_constraints
        if sem_deps.ordering_and_timeouts:
            compiled_context["ordering_and_timeouts"] = sem_deps.ordering_and_timeouts
        if sem_deps.flows_and_branches:
            compiled_context["flows_and_branches"] = sem_deps.flows_and_branches
        # Collect and attach unresolved conflicts or provenance
        unresolved_conflicts: List[Dict[str, Any]] = []
        if cp and hasattr(cp, "conflicts"):
            for c in (cp.conflicts or []):
                st = c.get("resolution_status", "UNRESOLVED") if isinstance(c, dict) else getattr(c, "resolution_status", "UNRESOLVED")
                if st == "UNRESOLVED":
                    unresolved_conflicts.append(c if isinstance(c, dict) else (c.to_dict() if hasattr(c, "to_dict") else vars(c)))
        elif efs_ir and hasattr(efs_ir, "conflicts"):
            for c in (efs_ir.conflicts or []):
                if getattr(c, "resolution_status", "UNRESOLVED") == "UNRESOLVED":
                    unresolved_conflicts.append(c.to_dict() if hasattr(c, "to_dict") else vars(c))

        # Check for opcode collisions in opcodes and instructions
        all_opcodes = list(compiled_context.get("opcodes", []))
        if efs_ir and hasattr(efs_ir, "opcodes") and efs_ir.opcodes:
            for op in efs_ir.opcodes:
                all_opcodes.append(op.to_dict() if hasattr(op, "to_dict") else op)

        seen_encodings: Dict[str, str] = {}
        for opc in all_opcodes:
            enc = str(opc.get("binary_encoding", "") or opc.get("opcode", "")).strip()
            mnem = str(opc.get("mnemonic", "") or opc.get("name", "")).strip()
            if enc and enc not in ("", "None", "undefined"):
                if enc in seen_encodings and seen_encodings[enc] != mnem:
                    unresolved_conflicts.append({
                        "category": "OPCODE",
                        "type": "OPCODE_COLLISION",
                        "description": f"Opcode binary encoding collision: '{mnem}' and '{seen_encodings[enc]}' share '{enc}'.",
                        "entities": [mnem, seen_encodings[enc]],
                        "resolution_status": "UNRESOLVED",
                        "is_blocking": True
                    })
                else:
                    seen_encodings[enc] = mnem

        # Check for collisions in efs_ir.instructions
        all_instructions = list(compiled_context.get("instructions", []))
        if efs_ir and hasattr(efs_ir, "instructions") and efs_ir.instructions:
            for inst in efs_ir.instructions:
                all_instructions.append(inst.to_dict() if hasattr(inst, "to_dict") else inst)

        seen_inst_opcs: Dict[str, str] = {}
        for inst in all_instructions:
            opc = str(inst.get("opcode", "")).strip()
            iname = str(inst.get("name", "") or inst.get("mnemonic", "")).strip()
            if opc and opc not in ("", "None", "undefined"):
                if opc in seen_inst_opcs and seen_inst_opcs[opc] != iname:
                    unresolved_conflicts.append({
                        "category": "OPCODE",
                        "type": "OPCODE_COLLISION",
                        "description": f"Opcode collision detected between '{iname}' and '{seen_inst_opcs[opc]}' on opcode '{opc}'.",
                        "entities": [iname, seen_inst_opcs[opc]],
                        "resolution_status": "UNRESOLVED",
                        "is_blocking": True
                    })
                else:
                    seen_inst_opcs[opc] = iname

        # Check for conflicting reset polarities across reset signals
        resets_in_cr = [cr for cr in clock_reset if cr.get("type") == "reset"]
        if len(resets_in_cr) > 1:
            pols = {cr.get("polarity") for cr in resets_in_cr if cr.get("polarity")}
            if len(pols) > 1:
                unresolved_conflicts.append({
                    "category": "RESET",
                    "type": "CONFLICTING_RESET_POLARITY",
                    "description": f"Conflicting reset polarities detected across reset signals: {[r.get('name') + ' (' + str(r.get('polarity')) + ')' for r in resets_in_cr]}",
                    "entities": [r.get("name") for r in resets_in_cr],
                    "resolution_status": "UNRESOLVED",
                    "is_blocking": True
                })

        compiled_context["unresolved_conflicts"] = unresolved_conflicts

        # 9. Grounding Validation
        validation = GroundingValidator.validate_rtl_context(compiled_context, efs_ir)
        if not validation.is_valid:
            logger.warning(f"RTL context grounding violations: {validation.violations}")

        # 10. Priority-Based Token Budget Enforcement
        pruned_context, exceeded, warning = TokenBudgetController.enforce_rtl_budget(
            compiled_context, max_tokens=max_budget
        )

        metrics.compiled_context_tokens = estimate_tokens(pruned_context)
        metrics.budget_exceeded = exceeded
        metrics.overflow_warning = warning
        metrics.calculate_reduction()

        pruned_context["_metrics"] = metrics.to_dict()
        return pruned_context
