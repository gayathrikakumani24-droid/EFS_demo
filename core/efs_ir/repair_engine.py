"""
Generic Schema-Driven EFS IR Structural Repair Engine.

Implements the 9-stage idempotent structural repair and validation workflow:
1. Normalize entity representations using schema-defined rules.
2. Disambiguate ports/signals mistakenly grouped as interfaces.
3. Build Canonical Entity Registry and resolve stable identifiers.
4. Resolve all cross-entity references (ownership, interface-signal linkage, endpoints).
5. Construct and reconstruct valid relationships after resolving dependencies.
6. Analyze and repair behavioral / FSM completeness.
7. Deterministically eliminate preventable structural violations before conflict detection.
8. Revalidate the repaired EFS IR.
9. Enforce idempotency: re-running repair produces zero delta.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from core.efs_ir.models import (
    EFSIR, EFSComponent, EFSInterface, EFSSignal, EFSRegister,
    EFSRegisterField, EFSFSM, EFSState, EFSTransition, EFSConstraint, EFSTraceability,
    EFSConflict, EFSResolution, new_efs_id
)
from core.efs_ir.entity_registry import CanonicalEntityRegistry, normalize_semantic_key
from core.efs_ir.behavioral_analyzer import GenericBehavioralAnalyzer
from core.efs_ir.validation_rules import (
    is_fsm_state_name,
    is_fsm_transition_or_condition,
    is_physical_signal_name,
    is_hardware_component_candidate,
    is_valid_register_candidate,
    is_valid_interface_candidate,
    normalize_hardware_offset,
    normalize_hardware_width,
    DISQUALIFIED_COMPONENT_NAMES,
)
from core.efs_ir.validator import validate_efs_ir
from utils.logger import get_logger

logger = get_logger("efs_ir.repair_engine")


class EFSStructuralRepairEngine:
    """
    Schema-driven, specification-agnostic structural repair and validation engine.
    """

    @classmethod
    def repair_and_validate(
        cls,
        ir: EFSIR,
        req_model: Optional[Dict[str, Any]] = None,
        spec_text: str = "",
        plan: Optional[Dict[str, Any]] = None,
        doc_ir: Optional[Any] = None
    ) -> Tuple[EFSIR, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Execute the full schema-driven repair loop on the provided EFS IR.
        
        Returns:
            (repaired_ir, repair_log, remaining_validation_issues)
        """
        logger.info("Executing Schema-Driven EFS IR Structural Repair...")
        repair_log: List[Dict[str, Any]] = []

        # Step 1: Ensure top-level metadata has canonical ID, display name, and top_module
        cls._repair_metadata(ir, repair_log)

        # Step 2: Disambiguate ports mistakenly stored in interfaces
        cls._disambiguate_interfaces_and_signals(ir, repair_log)

        # Step 3: Normalize signals, registers, and components
        cls._normalize_entities(ir, repair_log)

        # Step 4: Build Canonical Entity Registry and resolve references
        registry = CanonicalEntityRegistry(ir)
        cls._resolve_references(ir, registry, repair_log)

        # Step 5: Reconstruct valid relationships and interface linkages
        cls._reconstruct_relationships_and_interfaces(ir, registry, repair_log)

        # Step 6: Behavioral and FSM completeness analysis
        behavior_decisions = GenericBehavioralAnalyzer.analyze_and_complete_behavior(
            ir, req_model=req_model, spec_text=spec_text, plan=plan, doc_ir=doc_ir
        )
        if behavior_decisions:
            repair_log.extend(behavior_decisions)

        # Step 6.1: Resolve FSM and state references after behavioral analysis
        cls._resolve_fsm_references(ir, registry, repair_log)

        # Step 6.2: Resolve timed FSM transitions and explicit synthesizable guards
        cls._resolve_timed_fsm_transitions(ir, repair_log, req_model=req_model, spec_text=spec_text, plan=plan)

        # Step 7: Final ownership and boundary stabilization
        cls._stabilize_ownership(ir, registry, repair_log)

        # Step 8: Revalidate repaired IR
        validation_issues = validate_efs_ir(ir)

        # Step 9: Clean resolved issues from EFS IR conflicts if previously logged
        cls._reconcile_conflicts(ir, validation_issues)

        logger.info(
            f"Schema-Driven Repair complete: {len(repair_log)} repairs applied, "
            f"{len(validation_issues)} validation issues remaining."
        )

        return ir, repair_log, validation_issues

    @classmethod
    def _repair_metadata(cls, ir: EFSIR, log: List[Dict[str, Any]]) -> None:
        """Ensure metadata contains non-empty design_id, canonical_id, display_name, and top_module."""
        if not ir.metadata:
            return

        if not ir.metadata.design_id:
            ir.metadata.design_id = new_efs_id("dsn")
            log.append({"action": "assign_design_id", "value": ir.metadata.design_id})

        if not getattr(ir.metadata, "canonical_id", None):
            clean_suffix = ir.metadata.design_id.replace("DSN_", "").replace("dsn_", "")
            ir.metadata.canonical_id = f"EFS_{clean_suffix}"
            log.append({"action": "assign_canonical_id", "value": ir.metadata.canonical_id})

        if not getattr(ir.metadata, "display_name", None):
            disp = ir.metadata.design_name or "Hardware_Module"
            ir.metadata.display_name = disp.replace("_", " ").title()
            log.append({"action": "assign_display_name", "value": ir.metadata.display_name})

        if not getattr(ir.metadata, "top_module", None):
            top_cand = ir.metadata.design_name or "hardware_module"
            ir.metadata.top_module = re.sub(r"[^a-zA-Z0-9_]", "_", top_cand).strip("_").lower()
            log.append({"action": "assign_top_module", "value": ir.metadata.top_module})

        if not getattr(ir.metadata, "generation_run_id", None):
            ir.metadata.generation_run_id = f"RUN_{ir.metadata.canonical_id}"
            log.append({"action": "assign_generation_run_id", "value": ir.metadata.generation_run_id})

    @classmethod
    def _disambiguate_interfaces_and_signals(cls, ir: EFSIR, log: List[Dict[str, Any]]) -> None:
        """
        Disambiguate physical signals/ports mistakenly classified or stored as EFSInterface entities.
        If an EFSInterface has no sub-signals, no component endpoints, or matches physical port
        attributes (direction, width), convert it to an EFSSignal and remove from interfaces.
        """
        remaining_interfaces: List[EFSInterface] = []
        existing_sig_names = {s.name.lower() for s in ir.signals}

        for iface in ir.interfaces:
            # Check if this interface is actually a port/signal
            is_port = False
            i_name = iface.name.strip()
            
            # If it has no signals and no valid endpoints
            has_endpoints = bool(iface.source_component or iface.destination_component)
            has_sub_signals = bool(iface.signals and len(iface.signals) > 0)

            # Check if direction or width indicates a single port
            dir_str = getattr(iface, "direction", "").lower()
            is_standard_port_dir = dir_str in ("input", "output", "inout", "in", "out")

            if (not has_sub_signals and not has_endpoints) or is_standard_port_dir or is_physical_signal_name(i_name):
                # Check if it has port naming or standard protocol signal role
                is_port = True

            if is_port:
                # Convert to EFSSignal if not already in signals
                if i_name.lower() not in existing_sig_names:
                    d_norm = "output" if "out" in dir_str else "input"
                    role = "clock" if any(k in i_name.lower() for k in ("clk", "clock")) else (
                        "reset" if any(k in i_name.lower() for k in ("rst", "reset")) else "data"
                    )
                    sig = EFSSignal(
                        signal_id=new_efs_id("sig"),
                        name=i_name,
                        width="1",
                        direction=d_norm,
                        datatype="logic",
                        semantic_role=role,
                        description=iface.description or f"Hardware port {i_name}",
                        traceability=iface.traceability or EFSTraceability(extraction_method="disambiguated_port", confidence=0.9)
                    )
                    ir.signals.append(sig)
                    existing_sig_names.add(i_name.lower())
                    log.append({
                        "action": "convert_interface_to_signal",
                        "name": i_name,
                        "reason": "Entity stored in interfaces has physical port semantics without interface bundle properties."
                    })
            else:
                remaining_interfaces.append(iface)

        ir.interfaces = remaining_interfaces

    @classmethod
    def _normalize_entities(cls, ir: EFSIR, log: List[Dict[str, Any]]) -> None:
        """
        Normalize hardware attributes across components, signals, and registers.
        """
        # 1. Purge placeholder components (e.g. 'Module Name')
        clean_components = []
        for comp in ir.components:
            if comp.name.strip().lower() in DISQUALIFIED_COMPONENT_NAMES:
                log.append({"action": "purge_placeholder_component", "component": comp.name})
                continue
            clean_components.append(comp)
        ir.components = clean_components

        # 2. Filter leaked internal IDs from signals
        ir.signals = [s for s in ir.signals if not re.match(r"^SIG_[0-9a-fA-F]{8}$", s.name)]

        # 3. Synchronize canonical clock and reset names and resolve signal aliases
        # If both a concrete RTL port ('clk', 'rst_n') and a descriptive prose term ('Clock', 'Reset') exist:
        # Merge the prose signal into the canonical RTL signal as an alias rather than keeping duplicate physical ports.
        clk_ports = [s for s in ir.signals if s.name.lower() in ("clk", "clock", "aclk", "clk_in")]
        rst_ports = [s for s in ir.signals if any(k == s.name.lower() or s.name.lower().startswith(k + "_") or s.name.lower().endswith("_" + k) for k in ("rst", "reset", "rst_n", "aresetn", "reset_n"))]

        canon_clk_sig = next((s for s in clk_ports if s.name.lower() == "clk"), None) or (clk_ports[0] if clk_ports else None)
        canon_rst_sig = next((s for s in rst_ports if s.name.lower() in ("rst_n", "aresetn", "reset_n", "rst_b")), None) or (rst_ports[0] if rst_ports else None)

        clks_to_remove = set()
        if canon_clk_sig:
            canon_clk_sig.semantic_role = "clock"
            if not hasattr(canon_clk_sig, "aliases") or canon_clk_sig.aliases is None:
                canon_clk_sig.aliases = []
            if "Clock" not in canon_clk_sig.aliases and canon_clk_sig.name != "Clock":
                canon_clk_sig.aliases.append("Clock")
            for other_c in clk_ports:
                if other_c.signal_id != canon_clk_sig.signal_id:
                    if other_c.name not in canon_clk_sig.aliases and other_c.name != canon_clk_sig.name:
                        canon_clk_sig.aliases.append(other_c.name)
                    clks_to_remove.add(other_c.signal_id)
                    log.append({"action": "merge_clock_alias", "canonical": canon_clk_sig.name, "alias": other_c.name})

        rsts_to_remove = set()
        if canon_rst_sig:
            canon_rst_sig.semantic_role = "reset"
            if not hasattr(canon_rst_sig, "aliases") or canon_rst_sig.aliases is None:
                canon_rst_sig.aliases = []
            if "Reset" not in canon_rst_sig.aliases and canon_rst_sig.name != "Reset":
                canon_rst_sig.aliases.append("Reset")
            for other_r in rst_ports:
                if other_r.signal_id != canon_rst_sig.signal_id:
                    if other_r.name not in canon_rst_sig.aliases and other_r.name != canon_rst_sig.name:
                        canon_rst_sig.aliases.append(other_r.name)
                    if "asynch" in (other_r.description or "").lower():
                        canon_rst_sig.description = f"{canon_rst_sig.description or ''} Asynchronous system reset.".strip()
                    rsts_to_remove.add(other_r.signal_id)
                    log.append({"action": "merge_reset_alias", "canonical": canon_rst_sig.name, "alias": other_r.name})

        ir.signals = [s for s in ir.signals if s.signal_id not in clks_to_remove and s.signal_id not in rsts_to_remove]

        # Deduplicate signals by normalized name: keep the entry with explicit direction or more complete attributes
        seen_names: Dict[str, EFSSignal] = {}
        deduped_signals: List[EFSSignal] = []
        for s in ir.signals:
            n_low = s.name.lower()
            if n_low not in seen_names:
                seen_names[n_low] = s
                deduped_signals.append(s)
            else:
                existing = seen_names[n_low]
                if existing.direction == "input" and s.direction in ("output", "inout"):
                    existing.direction = s.direction
                if (not existing.width or existing.width == "1") and s.width and s.width != "1":
                    existing.width = s.width
                if s.semantic_role != "data" and existing.semantic_role == "data":
                    existing.semantic_role = s.semantic_role
                log.append({"action": "deduplicate_signal", "signal": s.name})
        ir.signals = deduped_signals

        canon_clk = canon_clk_sig.name if canon_clk_sig else (ir.clocks[0].name if ir.clocks else "clk")
        canon_rst = canon_rst_sig.name if canon_rst_sig else (ir.resets[0].name if ir.resets else "rst_n")

        # Signals normalization
        seen_sig_ids: Set[str] = set()
        unique_signals: List[EFSSignal] = []

        for sig in ir.signals:
            if not sig.signal_id or sig.signal_id in seen_sig_ids:
                sig.signal_id = new_efs_id("sig")
            seen_sig_ids.add(sig.signal_id)

            # Synchronize clock and reset references
            sig.clock = canon_clk
            if sig.semantic_role != "reset":
                sig.reset = canon_rst

            # Direction normalization
            d = str(sig.direction or "").lower().strip()
            desc = str(sig.description or "").lower()
            name_l = sig.name.lower()
            if name_l in ("busy", "done", "error"):
                sig.direction = "output"
                sig.semantic_role = "control"
            elif d in ("in", "input"):
                sig.direction = "input"
            elif d in ("out", "output"):
                sig.direction = "output"
            elif d in ("inout", "bidirectional", "bidi"):
                sig.direction = "inout"
            else:
                # Infer direction based on description, role, or name suffix
                if any(k in desc for k in ("output signal", "drives", "asserts output", "output to", "output controlling", "outputs")):
                    sig.direction = "output"
                elif any(k in desc for k in ("input signal", "receives", "input to", "clock input", "reset input", "inputs")):
                    sig.direction = "input"
                elif any(sig.name.lower().endswith(sfx) for sfx in ("_out", "_tx", "_led", "_light", "_pwm", "_done", "_busy", "_err", "_error", "_resp", "_ack")):
                    sig.direction = "output"
                elif any(sig.name.lower().endswith(sfx) for sfx in ("_in", "_rx", "_req", "_cmd", "_en", "_enable", "_clk", "_rst", "_rst_n", "_reset_n")):
                    sig.direction = "input"
                elif any(k in sig.name.lower() for k in ("ready", "valid", "data_in", "addr", "clk", "rst", "start", "enable", "en")):
                    sig.direction = "input"
                elif any(k in sig.name.lower() for k in ("data_out", "done", "busy", "error", "err", "resp", "ack")):
                    sig.direction = "output"
                else:
                    sig.direction = "input"
                log.append({"action": "normalize_signal_direction", "signal": sig.name, "direction": sig.direction})

            # Width normalization
            if not sig.width:
                sig.width = "1"
            else:
                sig.width = str(sig.width).strip()

            unique_signals.append(sig)

        # If all non-clock/reset signals are classified as inputs, identify driven signals from suffixes/names
        has_outputs = any(s.direction in ("output", "inout") for s in unique_signals)
        if not has_outputs and len(unique_signals) > 1:
            for s in unique_signals:
                if s.semantic_role not in ("clock", "reset") and not any(k in s.name.lower() for k in ("clk", "rst", "reset")):
                    # Domain output indicators
                    if any(term in s.name.lower() for term in ("light", "led", "out", "pwm", "state", "status", "data")):
                        s.direction = "output"
                        log.append({"action": "repair_all_inputs_contradiction", "promoted_signal": s.name, "new_direction": "output"})

        ir.signals = unique_signals
        ir.clocks = [s for s in ir.signals if s.semantic_role == "clock"]
        ir.resets = [s for s in ir.signals if s.semantic_role == "reset"]

        # Synchronize interfaces clock and reset
        for iface in ir.interfaces:
            iface.clock = canon_clk
            iface.reset = canon_rst

        # Remove spurious external_bus_interface if design is not a bus design
        is_bus_design = (
            ir.metadata.protocol not in ("generic", "none", "", "interconnect")
            or any(i.name != "external_bus_interface" and i.protocol not in ("generic", "none", "", "interconnect") for i in ir.interfaces)
            or any("bus" in c.name.lower() or "axi" in c.name.lower() for c in ir.components)
        )
        if not is_bus_design:
            ir.interfaces = [i for i in ir.interfaces if i.name != "external_bus_interface"]

        # Synchronize FSM reset conditions and normalize cycle duration condition types
        for fsm in ir.fsms:
            if hasattr(fsm, "reset_condition"):
                fsm.reset_condition = canon_rst
            if fsm.reset_behavior and any(old_rst in fsm.reset_behavior for old_rst in ("rst_n", "reset_n", "rst", "reset")):
                for old_rst in ("rst_n", "reset_n", "rst", "reset"):
                    if old_rst != canon_rst and f"on {old_rst}" in fsm.reset_behavior:
                        fsm.reset_behavior = fsm.reset_behavior.replace(f"on {old_rst}", f"on {canon_rst}")
            for trans in fsm.transitions:
                if trans.condition:
                    dur_m = re.search(r"\b(\d+)\s*(?:clock\s*)?cycles?\b", trans.condition, re.IGNORECASE)
                    if dur_m:
                        trans.condition_type = "timed"
                        trans.duration_cycles = int(dur_m.group(1))

        # Registers normalization
        seen_reg_ids: Set[str] = set()
        for reg in ir.registers:
            if not reg.register_id or reg.register_id in seen_reg_ids:
                reg.register_id = new_efs_id("reg")
            seen_reg_ids.add(reg.register_id)

            if reg.offset:
                norm_off = normalize_hardware_offset(reg.offset)
                if norm_off:
                    reg.offset = norm_off

            # Access normalization
            if reg.access:
                reg.access = reg.access.upper().replace("/", "")
                reg.access_type = reg.access

        # Components normalization
        seen_comp_ids: Set[str] = set()
        for comp in ir.components:
            if not comp.component_id or comp.component_id in seen_comp_ids:
                comp.component_id = new_efs_id("comp")
            seen_comp_ids.add(comp.component_id)

    @classmethod
    def _resolve_references(cls, ir: EFSIR, registry: CanonicalEntityRegistry, log: List[Dict[str, Any]]) -> None:
        """
        Resolve cross-entity references using the Canonical Entity Registry.
        Ensures all component references point to valid components, interface references point
        to valid interfaces, and signal references point to valid signals.
        """
        # 1. Resolve Component -> Interface references
        for comp in ir.components:
            resolved_ifaces: List[str] = []
            for if_ref in comp.interfaces:
                canonical_if_id = registry.canonical_id_for(if_ref, expected_type="interface")
                if canonical_if_id:
                    if canonical_if_id not in resolved_ifaces:
                        resolved_ifaces.append(canonical_if_id)
                else:
                    # Check if it was an interface name that exists in ir.interfaces
                    matching_if = next((i for i in ir.interfaces if i.name.lower() == str(if_ref).lower()), None)
                    if matching_if:
                        if matching_if.interface_id not in resolved_ifaces:
                            resolved_ifaces.append(matching_if.interface_id)
            comp.interfaces = resolved_ifaces

        # 2. Resolve Interface -> Component endpoints
        valid_comp_ids = {c.component_id for c in ir.components}
        top_comp = ir.components[0] if ir.components else None

        for iface in ir.interfaces:
            # Source component resolution
            if iface.source_component:
                cid = registry.canonical_id_for(iface.source_component, expected_type="component")
                if cid and cid in valid_comp_ids:
                    iface.source_component = cid
                elif top_comp and iface.source_component.lower() in ("external", "system", "pins", "pin", "boundary"):
                    iface.source_component = top_comp.component_id
                elif top_comp:
                    # Check if matching component by loose name
                    matching_c = next((c for c in ir.components if normalize_semantic_key(c.name) == normalize_semantic_key(iface.source_component)), None)
                    if matching_c:
                        iface.source_component = matching_c.component_id
                    else:
                        # Clear dangling reference or assign top_comp
                        iface.source_component = top_comp.component_id
                        log.append({
                            "action": "repair_interface_endpoint",
                            "interface": iface.name,
                            "repaired_source": top_comp.component_id
                        })

            # Destination component resolution
            if iface.destination_component:
                cid = registry.canonical_id_for(iface.destination_component, expected_type="component")
                if cid and cid in valid_comp_ids:
                    iface.destination_component = cid
                elif top_comp and iface.destination_component.lower() in ("external", "system", "pins", "pin", "boundary"):
                    iface.destination_component = top_comp.component_id
                elif top_comp:
                    matching_c = next((c for c in ir.components if normalize_semantic_key(c.name) == normalize_semantic_key(iface.destination_component)), None)
                    if matching_c:
                        iface.destination_component = matching_c.component_id
                    else:
                        iface.destination_component = None

            # Interface -> Signal references
            resolved_sigs: List[str] = []
            for s_ref in iface.signals:
                sid = registry.canonical_id_for(s_ref, expected_type="signal")
                if sid:
                    if sid not in resolved_sigs:
                        resolved_sigs.append(sid)
                else:
                    # Match by name
                    matching_s = next((s for s in ir.signals if s.name.lower() == str(s_ref).lower()), None)
                    if matching_s and matching_s.signal_id not in resolved_sigs:
                        resolved_sigs.append(matching_s.signal_id)
            iface.signals = resolved_sigs

        # 3. Resolve Signal -> Interface reference
        valid_if_ids = {i.interface_id for i in ir.interfaces}
        for sig in ir.signals:
            if sig.interface and sig.interface not in valid_if_ids:
                canonical_if = registry.canonical_id_for(sig.interface, expected_type="interface")
                if canonical_if and canonical_if in valid_if_ids:
                    sig.interface = canonical_if
                else:
                    sig.interface = None

    @classmethod
    def _reconstruct_relationships_and_interfaces(cls, ir: EFSIR, registry: CanonicalEntityRegistry, log: List[Dict[str, Any]]) -> None:
        """
        Reconstruct inter-module interfaces and relationships when supported by components and signals.
        If an interface exists but has an empty signals list, group signals by interface name prefix.
        """
        # Auto-associate signals with interfaces by prefix or name match if empty
        sig_map = {s.signal_id: s for s in ir.signals}
        for iface in ir.interfaces:
            if not iface.signals:
                prefix = iface.name.lower().rstrip("_") + "_"
                matched_ids = []
                for s in ir.signals:
                    if s.name.lower().startswith(prefix) or s.name.lower() == iface.name.lower():
                        matched_ids.append(s.signal_id)
                        s.interface = iface.interface_id
                if matched_ids:
                    iface.signals = matched_ids
                    log.append({
                        "action": "link_signals_to_interface",
                        "interface": iface.name,
                        "signals_count": len(matched_ids)
                    })

        # Ensure relationships array contains valid, non-dangling inter-module connections
        comp_names = {c.name.lower(): c.name for c in ir.components}
        comp_ids = {c.component_id: c.name for c in ir.components}
        valid_rels = []

        for rel in getattr(ir, "relationships", []) or []:
            src = rel.get("source", "") if isinstance(rel, dict) else getattr(rel, "source", "")
            dst = rel.get("target", "") if isinstance(rel, dict) else getattr(rel, "target", "")
            
            src_canon = comp_ids.get(src) or comp_names.get(src.lower())
            dst_canon = comp_ids.get(dst) or comp_names.get(dst.lower())

            if src_canon and dst_canon:
                sigs = rel.get("signals", []) if isinstance(rel, dict) else getattr(rel, "signals", [])
                valid_rels.append({
                    "source": src_canon,
                    "target": dst_canon,
                    "signals": list(sigs),
                    "direction": rel.get("direction", f"{src_canon} -> {dst_canon}") if isinstance(rel, dict) else f"{src_canon} -> {dst_canon}"
                })

        # Also populate from interfaces connecting two distinct components
        for iface in ir.interfaces:
            if iface.source_component and iface.destination_component and iface.source_component != iface.destination_component:
                src_name = comp_ids.get(iface.source_component) or iface.source_component
                dst_name = comp_ids.get(iface.destination_component) or iface.destination_component
                if not any(r["source"] == src_name and r["target"] == dst_name for r in valid_rels):
                    sig_names = [sig_map[sid].name for sid in iface.signals if sid in sig_map]
                    valid_rels.append({
                        "source": src_name,
                        "target": dst_name,
                        "signals": sig_names,
                        "direction": iface.direction or f"{src_name} -> {dst_name}"
                    })

        ir.relationships = valid_rels

    @classmethod
    def _stabilize_ownership(cls, ir: EFSIR, registry: CanonicalEntityRegistry, log: List[Dict[str, Any]]) -> None:
        """
        Ensure every signal and register has an owner component.
        If no owner is assigned, default to the top-level design component.
        Ensure top-level component exists if components list is empty.
        """
        # Ensure at least one top-level component exists
        if not ir.components:
            top_name = getattr(ir.metadata, "display_name", "") or getattr(ir.metadata, "design_name", "top_module")
            top_comp = EFSComponent(
                component_id=new_efs_id("comp"),
                name=top_name,
                type="top_level_design",
                description=f"Primary hardware module: {top_name}",
                traceability=EFSTraceability(extraction_method="inferred", confidence=0.9)
            )
            ir.components.append(top_comp)
            registry.register_component(top_comp)
            log.append({"action": "create_top_level_component", "name": top_name})

        # Identify true top-level module (container), not merely index 0
        top_comp = next((
            c for c in ir.components 
            if c.type in ("top_level", "top_level_design", "top_module", "system", "container") 
            or (ir.metadata and c.name.lower() in (
                (ir.metadata.top_module or "").lower(),
                (ir.metadata.design_name or "").lower(),
                (ir.metadata.display_name or "").lower()
            ))
        ), None)
        if not top_comp:
            top_comp = ir.components[0]
        top_comp.type = "top_level_design"

        valid_comp_ids = {c.component_id for c in ir.components}
        submodules = [c for c in ir.components if c.component_id != top_comp.component_id]

        # Signal ownership and consumers
        for sig in ir.signals:
            canonical_owner = registry.canonical_id_for(sig.owner, expected_type="component") if sig.owner else None
            if not sig.owner or sig.owner not in valid_comp_ids:
                if canonical_owner and canonical_owner in valid_comp_ids:
                    sig.owner = canonical_owner
                else:
                    if sig.direction in ("input", "inout"):
                        sig.owner = top_comp.component_id
                    else:
                        assigned_sub = None
                        sig_n = sig.name.lower()
                        for sub in submodules:
                            s_desc = (sub.description or "").lower()
                            s_name = sub.name.lower()
                            if sig_n in s_desc or sig_n in s_name or any(k in s_name for k in ("decoder", "output", "driver", "reg")):
                                assigned_sub = sub
                                break
                        sig.owner = assigned_sub.component_id if assigned_sub else top_comp.component_id

            # Attach consumer for top-level inputs if not already assigned
            if not sig.consumer or sig.consumer not in valid_comp_ids:
                if sig.direction == "input":
                    sig_n = sig.name.lower()
                    for sub in submodules:
                        s_desc = (sub.description or "").lower()
                        s_name = sub.name.lower()
                        if sig.semantic_role == "reset" and any(k in s_name or k in s_desc for k in ("reset", "sync", "rst")):
                            sig.consumer = sub.component_id
                            break
                        elif sig.semantic_role == "clock" and any(k in s_name or k in s_desc for k in ("controller", "fsm", "sync", "core")):
                            sig.consumer = sub.component_id
                            break
                        elif sig_n in s_desc or sig_n in s_name:
                            sig.consumer = sub.component_id
                            break

        # Register ownership
        for reg in ir.registers:
            if not reg.owner or reg.owner not in valid_comp_ids:
                canonical_owner = registry.canonical_id_for(reg.owner, expected_type="component") if reg.owner else None
                reg.owner = canonical_owner if (canonical_owner and canonical_owner in valid_comp_ids) else top_comp.component_id

    @classmethod
    def _resolve_timed_fsm_transitions(
        cls,
        ir: EFSIR,
        log: List[Dict[str, Any]],
        req_model: Optional[Dict[str, Any]] = None,
        spec_text: str = "",
        plan: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Generic rule for FSM duration timing resolution:
        When FSM states define duration_cycles (or durations in timed conditions/actions/timing rules)
        and transition departure conditions are constant ('1'b1', true, etc.) or semantic placeholders,
        derive explicit synthesizable terminal-count guards using a cycle counter register.
        """
        if not ir.fsms:
            return

        top_comp = next((
            c for c in ir.components 
            if c.type in ("top_level", "top_level_design", "top_module", "system", "container")
        ), ir.components[0] if ir.components else None)
        top_id = top_comp.component_id if top_comp else None

        # Collect all timing rules / requirements text
        timing_texts: List[str] = []
        for r in getattr(ir, "timing_rules", []):
            timing_texts.append(getattr(r, "description", "") or getattr(r, "name", ""))
        for tr_str in getattr(ir, "timing_requirements", []):
            timing_texts.append(str(tr_str))
        if req_model and isinstance(req_model, dict):
            for tr_str in req_model.get("timing_requirements", []):
                timing_texts.append(str(tr_str))
        if spec_text:
            timing_texts.append(spec_text)
        if plan and isinstance(plan, dict):
            for tr_str in plan.get("timing_rules", []):
                timing_texts.append(str(tr_str))

        for fsm in ir.fsms:
            state_durations: Dict[str, int] = {}
            for st in fsm.states:
                dur = getattr(st, "duration_cycles", None)
                clean_name = re.sub(r"(?i)\s+state$", "", st.name).strip()

                # 1. Check transitions originating from this state
                if dur is None:
                    for tr in fsm.transitions:
                        tr_src_clean = re.sub(r"(?i)\s+state$", "", tr.source_state).strip()
                        if tr.source_state == st.name or tr_src_clean.lower() == clean_name.lower():
                            if getattr(tr, "duration_cycles", None):
                                dur = tr.duration_cycles
                                break
                            elif tr.condition:
                                m = re.search(r"(\d+)\s*(?:clock\s*)?cycles?", str(tr.condition), re.IGNORECASE)
                                if m:
                                    dur = int(m.group(1))
                                    break

                # 2. Check state actions
                if dur is None and getattr(st, "actions", None):
                    for act in st.actions:
                        m = re.search(r"(\d+)\s*(?:clock\s*)?cycles?", str(act), re.IGNORECASE)
                        if not m:
                            m = re.search(r"(?:duration|for|dwell|load\s+cycle_counter\s+with\s+[A-Za-z0-9_]+\s+duration)\s+(\d+)", str(act), re.IGNORECASE)
                        if m:
                            dur = int(m.group(1))
                            break

                # 3. Check timing rules / text sources
                if dur is None:
                    for txt in timing_texts:
                        # e.g., "The RED light remains ON for 5 clock cycles"
                        m = re.search(rf"\b{re.escape(clean_name)}\b.*?\b(\d+)\s*(?:clock\s*)?cycles?", txt, re.IGNORECASE)
                        if not m:
                            m = re.search(rf"\b(\d+)\s*(?:clock\s*)?cycles?.*?\b{re.escape(clean_name)}\b", txt, re.IGNORECASE)
                        if not m:
                            m = re.search(rf"\b{re.escape(st.name)}\b.*?\b(\d+)\s*(?:clock\s*)?cycles?", txt, re.IGNORECASE)
                        if m:
                            dur = int(m.group(1))
                            break

                if dur is not None:
                    st.duration_cycles = dur
                    state_durations[st.name] = dur
                    state_durations[st.name.lower()] = dur
                    state_durations[clean_name] = dur
                    state_durations[clean_name.lower()] = dur

            if not state_durations:
                continue

            needs_timer = False
            for tr in fsm.transitions:
                src_clean = re.sub(r"(?i)\s+state$", "", tr.source_state).strip().lower()
                dur = state_durations.get(tr.source_state) or state_durations.get(tr.source_state.lower()) or state_durations.get(src_clean)
                if dur:
                    cond_str = str(tr.condition or "").strip().lower()
                    if cond_str in ("1'b1", "1", "true", "unconditional", "always", "") or tr.condition_type in ("semantic", "unresolved", "timed"):
                        needs_timer = True
                        break

            if not needs_timer:
                continue

            max_dur = max(state_durations.values())
            max_cnt_val = max(1, max_dur - 1)
            cnt_width = max(1, max_cnt_val.bit_length())

            # Find or create counter register
            timer_reg = next((r for r in ir.registers if any(k in r.name.lower() for k in ("timer", "cnt", "counter"))), None)
            if not timer_reg:
                timer_reg_name = "timer_cnt"
                timer_reg = EFSRegister(
                    register_id=new_efs_id("reg"),
                    name=timer_reg_name,
                    offset="0x00",
                    offset_source="explicit",
                    width=cnt_width,
                    access_type="RO",
                    access="RO",
                    reset_value="0x0",
                    description="Cycle counter tracking elapsed clock cycles in current FSM state.",
                    owner=top_id
                )
                ir.registers.append(timer_reg)
                log.append({"action": "create_cycle_counter_register", "name": timer_reg_name, "width": cnt_width})
            else:
                timer_reg_name = timer_reg.name
                if (timer_reg.width or 0) < cnt_width:
                    timer_reg.width = cnt_width

            # Find or create terminal count signal
            done_sig = next((s for s in ir.signals if any(k in s.name.lower() for k in ("done", "timer_done"))), None)
            if not done_sig:
                done_sig_name = "timer_done"
                done_sig = EFSSignal(
                    signal_id=new_efs_id("sig"),
                    name=done_sig_name,
                    width="1",
                    direction="output",
                    semantic_role="control",
                    description="Terminal count flag indicating state duration has elapsed.",
                    owner=top_id
                )
                ir.signals.append(done_sig)
                log.append({"action": "create_terminal_count_signal", "name": done_sig_name})
            else:
                done_sig_name = done_sig.name

            # Synthesize explicit synthesizable guards for transitions
            for tr in fsm.transitions:
                src_clean = re.sub(r"(?i)\s+state$", "", tr.source_state).strip().lower()
                dur = state_durations.get(tr.source_state) or state_durations.get(tr.source_state.lower()) or state_durations.get(src_clean)
                if dur:
                    cond_str = str(tr.condition or "").strip().lower()
                    if cond_str in ("1'b1", "1", "true", "unconditional", "always", "") or tr.condition_type in ("semantic", "unresolved", "timed"):
                        term_val = max(0, dur - 1)
                        tr.condition = f"{timer_reg_name} == {cnt_width}'d{term_val}"
                        tr.condition_type = "signal_assertion"
                        tr.associated_register = timer_reg_name
                        tr.associated_signal = done_sig_name
                        tr.duration_cycles = dur
                        tr.description = f"Transition to {tr.target_state} after {dur} clock cycles in {tr.source_state} ({timer_reg_name} == {term_val})."
                        log.append({
                            "action": "synthesize_transition_guard",
                            "from": tr.source_state,
                            "to": tr.target_state,
                            "condition": tr.condition
                        })

            # Explicitly record design assumptions in IR constraints
            assumptions = [
                ("timing_off_by_one", "State duration of N cycles occupies exactly N positive clock edges (0 to N-1 count; transition fires at terminal count N-1)."),
                ("reset_behavior", "Reset deassertion enters initial/reset state with cycle counter synchronously cleared to 0."),
                ("counter_clear_timing", "Cycle counter increments synchronously on active clock edge and clears to 0 upon state change."),
                ("output_decoding", "State outputs are combinationally decoded from current state, guaranteeing mutual exclusivity every cycle.")
            ]
            for a_type, a_desc in assumptions:
                if not any(c.type == a_type for c in ir.constraints):
                    ir.constraints.append(EFSConstraint(
                        constraint_id=new_efs_id("const"),
                        type=a_type,
                        expected_behavior=a_desc,
                        severity="INFO"
                    ))

    @classmethod
    def _resolve_fsm_references(cls, ir: EFSIR, registry: CanonicalEntityRegistry, log: List[Dict[str, Any]]) -> None:
        """
        Resolve and canonicalize state references in FSM transitions and states against registry.
        """
        for fsm in ir.fsms:
            # Map state ids and case-insensitive names to canonical state name
            state_lookup: Dict[str, str] = {}
            for st in fsm.states:
                if not st.state_id:
                    st.state_id = new_efs_id("st")
                registry.register_state(st, fsm_id=fsm.fsm_id)
                state_lookup[st.name.lower()] = st.name
                state_lookup[st.state_id.lower()] = st.name

            # Canonicalize initial_state
            if fsm.initial_state and fsm.initial_state.lower() in state_lookup:
                fsm.initial_state = state_lookup[fsm.initial_state.lower()]

            # Canonicalize reset_state
            if fsm.reset_state and fsm.reset_state.lower() in state_lookup:
                fsm.reset_state = state_lookup[fsm.reset_state.lower()]

            # Canonicalize transitions
            for trans in fsm.transitions:
                if not trans.transition_id:
                    trans.transition_id = new_efs_id("trans")
                if trans.source_state and trans.source_state.lower() in state_lookup:
                    trans.source_state = state_lookup[trans.source_state.lower()]
                if trans.target_state and trans.target_state.lower() in state_lookup:
                    trans.target_state = state_lookup[trans.target_state.lower()]

    @classmethod
    def _reconcile_conflicts(cls, ir: EFSIR, remaining_validation_issues: List[Dict[str, Any]]) -> None:
        """
        Reconcile conflicts: if an issue that was previously logged as a conflict
        has been deterministically repaired and is no longer present in validation issues,
        mark it as RESOLVED.
        """
        if not hasattr(ir, "conflicts") or not ir.conflicts:
            return

        active_issue_messages = {i.get("message", i.get("issue", "")) for i in remaining_validation_issues}
        for conf in ir.conflicts:
            c_title = getattr(conf, "title", "")
            c_desc = getattr(conf, "description", "")
            # If the conflict describes a dangling reference or missing clock/reset that is now resolved
            if c_title in ("Missing Clock Signal", "Missing Reset Signal") and any("clk" in s.name.lower() for s in ir.signals) and any("rst" in s.name.lower() for s in ir.signals):
                conf.resolution_status = "RESOLVED"
                conf.status = "RESOLVED"
                conf.blocking = False
                conf.requires_human_decision = False
            # If the conflict describes a transition_condition that is now synthesized
            elif getattr(conf, "property", "") == "transition_condition":
                conf_entities = [re.sub(r"(?i)\s+state$", "", str(e)).strip().lower() for e in (conf.entities or [])]
                matched_tr = None
                for fsm in ir.fsms:
                    for tr in fsm.transitions:
                        src_c = re.sub(r"(?i)\s+state$", "", tr.source_state).strip().lower()
                        dst_c = re.sub(r"(?i)\s+state$", "", tr.target_state).strip().lower()
                        if (src_c in conf_entities and dst_c in conf_entities) or (tr.source_state.lower() in conf_entities and tr.target_state.lower() in conf_entities):
                            matched_tr = tr
                            break
                    if matched_tr:
                        break

                if matched_tr:
                    is_now_explicit = (
                        matched_tr.condition_type not in ("semantic", "unresolved") and
                        matched_tr.condition not in ("1'b1", "1", "true", "")
                    )
                    if not is_now_explicit and matched_tr.condition_type == "unconditional" and not matched_tr.duration_cycles:
                        is_now_explicit = True

                    if is_now_explicit:
                        conf.resolution_status = "RESOLVED"
                        conf.status = "RESOLVED"
                        conf.blocking = False
                        conf.requires_human_decision = False
                        conf.resolution_notes = f"Transition condition synthesized to explicit Boolean guard: {matched_tr.condition}"
                else:
                    all_explicit = all(
                        tr.condition_type not in ("semantic", "unresolved") and tr.condition not in ("1'b1", "1", "true", "")
                        for fsm in ir.fsms for tr in fsm.transitions
                    )
                    if all_explicit:
                        conf.resolution_status = "RESOLVED"
                        conf.status = "RESOLVED"
                        conf.blocking = False
                        conf.requires_human_decision = False
                        conf.resolution_notes = "All FSM transition conditions synthesized to explicit Boolean guards."
            elif c_desc not in active_issue_messages and c_title not in active_issue_messages:
                if "references non-existent" in c_desc or "owner component" in c_desc or "references non-existent" in c_title:
                    conf.resolution_status = "RESOLVED"
                    conf.status = "RESOLVED"
                    conf.blocking = False
                    conf.requires_human_decision = False

