"""
Context Resolution and Entity Mapping module.

Provides canonical entity, signal, component, register, and relationship resolution.
Maps internal IDs (SIG_..., COMP_..., IF_..., REG_...) to canonical human-readable names
defined in EFS IR. Never invents names. Eliminates duplicates. Preserves source references.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from core.efs_ir.models import (
    EFSIR, EFSComponent, EFSInterface, EFSSignal, EFSRegister, EFSFSM,
    EFSTransaction, EFSConstraint, EFSTimingRule, EFSProtocolRule
)
from utils.logger import get_logger

logger = get_logger("context_compiler.resolver")


class ContextResolver:
    """Resolves and maps EFS IR entities to canonical human-readable names and relationships."""

    def __init__(self, efs_ir: Optional[EFSIR] = None):
        self.efs_ir = efs_ir
        self.component_by_id: Dict[str, EFSComponent] = {}
        self.component_by_name: Dict[str, EFSComponent] = {}
        self.signal_by_id: Dict[str, EFSSignal] = {}
        self.signal_by_name: Dict[str, EFSSignal] = {}
        self.interface_by_id: Dict[str, EFSInterface] = {}
        self.interface_by_name: Dict[str, EFSInterface] = {}
        self.register_by_id: Dict[str, EFSRegister] = {}
        self.register_by_name: Dict[str, EFSRegister] = {}
        self.fsm_by_id: Dict[str, EFSFSM] = {}
        self.fsm_by_name: Dict[str, EFSFSM] = {}

        # Canonical ID -> Human-Readable Name map
        self.id_to_name: Dict[str, str] = {}

        if self.efs_ir:
            self._index_efs_ir()

    def _index_efs_ir(self) -> None:
        """Index all canonical objects and establish ID -> Name mappings."""
        if not self.efs_ir:
            return

        # 1. Components
        for c in (self.efs_ir.components or []):
            c_name = (c.name or "").strip()
            if c.component_id:
                self.component_by_id[c.component_id] = c
                if c_name:
                    self.id_to_name[c.component_id] = c_name
            if c_name:
                self.component_by_name[c_name.lower()] = c

        # 2. Interfaces
        for iface in (self.efs_ir.interfaces or []):
            i_name = (iface.name or "").strip()
            if iface.interface_id:
                self.interface_by_id[iface.interface_id] = iface
                if i_name:
                    self.id_to_name[iface.interface_id] = i_name
            if i_name:
                self.interface_by_name[i_name.lower()] = iface

        # 3. Signals
        for s in (self.efs_ir.signals or []):
            s_name = (s.name or "").strip()
            if not s_name or s_name.startswith("SIG_"):
                if s.protocol_role:
                    s_name = s.protocol_role.lower()
                elif s.semantic_role and s.semantic_role not in ("data", "control"):
                    s_name = f"{s.semantic_role}_sig"
                else:
                    s_name = s.name or "sig_unnamed"

            if s.signal_id:
                self.signal_by_id[s.signal_id] = s
                if s_name:
                    self.id_to_name[s.signal_id] = s_name
            if s_name:
                self.signal_by_name[s_name.lower()] = s

        # 4. Registers
        for r in (self.efs_ir.registers or []):
            r_name = (r.name or "").strip()
            if r.register_id:
                self.register_by_id[r.register_id] = r
                if r_name:
                    self.id_to_name[r.register_id] = r_name
            if r_name:
                self.register_by_name[r_name.lower()] = r

        # 5. FSMs
        for f in (self.efs_ir.fsms or []):
            f_name = (f.name or "").strip()
            if f.fsm_id:
                self.fsm_by_id[f.fsm_id] = f
                if f_name:
                    self.id_to_name[f.fsm_id] = f_name
            if f_name:
                self.fsm_by_name[f_name.lower()] = f

    def resolve_signal_name(self, signal_ref: str) -> str:
        """Resolve a signal ID or name to its canonical human-readable name."""
        if not signal_ref:
            return ""
        ref = signal_ref.strip()
        if ref in self.id_to_name:
            return self.id_to_name[ref]
        s_obj = self.signal_by_id.get(ref) or self.signal_by_name.get(ref.lower())
        if s_obj and s_obj.name:
            return s_obj.name
        return ref

    def resolve_component_name(self, comp_ref: str) -> str:
        """Resolve a component ID or name to its canonical human-readable name."""
        if not comp_ref:
            return ""
        ref = comp_ref.strip()
        if ref in self.id_to_name:
            return self.id_to_name[ref]
        c_obj = self.component_by_id.get(ref) or self.component_by_name.get(ref.lower())
        if c_obj and c_obj.name:
            return c_obj.name
        return ref

    def resolve_interface_name(self, iface_ref: str) -> str:
        """Resolve an interface ID or name to its canonical human-readable name."""
        if not iface_ref:
            return ""
        ref = iface_ref.strip()
        if ref in self.id_to_name:
            return self.id_to_name[ref]
        i_obj = self.interface_by_id.get(ref) or self.interface_by_name.get(ref.lower())
        if i_obj and i_obj.name:
            return i_obj.name
        return ref

    def scrub_internal_ids(self, text: str) -> str:
        """
        Replace any internal IDs (SIG_..., COMP_..., IF_..., REG_...) present in text
        with canonical human-readable names.
        """
        if not text:
            return ""
        result = text
        for eid, name in self.id_to_name.items():
            if eid in result:
                result = result.replace(eid, name)
        return result

    def get_component_signals(self, component: EFSComponent) -> List[EFSSignal]:
        """
        Retrieve all signals connected to or owned by the given component,
        including interface signals, clock, reset, and direct signals.
        """
        if not self.efs_ir:
            return []

        comp_id = component.component_id
        comp_name_lower = (component.name or "").lower()
        is_top_level = (
            component.type in ("top_level", "top_level_design", "top_module", "system", "container") or
            (self.efs_ir.metadata and comp_name_lower in (
                (self.efs_ir.metadata.top_module or "").lower(),
                (self.efs_ir.metadata.design_name or "").lower(),
                (self.efs_ir.metadata.display_name or "").lower()
            )) or
            len(self.efs_ir.components) <= 1
        )

        # Interface IDs attached to this component
        comp_interface_ids = set(component.interfaces or [])
        for iface in (self.efs_ir.interfaces or []):
            if iface.source_component == comp_id or iface.name.lower() in [i.lower() for i in comp_interface_ids]:
                comp_interface_ids.add(iface.interface_id)
                comp_interface_ids.add(iface.name)

        matched_signals: List[EFSSignal] = []
        seen_sig_names: Set[str] = set()

        for s in (self.efs_ir.signals or []):
            s_name = (s.name or "").strip()
            s_owner = (s.owner or "").lower()
            s_consumer = (getattr(s, "consumer", "") or "").lower()
            s_iface = s.interface or ""

            # Top-level module interface includes all declared top-level external ports
            if is_top_level and s.direction in ("input", "output", "inout"):
                if s_name.lower() not in seen_sig_names:
                    seen_sig_names.add(s_name.lower())
                    matched_signals.append(s)
                continue

            is_direct_owner = (s.owner == comp_id) or (s_owner == comp_name_lower)
            is_direct_consumer = (getattr(s, "consumer", None) == comp_id) or (s_consumer == comp_name_lower)
            iface_sig_refs = [
                sig_ref for iface in (self.efs_ir.interfaces or [])
                if iface.interface_id in comp_interface_ids or (iface.name and iface.name.lower() in [i.lower() for i in comp_interface_ids])
                for sig_ref in (iface.signals or [])
            ]
            iface_sig_refs_lower = {str(r).lower() for r in iface_sig_refs}
            is_iface_member = (
                s_iface in comp_interface_ids
                or (s_iface and s_iface.lower() in [i.lower() for i in comp_interface_ids])
                or s.signal_id in iface_sig_refs
                or s_name.lower() in iface_sig_refs_lower
            )
            is_decoder = "decoder" in comp_name_lower or "decoder" in (getattr(component, "type", "") or "").lower()
            is_clock_reset = s.semantic_role in ("clock", "reset") and not is_decoder and (
                getattr(s, "clock", None) == component.clock_domain or
                getattr(s, "reset", None) == component.reset_domain or
                not s.owner or
                s.owner == comp_id or
                s_owner == comp_name_lower or
                is_direct_consumer or
                not component.clock_domain
            )

            if is_direct_owner or is_direct_consumer or is_iface_member or is_clock_reset:
                if s_name.lower() not in seen_sig_names:
                    seen_sig_names.add(s_name.lower())
                    matched_signals.append(s)

        return matched_signals

    def get_component_registers(self, component: EFSComponent) -> List[EFSRegister]:
        """Retrieve all registers owned by the given component."""
        if not self.efs_ir:
            return []
        comp_id = component.component_id
        comp_name_lower = (component.name or "").lower()
        is_decoder = "decoder" in comp_name_lower or "decoder" in (getattr(component, "type", "") or "").lower()
        if is_decoder:
            return []

        matched: List[EFSRegister] = []
        seen_reg_names: Set[str] = set()
        for r in (self.efs_ir.registers or []):
            r_owner = (r.owner or "").lower()
            if r.owner == comp_id or r_owner == comp_name_lower:
                r_name = (r.name or "").strip()
                if r_name.lower() not in seen_reg_names:
                    seen_reg_names.add(r_name.lower())
                    matched.append(r)
        if not matched and not is_decoder and (len(self.efs_ir.components) <= 1 or not any(getattr(r, "owner", None) for r in self.efs_ir.registers)):
            matched = list(self.efs_ir.registers or [])
        return matched

    def get_component_fsm(self, component: EFSComponent) -> Optional[EFSFSM]:
        """Retrieve the primary FSM associated with the given component."""
        if not self.efs_ir or not self.efs_ir.fsms:
            return None
        comp_id = component.component_id
        comp_name_lower = (component.name or "").lower()
        is_decoder = "decoder" in comp_name_lower or "decoder" in (getattr(component, "type", "") or "").lower()
        if is_decoder:
            return None

        for f in self.efs_ir.fsms:
            f_owner = (getattr(f, "owner", "") or "").lower()
            f_name = (f.name or "").lower()
            if f_owner == comp_id or f_owner == comp_name_lower or comp_name_lower in f_name:
                return f
        # Default to first FSM if single component or no owner specified
        if not is_decoder and (len(self.efs_ir.components) <= 1 or not any(getattr(f, "owner", None) for f in self.efs_ir.fsms)):
            return self.efs_ir.fsms[0]
        return None

    def get_source_refs(self, obj: Any) -> List[str]:
        """Extract source references (requirement IDs, sections) from an EFS IR object."""
        refs: List[str] = []
        if hasattr(obj, "source_requirements") and obj.source_requirements:
            refs.extend(obj.source_requirements)
        if hasattr(obj, "source_chunks") and obj.source_chunks:
            refs.extend(obj.source_chunks)
        if hasattr(obj, "citation") and obj.citation:
            refs.append(str(obj.citation))
        if hasattr(obj, "source_evidence") and obj.source_evidence:
            refs.append(str(obj.source_evidence))
        return list(dict.fromkeys(r for r in refs if r))

    @staticmethod
    def deduplicate_list_of_dicts(items: List[Dict[str, Any]], key_fields: List[str]) -> List[Dict[str, Any]]:
        """Deduplicate a list of dictionaries based on a composite key."""
        seen = set()
        deduped = []
        for item in items:
            key = tuple(str(item.get(k, "")).strip().lower() for k in key_fields)
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped
