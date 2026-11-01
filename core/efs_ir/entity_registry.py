"""
Canonical Entity Registry & Generic Reference Resolver for EFS IR.

Maintains a typed, canonical entity registry indexed by stable identifiers
and semantic keys (names, normalized names, aliases).
Provides deterministic reference resolution, deduplication, and schema-driven
distinction between components, interfaces, signals/ports, registers, and FSM states.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from core.efs_ir.models import (
    EFSIR, EFSComponent, EFSInterface, EFSSignal, EFSRegister,
    EFSFSM, EFSState, EFSTransition, EFSMemoryRegion, EFSInstruction,
    EFSOpcode, new_efs_id
)
from core.efs_ir.validation_rules import (
    is_fsm_state_name,
    is_fsm_transition_or_condition,
    is_physical_signal_name,
    is_hardware_component_candidate,
    is_valid_register_candidate,
)
from utils.logger import get_logger

logger = get_logger("efs_ir.entity_registry")


def normalize_semantic_key(name: Any) -> str:
    """
    Produce a canonical, normalized semantic key for hardware entity matching.
    Case-insensitive, whitespace- and underscore-collapsed, trailing punctuation stripped.
    """
    if name is None:
        return ""
    s = str(name).strip("` \t\r\n'\"")
    # Collapse multiple whitespace, dashes, and underscores to a single underscore
    s = re.sub(r"[\s\-_]+", "_", s).lower()
    return s.strip("_")


class CanonicalEntityRegistry:
    """
    Typed Canonical Entity Registry for EFS IR.
    
    Provides:
    1. Stable lookup by canonical ID (e.g. COMP_..., IF_..., SIG_..., REG_...).
    2. Lookup by exact name, normalized semantic key, and registered aliases.
    3. Type-safe reference resolution.
    4. Deterministic deduplication with automatic cascading reference updates.
    """

    def __init__(self, efs_ir: Optional[EFSIR] = None):
        # Primary storage: id -> entity
        self.components: Dict[str, EFSComponent] = {}
        self.interfaces: Dict[str, EFSInterface] = {}
        self.signals: Dict[str, EFSSignal] = {}
        self.registers: Dict[str, EFSRegister] = {}
        self.fsms: Dict[str, EFSFSM] = {}
        self.states: Dict[str, EFSState] = {}
        self.memories: Dict[str, EFSMemoryRegion] = {}
        self.instructions: Dict[str, EFSInstruction] = {}
        self.opcodes: Dict[str, EFSOpcode] = {}

        # Semantic key maps: (entity_type, normalized_key) -> canonical_id
        self._key_to_id: Dict[Tuple[str, str], str] = {}
        # Global key map (any type) for fallback cross-type resolution: normalized_key -> (entity_type, canonical_id)
        self._global_key_map: Dict[str, List[Tuple[str, str]]] = {}

        # Alias tracking: canonical_id -> Set[alias]
        self._id_to_aliases: Dict[str, Set[str]] = {}

        # Canonical ID remapping (old_id -> new_canonical_id) for merged duplicates
        self.id_redirects: Dict[str, str] = {}

        if efs_ir:
            self.load_from_efs_ir(efs_ir)

    def load_from_efs_ir(self, ir: EFSIR) -> None:
        """Populate the registry from an existing EFS IR instance."""
        for c in ir.components:
            self.register_component(c)
        for i in ir.interfaces:
            self.register_interface(i)
        for s in ir.signals:
            self.register_signal(s)
        for r in ir.registers:
            self.register_register(r)
        for f in ir.fsms:
            self.register_fsm(f)
        for m in getattr(ir, "memory_regions", []):
            self.register_memory(m)
        for inst in getattr(ir, "instructions", []):
            self.register_instruction(inst)
        for opc in getattr(ir, "opcodes", []):
            self.register_opcode(opc)

    def _index_keys(self, entity_id: str, entity_type: str, *keys: Any) -> None:
        """Index one or more semantic keys for an entity."""
        for k in keys:
            if not k:
                continue
            norm = normalize_semantic_key(k)
            if not norm:
                continue
            type_tuple = (entity_type, norm)
            if type_tuple not in self._key_to_id:
                self._key_to_id[type_tuple] = entity_id

            if norm not in self._global_key_map:
                self._global_key_map[norm] = []
            if (entity_type, entity_id) not in self._global_key_map[norm]:
                self._global_key_map[norm].append((entity_type, entity_id))

    def register_component(self, comp: EFSComponent) -> str:
        """Register a component entity. Returns its canonical ID."""
        cid = comp.component_id or new_efs_id("comp")
        comp.component_id = cid
        self.components[cid] = comp
        self._index_keys(cid, "component", cid, comp.name)
        return cid

    def register_interface(self, iface: EFSInterface) -> str:
        """Register an interface entity. Returns its canonical ID."""
        iid = iface.interface_id or new_efs_id("if")
        iface.interface_id = iid
        self.interfaces[iid] = iface
        self._index_keys(iid, "interface", iid, iface.name)
        return iid

    def register_signal(self, sig: EFSSignal) -> str:
        """Register a signal entity. Returns its canonical ID."""
        sid = sig.signal_id or new_efs_id("sig")
        sig.signal_id = sid
        self.signals[sid] = sig
        self._index_keys(sid, "signal", sid, sig.name)
        return sid

    def register_register(self, reg: EFSRegister) -> str:
        """Register a register entity. Returns its canonical ID."""
        rid = reg.register_id or new_efs_id("reg")
        reg.register_id = rid
        self.registers[rid] = reg
        self._index_keys(rid, "register", rid, reg.name)
        return rid

    def register_fsm(self, fsm: EFSFSM) -> str:
        """Register an FSM entity and its states. Returns its canonical ID."""
        fid = fsm.fsm_id or new_efs_id("fsm")
        fsm.fsm_id = fid
        self.fsms[fid] = fsm
        self._index_keys(fid, "fsm", fid, fsm.name)
        for st in getattr(fsm, "states", []):
            self.register_state(st, fid)
        return fid

    def register_state(self, state: EFSState, fsm_id: Optional[str] = None) -> str:
        """Register an FSM state entity. Returns its canonical ID."""
        sid = state.state_id or new_efs_id("st")
        state.state_id = sid
        self.states[sid] = state
        self._index_keys(sid, "state", sid, state.name)
        return sid

    def register_memory(self, mem: EFSMemoryRegion) -> str:
        """Register a memory region entity. Returns its canonical ID."""
        mid = mem.region_id or new_efs_id("memreg")
        mem.region_id = mid
        self.memories[mid] = mem
        self._index_keys(mid, "memory", mid, mem.name)
        return mid

    def register_instruction(self, inst: EFSInstruction) -> str:
        """Register an instruction entity. Returns its canonical ID."""
        iid = inst.instruction_id or new_efs_id("instr")
        inst.instruction_id = iid
        self.instructions[iid] = inst
        self._index_keys(iid, "instruction", iid, inst.name, inst.mnemonic)
        return iid

    def register_opcode(self, opc: EFSOpcode) -> str:
        """Register an opcode entity. Returns its canonical ID."""
        oid = opc.opcode_id or new_efs_id("opc")
        opc.opcode_id = oid
        self.opcodes[oid] = opc
        self._index_keys(oid, "opcode", oid, opc.mnemonic, opc.binary_encoding)
        return oid

    def register_alias(self, entity_id: str, entity_type: str, alias: str) -> None:
        """Register an alias for an existing canonical entity."""
        if not alias or not entity_id:
            return
        if entity_id not in self._id_to_aliases:
            self._id_to_aliases[entity_id] = set()
        self._id_to_aliases[entity_id].add(alias)
        self._index_keys(entity_id, entity_type, alias)

    def resolve(self, ref_key: Any, expected_type: Optional[str] = None) -> Optional[Any]:
        """
        Resolve a reference key (ID, raw name, normalized name, or alias) to its canonical entity.
        If expected_type is provided, restricts resolution to that entity type.
        Follows ID redirects if the entity was merged.
        """
        if ref_key is None:
            return None
        raw_str = str(ref_key).strip("` \t\r\n'\"")
        if not raw_str:
            return None

        # Check if ID was redirected
        effective_key = self.id_redirects.get(raw_str, raw_str)

        # 1. Direct ID lookup
        if expected_type:
            entity = self._lookup_by_id(effective_key, expected_type)
            if entity:
                return entity
        else:
            for etype in ("component", "interface", "signal", "register", "fsm", "state", "memory", "instruction", "opcode"):
                entity = self._lookup_by_id(effective_key, etype)
                if entity:
                    return entity

        # 2. Typed semantic key lookup
        norm = normalize_semantic_key(effective_key)
        if expected_type:
            cid = self._key_to_id.get((expected_type, norm))
            if cid:
                cid = self.id_redirects.get(cid, cid)
                return self._lookup_by_id(cid, expected_type)

        # 3. Global semantic key lookup
        candidates = self._global_key_map.get(norm, [])
        if expected_type:
            candidates = [c for c in candidates if c[0] == expected_type]

        if candidates:
            # Return first valid candidate of expected type, or first overall
            target_type, target_id = candidates[0]
            target_id = self.id_redirects.get(target_id, target_id)
            return self._lookup_by_id(target_id, target_type)

        return None

    def _lookup_by_id(self, entity_id: str, entity_type: str) -> Optional[Any]:
        """Direct lookup by primary identifier for a given type."""
        store_map = {
            "component": self.components,
            "interface": self.interfaces,
            "signal": self.signals,
            "register": self.registers,
            "fsm": self.fsms,
            "state": self.states,
            "memory": self.memories,
            "instruction": self.instructions,
            "opcode": self.opcodes,
        }
        store = store_map.get(entity_type)
        if store and entity_id in store:
            return store[entity_id]
        return None

    def canonical_id_for(self, ref_key: Any, expected_type: Optional[str] = None) -> Optional[str]:
        """Resolve a reference key to its canonical entity ID string."""
        entity = self.resolve(ref_key, expected_type)
        if entity:
            for attr in ("component_id", "interface_id", "signal_id", "register_id", "fsm_id", "state_id", "region_id", "instruction_id", "opcode_id"):
                if hasattr(entity, attr):
                    return getattr(entity, attr)
        return None

    def canonical_name_for(self, ref_key: Any, expected_type: Optional[str] = None) -> Optional[str]:
        """Resolve a reference key to its canonical entity name."""
        entity = self.resolve(ref_key, expected_type)
        if entity and hasattr(entity, "name"):
            return entity.name
        return None

    @staticmethod
    def classify_interface_candidate(item: Dict[str, Any]) -> str:
        """
        Generic, specification-agnostic classification of an entry in an 'interfaces' list.
        Returns:
            'SIGNAL_PORT': Item represents a physical signal/port (has width/direction, no sub-signals, matches signal semantics).
            'INTERFACE_GROUP': Item represents an interface bundle/channel (has sub-signals, endpoints, or protocol bus).
        """
        if not isinstance(item, dict):
            return "SIGNAL_PORT"

        name = str(item.get("name", "")).strip()
        attrs = dict(item)

        # If it explicitly contains sub-signals, it is an interface bundle
        sub_signals = item.get("signals")
        if isinstance(sub_signals, list) and len(sub_signals) > 0:
            return "INTERFACE_GROUP"

        # If it has explicit source and destination component endpoints connecting two components
        if item.get("source_component") or item.get("destination_component") or (item.get("source") and item.get("target")):
            return "INTERFACE_GROUP"

        # If it has single physical wire/bus attributes like width and pin direction
        has_width = bool(item.get("width"))
        has_dir = str(item.get("direction", "")).lower() in ("input", "output", "inout", "in", "out")
        if has_width or has_dir:
            return "SIGNAL_PORT"

        # Check physical signal keyword match
        if is_physical_signal_name(name, attrs):
            return "SIGNAL_PORT"

        # Check standard interface keywords
        name_lower = name.lower()
        if any(k in name_lower for k in ("_bus", "interface", "_channel", "_port", "_if")):
            return "INTERFACE_GROUP"

        # Default: if no endpoints and no sub-signals, it is safer to treat as a port
        return "SIGNAL_PORT"
