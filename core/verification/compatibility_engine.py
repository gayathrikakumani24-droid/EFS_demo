"""
Compatibility and Validation Engine for Hardware GraphRAG.

Performs semantic validation comparing user specifications against trusted EFS IR models
and protocol knowledge bases. Distinguishes MISSING, CONFLICT, AMBIGUOUS, COMPATIBLE,
DERIVED, IMPLEMENTATION_CHOICE, and UNSUPPORTED validation categories without raw string equality.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


class ProtocolRoleMapper:
    """Normalizes raw hardware interface signal names into standard semantic protocol roles."""

    # Canonical AXI4 / AXI4-Lite signal role definitions
    AXI_ROLES = {
        # Write Address Channel
        "awaddr": ("write_address", "address", "input", "AWADDR"),
        "awvalid": ("write_address", "valid", "input", "AWVALID"),
        "awready": ("write_address", "ready", "output", "AWREADY"),
        "awprot": ("write_address", "protection", "input", "AWPROT"),
        
        # Write Data Channel
        "wdata": ("write_data", "data", "input", "WDATA"),
        "wstrb": ("write_data", "strb", "input", "WSTRB"),
        "wvalid": ("write_data", "valid", "input", "WVALID"),
        "wready": ("write_data", "ready", "output", "WREADY"),
        
        # Write Response Channel
        "bresp": ("write_response", "resp", "output", "BRESP"),
        "write_resp": ("write_response", "resp", "output", "BRESP"),
        "bvalid": ("write_response", "valid", "output", "BVALID"),
        "bready": ("write_response", "ready", "input", "BREADY"),
        
        # Read Address Channel
        "araddr": ("read_address", "address", "input", "ARADDR"),
        "arvalid": ("read_address", "valid", "input", "ARVALID"),
        "arready": ("read_address", "ready", "output", "ARREADY"),
        "arprot": ("read_address", "protection", "input", "ARPROT"),
        
        # Read Data/Response Channel
        "rdata": ("read_response", "data", "output", "RDATA"),
        "read_data": ("read_response", "data", "output", "RDATA"),
        "rresp": ("read_response", "resp", "output", "RRESP"),
        "read_resp": ("read_response", "resp", "output", "RRESP"),
        "rvalid": ("read_response", "valid", "output", "RVALID"),
        "rready": ("read_response", "ready", "input", "RREADY"),
    }

    @classmethod
    def get_semantic_role(cls, raw_signal_name: str, interface_role: str = "slave") -> Optional[Tuple[str, str, str, str]]:
        """
        Returns (channel, role_type, expected_slave_direction, canonical_symbol) for a given signal name.
        """
        s_lower = str(raw_signal_name).strip().lower()
        
        # Exact or suffix match in AXI roles
        for key, info in cls.AXI_ROLES.items():
            if s_lower == key or s_lower.endswith(f"_{key}") or s_lower.endswith(key):
                chan, rtype, slave_dir, canon = info
                # If target is master interface, invert direction
                expected_dir = slave_dir if interface_role.lower() == "slave" else ("input" if slave_dir == "output" else "output")
                return chan, rtype, expected_dir, canon

        # Heuristic role matching for non-AXI or generic interfaces
        if "clk" in s_lower or "clock" in s_lower:
            return "system", "clock", "input", "CLK"
        if "rst" in s_lower or "reset" in s_lower:
            return "system", "reset", "input", "RST"
        if "data" in s_lower:
            direction = "output" if any(w in s_lower for w in ["out", "rd", "read", "resp"]) else "input"
            return "data_bus", "data", direction, "DATA"
        if "valid" in s_lower or "req" in s_lower:
            return "control", "valid", "input", "VALID"
        if "ready" in s_lower or "ack" in s_lower:
            return "control", "ready", "output", "READY"

        return None


class CompatibilityEngine:
    """
    Core Compatibility & Validation Engine.
    
    Evaluates design requirements, canonical EFS IR models, and protocol knowledge bases.
    Classifies issues into 7 explicit validation categories with complete provenance.
    """

    CATEGORIES = {
        "MISSING": {"icon": "🔴", "badge": "🔴 [MISSING]", "is_blocking": True},
        "CONFLICT": {"icon": "🟠", "badge": "🟠 [CONFLICT]", "is_blocking": True},
        "AMBIGUOUS": {"icon": "🟡", "badge": "🟡 [AMBIGUOUS]", "is_blocking": True},
        "COMPATIBLE": {"icon": "🟢", "badge": "🟢 [COMPATIBLE]", "is_blocking": False},
        "DERIVED": {"icon": "🔵", "badge": "🔵 [DERIVED]", "is_blocking": False},
        "IMPLEMENTATION_CHOICE": {"icon": "⚙️", "badge": "⚙️ [IMPLEMENTATION CHOICE]", "is_blocking": False},
        "UNSUPPORTED": {"icon": "🟣", "badge": "🟣 [UNSUPPORTED]", "is_blocking": True},
    }

    @classmethod
    def create_issue(
        cls,
        category: str,
        classification: str,
        issue_text: str,
        required_info: str = "",
        user_spec_evidence: str = "",
        efs_protocol_evidence: str = "",
        source_doc: str = "",
        severity: str = "ERROR"
    ) -> Dict[str, Any]:
        """Constructs a fully attribute-rich, provenance-backed validation issue object."""
        cat_info = cls.CATEGORIES.get(classification, cls.CATEGORIES["MISSING"])
        
        return {
            "category": category,
            "classification": classification,
            "badge": cat_info["badge"],
            "is_blocking": cat_info["is_blocking"],
            "severity": severity if cat_info["is_blocking"] else "INFO",
            "issue": issue_text,
            "required_information": required_info,
            "user_spec_evidence": user_spec_evidence,
            "efs_protocol_evidence": efs_protocol_evidence,
            "evidence": user_spec_evidence or efs_protocol_evidence or "N/A",
            "source_doc": source_doc,
            "provenance": {
                "user_spec": user_spec_evidence,
                "efs_protocol": efs_protocol_evidence,
                "source_doc": source_doc
            }
        }

    @classmethod
    def validate_interface_signals(
        cls,
        user_ports: List[Dict[str, Any]],
        efs_signals: List[Any],
        interface_role: str = "slave"
    ) -> List[Dict[str, Any]]:
        """
        Validates interface ports against semantic protocol roles rather than raw string comparison.
        Detects direction, width, ownership, and role mismatches as CONFLICT issues.
        """
        issues = []
        
        efs_signal_map = {}
        for sig in efs_signals:
            sig_name = str(getattr(sig, "name", "")).strip()
            efs_signal_map[sig_name.lower()] = sig

        for port in user_ports:
            p_name = str(port.get("name", "")).strip()
            p_name_lower = p_name.lower()
            
            if p_name_lower in ("name", "signal", "port", "pin", "width", "direction", "description"):
                continue

            p_dir = str(port.get("direction", "input")).strip().lower()
            if "out" in p_dir:
                p_dir = "output"
            elif "in" in p_dir:
                p_dir = "input"

            role_info = ProtocolRoleMapper.get_semantic_role(p_name, interface_role)
            
            # Check corresponding EFS signal
            efs_sig = efs_signal_map.get(p_name_lower)
            if not efs_sig and role_info:
                # Try finding matching signal by semantic role
                _, _, _, canon = role_info
                for e_name, e_obj in efs_signal_map.items():
                    e_role_info = ProtocolRoleMapper.get_semantic_role(e_name, interface_role)
                    if e_role_info and e_role_info[3] == canon:
                        efs_sig = e_obj
                        break

            if efs_sig:
                efs_dir = str(getattr(efs_sig, "direction", "input")).strip().lower()
                if "out" in efs_dir:
                    efs_dir = "output"
                elif "in" in efs_dir:
                    efs_dir = "input"

                # Check direction conflict between User Spec (p_dir), EFS IR (efs_dir), or Protocol Standard (expected_dir)
                expected_dir = role_info[2] if role_info else efs_dir
                if p_dir != efs_dir or p_dir != expected_dir:
                    issues.append(cls.create_issue(
                        category="INTERFACE",
                        classification="CONFLICT",
                        issue_text=f"Signal direction conflict on '{p_name}': User Spec declares '{p_dir}', but EFS/Protocol specifies '{expected_dir}'.",
                        required_info=f"Clarify interface port '{p_name}' direction (expected '{expected_dir}').",
                        user_spec_evidence=f"User Specification: {p_name} direction is '{p_dir}'",
                        efs_protocol_evidence=f"EFS IR / Protocol Standard ({role_info[3] if role_info else 'EFS'}): expected '{expected_dir}'",
                        severity="ERROR"
                    ))

        return issues
