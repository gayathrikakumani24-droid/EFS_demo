"""
ContextPack data model.

Represents a minimal, query-driven, dependency-aware, protocol-aware, and token-budgeted
slice of the hardware design space passed to LLM generation agents.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


from core.efs_ir.models import strip_nulls


@dataclass
class TaskIntent:
    """Structured representation of user query intent."""
    task: str = "RTL_GENERATION"  # RTL_GENERATION, SYSTEMVERILOG_GENERATION, VHDL_GENERATION, UVM_GENERATION, SVA_GENERATION, PLANTUML_GENERATION, TESTBENCH_GENERATION, FSM_GENERATION
    target: str = ""
    interface: Optional[str] = None
    language: str = "Verilog"
    raw_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TaskIntent:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ContextPack:
    """
    Task-specific Context Pack passed to LLM generation agents.
    Contains ONLY the resolved target component, its dependencies,
    relevant interface channels, protocol rules, and source evidence.
    """
    task: TaskIntent = field(default_factory=TaskIntent)
    target_component: Dict[str, Any] = field(default_factory=dict)
    interfaces: List[Dict[str, Any]] = field(default_factory=list)
    signals: List[Dict[str, Any]] = field(default_factory=list)
    registers: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    fsm: Dict[str, Any] = field(default_factory=dict)
    transactions: List[Dict[str, Any]] = field(default_factory=list)
    functional_requirements: List[Dict[str, Any]] = field(default_factory=list)
    timing_rules: List[Dict[str, Any]] = field(default_factory=list)
    protocol_rules: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    source_evidence: List[Dict[str, Any]] = field(default_factory=list)
    protocol_source_evidence: List[Dict[str, Any]] = field(default_factory=list)
    relevant_protocol_chunks: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)

    # Context Statistics
    total_full_efs_objects: int = 0
    selected_efs_objects: int = 0
    total_source_chunks: int = 0
    selected_source_chunks: int = 0
    total_protocol_rules: int = 0
    selected_protocol_rules: int = 0
    estimated_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "task": self.task.to_dict() if isinstance(self.task, TaskIntent) else self.task,
            "target_component": self.target_component,
            "user_requirements": self.functional_requirements,
            "target_interfaces": self.interfaces,
            "target_signals": self.signals,
            "target_dependencies": self.dependencies,
            "target_registers": self.registers,
            "target_fsm": self.fsm,
            "target_transactions": self.transactions,
            "relevant_timing_constraints": self.timing_rules,
            "relevant_protocol_rules": self.protocol_rules,
            "relevant_protocol_chunks": self.relevant_protocol_chunks,
            "user_source_evidence": self.source_evidence,
            "protocol_source_evidence": self.protocol_source_evidence,
            "conflicts": self.conflicts,
            "stats": {
                "total_full_efs_objects": self.total_full_efs_objects,
                "selected_efs_objects": self.selected_efs_objects,
                "total_source_chunks": self.total_source_chunks,
                "selected_source_chunks": self.selected_source_chunks,
                "total_protocol_rules": self.total_protocol_rules,
                "selected_protocol_rules": self.selected_protocol_rules,
                "estimated_tokens": self.estimated_tokens,
            }
        }
        return strip_nulls(d)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContextPack:
        task_data = data.get("task", {})
        task = TaskIntent.from_dict(task_data) if isinstance(task_data, dict) else TaskIntent()
        stats = data.get("stats", {})
        return cls(
            task=task,
            target_component=data.get("target_component", {}),
            interfaces=data.get("target_interfaces", data.get("interfaces", [])),
            signals=data.get("target_signals", data.get("signals", [])),
            registers=data.get("target_registers", data.get("registers", [])),
            dependencies=data.get("target_dependencies", data.get("dependencies", [])),
            fsm=data.get("target_fsm", data.get("fsm", {})),
            transactions=data.get("target_transactions", data.get("transactions", [])),
            functional_requirements=data.get("user_requirements", data.get("functional_requirements", [])),
            timing_rules=data.get("relevant_timing_constraints", data.get("timing_rules", [])),
            protocol_rules=data.get("relevant_protocol_rules", data.get("protocol_rules", [])),
            constraints=data.get("constraints", []),
            source_evidence=data.get("user_source_evidence", data.get("source_evidence", [])),
            protocol_source_evidence=data.get("protocol_source_evidence", []),
            relevant_protocol_chunks=data.get("relevant_protocol_chunks", []),
            conflicts=data.get("conflicts", []),
            total_full_efs_objects=stats.get("total_full_efs_objects", 0),
            selected_efs_objects=stats.get("selected_efs_objects", 0),
            total_source_chunks=stats.get("total_source_chunks", 0),
            selected_source_chunks=stats.get("selected_source_chunks", 0),
            total_protocol_rules=stats.get("total_protocol_rules", 0),
            selected_protocol_rules=stats.get("selected_protocol_rules", 0),
            estimated_tokens=stats.get("estimated_tokens", 0),
        )

    def estimate_token_count(self) -> int:
        """Rough estimation of token count (~4 characters per token)."""
        dumped = json.dumps(self.to_dict())
        tokens = max(1, len(dumped) // 4)
        self.estimated_tokens = tokens
        return tokens

    def to_formatted_prompt_text(self) -> str:
        """Format ContextPack as a clean text block for LLM prompts."""
        lines = []
        lines.append("=== TASK INTENT ===")
        lines.append(f"Task Type: {self.task.task}")
        lines.append(f"Target Component: {self.task.target}")
        if self.task.interface:
            lines.append(f"Target Interface: {self.task.interface}")
        lines.append(f"Target Language: {self.task.language}")
        lines.append("")

        lines.append("=== TARGET COMPONENT DETAILS ===")
        lines.append(json.dumps(self.target_component, indent=2))
        lines.append("")

        if self.interfaces:
            lines.append("=== RELEVANT INTERFACES ===")
            lines.append(json.dumps(self.interfaces, indent=2))
            lines.append("")

        if self.signals:
            lines.append("=== RELEVANT SIGNALS ===")
            lines.append(json.dumps(self.signals, indent=2))
            lines.append("")

        if self.registers:
            lines.append("=== RELEVANT REGISTERS ===")
            lines.append(json.dumps(self.registers, indent=2))
            lines.append("")

        if self.dependencies:
            lines.append("=== DIRECT DEPENDENCIES ===")
            lines.append(json.dumps(self.dependencies, indent=2))
            lines.append("")

        if self.fsm:
            lines.append("=== FINITE STATE MACHINE (FSM) ===")
            lines.append(json.dumps(self.fsm, indent=2))
            lines.append("")

        if self.transactions:
            lines.append("=== RELEVANT TRANSACTIONS ===")
            lines.append(json.dumps(self.transactions, indent=2))
            lines.append("")

        if self.functional_requirements:
            lines.append("=== FUNCTIONAL REQUIREMENTS ===")
            lines.append(json.dumps(self.functional_requirements, indent=2))
            lines.append("")

        if self.protocol_rules:
            lines.append("=== PROTOCOL SPECIFIC RULES ===")
            lines.append(json.dumps(self.protocol_rules, indent=2))
            lines.append("")

        if self.timing_rules:
            lines.append("=== TIMING RULES ===")
            lines.append(json.dumps(self.timing_rules, indent=2))
            lines.append("")

        if self.constraints:
            lines.append("=== CONSTRAINTS & ASSERTIONS INTENTS ===")
            lines.append(json.dumps(self.constraints, indent=2))
            lines.append("")

        if self.source_evidence:
            lines.append("=== SOURCE SPECIFICATION EVIDENCE ===")
            for ev in self.source_evidence:
                ref = f"Doc:{ev.get('document_id', 'doc')} p.{ev.get('page', '?')} Sec:{ev.get('section', '?')} [Chunk:{ev.get('chunk_id', '')}]"
                lines.append(f"[{ref}]\n{ev.get('text', '')}\n")
            lines.append("")

        if self.conflicts:
            lines.append("=== UNRESOLVED SPECIFICATION CONFLICTS ===")
            lines.append(json.dumps(self.conflicts, indent=2))
            lines.append("")

        return "\n".join(lines)
