"""
Generation Context Contract and Pipeline Boundary.

Defines the explicit GenerationContext contract that serves as the boundary between
upstream knowledge extraction (Requirement Model, EFS IR, Design Plan, Protocol Rules)
and downstream LLM generation prompts.

Guarantees:
1. Prompts consume ONLY the minimum sufficient compiled context.
2. Raw specification dumps, DocumentIR text chunks, and massive unneeded graph dumps
   are stripped at this boundary.
3. Behavioral semantics (entities, interfaces, signals, registers, FSM states/transitions,
   opcodes, instructions, constraints, dependencies) are strictly preserved.
4. Token metrics are recorded at every stage:
   - original_context_tokens
   - compiled_context_tokens
   - final_prompt_tokens
   - final_input_tokens
   - provider_input_tokens (when reported by LLM API)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from utils.logger import get_logger
from core.context_compiler.base import estimate_tokens
from core.context_compiler.telemetry import record_generation_telemetry, CompilationMetrics

logger = get_logger("context_compiler.generation_context")


@dataclass
class GenerationContext:
    """
    Explicit, specification-agnostic contract for LLM generation context.
    Contains ONLY information required by the selected artifact.
    """
    artifact_type: str = "generic"
    design_intent: str = ""
    target_component: str = "target_module"
    entities: List[Dict[str, Any]] = field(default_factory=list)
    interfaces: List[Dict[str, Any]] = field(default_factory=list)
    signals: List[Dict[str, Any]] = field(default_factory=list)
    registers: List[Dict[str, Any]] = field(default_factory=list)
    behavior: List[Dict[str, Any]] = field(default_factory=list)
    fsm: Dict[str, Any] = field(default_factory=dict)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    traceability: List[str] = field(default_factory=list)
    protocol_rules: List[Dict[str, Any]] = field(default_factory=list)
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    opcodes: List[Dict[str, Any]] = field(default_factory=list)
    instructions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    entity_provenance: Dict[str, Any] = field(default_factory=dict)
    unresolved_entities: List[Dict[str, Any]] = field(default_factory=list)
    projection_rationale: Dict[str, Any] = field(default_factory=dict)

    # Telemetry measurements
    original_context_tokens: int = 0
    compiled_context_tokens: int = 0
    final_prompt_tokens: int = 0
    final_input_tokens: int = 0
    provider_input_tokens: Optional[int] = None
    output_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert contract to a clean dictionary omitting empty fields."""
        d: Dict[str, Any] = {
            "artifact_type": self.artifact_type,
            "target_component": self.target_component,
        }
        if self.design_intent:
            d["design_intent"] = self.design_intent
        if self.entities:
            d["entities"] = self.entities
        if self.interfaces:
            d["interfaces"] = self.interfaces
        if self.signals:
            d["signals"] = self.signals
        if self.registers:
            d["registers"] = self.registers
        if self.fsm:
            d["fsm"] = self.fsm
        if self.behavior:
            d["behavior"] = self.behavior
        if self.constraints:
            d["constraints"] = self.constraints
        if self.dependencies:
            d["dependencies"] = self.dependencies
        if self.conflicts:
            d["conflicts"] = self.conflicts
        if self.protocol_rules:
            d["protocol_rules"] = self.protocol_rules
        if self.parameters:
            d["parameters"] = self.parameters
        if self.opcodes:
            d["opcodes"] = self.opcodes
        if self.instructions:
            d["instructions"] = self.instructions
        if self.traceability:
            d["traceability"] = self.traceability
        if self.metadata:
            d["metadata"] = self.metadata
        if self.entity_provenance:
            d["entity_provenance"] = self.entity_provenance
        if self.unresolved_entities:
            d["unresolved_entities"] = self.unresolved_entities
        if self.projection_rationale:
            d["projection_rationale"] = self.projection_rationale
        return d

    def to_prompt_json(self, indent: int = 2) -> str:
        """Serialize contract to clean JSON for inclusion in LLM prompt."""
        return json.dumps(self.to_dict(), indent=indent, default=str, ensure_ascii=False)

    def to_prompt_payload(self) -> str:
        """Standardized formatted prompt payload for generation prompts."""
        return self.to_prompt_json(indent=2)

    def calculate_metrics(self, prompt_text: str, system_prompt: str = "") -> None:
        """Calculate and update token metrics at the LLM boundary."""
        if not self.compiled_context_tokens:
            self.compiled_context_tokens = estimate_tokens(self.to_dict())
        self.final_prompt_tokens = estimate_tokens(prompt_text)
        self.final_input_tokens = self.final_prompt_tokens + (estimate_tokens(system_prompt) if system_prompt else 0)

    def log_telemetry(
        self,
        generation_name: Optional[str] = None,
        model: Optional[str] = None,
        provider_tokens: Optional[int] = None,
        output_tokens: int = 0
    ) -> None:
        """Record boundary telemetry to logs and Langfuse."""
        if isinstance(provider_tokens, (int, float)):
            self.provider_input_tokens = int(provider_tokens)
        else:
            self.provider_input_tokens = None
        if isinstance(output_tokens, (int, float)):
            self.output_tokens = int(output_tokens)
        else:
            self.output_tokens = 0

        actual_input = self.provider_input_tokens or self.final_input_tokens or self.compiled_context_tokens
        orig = max(self.original_context_tokens, 1)
        reduction_ratio = round(max(0.0, (orig - actual_input) / orig) * 100, 2)
        end_to_end_ratio = round(max(0.0, (orig - self.final_input_tokens) / orig) * 100, 2) if self.final_input_tokens else reduction_ratio

        # Structured debug log as mandated by specification
        logger.info(
            f"[GenerationContext] artifact={self.artifact_type} | "
            f"original_tokens={self.original_context_tokens} | "
            f"compiled_tokens={self.compiled_context_tokens} | "
            f"final_prompt_tokens={self.final_prompt_tokens} | "
            f"final_input_tokens={self.final_input_tokens} | "
            f"provider_input_tokens={self.provider_input_tokens or 'N/A'} | "
            f"output_tokens={self.output_tokens} | "
            f"context_reduction={reduction_ratio}% | "
            f"end_to_end_reduction={end_to_end_ratio}%"
        )
        logger.info(
            f"[ContextCompiler] compiler=GenerationContext({self.artifact_type}) | "
            f"reduction={reduction_ratio}%"
        )

        metrics = CompilationMetrics(
            artifact_type=self.artifact_type,
            compiler_used=f"GenerationContext.{self.artifact_type}",
            original_context_tokens=self.original_context_tokens,
            compiled_context_tokens=self.compiled_context_tokens,
            final_input_tokens=self.final_input_tokens,
            output_tokens=self.output_tokens,
            context_reduction_ratio=reduction_ratio,
            extra_details={
                "provider_input_tokens": self.provider_input_tokens,
                "end_to_end_prompt_reduction_ratio": end_to_end_ratio
            }
        )
        record_generation_telemetry(metrics, generation_name=generation_name, model=model)

    @classmethod
    def from_requirement_model(
        cls,
        req_model: Any,
        artifact_type: str = "planning",
        token_budget: int = 3500
    ) -> GenerationContext:
        """
        Compile RequirementIR into a compact GenerationContext.
        Filters out raw DocumentIR text blocks, full text chunks, and massive edge lists.
        """
        orig_tokens = estimate_tokens(req_model)
        
        ctx = cls(artifact_type=artifact_type)
        ctx.original_context_tokens = max(orig_tokens, 1)

        # 1. Design Intent & Metadata
        if hasattr(req_model, "doc_ir") and getattr(req_model.doc_ir, "summary", None):
            ctx.design_intent = req_model.doc_ir.summary[:300]
        elif isinstance(req_model, dict) and req_model.get("design_name"):
            ctx.design_intent = f"Design for {req_model.get('design_name')}"
        else:
            ctx.design_intent = "Synthesizable hardware module implementation"

        # 2. Extract Entities
        entities_src = getattr(req_model, "entities", []) if hasattr(req_model, "entities") else (req_model.get("entities", []) if isinstance(req_model, dict) else [])
        for e in entities_src:
            name = getattr(e, "name", None) or (e.get("name") if isinstance(e, dict) else str(e))
            etype = getattr(e, "type", None) or (e.get("type", "entity") if isinstance(e, dict) else "entity")
            desc = getattr(e, "description", None) or (e.get("description", "") if isinstance(e, dict) else "")
            
            # Map into target categories
            elower = etype.lower()
            if elower in ("signal", "port", "clock", "reset"):
                ctx.signals.append({
                    "name": name,
                    "direction": getattr(e, "direction", None) or (e.get("direction", "input") if isinstance(e, dict) else "input"),
                    "width": str(getattr(e, "width", None) or (e.get("width", "1") if isinstance(e, dict) else "1")),
                    "semantic_role": elower if elower in ("clock", "reset") else "data"
                })
            elif elower in ("register", "reg"):
                ctx.registers.append({
                    "name": name,
                    "offset": getattr(e, "offset", None) or (e.get("offset", "0x0") if isinstance(e, dict) else "0x0"),
                    "access": getattr(e, "access", None) or (e.get("access", "RW") if isinstance(e, dict) else "RW"),
                    "description": desc[:100] if desc else ""
                })
            elif elower in ("parameter", "param", "generic"):
                ctx.parameters.append({
                    "name": name,
                    "value": getattr(e, "value", None) or (e.get("value", "") if isinstance(e, dict) else "")
                })
            elif elower in ("opcode", "encoding"):
                ctx.opcodes.append({
                    "mnemonic": name,
                    "encoding": getattr(e, "value", None) or getattr(e, "encoding", None) or (e.get("encoding", "") if isinstance(e, dict) else "")
                })
            elif elower in ("instruction", "inst"):
                ctx.instructions.append({
                    "name": name,
                    "description": desc[:100] if desc else ""
                })
            elif elower in ("module", "component", "submodule", "unit", "controller"):
                ctx.entities.append({
                    "name": name,
                    "type": etype,
                    "purpose": desc[:120] if desc else ""
                })

        # 3. Extract Interfaces
        if hasattr(req_model, "interfaces") and req_model.interfaces:
            for iface in req_model.interfaces:
                i_dict = iface.to_dict() if hasattr(iface, "to_dict") else iface
                if isinstance(i_dict, dict):
                    ctx.interfaces.append({
                        "name": i_dict.get("name", "interface"),
                        "protocol": i_dict.get("protocol", "General"),
                        "signals": [s.get("name") if isinstance(s, dict) else str(s) for s in i_dict.get("signals", [])][:12]
                    })
        elif isinstance(req_model, dict) and req_model.get("interfaces"):
            for iface in req_model.get("interfaces", []):
                if isinstance(iface, dict):
                    ctx.interfaces.append({
                        "name": iface.get("name", "interface"),
                        "protocol": iface.get("protocol", "General")
                    })

        # 4. Extract Requirements / Functional Behaviors
        reqs_src = getattr(req_model, "requirements", []) if hasattr(req_model, "requirements") else (req_model.get("requirements", []) if isinstance(req_model, dict) else [])
        states_seen = set()
        transitions = []
        for r in reqs_src:
            st = getattr(r, "state", None) or (r.get("state") if isinstance(r, dict) else None)
            nxt = getattr(r, "next_state", None) or (r.get("next_state") if isinstance(r, dict) else None)
            stmt = getattr(r, "statement", None) or (r.get("statement", "") if isinstance(r, dict) else "")
            cond = getattr(r, "condition", None) or (r.get("condition") if isinstance(r, dict) else None)
            act = getattr(r, "action", None) or (r.get("action") if isinstance(r, dict) else None)
            rtype = getattr(r, "type", "") or (r.get("type", "") if isinstance(r, dict) else "")

            if st:
                states_seen.add(st)
            if nxt:
                states_seen.add(nxt)
            if st and nxt:
                transitions.append({
                    "from": st,
                    "to": nxt,
                    "condition": cond or stmt[:80],
                    "action": act or ""
                })
            elif rtype.upper() in ("BEHAVIOR", "FUNCTIONAL_BEHAVIOR", "SEQUENCE", "FLOW", "ALGORITHM"):
                if stmt:
                    ctx.behavior.append({
                        "statement": stmt[:150],
                        "condition": cond or "",
                        "action": act or ""
                    })
            elif rtype.upper() in ("CONSTRAINT", "HOLD", "GUARD", "TIMING"):
                if stmt:
                    ctx.constraints.append({"statement": stmt[:120]})

        if states_seen:
            ctx.fsm = {
                "states": list(states_seen),
                "transitions": transitions[:20]
            }

        # 5. Extract Conflicts
        conflicts_src = getattr(req_model, "conflicts", []) if hasattr(req_model, "conflicts") else (req_model.get("conflicts", []) if isinstance(req_model, dict) else [])
        for c in conflicts_src:
            c_dict = c.to_dict() if hasattr(c, "to_dict") else c
            if isinstance(c_dict, dict) and c_dict.get("resolution_status") == "UNRESOLVED":
                ctx.conflicts.append({
                    "title": c_dict.get("title", ""),
                    "description": c_dict.get("description", "")[:120]
                })

        # 6. Apply Priority Budgeting
        ctx.compiled_context_tokens = estimate_tokens(ctx.to_dict())
        if ctx.compiled_context_tokens > token_budget:
            # Prune secondary behavior descriptions
            if len(ctx.behavior) > 10:
                ctx.behavior = ctx.behavior[:10]
            if len(ctx.constraints) > 10:
                ctx.constraints = ctx.constraints[:10]
            ctx.compiled_context_tokens = estimate_tokens(ctx.to_dict())

        return ctx

    @classmethod
    def from_spec_inputs(
        cls,
        req_model: Any,
        plan: Dict[str, Any],
        protocol_rules: List[Dict[str, Any]],
        artifact_type: str = "design_spec",
        token_budget: int = 4000
    ) -> GenerationContext:
        """
        Compile inputs for formal design specification generation without full raw DocumentIR dumps.
        """
        base_ctx = cls.from_requirement_model(req_model, artifact_type=artifact_type, token_budget=token_budget // 2)
        
        # Add compact architecture plan
        filtered_plan = {}
        if isinstance(plan, dict):
            for k in ("system_architecture", "submodules", "clock_domains", "interface_mapping", "fsm_architecture", "registers_summary"):
                if k in plan:
                    filtered_plan[k] = plan[k]
        base_ctx.dependencies = filtered_plan.get("submodules", [])
        base_ctx.metadata["plan_overview"] = filtered_plan.get("system_architecture", "")
        if filtered_plan.get("clock_domains"):
            base_ctx.metadata["clock_domains"] = filtered_plan["clock_domains"]

        # Add sliced protocol rules
        sliced_rules = []
        for r in protocol_rules or []:
            if isinstance(r, dict):
                cit = r.get("citation") or r.get("chunk_id") or "spec"
                txt = r.get("text", "").strip()[:200]
                sliced_rules.append({"citation": cit, "rule": txt})
            elif isinstance(r, str):
                sliced_rules.append({"citation": "spec", "rule": r[:200]})
        base_ctx.protocol_rules = sliced_rules[:15]

        # Calculate original context size (req_model + plan + rules)
        orig = estimate_tokens(req_model) + estimate_tokens(plan) + estimate_tokens(protocol_rules)
        base_ctx.original_context_tokens = max(orig, 1)
        base_ctx.compiled_context_tokens = estimate_tokens(base_ctx.to_dict())
        return base_ctx

    @classmethod
    def from_compiled_rtl(cls, compiled_rtl: Dict[str, Any], token_budget: int = 8000) -> GenerationContext:
        """Adapt output of RTLContextCompiler to the GenerationContext contract."""
        mod = compiled_rtl.get("module", {})
        ctx = cls(
            artifact_type="rtl",
            target_component=mod.get("name", "target_module"),
            design_intent=mod.get("description", ""),
            signals=compiled_rtl.get("signals", compiled_rtl.get("ports", [])),
            registers=compiled_rtl.get("registers", []),
            fsm=compiled_rtl.get("fsm", {}),
            behavior=compiled_rtl.get("datapath", []),
            protocol_rules=compiled_rtl.get("protocol_rules", []),
            parameters=compiled_rtl.get("parameters", []),
            opcodes=compiled_rtl.get("opcodes", []),
            instructions=compiled_rtl.get("instructions", []),
            traceability=compiled_rtl.get("source_refs", [])
        )
        if "design_plan" in compiled_rtl:
            ctx.metadata["design_plan"] = compiled_rtl["design_plan"]
        if "clock_reset" in compiled_rtl:
            ctx.metadata["clock_reset"] = compiled_rtl["clock_reset"]

        # Populate entity provenance
        provenance_map: Dict[str, Any] = {}
        for s in compiled_rtl.get("signals", compiled_rtl.get("ports", [])):
            s_name = s.get("name") if isinstance(s, dict) else str(s)
            if s_name:
                w_prov = s.get("width_provenance", "explicit_numeric") if isinstance(s, dict) else "explicit"
                provenance_map[f"signal:{s_name}"] = {
                    "origin": "INHERITED" if w_prov != "inferred_bus_width" else "DERIVED",
                    "width_provenance": w_prov,
                    "source_ref": s.get("source_references", []) if isinstance(s, dict) else []
                }
        for r in compiled_rtl.get("registers", []):
            r_name = r.get("name") if isinstance(r, dict) else str(r)
            if r_name:
                provenance_map[f"register:{r_name}"] = {
                    "origin": "INHERITED" if (isinstance(r, dict) and r.get("source") == "EFS") else "DERIVED",
                    "reset_source": r.get("reset_source", "explicit") if isinstance(r, dict) else "explicit"
                }
        ctx.entity_provenance = provenance_map

        # Populate unresolved entities
        unresolved = list(compiled_rtl.get("unresolved_entities", []))
        if "unresolved_conflicts" in compiled_rtl:
            unresolved.extend(compiled_rtl["unresolved_conflicts"])
        ctx.unresolved_entities = unresolved

        # Populate projection rationale
        ctx.projection_rationale = {
            "target_component": mod.get("name", "target_module"),
            "component_type": mod.get("type", "module"),
            "is_top_level": mod.get("type") in ("top_level", "top_level_design", "top_module"),
            "compiled_ports_count": len(compiled_rtl.get("ports", [])),
            "compiled_registers_count": len(compiled_rtl.get("registers", [])),
            "has_fsm": bool(compiled_rtl.get("fsm")),
        }

        raw_metrics = compiled_rtl.get("_metrics", {})
        ctx.original_context_tokens = raw_metrics.get("original_context_tokens", estimate_tokens(compiled_rtl))
        ctx.compiled_context_tokens = estimate_tokens(ctx.to_dict())
        return ctx

    @classmethod
    def from_compiled_diagram(cls, compiled_diag: Dict[str, Any], token_budget: int = 4000) -> GenerationContext:
        """Adapt output of DiagramContextCompiler to the GenerationContext contract."""
        ctx = cls(
            artifact_type="diagram",
            target_component=compiled_diag.get("design_name", "system"),
            entities=compiled_diag.get("components", []),
            interfaces=compiled_diag.get("interfaces", []),
            signals=compiled_diag.get("signals", []),
            registers=compiled_diag.get("registers", []),
            fsm=compiled_diag.get("fsm", {}),
            behavior=compiled_diag.get("sequence_flows", [])
        )
        raw_metrics = compiled_diag.get("_metrics", {})
        ctx.original_context_tokens = raw_metrics.get("original_context_tokens", estimate_tokens(compiled_diag))
        ctx.compiled_context_tokens = estimate_tokens(ctx.to_dict())
        return ctx
