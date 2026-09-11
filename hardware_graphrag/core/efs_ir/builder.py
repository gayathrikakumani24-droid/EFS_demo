"""
EFS IR Builder.

Builds a canonical EFSIR instance from the Requirement Model, Architecture Plan,
retrieved protocol rules, and the ingestion registry (to establish source traceability).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from core.registry import get_registry
from core.efs_ir.models import (
    EFSIR, EFSMetadata, EFSTraceability, EFSComponent, EFSInterface, EFSSignal,
    EFSRegister, EFSRegisterField, EFSFSM, EFSState, EFSTransition,
    EFSConstraint, EFSTimingRule, EFSProtocolRule, EFSFlow, EFSFlowStep,
    EFSAssertionIntent, EFSCoverageIntent, new_efs_id,
    EFSInstruction, EFSInstructionField, EFSOpcode, EFSMemoryRegion,
    EFSDataStructure, EFSError, EFSPerformanceRequirement, EFSConflict
)


class EFSIRBuilder:
    """Builder class to assemble the EFS IR from pipeline and design flow artifacts."""

    def __init__(
        self,
        requirement_model: Optional[Dict[str, Any]] = None,
        plan: Optional[Dict[str, Any]] = None,
        protocol_rules: Optional[List[Dict[str, Any]]] = None,
        design_spec: str = "",
        req_model: Optional[Dict[str, Any]] = None
    ):
        self.req_model = requirement_model or req_model or {}
        self.plan = plan or {}
        self.protocol_rules = protocol_rules or []
        self.design_spec = design_spec
        
        self.registry = get_registry()
        self.ir = EFSIR()
        
        # Helper mappings for ID resolution
        self.name_to_component_id: Dict[str, str] = {}
        self.name_to_interface_id: Dict[str, str] = {}
        self.name_to_signal_id: Dict[str, str] = {}
        self.name_to_register_id: Dict[str, str] = {}

    def _find_entity_traceability(self, entity_name: str) -> EFSTraceability:
        """Scan the pipeline registry to find source document / chunk traceability for a name."""
        entity_name_lower = entity_name.lower().replace("_", " ")
        for doc_id, entities in self.registry.entities.items():
            for ent in entities:
                ent_name_lower = ent.name.lower().replace("_", " ")
                # Match canonical name or aliases
                if ent_name_lower == entity_name_lower or any(a.lower().replace("_", " ") == entity_name_lower for a in ent.aliases):
                    return EFSTraceability(
                        doc_id=doc_id,
                        chunk_id=ent.chunk_id,
                        page=ent.page,
                        chapter=ent.chapter,
                        section=ent.section,
                        original_text=ent.original_text,
                        source_entity_id=ent.entity_id,
                        extraction_method="pipeline_normalized",
                        confidence=0.95
                    )
        return EFSTraceability(
            doc_id="user_spec",
            chunk_id="USER_SPEC_TEXT",
            original_text=f"Specification entry for '{entity_name}'",
            extraction_method="heuristic_extracted",
            confidence=0.85
        )

    def _find_relationship_traceability(self, source: str, target: str) -> EFSTraceability:
        """Scan the pipeline registry for relationships between two entities for traceability."""
        src_lower = source.lower()
        tgt_lower = target.lower()
        for doc_id, rels in self.registry.relationships.items():
            for rel in rels:
                if rel.source.lower() == src_lower and rel.target.lower() == tgt_lower:
                    return EFSTraceability(
                        doc_id=doc_id,
                        chunk_id=rel.chunk_id,
                        source_relationship_id=rel.rel_id,
                        original_text=rel.evidence,
                        extraction_method="pipeline_relation",
                        confidence=0.9
                    )
        return EFSTraceability(extraction_method="heuristic_relationship", confidence=0.4)

    def build(self) -> EFSIR:
        """Execute the end-to-end building sequence."""
        self._build_metadata()
        self._build_components()
        self._build_interfaces_and_signals()
        self._build_registers()
        self._build_instructions()
        self._build_opcodes()
        self._build_memory_regions()
        self._build_data_structures()
        self._build_errors()
        self._build_performance_requirements()
        self._build_fsms()
        self._build_protocol_rules_and_constraints()
        self._build_flows()
        self._build_assertion_intent()
        self._build_coverage_intent()
        self._build_conflicts()
        return self.ir

    def _build_metadata(self):
        """Build design metadata with canonical ID, display name, and aliases."""
        import re
        proto = "generic"
        if self.req_model.get("protocol_references"):
            proto = self.req_model["protocol_references"][0]
            
        docs = []
        if self.registry.documents:
            docs = [doc.filename for doc in self.registry.documents.values()]
            
        # Extract display name from spec text header, req_model, or plan
        raw_display_name = ""
        if self.design_spec:
            name_match = re.search(r"# Hardware Design Specification:\s*([A-Za-z0-9_\-\s]+)", self.design_spec)
            if name_match:
                raw_display_name = name_match.group(1).strip()
        if not raw_display_name and self.req_model.get("title"):
            raw_display_name = self.req_model["title"]
        if not raw_display_name:
            raw_display_name = self.plan.get("system_architecture", "Controller_Design").split()[0].replace(",", "").replace(".", "")
            
        display_name = raw_display_name or "Target_Module"
        design_name = re.sub(r"[^a-zA-Z0-9_]", "_", display_name).strip("_").lower()
        canonical_id = re.sub(r"[^a-zA-Z0-9]", "", display_name).lower()
        
        # Build comprehensive alias list
        aliases = []
        for name_variant in (display_name, design_name, canonical_id):
            if name_variant and name_variant not in aliases:
                aliases.append(name_variant)
                
        # Add camelCase / PascalCase / snake_case variants
        words = [w for w in re.split(r"[^a-zA-Z0-9]+", display_name) if w]
        if words:
            pascal_name = "".join(w.capitalize() for w in words)
            camel_name = words[0].lower() + "".join(w.capitalize() for w in words[1:])
            snake_name = "_".join(w.lower() for w in words)
            for v in (pascal_name, camel_name, snake_name):
                if v and v not in aliases:
                    aliases.append(v)
                    
            # Non-prefix and inflection alias variants (e.g. RegisteredControlledDataEngine)
            core_words = [w for w in words if w.lower() not in ("simple", "generic", "custom", "top", "main")]
            if core_words:
                core_pascal = "".join(w.capitalize() for w in core_words)
                for v in (core_pascal, core_pascal.replace("Register", "Registered"), core_pascal.replace("Registered", "Register")):
                    if v and v not in aliases:
                        aliases.append(v)

        self.ir.metadata = EFSMetadata(
            design_name=design_name,
            canonical_id=canonical_id,
            display_name=display_name,
            aliases=aliases,
            protocol=proto,
            source_documents=docs,
            specification_revision="1.0",
            ir_schema_version="1.0"
        )

    def _build_components(self):
        """Map architecture plan submodules into EFSComponents and register top-level design target."""
        # 1. Add submodules from plan
        for sub in self.plan.get("submodules", []):
            comp_id = new_efs_id("comp")
            name = sub.get("name", "Unnamed_Submodule")
            comp = EFSComponent(
                component_id=comp_id,
                name=name,
                type=name.lower(),
                description=sub.get("purpose", ""),
                traceability=self._find_entity_traceability(name)
            )
            self.ir.components.append(comp)
            self.name_to_component_id[name] = comp_id
            
        # 2. Add memories from requirement model as components
        for mem in self.req_model.get("memories", []):
            comp_id = new_efs_id("comp")
            name = mem.get("name", "FIFO_Buffer")
            comp = EFSComponent(
                component_id=comp_id,
                name=name,
                type="memory",
                description=f"{mem.get('description', '')} (Depth: {mem.get('depth', 0)}, Width: {mem.get('width', 0)})",
                traceability=self._find_entity_traceability(name)
            )
            self.ir.components.append(comp)
            self.name_to_component_id[name] = comp_id

        # 3. Add Top-Level Composite Design Component if not already present
        top_name = getattr(self.ir.metadata, "display_name", "") or getattr(self.ir.metadata, "design_name", "top_module")
        if not any(c.name.lower() == top_name.lower() for c in self.ir.components):
            top_comp_id = new_efs_id("comp")
            top_comp = EFSComponent(
                component_id=top_comp_id,
                name=top_name,
                type="top_level_design",
                description=f"Top-Level Composite Design Module: {top_name}",
                traceability=self._find_entity_traceability(top_name)
            )
            self.ir.components.insert(0, top_comp)
            self.name_to_component_id[top_name] = top_comp_id

    def _build_interfaces_and_signals(self):
        """Build port signals and group interfaces."""
        # Determine clock and reset signal configurations
        clk_sigs = []
        rst_sigs = []
        for cr in self.req_model.get("clocks_resets", []):
            sig_id = new_efs_id("sig")
            name = cr.get("name", "clk")
            sig = EFSSignal(
                signal_id=sig_id,
                name=name,
                width="1",
                direction="input",
                datatype="logic",
                semantic_role="clock" if cr.get("type") == "clock" else "reset",
                description=cr.get("description", ""),
                traceability=self._find_entity_traceability(name)
            )
            self.ir.signals.append(sig)
            self.name_to_signal_id[name] = sig_id
            if cr.get("type") == "clock":
                clk_sigs.append(name)
            else:
                rst_sigs.append(name)

        primary_clk = clk_sigs[0] if clk_sigs else "clk"
        primary_rst = rst_sigs[0] if rst_sigs else "rst_n"

        # Map external interface port list from requirement model or design spec text
        ports = self.req_model.get("interfaces", [])
        if not ports and self.design_spec:
            try:
                from core.agents.rtl_agent import parse_markdown_spec
                parsed_spec = parse_markdown_spec(self.design_spec)
                ports = parsed_spec.get("interfaces", [])
            except Exception:
                pass

        interface_signals_ids = []
        for port in ports:
            name = str(port.get("name", "")).strip("` \t\r\n'\"")
            if name.lower() in ("name", "signal", "port", "signal name", "port name", "signal_name", "port_name", "pin", "width", "direction", "description", "protocol/description"):
                continue
            from core.verification.compatibility_engine import ProtocolRoleMapper

            sig_id = new_efs_id("sig")
            raw_dir = str(port.get("direction", "")).strip().lower()
            
            # Infer interface role (slave vs master)
            iface_role = "slave" if "slave" in self.design_spec.lower() or "target" in self.design_spec.lower() or "slave" in str(self.req_model).lower() else "master"
            role_info = ProtocolRoleMapper.get_semantic_role(name, iface_role)

            if "in" in raw_dir and "out" in raw_dir:
                direction = "inout"
            elif "out" in raw_dir:
                direction = "output"
            elif "in" in raw_dir:
                direction = "input"
            elif role_info:
                # Deduce correct direction from protocol role mapping if raw direction is absent/ambiguous
                direction = role_info[2]
            else:
                direction = "input"

            width = str(port.get("width", "1"))
            semantic_role = role_info[1] if role_info else ("control" if any(k in name.lower() for k in ("valid", "ready", "enable", "start", "done", "busy")) else "data")
            
            sig = EFSSignal(
                signal_id=sig_id,
                name=name,
                width=width,
                direction=direction,
                datatype="logic",
                clock=primary_clk,
                reset=primary_rst,
                description=port.get("description", ""),
                semantic_role=semantic_role,
                traceability=self._find_entity_traceability(name)
            )
            self.ir.signals.append(sig)
            self.name_to_signal_id[name] = sig_id
            interface_signals_ids.append(sig_id)

        # Create a generic external interface connection block based strictly on extracted signals
        if interface_signals_ids:
            iface_id = new_efs_id("if")
            iface = EFSInterface(
                interface_id=iface_id,
                name="external_bus_interface",
                protocol=self.ir.metadata.protocol,
                role="slave" if "slave" in self.design_spec.lower() or "target" in self.design_spec.lower() else "master",
                signals=interface_signals_ids,
                clock=primary_clk,
                reset=primary_rst,
                description="Primary bus port boundary interface.",
                traceability=self._find_entity_traceability(self.ir.metadata.protocol)
            )
            self.ir.interfaces.append(iface)
            self.name_to_interface_id[iface.name] = iface_id
            
            # Map interface back to components
            for comp in self.ir.components:
                if "controller" in comp.name.lower() or "fsm" in comp.name.lower():
                    comp.interfaces.append(iface_id)

    def _build_registers(self):
        """Map requirement model register layout."""
        for reg in self.req_model.get("registers", []):
            reg_id = new_efs_id("reg")
            name = reg.get("name", "CONTROL_REG")
            
            reg_width = reg.get("width", 32)
            if isinstance(reg_width, str):
                try:
                    reg_width = int(reg_width)
                except ValueError:
                    reg_width = 32

            fields = []
            used_bits = set()
            next_available_bit = 0

            for f in reg.get("fields", []):
                msb, lsb = None, None
                if "msb" in f and "lsb" in f:
                    try:
                        msb = int(f["msb"])
                        lsb = int(f["lsb"])
                    except (TypeError, ValueError):
                        pass

                if msb is None:
                    bits_str = str(f.get("bits", "")).strip()
                    if ":" in bits_str:
                        try:
                            parts = bits_str.replace("[", "").replace("]", "").split(":")
                            msb, lsb = int(parts[0]), int(parts[1])
                        except ValueError:
                            pass
                    elif bits_str.isdigit():
                        try:
                            msb = lsb = int(bits_str)
                        except ValueError:
                            pass

                f_width = f.get("width", 1)
                if isinstance(f_width, str) and f_width.isdigit():
                    f_width = int(f_width)
                if not isinstance(f_width, int) or f_width < 1:
                    f_width = abs((msb or 0) - (lsb or 0)) + 1 if (msb is not None and lsb is not None) else 1

                # If msb/lsb missing or overlaps already allocated bits, resolve to next available bit offset
                if msb is None or lsb is None:
                    while any((next_available_bit + i) in used_bits for i in range(f_width)):
                        next_available_bit += 1
                    lsb = next_available_bit
                    msb = lsb + f_width - 1
                else:
                    low, high = min(lsb, msb), max(lsb, msb)
                    if any(b in used_bits for b in range(low, high + 1)):
                        while any((next_available_bit + i) in used_bits for i in range(f_width)):
                            next_available_bit += 1
                        lsb = next_available_bit
                        msb = lsb + f_width - 1

                low, high = min(lsb, msb), max(lsb, msb)
                for b in range(low, high + 1):
                    used_bits.add(b)
                next_available_bit = max(next_available_bit, high + 1)

                fields.append(EFSRegisterField(
                    name=f.get("name", "FIELD"),
                    msb=high,
                    lsb=low,
                    width=(high - low + 1),
                    access=f.get("access", reg.get("access", "RW")),
                    reset_value=str(f.get("reset_val", f.get("reset_value", "0"))),
                    description=f.get("description", "")
                ))

            max_field_msb = max([f.msb for f in fields], default=31)
            if max_field_msb >= reg_width:
                reg_width = max_field_msb + 1

            efs_reg = EFSRegister(
                register_id=reg_id,
                name=name,
                offset=reg.get("offset", "0x0"),
                width=reg_width,
                access_type=reg.get("access", "RW"),
                fields=fields,
                description=reg.get("description", ""),
                traceability=self._find_entity_traceability(name)
            )
            self.ir.registers.append(efs_reg)
            self.name_to_register_id[name] = reg_id

        # If no explicit registers were parsed from the document, keep registers as empty array without synthetic defaults

    def _build_fsms(self):
        """Build Finite State Machine structures."""
        fsm_info = self.req_model.get("fsm_info", {})
        if not fsm_info:
            return

        states = []
        for state_name in fsm_info.get("states", []):
            state_id = new_efs_id("st")
            
            # Find actions or outputs related to this state from the plan
            plan_actions = ""
            for s in self.plan.get("fsm_architecture", {}).get("states", []):
                if isinstance(s, dict) and s.get("name") == state_name:
                    plan_actions = s.get("actions", "")
            
            states.append(EFSState(
                state_id=state_id,
                name=state_name,
                actions=[plan_actions] if plan_actions else []
            ))

        transitions = []
        for trans in fsm_info.get("transitions", []):
            trans_id = new_efs_id("trans")
            transitions.append(EFSTransition(
                transition_id=trans_id,
                source_state=trans.get("from"),
                target_state=trans.get("to"),
                condition=trans.get("condition", "1"),
                traceability=self._find_relationship_traceability(trans.get("from"), trans.get("to"))
            ))

        fsm_id = new_efs_id("fsm")
        fsm = EFSFSM(
            fsm_id=fsm_id,
            name="Main_Controller_FSM",
            states=states,
            initial_state=states[0].name if states else "IDLE",
            transitions=transitions,
            traceability=self._find_entity_traceability("state machine")
        )
        self.ir.fsms.append(fsm)

    def _build_protocol_rules_and_constraints(self):
        """Map retrieved protocol spec chunks and relationships to ProtocolRules and Constraints."""
        # 1. Map retrieved protocol rules from database
        for r in self.protocol_rules:
            rule_id = new_efs_id("prule")
            self.ir.protocol_rules.append(EFSProtocolRule(
                rule_id=rule_id,
                name=r.get("citation", "Protocol Clause"),
                type="handshake" if "handshake" in r.get("text", "").lower() else "general",
                description=r.get("text", ""),
                traceability=EFSTraceability(
                    chunk_id=r.get("chunk_id"),
                    original_text=r.get("text"),
                    extraction_method="graphrag_retrieval",
                    confidence=r.get("score", 1.0)
                )
            ))

        # 2. Extract constraints from protocol rules or specifications
        # For example, AXI4 handshake stability or VALID/READY dependencies
        for p_sig in self.ir.signals:
            if "valid" in p_sig.name.lower():
                # Add handshake stability constraint
                const_id = new_efs_id("const")
                self.ir.constraints.append(EFSConstraint(
                    constraint_id=const_id,
                    type="stability",
                    source_objects=[p_sig.name],
                    condition=f"asserted({p_sig.name})",
                    expected_behavior=f"stable({p_sig.name}) until handshaked",
                    severity="CRITICAL",
                    protocol=self.ir.metadata.protocol,
                    traceability=self._find_entity_traceability(p_sig.name)
                ))

        # 3. Add timing constraints from requirement model
        for idx, tc in enumerate(self.req_model.get("timing_constraints", [])):
            rule_id = new_efs_id("time")
            self.ir.timing_rules.append(EFSTimingRule(
                rule_id=rule_id,
                type="cycle_relationship" if "cycle" in tc.lower() else "latency",
                cycle_relationship=tc,
                traceability=self._find_entity_traceability("timing")
            ))

    def _build_flows(self):
        """Construct transactional Flows representing sequence steps."""
        fsm_info = self.req_model.get("fsm_info", {})
        if not fsm_info or not fsm_info.get("states"):
            return

        steps = []
        for idx, state in enumerate(fsm_info.get("states", [])):
            step_id = new_efs_id("fstep")
            steps.append(EFSFlowStep(
                step_id=step_id,
                action_description=f"Controller transitions to state {state} to run operations.",
                state_transitions=[state]
            ))

        flow_id = new_efs_id("flow")
        flow = EFSFlow(
            flow_id=flow_id,
            name="Primary_Operation_Flow",
            ordered_steps=steps,
            traceability=self._find_entity_traceability("flow")
        )
        self.ir.flows.append(flow)

    def _build_assertion_intent(self):
        """Compile SVA intent assertions from constraints without hardcoded protocol fallbacks."""
        primary_clk = next((s.name for s in self.ir.signals if s.semantic_role == "clock"), "clk")
        primary_rst = next((s.name for s in self.ir.signals if s.semantic_role == "reset"), "rst_n")

        for const in self.ir.constraints:
            if const.type in ("stability", "handshake"):
                ast_id = new_efs_id("ast")
                valid_sigs = [s.name for s in self.ir.signals if "valid" in s.name.lower()]
                rdy_sigs = [s.name for s in self.ir.signals if "ready" in s.name.lower()]

                if const.source_objects:
                    sig_name = const.source_objects[0]
                elif valid_sigs:
                    sig_name = valid_sigs[0]
                else:
                    continue  # Skip handshake assertion if no signal is specified or present

                matched_rdy = rdy_sigs[0] if rdy_sigs else ""
                antecedent_str = f"{sig_name} && !{matched_rdy}" if matched_rdy else f"asserted({sig_name})"

                self.ir.assertions.append(EFSAssertionIntent(
                    assertion_id=ast_id,
                    derived_constraint_id=const.constraint_id,
                    property_type="assert",
                    clock=primary_clk,
                    reset_condition=primary_rst,
                    antecedent=antecedent_str,
                    consequent=f"$stable({sig_name})",
                    temporal_behavior="implication",
                    severity="error",
                    traceability=const.traceability
                ))

    def _build_coverage_intent(self):
        """Map coverage items to FSM states and transitions."""
        for fsm in self.ir.fsms:
            # 1. State Coverage
            cov_id = new_efs_id("cov")
            self.ir.coverage.append(EFSCoverageIntent(
                coverage_id=cov_id,
                type="state",
                target_object_id=fsm.fsm_id,
                description=f"Verify coverage of all states in FSM '{fsm.name}': {', '.join([s.name for s in fsm.states])}.",
                traceability=fsm.traceability
            ))

            # 2. Transition Coverage
            cov_id = new_efs_id("cov")
            self.ir.coverage.append(EFSCoverageIntent(
                coverage_id=cov_id,
                type="transition",
                target_object_id=fsm.fsm_id,
                description=f"Verify coverage of all state-to-state transitions in FSM '{fsm.name}'.",
                traceability=fsm.traceability
            ))

    def _build_instructions(self):
        """Map instructions from req_model."""
        for instr in self.req_model.get("instructions", []):
            instr_id = new_efs_id("instr")
            name = instr.get("name", "instruction")
            fields = []
            for f in instr.get("fields", []):
                fields.append(EFSInstructionField(
                    name=f.get("name", "FIELD"),
                    width=f.get("width", 0),
                    msb=f.get("msb", 0),
                    lsb=f.get("lsb", 0),
                    description=f.get("description", "")
                ))
            self.ir.instructions.append(EFSInstruction(
                instruction_id=instr_id,
                name=name,
                width=instr.get("width", 32),
                fields=fields,
                description=instr.get("description", ""),
                traceability=self._find_entity_traceability(name)
            ))

    def _build_opcodes(self):
        """Map opcodes from req_model."""
        for opc in self.req_model.get("opcodes", []):
            opc_id = new_efs_id("opc")
            mnemonic = opc.get("mnemonic", "OPC")
            self.ir.opcodes.append(EFSOpcode(
                opcode_id=opc_id,
                mnemonic=mnemonic,
                binary_encoding=opc.get("binary_encoding", "UNKNOWN"),
                operation=opc.get("operation", ""),
                operands=opc.get("operands", []),
                traceability=self._find_entity_traceability(mnemonic)
            ))

    def _build_memory_regions(self):
        """Map memory regions from req_model."""
        for mr in self.req_model.get("memory_regions", []):
            mr_id = new_efs_id("memreg")
            name = mr.get("name", "memory_region")
            self.ir.memory_regions.append(EFSMemoryRegion(
                region_id=mr_id,
                name=name,
                type=mr.get("type", "SRAM"),
                base_address=mr.get("base_address", ""),
                size=mr.get("size", ""),
                depth=mr.get("depth", 0),
                width=mr.get("width", 0),
                description=mr.get("description", ""),
                traceability=self._find_entity_traceability(name)
            ))

    def _build_data_structures(self):
        """Map data structures from req_model."""
        for ds in self.req_model.get("data_structures", []):
            ds_id = new_efs_id("dstruct")
            name = ds.get("name", "data_structure")
            self.ir.data_structures.append(EFSDataStructure(
                structure_id=ds_id,
                name=name,
                fields=ds.get("fields", []),
                description=ds.get("description", ""),
                traceability=self._find_entity_traceability(name)
            ))

    def _build_errors(self):
        """Map errors from req_model."""
        for err in self.req_model.get("errors", []):
            err_id = new_efs_id("err")
            name = err.get("name", "error")
            self.ir.errors.append(EFSError(
                error_id=err_id,
                name=name,
                code=err.get("code", ""),
                condition=err.get("condition", ""),
                behavior=err.get("behavior", ""),
                traceability=self._find_entity_traceability(name)
            ))

    def _build_performance_requirements(self):
        """Map performance requirements from req_model."""
        for pr in self.req_model.get("performance_requirements", []):
            pr_id = new_efs_id("perf")
            metric = pr.get("metric", "throughput")
            self.ir.performance_requirements.append(EFSPerformanceRequirement(
                perf_id=pr_id,
                metric=metric,
                target_value=pr.get("target_value", ""),
                condition=pr.get("condition", ""),
                traceability=self._find_entity_traceability(metric)
            ))

    def _build_conflicts(self):
        """Map extracted conflicts and perform programmatic checks."""
        # 1. Map conflicts from req_model
        for conf in self.req_model.get("spec_conflicts", []):
            conflict_id = new_efs_id("conf")
            self.ir.conflicts.append(EFSConflict(
                conflict_id=conflict_id,
                type=conf.get("type", "AMBIGUITY"),
                severity=conf.get("severity", "MAJOR"),
                source_objects=conf.get("source_objects", []),
                evidence=conf.get("evidence", ""),
                description=conf.get("description", ""),
                candidates=conf.get("candidates", []),
                conflicting_values=conf.get("conflicting_values", []),
                resolution_status="UNRESOLVED",
                traceability=self._find_entity_traceability(conf.get("type", "conflict"))
            ))

        # 2. Programmatic Check: Duplicate Opcode Assignments
        opcode_map = {}
        for opc in self.ir.opcodes:
            enc = opc.binary_encoding.strip()
            if enc and enc != "UNKNOWN":
                opcode_map.setdefault(enc, []).append(opc)
                
        for enc, opcs in opcode_map.items():
            if len(opcs) > 1:
                conflict_id = new_efs_id("conf")
                mnemonics = [o.mnemonic for o in opcs]
                ids = [o.opcode_id for o in opcs]
                self.ir.conflicts.append(EFSConflict(
                    conflict_id=conflict_id,
                    type="DUPLICATE_OPCODE",
                    severity="CRITICAL",
                    source_objects=ids,
                    evidence=f"Opcode binary value '{enc}' is assigned to multiple instructions: {', '.join(mnemonics)}",
                    description=f"Duplicate opcode assignment detected for value {enc}",
                    candidates=mnemonics,
                    conflicting_values=[enc],
                    resolution_status="UNRESOLVED",
                    traceability=EFSTraceability(extraction_method="programmatic_check", confidence=1.0)
                ))

        # 3. Programmatic Check: Overlapping Register Fields
        for reg in self.ir.registers:
            field_bits = {}
            for f in reg.fields:
                for bit in range(f.lsb, f.msb + 1):
                    field_bits.setdefault(bit, []).append(f.name)
            overlaps = {bit: names for bit, names in field_bits.items() if len(names) > 1}
            if overlaps:
                conflict_id = new_efs_id("conf")
                overlap_desc = "; ".join([f"bit {b} in {', '.join(names)}" for b, names in overlaps.items()])
                self.ir.conflicts.append(EFSConflict(
                    conflict_id=conflict_id,
                    type="OVERLAPPING_REGISTER_FIELDS",
                    severity="CRITICAL",
                    source_objects=[reg.register_id],
                    evidence=f"Register '{reg.name}' has overlapping field definitions: {overlap_desc}",
                    description=f"Register '{reg.name}' has fields sharing the same bit locations",
                    resolution_status="UNRESOLVED",
                    traceability=EFSTraceability(extraction_method="programmatic_check", confidence=1.0)
                ))
