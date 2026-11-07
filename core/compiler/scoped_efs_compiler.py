"""
Scoped EFS IR Compiler module.

Compiles query-time retrieved design and protocol knowledge into a unified,
protocol-aware, traceable, scoped EFS Intermediate Representation (EFSIR).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from core.efs_ir.models import (
    EFSIR, EFSComponent, EFSInterface, EFSSignal, EFSRegister, EFSRegisterField,
    EFSInstruction, EFSOpcode, EFSFSM, EFSState, EFSTransition,
    EFSFunctionalBehavior, EFSProtocolRule, EFSConstraint, EFSTimingRule,
    EFSTraceability, EFSMetadata, EFSConflict, EFSSourceDocument
)
from core.retrieval.query_interpreter import ScopedQueryIntent
from core.retrieval.scope_resolver import ResolvedScope
from utils.logger import get_logger

logger = get_logger("compiler.scoped_efs")


class ScopedEFSCompiler:
    """
    Query-driven on-demand compiler that generates scoped EFS IR
    from retrieved design and protocol knowledge.
    """

    def __init__(
        self,
        intent: ScopedQueryIntent,
        scope: ResolvedScope,
        protocol_rules: Optional[List[Dict[str, Any]]] = None,
        design_spec: str = "",
        generation_run_id: str = "",
        doc_ir: Optional[Any] = None
    ):
        self.intent = intent
        self.scope = scope
        self.protocol_rules = protocol_rules or []
        self.design_spec = design_spec
        self.generation_run_id = generation_run_id or f"RUN_{int(time.time() * 1000)}"
        self.doc_ir = doc_ir

    def compile(self) -> EFSIR:
        """
        Compile resolved scope and protocol rules into an executable, scoped EFSIR.
        """
        logger.info(f"Compiling scoped EFS IR for target component '{self.scope.target_component_name}'...")

        primary_protocol = (
            self.intent.referenced_protocols[0]
            if self.intent.referenced_protocols
            else (self.scope.interfaces[0].get("protocol") if self.scope.interfaces else "GENERIC")
        ) or "GENERIC"

        # 1. Metadata
        doc_filename = ""
        doc_id = ""
        if self.doc_ir:
            doc_id = getattr(self.doc_ir, "document_id", "")
            meta = getattr(self.doc_ir, "metadata", {}) or {}
            doc_filename = meta.get("filename", "")

        metadata = EFSMetadata(
            ir_schema_version="2.0",
            design_id=f"DSN_{self.scope.target_component_name.upper().replace(' ', '_')}",
            design_name=self.scope.target_component_name,
            display_name=self.scope.target_component_name,
            top_module=self.scope.target_component_name,
            protocol=primary_protocol,
            protocol_version="1.0",
            description=f"Scoped hardware representation for {self.scope.target_component_name}",
            author="EFS Query-Driven Compiler",
            generation_run_id=self.generation_run_id,
            document_id=doc_id,
            source_documents=[EFSSourceDocument(name=doc_filename)] if doc_filename else [EFSSourceDocument(name="user_specification")],
        )

        # 2. Components (Target Component + Boundary Modules)
        components: List[EFSComponent] = []
        target_comp_id = f"COMP_{self.scope.target_component_name.upper().replace(' ', '_')}"
        target_comp = EFSComponent(
            component_id=target_comp_id,
            name=self.scope.target_component_name,
            type="top_level_design" if self.intent.requested_scope == "FULL_SYSTEM" else self.scope.target_component_type,
            description=f"Target hardware module: {self.scope.target_component_name}",
            clock_domain=self.scope.clock_domain or "clk",
            reset_domain=self.scope.reset_domain or "rst_n",
            interfaces=[i["name"] for i in self.scope.interfaces],
            traceability=EFSTraceability(
                doc_id=doc_id or "user_spec",
                chunk_id=self.scope.source_chunk_ids[0] if self.scope.source_chunk_ids else "",
            )
        )
        components.append(target_comp)

        for conn in self.scope.connected_components:
            components.append(EFSComponent(
                component_id=f"COMP_{conn['name'].upper().replace(' ', '_')}",
                name=conn["name"],
                type="submodule",
                description=f"Adjacent boundary component: {conn['name']}",
            ))

        # 3. Interfaces
        interfaces: List[EFSInterface] = []
        for iface in self.scope.interfaces:
            if_name = iface.get("name", "iface")
            interfaces.append(EFSInterface(
                interface_id=f"IF_{if_name.upper().replace(' ', '_')}",
                name=if_name,
                protocol=iface.get("protocol", primary_protocol),
                role=iface.get("role", "slave"),
                description=iface.get("description", f"Interface {if_name}"),
                traceability=EFSTraceability(
                    doc_id=doc_id or "user_spec",
                    chunk_id=iface.get("chunk_id", ""),
                )
            ))

        # 4. Signals and Ports
        signals: List[EFSSignal] = []
        clocks: List[EFSSignal] = []
        resets: List[EFSSignal] = []

        for p in self.scope.ports:
            p_name = p.get("name", "")
            if not p_name:
                continue

            sem_role = p.get("semantic_role", "data")
            is_clk = sem_role == "clock" or "clk" in p_name.lower()
            is_rst = sem_role == "reset" or "rst" in p_name.lower()

            sig_obj = EFSSignal(
                signal_id=f"SIG_{p_name.upper()}",
                name=p_name,
                width=str(p.get("width", 1)),
                direction=p.get("direction", "input"),
                datatype=p.get("datatype", "logic"),
                semantic_role=sem_role,
                clock=self.scope.clock_domain or "clk",
                reset=self.scope.reset_domain or "rst_n",
                owner=target_comp_id if p.get("direction") == "output" else "",
                consumer=target_comp_id if p.get("direction") == "input" else "",
                description=p.get("description", ""),
                traceability=EFSTraceability(
                    doc_id=p.get("doc_id") or doc_id or "user_spec",
                    chunk_id=p.get("chunk_id", ""),
                )
            )
            signals.append(sig_obj)
            if is_clk:
                clocks.append(sig_obj)
            if is_rst:
                resets.append(sig_obj)

        # 5. Registers and Fields
        registers: List[EFSRegister] = []
        for reg in self.scope.registers:
            r_name = reg.get("name", "")
            if not r_name:
                continue
            registers.append(EFSRegister(
                register_id=f"REG_{r_name.upper()}",
                name=r_name,
                address=reg.get("address", "0x0"),
                offset=reg.get("offset", "0x0"),
                width=reg.get("width", 32),
                access_type=reg.get("access_type", "RW"),
                description=reg.get("description", ""),
                owner=target_comp_id,
                traceability=EFSTraceability(
                    doc_id=doc_id or "user_spec",
                    chunk_id=reg.get("chunk_id", ""),
                )
            ))

        # 6. Instructions and Opcodes
        instructions: List[EFSInstruction] = []
        for inst in self.scope.instructions:
            i_name = inst.get("name", "")
            if not i_name:
                continue
            instructions.append(EFSInstruction(
                instruction_id=f"INSTR_{i_name.upper()}",
                name=i_name,
                width=inst.get("width", 32),
                description=inst.get("description", ""),
                traceability=EFSTraceability(
                    doc_id=doc_id or "user_spec",
                    chunk_id=inst.get("chunk_id", ""),
                )
            ))

        opcodes: List[EFSOpcode] = []
        for op in self.scope.opcodes:
            mnemonic = op.get("mnemonic", "")
            if not mnemonic:
                continue
            opcodes.append(EFSOpcode(
                opcode_id=f"OPC_{mnemonic.upper()}",
                mnemonic=mnemonic,
                binary_encoding=op.get("binary_encoding", ""),
                operation=op.get("operation", ""),
                traceability=EFSTraceability(
                    doc_id=doc_id or "user_spec",
                    chunk_id=op.get("chunk_id", ""),
                )
            ))

        # 7. FSMs
        fsms: List[EFSFSM] = []
        for f in self.scope.fsms:
            f_name = f.get("name", "MainFSM")
            states = [
                EFSState(state_id=f"ST_{s}", name=s)
                for s in f.get("states", ["IDLE"])
            ]
            fsms.append(EFSFSM(
                fsm_id=f"FSM_{f_name.upper()}",
                name=f_name,
                initial_state=f.get("initial_state", "IDLE"),
                states=states,
            ))

        # 8. Functional Behavior
        behaviors: List[EFSFunctionalBehavior] = []
        for idx, rule in enumerate(self.scope.behavioral_rules):
            behaviors.append(EFSFunctionalBehavior(
                behavior_id=rule.get("rule_id") or f"BEH_{idx+1:03d}",
                description=rule.get("statement", ""),
                trigger=rule.get("condition", ""),
                action=rule.get("action", ""),
                traceability=EFSTraceability(
                    doc_id=doc_id or "user_spec",
                    chunk_id=rule.get("source_ref", ""),
                )
            ))

        # 9. Structured Protocol Rules
        structured_proto_rules: List[EFSProtocolRule] = []
        proto_citations: List[str] = []
        proto_chunk_ids: List[str] = []

        flat_rules = []
        for r in self.protocol_rules:
            if isinstance(r, list):
                flat_rules.extend(r)
            else:
                flat_rules.append(r)

        for idx, rule in enumerate(flat_rules):
            r_text = ""
            r_proto = primary_protocol
            r_cid = ""
            r_cite = ""

            if isinstance(rule, dict):
                r_text = rule.get("text") or rule.get("description") or rule.get("statement") or str(rule)
                r_proto = rule.get("protocol") or primary_protocol
                r_cid = rule.get("chunk_id", "")
                r_cite = rule.get("citation", "")
            else:
                r_text = str(rule)

            if r_cid:
                proto_chunk_ids.append(r_cid)
            if r_cite:
                proto_citations.append(r_cite)

            # Classify rule type
            r_type = "GENERAL_PROTOCOL"
            r_lower = r_text.lower()
            if any(k in r_lower for k in ["valid", "ready", "handshake"]):
                r_type = "HANDSHAKE"
            elif any(k in r_lower for k in ["reset", "rst"]):
                r_type = "RESET"
            elif any(k in r_lower for k in ["timing", "hold", "setup", "delay"]):
                r_type = "TIMING"
            elif any(k in r_lower for k in ["burst", "aligned", "wrap"]):
                r_type = "BURST"
            elif any(k in r_lower for k in ["ordering", "order", "id"]):
                r_type = "ORDERING"

            structured_proto_rules.append(EFSProtocolRule(
                rule_id=f"PR_{idx+1:03d}",
                name=f"{r_proto} {r_type} Rule #{idx+1}",
                type=r_type,
                description=r_text,
                details={
                    "protocol": r_proto,
                    "severity": "MANDATORY",
                    "source_references": [r_cid] if r_cid else ([r_cite] if r_cite else []),
                    "applies_to_signals": [s.name for s in signals if any(k in s.name.lower() for k in ["valid", "ready", "addr", "data", "resp"])],
                },
                traceability=EFSTraceability(
                    doc_id="protocol_specification",
                    chunk_id=r_cid,
                )
            ))

        # 10. Protocol Context Slice
        protocol_context = {
            "protocol_name": primary_protocol,
            "interfaces": [i.to_dict() for i in interfaces],
            "channels": [
                "read_address", "write_address", "write_data",
                "write_response", "read_data"
            ] if "AXI" in primary_protocol.upper() else ["transfer"],
            "handshake_rules": [r.description for r in structured_proto_rules if r.type == "HANDSHAKE"],
            "timing_rules": [r.description for r in structured_proto_rules if r.type == "TIMING"],
            "ordering_rules": [r.description for r in structured_proto_rules if r.type == "ORDERING"],
            "citations": sorted(set(proto_citations)),
            "source_chunk_ids": sorted(set(proto_chunk_ids)),
        }

        # 11. Preserved Specification Gaps & Conflicts
        conflicts: List[EFSConflict] = []
        for gap_msg in self.scope.specification_gaps:
            conflicts.append(EFSConflict(
                conflict_id=f"GAP_{len(conflicts)+1}",
                category="SPECIFICATION_GAP",
                severity="WARNING",
                description=gap_msg,
                affected_entities=[self.scope.target_component_name],
                blocking=False,
                resolution_status="UNRESOLVED",
            ))

        efs_ir = EFSIR(
            metadata=metadata,
            components=components,
            interfaces=interfaces,
            signals=signals,
            clocks=clocks,
            resets=resets,
            registers=registers,
            instructions=instructions,
            opcodes=opcodes,
            fsms=fsms,
            functional_behavior=behaviors,
            protocol_rules=structured_proto_rules,
            protocol_context=protocol_context,
            conflicts=conflicts,
        )

        logger.info(
            f"Successfully compiled scoped EFS IR: {len(components)} components, "
            f"{len(signals)} signals, {len(registers)} registers, {len(opcodes)} opcodes, "
            f"{len(structured_proto_rules)} protocol rules."
        )

        return efs_ir
