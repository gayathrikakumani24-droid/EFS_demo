"""
Generic EFS IR Compiler (Sections 15, 16, 17, 21, 22, 23, 29).

Compiles protocol-independent Requirement IR into canonical structured EFS IR.
Maps generic semantic constructs (entities, instructions, opcodes, memory regions,
data structures, errors, performance, state transitions, conditions, events, transactions)
dynamically into EFS IR models with full source traceability.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional
from core.efs_ir.models import (
    EFSIR, EFSAssertionIntent, EFSComponent, EFSConflict, EFSConstraint,
    EFSCoverageIntent, EFSDataStructure, EFSError, EFSExecutionModel, EFSFSM, EFSFlow,
    EFSFlowStep, EFSInstruction, EFSInstructionField, EFSInstructionFormat, EFSInterface,
    EFSMemoryRegion, EFSMetadata, EFSOpcode, EFSPerformanceRequirement, EFSProtocolRule,
    EFSRegister, EFSRegisterField, EFSRequirement, EFSSignal, EFSState, EFSTimingRule,
    EFSTraceability, EFSTransaction, EFSTransition, new_efs_id
)
from core.requirement_ir.models import (
    AtomicRequirement, DiscoveredEntity, RequirementIR, SpecConflict
)
from utils.logger import get_logger

logger = get_logger("compiler.efs_compiler")


class RequirementToEFSCompiler:
    """Generic Specification Compiler producing canonical EFS IR from Requirement IR."""

    def __init__(self, req_ir: RequirementIR):
        self.req_ir = req_ir
        self.efs = EFSIR()
        self.entity_map: Dict[str, str] = {}  # name -> efs_id

    def compile(self) -> EFSIR:
        """Execute end-to-end generic compilation sequence."""
        # Source Validation Gate
        req_doc_id = self.req_ir.document_id or self.req_ir.doc_ir.document_id
        doc_ir_id = self.req_ir.doc_ir.document_id
        if req_doc_id and doc_ir_id and req_doc_id != doc_ir_id:
            logger.error(f"DOCUMENT_ID_MISMATCH: RequirementIR ({req_doc_id}) != DocumentIR ({doc_ir_id})")
            raise ValueError(f"DOCUMENT_ID_MISMATCH: RequirementIR document_id '{req_doc_id}' does not match DocumentIR '{doc_ir_id}'")

        self._compile_metadata()
        self._compile_entities()
        self._compile_requirements_and_behaviors()
        self._compile_conflicts()
        return self.efs

    def _compile_metadata(self) -> None:
        doc_id = self.req_ir.doc_ir.document_id
        doc_hash = self.req_ir.doc_ir.document_hash or self.req_ir.document_hash
        ing_id = self.req_ir.doc_ir.ingestion_id or self.req_ir.ingestion_id
        filename = self.req_ir.doc_ir.metadata.get("filename") or f"{doc_id}.{self.req_ir.doc_ir.source_type}"
        self.efs.metadata = EFSMetadata(
            design_name=f"Design_{doc_id[:8]}",
            protocol="discovered_generic",
            document_id=doc_id,
            document_hash=doc_hash,
            ingestion_id=ing_id,
            source_documents=[filename],
            specification_revision=self.req_ir.doc_ir.document_version,
            ir_schema_version="2.0"
        )

    def _compile_entities(self) -> None:
        """Map discovered entities into canonical EFS IR arrays."""
        for ent in self.req_ir.entities:
            t = ent.type.lower()
            traceability = self._create_traceability_for_entity(ent)

            # 1. Modules & Components
            if t in ("module", "subsystem", "component", "arbiter", "controller", "engine", "scheduler", "queue"):
                comp_id = new_efs_id("comp")
                comp = EFSComponent(
                    component_id=comp_id,
                    name=ent.name,
                    type=t,
                    description=ent.description,
                    traceability=traceability
                )
                self.efs.components.append(comp)
                self.entity_map[ent.name] = comp_id

            # 2. Interfaces & Channels
            elif t in ("interface", "channel"):
                if_id = new_efs_id("if")
                iface = EFSInterface(
                    interface_id=if_id,
                    name=ent.name,
                    protocol="discovered_protocol",
                    role=ent.attributes.get("role", "initiator"),
                    description=ent.description,
                    traceability=traceability
                )
                self.efs.interfaces.append(iface)
                self.entity_map[ent.name] = if_id

            # 3. Signals & Ports
            elif t in ("signal", "port", "clock", "reset"):
                sig_id = new_efs_id("sig")
                w = str(ent.attributes.get("width", "1"))
                d = str(ent.attributes.get("direction", "input"))
                role = "clock" if t == "clock" else ("reset" if t == "reset" else "data")
                sig = EFSSignal(
                    signal_id=sig_id,
                    name=ent.name,
                    width=w,
                    direction=d,
                    semantic_role=role,
                    description=ent.description,
                    traceability=traceability
                )
                self.efs.signals.append(sig)
                self.entity_map[ent.name] = sig_id

            # 4. Registers & Register Fields
            elif t in ("register", "reg", "field"):
                reg_id = new_efs_id("reg")
                offset = str(ent.attributes.get("offset", "0x0"))
                access = str(ent.attributes.get("access", "RW"))
                reg = EFSRegister(
                    register_id=reg_id,
                    name=ent.name,
                    offset=offset,
                    access_type=access,
                    description=ent.description,
                    traceability=traceability
                )
                self.efs.registers.append(reg)
                self.entity_map[ent.name] = reg_id

            # 5. Instructions
            elif t in ("instruction", "inst", "mnemonic"):
                inst_id = new_efs_id("instr")
                op_val = str(ent.attributes.get("opcode", ""))
                fmt_val = str(ent.attributes.get("format", ""))
                inst = EFSInstruction(
                    instruction_id=inst_id,
                    name=ent.name,
                    mnemonic=ent.name,
                    opcode=op_val,
                    format=fmt_val,
                    description=ent.description,
                    traceability=traceability
                )
                self.efs.instructions.append(inst)
                self.entity_map[ent.name] = inst_id

            # 5b. Instruction Formats
            elif t in ("instruction_format", "format"):
                fmt_id = new_efs_id("fmt")
                fld_objs = [EFSInstructionField(name=f.get("name", ""), width=int(f.get("width", 0)))
                            for f in ent.attributes.get("fields", [])]
                fmt_obj = EFSInstructionFormat(
                    format_id=fmt_id,
                    name=ent.name,
                    total_width=int(ent.attributes.get("total_width", 32)),
                    fields=fld_objs,
                    traceability=traceability
                )
                self.efs.instruction_formats.append(fmt_obj)
                self.entity_map[ent.name] = fmt_id

            # 6. Opcodes
            elif t in ("opcode", "encoding"):
                op_id = new_efs_id("opc")
                op_hex = str(ent.attributes.get("opcode_hex", ent.attributes.get("value", ent.name)))
                opc = EFSOpcode(
                    opcode_id=op_id,
                    mnemonic=str(ent.attributes.get("instruction", ent.name)),
                    binary_encoding=op_hex,
                    operation=ent.description,
                    traceability=traceability
                )
                self.efs.opcodes.append(opc)
                self.entity_map[ent.name] = op_id

            # 7. Memory Regions & Caches
            elif t in ("memory", "memory_region", "memory_resource", "sram", "dram", "cache", "buffer", "fifo"):
                mem_id = new_efs_id("memreg")
                base_addr = str(ent.attributes.get("base_address", "0x0"))
                sz = str(ent.attributes.get("size", "0"))
                mem = EFSMemoryRegion(
                    region_id=mem_id,
                    name=ent.name,
                    type=t,
                    base_address=base_addr,
                    size=sz,
                    description=ent.description,
                    traceability=traceability
                )
                self.efs.memory_regions.append(mem)
                self.entity_map[ent.name] = mem_id

            # 8. Data Structures & Descriptors
            elif t in ("data_structure", "descriptor", "packet", "message"):
                ds_id = new_efs_id("dstruct")
                fld_dicts = [{"name": f} if isinstance(f, str) else f for f in ent.attributes.get("fields", [])]
                ds = EFSDataStructure(
                    structure_id=ds_id,
                    name=ent.name,
                    fields=fld_dicts,
                    description=ent.description,
                    traceability=traceability
                )
                self.efs.data_structures.append(ds)
                self.entity_map[ent.name] = ds_id

            # 8b. Execution Strategy / Model
            elif t in ("execution_strategy", "execution_model", "scheduling_policy"):
                ex_id = new_efs_id("exec")
                ex_obj = EFSExecutionModel(
                    execution_id=ex_id,
                    name=ent.name,
                    type=str(ent.attributes.get("type", "PIPELINE")),
                    description=ent.description,
                    traceability=traceability
                )
                self.efs.execution_models.append(ex_obj)
                self.entity_map[ent.name] = ex_id

            # 9. Errors & Faults
            elif t in ("error", "exception", "fault"):
                err_id = new_efs_id("err")
                err_code = str(ent.attributes.get("code", ""))
                err = EFSError(
                    error_id=err_id,
                    name=ent.name,
                    code=err_code,
                    condition=ent.description,
                    traceability=traceability
                )
                self.efs.errors.append(err)
                self.entity_map[ent.name] = err_id

            # 10. Performance / Power / Security Requirements
            elif t in ("performance_requirement", "power_requirement", "security_requirement"):
                perf_id = new_efs_id("perf")
                metric = str(ent.attributes.get("metric", t.replace("_requirement", "")))
                val = str(ent.attributes.get("target_value", str(ent.attributes.get("value", ""))))
                unit = str(ent.attributes.get("unit", ""))
                target = f"{val} {unit}".strip()
                perf = EFSPerformanceRequirement(
                    perf_id=perf_id,
                    metric=metric,
                    target_value=target or ent.name,
                    condition=ent.description,
                    traceability=traceability
                )
                self.efs.performance_requirements.append(perf)
                self.entity_map[ent.name] = perf_id

    def _compile_requirements_and_behaviors(self) -> None:
        """Map generic requirements into EFS FSMs, Constraints, Flows, Requirements, and Timings."""
        states_map: Dict[str, EFSState] = {}
        transitions: List[EFSTransition] = []
        flow_steps: List[EFSFlowStep] = []

        for req in self.req_ir.requirements:
            t = req.type.upper()
            traceability = EFSTraceability(
                doc_id=req.source.document_id,
                chunk_id=req.source.block_id,
                page=req.source.page,
                section=req.source.section,
                original_text=req.statement,
                extraction_method=req.extraction_method,
                confidence=req.confidence
            )

            # Map into general EFSRequirement
            efs_req = EFSRequirement(
                req_id=new_efs_id("req"),
                title=f"Requirement_{req.requirement_id}",
                category=req.knowledge_status,
                requirement_type=t.lower(),
                target_object=req.subject or (req.entities[0] if req.entities else ""),
                description=req.statement,
                is_critical=(t in ("INTERFACE", "HOLD", "CONSTRAINT", "ERROR")),
                traceability=traceability
            )
            self.efs.requirements.append(efs_req)

            # 1. State transition mapping (state / next_state -> EFS FSM)
            st_name = req.state or ""
            next_st = req.next_state or ""
            if not (st_name and next_st) and "transition" in req.statement.lower():
                trans_m = re.search(r"transition(?:s)?\s+from\s+([A-Za-z0-9_]+)\s+to\s+([A-Za-z0-9_]+)", req.statement, re.IGNORECASE)
                if trans_m:
                    st_name = trans_m.group(1).upper()
                    next_st = trans_m.group(2).upper()

            if st_name and next_st:
                if st_name not in states_map:
                    states_map[st_name] = EFSState(name=st_name)
                if next_st not in states_map:
                    states_map[next_st] = EFSState(name=next_st)

                transitions.append(EFSTransition(
                    transition_id=new_efs_id("trans"),
                    source_state=st_name,
                    target_state=next_st,
                    condition=req.condition or req.statement,
                    action=req.action or "",
                    traceability=traceability
                ))

            # 2. Timing / Timeout mapping
            if req.timing or t in ("TIMING", "TIMEOUT"):
                rule = EFSTimingRule(
                    rule_id=new_efs_id("time"),
                    type="timeout" if "timeout" in req.statement.lower() else "latency",
                    cycle_relationship=req.timing or req.statement,
                    traceability=traceability
                )
                self.efs.timing_rules.append(rule)

            # 3. Constraints & Handshake conditions
            if req.condition or t in ("CONSTRAINT", "HOLD", "GUARD"):
                const = EFSConstraint(
                    constraint_id=new_efs_id("const"),
                    type="handshake" if "assert" in req.statement.lower() else "generic_constraint",
                    condition=req.condition or req.statement,
                    expected_behavior=req.statement,
                    traceability=traceability,
                    confidence=req.confidence
                )
                self.efs.constraints.append(const)

            # 4. Transactions & Sequences
            if t in ("TRANSACTION", "SEQUENCE", "FLOW"):
                trans_id = new_efs_id("txn")
                txn = EFSTransaction(
                    transaction_id=trans_id,
                    name=req.subject or f"Transaction_{req.requirement_id}",
                    type=t.lower(),
                    description=req.statement,
                    traceability=traceability
                )
                self.efs.transactions.append(txn)

            # 5. Behavioral Flow steps
            step = EFSFlowStep(
                step_id=new_efs_id("fstep"),
                action_description=req.statement,
                condition=req.condition or "",
                signal_behavior=req.action or ""
            )
            flow_steps.append(step)

        # Assemble compiled FSM if state transitions were discovered
        if states_map and transitions:
            fsm = EFSFSM(
                fsm_id=new_efs_id("fsm"),
                name="Discovered_Controller_FSM",
                states=list(states_map.values()),
                transitions=transitions,
                initial_state="IDLE" if "IDLE" in states_map else list(states_map.keys())[0],
                description="Synthesized state machine derived from requirement specification rules."
            )
            self.efs.fsms.append(fsm)

            # Auto-synthesize missing control signals referenced in transition conditions
            known_sig_names = {s.name.lower() for s in self.efs.signals}
            for tr in transitions:
                cond_text = tr.condition or ""
                ref_signals = re.findall(r"\b([a-zA-Z0-9_]{2,30})\b", cond_text)
                for sig in ref_signals:
                    sig_lower = sig.lower()
                    if sig_lower not in known_sig_names and sig_lower not in ("when", "is", "asserted", "the", "and", "or", "to", "from", "valid", "ready", "high", "low", "1", "0", "true", "false", "rule", "transition", "if"):
                        known_sig_names.add(sig_lower)
                        self.efs.signals.append(EFSSignal(
                            signal_id=new_efs_id("sig"),
                            name=sig,
                            width="1",
                            direction="input",
                            datatype="logic",
                            semantic_role="control",
                            description=f"Auto-derived control trigger for FSM transition '{tr.source_state} -> {tr.target_state}'",
                            traceability=tr.traceability
                        ))

        # Assemble compiled behavioral Flow
        if flow_steps:
            flow = EFSFlow(
                flow_id=new_efs_id("flow"),
                name="Discovered_Transaction_Flow",
                ordered_steps=flow_steps
            )
            self.efs.flows.append(flow)

    def _compile_conflicts(self) -> None:
        """Map requirement IR conflicts into EFSConflicts."""
        for c in self.req_ir.conflicts:
            efs_conf = EFSConflict(
                conflict_id=c.conflict_id,
                type=c.type,
                severity=c.severity,
                description=c.description,
                conflicting_values=c.conflicting_values,
                resolution_status=c.status,
                traceability=EFSTraceability(
                    doc_id=c.sources[0].document_id if c.sources else None,
                    chunk_id=c.sources[0].block_id if c.sources else None
                )
            )
            self.efs.conflicts.append(efs_conf)

    def _create_traceability_for_entity(self, ent: DiscoveredEntity) -> EFSTraceability:
        doc_id = self.req_ir.doc_ir.document_id
        doc_hash = self.req_ir.doc_ir.document_hash or self.req_ir.document_hash
        ing_id = self.req_ir.doc_ir.ingestion_id or self.req_ir.ingestion_id
        if ent.source_references:
            loc = ent.source_references[0]
            return EFSTraceability(
                doc_id=loc.document_id or doc_id,
                document_hash=loc.document_hash or doc_hash,
                ingestion_id=ing_id,
                chunk_id=loc.block_id,
                block_id=loc.block_id,
                page=loc.page,
                section=loc.section,
                original_text=loc.original_text,
                source_entity_id=ent.entity_id,
                extraction_method="compiler_mapped",
                confidence=0.95
            )
        return EFSTraceability(
            doc_id=doc_id,
            document_hash=doc_hash,
            ingestion_id=ing_id,
            extraction_method="compiler_mapped",
            confidence=0.8
        )
