"""
Source-Backed Recovery Engine for EFS IR.

Searches the original document source evidence index before completeness validation.
If missing properties are present in the source document, recovers them into EFS IR
with proper status ("explicit", "partial", or "inferred") and verbatim source provenance.

RULE: The validator/recovery engine NEVER invents arbitrary hardware values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from core.efs_ir.models import (
    EFSIR, EFSComponent, EFSConstraint, EFSFSM, EFSFlow, EFSFlowStep, EFSInterface,
    EFSMemoryRegion, EFSPerformanceRequirement, EFSRegister, EFSRegisterField, EFSSignal, EFSState,
    EFSTimingRule, EFSTraceability, EFSTransaction, EFSTransition, new_efs_id
)
from core.extraction.dedicated_extractors import (
    COMMUNICATION_RELATIONS, DedicatedConstraintExtractor, DedicatedFSMExtractor,
    DedicatedInterfaceExtractor, DedicatedRegisterExtractor, DedicatedRelationshipExtractor
)
from core.efs_ir.validation_rules import (
    is_fsm_state_name,
    is_fsm_transition_or_condition,
    is_physical_signal_name,
    is_hardware_component_candidate,
    is_valid_interface_candidate,
)
from core.extraction.source_evidence import SourceEvidence, SourceEvidenceIndex
from core.requirement_ir.models import DocumentBlock, DocumentIR
from utils.logger import get_logger

logger = get_logger("extraction.source_recovery")


@dataclass
class RecoveryRecord:
    property_name: str
    target_entity: str
    status: str  # explicit | partial | inferred | unresolved
    recovered_items_count: int = 0
    source_evidence: Optional[SourceEvidence] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "property_name": self.property_name,
            "target_entity": self.target_entity,
            "status": self.status,
            "recovered_items_count": self.recovered_items_count,
            "source_evidence": self.source_evidence.to_dict() if self.source_evidence else None,
            "notes": self.notes
        }


@dataclass
class SourceRecoveryReport:
    total_properties_checked: int = 0
    explicit_recovered: int = 0
    partial_recovered: int = 0
    inferred_recovered: int = 0
    unresolved_gaps: int = 0
    records: List[RecoveryRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_properties_checked": self.total_properties_checked,
            "explicit_recovered": self.explicit_recovered,
            "partial_recovered": self.partial_recovered,
            "inferred_recovered": self.inferred_recovered,
            "unresolved_gaps": self.unresolved_gaps,
            "records": [r.to_dict() for r in self.records]
        }


class SourceBackedRecovery:
    """
    Executes source-backed recovery pass on EFS IR using the source document.
    Ensures that properties explicitly present in the source are not falsely
    flagged as missing or hallucinated.
    """

    @classmethod
    def recover_efs_ir(
        cls,
        efs_ir: EFSIR,
        doc_ir: DocumentIR,
        evidence_index: Optional[SourceEvidenceIndex] = None
    ) -> SourceRecoveryReport:
        if evidence_index is None:
            evidence_index = SourceEvidenceIndex(doc_ir)

        report = SourceRecoveryReport()
        doc_name = doc_ir.metadata.get("filename", "specification")

        # ----------------------------------------------------------------------
        # 1. Clock & Reset Recovery
        # ----------------------------------------------------------------------
        report.total_properties_checked += 1
        clk_signals = [s for s in (efs_ir.signals or []) if (s.semantic_role or "").lower() == "clock" or "clk" in s.name.lower()]
        rst_signals = [s for s in (efs_ir.signals or []) if (s.semantic_role or "").lower() == "reset" or "rst" in s.name.lower()]

        if not clk_signals:
            ev_list = evidence_index.search(["clock", "clk", "system clock"], match_any=True)
            if ev_list:
                ev = ev_list[0]
                # Extract signal name if present e.g. "input wire clk" or "Clock : System clock"
                clk_name = "clk"
                m = re.search(r"\b([a-zA-Z0-9_]*clk[a-zA-Z0-9_]*)\b", ev.text, re.I)
                if m:
                    clk_name = m.group(1)

                trace = EFSTraceability(
                    doc_id=doc_ir.document_id,
                    source_document=doc_name,
                    block_id=ev.block_id,
                    section=ev.section,
                    original_text=ev.text,
                    extraction_method="source_backed_recovery",
                    confidence=1.0
                )
                sig = EFSSignal(
                    signal_id=new_efs_id("sig"),
                    name=clk_name,
                    width="1",
                    direction="input",
                    semantic_role="clock",
                    description=f"Clock signal recovered from source: {ev.text}",
                    traceability=trace
                )
                efs_ir.signals.append(sig)
                if not any(s.signal_id == sig.signal_id for s in efs_ir.clocks):
                    efs_ir.clocks.append(sig)
                report.explicit_recovered += 1
                report.records.append(RecoveryRecord("clock", clk_name, "explicit", 1, ev, "Recovered clock from source text"))
            else:
                report.unresolved_gaps += 1
                report.records.append(RecoveryRecord("clock", "Clock", "unresolved", 0, None, "No clock evidence in source"))

        if not rst_signals:
            ev_list = evidence_index.search(["reset", "rst", "asynchronous reset"], match_any=True)
            if ev_list:
                ev = ev_list[0]
                rst_name = "rst_n"
                m = re.search(r"\b([a-zA-Z0-9_]*rst[a-zA-Z0-9_]*)\b", ev.text, re.I)
                if m:
                    rst_name = m.group(1)
                elif "asynchronous" in ev.text.lower() or "active-low" in ev.text.lower():
                    rst_name = "rst_n"
                else:
                    rst_name = "rst"

                trace = EFSTraceability(
                    doc_id=doc_ir.document_id,
                    source_document=doc_name,
                    block_id=ev.block_id,
                    section=ev.section,
                    original_text=ev.text,
                    extraction_method="source_backed_recovery",
                    confidence=1.0
                )
                sig = EFSSignal(
                    signal_id=new_efs_id("sig"),
                    name=rst_name,
                    width="1",
                    direction="input",
                    semantic_role="reset",
                    description=f"Reset signal recovered from source: {ev.text}",
                    traceability=trace
                )
                efs_ir.signals.append(sig)
                if not any(s.signal_id == sig.signal_id for s in efs_ir.resets):
                    efs_ir.resets.append(sig)
                report.explicit_recovered += 1
                report.records.append(RecoveryRecord("reset", rst_name, "explicit", 1, ev, "Recovered reset from source text"))
            else:
                report.unresolved_gaps += 1
                report.records.append(RecoveryRecord("reset", "Reset", "unresolved", 0, None, "No reset evidence in source"))

        # ----------------------------------------------------------------------
        # 2. Registers Recovery
        # ----------------------------------------------------------------------
        report.total_properties_checked += 1
        if not efs_ir.registers:
            recovered_regs = DedicatedRegisterExtractor.extract_registers(doc_ir, evidence_index)
            if recovered_regs:
                for r in recovered_regs:
                    trace = EFSTraceability(
                        doc_id=doc_ir.document_id,
                        source_document=doc_name,
                        block_id=r.evidence.block_id,
                        section=r.evidence.section,
                        original_text=r.evidence.text,
                        extraction_method="source_backed_recovery",
                        confidence=0.95
                    )
                    reg_fields = []
                    for f in getattr(r, "fields", []):
                        if isinstance(f, dict):
                            reg_fields.append(EFSRegisterField(
                                name=f.get("name", ""),
                                msb=f.get("msb", 0),
                                lsb=f.get("lsb", 0),
                                width=f.get("width", 1),
                                access=f.get("access", r.access),
                                description=f.get("description", "")
                            ))
                        elif isinstance(f, EFSRegisterField):
                            reg_fields.append(f)

                    reg_obj = EFSRegister(
                        register_id=new_efs_id("reg"),
                        name=r.name,
                        address=r.address,
                        offset=r.offset,
                        offset_source="explicit" if r.offset else "missing",
                        width=r.width or 32,
                        access=r.access,
                        access_type=r.access,
                        reset_value=r.reset_value,
                        fields=reg_fields,
                        description=r.description,
                        traceability=trace
                    )
                    efs_ir.registers.append(reg_obj)

                has_explicit = any(r.status == "explicit" for r in recovered_regs)
                stat = "explicit" if has_explicit else "partial"
                if stat == "explicit":
                    report.explicit_recovered += 1
                else:
                    report.partial_recovered += 1

                first_ev = recovered_regs[0].evidence if recovered_regs else None
                report.records.append(RecoveryRecord("registers", f"{len(recovered_regs)} registers", stat, len(recovered_regs), first_ev, "Recovered registers from source specification"))
            else:
                report.records.append(RecoveryRecord("registers", "Registers", "not_applicable", 0, None, "No register evidence found in source"))

        # ----------------------------------------------------------------------
        # 3. Interfaces & Component Relationships Recovery
        # ----------------------------------------------------------------------
        report.total_properties_checked += 1
        known_comp_names = [c.name for c in (efs_ir.components or [])]
        known_reg_names = [r.name for r in (efs_ir.registers or [])]
        known_mem_names = [m.name for m in (getattr(efs_ir, "memory_regions", []) or [])]

        accepted_rels, rejected_rels, unresolved_rels = DedicatedRelationshipExtractor.extract_candidates(
            doc_ir=doc_ir,
            known_components=known_comp_names,
            known_registers=known_reg_names,
            known_memories=known_mem_names,
            evidence_index=evidence_index
        )

        # 1. Process rejected candidate descriptions: preserve as component/register descriptions
        for rej in rejected_rels:
            if rej.category == "component_description":
                for comp in efs_ir.components:
                    if comp.name and (comp.name.lower() == (rej.resolved_source or rej.source).lower()):
                        if not comp.description or comp.description.startswith("Component referenced"):
                            comp.description = rej.evidence.text
                        break
            elif rej.category == "register_description":
                for reg in efs_ir.registers:
                    if reg.name and (reg.name.lower() == (rej.resolved_source or rej.source).lower()):
                        if not reg.description:
                            reg.description = rej.evidence.text
                        break

        # 2. Process accepted relationships: instantiate EFSInterface for communication, EFSFlow for flows
        recovered_interfaces_count = 0
        recovered_flows_count = 0

        for rel in accepted_rels:
            src_name = rel.resolved_source or rel.source
            dst_name = rel.resolved_target or rel.target

            # Hard semantic boundary: If either endpoint is a signal or FSM state, do NOT create interface or component!
            if is_fsm_state_name(src_name) or is_fsm_state_name(dst_name) or is_physical_signal_name(src_name) or is_physical_signal_name(dst_name):
                logger.info(f"Skipping relationship {src_name} -> {dst_name}: endpoint is signal or FSM state.")
                continue

            _known_comp_set = set(known_comp_names) | {c.name for c in efs_ir.components}

            # Verify both entities exist in valid EFS IR entities (components or memory regions)
            src_comp = next((c for c in efs_ir.components if c.name.lower() == src_name.lower()), None)
            dst_comp = next((c for c in efs_ir.components if c.name.lower() == dst_name.lower()), None)

            # Only match against pre-existing known architectural components, never arbitrary text
            if not src_comp and any(c.lower() == src_name.lower() for c in known_comp_names):
                src_comp = next((c for c in efs_ir.components if c.name.lower() == src_name.lower()), None)
            if not dst_comp and any(c.lower() == dst_name.lower() for c in known_comp_names):
                dst_comp = next((c for c in efs_ir.components if c.name.lower() == dst_name.lower()), None)

            # If either component is not in efs_ir, register it ONLY if it is a valid architectural component
            if not src_comp:
                if not is_hardware_component_candidate(src_name, known_components=_known_comp_set):
                    continue
                c_id = new_efs_id("comp")
                src_comp = EFSComponent(
                    component_id=c_id,
                    name=src_name,
                    type="architectural_component",
                    description=f"Architectural component from {rel.evidence.section}",
                    traceability=EFSTraceability(
                        doc_id=doc_ir.document_id,
                        source_document=doc_name,
                        block_id=rel.evidence.block_id,
                        section=rel.evidence.section,
                        original_text=rel.evidence.text,
                        extraction_method="source_backed_recovery"
                    )
                )
                efs_ir.components.append(src_comp)

            if not dst_comp and not any(m.name.lower() == dst_name.lower() for m in getattr(efs_ir, "memory_regions", [])):
                if not is_hardware_component_candidate(dst_name, known_components=_known_comp_set):
                    continue
                c_id = new_efs_id("comp")
                dst_comp = EFSComponent(
                    component_id=c_id,
                    name=dst_name,
                    type="architectural_component",
                    description=f"Architectural component from {rel.evidence.section}",
                    traceability=EFSTraceability(
                        doc_id=doc_ir.document_id,
                        source_document=doc_name,
                        block_id=rel.evidence.block_id,
                        section=rel.evidence.section,
                        original_text=rel.evidence.text,
                        extraction_method="source_backed_recovery"
                    )
                )
                efs_ir.components.append(dst_comp)

            has_valid_src = bool(src_comp or any(c.name.lower() == src_name.lower() for c in efs_ir.components))
            has_valid_dst = bool(
                dst_comp
                or any(c.name.lower() == dst_name.lower() for c in efs_ir.components)
                or any(m.name.lower() == dst_name.lower() for m in getattr(efs_ir, "memory_regions", []))
            )

            # RULE: Both entities must resolve to valid hardware entities before creating interface or flow
            if not (has_valid_src and has_valid_dst):
                logger.info(f"Skipping relationship: unvalidated endpoint (src={src_name}: {has_valid_src}, dst={dst_name}: {has_valid_dst})")
                continue

            # Populate canonical efs_ir.relationships for ALL valid accepted hardware relationships
            if not hasattr(efs_ir, "relationships") or efs_ir.relationships is None:
                efs_ir.relationships = []
            rel_item = {
                "source": src_name,
                "target": dst_name,
                "relation": rel.relation,
                "category": rel.category,
                "signals": list(rel.signals),
                "direction": rel.direction or f"{src_name} -> {dst_name}",
                "evidence": getattr(rel.evidence, "text", "") if rel.evidence else ""
            }
            if not any(r.get("source") == src_name and r.get("target") == dst_name and r.get("relation") == rel.relation for r in efs_ir.relationships):
                efs_ir.relationships.append(rel_item)

            valid_if, rej_reason = is_valid_interface_candidate(
                src_name, dst_name, {"relation": rel.relation}, known_components=_known_comp_set
            )
            is_interface = valid_if and (rel.category == "interface" or rel.relation in COMMUNICATION_RELATIONS)

            if is_interface:
                clean_if_name = f"{src_name}_{rel.relation}_{dst_name}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                if not any(
                    (i.source_component == src_name and getattr(i, "destination_component", None) == dst_name)
                    or i.name == clean_if_name
                    for i in efs_ir.interfaces
                ):
                    if_id = new_efs_id("if")
                    efs_ir.interfaces.append(EFSInterface(
                        interface_id=if_id,
                        name=clean_if_name,
                        protocol="interconnect",
                        source_component=src_name,
                        destination_component=dst_name,
                        signals=list(rel.signals),
                        direction=rel.direction or f"{src_name} -> {dst_name}",
                        description=f"{src_name} {rel.relation} {dst_name} (evidence: {rel.evidence.text})",
                        traceability=EFSTraceability(
                            doc_id=doc_ir.document_id,
                            source_document=doc_name,
                            block_id=rel.evidence.block_id,
                            section=rel.evidence.section,
                            original_text=rel.evidence.text,
                            extraction_method="source_backed_recovery"
                        )
                    ))
                    recovered_interfaces_count += 1

                # Ensure interface signals are registered in efs_ir.signals
                for sig in rel.signals:
                    if sig and not any(s.name.lower() == sig.lower() for s in efs_ir.signals):
                        sig_dir = "output" if "sends" in rel.relation else ("input" if "returns" in rel.relation else "inout")
                        
                        # Preserve explicit width from source evidence if present; do not invent width
                        recovered_width = "unspecified"
                        if rel.evidence and getattr(rel.evidence, "text", None):
                            ev_text = str(rel.evidence.text)
                            m_bit = re.search(r"\b(\d+)\s*(?:bits?|bit|-bit)\b", ev_text, re.IGNORECASE)
                            if m_bit:
                                recovered_width = m_bit.group(1)
                            else:
                                m_range = re.search(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]", ev_text)
                                if m_range:
                                    recovered_width = str(abs(int(m_range.group(1)) - int(m_range.group(2))) + 1)
                        
                        efs_ir.signals.append(EFSSignal(
                            signal_id=new_efs_id("sig"),
                            name=sig,
                            direction=sig_dir,
                            width=recovered_width,
                            semantic_role="data",
                            owner=src_comp.component_id if src_comp else None,
                            consumer=dst_comp.component_id if dst_comp else None,
                            description=f"Interface signal connecting {src_name} to {dst_name}",
                            traceability=EFSTraceability(
                                doc_id=doc_ir.document_id,
                                source_document=doc_name,
                                section=rel.evidence.section,
                                original_text=rel.evidence.text,
                                extraction_method="source_backed_recovery"
                            )
                        ))
            else:
                clean_flow_name = f"{src_name}_{rel.relation}_{dst_name}".replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                if not any(f.name == clean_flow_name for f in efs_ir.flows):
                    flow_id = new_efs_id("flow")
                    efs_ir.flows.append(EFSFlow(
                        flow_id=flow_id,
                        name=clean_flow_name,
                        components=[src_name, dst_name],
                        ordered_steps=[
                            EFSFlowStep(
                                component=src_name,
                                action_description=f"{src_name} {rel.relation} {dst_name}",
                                condition=f"evidence: {rel.evidence.text}"
                            )
                        ],
                        traceability=EFSTraceability(
                            doc_id=doc_ir.document_id,
                            source_document=doc_name,
                            block_id=rel.evidence.block_id,
                            section=rel.evidence.section,
                            original_text=rel.evidence.text,
                            extraction_method="source_backed_recovery"
                        )
                    ))
                    recovered_flows_count += 1

        if recovered_interfaces_count > 0 or recovered_flows_count > 0:
            report.explicit_recovered += 1
            tot = recovered_interfaces_count + recovered_flows_count
            first_ev = accepted_rels[0].evidence if accepted_rels else None
            report.records.append(RecoveryRecord(
                "relationships",
                f"{tot} qualified relationships ({recovered_interfaces_count} interfaces, {recovered_flows_count} flows)",
                "explicit",
                tot,
                first_ev,
                f"Recovered {recovered_interfaces_count} interfaces and {recovered_flows_count} flows; rejected {len(rejected_rels)} candidate descriptions"
            ))
        elif len(efs_ir.components) > 1 and not any(i.source_component for i in efs_ir.interfaces):
            report.unresolved_gaps += 1
            report.records.append(RecoveryRecord("relationships", "Module Relationships", "unresolved", 0, None, "Multi-component design without relational evidence"))

        # Dedicated Interface recovery if interfaces list is empty
        if not efs_ir.interfaces:
            rec_ifaces = DedicatedInterfaceExtractor.extract_interfaces(doc_ir, evidence_index)
            if rec_ifaces:
                for iface_data in rec_ifaces:
                    if_id = new_efs_id("if")
                    efs_ir.interfaces.append(EFSInterface(
                        interface_id=if_id,
                        name=iface_data.name,
                        protocol=iface_data.protocol,
                        role=iface_data.direction,
                        signals=iface_data.signals,
                        description=iface_data.purpose,
                        traceability=EFSTraceability(
                            doc_id=doc_ir.document_id,
                            source_document=doc_name,
                            block_id=iface_data.source_evidence.block_id,
                            section=iface_data.source_evidence.section,
                            original_text=iface_data.source_evidence.text,
                            extraction_method="source_backed_recovery"
                        )
                    ))
                report.explicit_recovered += 1
                report.records.append(RecoveryRecord("interfaces", f"{len(rec_ifaces)} interfaces", "explicit", len(rec_ifaces), rec_ifaces[0].source_evidence, "Recovered explicit hardware interfaces"))

        # ----------------------------------------------------------------------
        # 4. Constraints & Timing Recovery
        # ----------------------------------------------------------------------
        report.total_properties_checked += 1
        has_constraints = bool(
            (efs_ir.constraints and len(efs_ir.constraints) > 0)
            or (getattr(efs_ir, "timing_rules", None) and len(getattr(efs_ir, "timing_rules")) > 0)
            or (getattr(efs_ir, "performance_requirements", None) and len(getattr(efs_ir, "performance_requirements")) > 0)
        )
        if not has_constraints:
            rec_constraints = DedicatedConstraintExtractor.extract_constraints(doc_ir, evidence_index)
            if rec_constraints:
                for c in rec_constraints:
                    trace = EFSTraceability(
                        doc_id=doc_ir.document_id,
                        source_document=doc_name,
                        block_id=c.evidence.block_id,
                        section=c.evidence.section,
                        original_text=c.evidence.text,
                        extraction_method="source_backed_recovery",
                        confidence=1.0
                    )
                    if c.category in ("frequency", "latency", "clock_period"):
                        efs_ir.timing_rules.append(EFSTimingRule(
                            rule_id=new_efs_id("time"),
                            type=c.category,
                            cycle_relationship=c.value,
                            traceability=trace
                        ))
                    efs_ir.performance_requirements.append(EFSPerformanceRequirement(
                        perf_id=new_efs_id("perf"),
                        metric=c.category,
                        target_value=c.value,
                        condition=c.original_text,
                        traceability=trace
                    ))
                report.explicit_recovered += 1
                report.records.append(RecoveryRecord("constraints", f"{len(rec_constraints)} constraints", "explicit", len(rec_constraints), rec_constraints[0].evidence, "Recovered hardware constraints from source"))
            else:
                report.unresolved_gaps += 1
                report.records.append(RecoveryRecord("constraints", "Constraints", "unresolved", 0, None, "No constraint evidence found in source"))

        # ----------------------------------------------------------------------
        # 5. FSM Recovery
        # ----------------------------------------------------------------------
        report.total_properties_checked += 1
        if not efs_ir.fsms:
            sig_names = [s.name for s in (efs_ir.signals or [])]
            rec_fsm = DedicatedFSMExtractor.extract_fsm(doc_ir, sig_names, evidence_index)
            if rec_fsm:
                trace = EFSTraceability(
                    doc_id=doc_ir.document_id,
                    source_document=doc_name,
                    block_id=rec_fsm.evidence.block_id,
                    section=rec_fsm.evidence.section,
                    original_text=rec_fsm.evidence.text,
                    extraction_method="source_backed_recovery",
                    confidence=0.95
                )
                states_objs = [EFSState(state_id=new_efs_id("st"), name=st) for st in rec_fsm.states]
                trans_objs = []
                for t in rec_fsm.transitions:
                    src = getattr(t, "source_state", None) or (t.get("source_state") if isinstance(t, dict) else "")
                    dst = getattr(t, "target_state", None) or (t.get("target_state") if isinstance(t, dict) else "")
                    cond = getattr(t, "condition", None) or (t.get("condition", "") if isinstance(t, dict) else "")
                    assoc_sig = getattr(t, "associated_signal", None) or (t.get("associated_signal") if isinstance(t, dict) else None)
                    assoc_reg = getattr(t, "associated_register", None) or (t.get("associated_register") if isinstance(t, dict) else None)
                    cond_type = getattr(t, "condition_type", None) or (t.get("condition_type", "explicit") if isinstance(t, dict) else "explicit")
                    ev_text = getattr(t, "evidence", None) or (t.get("evidence", "") if isinstance(t, dict) else "")
                    trans_objs.append(
                        EFSTransition(
                            transition_id=new_efs_id("trans"),
                            source_state=src,
                            target_state=dst,
                            condition=cond,
                            associated_signal=assoc_sig,
                            associated_register=assoc_reg,
                            condition_type=cond_type,
                            evidence=ev_text,
                            traceability=trace
                        )
                    )

                efs_fsm = EFSFSM(
                    fsm_id=new_efs_id("fsm"),
                    name=rec_fsm.name,
                    states=states_objs,
                    transitions=trans_objs,
                    initial_state=rec_fsm.reset_state or (rec_fsm.states[0] if rec_fsm.states else "IDLE"),
                    terminal_states=getattr(rec_fsm, "terminal_states", []),
                    reset_state=getattr(rec_fsm, "reset_state", ""),
                    reset_behavior=getattr(rec_fsm, "reset_behavior", ""),
                    traceability=trace
                )
                efs_ir.fsms.append(efs_fsm)

                # Structure FSM transitions as relationships inside EFSFlow
                if trans_objs:
                    flow_steps = [
                        EFSFlowStep(
                            action_description=f"State transition: {tr.source_state} -> {tr.target_state}",
                            condition=tr.condition,
                            state_transitions=[f"{tr.source_state}->{tr.target_state}"],
                            signal_behavior=f"Triggered by {tr.associated_signal}" if tr.associated_signal else ""
                        )
                        for tr in trans_objs
                    ]
                    efs_ir.flows.append(EFSFlow(
                        flow_id=new_efs_id("flow"),
                        name=f"{rec_fsm.name}_Flow",
                        ordered_steps=flow_steps,
                        state_transitions=[f"{tr.source_state}->{tr.target_state}" for tr in trans_objs],
                        traceability=trace
                    ))
                if rec_fsm.status == "explicit":
                    report.explicit_recovered += 1
                elif rec_fsm.status == "partial":
                    report.partial_recovered += 1
                else:
                    report.inferred_recovered += 1
                report.records.append(RecoveryRecord("fsm", rec_fsm.name, rec_fsm.status, len(rec_fsm.states), rec_fsm.evidence, f"Recovered FSM ({rec_fsm.status}) from source control signals"))
            else:
                report.unresolved_gaps += 1
                report.records.append(RecoveryRecord("fsm", "FSM", "unresolved", 0, None, "No FSM or control signal evidence in source"))

        # ----------------------------------------------------------------------
        # 6. Memory Regions Recovery
        # ----------------------------------------------------------------------
        report.total_properties_checked += 1
        if not efs_ir.memory_regions:
            mem_evs = evidence_index.search(["sram", "cache", "l1", "l2", "memory", "buffer", "dma"], match_any=True)
            if mem_evs:
                for idx, ev in enumerate(mem_evs[:4], 1):
                    # Extract memory name if possible
                    m_name = f"Memory_Region_{idx}"
                    line_clean = ev.text.strip().replace(":", " : ")
                    name_m = re.search(r"([A-Za-z0-9_\s]{2,30}\s*(?:sram|cache|memory|buffer))\b", line_clean, re.I)
                    if name_m:
                        m_name = name_m.group(1).strip()

                    size_m = re.search(r"(\d+\s*(?:kb|mb|gb|bytes?|words?))", ev.text, re.I)
                    sz_val = size_m.group(1) if size_m else "unspecified"

                    trace = EFSTraceability(
                        doc_id=doc_ir.document_id,
                        source_document=doc_name,
                        block_id=ev.block_id,
                        section=ev.section,
                        original_text=ev.text,
                        extraction_method="source_backed_recovery"
                    )
                    efs_ir.memory_regions.append(EFSMemoryRegion(
                        region_id=new_efs_id("memreg"),
                        name=m_name,
                        type="memory_resource",
                        size=sz_val,
                        description=f"Recovered from: {ev.text}",
                        traceability=trace
                    ))
                report.explicit_recovered += 1
                report.records.append(RecoveryRecord("memory", f"{len(efs_ir.memory_regions)} memory regions", "explicit", len(efs_ir.memory_regions), mem_evs[0], "Recovered memory hierarchy from source specification"))
            else:
                report.unresolved_gaps += 1
                report.records.append(RecoveryRecord("memory", "Memory", "unresolved", 0, None, "No memory evidence found in source"))

        # Post-recovery semantic cleanup: prune misclassified components and interfaces
        efs_ir.components = [
            c for c in (efs_ir.components or [])
            if not is_fsm_state_name(c.name) and not is_physical_signal_name(c.name)
        ]
        valid_ifaces = []
        for i in (efs_ir.interfaces or []):
            if is_physical_signal_name(i.name) or is_fsm_state_name(i.name):
                continue
            s_c = i.source_component or ""
            d_c = getattr(i, "destination_component", None) or ""
            if s_c or d_c:
                valid_if, _ = is_valid_interface_candidate(s_c, d_c)
                if not valid_if:
                    continue
            valid_ifaces.append(i)
        efs_ir.interfaces = valid_ifaces

        logger.info(
            f"SourceBackedRecovery complete: explicit={report.explicit_recovered}, partial={report.partial_recovered}, "
            f"inferred={report.inferred_recovered}, unresolved_gaps={report.unresolved_gaps}."
        )
        return report
