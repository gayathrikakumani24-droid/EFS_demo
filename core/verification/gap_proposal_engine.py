"""
Evidence-Grounded AI Proposal Engine for Hardware Specification Gaps.

Generates structured, technically sound resolution proposals for detected specification
gaps. Strictly enforces evidence classification:
- SOURCE_SUPPORTED: Explicitly documented in spec.
- DERIVED: Logically derived with documented derivation.
- IMPLEMENTATION_CHOICE: Synthesizable engineering choice not mandated by spec.
- UNSUPPORTED_ASSUMPTION: Speculation lacking evidence.

Returns NO_SAFE_PROPOSAL with focused human clarification questions whenever
the specification provides insufficient evidence.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from core.efs_ir.models import EFSIR
from core.efs_ir.specification_gap import (
    EvidenceClassification,
    GapResolutionStatus,
    GapSeverity,
    SpecificationGap,
    SpecificationGapCategory,
    SpecificationGapProposal,
)
from utils.logger import get_logger

logger = get_logger("verification.gap_proposal_engine")


class GapProposalEngine:
    """
    Evidence-grounded engine synthesizing proposals for hardware specification gaps.
    Works deterministically first, querying available source evidence, EFS IR entities,
    and protocol contracts.
    """

    @classmethod
    def analyze_and_propose(
        cls,
        gap: SpecificationGap,
        efs_ir: Optional[EFSIR] = None,
        context_pack: Optional[Any] = None,
        design_spec: Optional[str] = None,
        approved_resolutions: Optional[List[Any]] = None,
        llm_client: Optional[Any] = None,
    ) -> List[SpecificationGapProposal]:
        """
        Analyze the gap against available evidence and return structured candidate proposals.
        """
        spec_text = str(design_spec or (getattr(efs_ir.metadata, "description", "") if efs_ir and hasattr(efs_ir, "metadata") else "")).strip()
        entity = gap.affected_entity.strip()
        cat = gap.category

        # Collect available context evidence
        relevant_evidence, doc_refs = cls._extract_relevant_evidence(entity, spec_text, efs_ir, context_pack)

        proposals: List[SpecificationGapProposal] = []

        if cat == SpecificationGapCategory.UNSPECIFIED_REGISTER_UPDATE:
            proposals = cls._propose_register_update(gap, entity, relevant_evidence, doc_refs, efs_ir, spec_text)
        elif cat == SpecificationGapCategory.MISSING_SIGNAL_DRIVER:
            proposals = cls._propose_signal_driver(gap, entity, relevant_evidence, doc_refs, efs_ir, spec_text)
        elif cat == SpecificationGapCategory.UNSPECIFIED_RESET_BEHAVIOR:
            proposals = cls._propose_reset_behavior(gap, entity, relevant_evidence, doc_refs, efs_ir, spec_text)
        elif cat == SpecificationGapCategory.UNSPECIFIED_CONTROL_FLOW:
            proposals = cls._propose_control_flow(gap, entity, relevant_evidence, doc_refs, efs_ir, spec_text)
        elif cat == SpecificationGapCategory.INCOMPLETE_PROTOCOL_BEHAVIOR:
            proposals = cls._propose_protocol_behavior(gap, entity, relevant_evidence, doc_refs, efs_ir, spec_text)
        elif cat == SpecificationGapCategory.CONFLICTING_REQUIREMENTS:
            proposals = cls._propose_conflict_resolution(gap, entity, relevant_evidence, doc_refs, efs_ir, spec_text)
        else:
            proposals = cls._propose_generic_gap(gap, entity, relevant_evidence, doc_refs, efs_ir, spec_text)

        # Apply Proposal Safety Gate (Phase 3.2.1)
        safe_proposals: List[SpecificationGapProposal] = []
        for p in proposals:
            is_safe, safety_reason = cls.evaluate_proposal_safety(
                proposal=p,
                gap=gap,
                evidence=relevant_evidence,
                efs_ir=efs_ir,
                approved_resolutions=approved_resolutions
            )
            if not is_safe and p.is_safe_proposal:
                safe_proposals.append(SpecificationGapProposal(
                    proposal_id=p.proposal_id,
                    gap_id=gap.gap_id,
                    title=f"Unsafe Proposal Suppressed by Safety Gate: {p.title}",
                    proposed_resolution="NO_SAFE_PROPOSAL",
                    detailed_behavioral_explanation=f"Safety gate blocked proposal: {safety_reason}",
                    is_safe_proposal=False,
                    clarification_questions=[
                        f"Please provide explicit specification for '{gap.affected_entity}': {safety_reason}"
                    ],
                    evidence_classification=EvidenceClassification.UNSUPPORTED_ASSUMPTION,
                    confidence_level=0.0
                ))
            else:
                safe_proposals.append(p)
        proposals = safe_proposals

        # Ensure all proposals are properly linked to gap_id
        for p in proposals:
            p.gap_id = gap.gap_id
            if not p.supporting_evidence and relevant_evidence:
                p.supporting_evidence = relevant_evidence[:300]
            if not p.evidence_references and doc_refs:
                p.evidence_references = doc_refs

        gap.proposals = proposals
        if proposals:
            gap.resolution_status = GapResolutionStatus.PROPOSED
        return proposals

    @classmethod
    def evaluate_proposal_safety(
        cls,
        proposal: SpecificationGapProposal,
        gap: SpecificationGap,
        evidence: str,
        efs_ir: Optional[EFSIR] = None,
        approved_resolutions: Optional[List[Any]] = None
    ) -> Tuple[bool, str]:
        """
        Phase 3.2.1 Proposal Safety Gate:
        Verifies that candidate proposals do not fabricate hardware behavior,
        do not invent arithmetic without source support, and do not conflict
        with previously approved decisions.
        """
        if not proposal.is_safe_proposal or proposal.proposed_resolution == "NO_SAFE_PROPOSAL":
            return False, "Proposal explicitly indicates missing specification information."

        if not gap.affected_entity:
            return False, "Target hardware entity is undefined."

        res_str = proposal.proposed_resolution.lower()

        # Safety Check: Prohibit fabricated arithmetic/bitwise operations without source evidence
        if any(op in res_str for op in ("+ 32'd", "- 32'd", "* 32'd", "+ 1", "- 1", " ^ 32'h", " & 32'h")):
            if not any(k in evidence.lower() for k in ("add", "increment", "plus", "+", "sum", "sub", "counter", "count", "invert", "xor", "mask")):
                return False, "Arithmetic or bitwise transformation fabricated without source requirement support."

        # Safety Check: Consistency with existing approved resolutions
        if approved_resolutions:
            for res in approved_resolutions:
                r_ent = str(getattr(res, "target_entity", "") or "").lower()
                r_val = str(getattr(res, "resolved_value", "") or getattr(res, "proposed_change", "") or "")
                if r_ent and r_ent == gap.affected_entity.lower():
                    if r_val and r_val != proposal.proposed_resolution and getattr(res, "approval_status", "") == "APPROVED":
                        return False, f"Conflicts with previously approved decision for '{r_ent}': {r_val}"

        return True, "Proposal passed all safety checks."

    @classmethod
    def _extract_relevant_evidence(
        cls,
        entity: str,
        spec_text: str,
        efs_ir: Optional[EFSIR],
        context_pack: Optional[Any]
    ) -> Tuple[str, List[str]]:
        """Extract matching sentences or lines referencing the entity from specification and EFS IR."""
        evidence_snippets: List[str] = []
        doc_refs: List[str] = []

        if not entity:
            return "", []

        # 1. Search in spec_text
        if spec_text:
            lines = spec_text.splitlines()
            ent_pattern = re.compile(rf"\b{re.escape(entity)}\b", re.IGNORECASE)
            for idx, line in enumerate(lines):
                if ent_pattern.search(line):
                    snippet = line.strip()
                    if snippet and snippet not in evidence_snippets:
                        evidence_snippets.append(snippet)
                        doc_refs.append(f"spec_line_{idx+1}")

        # 2. Search in EFS IR functional behaviors
        if efs_ir and hasattr(efs_ir, "functional_behavior") and efs_ir.functional_behavior:
            for fb in efs_ir.functional_behavior:
                desc = getattr(fb, "description", "") or ""
                action = getattr(fb, "action", "") or ""
                if entity.lower() in desc.lower() or entity.lower() in action.lower():
                    snippet = f"Behavior '{fb.name}': {desc} (Action: {action})"
                    if snippet not in evidence_snippets:
                        evidence_snippets.append(snippet)
                        doc_refs.append(getattr(fb, "behavior_id", "efs_behavior"))

        # 3. Search in Protocol Rules
        if efs_ir and hasattr(efs_ir, "protocol_rules") and efs_ir.protocol_rules:
            for pr in efs_ir.protocol_rules:
                r_sigs = getattr(pr, "signals", []) or []
                r_desc = getattr(pr, "description", "") or ""
                if entity in r_sigs or any(entity.lower() == s.lower() for s in r_sigs) or entity.lower() in r_desc.lower():
                    snippet = f"Protocol Rule '{pr.rule_id}': {r_desc}"
                    if snippet not in evidence_snippets:
                        evidence_snippets.append(snippet)
                        doc_refs.append(getattr(pr, "rule_id", "proto_rule"))

        combined = " | ".join(evidence_snippets) if evidence_snippets else ""
        return combined, doc_refs

    @classmethod
    def _propose_register_update(
        cls,
        gap: SpecificationGap,
        entity: str,
        evidence: str,
        doc_refs: List[str],
        efs_ir: Optional[EFSIR],
        spec_text: str
    ) -> List[SpecificationGapProposal]:
        """Propose update logic for unspecified register."""
        proposals: List[SpecificationGapProposal] = []
        ent_lower = entity.lower()

        # Specific hardware register roles
        if "status" in ent_lower:
            sig_names = [s.name for s in getattr(efs_ir, "signals", [])] if efs_ir else []
            done_sig = next((s for s in sig_names if any(k in s.lower() for k in ("done", "finish", "complete"))), "1'b0")
            err_sig = next((s for s in sig_names if any(k in s.lower() for k in ("err", "error", "fault"))), "1'b0")
            prop_status = SpecificationGapProposal(
                title=f"Status Register Capture for '{entity}'",
                proposed_resolution=f"always @(posedge clk) {entity} <= {{{err_sig}, {done_sig}}};",
                detailed_behavioral_explanation=f"Update status register '{entity}' synchronously with execution flags (error, done).",
                proposed_rtl_semantics={
                    "type": "SEQUENTIAL_UPDATE",
                    "target": entity,
                    "next_value": f"{{{err_sig}, {done_sig}}}",
                    "enable": "1'b1",
                    "hold_behavior": "PRESERVE"
                },
                relevant_registers=[entity],
                relevant_signals=[s for s in [done_sig, err_sig] if s != "1'b0"],
                conditions_and_timing="Synchronous continuous status update",
                expected_effect=f"Captures current module operational status in '{entity}'.",
                supporting_evidence=evidence or "Standard hardware status register semantics.",
                evidence_references=doc_refs,
                reasoning="Status registers reflect operational state flags.",
                explicit_assumptions=["Status flags settle synchronously"],
                confidence_level=0.80,
                evidence_classification=EvidenceClassification.IMPLEMENTATION_CHOICE,
                is_safe_proposal=True
            )
            return [prop_status]

        # Collect declared interface input ports
        in_ports: List[str] = []
        if efs_ir:
            in_ports = [s.name for s in getattr(efs_ir, "signals", []) if getattr(s, "direction", "") == "input"]
            if not in_ports:
                in_ports = [p.name for p in getattr(efs_ir, "interface_ports", []) if getattr(p, "direction", "") == "input"]

        valid_in_port = next((p for p in in_ports if "valid" in p.lower()), None)
        en_in_port = next((p for p in in_ports if any(k in p.lower() for k in ("en", "enable", "start", "req"))), None)

        if "cfg" in ent_lower or "config" in ent_lower:
            cfg_data_in = next((p for p in in_ports if any(k in p.lower() for k in ("instr", "data", "rdata", "wdata", "din", "cfg", "config"))), None) or (in_ports[0] if in_ports else "32'h0")
            cfg_en_in = valid_in_port or en_in_port or "1'b1"
            prop_cfg = SpecificationGapProposal(
                title=f"Configuration Register Hold/Update for '{entity}'",
                proposed_resolution=f"if ({cfg_en_in}) {entity} <= {cfg_data_in};",
                detailed_behavioral_explanation=f"Retain configuration value in '{entity}', updating synchronously when '{cfg_en_in}' is active.",
                proposed_rtl_semantics={
                    "type": "SEQUENTIAL_UPDATE",
                    "target": entity,
                    "enable": cfg_en_in,
                    "next_value": cfg_data_in,
                    "hold_behavior": "PRESERVE"
                },
                relevant_registers=[entity],
                relevant_signals=[cfg_en_in, cfg_data_in] if cfg_en_in != "1'b1" else [cfg_data_in],
                conditions_and_timing="Synchronous register write enable",
                expected_effect=f"Maintains configuration parameters in '{entity}'.",
                supporting_evidence=evidence or "Standard configuration register hold semantics.",
                evidence_references=doc_refs,
                reasoning="Configuration registers hold written settings across operations.",
                explicit_assumptions=["Configuration is programmed during setup"],
                confidence_level=0.75,
                evidence_classification=EvidenceClassification.IMPLEMENTATION_CHOICE,
                is_safe_proposal=True
            )
            return [prop_cfg]

        matched_input = next((p for p in in_ports if any(k in p.lower() for k in ("data", "in", "din", "payload", "val", "wdata", "rdata", "rx"))), None)
        if not matched_input and in_ports:
            matched_input = next((p for p in in_ports if not any(k in p.lower() for k in ("clk", "clock", "rst", "reset"))), None)
        if not matched_input:
            matched_input = "data_in"

        # Alternative 1: Synchronous load gated by explicit operation / enable condition
        cond_1 = valid_in_port or en_in_port or "1'b1"
        data_src_1 = matched_input
        prop_1 = SpecificationGapProposal(
            title=f"Clock-Enabled Synchronous Update for '{entity}'",
            proposed_resolution=f"if ({cond_1}) {entity} <= {data_src_1};",
            detailed_behavioral_explanation=(
                f"Update register '{entity}' synchronously on clock edge with '{data_src_1}' when '{cond_1}' is asserted. "
                f"When '{cond_1}' is deasserted, implicitly retain current state."
            ),
            proposed_rtl_semantics={
                "type": "SEQUENTIAL_UPDATE",
                "target": entity,
                "enable": cond_1,
                "next_value": data_src_1,
                "hold_behavior": "PRESERVE"
            },
            relevant_registers=[entity],
            relevant_signals=[cond_1, data_src_1] if cond_1 != "1'b1" else [data_src_1],
            conditions_and_timing="Synchronous to clk edge, 1-cycle latency",
            expected_effect=f"Register '{entity}' captures '{data_src_1}' on active '{cond_1}'.",
            supporting_evidence=evidence or "Register declared in interface with input data ports.",
            evidence_references=doc_refs,
            reasoning=f"Standard clocked register loading semantics matching available datapath inputs.",
            explicit_assumptions=[f"Input '{cond_1}' acts as write enable", f"Input '{data_src_1}' is valid during write"],
            potential_risks=["Potential unintended overwrite if enable signal timing is misaligned"],
            confidence_level=0.75,
            evidence_classification=EvidenceClassification.IMPLEMENTATION_CHOICE,
            is_safe_proposal=True
        )
        proposals.append(prop_1)

        # Alternative 2: State-based / lifecycle update (e.g. update during EXECUTE / COMPUTE state)
        fsm_states = [s.name for f in getattr(efs_ir, "fsms", []) for s in getattr(f, "states", [])] if efs_ir else []
        exec_state = next((s for s in fsm_states if any(k in s.lower() for k in ("exec", "run", "calc", "compute", "load"))), None) or (fsm_states[-1] if fsm_states else "EXEC")
        prop_2 = SpecificationGapProposal(
            title=f"State-Gated Transfer for '{entity}'",
            proposed_resolution=f"if (current_state == {exec_state}) {entity} <= {data_src_1};",
            detailed_behavioral_explanation=(
                f"Update register '{entity}' only while controller is in state '{exec_state}'. "
                f"Preserves value across other operational states."
            ),
            proposed_rtl_semantics={
                "type": "SEQUENTIAL_UPDATE",
                "target": entity,
                "condition": f"current_state == {exec_state}",
                "next_value": data_src_1,
                "hold_behavior": "PRESERVE"
            },
            relevant_registers=[entity],
            relevant_fsm_states=[exec_state],
            conditions_and_timing=f"Active during {exec_state} state",
            expected_effect=f"Limits updates of '{entity}' strictly to {exec_state} phase.",
            supporting_evidence=evidence or f"FSM includes {exec_state} state.",
            evidence_references=doc_refs,
            reasoning="Aligns register write window with controller execution lifecycle.",
            explicit_assumptions=[f"Computation results for '{entity}' settle during '{exec_state}'"],
            potential_risks=[f"Data will not be captured if FSM does not enter '{exec_state}'"],
            confidence_level=0.65,
            evidence_classification=EvidenceClassification.IMPLEMENTATION_CHOICE,
            is_safe_proposal=True
        )
        proposals.append(prop_2)

        return proposals

    @classmethod
    def _propose_signal_driver(
        cls,
        gap: SpecificationGap,
        entity: str,
        evidence: str,
        doc_refs: List[str],
        efs_ir: Optional[EFSIR],
        spec_text: str
    ) -> List[SpecificationGapProposal]:
        """Propose driving logic for unassigned output port."""
        proposals: List[SpecificationGapProposal] = []
        ent_lower = entity.lower()

        # Check for done / completion signals
        if any(k in ent_lower for k in ("done", "complete", "finish", "ready")):
            # Alternative 1: FSM State Indicator (Level)
            fsm_states = [s.name for f in getattr(efs_ir, "fsms", []) for s in getattr(f, "states", [])] if efs_ir else []
            done_state = next((s for s in fsm_states if any(k in s.lower() for k in ("done", "finish", "complete"))), None) or (fsm_states[-1] if fsm_states else "DONE")
            prop_level = SpecificationGapProposal(
                title=f"Level Completion Flag: '{entity}' Asserted in {done_state}",
                proposed_resolution=f"assign {entity} = (current_state == {done_state});",
                detailed_behavioral_explanation=f"Drive output '{entity}' high whenever controller state is '{done_state}'.",
                proposed_rtl_semantics={
                    "type": "CONTINUOUS_ASSIGNMENT",
                    "target": entity,
                    "expression": f"current_state == {done_state}"
                },
                relevant_signals=[entity],
                relevant_fsm_states=[done_state],
                conditions_and_timing="Combinational decode of current_state register",
                expected_effect=f"'{entity}' remains asserted as long as FSM dwells in {done_state}.",
                supporting_evidence=evidence or f"FSM contains state '{done_state}'.",
                evidence_references=doc_refs,
                reasoning="Standard controller completion output decode.",
                explicit_assumptions=[f"Downstream consumer expects level indicator during {done_state}"],
                potential_risks=["Consumer expecting single-cycle pulse may trigger multiple times"],
                confidence_level=0.85,
                evidence_classification=EvidenceClassification.IMPLEMENTATION_CHOICE,
                is_safe_proposal=True
            )
            proposals.append(prop_level)

            # Alternative 2: Single-Cycle Strobe / Pulse
            prop_pulse = SpecificationGapProposal(
                title=f"Single-Cycle Pulse on Completion: '{entity}'",
                proposed_resolution=f"assign {entity} = (current_state == {done_state}) && (next_state != {done_state});",
                detailed_behavioral_explanation=f"Pulse '{entity}' high for exactly one clock cycle as controller exits or enters '{done_state}'.",
                proposed_rtl_semantics={
                    "type": "CONTINUOUS_ASSIGNMENT",
                    "target": entity,
                    "expression": f"(current_state == {done_state}) && (next_state != {done_state})"
                },
                relevant_signals=[entity],
                relevant_fsm_states=[done_state],
                conditions_and_timing="Single cycle strobe",
                expected_effect=f"Asserts '{entity}' for one clock period.",
                supporting_evidence=evidence or "Standard synchronous strobe semantics.",
                evidence_references=doc_refs,
                reasoning="Prevents repeated triggers in edge-sensitive consumers.",
                explicit_assumptions=["Consumer requires single-cycle edge/pulse notification"],
                potential_risks=["Consumer with slow polling clock may miss single-cycle pulse"],
                confidence_level=0.75,
                evidence_classification=EvidenceClassification.IMPLEMENTATION_CHOICE,
                is_safe_proposal=True
            )
            proposals.append(prop_pulse)
            return proposals

        # Check for error / fault signals
        if any(k in ent_lower for k in ("err", "error", "fault", "overflow")):
            prop_err = SpecificationGapProposal(
                title=f"Error Detection Assertion for '{entity}'",
                proposed_resolution=f"assign {entity} = 1'b0; // Specification omitted error conditions",
                detailed_behavioral_explanation=(
                    f"Specification declares '{entity}', but does not define any invalid opcode, overflow, "
                    f"parity, or timeout conditions. A safe baseline tie-off requires explicit human authorization."
                ),
                proposed_rtl_semantics={
                    "type": "CONTINUOUS_ASSIGNMENT",
                    "target": entity,
                    "expression": "1'b0"
                },
                relevant_signals=[entity],
                conditions_and_timing="Constant default",
                expected_effect=f"Ties '{entity}' inactive until error rules are specified.",
                supporting_evidence=evidence or "Output declared without error condition rules.",
                evidence_references=doc_refs,
                reasoning="Prevents spurious error alerts when no fault checks exist.",
                explicit_assumptions=["Design has no active fault detection logic in current phase"],
                potential_risks=["Silent failure if an actual runtime error condition occurs"],
                confidence_level=0.60,
                evidence_classification=EvidenceClassification.IMPLEMENTATION_CHOICE,
                is_safe_proposal=True,
                clarification_questions=[
                    f"Which fault conditions should assert output '{entity}' (e.g. arithmetic overflow, illegal opcode, parity error)?"
                ]
            )
            return [prop_err]

        # Check for protocol ready signals (e.g., axi_rready, axi_bready, ready)
        if any(k in ent_lower for k in ("rready", "bready")) or ent_lower.endswith("_ready") or ent_lower == "ready":
            prop_ready = SpecificationGapProposal(
                title=f"Ready Handshake Assertion for '{entity}'",
                proposed_resolution=f"assign {entity} = 1'b1; // Default ready to accept transactions",
                detailed_behavioral_explanation=f"Assert ready signal '{entity}' high to accept transactions without stall or backpressure.",
                proposed_rtl_semantics={"type": "CONTINUOUS_ASSIGNMENT", "target": entity, "expression": "1'b1"},
                relevant_signals=[entity],
                conditions_and_timing="Combinational handshake assertion",
                expected_effect=f"Maintains '{entity}' active to receive incoming responses or data.",
                supporting_evidence=evidence or "Standard bus ready handshake protocol convention.",
                evidence_references=doc_refs,
                reasoning="Always-ready assertion prevents upstream transaction deadlocks.",
                explicit_assumptions=["Receiver is always prepared to accept bus transfers"],
                confidence_level=0.85,
                evidence_classification=EvidenceClassification.IMPLEMENTATION_CHOICE,
                is_safe_proposal=True
            )
            return [prop_ready]

        # Check for protocol valid signals (e.g., axi_arvalid, axi_awvalid, axi_wvalid, valid)
        if any(k in ent_lower for k in ("arvalid", "awvalid", "wvalid")) or ent_lower.endswith("_valid") or ent_lower == "valid":
            fsm_states = [s.name for f in getattr(efs_ir, "fsms", []) for s in getattr(f, "states", [])] if efs_ir else []
            active_st = next((s for s in fsm_states if any(k in s.lower() for k in ("req", "read", "write", "addr", "send", "tx", "run", "exec"))), None) or (fsm_states[-1] if fsm_states else "EXEC")
            prop_valid = SpecificationGapProposal(
                title=f"Transaction Valid Initiation for '{entity}'",
                proposed_resolution=f"assign {entity} = (current_state == {active_st});",
                detailed_behavioral_explanation=f"Assert '{entity}' high while controller is in state '{active_st}' to initiate bus transfer.",
                proposed_rtl_semantics={"type": "CONTINUOUS_ASSIGNMENT", "target": entity, "expression": f"current_state == {active_st}"},
                relevant_signals=[entity],
                relevant_fsm_states=[active_st],
                conditions_and_timing=f"Active during state {active_st}",
                expected_effect=f"Drives '{entity}' high to request bus transfer during {active_st}.",
                supporting_evidence=evidence or f"Protocol valid handshake signal paired with {active_st} state.",
                evidence_references=doc_refs,
                reasoning="Aligns bus transaction request with controller active phase.",
                explicit_assumptions=[f"Bus transaction takes place during {active_st}"],
                confidence_level=0.80,
                evidence_classification=EvidenceClassification.IMPLEMENTATION_CHOICE,
                is_safe_proposal=True
            )
            return [prop_valid]

        # Check for byte strobe signals (e.g., axi_wstrb, wstrb, strb)
        if any(k in ent_lower for k in ("wstrb", "strb")):
            prop_strb = SpecificationGapProposal(
                title=f"All-Byte Strobe Enable for '{entity}'",
                proposed_resolution=f"assign {entity} = '1; // Enable all byte lanes",
                detailed_behavioral_explanation=f"Assert all byte lanes of strobe output '{entity}'.",
                proposed_rtl_semantics={"type": "CONTINUOUS_ASSIGNMENT", "target": entity, "expression": "'1"},
                relevant_signals=[entity],
                conditions_and_timing="Constant full-word write strobe",
                expected_effect=f"Enables all byte lanes without partial masking.",
                supporting_evidence=evidence or "Standard bus full-word write strobe convention.",
                evidence_references=doc_refs,
                reasoning="Default write strobe enables full payload bus width.",
                explicit_assumptions=["Transactions perform full-word updates"],
                confidence_level=0.85,
                evidence_classification=EvidenceClassification.IMPLEMENTATION_CHOICE,
                is_safe_proposal=True
            )
            return [prop_strb]

        # Check for burst last signals (e.g., axi_wlast, wlast, last)
        if any(k in ent_lower for k in ("wlast", "last")):
            prop_last = SpecificationGapProposal(
                title=f"Single-Beat Terminal Flag for '{entity}'",
                proposed_resolution=f"assign {entity} = 1'b1; // Single-beat burst",
                detailed_behavioral_explanation=f"Assert '{entity}' high indicating single-beat transfer bursts.",
                proposed_rtl_semantics={"type": "CONTINUOUS_ASSIGNMENT", "target": entity, "expression": "1'b1"},
                relevant_signals=[entity],
                conditions_and_timing="Constant single-beat flag",
                expected_effect=f"Signals terminal beat on every write transfer.",
                supporting_evidence=evidence or "Standard single transfer burst protocol convention.",
                evidence_references=doc_refs,
                reasoning="Single-beat transactions assert last on every transfer beat.",
                explicit_assumptions=["Burst length is 1 beat"],
                confidence_level=0.80,
                evidence_classification=EvidenceClassification.IMPLEMENTATION_CHOICE,
                is_safe_proposal=True
            )
            return [prop_last]

        # Check for address bus outputs (e.g., axi_araddr, axi_awaddr, addr)
        if any(k in ent_lower for k in ("araddr", "awaddr")) or ent_lower.endswith("_addr") or ent_lower == "addr":
            reg_names = [r.name for r in getattr(efs_ir, "registers", [])] if efs_ir else []
            addr_reg = next((r for r in reg_names if "addr" in r.lower()), None)
            drv_expr = addr_reg if addr_reg else "'0"
            prop_addr = SpecificationGapProposal(
                title=f"Address Bus Driver for '{entity}'",
                proposed_resolution=f"assign {entity} = {drv_expr};",
                detailed_behavioral_explanation=f"Drive output address '{entity}' from {f'internal register {addr_reg}' if addr_reg else 'default zero base'}.",
                proposed_rtl_semantics={"type": "CONTINUOUS_ASSIGNMENT", "target": entity, "expression": drv_expr},
                relevant_signals=[entity],
                relevant_registers=[addr_reg] if addr_reg else [],
                conditions_and_timing="Combinational address output",
                expected_effect=f"Provides address for memory/bus transactions.",
                supporting_evidence=evidence or "Standard address bus connection.",
                evidence_references=doc_refs,
                reasoning="Routes internal addressing source to external bus address port.",
                explicit_assumptions=["Address is stable across bus transfer"],
                confidence_level=0.75,
                evidence_classification=EvidenceClassification.IMPLEMENTATION_CHOICE,
                is_safe_proposal=True
            )
            return [prop_addr]

        # Check for data bus outputs (e.g., axi_wdata, wdata, data_out, dout)
        if any(k in ent_lower for k in ("wdata", "data_out", "dout")):
            reg_names = [r.name for r in getattr(efs_ir, "registers", [])] if efs_ir else []
            data_reg = next((r for r in reg_names if any(k in r.lower() for k in ("data", "payload", "tx", "buf", "out"))), None)
            drv_expr = data_reg if data_reg else "'0"
            prop_data = SpecificationGapProposal(
                title=f"Data Bus Output Driver for '{entity}'",
                proposed_resolution=f"assign {entity} = {drv_expr};",
                detailed_behavioral_explanation=f"Drive output data payload '{entity}' from {f'internal register {data_reg}' if data_reg else 'default zero'}.",
                proposed_rtl_semantics={"type": "CONTINUOUS_ASSIGNMENT", "target": entity, "expression": drv_expr},
                relevant_signals=[entity],
                relevant_registers=[data_reg] if data_reg else [],
                conditions_and_timing="Combinational datapath output",
                expected_effect=f"Drives write data payload onto bus.",
                supporting_evidence=evidence or "Standard write data output connection.",
                evidence_references=doc_refs,
                reasoning="Routes internal register or datapath buffer to output data port.",
                explicit_assumptions=["Write data is valid during transaction"],
                confidence_level=0.75,
                evidence_classification=EvidenceClassification.IMPLEMENTATION_CHOICE,
                is_safe_proposal=True
            )
            return [prop_data]

        # Generic output port without evidence
        has_driver_intent = any(
            k in evidence.lower() for k in (
                "driven by", "drives", "equal", "assign", "<=", "=", "connected to",
                "routes", "transmits", "carries", "computed", "calculated", "sum",
                "product", "inverted", "buffer", "registers", "alu_out", "acc_reg"
            )
        )
        if not evidence or not has_driver_intent:
            prop = SpecificationGapProposal(
                title=f"Insufficient Evidence to Drive Output '{entity}'",
                proposed_resolution="NO_SAFE_PROPOSAL",
                detailed_behavioral_explanation=(
                    f"Output port '{entity}' is declared in the interface, but the specification does not "
                    f"define its driving equation or functional relationship to input signals or registers. "
                    f"The existence of input signals does not establish the intended output function."
                ),
                is_safe_proposal=False,
                clarification_questions=[
                    f"What equation, internal register, or input signal drives output '{entity}'?",
                    f"Under what functional conditions should output '{entity}' be asserted or valid?"
                ],
                evidence_classification=EvidenceClassification.UNSUPPORTED_ASSUMPTION,
                confidence_level=0.0
            )
            return [prop]

        # Output with partial evidence
        prop_derived = SpecificationGapProposal(
            title=f"Derived Combinational Driver for '{entity}'",
            proposed_resolution=f"assign {entity} = ...; // Grounded in: {evidence[:80]}",
            detailed_behavioral_explanation=f"Grounded derivation based on specification evidence: {evidence}",
            relevant_signals=[entity],
            supporting_evidence=evidence,
            evidence_references=doc_refs,
            reasoning="Derived from contextual spec mentions.",
            confidence_level=0.5,
            evidence_classification=EvidenceClassification.DERIVED,
            is_safe_proposal=True
        )
        return [prop_derived]

    @classmethod
    def _propose_reset_behavior(
        cls,
        gap: SpecificationGap,
        entity: str,
        evidence: str,
        doc_refs: List[str],
        efs_ir: Optional[EFSIR],
        spec_text: str
    ) -> List[SpecificationGapProposal]:
        """Propose reset semantics when polarity or reset values are unspecified."""
        proposals: List[SpecificationGapProposal] = []

        is_source_supported = bool(evidence and any(k in evidence.lower() for k in ("active-low", "active_low", "asynchronous", "async", "active-high", "active_high", "synchronous", "sync")))
        prop_async_low = SpecificationGapProposal(
            title="Asynchronous Active-Low Reset (rst_n)",
            proposed_resolution="always @(posedge clk or negedge rst_n) if (!rst_n) ...",
            detailed_behavioral_explanation="Standard industry convention: active-low asynchronous reset.",
            proposed_rtl_semantics={"reset_type": "ASYNC", "polarity": "ACTIVE_LOW", "signal": "rst_n"},
            supporting_evidence=evidence or "Standard ASIC/FPGA digital design methodology.",
            evidence_references=doc_refs,
            reasoning="Active-low reset minimizes power during active operation and aligns with standard cell libraries.",
            explicit_assumptions=["Reset deassertion is synchronized externally"],
            potential_risks=["Requires reset recovery/removal timing checks"],
            confidence_level=0.90,
            evidence_classification=EvidenceClassification.SOURCE_SUPPORTED if is_source_supported else EvidenceClassification.IMPLEMENTATION_CHOICE,
            is_safe_proposal=True
        )
        proposals.append(prop_async_low)

        prop_sync_high = SpecificationGapProposal(
            title="Synchronous Active-High Reset (rst)",
            proposed_resolution="always @(posedge clk) if (rst) ...",
            detailed_behavioral_explanation="Synchronous active-high reset, ideal for pure FPGA fabric targets.",
            proposed_rtl_semantics={"reset_type": "SYNC", "polarity": "ACTIVE_HIGH", "signal": "rst"},
            supporting_evidence=evidence or "Standard FPGA architectural design recommendation.",
            evidence_references=doc_refs,
            reasoning="Pure synchronous reset eliminates glitch vulnerability and simplifies static timing analysis.",
            explicit_assumptions=["Clock is active and stable during reset assertion"],
            potential_risks=["Registers cannot be reset if clock is stopped"],
            confidence_level=0.85,
            evidence_classification=EvidenceClassification.IMPLEMENTATION_CHOICE,
            is_safe_proposal=True
        )
        proposals.append(prop_sync_high)

        return proposals

    @classmethod
    def _propose_control_flow(
        cls,
        gap: SpecificationGap,
        entity: str,
        evidence: str,
        doc_refs: List[str],
        efs_ir: Optional[EFSIR],
        spec_text: str
    ) -> List[SpecificationGapProposal]:
        """Propose FSM transition guard when condition is omitted or vague."""
        # Check if handshake signals exist in interface
        sig_names = [s.name for s in getattr(efs_ir, "signals", [])] if efs_ir else []
        hs_sig = next((s for s in sig_names if any(k in s.lower() for k in ("start", "valid", "req", "enable", "go"))), None)

        if hs_sig:
            prop = SpecificationGapProposal(
                title=f"Guard Transition with Input Handshake '{hs_sig}'",
                proposed_resolution=f"{entity}: transition condition = ({hs_sig} == 1'b1)",
                detailed_behavioral_explanation=f"Trigger state transition when input handshake '{hs_sig}' is asserted.",
                proposed_rtl_semantics={"type": "FSM_TRANSITION", "guard": f"{hs_sig} == 1'b1"},
                relevant_signals=[hs_sig],
                conditions_and_timing="Synchronous transition condition evaluated on posedge clk",
                expected_effect=f"Advances controller from current state upon active '{hs_sig}'.",
                supporting_evidence=evidence or f"Input '{hs_sig}' present in interface.",
                evidence_references=doc_refs,
                reasoning=f"Uses available control signal '{hs_sig}' to initiate operation.",
                explicit_assumptions=[f"Signal '{hs_sig}' is synchronously driven"],
                confidence_level=0.80,
                evidence_classification=EvidenceClassification.DERIVED,
                is_safe_proposal=True
            )
            return [prop]

        # No safe condition available
        prop_no = SpecificationGapProposal(
            title=f"Missing FSM Transition Condition for '{entity}'",
            proposed_resolution="NO_SAFE_PROPOSAL",
            detailed_behavioral_explanation=(
                f"State transition '{entity}' has no boolean guard or trigger condition in the specification. "
                f"No input control or handshake signals were found to establish transition criteria."
            ),
            is_safe_proposal=False,
            clarification_questions=[
                f"What signal condition triggers transition '{entity}'?",
                f"Should the transition occur unconditionally after a single cycle (dwell = 1'b1)?"
            ],
            evidence_classification=EvidenceClassification.UNSUPPORTED_ASSUMPTION,
            confidence_level=0.1
        )
        return [prop_no]

    @classmethod
    def _propose_protocol_behavior(
        cls,
        gap: SpecificationGap,
        entity: str,
        evidence: str,
        doc_refs: List[str],
        efs_ir: Optional[EFSIR],
        spec_text: str
    ) -> List[SpecificationGapProposal]:
        """Propose application-level trigger vs protocol handshake separation."""
        prop = SpecificationGapProposal(
            title=f"Application Initiation Trigger for Protocol Rule '{entity}'",
            proposed_resolution="Initiate transaction on explicit internal request flag, hold valid until ready",
            detailed_behavioral_explanation=(
                f"Protocol rule '{entity}' defines bus handshaking (valid/ready), but the specification does "
                f"not define the application-level condition that triggers the controller to initiate the read/write. "
                f"Separate protocol compliance from internal application initiation."
            ),
            proposed_rtl_semantics={
                "type": "PROTOCOL_TRIGGER",
                "trigger_condition": "start_tx == 1'b1",
                "handshake": "valid && ready"
            },
            supporting_evidence=evidence,
            evidence_references=doc_refs,
            reasoning="Adheres to protocol stability constraints while exposing missing application trigger.",
            explicit_assumptions=["Internal FSM or status register will trigger transaction"],
            confidence_level=0.70,
            evidence_classification=EvidenceClassification.IMPLEMENTATION_CHOICE,
            is_safe_proposal=True,
            clarification_questions=[
                "What internal event or instruction triggers this bus transaction?"
            ]
        )
        return [prop]

    @classmethod
    def _propose_conflict_resolution(
        cls,
        gap: SpecificationGap,
        entity: str,
        evidence: str,
        doc_refs: List[str],
        efs_ir: Optional[EFSIR],
        spec_text: str
    ) -> List[SpecificationGapProposal]:
        """Propose options to resolve conflicting specification requirements."""
        proposals: List[SpecificationGapProposal] = []

        prop_1 = SpecificationGapProposal(
            title=f"Resolve Conflict by Prioritizing Interface Definition for '{entity}'",
            proposed_resolution="Adopt interface declaration parameters",
            detailed_behavioral_explanation="Prioritizes top-level structural interface declaration over prose text.",
            supporting_evidence=evidence,
            evidence_references=doc_refs,
            reasoning="Interface declarations define physical boundary contracts.",
            confidence_level=0.70,
            evidence_classification=EvidenceClassification.IMPLEMENTATION_CHOICE,
            is_safe_proposal=True
        )
        proposals.append(prop_1)

        prop_2 = SpecificationGapProposal(
            title=f"Resolve Conflict by Prioritizing Functional Prose for '{entity}'",
            proposed_resolution="Adopt functional behavior specification values",
            detailed_behavioral_explanation="Prioritizes detailed functional behavioral text over preliminary header.",
            supporting_evidence=evidence,
            evidence_references=doc_refs,
            reasoning="Functional prose often contains detailed architectural updates.",
            confidence_level=0.70,
            evidence_classification=EvidenceClassification.IMPLEMENTATION_CHOICE,
            is_safe_proposal=True
        )
        proposals.append(prop_2)

        return proposals

    @classmethod
    def _propose_generic_gap(
        cls,
        gap: SpecificationGap,
        entity: str,
        evidence: str,
        doc_refs: List[str],
        efs_ir: Optional[EFSIR],
        spec_text: str
    ) -> List[SpecificationGapProposal]:
        """Fallback for generic unclassified gaps."""
        if not evidence:
            return [
                SpecificationGapProposal(
                    title=f"Unresolved Requirement for '{entity}'",
                    proposed_resolution="NO_SAFE_PROPOSAL",
                    detailed_behavioral_explanation=f"Specification lacks sufficient evidence to resolve gap '{gap.gap_id}'.",
                    is_safe_proposal=False,
                    clarification_questions=[f"Please clarify requirement for '{entity}': {gap.description}"],
                    evidence_classification=EvidenceClassification.UNSUPPORTED_ASSUMPTION,
                    confidence_level=0.1
                )
            ]
        return [
            SpecificationGapProposal(
                title=f"Grounded Clarification for '{entity}'",
                proposed_resolution=f"Align with documented text: {evidence[:100]}",
                detailed_behavioral_explanation=evidence,
                supporting_evidence=evidence,
                evidence_references=doc_refs,
                confidence_level=0.6,
                evidence_classification=EvidenceClassification.DERIVED,
                is_safe_proposal=True
            )
        ]
