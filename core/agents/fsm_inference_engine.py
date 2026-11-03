"""
FSM Inference and Missing-Requirement Engine.

Handles the four required FSM cases:
1. Explicit FSM: Preserves specified states and transitions.
2. Inferred FSM: When no explicit FSM exists but sequential behavior has distinct phases, proposes an inferred FSM tagged as INFERRED with HITL routing.
3. Insufficient behavioral information: Identifies missing requirements, marks SPECIFICATION_INCOMPLETE, and requests clarification.
4. FSM unnecessary: Pure combinational or simple datapath designs generate plans without an artificial FSM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.efs_ir.models import EFSIR, EFSFSM, EFSState, EFSTransition
from core.efs_ir.rtl_plan import (
    EvidenceClassification,
    PlanFSM,
    PlanFSMTransition,
)
from core.agents.spec_classifier import classify_hardware_architecture, ArchitectureClass
from utils.logger import get_logger

logger = get_logger("agents.fsm_inference_engine")


class FSMHandlingCase(str, Enum):
    """The four canonical FSM handling modes."""
    EXPLICIT_FSM = "EXPLICIT_FSM"
    INFERRED_FSM = "INFERRED_FSM"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"
    FSM_UNNECESSARY = "FSM_UNNECESSARY"


@dataclass
class FSMInferenceResult:
    """Result of FSM analysis and inference."""
    case: FSMHandlingCase
    fsm: Optional[PlanFSM] = None
    requires_hitl_approval: bool = False
    missing_requirements: List[Dict[str, Any]] = field(default_factory=list)
    inference_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case": self.case.value,
            "fsm": self.fsm.to_dict() if self.fsm else None,
            "requires_hitl_approval": self.requires_hitl_approval,
            "missing_requirements": self.missing_requirements,
            "inference_notes": self.inference_notes,
        }


class FSMInferenceEngine:
    """Analyzes specifications and EFS IR to deterministically handle FSMs."""

    @staticmethod
    def resolve_fsm(
        efs_ir: Optional[EFSIR] = None,
        context_pack: Optional[Any] = None,
        spec_text: str = "",
        compiled_rtl_context: Optional[Dict[str, Any]] = None,
    ) -> FSMInferenceResult:
        """
        Determine which of the 4 FSM cases applies and generate the appropriate PlanFSM or diagnostics.
        """
        text_lower = (spec_text or "").lower()

        # 1. First check if FSM is unnecessary (Case 4: Pure combinational or simple register datapath)
        arch_res = classify_hardware_architecture(efs_ir=efs_ir, spec_text=spec_text)

        # Detect pure combinational
        if arch_res.primary_class == ArchitectureClass.COMBINATIONAL or (not arch_res.has_clock and not arch_res.has_reset and not arch_res.requires_fsm):
            logger.info("FSM Case 4: Pure combinational circuit - FSM is unnecessary.")
            return FSMInferenceResult(
                case=FSMHandlingCase.FSM_UNNECESSARY,
                fsm=None,
                requires_hitl_approval=False,
                inference_notes="Pure combinational logic does not require an FSM."
            )

        # Detect simple register/counter datapath without multi-state controller
        is_simple_pipeline_or_counter = (
            arch_res.primary_class in (ArchitectureClass.DATAPATH, ArchitectureClass.SEQUENTIAL, ArchitectureClass.ARITHMETIC_PROCESSING_UNIT)
            and not arch_res.requires_fsm
            and not re.search(r"\b(state\s+machine|fsm|states:|transition)\b", text_lower)
        )
        if is_simple_pipeline_or_counter:
            logger.info("FSM Case 4: Simple datapath/register/counter - FSM is unnecessary.")
            return FSMInferenceResult(
                case=FSMHandlingCase.FSM_UNNECESSARY,
                fsm=None,
                requires_hitl_approval=False,
                inference_notes="Standard register transfer / counter datapath does not require an artificial FSM."
            )

        # 2. Check for Explicit FSM in EFS IR or ContextPack (Case 1: Explicit FSM)
        target_fsm: Optional[Any] = None
        if context_pack:
            cp_dict = context_pack.to_dict() if hasattr(context_pack, "to_dict") else context_pack
            if isinstance(cp_dict, dict) and cp_dict.get("target_fsm"):
                target_fsm = cp_dict["target_fsm"]
            elif isinstance(cp_dict, dict) and cp_dict.get("fsm"):
                target_fsm = cp_dict["fsm"]

        if not target_fsm and efs_ir and getattr(efs_ir, "fsms", None) and len(efs_ir.fsms) > 0:
            target_fsm = efs_ir.fsms[0]

        if target_fsm:
            # Check for Case 3 (Insufficient behavioral information inside explicit FSM)
            raw_transitions = (
                target_fsm.get("transitions", [])
                if isinstance(target_fsm, dict)
                else getattr(target_fsm, "transitions", [])
            )
            raw_states = (
                target_fsm.get("states", [])
                if isinstance(target_fsm, dict)
                else getattr(target_fsm, "states", [])
            )

            state_dur_map: Dict[str, int] = {}
            for s in raw_states:
                s_name = s.get("name") if isinstance(s, dict) else getattr(s, "name", str(s))
                s_dur = s.get("duration_cycles") if isinstance(s, dict) else getattr(s, "duration_cycles", None)
                if s_dur is not None:
                    state_dur_map[s_name] = int(s_dur)
                    state_dur_map[re.sub(r"(?i)\s+state$", "", s_name).strip()] = int(s_dur)

            missing_conds = []
            for tr in raw_transitions:
                cond = tr.get("condition") if isinstance(tr, dict) else getattr(tr, "condition", None)
                cond_str = cond.get("expression", "") if isinstance(cond, dict) else str(cond or "")
                s_st = tr.get("from_state", tr.get("from", "")) if isinstance(tr, dict) else getattr(tr, "source_state", "")
                t_st = tr.get("to_state", tr.get("to", "")) if isinstance(tr, dict) else getattr(tr, "target_state", "")
                
                tr_dur = tr.get("duration_cycles") if isinstance(tr, dict) else getattr(tr, "duration_cycles", None)
                if tr_dur is None:
                    tr_dur = state_dur_map.get(s_st) or state_dur_map.get(re.sub(r"(?i)\s+state$", "", s_st).strip())

                if cond_str.strip() == "" and tr_dur is None:
                    missing_conds.append({
                        "category": "FSM",
                        "issue": f"Transition '{s_st}' -> '{t_st}' has no trigger condition",
                        "affected_state": s_st,
                        "target_state": t_st,
                        "required_information": f"Specify condition triggering transition from '{s_st}' to '{t_st}'."
                    })

            if missing_conds:
                logger.info(f"FSM Case 3: Insufficient behavioral information in FSM ({len(missing_conds)} missing conditions).")
                return FSMInferenceResult(
                    case=FSMHandlingCase.INSUFFICIENT_INFORMATION,
                    fsm=None,
                    requires_hitl_approval=True,
                    missing_requirements=missing_conds,
                    inference_notes="FSM declared with empty transition conditions."
                )

            # Build Explicit PlanFSM
            clean_states = [s.get("name") if isinstance(s, dict) else (getattr(s, "name", str(s))) for s in raw_states]
            init_st = (
                target_fsm.get("initial_state")
                if isinstance(target_fsm, dict)
                else getattr(target_fsm, "initial_state", "")
            ) or (clean_states[0] if clean_states else "IDLE")

            plan_transitions = []
            for tr in raw_transitions:
                s_st = tr.get("from_state", tr.get("from", "")) if isinstance(tr, dict) else getattr(tr, "source_state", "")
                t_st = tr.get("to_state", tr.get("to", "")) if isinstance(tr, dict) else getattr(tr, "target_state", "")
                cond = tr.get("condition") if isinstance(tr, dict) else getattr(tr, "condition", "1'b1")
                cond_expr = cond.get("expression", "1'b1") if isinstance(cond, dict) else str(cond or "1'b1")
                evid = tr.get("source_evidence", "") if isinstance(tr, dict) else getattr(tr, "evidence", "")
                
                tr_dur = tr.get("duration_cycles") if isinstance(tr, dict) else getattr(tr, "duration_cycles", None)
                if tr_dur is None:
                    tr_dur = state_dur_map.get(s_st) or state_dur_map.get(re.sub(r"(?i)\s+state$", "", s_st).strip())

                if tr_dur is not None and (cond_expr.strip() in ("", "1'b1", "1", "true", "unconditional", "timed") or "cycle" in cond_expr.lower()):
                    term_val = max(0, int(tr_dur) - 1)
                    w = max(4, term_val.bit_length())
                    cond_expr = f"timer_cnt >= {w}'d{term_val}"

                plan_transitions.append(
                    PlanFSMTransition(
                        from_state=s_st,
                        to_state=t_st,
                        condition_expression=cond_expr,
                        is_fully_defined=True,
                        classification=EvidenceClassification.EXPLICIT,
                        evidence=evid
                    )
                )

            plan_fsm = PlanFSM(
                states=clean_states,
                initial_state=init_st,
                transitions=plan_transitions,
                fsm_required=True,
                fsm_type="explicit",
                classification=EvidenceClassification.EXPLICIT
            )
            logger.info(f"FSM Case 1: Explicit FSM preserved ({len(clean_states)} states, {len(plan_transitions)} transitions).")
            return FSMInferenceResult(
                case=FSMHandlingCase.EXPLICIT_FSM,
                fsm=plan_fsm,
                requires_hitl_approval=False,
                inference_notes="Explicit FSM extracted from specification / EFS IR."
            )

        # 3. Check for Inferred FSM (Case 2: Sequential design with distinct control phases)
        # Search text for state names or sequential phases (e.g. S_IDLE, S_COUNT, etc.)
        state_matches = re.findall(r"\bS_[A-Za-z0-9_]+\b", spec_text)
        if not state_matches:
            # Match "state RED" or "states: RED, GREEN"
            state_matches = re.findall(r"\b(?:states?[:\s]+|state\s+)([A-Z][A-Z0-9_]{1,})\b", spec_text)
        if not state_matches:
            # Match "RED state", "GREEN state", "YELLOW state"
            state_matches = re.findall(r"\b([A-Z][A-Z0-9_]{1,})\s+states?\b", spec_text)
        if not state_matches:
            # Match Markdown table rows where first column is a state: | RED | 1 | 0 | ...
            table_states = re.findall(r"\|\s*([A-Z][A-Z0-9_]{1,})\s*\|[^|\n]+\|", spec_text)
            clean_tbl_states = [s for s in table_states if s not in ("STATE", "NAME", "PORT", "SIGNAL", "INPUT", "OUTPUT", "REGISTER")]
            if clean_tbl_states:
                state_matches = clean_tbl_states
        if not state_matches and ("controller" in text_lower or "sequence" in text_lower or "dwell" in text_lower or "traffic" in text_lower):
            # Check for phase words
            phase_candidates = []
            for cand in ["IDLE", "WAIT", "READY", "RUN", "ACTIVE", "COUNT", "DONE", "ERROR", "TX", "RX", "RED", "GREEN", "YELLOW"]:
                if re.search(rf"\b{cand}\b", spec_text):
                    phase_candidates.append(cand)
            if len(phase_candidates) >= 2:
                state_matches = phase_candidates

        if state_matches:
            dedup_states = list(dict.fromkeys(state_matches))
            # Propose inferred transitions between states

            def _find_state_dwell_condition(state_name: str) -> Optional[str]:
                clean_name = state_name.replace("S_", "").strip()
                # 1. Search markdown tables: | RED | ... | 5 cycles |
                tbl_m = re.search(rf"\|\s*{re.escape(clean_name)}\s*\|[^|\n]*\|[^|\n]*\|[^|\n]*\|\s*(\d+)\s*(?:clock\s*)?cycles?", spec_text, re.IGNORECASE)
                if not tbl_m:
                    tbl_m = re.search(rf"\|\s*{re.escape(clean_name)}\s*\|[^|\n]*?(\d+)\s*(?:clock\s*)?cycles?", spec_text, re.IGNORECASE)
                # 2. Search prose: RED light remains ON for 5 clock cycles
                if not tbl_m:
                    tbl_m = re.search(rf"\b{re.escape(clean_name)}\b.*?(?:remains?\s+(?:on|in)?\s+for|dwells?\s+(?:in|for)?|duration\s*(?:is|of)?|for)\s*(\d+)\s*(?:clock\s*)?cycles?", spec_text, re.IGNORECASE)
                if tbl_m:
                    dur = int(tbl_m.group(1))
                    term_val = max(0, dur - 1)
                    w = max(4, term_val.bit_length())
                    return f"timer_cnt >= {w}'d{term_val}"
                return None

            inferred_transitions = []
            for i in range(len(dedup_states) - 1):
                s_from = dedup_states[i]
                s_to = dedup_states[i + 1]
                dwell_cond = _find_state_dwell_condition(s_from)
                cond_expr = dwell_cond or ("start == 1'b1" if (i == 0 and "start" in spec_text.lower()) else "1'b1")
                inferred_transitions.append(
                    PlanFSMTransition(
                        from_state=s_from,
                        to_state=s_to,
                        condition_expression=cond_expr,
                        is_fully_defined=True,
                        classification=EvidenceClassification.INFERRED,
                        evidence=f"Inferred sequence transition from '{s_from}' to '{s_to}'" + (f" with duration condition '{dwell_cond}'" if dwell_cond else "")
                    )
                )
            # Add loop back to initial state
            last_state = dedup_states[-1]
            last_dwell = _find_state_dwell_condition(last_state)
            last_cond = last_dwell or "1'b1"
            inferred_transitions.append(
                PlanFSMTransition(
                    from_state=last_state,
                    to_state=dedup_states[0],
                    condition_expression=last_cond,
                    is_fully_defined=True,
                    classification=EvidenceClassification.INFERRED,
                    evidence=f"Inferred reset/loop transition back to '{dedup_states[0]}'" + (f" with duration condition '{last_dwell}'" if last_dwell else "")
                )
            )

            plan_fsm = PlanFSM(
                states=dedup_states,
                initial_state=dedup_states[0],
                transitions=inferred_transitions,
                fsm_required=True,
                fsm_type="inferred",
                classification=EvidenceClassification.INFERRED
            )
            logger.info(f"FSM Case 2: Inferred FSM proposed ({len(dedup_states)} states). Routing through HITL.")
            return FSMInferenceResult(
                case=FSMHandlingCase.INFERRED_FSM,
                fsm=plan_fsm,
                requires_hitl_approval=True,
                inference_notes="Sequential controller phases inferred from specification text; requires human approval."
            )

        # 4. Check if sequential behavior is present but completely unstated (Case 3)
        if "fsm" in text_lower or "sequence controller" in text_lower or "state machine" in text_lower:
            logger.info("FSM Case 3: FSM mentioned but states/transitions unstated.")
            return FSMInferenceResult(
                case=FSMHandlingCase.INSUFFICIENT_INFORMATION,
                fsm=None,
                requires_hitl_approval=True,
                missing_requirements=[{
                    "category": "FSM",
                    "issue": "State machine behavior mentioned but states and transitions are completely unspecified.",
                    "required_information": "Provide explicit FSM states, conditions, and transition table."
                }],
                inference_notes="FSM controller mentioned without defining states or transitions."
            )

        # Fallback to Case 4 (FSM Unnecessary)
        return FSMInferenceResult(
            case=FSMHandlingCase.FSM_UNNECESSARY,
            fsm=None,
            requires_hitl_approval=False,
            inference_notes="No FSM requirements identified in specification."
        )
