"""
Generic Behavioral and FSM Completeness Analyzer.

Analyzes hardware specifications and EFS IR models to:
1. Determine whether a design is sequential or purely combinational.
2. Detect both explicitly defined and implicitly described sequential control.
3. Infer missing behavioral elements (states, transitions, control signals, reset behavior)
   ONLY when supported by extracted requirements, while preserving provenance.
4. Avoid assuming that every design requires an FSM, counter, clock, or reset.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from core.efs_ir.models import (
    EFSIR, EFSFSM, EFSState, EFSTransition, EFSSignal, EFSTimingRule, EFSTraceability,
    new_efs_id
)
from core.efs_ir.validation_rules import (
    is_fsm_state_name,
    is_fsm_transition_or_condition,
    is_physical_signal_name
)
from utils.logger import get_logger

logger = get_logger("efs_ir.behavioral_analyzer")


class GenericBehavioralAnalyzer:
    """
    Specification-agnostic behavioral analyzer for hardware designs.
    """

    @classmethod
    def is_sequential_design(
        cls,
        ir: Optional[EFSIR] = None,
        req_model: Optional[Dict[str, Any]] = None,
        spec_text: str = "",
        plan: Optional[Dict[str, Any]] = None,
        doc_ir: Optional[Any] = None
    ) -> bool:
        """
        Determine if the hardware design exhibits or requires sequential behavior.
        A design is sequential if:
        1. It has explicit registers, clocks, resets, or FSM definitions.
        2. Its requirements/specifications describe multi-cycle behavior, latches, flip-flops,
           clock edges, state transitions, or delay cycles.
        If none of these are present and only combinational operations exist, it is combinational.
        """
        # 1. Existing IR evidence
        if ir is not None:
            if ir.registers or ir.fsms or ir.clocks or ir.resets:
                return True
            if ir.components:
                for comp in ir.components:
                    c_name = comp.name.lower()
                    c_type = (comp.type or "").lower()
                    if any(k in c_name or k in c_type for k in ("core", "cpu", "processor", "controller", "fsm", "counter", "timer", "engine", "sequencer", "pipeline")):
                        return True
            for sig in ir.signals:
                if sig.semantic_role in ("clock", "reset"):
                    return True
                s_name = sig.name.lower()
                if any(k in s_name for k in ("clk", "clock", "rst", "reset")):
                    return True
            if getattr(ir, "timing_rules", None):
                for tr in ir.timing_rules:
                    if any(k in (tr.cycle_relationship or "").lower() for k in ("cycle", "clock", "delay", "remains on")):
                        return True

        # 2. req_model / plan evidence
        if req_model:
            if req_model.get("registers") or req_model.get("clocks_resets") or req_model.get("fsm_info") or req_model.get("states"):
                return True
            for cr in req_model.get("clocks_resets", []):
                return True
        if plan and isinstance(plan, dict):
            if plan.get("fsm_architecture") or plan.get("state_machine") or plan.get("clock_domains"):
                return True

        # 3. Textual evidence check
        ir_dsn = getattr(ir.metadata, "design_name", "") if (ir and getattr(ir, "metadata", None)) else ""
        combined_text = (spec_text + " " + (ir_dsn or "")).lower()
        if doc_ir:
            for b in getattr(doc_ir, "blocks", []):
                if getattr(b, "text", None):
                    combined_text += " " + b.text.lower()
        if req_model and isinstance(req_model, dict):
            for r in req_model.get("requirements", []) + req_model.get("functional_requirements", []) + req_model.get("timing_rules", []):
                stmt = r.get("statement", "") if isinstance(r, dict) else str(r)
                combined_text += " " + stmt.lower()
        if plan and isinstance(plan, dict):
            p_fsm = plan.get("fsm_architecture") or plan.get("state_machine") or {}
            if isinstance(p_fsm, dict) and p_fsm.get("description"):
                combined_text += " " + str(p_fsm["description"]).lower()

        # Filter out negated clock/reset phrases (e.g., "without clock or reset", "no clock")
        cleaned_text = re.sub(r"\b(?:without|no|neither)\s+[^.;\n]*?\b(?:clocks?|resets?|registers?|fsm)(?:\s+(?:or|and|nor)\s+(?:clocks?|resets?|registers?|fsm))*\b", " ", combined_text)

        sequential_patterns = [
            r"\b(?:clock|clk|posedge|negedge|clock\s+edge|active\s+edge)\b",
            r"\b(?:reset|rst|active-low\s+reset|active-high\s+reset)\b",
            r"\b(?:fsm|state\s+machine|states?|transitions?|next\s+state|current\s+state)\b",
            r"\b(?:operating\s+states?|operating\s+modes?|switches\s+to|advances\s+to)\b",
            r"\b(?:flip-flop|flip\s+flop|register|latch|counter|count\s+up|count\s+down)\b",
            r"\b(?:\d+\s+cycles?|clock\s+cycles?|cycle-accurate|sequential|multi-cycle|dwell\s+time)\b",
        ]
        for pat in sequential_patterns:
            if re.search(pat, cleaned_text):
                return True

        return False

    @classmethod
    def analyze_and_complete_behavior(
        cls,
        ir: EFSIR,
        req_model: Optional[Dict[str, Any]] = None,
        spec_text: str = "",
        plan: Optional[Dict[str, Any]] = None,
        doc_ir: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze behavioral completeness and infer supported sequential structures.
        Returns a list of behavioral decisions made (with provenance).
        """
        decisions: List[Dict[str, Any]] = []

        # Extract nested plan / doc_ir from req_model if not explicitly provided
        if plan is None and isinstance(req_model, dict) and "plan" in req_model:
            plan = req_model["plan"]
        if doc_ir is None and isinstance(req_model, dict) and "doc_ir" in req_model:
            doc_ir = req_model["doc_ir"]

        is_seq = cls.is_sequential_design(ir, req_model, spec_text, plan=plan, doc_ir=doc_ir)

        # Purely combinational designs must NOT have sequential clocks, resets, or FSMs inferred
        if not is_seq:
            return decisions

        # 1. Clock and Reset Analysis
        clk_names = {s.name.lower() for s in ir.signals if s.semantic_role == "clock"}
        rst_names = {s.name.lower() for s in ir.signals if s.semantic_role == "reset"}

        # Also check signal names
        for s in ir.signals:
            if any(k in s.name.lower() for k in ("clk", "clock")) and s.name.lower() not in clk_names:
                s.semantic_role = "clock"
                clk_names.add(s.name.lower())
            elif any(k in s.name.lower() for k in ("rst", "reset")) and s.name.lower() not in rst_names:
                s.semantic_role = "reset"
                rst_names.add(s.name.lower())

        # Infer primary clock if missing in sequential design
        if not clk_names:
            clk_sig = EFSSignal(
                signal_id=new_efs_id("sig"),
                name="clk",
                width="1",
                direction="input",
                datatype="logic",
                semantic_role="clock",
                description="Primary system clock signal.",
                traceability=EFSTraceability(
                    extraction_method="inferred",
                    confidence=0.85,
                    original_text="Inferred primary clock for sequential hardware logic."
                )
            )
            ir.signals.insert(0, clk_sig)
            ir.clocks.append(clk_sig)
            decisions.append({
                "action": "infer_clock",
                "signal": "clk",
                "provenance": "inferred",
                "reason": "Sequential hardware logic requires an active clock input."
            })

        # Infer primary reset if missing in sequential design
        if not rst_names:
            rst_sig = EFSSignal(
                signal_id=new_efs_id("sig"),
                name="rst_n",
                width="1",
                direction="input",
                datatype="logic",
                semantic_role="reset",
                description="Primary active-low system reset signal.",
                traceability=EFSTraceability(
                    extraction_method="inferred",
                    confidence=0.85,
                    original_text="Inferred primary active-low reset for sequential hardware initialization."
                )
            )
            ir.signals.insert(1, rst_sig)
            ir.resets.append(rst_sig)
            decisions.append({
                "action": "infer_reset",
                "signal": "rst_n",
                "provenance": "inferred",
                "reason": "Sequential hardware logic requires a reset mechanism for deterministic initialization."
            })

        # Synchronize clocks & resets
        ir.clocks = [s for s in ir.signals if s.semantic_role == "clock"]
        ir.resets = [s for s in ir.signals if s.semantic_role == "reset"]

        # 2. FSM Analysis and Completion
        cls._analyze_and_complete_fsm(ir, req_model, spec_text, plan, doc_ir, decisions)

        return decisions

    @classmethod
    def _analyze_and_complete_fsm(
        cls,
        ir: EFSIR,
        req_model: Optional[Dict[str, Any]],
        spec_text: str,
        plan: Optional[Dict[str, Any]],
        doc_ir: Optional[Any],
        decisions: List[Dict[str, Any]]
    ) -> None:
        """
        Analyze FSM completeness:
        - If FSM states exist, ensure transitions, conditions, and references are valid.
        - If requirements describe a sequential state progression, construct a canonical EFSFSM
          grounded in the extracted requirements, plan, timing rules, and source text.
        - Do not construct an artificial FSM if the design lacks state behavior.
        """
        # Step A: Validate and normalize existing FSMs
        for fsm in ir.fsms:
            if not fsm.name:
                design_n = getattr(ir.metadata, "design_name", "") or "main"
                fsm.name = f"{design_n}_fsm"

            if not fsm.reset_state and fsm.initial_state:
                fsm.reset_state = fsm.initial_state
            if not fsm.reset_behavior and ir.resets:
                rst_name = ir.resets[0].name
                rst_pol = "low" if "n" in rst_name.lower() else "high"
                fsm.reset_behavior = f"Asynchronous active-{rst_pol} reset to {fsm.reset_state or fsm.initial_state} on {rst_name}"

            defined_states = {s.name for s in fsm.states}
            for tr in fsm.transitions:
                if tr.source_state and tr.source_state not in defined_states:
                    st = EFSState(
                        state_id=new_efs_id("st"),
                        name=tr.source_state,
                    )
                    fsm.states.append(st)
                    defined_states.add(tr.source_state)
                    decisions.append({
                        "action": "add_fsm_state",
                        "provenance": "inferred",
                        "fsm": fsm.name,
                        "state": tr.source_state,
                        "reason": f"State referenced in transition {tr.source_state} -> {tr.target_state}."
                    })
                if tr.target_state and tr.target_state not in defined_states:
                    st = EFSState(
                        state_id=new_efs_id("st"),
                        name=tr.target_state,
                    )
                    fsm.states.append(st)
                    defined_states.add(tr.target_state)
                    decisions.append({
                        "action": "add_fsm_state",
                        "provenance": "inferred",
                        "fsm": fsm.name,
                        "state": tr.target_state,
                        "reason": f"State referenced in transition {tr.source_state} -> {tr.target_state}."
                    })

            # If states exist but transitions are missing, infer sequential transitions
            if fsm.states and not fsm.transitions:
                for i in range(len(fsm.states)):
                    st_cur = fsm.states[i]
                    s_cur = st_cur.name
                    s_nxt = fsm.states[(i + 1) % len(fsm.states)].name
                    dur = getattr(st_cur, "duration_cycles", None)
                    if dur is not None and dur > 0:
                        t_cond = f"{dur} clock cycles"
                        t_type = "timed"
                    else:
                        t_cond = ""
                        t_type = "unresolved"
                    fsm.transitions.append(EFSTransition(
                        transition_id=new_efs_id("trans"),
                        source_state=s_cur,
                        target_state=s_nxt,
                        condition=t_cond,
                        condition_type=t_type,
                        duration_cycles=dur,
                        traceability=fsm.traceability
                    ))

            # Ensure initial and reset state
            if not fsm.initial_state and fsm.states:
                fsm.initial_state = fsm.states[0].name
            if not fsm.reset_state and fsm.initial_state:
                fsm.reset_state = fsm.initial_state

        # Step B: If no FSM exists in IR, check behavioral evidence to infer canonical FSM
        if not ir.fsms:
            cls._infer_fsm_from_evidence(ir, req_model, spec_text, plan, doc_ir, decisions)

    @classmethod
    def _infer_fsm_from_evidence(
        cls,
        ir: EFSIR,
        req_model: Optional[Dict[str, Any]],
        spec_text: str,
        plan: Optional[Dict[str, Any]],
        doc_ir: Optional[Any],
        decisions: List[Dict[str, Any]]
    ) -> None:
        """
        Generic, specification-agnostic inference of an FSM from behavioral evidence,
        including state names, timing rules, dwell counters, transitions, and state outputs.
        """
        # Collect all text and structured descriptions
        text_corpus: List[str] = []
        if spec_text:
            text_corpus.append(spec_text)
        if doc_ir:
            for b in getattr(doc_ir, "blocks", []):
                if getattr(b, "text", None):
                    text_corpus.append(b.text)
        for tr in ir.timing_rules:
            if tr.cycle_relationship:
                text_corpus.append(tr.cycle_relationship)
        for fb in ir.functional_behavior:
            if fb.description:
                text_corpus.append(fb.description)
            if fb.trigger:
                text_corpus.append(fb.trigger)
            if fb.expected_result:
                text_corpus.append(fb.expected_result)
        for fl in ir.flows:
            for st in getattr(fl, "ordered_steps", []):
                if getattr(st, "action_description", None):
                    text_corpus.append(st.action_description)
        if req_model and isinstance(req_model, dict):
            for r in req_model.get("requirements", []) + req_model.get("functional_requirements", []) + req_model.get("timing_rules", []):
                stmt = r.get("statement", "") if isinstance(r, dict) else str(r)
                if stmt:
                    text_corpus.append(stmt)
        if plan and isinstance(plan, dict):
            p_fsm = plan.get("fsm_architecture") or plan.get("state_machine") or plan.get("fsm") or {}
            if isinstance(p_fsm, dict) and p_fsm.get("description"):
                text_corpus.append(str(p_fsm["description"]))

        full_text = "\n".join(text_corpus)
        if not full_text.strip() and not plan and not req_model:
            return

        # 1. Discover Candidate States (preserving document order)
        discovered_states: List[str] = []
        discovered_set: Set[str] = set()
        plan_state_data: Dict[str, Dict[str, Any]] = {}

        # Check if control signals (signals other than clock and reset) exist in IR
        control_signals = [
            s for s in ir.signals
            if (s.semantic_role or "").lower() not in ("clock", "reset")
            and not any(k in s.name.lower() for k in ("clk", "clock", "rst", "reset"))
        ]
        has_source_evidence = bool(
            full_text.strip() or
            (ir.timing_rules and len(ir.timing_rules) > 0) or
            (req_model and (req_model.get("fsm_info") or req_model.get("states") or req_model.get("timing_rules")))
        )
        has_plan_transitions = bool(
            plan and isinstance(plan, dict) and
            (plan.get("fsm_architecture", {}).get("transitions") or plan.get("transitions"))
        )

        # 1a. Extract from plan if supported by source evidence, plan transitions, or control signals
        if plan and isinstance(plan, dict):
            if has_source_evidence or control_signals or has_plan_transitions:
                p_fsm = plan.get("fsm_architecture") or plan.get("state_machine") or plan.get("fsm") or {}
                if isinstance(p_fsm, dict):
                    for s in p_fsm.get("states", []):
                        s_name = (s.get("name") if isinstance(s, dict) else str(s)).strip().upper()
                        if s_name:
                            plan_state_data[s_name] = s if isinstance(s, dict) else {"name": s_name}
                            if s_name not in discovered_set:
                                discovered_states.append(s_name)
                                discovered_set.add(s_name)

        # 1b. Extract from req_model if present
        if req_model and isinstance(req_model, dict):
            f_info = req_model.get("fsm_info") or {}
            raw_states = f_info.get("states") or req_model.get("states") or []
            for s in raw_states:
                s_name = (s.get("name") if isinstance(s, dict) else str(s)).strip().upper()
                if s_name and s_name not in discovered_set:
                    discovered_states.append(s_name)
                    discovered_set.add(s_name)

        # 1c. Generic NLP pattern extraction from specification text & timing rules
        reserved_words = {
            "THE", "AND", "FOR", "NOT", "NOTE", "RULE", "SECTION", "TABLE", "CLK", "CLOCK",
            "RESET", "RST", "RST_N", "INPUT", "OUTPUT", "INOUT", "MODULE", "CONTROLLER",
            "SYSTEM", "DEVICE", "SIGNAL", "SIGNALS", "PORT", "PORTS", "DATA", "WIRE",
            "LOGIC", "REGISTER", "REGISTERS", "REG", "MEMORY", "BUS", "INTERFACE",
            "REQ", "ACK", "TRUE", "FALSE", "HIGH", "LOW", "OF", "WITH", "FROM", "WHEN",
            "STATE", "STATES", "MODE", "MODES", "ALL", "ONE", "ANY", "EACH", "THIS",
            "WHICH", "THAT", "WHO", "WHERE", "WHILE", "THEN", "AFTER", "BEFORE", "DURING",
            "UNTIL", "ONCE", "SINCE", "BECAUSE", "ALTHOUGH", "THOUGH", "UNLESS", "THERE",
            "HERE", "SOME", "MORE", "LESS", "OTHER", "ANOTHER", "EITHER", "NEITHER", "BOTH",
            "EVERY", "NONE", "ALSO", "THESE", "THOSE", "SUCH", "SAME"
        }

        # Patterns matching operating states or modes
        state_patterns = [
            r"\b(?:switches to|advances to|transitions to|enters|moves to|remains in|during|after)\s+(?:the\s+)?([A-Za-z0-9_]{2,24})\s+(?:state|mode|phase)\b",
            r"\b([A-Z0-9_]{2,20})\s+(?:state|mode|phase)\b",
            r"\b([A-Za-z0-9_]{2,20})\s+(?:light|output|flag|signal|phase)?\s*(?:(?:remains\s+|is\s+)?(?:on|active|high|asserted))\s+for\s+\d+\s+(?:clock\s+)?cycles?",
            r"(?:(?:fsm\s+)?states|modes|phases)\s*:\s*([A-Za-z0-9_,\s]+)",
            r"(?:cycles through|switches between|operates across)\s+([A-Za-z0-9_,\s]+)",
            r"\b([A-Za-z0-9_]{2,20})\s*(?:->|-->|=>)\s*\b([A-Za-z0-9_]{2,20})\b",
            r"transition(?:s)?\s+from\s+([A-Za-z0-9_]+)\s+to\s+([A-Za-z0-9_]+)",
            r"^[-*]\s*\b([A-Z0-9_]{2,20})\b\s*:\s*(?:state|mode|phase|waiting|operates|handles|active|[A-Za-z\s]+remains\s+ON)"
        ]

        for pat in state_patterns:
            for match in re.finditer(pat, full_text, re.IGNORECASE | re.MULTILINE):
                groups = match.groups()
                for grp in groups:
                    if not grp:
                        continue
                    # Might contain comma-separated list
                    items = [x.strip() for x in re.split(r"[,;/]|\band\b", grp) if x.strip()]
                    for item in items:
                        clean_item = item.strip().upper()
                        if clean_item and clean_item not in reserved_words and len(clean_item) >= 2:
                            if re.match(r"^[A-Z_][A-Z0-9_]*$", clean_item) and not clean_item.isdigit():
                                if clean_item not in discovered_set:
                                    discovered_states.append(clean_item)
                                    discovered_set.add(clean_item)

        # Sort states by first appearance in full_text if discovered from text patterns
        if full_text:
            def _first_pos(st: str) -> int:
                pos = full_text.upper().find(st)
                return pos if pos >= 0 else 999999
            discovered_states.sort(key=_first_pos)

        # Require at least 2 distinct states to infer a genuine FSM
        if len(discovered_states) < 2:
            return

        # 2. Build State Entities with Output Associations and Actions
        fsm_states: List[EFSState] = []
        fsm_outputs: Set[str] = set()

        for st_name in discovered_states:
            st_data = plan_state_data.get(st_name, {})
            actions = []
            if st_data.get("actions"):
                a_val = st_data["actions"]
                actions = [a_val] if isinstance(a_val, str) else list(a_val)

            outputs: Dict[str, str] = {}
            if st_data.get("outputs") and isinstance(st_data["outputs"], dict):
                outputs.update(st_data["outputs"])

            # Associate physical signals with state names
            # E.g. signal 'red_light' matches state 'RED'
            s_low = st_name.lower()
            for sig in ir.signals:
                sig_low = sig.name.lower()
                is_match = (
                    sig_low == s_low or
                    sig_low.startswith(f"{s_low}_") or
                    sig_low.endswith(f"_{s_low}") or
                    f"_{s_low}_" in sig_low or
                    s_low in (sig.description or "").lower() or
                    f"{sig_low} output for the {s_low} state" in full_text.lower() or
                    f"{s_low} light remains on" in full_text.lower()
                )
                if is_match:
                    outputs[sig.name] = "1'b1"
                    fsm_outputs.add(sig.name)
                    act_str = f"{sig.name} = 1'b1"
                    if not any(sig.name in a for a in actions):
                        actions.append(act_str)

            fsm_states.append(EFSState(
                state_id=new_efs_id("st"),
                name=st_name,
                outputs=outputs,
                actions=actions
            ))

        # Enforce mutual exclusivity on state outputs if multiple states have single-active signals
        if len(fsm_outputs) >= 2:
            for st in fsm_states:
                for o_sig in fsm_outputs:
                    if o_sig not in st.outputs:
                        st.outputs[o_sig] = "1'b0"

        # 3. Derive Transitions and Timing Guards
        fsm_trans: List[EFSTransition] = []

        # 3a. Check explicit transitions from req_model or plan
        explicit_t_defs = []
        if req_model and isinstance(req_model, dict):
            explicit_t_defs.extend(req_model.get("fsm_info", {}).get("transitions", []) or req_model.get("transitions", []))
        if plan and isinstance(plan, dict):
            p_fsm = plan.get("fsm_architecture") or plan.get("state_machine") or plan.get("fsm") or {}
            if isinstance(p_fsm, dict):
                explicit_t_defs.extend(p_fsm.get("transitions", []))

        for tr in explicit_t_defs:
            src = (tr.get("from") or tr.get("source_state") or tr.get("source") or "").strip().upper()
            dst = (tr.get("to") or tr.get("target_state") or tr.get("target") or "").strip().upper()
            cond = tr.get("condition") or "1'b1"
            if src in discovered_set and dst in discovered_set:
                fsm_trans.append(EFSTransition(
                    transition_id=new_efs_id("trans"),
                    source_state=src,
                    target_state=dst,
                    condition=str(cond),
                    condition_type="boolean" if any(k in str(cond) for k in ("==", "!=", "1'b")) else "semantic",
                    traceability=EFSTraceability(extraction_method="source_extracted", confidence=0.95)
                ))

        # 3b. Extract direct transition notation from text: A -> B or from A to B
        if not fsm_trans:
            matches_a = re.findall(r"\b([A-Za-z0-9_]{2,20})\b\s*(?:->|-->|=>)\s*\b([A-Za-z0-9_]{2,20})\b(?:\s*(?:[:]|when|on|if|after)\s*([^\n\r,;.]+))?", full_text, re.IGNORECASE)
            matches_b = re.findall(r"(?:transition(?:s)?\s+from|(?:from\s+))\b([A-Za-z0-9_]{2,20})\b\s*,?\s*(?:it\s+)?(?:transition(?:s)?\s+to|advances?\s+to|moves?\s+to|resets?\s+to|switches?\s+to)\s+\b([A-Za-z0-9_]{2,20})\b(?:\s*(?:when|on|if|after|:)\s*([^\n\r,;.]+))?", full_text, re.IGNORECASE)
            for src_raw, dst_raw, cond_raw in list(matches_a) + list(matches_b):
                src_c = src_raw.strip().upper()
                dst_c = dst_raw.strip().upper()
                if src_c in discovered_set and dst_c in discovered_set and src_c != dst_c:
                    if not any(t.source_state == src_c and t.target_state == dst_c for t in fsm_trans):
                        fsm_trans.append(EFSTransition(
                            transition_id=new_efs_id("trans"),
                            source_state=src_c,
                            target_state=dst_c,
                            condition=(cond_raw or "1'b1").strip(),
                            condition_type="boolean" if any(k in str(cond_raw) for k in ("==", "!=", "1'b")) else "semantic",
                            traceability=EFSTraceability(extraction_method="behavioral_inference", confidence=0.90)
                        ))

        # 3c. If no explicit transitions, infer sequential and cyclic transitions between states
        if not fsm_trans and len(fsm_states) >= 2:
            for i in range(len(fsm_states)):
                src = fsm_states[i].name
                dst = fsm_states[(i + 1) % len(fsm_states)].name

                # Search timing constraints / duration for src state
                cond = "auto_advance"
                cond_type = "semantic"
                dur_m = re.search(rf"\b{re.escape(src.lower())}\b[^\n\r.]*?(?:for|remains\s+on\s+for|after)\s+(\d+)\s+(?:clock\s+)?cycles?", full_text, re.IGNORECASE)
                if not dur_m:
                    dur_m = re.search(rf"(?:for|remains\s+on\s+for|after)\s+(\d+)\s+(?:clock\s+)?cycles?[^\n\r.]*?\b{re.escape(src.lower())}\b", full_text, re.IGNORECASE)
                duration_cycles = None
                if dur_m:
                    cycles = int(dur_m.group(1))
                    cond = f"{cycles} clock cycles"
                    cond_type = "timed"
                    duration_cycles = cycles
                else:
                    start_sig = next((s.name for s in ir.signals if "start" in s.name.lower()), None)
                    done_sig = next((s.name for s in ir.signals if "done" in s.name.lower()), None)
                    if i == 0 and start_sig:
                        cond = f"{start_sig} asserted"
                        cond_type = "signal_assertion"
                    elif i == len(fsm_states) - 2 and done_sig:
                        cond = f"{done_sig} asserted"
                        cond_type = "signal_assertion"

                fsm_trans.append(EFSTransition(
                    transition_id=new_efs_id("trans"),
                    source_state=src,
                    target_state=dst,
                    condition=cond,
                    condition_type=cond_type,
                    duration_cycles=duration_cycles,
                    traceability=EFSTraceability(extraction_method="behavioral_inference", confidence=0.90)
                ))

        # 4. Determine Initial State and Reset Behavior
        initial_st = fsm_states[0].name
        rst_match = re.search(r"(?:after\s+reset\s+(?:is\s+released\s*,?\s*)?(?:the\s+)?|resets?\s+to\s+(?:the\s+)?|initial\s+state\s+(?:is\s+)?)\b([A-Za-z0-9_]+)\b", full_text, re.IGNORECASE)
        if rst_match:
            cand_rst = rst_match.group(1).strip().upper()
            if cand_rst in discovered_set:
                initial_st = cand_rst

        rst_sig_name = ir.resets[0].name if ir.resets else "rst_n"
        clk_sig_name = ir.clocks[0].name if ir.clocks else "clk"
        rst_pol = "low" if "n" in rst_sig_name.lower() else "high"
        reset_behavior = f"Asynchronous active-{rst_pol} reset to {initial_st} on {rst_sig_name}"

        # 5. Maintain Timing Rules Alignment
        for st in fsm_states:
            dur_m = re.search(rf"\b{re.escape(st.name.lower())}\b[^\n\r.]*?(?:for|remains\s+on\s+for|after)\s+(\d+)\s+(?:clock\s+)?cycles?", full_text, re.IGNORECASE)
            if dur_m:
                cycles = dur_m.group(1)
                r_name = f"{st.name.lower()}_dwell_time"
                if not any(r.name.lower() == r_name for r in ir.timing_rules):
                    ir.timing_rules.append(EFSTimingRule(
                        rule_id=new_efs_id("time"),
                        name=r_name,
                        type="cycle_relationship",
                        cycle_relationship=f"The {st.name} state remains active for {cycles} clock cycles.",
                        clock_signal=clk_sig_name,
                        traceability=EFSTraceability(extraction_method="behavioral_inference", confidence=0.90)
                    ))

        # 6. Construct Canonical EFSFSM
        design_name = getattr(ir.metadata, "design_name", "") or "controller"
        created_fsm = EFSFSM(
            fsm_id=new_efs_id("fsm"),
            name=f"{design_name}_fsm",
            initial_state=initial_st,
            terminal_states=[],
            reset_state=initial_st,
            reset_behavior=reset_behavior,
            encoding="binary",
            states=fsm_states,
            transitions=fsm_trans,
            outputs=sorted(list(fsm_outputs)),
            traceability=EFSTraceability(
                extraction_method="behavioral_inference",
                confidence=0.90,
                original_text="Inferred canonical FSM from behavioral sequence and timing requirements"
            )
        )
        ir.fsms.append(created_fsm)
        decisions.append({
            "action": "infer_fsm_from_behavioral_evidence",
            "fsm": created_fsm.name,
            "states": [s.name for s in fsm_states],
            "transitions_count": len(fsm_trans),
            "initial_state": initial_st,
            "outputs": list(fsm_outputs),
            "provenance": "behavioral_inference"
        })
