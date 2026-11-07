"""
Structured Specification Gap and Proposal Models for Human-in-the-Loop (HITL) Resolution.

Preserves the critical distinction between:
- Explicitly specified requirements (SOURCE_SUPPORTED)
- Logically derived requirements (DERIVED)
- Implementation choices (IMPLEMENTATION_CHOICE)
- Unsupported assumptions (UNSUPPORTED_ASSUMPTION)
- Genuine specification gaps requiring human authorization.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from core.efs_ir.models import EFSConflict, new_efs_id
from core.efs_ir.rtl_plan import PlanOrigin, PlanUnresolvedLogic


class SpecificationGapCategory(str, Enum):
    """Categorization of specification gaps and ambiguities in hardware models."""
    MISSING_BEHAVIOR = "MISSING_BEHAVIOR"
    AMBIGUOUS_REQUIREMENT = "AMBIGUOUS_REQUIREMENT"
    CONFLICTING_REQUIREMENTS = "CONFLICTING_REQUIREMENTS"
    MISSING_SIGNAL_DRIVER = "MISSING_SIGNAL_DRIVER"
    UNSPECIFIED_REGISTER_UPDATE = "UNSPECIFIED_REGISTER_UPDATE"
    UNSPECIFIED_RESET_BEHAVIOR = "UNSPECIFIED_RESET_BEHAVIOR"
    UNSPECIFIED_CONTROL_FLOW = "UNSPECIFIED_CONTROL_FLOW"
    INCOMPLETE_PROTOCOL_BEHAVIOR = "INCOMPLETE_PROTOCOL_BEHAVIOR"
    UNSUPPORTED_IMPLEMENTATION = "UNSUPPORTED_IMPLEMENTATION"


class GapSeverity(str, Enum):
    """Severity classification for hardware specification gaps."""
    WARNING = "WARNING"
    INCOMPLETE = "INCOMPLETE"
    BLOCKING = "BLOCKING"
    UNSUPPORTED = "UNSUPPORTED"
    ERROR = "ERROR"


class EvidenceClassification(str, Enum):
    """
    Evidence grounding classification for proposed hardware statements.
    Guarantees AI proposals never misrepresent implementation choices as source requirements.
    """
    SOURCE_SUPPORTED = "SOURCE_SUPPORTED"           # Explicitly mandated in the spec
    DERIVED = "DERIVED"                             # Deterministically derived with mathematical/logical proof
    IMPLEMENTATION_CHOICE = "IMPLEMENTATION_CHOICE" # Viable RTL design choice not mandated by spec
    UNSUPPORTED_ASSUMPTION = "UNSUPPORTED_ASSUMPTION" # Speculative assumption lacking evidence


class GapResolutionStatus(str, Enum):
    """Resolution lifecycle status for a specification gap."""
    UNRESOLVED = "UNRESOLVED"
    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    DEFERRED = "DEFERRED"
    EDITED = "EDITED"


class HITLAction(str, Enum):
    """Actions available to human engineer during specification gap review."""
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    EDIT = "EDIT"
    REQUEST_ALTERNATIVES = "REQUEST_ALTERNATIVES"
    REQUEST_CLARIFICATION = "REQUEST_CLARIFICATION"
    DEFER = "DEFER"


@dataclass
class SpecificationGapProposal:
    """
    Structured, evidence-grounded AI proposal resolving a hardware specification gap.
    """
    proposal_id: str = field(default_factory=lambda: f"PROP_{uuid.uuid4().hex[:8]}")
    gap_id: str = ""
    title: str = ""
    proposed_resolution: str = ""
    detailed_behavioral_explanation: str = ""
    proposed_rtl_semantics: Dict[str, Any] = field(default_factory=dict)
    relevant_signals: List[str] = field(default_factory=list)
    relevant_registers: List[str] = field(default_factory=list)
    relevant_fsm_states: List[str] = field(default_factory=list)
    conditions_and_timing: str = ""
    expected_effect: str = ""
    supporting_evidence: str = ""
    evidence_references: List[str] = field(default_factory=list)
    reasoning: str = ""
    explicit_assumptions: List[str] = field(default_factory=list)
    potential_risks: List[str] = field(default_factory=list)
    confidence_level: float = 0.8
    evidence_classification: EvidenceClassification = EvidenceClassification.IMPLEMENTATION_CHOICE
    is_safe_proposal: bool = True
    clarification_questions: List[str] = field(default_factory=list)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence_classification"] = self.evidence_classification.value if isinstance(self.evidence_classification, Enum) else str(self.evidence_classification)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SpecificationGapProposal:
        ev_cls = data.get("evidence_classification", EvidenceClassification.IMPLEMENTATION_CHOICE.value)
        try:
            ev_enum = EvidenceClassification(ev_cls)
        except Exception:
            ev_enum = EvidenceClassification.IMPLEMENTATION_CHOICE
        d = dict(data)
        d["evidence_classification"] = ev_enum
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class HITLDecisionRecord:
    """
    Auditable record of a human engineer's review decision for a specification gap proposal.
    """
    decision_id: str = field(default_factory=lambda: f"DEC_{uuid.uuid4().hex[:8]}")
    gap_id: str = ""
    proposal_id: Optional[str] = None
    action: HITLAction = HITLAction.APPROVE
    reviewer: str = "human_engineer"
    decision_rationale: str = ""
    edited_resolution: Optional[str] = None
    original_proposal: Optional[Dict[str, Any]] = None
    source_evidence_references: List[str] = field(default_factory=list)
    previous_gap_status: str = "UNRESOLVED"
    resulting_gap_status: str = "APPROVED"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action.value if isinstance(self.action, Enum) else str(self.action)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HITLDecisionRecord:
        act_val = data.get("action", HITLAction.APPROVE.value)
        try:
            act_enum = HITLAction(act_val)
        except Exception:
            act_enum = HITLAction.APPROVE
        d = dict(data)
        d["action"] = act_enum
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SpecificationGap:
    """
    A canonical, structured representation of an unresolved specification gap,
    ambiguity, or missing hardware behavior.
    """
    gap_id: str = field(default_factory=lambda: f"GAP_{uuid.uuid4().hex[:8]}")
    component_id: str = ""
    affected_entity: str = ""
    category: SpecificationGapCategory = SpecificationGapCategory.MISSING_BEHAVIOR
    description: str = ""
    severity: GapSeverity = GapSeverity.BLOCKING
    blocks_generation: bool = True
    requires_human_decision: bool = True
    source_requirement_ids: List[str] = field(default_factory=list)
    relevant_entity_ids: List[str] = field(default_factory=list)
    related_signals: List[str] = field(default_factory=list)
    related_registers: List[str] = field(default_factory=list)
    related_fsm_states: List[str] = field(default_factory=list)
    related_interfaces: List[str] = field(default_factory=list)
    source_evidence: str = ""
    document_references: List[Dict[str, Any]] = field(default_factory=list)
    resolution_status: GapResolutionStatus = GapResolutionStatus.UNRESOLVED
    proposals: List[SpecificationGapProposal] = field(default_factory=list)
    selected_proposal_id: Optional[str] = None
    decision: Optional[HITLDecisionRecord] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value if isinstance(self.category, Enum) else str(self.category)
        d["severity"] = self.severity.value if isinstance(self.severity, Enum) else str(self.severity)
        d["resolution_status"] = self.resolution_status.value if isinstance(self.resolution_status, Enum) else str(self.resolution_status)
        d["proposals"] = [p.to_dict() if hasattr(p, "to_dict") else p for p in self.proposals]
        d["decision"] = self.decision.to_dict() if self.decision else None
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SpecificationGap:
        cat_val = data.get("category", SpecificationGapCategory.MISSING_BEHAVIOR.value)
        try:
            cat_enum = SpecificationGapCategory(cat_val)
        except Exception:
            cat_enum = SpecificationGapCategory.MISSING_BEHAVIOR

        sev_val = data.get("severity", GapSeverity.BLOCKING.value)
        try:
            sev_enum = GapSeverity(sev_val)
        except Exception:
            sev_enum = GapSeverity.BLOCKING

        stat_val = data.get("resolution_status", GapResolutionStatus.UNRESOLVED.value)
        try:
            stat_enum = GapResolutionStatus(stat_val)
        except Exception:
            stat_enum = GapResolutionStatus.UNRESOLVED

        proposals = [
            SpecificationGapProposal.from_dict(p) if isinstance(p, dict) else p
            for p in data.get("proposals", [])
        ]
        dec_data = data.get("decision")
        decision = HITLDecisionRecord.from_dict(dec_data) if isinstance(dec_data, dict) else dec_data

        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k not in ("category", "severity", "resolution_status", "proposals", "decision")}
        return cls(
            category=cat_enum,
            severity=sev_enum,
            resolution_status=stat_enum,
            proposals=proposals,
            decision=decision,
            **kwargs
        )

    @classmethod
    def from_unresolved_logic(
        cls,
        unres: Union[PlanUnresolvedLogic, Dict[str, Any]],
        component_id: str = "",
        spec_text: str = ""
    ) -> SpecificationGap:
        """Create a SpecificationGap from an RTL plan PlanUnresolvedLogic entry."""
        u_dict = unres.to_dict() if hasattr(unres, "to_dict") else (unres if isinstance(unres, dict) else {})
        entity = u_dict.get("affected_entity", "")
        desc = u_dict.get("description", "")
        missing = u_dict.get("missing_information", "")
        sev_str = str(u_dict.get("severity", "WARNING")).upper()
        blocks = bool(u_dict.get("blocks_generation", False))
        comp = u_dict.get("affected_component") or component_id

        # Classify gap category
        entity_lower = entity.lower()
        desc_lower = desc.lower()

        if "output port" in desc_lower or "driving" in desc_lower or "missing signal driver" in desc_lower:
            cat = SpecificationGapCategory.MISSING_SIGNAL_DRIVER
            sev = GapSeverity.BLOCKING
            blocks = True
        elif "register" in desc_lower or "update condition" in desc_lower or "load equation" in desc_lower:
            cat = SpecificationGapCategory.UNSPECIFIED_REGISTER_UPDATE
            sev = GapSeverity.WARNING if not blocks else GapSeverity.BLOCKING
        elif "reset" in desc_lower:
            cat = SpecificationGapCategory.UNSPECIFIED_RESET_BEHAVIOR
            sev = GapSeverity.WARNING
        elif "transition" in desc_lower or "fsm" in desc_lower:
            cat = SpecificationGapCategory.UNSPECIFIED_CONTROL_FLOW
            sev = GapSeverity.BLOCKING
            blocks = True
        elif "protocol" in desc_lower or "handshake" in desc_lower:
            cat = SpecificationGapCategory.INCOMPLETE_PROTOCOL_BEHAVIOR
            sev = GapSeverity.WARNING
        elif "conflict" in desc_lower:
            cat = SpecificationGapCategory.CONFLICTING_REQUIREMENTS
            sev = GapSeverity.BLOCKING
            blocks = True
        else:
            cat = SpecificationGapCategory.MISSING_BEHAVIOR
            sev = GapSeverity(sev_str) if sev_str in GapSeverity.__members__ else GapSeverity.BLOCKING

        full_desc = f"{desc} Missing: {missing}" if missing else desc
        stable_id = f"GAP_{abs(hash(f'{comp}_{entity}_{cat.value}')) % 100000:05d}"

        return cls(
            gap_id=stable_id,
            component_id=comp,
            affected_entity=entity,
            category=cat,
            description=full_desc,
            severity=sev,
            blocks_generation=blocks,
            requires_human_decision=bool(u_dict.get("requires_human_decision", blocks)),
            related_signals=[entity] if "port" in desc_lower or "signal" in desc_lower else [],
            related_registers=[entity] if "register" in desc_lower else [],
            source_evidence=spec_text[:400] if spec_text else "",
            resolution_status=GapResolutionStatus.UNRESOLVED
        )

    @classmethod
    def from_efs_conflict(cls, conf: Union[EFSConflict, Dict[str, Any]]) -> SpecificationGap:
        """Create a SpecificationGap from an EFSConflict object."""
        c_dict = conf.to_dict() if hasattr(conf, "to_dict") else (conf if isinstance(conf, dict) else {})
        cid = c_dict.get("conflict_id", "") or c_dict.get("id", "")
        cat_str = str(c_dict.get("category", "GENERAL")).upper()
        desc = c_dict.get("description", "") or c_dict.get("title", "")
        evid = c_dict.get("evidence", "") or desc
        entities = c_dict.get("entities", []) or c_dict.get("affected_entities", [])
        primary_entity = entities[0] if entities else cid

        # Map to category
        c_type = str(c_dict.get("type", "")).upper()
        if "REGISTER" in c_type or "REGISTER" in cat_str:
            cat = SpecificationGapCategory.UNSPECIFIED_REGISTER_UPDATE
        elif "RESET" in c_type or "RESET" in cat_str:
            cat = SpecificationGapCategory.UNSPECIFIED_RESET_BEHAVIOR
        elif "FSM" in c_type or "TRANSITION" in c_type or "FSM" in cat_str:
            cat = SpecificationGapCategory.UNSPECIFIED_CONTROL_FLOW
        elif "PROTOCOL" in c_type or "HANDSHAKE" in c_type:
            cat = SpecificationGapCategory.INCOMPLETE_PROTOCOL_BEHAVIOR
        elif "CONFLICT" in c_type or "DUPLICATE" in c_type:
            cat = SpecificationGapCategory.CONFLICTING_REQUIREMENTS
        elif "SIGNAL" in c_type or "PORT" in c_type:
            cat = SpecificationGapCategory.MISSING_SIGNAL_DRIVER
        else:
            cat = SpecificationGapCategory.MISSING_BEHAVIOR

        blocking = bool(c_dict.get("blocking", True))
        sev = GapSeverity.BLOCKING if blocking else GapSeverity.WARNING
        status_str = str(c_dict.get("resolution_status", c_dict.get("status", "UNRESOLVED"))).upper()
        res_status = GapResolutionStatus.APPROVED if status_str in ("RESOLVED", "APPROVED") else GapResolutionStatus.UNRESOLVED

        return cls(
            gap_id=cid or f"GAP_{abs(hash(f'{primary_entity}_{desc}')) % 100000:05d}",
            component_id="",
            affected_entity=primary_entity,
            category=cat,
            description=desc,
            severity=sev,
            blocks_generation=blocking,
            requires_human_decision=bool(c_dict.get("requires_human_decision", True)),
            relevant_entity_ids=entities,
            source_evidence=evid,
            resolution_status=res_status
        )
