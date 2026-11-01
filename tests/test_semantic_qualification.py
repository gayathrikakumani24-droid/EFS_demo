"""
Generic Regression Tests for Semantic Qualification and Interface Classification.

Tests Cases A through H as specified:
- CASE A: Signal ownership ("counter controller has input clk" -> EFSSignal(clk), NO EFSInterface)
- CASE B: Output signal ("counter controller produces busy" -> EFSSignal(busy), NO EFSInterface)
- CASE C: FSM transition ("IDLE -> COUNTING when start == 1" -> FSM transition, NO EFSInterface)
- CASE D: FSM transition ("COUNTING -> DONE when count == 10" -> FSM transition, NO EFSInterface)
- CASE E: Actual interface ("AXI4 interface connects compute engine to memory" -> EFSInterface)
- CASE F: Component description ("Control Unit : Manages execution for multi-cycle operations" -> component description, NO EFSInterface)
- CASE G: Optional memory (A design with no memory dependency -> NO false "Missing Memory Interface" critical conflict)
- CASE H: Memory design ("Controller reads and writes external memory through AXI4." -> memory/interface extraction)
"""

import pytest
from core.requirement_ir.models import (
    AtomicRequirement, DiscoveredEntity, DocumentBlock, DocumentIR,
    RequirementIR, SourceLocation, new_req_id
)
from core.extraction.requirement_extractor import GenericRequirementExtractor
from core.compiler.efs_compiler import RequirementToEFSCompiler
from core.efs_ir.models import EFSIR, EFSComponent, EFSInterface, EFSSignal, EFSFSM, EFSState, EFSTransition, EFSRegister, EFSRegisterField
from core.efs_ir.validation_rules import (
    is_fsm_state_name,
    is_fsm_transition_or_condition,
    is_physical_signal_name,
    is_hardware_component_candidate,
    is_valid_interface_candidate,
)
from core.verification.conflict_resolution_engine import ConflictResolutionEngine, RegisterRequirementStatus
from core.extraction.source_evidence import SourceEvidenceIndex
from core.extraction.dedicated_extractors import DedicatedRegisterExtractor


def _build_doc_ir(text: str, doc_id: str = "test_doc") -> DocumentIR:
    blocks = [
        DocumentBlock(
            block_id="BLK_001",
            type="paragraph",
            section_id="SEC_1",
            section_title="Architecture",
            page=1,
            text=text
        )
    ]
    return DocumentIR(
        document_id=doc_id,
        source_type="docx",
        metadata={"filename": "test.docx"},
        sections=[{"section_id": "SEC_1", "title": "Architecture", "page_start": 1, "text": text}],
        blocks=blocks,
        tables=[]
    )


class TestSemanticClassificationBoundary:
    """Targeted tests for hard semantic boundary functions."""

    def test_case_a_signal_classification(self):
        """CASE A: Signal ownership input clk."""
        assert is_physical_signal_name("clk") is True
        assert is_physical_signal_name("rst_n") is True
        assert is_physical_signal_name("start") is True
        assert is_hardware_component_candidate("clk") is False

        # clk -> counter controller is NOT a valid interface candidate
        valid, reason = is_valid_interface_candidate("clk", "counter controller")
        assert valid is False
        assert "EFSSignal" in reason

    def test_case_b_output_signal_classification(self):
        """CASE B: Output signal busy, done, count[7:0]."""
        assert is_physical_signal_name("busy") is True
        assert is_physical_signal_name("done") is True
        assert is_physical_signal_name("count[7:0]") is True
        assert is_hardware_component_candidate("busy") is False

        # counter controller -> busy is NOT a valid interface candidate
        valid, reason = is_valid_interface_candidate("counter controller", "busy")
        assert valid is False
        assert "EFSSignal" in reason

    def test_case_c_fsm_transition_classification(self):
        """CASE C: FSM transition IDLE -> COUNTING when start == 1."""
        assert is_fsm_state_name("IDLE") is True
        assert is_fsm_state_name("COUNTING") is True
        assert is_hardware_component_candidate("IDLE") is False
        assert is_hardware_component_candidate("COUNTING") is False
        assert is_fsm_transition_or_condition("start == 1") is True

        valid, reason = is_valid_interface_candidate("IDLE", "COUNTING")
        assert valid is False
        assert "EFSFSM.transitions" in reason

    def test_case_d_fsm_transition_condition(self):
        """CASE D: FSM transition COUNTING -> DONE when count == 10."""
        assert is_fsm_state_name("DONE") is True
        assert is_fsm_transition_or_condition("count == 10") is True

        valid, reason = is_valid_interface_candidate("COUNTING", "DONE")
        assert valid is False
        assert "EFSFSM.transitions" in reason

    def test_case_e_actual_interface(self):
        """CASE E: Actual interface connects compute engine to memory."""
        assert is_hardware_component_candidate("compute engine") is True
        valid, reason = is_valid_interface_candidate("compute engine", "memory")
        assert valid is True
        assert reason == ""

    def test_case_f_component_description(self):
        """CASE F: Component description protection."""
        desc = "Manages execution for multi-cycle operations"
        valid, reason = is_valid_interface_candidate(
            "Control Unit", desc, attributes={"relation": "manages"}
        )
        assert valid is False


class TestPipelineExtractionAndCompiler:
    """Integration tests verifying extraction and EFS IR construction."""

    def test_case_a_and_b_signal_ownership_pipeline(self):
        """CASE A & B: Ensure signal ownership creates EFSSignal and NO EFSInterface."""
        doc_ir = DocumentIR(document_id="doc_signals")
        req_ir = RequirementIR(
            doc_ir=doc_ir,
            document_id="doc_signals",
            entities=[
                DiscoveredEntity(
                    entity_id="ent_ctrl",
                    name="counter controller",
                    type="component",
                    description="Simple synchronous counter controller"
                ),
                DiscoveredEntity(
                    entity_id="ent_clk",
                    name="clk",
                    type="signal",
                    attributes={"direction": "input", "owner": "counter controller", "semantic_role": "clock"}
                ),
                DiscoveredEntity(
                    entity_id="ent_busy",
                    name="busy",
                    type="signal",
                    attributes={"direction": "output", "owner": "counter controller"}
                ),
                # Misclassified candidate from LLM
                DiscoveredEntity(
                    entity_id="ent_rel1",
                    name="clk_counter controller_interface",
                    type="interface",
                    attributes={"source": "clk", "target": "counter controller"}
                ),
                DiscoveredEntity(
                    entity_id="ent_rel2",
                    name="counter controller_busy_interface",
                    type="interface",
                    attributes={"source": "counter controller", "target": "busy"}
                ),
            ],
            requirements=[],
            edges=[]
        )

        compiler = RequirementToEFSCompiler(req_ir)
        efs_ir = compiler.compile()

        # Interfaces must NOT include signal ownership
        iface_names = [i.name for i in efs_ir.interfaces]
        assert "clk_counter controller_interface" not in iface_names
        assert "counter controller_busy_interface" not in iface_names
        assert "clk" not in iface_names
        assert "busy" not in iface_names

        # Signals must be preserved
        sig_names = [s.name for s in efs_ir.signals]
        assert "clk" in sig_names
        assert "busy" in sig_names

    def test_case_c_and_d_fsm_transition_pipeline(self):
        """CASE C & D: Ensure FSM transitions do not create interfaces or components."""
        doc_ir = DocumentIR(document_id="doc_fsm")
        req_ir = RequirementIR(
            doc_ir=doc_ir,
            document_id="doc_fsm",
            entities=[
                # Misclassified FSM states as components
                DiscoveredEntity(entity_id="ent_idle", name="IDLE", type="component"),
                DiscoveredEntity(entity_id="ent_cnt", name="COUNTING", type="component"),
                DiscoveredEntity(entity_id="ent_done", name="DONE", type="component"),
                # Misclassified transition interfaces
                DiscoveredEntity(
                    entity_id="ent_if1",
                    name="IDLE_COUNTING_interface",
                    type="interface",
                    attributes={"source": "IDLE", "target": "COUNTING", "signals": ["start == 1"]}
                ),
                DiscoveredEntity(
                    entity_id="ent_if2",
                    name="COUNTING_DONE_interface",
                    type="interface",
                    attributes={"source": "COUNTING", "target": "DONE", "signals": ["count == 10"]}
                ),
                DiscoveredEntity(
                    entity_id="ent_if3",
                    name="DONE_IDLE_interface",
                    type="interface",
                    attributes={"source": "DONE", "target": "IDLE", "signals": ["next clock cycle"]}
                ),
            ],
            requirements=[],
            edges=[]
        )

        compiler = RequirementToEFSCompiler(req_ir)
        efs_ir = compiler.compile()

        # FSM states must NOT be in components
        comp_names = [c.name for c in efs_ir.components]
        assert "IDLE" not in comp_names
        assert "COUNTING" not in comp_names
        assert "DONE" not in comp_names

        # Transition interfaces must NOT exist
        iface_names = [i.name for i in efs_ir.interfaces]
        assert "IDLE_COUNTING_interface" not in iface_names
        assert "COUNTING_DONE_interface" not in iface_names
        assert "DONE_IDLE_interface" not in iface_names

    def test_case_g_optional_memory_no_conflict(self):
        """CASE G: Design with no memory dependency has NO 'Missing Memory Interface' conflict."""
        spec_text = """
        # Counter Controller Spec
        A simple synchronous counter controller with clk, rst_n, start, busy, done, count.
        No memory interfaces or external memory required.
        """
        doc_ir = _build_doc_ir(spec_text)
        evidence_index = SourceEvidenceIndex(doc_ir)

        efs_ir = EFSIR(
            metadata=None,
            components=[EFSComponent(component_id="comp_1", name="counter controller", type="controller")],
            signals=[
                EFSSignal(signal_id="sig_1", name="clk", direction="input", semantic_role="clock"),
                EFSSignal(signal_id="sig_2", name="rst_n", direction="input", semantic_role="reset"),
            ],
            interfaces=[],
            memory_regions=[],
            fsms=[
                EFSFSM(
                    fsm_id="fsm_1",
                    name="Counter_FSM",
                    states=[EFSState(state_id="st_1", name="IDLE"), EFSState(state_id="st_2", name="COUNTING")],
                    transitions=[EFSTransition(transition_id="tr_1", source_state="IDLE", target_state="COUNTING", condition="start == 1")]
                )
            ]
        )

        conflicts = ConflictResolutionEngine.validate_hardware_completeness(
            efs_ir, doc_ir=doc_ir, evidence_index=evidence_index
        )
        mem_conflicts = [c for c in conflicts if "Memory" in (c.title or "")]
        assert len(mem_conflicts) == 0, f"Expected 0 memory conflicts, got: {[c.title for c in mem_conflicts]}"

    def test_case_h_memory_design_flags_when_missing(self):
        """CASE H: Design specifying memory requirement triggers memory recovery or validation."""
        spec_text = """
        # Controller with Memory Spec
        Controller reads and writes external memory through AXI4 interface.
        Data memory buffer holds incoming packets.
        """
        doc_ir = _build_doc_ir(spec_text)
        evidence_index = SourceEvidenceIndex(doc_ir)

        efs_ir = EFSIR(
            metadata=None,
            components=[EFSComponent(component_id="comp_1", name="Controller", type="controller")],
            signals=[
                EFSSignal(signal_id="sig_1", name="clk", direction="input", semantic_role="clock"),
            ],
            interfaces=[],
            memory_regions=[]  # Missing memory initially!
        )

        conflicts = ConflictResolutionEngine.validate_hardware_completeness(
            efs_ir, doc_ir=doc_ir, evidence_index=evidence_index
        )
        mem_conflicts = [c for c in conflicts if "Memory" in (c.title or "")]
        # Either memory was recovered from source or completeness conflict was flagged
        assert len(efs_ir.memory_regions) > 0 or len(mem_conflicts) > 0, "Expected memory extraction or conflict."


class TestInterModuleRelationshipCompleteness:
    """
    Tests for inter-module relationship validation semantics:
    - CASE A: Single controller ("Counter Controller has clk, rst_n, start, busy, done and count.")
      Expected: NO Missing Inter-Module Relationships conflict.
    - CASE B: Internal FSM ("IDLE transitions to COUNTING when start is asserted.")
      Expected: FSM transition. NO inter-module relationship conflict.
    - CASE C: Actual modules ("Compute Engine communicates with DMA Engine through AXI4.")
      Expected: Compute Engine and DMA Engine are components. AXI4 relationship/interface is present.
    - CASE D: Missing actual relationship ("Compute Engine communicates with DMA Engine through AXI4.")
      IR: Compute Engine, DMA Engine, NO interface
      Expected: Missing Inter-Module Relationships conflict.
    - CASE E: Signals are not modules ("Controller receives start and produces done.")
      Expected: Signals. NO inter-module relationship requirement.
    """

    def test_case_a_single_controller_no_rel_conflict(self):
        """CASE A: Single controller with multiple signals has NO 'Missing Inter-Module Relationships' conflict."""
        spec_text = "Counter Controller has clk, rst_n, start, busy, done and count."
        doc_ir = _build_doc_ir(spec_text)
        evidence_index = SourceEvidenceIndex(doc_ir)

        efs_ir = EFSIR(
            metadata=None,
            components=[
                EFSComponent(component_id="comp_1", name="Counter Controller", type="controller"),
                # Even if architecture planner created internal slices
                EFSComponent(component_id="comp_2", name="Counter_Datapath", type="module"),
                EFSComponent(component_id="comp_3", name="Control_FSM", type="fsm"),
            ],
            signals=[
                EFSSignal(signal_id="sig_1", name="clk", direction="input", semantic_role="clock"),
                EFSSignal(signal_id="sig_2", name="rst_n", direction="input", semantic_role="reset"),
                EFSSignal(signal_id="sig_3", name="start", direction="input"),
                EFSSignal(signal_id="sig_4", name="busy", direction="output"),
                EFSSignal(signal_id="sig_5", name="done", direction="output"),
                EFSSignal(signal_id="sig_6", name="count", direction="output", width="8"),
            ],
            interfaces=[]
        )

        conflicts = ConflictResolutionEngine.validate_hardware_completeness(
            efs_ir, doc_ir=doc_ir, evidence_index=evidence_index
        )
        rel_conflicts = [c for c in conflicts if c.title == "Missing Inter-Module Relationships"]
        assert len(rel_conflicts) == 0, f"Expected 0 relationship conflicts, got: {[c.title for c in rel_conflicts]}"

    def test_case_b_internal_fsm_no_rel_conflict(self):
        """CASE B: Internal FSM states are not treated as hardware components or inter-module relationships."""
        spec_text = "IDLE transitions to COUNTING when start is asserted. COUNTING transitions to DONE when count is 10."
        doc_ir = _build_doc_ir(spec_text)
        evidence_index = SourceEvidenceIndex(doc_ir)

        efs_ir = EFSIR(
            metadata=None,
            components=[
                EFSComponent(component_id="comp_1", name="Counter Controller", type="controller"),
            ],
            signals=[
                EFSSignal(signal_id="sig_1", name="clk", direction="input", semantic_role="clock"),
                EFSSignal(signal_id="sig_2", name="rst_n", direction="input", semantic_role="reset"),
            ],
            fsms=[
                EFSFSM(
                    fsm_id="fsm_1",
                    name="Controller_FSM",
                    states=[
                        EFSState(state_id="s1", name="IDLE"),
                        EFSState(state_id="s2", name="COUNTING"),
                        EFSState(state_id="s3", name="DONE")
                    ],
                    transitions=[
                        EFSTransition(transition_id="t1", source_state="IDLE", target_state="COUNTING", condition="start == 1"),
                        EFSTransition(transition_id="t2", source_state="COUNTING", target_state="DONE", condition="count == 10")
                    ]
                )
            ],
            interfaces=[]
        )

        conflicts = ConflictResolutionEngine.validate_hardware_completeness(
            efs_ir, doc_ir=doc_ir, evidence_index=evidence_index
        )
        rel_conflicts = [c for c in conflicts if c.title == "Missing Inter-Module Relationships"]
        assert len(rel_conflicts) == 0

    def test_case_c_actual_modules_with_relationship(self):
        """CASE C: Actual communicating modules with extracted AXI4 interface raises NO conflict."""
        spec_text = "Compute Engine communicates with DMA Engine through AXI4."
        doc_ir = _build_doc_ir(spec_text)
        evidence_index = SourceEvidenceIndex(doc_ir)

        efs_ir = EFSIR(
            metadata=None,
            components=[
                EFSComponent(component_id="comp_ce", name="Compute Engine", type="module"),
                EFSComponent(component_id="comp_dma", name="DMA Engine", type="module"),
            ],
            signals=[
                EFSSignal(signal_id="sig_1", name="clk", direction="input", semantic_role="clock"),
                EFSSignal(signal_id="sig_2", name="rst_n", direction="input", semantic_role="reset"),
            ],
            interfaces=[
                EFSInterface(
                    interface_id="if_axi4",
                    name="AXI4_Compute_DMA",
                    protocol="AXI4",
                    source_component="Compute Engine",
                    destination_component="DMA Engine"
                )
            ]
        )

        conflicts = ConflictResolutionEngine.validate_hardware_completeness(
            efs_ir, doc_ir=doc_ir, evidence_index=evidence_index
        )
        rel_conflicts = [c for c in conflicts if c.title == "Missing Inter-Module Relationships"]
        assert len(rel_conflicts) == 0

    def test_case_d_missing_actual_relationship_flags_conflict(self):
        """CASE D: Actual communicating modules described in source missing interface DOES flag conflict."""
        spec_text = "Compute Engine communicates with DMA Engine through AXI4."
        doc_ir = _build_doc_ir(spec_text)
        evidence_index = SourceEvidenceIndex(doc_ir)

        efs_ir = EFSIR(
            metadata=None,
            components=[
                EFSComponent(component_id="comp_ce", name="Compute Engine", type="module"),
                EFSComponent(component_id="comp_dma", name="DMA Engine", type="module"),
            ],
            signals=[
                EFSSignal(signal_id="sig_1", name="clk", direction="input", semantic_role="clock"),
                EFSSignal(signal_id="sig_2", name="rst_n", direction="input", semantic_role="reset"),
            ],
            interfaces=[]  # MISSING interface!
        )

        from unittest.mock import patch
        with patch("core.extraction.source_recovery.SourceBackedRecovery.recover_efs_ir", return_value=None):
            conflicts = ConflictResolutionEngine.validate_hardware_completeness(
                efs_ir, doc_ir=doc_ir, evidence_index=evidence_index
            )
        rel_conflicts = [c for c in conflicts if c.title == "Missing Inter-Module Relationships"]
        assert len(rel_conflicts) == 1
        assert "Module_Relationships" in rel_conflicts[0].entities

    def test_case_e_signals_are_not_modules_no_rel_conflict(self):
        """CASE E: Controller receiving start and producing done involves signals, not inter-module relationships."""
        spec_text = "Controller receives start and produces done."
        doc_ir = _build_doc_ir(spec_text)
        evidence_index = SourceEvidenceIndex(doc_ir)

        efs_ir = EFSIR(
            metadata=None,
            components=[
                EFSComponent(component_id="comp_1", name="Controller", type="controller"),
            ],
            signals=[
                EFSSignal(signal_id="sig_1", name="clk", direction="input", semantic_role="clock"),
                EFSSignal(signal_id="sig_2", name="rst_n", direction="input", semantic_role="reset"),
                EFSSignal(signal_id="sig_3", name="start", direction="input"),
                EFSSignal(signal_id="sig_4", name="done", direction="output"),
            ],
            interfaces=[]
        )

        conflicts = ConflictResolutionEngine.validate_hardware_completeness(
            efs_ir, doc_ir=doc_ir, evidence_index=evidence_index
        )
        rel_conflicts = [c for c in conflicts if c.title == "Missing Inter-Module Relationships"]
        assert len(rel_conflicts) == 0


class TestRegisterSourceEvidenceModel:
    """
    Regression test suite for the 4-State Register Source-Evidence Model:
    1. Counter Controller: 0 registers in source -> NOT_APPLICABLE -> 0 conflicts.
    2. Real Register Specification: CTRL & STATUS with fields -> REQUIRED_AND_PRESENT -> 0 conflicts.
    3. Missing Register Specification: Source requires CTRL & STATUS, IR has none -> REQUIRED_BUT_MISSING -> 1 conflict.
    """

    def test_counter_controller_register_not_applicable(self):
        """Counter Controller has no registers in source; status must be NOT_APPLICABLE and 0 conflicts generated."""
        spec_text = (
            "Design a simple synchronous counter controller with clk, rst_n, start, busy, done, count[7:0].\n"
            "clk is an input clock signal. rst_n is an active-low reset signal.\n"
            "start is an input signal that triggers counting.\n"
            "busy is an output signal indicating counting or done.\n"
            "done is an output signal asserted when count reaches target.\n"
            "count[7:0] is an 8-bit counter output.\n"
            "The controller contains an FSM with IDLE, COUNTING, and DONE states.\n"
            "IDLE transitions to COUNTING when start == 1.\n"
            "COUNTING transitions to DONE when count == 10.\n"
            "After one clock cycle in DONE, return to IDLE."
        )
        doc_ir = _build_doc_ir(spec_text)
        evidence_index = SourceEvidenceIndex(doc_ir)

        comp = EFSComponent(component_id="comp_counter", name="Counter_Controller", type="controller")
        sig_clk = EFSSignal(signal_id="sig_clk", name="clk", direction="input", width="1", semantic_role="clock", owner="comp_counter")
        sig_rst = EFSSignal(signal_id="sig_rst", name="rst_n", direction="input", width="1", semantic_role="reset", owner="comp_counter")
        sig_start = EFSSignal(signal_id="sig_start", name="start", direction="input", width="1", semantic_role="control", owner="comp_counter")
        sig_busy = EFSSignal(signal_id="sig_busy", name="busy", direction="output", width="1", semantic_role="status", owner="comp_counter")
        sig_done = EFSSignal(signal_id="sig_done", name="done", direction="output", width="1", semantic_role="status", owner="comp_counter")
        sig_count = EFSSignal(signal_id="sig_count", name="count", direction="output", width="8", semantic_role="data", owner="comp_counter")

        fsm = EFSFSM(
            fsm_id="fsm_counter",
            name="Counter_Controller_FSM",
            states=[
                EFSState(state_id="st_idle", name="IDLE"),
                EFSState(state_id="st_counting", name="COUNTING"),
                EFSState(state_id="st_done", name="DONE")
            ],
            transitions=[
                EFSTransition(transition_id="tr_1", source_state="IDLE", target_state="COUNTING", condition="start == 1"),
                EFSTransition(transition_id="tr_2", source_state="COUNTING", target_state="DONE", condition="count == 10"),
                EFSTransition(transition_id="tr_3", source_state="DONE", target_state="IDLE", condition="next clock cycle")
            ],
            initial_state="IDLE"
        )

        efs_ir = EFSIR(
            components=[comp],
            signals=[sig_clk, sig_rst, sig_start, sig_busy, sig_done, sig_count],
            clocks=[sig_clk],
            resets=[sig_rst],
            fsms=[fsm],
            interfaces=[],
            registers=[]
        )

        # Classify register requirement
        reg_status, reg_ev, reg_reason = ConflictResolutionEngine.classify_register_requirement(
            efs_ir, doc_ir=doc_ir, evidence_index=evidence_index
        )
        assert reg_status == RegisterRequirementStatus.NOT_APPLICABLE
        assert len(efs_ir.registers) == 0

        # Validate completeness
        conflicts = ConflictResolutionEngine.validate_hardware_completeness(
            efs_ir, doc_ir=doc_ir, evidence_index=evidence_index
        )
        reg_conflicts = [c for c in conflicts if "Register" in (c.title or "") or c.property == "registers"]
        mem_conflicts = [c for c in conflicts if "Memory" in (c.title or "")]
        rel_conflicts = [c for c in conflicts if "Inter-Module" in (c.title or "")]

        assert len(reg_conflicts) == 0, f"Expected 0 register conflicts, got: {[c.title for c in reg_conflicts]}"
        assert len(mem_conflicts) == 0, f"Expected 0 memory conflicts, got: {[c.title for c in mem_conflicts]}"
        assert len(rel_conflicts) == 0, f"Expected 0 relationship conflicts, got: {[c.title for c in rel_conflicts]}"
        assert len(conflicts) == 0, f"Expected 0 conflicts, got: {[c.title for c in conflicts]}"
        assert len(efs_ir.interfaces) == 0
        assert len(efs_ir.signals) == 6
        assert len(efs_ir.fsms[0].transitions) == 3

    def test_real_register_regression(self):
        """Source explicitly describes CTRL and STATUS registers with fields; extraction succeeds and 0 conflicts raised."""
        spec_text = (
            "The hardware exposes configuration registers CTRL and STATUS.\n"
            "CTRL contains an enable field.\n"
            "STATUS contains a done field.\n"
            "CTRL is written by software and STATUS is read by software."
        )
        doc_ir = _build_doc_ir(spec_text)
        evidence_index = SourceEvidenceIndex(doc_ir)

        # Verify DedicatedRegisterExtractor extracts CTRL and STATUS with fields
        extracted = DedicatedRegisterExtractor.extract_registers(doc_ir, evidence_index)
        assert len(extracted) == 2, f"Expected 2 registers, got {[r.name for r in extracted]}"
        ctrl_reg = next((r for r in extracted if r.name == "CTRL"), None)
        status_reg = next((r for r in extracted if r.name == "STATUS"), None)
        assert ctrl_reg is not None
        assert status_reg is not None
        assert any(f.get("name") == "enable" for f in ctrl_reg.fields)
        assert any(f.get("name") == "done" for f in status_reg.fields)
        assert ctrl_reg.access == "WO"
        assert status_reg.access == "RO"

        # Build EFS IR with the extracted registers
        comp = EFSComponent(name="Hardware_Module", type="module")
        sig_clk = EFSSignal(name="clk", direction="input", semantic_role="clock")
        sig_rst = EFSSignal(name="rst_n", direction="input", semantic_role="reset")
        efs_ir = EFSIR(
            components=[comp],
            signals=[sig_clk, sig_rst],
            clocks=[sig_clk],
            resets=[sig_rst],
            registers=[
                EFSRegister(name="CTRL", access="WO", fields=[EFSRegisterField(name="enable", width=1, access="WO")]),
                EFSRegister(name="STATUS", access="RO", fields=[EFSRegisterField(name="done", width=1, access="RO")]),
            ]
        )

        reg_status, reg_ev, reg_reason = ConflictResolutionEngine.classify_register_requirement(
            efs_ir, doc_ir=doc_ir, evidence_index=evidence_index
        )
        assert reg_status in (RegisterRequirementStatus.REQUIRED_AND_PRESENT, "REQUIRED")

        conflicts = ConflictResolutionEngine.validate_hardware_completeness(
            efs_ir, doc_ir=doc_ir, evidence_index=evidence_index
        )
        reg_conflicts = [c for c in conflicts if "Register" in (c.title or "") or c.property == "registers"]
        assert len(reg_conflicts) == 0, f"Expected 0 register conflicts, got: {[c.title for c in reg_conflicts]}"

    def test_missing_register_regression(self):
        """Source explicitly describes CTRL and STATUS registers, but EFS IR has none; raises Missing Register Definitions conflict."""
        spec_text = "The module contains a CTRL configuration register and STATUS register."
        doc_ir = _build_doc_ir(spec_text)
        evidence_index = SourceEvidenceIndex(doc_ir)

        comp = EFSComponent(name="Hardware_Module", type="module")
        sig_clk = EFSSignal(name="clk", direction="input", semantic_role="clock")
        sig_rst = EFSSignal(name="rst_n", direction="input", semantic_role="reset")
        efs_ir = EFSIR(
            components=[comp],
            signals=[sig_clk, sig_rst],
            clocks=[sig_clk],
            resets=[sig_rst],
            registers=[]  # Intentionally missing!
        )

        reg_status, reg_ev, reg_reason = ConflictResolutionEngine.classify_register_requirement(
            efs_ir, doc_ir=doc_ir, evidence_index=evidence_index
        )
        assert reg_status == RegisterRequirementStatus.REQUIRED_BUT_MISSING

        # When running completeness validation without doc_ir (simulating unrecovered extraction gap)
        conflicts = ConflictResolutionEngine.validate_hardware_completeness(
            efs_ir, evidence_index=evidence_index
        )
        reg_conflicts = [c for c in conflicts if "Register" in (c.title or "") or c.property == "registers"]
        assert len(reg_conflicts) == 1, f"Expected 1 register conflict, got: {[c.title for c in reg_conflicts]}"
        conflict = reg_conflicts[0]
        assert conflict.title == "Missing Register Definitions"
        assert conflict.property == "registers"
        assert conflict.severity == "CRITICAL"
        assert conflict.blocking is True
        assert "CTRL configuration register" in conflict.evidence or spec_text in conflict.evidence
        assert len(efs_ir.registers) == 0  # No fabricated registers!


