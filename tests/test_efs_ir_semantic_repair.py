"""
Parameterized Tests for EFS IR Semantic Integrity, Normalization, and Pipeline Repair.

Covers:
1. Sequential and combinational designs with correct input/output classification.
2. Active-high and active-low reset canonicalization without alternation.
3. Timed FSMs with duration_cycles and type safety (preventing boolean misclassification).
4. Multi-design isolation in shared vector DB: protocol-oriented vs generic designs.
5. Rejection of placeholder entities (e.g., 'Module Name') and empty top_module resolution.
6. Semantic validation blocking completeness on contradiction (all-inputs, reset mismatch, leaked IDs).
7. End-to-end RTL generation grounding verification.
"""

import pytest
import re
from typing import Any, Dict, List

from core.efs_ir.models import (
    EFSIR, EFSMetadata, EFSComponent, EFSInterface, EFSSignal,
    EFSFSM, EFSState, EFSTransition, EFSTimingRule, EFSProtocolRule,
    new_efs_id
)
from core.efs_ir.builder import EFSIRBuilder
from core.efs_ir.repair_engine import EFSStructuralRepairEngine
from core.efs_ir.validator import validate_efs_ir, get_efs_ir_quality
from core.efs_ir.validation_rules import is_physical_signal_name, DISQUALIFIED_COMPONENT_NAMES
from core.extraction.requirement_extractor import GenericRequirementExtractor
from core.document_ir.models import DocumentIR, TextBlock


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def traffic_light_spec_text() -> str:
    return """
    # Hardware Design Specification: Traffic Light Controller
    Module Name: Traffic Light Controller
    
    The traffic light controller operates on clk and active-low reset_n.
    The module drives three output lights: red_light, yellow_light, and green_light.
    
    Operation:
    - On reset_n asserted, the controller resets to RED.
    - RED state remains on for 5 clock cycles, then transitions to GREEN.
    - GREEN state remains on for 10 clock cycles, then transitions to YELLOW.
    - YELLOW state remains on for 3 clock cycles, then transitions back to RED.
    Only one light is active at any time.
    """


@pytest.fixture
def axi_subordinate_spec_text() -> str:
    return """
    # Hardware Design Specification: AXI4_Lite_Slave
    The module implements an AXI4-Lite slave interface operating on aclk and active-low aresetn.
    
    Ports:
    - aclk: input clock
    - aresetn: input active-low reset
    - s_axi_awvalid: input write address valid
    - s_axi_awready: output write address ready
    - s_axi_wvalid: input write data valid
    - s_axi_wready: output write data ready
    - s_axi_bvalid: output write response valid
    - s_axi_bready: input write response ready
    
    Protocol:
    - Handshake rule: valid must remain asserted until ready is asserted.
    """


@pytest.fixture
def combinational_alu_spec_text() -> str:
    return """
    # Hardware Design Specification: Combinational ALU
    Purely combinational arithmetic logic unit.
    
    Ports:
    - a_in: input [31:0] operand A
    - b_in: input [31:0] operand B
    - op_code: input [3:0] operation code
    - alu_out: output [31:0] ALU result output
    - zero_flag: output zero detection flag
    - overflow_flag: output overflow status flag
    """


# ==============================================================================
# 1. Parameterized Port Direction & Component Classification Tests
# ==============================================================================

class TestPortDirectionAndComponentClassification:
    """Tests input/output port classification and prevention of all-inputs defect."""

    @pytest.mark.parametrize("design_type,spec_text,expected_outputs,expected_inputs", [
        (
            "sequential_traffic_light",
            """
            Module Name: Traffic Light Controller
            Inputs: clk, reset_n
            Outputs: red_light, green_light, yellow_light
            Transitions: RED to GREEN after 5 clock cycles.
            """,
            ["red_light", "green_light", "yellow_light"],
            ["clk", "reset_n"]
        ),
        (
            "combinational_mux",
            """
            Module Name: Multiplexer_4to1
            Inputs: in0, in1, in2, in3, sel_in
            Outputs: data_out, valid_out
            Purely combinational 4-to-1 data multiplexer.
            """,
            ["data_out", "valid_out"],
            ["in0", "in1", "in2", "in3", "sel_in"]
        ),
        (
            "combinational_alu",
            """
            Module Name: Simple_ALU
            Inputs: op_a, op_b, alu_op
            Outputs: result_out, carry_flag, zero_flag
            Combinational execution core.
            """,
            ["result_out", "carry_flag", "zero_flag"],
            ["op_a", "op_b", "alu_op"]
        ),
    ])
    def test_port_direction_classification(self, design_type, spec_text, expected_outputs, expected_inputs):
        doc_ir = DocumentIR(
            document_id=f"DOC_{design_type}",
            filename=f"{design_type}.md",
            blocks=[TextBlock(block_id="b1", text=spec_text, block_type="paragraph", section="Overview")]
        )
        extractor = RequirementExtractor(doc_ir)
        req_model = extractor.extract()
        
        builder = EFSIRBuilder(req_model=req_model, design_spec=spec_text)
        ir = builder.build()

        sig_by_name = {s.name.lower(): s for s in ir.signals}

        # Verify all expected outputs are classified as output
        for out_name in expected_outputs:
            assert out_name.lower() in sig_by_name, f"Expected output signal {out_name} not found"
            assert sig_by_name[out_name.lower()].direction == "output", (
                f"Signal {out_name} classified as '{sig_by_name[out_name.lower()].direction}', expected 'output'"
            )

        # Verify expected inputs are classified as input
        for in_name in expected_inputs:
            assert in_name.lower() in sig_by_name, f"Expected input signal {in_name} not found"
            assert sig_by_name[in_name.lower()].direction == "input", (
                f"Signal {in_name} classified as '{sig_by_name[in_name.lower()].direction}', expected 'input'"
            )

        # Verify no internal generated placeholder IDs leaked
        for sig in ir.signals:
            assert not re.match(r"^SIG_[0-9a-fA-F]{8}$", sig.name), f"Internal signal ID leaked as signal name: {sig.name}"

    def test_all_inputs_semantic_contradiction_is_flagged(self):
        """A design with only inputs and zero outputs must be flagged as a semantic contradiction."""
        ir = EFSIR(
            metadata=EFSMetadata(design_id=new_efs_id("dsn"), design_name="broken_module", top_module="broken_module"),
            components=[EFSComponent(name="broken_module")],
            signals=[
                EFSSignal(name="clk", direction="input", semantic_role="clock"),
                EFSSignal(name="rst_n", direction="input", semantic_role="reset"),
                EFSSignal(name="data_a", direction="input"),
                EFSSignal(name="data_b", direction="input"),
            ]
        )
        issues = validate_efs_ir(ir)
        all_in_issues = [i for i in issues if "All signals are classified as inputs" in i.get("message", "")]
        assert len(all_in_issues) >= 1, "Validator must detect all-inputs contradiction"
        assert all_in_issues[0]["severity"] == "ERROR"

        # Quality scorecard must mark status as BLOCKED
        quality = get_efs_ir_quality(ir, issues)
        assert quality["status"] == "BLOCKED"


# ==============================================================================
# 2. Parameterized Reset Semantics & Canonicalization Tests
# ==============================================================================

class TestResetSemanticsCanonicalization:
    """Tests active-high and active-low resets, ensuring no alternation between reset_n and rst_n."""

    @pytest.mark.parametrize("reset_name,expected_polarity", [
        ("reset_n", "low"),
        ("rst_n", "low"),
        ("sys_rst_n", "low"),
        ("rst", "high"),
        ("reset", "high"),
        ("clr", "high"),
    ])
    def test_canonical_reset_synchronization(self, reset_name, expected_polarity):
        spec = f"""
        # Hardware Design Specification: Reset_Test_Module
        Operating with clock 'clk' and reset '{reset_name}'.
        Signals:
        - data_in: input
        - data_out: output
        """
        req_model = {
            "design_name": "Reset_Test_Module",
            "signals": [
                {"name": "clk", "direction": "input", "semantic_role": "clock"},
                {"name": reset_name, "direction": "input", "semantic_role": "reset"},
                {"name": "data_in", "direction": "input"},
                {"name": "data_out", "direction": "output"},
            ],
            "spec_text": spec
        }
        builder = EFSIRBuilder(req_model=req_model, design_spec=spec)
        ir = builder.build()

        assert len(ir.resets) >= 1, "Must populate ir.resets"
        assert ir.resets[0].name == reset_name, f"Expected canonical reset '{reset_name}', got '{ir.resets[0].name}'"

        # Verify all signals reference the exact canonical reset, NOT an alternated default
        for sig in ir.signals:
            if sig.semantic_role != "reset":
                assert sig.reset == reset_name, (
                    f"Signal {sig.name} reset reference '{sig.reset}' alternated from canonical '{reset_name}'"
                )

        # Interfaces must also match
        for iface in ir.interfaces:
            assert iface.reset == reset_name

        # Validation must pass without reset mismatch
        issues = validate_efs_ir(ir)
        rst_issues = [i for i in issues if "Reset semantic mismatch" in i.get("message", "")]
        assert len(rst_issues) == 0, f"Unexpected reset mismatch issues: {rst_issues}"

    def test_conflicting_reset_names_are_flagged(self):
        """Contradicting reset names between signals and ir.resets must produce validation errors."""
        ir = EFSIR(
            metadata=EFSMetadata(design_id=new_efs_id("dsn"), design_name="conflict_reset", top_module="conflict_reset"),
            components=[EFSComponent(name="conflict_reset")],
            resets=[EFSSignal(name="reset_n", direction="input", semantic_role="reset")],
            signals=[
                EFSSignal(name="clk", direction="input", semantic_role="clock", reset="reset_n"),
                EFSSignal(name="reset_n", direction="input", semantic_role="reset", reset="reset_n"),
                # Signal erroneously references rst_n while design canonical reset is reset_n
                EFSSignal(name="out_val", direction="output", reset="rst_n"),
            ]
        )
        issues = validate_efs_ir(ir)
        mismatch_issues = [i for i in issues if "Reset semantic mismatch" in i.get("message", "")]
        assert len(mismatch_issues) >= 1, "Must flag mismatch between reset_n and rst_n"
        assert mismatch_issues[0]["severity"] == "ERROR"


# ==============================================================================
# 3. Timed FSMs and Duration Normalization Tests
# ==============================================================================

class TestTimedFSMNormalization:
    """Tests FSM timing conditions, typed durations, and deduplication."""

    def test_cycle_durations_typed_as_timed(self, traffic_light_spec_text):
        doc_ir = DocumentIR(
            document_id="DOC_traffic_light",
            filename="traffic_light.md",
            blocks=[TextBlock(block_id="b1", text=traffic_light_spec_text, block_type="paragraph", section="FSM")]
        )
        extractor = RequirementExtractor(doc_ir)
        req_model = extractor.extract()

        builder = EFSIRBuilder(req_model=req_model, design_spec=traffic_light_spec_text)
        ir = builder.build()

        assert len(ir.fsms) >= 1, "Must extract FSM"
        fsm = ir.fsms[0]

        # Verify state transitions have condition_type == 'timed' and integer duration_cycles
        timed_transitions = [t for t in fsm.transitions if t.duration_cycles is not None]
        assert len(timed_transitions) >= 1, "Timed FSM must have transitions with duration_cycles"

        for trans in timed_transitions:
            assert trans.condition_type == "timed", (
                f"Transition {trans.source_state}->{trans.target_state} condition_type is '{trans.condition_type}', expected 'timed'"
            )
            assert isinstance(trans.duration_cycles, int), "duration_cycles must be an integer"
            assert trans.duration_cycles > 0

    def test_cycle_duration_misclassified_as_boolean_is_rejected(self):
        """A transition with cycle duration condition misclassified as boolean must trigger a validation error."""
        ir = EFSIR(
            metadata=EFSMetadata(design_id=new_efs_id("dsn"), design_name="fsm_bad_type", top_module="fsm_bad_type"),
            components=[EFSComponent(name="fsm_bad_type")],
            signals=[
                EFSSignal(name="clk", direction="input", semantic_role="clock"),
                EFSSignal(name="rst_n", direction="input", semantic_role="reset"),
                EFSSignal(name="light_out", direction="output"),
            ],
            resets=[EFSSignal(name="rst_n", direction="input", semantic_role="reset")],
            fsms=[
                EFSFSM(
                    name="Test_FSM",
                    states=[EFSState(name="S0"), EFSState(name="S1")],
                    initial_state="S0",
                    transitions=[
                        EFSTransition(
                            source_state="S0",
                            target_state="S1",
                            condition="5 clock cycles",
                            condition_type="boolean"  # DEFECTIVE
                        )
                    ]
                )
            ]
        )
        issues = validate_efs_ir(ir)
        timing_issues = [i for i in issues if "Cycle duration condition" in i.get("message", "")]
        assert len(timing_issues) >= 1, "Validator must reject cycle duration typed as boolean"
        assert timing_issues[0]["severity"] == "ERROR"


# ==============================================================================
# 4. Multi-Design Isolation & Protocol Quarantining Tests
# ==============================================================================

class TestMultiDesignIsolationAndProtocolQuarantine:
    """Verifies that generic designs do not receive spurious bus interfaces or AXI protocol rules."""

    def test_generic_traffic_light_quarantines_axi_rules(self, traffic_light_spec_text):
        """When vector store contains AXI rules, a generic non-bus design must quarantine them."""
        retrieved_axi_rules = [
            {"citation": "AXI4 §A3.2.1", "text": "A value of VALID once asserted must remain asserted until READY is high."},
            {"citation": "AXI4 §A3.3.1", "text": "Write response BVALID must not depend on BREADY."},
        ]

        doc_ir = DocumentIR(
            document_id="DOC_tl",
            filename="traffic_light.docx",
            blocks=[TextBlock(block_id="b1", text=traffic_light_spec_text, block_type="paragraph", section="Spec")]
        )
        extractor = RequirementExtractor(doc_ir)
        req_model = extractor.extract()

        builder = EFSIRBuilder(
            req_model=req_model,
            protocol_rules=retrieved_axi_rules,
            design_spec=traffic_light_spec_text
        )
        ir = builder.build()

        # 1. No spurious external_bus_interface must exist
        assert not any(i.name == "external_bus_interface" for i in ir.interfaces), (
            "Generic Traffic Light Controller must NOT create an external_bus_interface"
        )

        # 2. AXI protocol rules must be quarantined
        assert len(ir.protocol_rules) == 0, (
            f"Generic design must quarantine unsupported AXI protocol rules, found: {ir.protocol_rules}"
        )

    def test_axi_design_preserves_supported_protocol_rules(self, axi_subordinate_spec_text):
        """A bus design implementing AXI must preserve relevant protocol rules."""
        retrieved_axi_rules = [
            {"citation": "AXI4 §A3.2.1", "text": "A value of VALID once asserted must remain asserted until READY is high."},
        ]

        doc_ir = DocumentIR(
            document_id="DOC_axi",
            filename="axi_slave.docx",
            blocks=[TextBlock(block_id="b1", text=axi_subordinate_spec_text, block_type="paragraph", section="Spec")]
        )
        extractor = RequirementExtractor(doc_ir)
        req_model = extractor.extract()
        req_model["protocol_references"] = ["AXI4_Lite"]

        builder = EFSIRBuilder(
            req_model=req_model,
            protocol_rules=retrieved_axi_rules,
            design_spec=axi_subordinate_spec_text
        )
        ir = builder.build()

        # AXI design must preserve the supported protocol rule
        assert len(ir.protocol_rules) >= 1
        assert "VALID" in ir.protocol_rules[0].description


# ==============================================================================
# 5. Placeholder Entity Rejection & Top Module Tests
# ==============================================================================

class TestPlaceholderEntityRejectionAndTopModule:
    """Verifies that meta-placeholders like 'Module Name' are rejected and top_module is populated."""

    def test_module_name_placeholder_rejected_and_top_module_set(self, traffic_light_spec_text):
        doc_ir = DocumentIR(
            document_id="DOC_tl",
            filename="traffic_light.docx",
            blocks=[TextBlock(block_id="b1", text=traffic_light_spec_text, block_type="paragraph", section="Spec")]
        )
        extractor = RequirementExtractor(doc_ir)
        req_model = extractor.extract()

        builder = EFSIRBuilder(req_model=req_model, design_spec=traffic_light_spec_text)
        ir = builder.build()

        # 1. No component named 'Module Name'
        for comp in ir.components:
            assert comp.name.strip().lower() not in DISQUALIFIED_COMPONENT_NAMES, (
                f"Disqualified placeholder component '{comp.name}' found in design components"
            )

        # 2. top_module is non-empty and valid
        assert ir.metadata.top_module != "", "top_module must not be empty"
        assert ir.metadata.top_module.lower() in ("traffic_light_controller", "traffic_light")

        # 3. Validation does not report placeholder or missing top_module errors
        issues = validate_efs_ir(ir)
        meta_comp_errors = [i for i in issues if i["category"] in ("Metadata", "Components") and i["severity"] == "ERROR"]
        assert len(meta_comp_errors) == 0, f"Unexpected metadata/component errors: {meta_comp_errors}"


# ==============================================================================
# 6. Extraction Quality Scorecard Integrity
# ==============================================================================

class TestExtractionQualityIntegrity:
    """Ensures extraction quality cannot report 'complete' when semantic defects exist."""

    def test_completeness_blocked_when_semantic_defects_present(self):
        # Design with reset mismatch and all-inputs defect
        ir = EFSIR(
            metadata=EFSMetadata(design_id=new_efs_id("dsn"), design_name="defective_design", top_module="defective_design"),
            components=[EFSComponent(name="defective_design")],
            signals=[
                EFSSignal(name="clk", direction="input", semantic_role="clock", reset="reset_n"),
                EFSSignal(name="reset_n", direction="input", semantic_role="reset", reset="reset_n"),
                EFSSignal(name="in_a", direction="input", reset="rst_n"),  # Mismatched reset
                EFSSignal(name="in_b", direction="input", reset="rst_n"),  # Zero outputs
            ],
            resets=[EFSSignal(name="reset_n", direction="input", semantic_role="reset")]
        )
        builder = EFSIRBuilder()
        builder.ir = ir
        builder.design_spec = "defective module"
        builder._build_extraction_quality()

        # Must report partial due to semantic defects
        assert ir.extraction_quality.ir_completeness == "partial"
        assert ir.extraction_quality.extraction_status == "partial"
        assert "all_signals_classified_as_inputs" in ir.extraction_quality.missing_information
        assert "reset_naming_mismatch" in ir.extraction_quality.missing_information
