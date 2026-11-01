"""
Regression tests for Target Resolution and Component Extraction Pipeline.

Tests:
1. InstructionDecoder (PascalCase) -> Instruction Decoder
2. Instruction Decoder (Spaced) -> Instruction Decoder
3. instruction_decoder (snake_case) -> Instruction Decoder
4. instruction-decoder (kebab-case) -> Instruction Decoder
5. InstrucionDecoder (typo) -> controlled fuzzy match or candidate suggestion, NEVER Register_File
6. CompletelyUnknownUnit -> TARGET_NOT_FOUND, NEVER Register_File
7. Existing targets (Register_File, FSM_Control_Unit) resolve correctly
8. User Spec regression verification
"""

import pytest
from core.efs_ir.models import EFSIR, EFSComponent, EFSMetadata
from core.retrieval.target_resolver import resolve_target, TargetResolutionResult
from core.normalization.target_normalizer import normalize_target_name


@pytest.fixture
def sample_efs_ir():
    """Create a sample EFS IR representing extracted hardware components."""
    ir = EFSIR()
    ir.metadata = EFSMetadata(
        design_name="matrix_multi_core",
        canonical_id="matrixmulticore",
        display_name="Matrix Multi Core Compute Engine",
        aliases=["MatrixMultiCore", "matrix_compute"]
    )
    ir.components = [
        EFSComponent(component_id="COMP_001", name="Instruction Decoder", type="component", description="Parses assembly instructions"),
        EFSComponent(component_id="COMP_002", name="Register_File", type="component", description="Storage for registers"),
        EFSComponent(component_id="COMP_003", name="64 Matrix ALU", type="component", description="64-element matrix ALU"),
        EFSComponent(component_id="COMP_004", name="Control Unit", type="component", description="FSM Control Unit"),
        EFSComponent(component_id="COMP_005", name="Memory Interface", type="component", description="Memory boundary interface"),
        EFSComponent(component_id="COMP_006", name="FSM_Control_Unit", type="component", description="State machine controller")
    ]
    return ir


def test_normalization_utility():
    """Verify target name normalization across spacing, casing, and punctuation variants."""
    assert normalize_target_name("Instruction Decoder") == "instructiondecoder"
    assert normalize_target_name("InstructionDecoder") == "instructiondecoder"
    assert normalize_target_name("instruction_decoder") == "instructiondecoder"
    assert normalize_target_name("instruction-decoder") == "instructiondecoder"
    assert normalize_target_name("InstrucionDecoder") == "instruciondecoder"


def test_1_pascal_case(sample_efs_ir):
    """TEST 1: Query InstructionDecoder resolves to Instruction Decoder."""
    res = resolve_target("InstructionDecoder", sample_efs_ir)
    assert res.is_success()
    assert res.target_component.name == "Instruction Decoder"


def test_2_spaced(sample_efs_ir):
    """TEST 2: Query 'Instruction Decoder' resolves to Instruction Decoder."""
    res = resolve_target("Instruction Decoder", sample_efs_ir)
    assert res.is_success()
    assert res.target_component.name == "Instruction Decoder"


def test_3_snake_case(sample_efs_ir):
    """TEST 3: Query 'instruction_decoder' resolves to Instruction Decoder."""
    res = resolve_target("instruction_decoder", sample_efs_ir)
    assert res.is_success()
    assert res.target_component.name == "Instruction Decoder"


def test_4_kebab_case(sample_efs_ir):
    """TEST 4: Query 'instruction-decoder' resolves to Instruction Decoder."""
    res = resolve_target("instruction-decoder", sample_efs_ir)
    assert res.is_success()
    assert res.target_component.name == "Instruction Decoder"


def test_5_typo_tolerance(sample_efs_ir):
    """TEST 5: Minor typo InstrucionDecoder resolves to Instruction Decoder or gives candidates, NEVER Register_File."""
    res = resolve_target("InstrucionDecoder", sample_efs_ir)
    if res.is_success():
        assert res.target_component.name == "Instruction Decoder"
    else:
        assert res.status == "TARGET_NOT_FOUND"
        assert res.target_component is None
    # Crucial assertion: Must NEVER resolve to Register_File
    if res.target_component:
        assert res.target_component.name != "Register_File"


def test_6_unknown_target(sample_efs_ir):
    """TEST 6: Query for unknown unit returns TARGET_NOT_FOUND, NEVER Register_File."""
    res = resolve_target("CompletelyUnknownUnit", sample_efs_ir)
    assert res.status == "TARGET_NOT_FOUND"
    assert res.target_component is None
    # Crucial assertion: Must NEVER resolve to Register_File or Control Unit
    assert "TARGET_RESOLUTION_FAILED" in res.error_message


def test_7_existing_targets(sample_efs_ir):
    """TEST 7: Verify existing targets like Register_File and FSM_Control_Unit resolve correctly."""
    res_reg = resolve_target("Register_File", sample_efs_ir)
    assert res_reg.is_success()
    assert res_reg.target_component.name == "Register_File"

    res_fsm = resolve_target("FSM_Control_Unit", sample_efs_ir)
    assert res_fsm.is_success()
    assert res_fsm.target_component.name == "FSM_Control_Unit"


def test_8_user_spec_regression():
    """TEST 8: Verify User Spec parsing/extraction imports and behavior remain functional."""
    try:
        from core.agents.requirement_agent import analyze_requirements
        req = analyze_requirements("The system shall implement a 32-bit register file.")
        assert req is not None
    except Exception as e:
        pytest.fail(f"User Spec parsing regression failure: {e}")
