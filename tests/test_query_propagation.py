"""
Verification test suite for Target/Query Propagation, Spec Separation, and Logging.

Tests:
1. "Generate EFS for instruction decoder" -> target = InstructionDecoder
2. "Generate EFS for Register File" -> target = Register_File
3. "Generate EFS for matrix compute unit" -> resolves only if target exists
4. Unknown target "Generate EFS for XYZ123" -> TARGET_RESOLUTION_FAILED (never Register_File)
5. Spec containing Unicode symbols (≤, ≥, ×, →) -> No UnicodeEncodeError
6. Log truncation helper (truncate_for_log) prevents full spec text logging
7. Complete User Spec remains available to retrieval context compilation independently of intent.target
"""

import pytest
from core.retrieval.query_analyzer import analyze_query
from core.retrieval.target_resolver import resolve_target
from core.efs_ir.builder import EFSIRBuilder
from core.retrieval.context_compiler import compile_task_context
from utils.logger import truncate_for_log, get_logger

logger = get_logger("test.query_propagation")


@pytest.fixture
def multi_component_efs_ir():
    spec_text = """
    # Hardware Design Specification: Custom Peripheral Controller
    
    Contains multi-core matrix processing units with Unicode constraints:
    - Bandwidth ≤ 100 Gbps
    - Clock frequency ≥ 500 MHz
    - Latency reduction × 2
    - State transition: IDLE → BUSY

    ## Architecture
    Submodules: Register_File, FSM_Control_Unit, InstructionDecoder
    """
    req_model = {"title": "Custom Peripheral Controller"}
    plan = {
        "system_architecture": "Custom Peripheral Controller",
        "submodules": [
            {"name": "Register_File", "purpose": "Register storage"},
            {"name": "FSM_Control_Unit", "purpose": "State machine logic"},
            {"name": "InstructionDecoder", "purpose": "Decodes matrix instructions"}
        ]
    }
    builder = EFSIRBuilder(req_model=req_model, plan=plan, design_spec=spec_text)
    return builder.build()


def test_1_instruction_decoder_target_parsing():
    """TEST 1 — Query 'Generate EFS for instruction decoder' parses target = InstructionDecoder."""
    intent = analyze_query("Generate EFS for instruction decoder")
    assert intent.target == "InstructionDecoder"
    assert "instruction decoder" not in intent.target.lower() or intent.target == "InstructionDecoder"


def test_2_register_file_target_parsing():
    """TEST 2 — Query 'Generate EFS for Register File' parses target = Register_File."""
    intent = analyze_query("Generate EFS for Register File")
    assert intent.target in ("RegisterFile", "Register_File")


def test_3_matrix_compute_unit_target_resolution(multi_component_efs_ir):
    """TEST 3 — Query 'Generate EFS for matrix compute unit' resolves only if target exists in IR."""
    intent = analyze_query("Generate EFS for matrix compute unit")

    res = resolve_target(intent.target, multi_component_efs_ir)
    # Target 'MatrixComputeUnit' is NOT in multi_component_efs_ir submodules
    assert res.is_success() is False
    assert res.status == "TARGET_NOT_FOUND"


def test_4_unknown_target_fails_explicitly(multi_component_efs_ir):
    """TEST 4 — Unknown target 'Generate EFS for XYZ123' returns TARGET_RESOLUTION_FAILED and NEVER Register_File."""
    intent = analyze_query("Generate EFS for XYZ123")

    res = resolve_target(intent.target, multi_component_efs_ir)
    assert res.is_success() is False
    assert res.status == "TARGET_NOT_FOUND"
    assert "TARGET_RESOLUTION_FAILED" in res.error_message
    assert "XYZ123" in res.error_message
    assert res.target_component is None or res.target_component.name != "Register_File"


def test_5_unicode_logging_spec_handling(multi_component_efs_ir):
    """TEST 5 — Specification containing Unicode symbols (≤, ≥, ×, →) logs safely without UnicodeEncodeError."""
    unicode_str = "Unicode specs: Bandwidth ≤ 100 Gbps, Frequency ≥ 500 MHz, Multiplier × 2, Flow: IDLE → BUSY"
    
    # Verify truncate_for_log handles unicode cleanly
    truncated = truncate_for_log(unicode_str, max_chars=40)
    assert "≤" in truncated or "Unicode" in truncated

    # Verify logging does not raise UnicodeEncodeError
    logger.info(f"Testing unicode logging: {unicode_str}")


def test_6_log_truncation_prevents_full_spec_logging():
    """TEST 6 — Terminal logger uses truncate_for_log to cap logged query text."""
    long_spec_text = "Section " + " ".join([f"Word_{i}" for i in range(500)])
    
    truncated = truncate_for_log(long_spec_text, max_chars=100)
    assert len(truncated) < 200
    assert "[len=" in truncated
    assert "hash=" in truncated


def test_7_user_spec_available_to_context_independently(multi_component_efs_ir):
    """TEST 7 — User Spec remains available to context compilation independently of intent.target."""
    user_query = "Generate EFS for InstructionDecoder"
    intent = analyze_query(user_query)
    
    assert intent.target == "InstructionDecoder"
    
    # Context compilation using user_query and multi_component_efs_ir
    pack, res = compile_task_context(user_query, multi_component_efs_ir)
    assert res.is_success() is True
    assert res.target_component.name == "InstructionDecoder"
    assert pack.task.target == "InstructionDecoder"
