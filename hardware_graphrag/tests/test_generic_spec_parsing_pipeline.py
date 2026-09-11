"""
Test suite for Generic Specification Parsing -> Requirement IR -> EFS IR Pipeline.
"""

import os
import pytest
from core.pipeline import run_pipeline
from core.requirement_ir.models import DocumentIR, DocumentBlock, RequirementIR, AtomicRequirement, DiscoveredEntity, SpecConflict
from core.compiler.efs_compiler import RequirementToEFSCompiler
from core.requirement_ir.validator import RequirementValidator


SAMPLE_HARDWARE_SPEC_MD = """# AI Matrix Coprocessor Hardware Specification

## 1. Architecture & Execution Engine
The AI Matrix Coprocessor executes matrix multiplication and element-wise matrix arithmetic operations.
The execution core operates at a target maximum frequency of 100 MHz and requires 4 cycles latency per matrix operation.

## 2. External Interface Port List
- clk: input, 1 bit, System clock signal
- rst_n: input, 1 bit, Active-low asynchronous reset
- start_op: input, 1 bit, Control start signal
- op_code: input, 8 bits, Opcode selector input
- matrix_data_in: input, 64 bits, Streaming matrix payload data
- matrix_data_out: output, 64 bits, Output matrix result payload
- busy: output, 1 bit, Asserted high when coprocessor is busy
- done: output, 1 bit, Asserted high when operation completes

## 3. Instruction Set & Opcodes
| Instruction | Opcode | Format | Description |
|---|---|---|---|
| MLOAD | 0x01 | [63:0] | Load matrix data from external memory into L1 SRAM buffer |
| MSTORE | 0x02 | [63:0] | Store result matrix data back to external memory |
| MADD | 0x03 | [63:0] | Perform element-wise 64-bit matrix addition |
| MMUL | 0x04 | [63:0] | Perform 4x4 matrix multiplication |

## 4. Hierarchical Memory & Data Structures
- L1_Matrix_SRAM: memory_region, base_address=0x80000000, size=64KB, depth=1024, width=64 bits. Stores tile operands.
- Matrix_Descriptor: data_structure, width=128 bits. Contains rows, columns, stride, and precision encoding fields.

## 5. Control Register Layout
| Register | Offset | Access | Reset | Description |
|---|---|---|---|---|
| CTRL_REG | 0x00 | RW | 0x00000000 | Enable, Start, Interrupt Mask |
| STAT_REG | 0x04 | RO | 0x00000000 | Busy, Done, Error Flags |

## 6. Error Handling & Exception Conditions
| Error Name | Code | Description |
|---|---|---|
| INVALID_OPCODE | 0xE01 | Raised when an undefined opcode encoding is passed |
| MATRIX_OVERFLOW | 0xE02 | Raised when element computation exceeds 64-bit precision |

## 7. State Machine
The control unit transitions:
- IDLE -> FETCH: when start_op is asserted
- FETCH -> EXECUTE: when MLOAD completes
- EXECUTE -> DONE: when operation completes
- DONE -> IDLE: when acknowledge signal is received

## 8. Specification Contradictions
Note: In Section 2, matrix_data_in is specified as 64 bits width. In a legacy subsection, matrix_data_in input is declared as 32 bits width.
"""


def test_generic_spec_pipeline_end_to_end(tmp_path):
    """Test full parsing of a generic AI coprocessor specification without protocol hardcoding."""
    spec_file = tmp_path / "matrix_coprocessor_spec.md"
    spec_file.write_text(SAMPLE_HARDWARE_SPEC_MD, encoding="utf-8")

    res = run_pipeline(str(spec_file), filename="matrix_coprocessor_spec.md")

    # 1. Pipeline result checks
    assert res.doc is not None
    assert res.req_ir is not None
    assert res.efs_ir is not None
    assert len(res.errors) == 0

    # 2. Check full JSON output keys
    full_json = res.to_full_json()
    required_keys = [
        "document", "sections", "entities", "requirements", "relationships",
        "behaviors", "transactions", "states", "transitions", "conditions",
        "events", "timing", "interfaces", "registers", "operations",
        "instructions", "memory", "data_structures", "errors", "performance",
        "power", "security", "subflows", "constraints", "conflicts",
        "source_traceability", "efs", "extraction_coverage", "missing_information_report"
    ]
    for key in required_keys:
        assert key in full_json, f"Missing required top-level JSON key '{key}'"

    # 3. Verify extracted instructions & opcodes
    inst_names = {i.mnemonic for i in res.efs_ir.instructions}
    assert "MLOAD" in inst_names or "MADD" in inst_names or "MMUL" in inst_names

    # 4. Verify ZERO unrequested protocol ports (e.g. s_axi_wdata) were force-injected
    sig_names = [s.name for s in res.efs_ir.signals]
    assert not any(s.startswith("s_axi_") for s in sig_names)
    assert "matrix_data_in" in sig_names or "clk" in sig_names

    # 5. Verify Memory Regions & Errors extracted
    assert len(res.efs_ir.memory_regions) > 0 or any(e.type == "memory_region" for e in res.req_ir.entities)
    assert len(res.efs_ir.errors) > 0 or any(e.type == "error" for e in res.req_ir.entities)


def test_validator_detects_width_contradiction():
    """Test that RequirementValidator surfaces bit width contradictions as explicit SpecConflicts."""
    req1 = AtomicRequirement(
        requirement_id="REQ_001",
        statement="Signal matrix_data_in is 64 bits width.",
        subject="matrix_data_in"
    )
    req2 = AtomicRequirement(
        requirement_id="REQ_002",
        statement="Signal matrix_data_in is 32 bits width.",
        subject="matrix_data_in"
    )
    ent = DiscoveredEntity(name="matrix_data_in", type="port")
    req_ir = RequirementIR(requirements=[req1, req2], entities=[ent])

    validator = RequirementValidator(req_ir)
    conflicts, status = validator.validate()

    assert len(conflicts) > 0
    assert any(c.type == "WIDTH_MISMATCH" for c in conflicts)
    assert status["has_critical_conflicts"] is True
