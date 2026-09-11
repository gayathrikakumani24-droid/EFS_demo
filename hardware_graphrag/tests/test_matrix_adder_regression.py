"""
Regression test suite for Hardware GraphRAG + EFS IR Context Compilation Layer.

Tests end-to-end targeted generation for Matrix Adder, Matrix Subtractor, and Matrix Multiplier
within a 30-40 page multi-module specification system.
"""

import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.efs_ir.models import (
    EFSIR, EFSComponent, EFSInterface, EFSSignal, EFSRegister,
    EFSRegisterField, EFSFSM, EFSState, EFSTransition, EFSMetadata
)
from core.retrieval.context_compiler import compile_task_context
from core.agents.orchestrator import run_design_flow


def create_mock_matrix_system_spec() -> str:
    """Mock text specification representing a 30-40 page hardware requirement specification."""
    return """
# Matrix Compute System Hardware Specification

## 1. System Overview
The Matrix Compute System is a high-performance hardware accelerator containing multiple sub-blocks:
- Matrix Adder
- Matrix Subtractor
- Matrix Multiplier
- Register File
- DMA Controller
- Scheduler Engine
- Memory Controller

## 2. Matrix Adder Submodule
The Matrix Adder accepts two 64-bit operands (Operand A and Operand B) and supports matrix dimensions up to 32x32.
Completion is indicated by the done signal.
The Matrix Adder connects to an AXI4-Lite slave control interface.

### Signals
| Signal | Direction | Width | Description |
| clk | input | 1 | Clock signal |
| rst_n | input | 1 | Active-low reset |
| operand_a | input | 64 | Matrix A operand input |
| operand_b | input | 64 | Matrix B operand input |
| result | output | 64 | Matrix addition output |
| done | output | 1 | Operation completion flag |
| s_axi_awvalid | input | 1 | AXI Write Address Valid |
| s_axi_awready | output | 1 | AXI Write Address Ready |
| s_axi_wdata | input | 32 | AXI Write Data |
| s_axi_wvalid | input | 1 | AXI Write Data Valid |

### Registers
| Register | Offset | Access | Description |
| ADDER_CTRL | 0x00 | RW | Control register for Matrix Adder |
| ADDER_STATUS | 0x04 | RO | Status register |

### FSM States
IDLE, FETCH_OPS, ADD_COMPUTE, OUTPUT_DONE

## 3. Matrix Subtractor Submodule
The Matrix Subtractor performs 64-bit matrix subtraction operations.

## 4. Matrix Multiplier Submodule
The Matrix Multiplier performs 64-bit systolic array matrix multiplications.
"""


def test_matrix_adder_rtl_regression():
    """Regression Test 1: Generate Verilog RTL for MatrixAdder."""
    raw_spec = create_mock_matrix_system_spec()
    query = "Generate synthesizable Verilog RTL for MatrixAdder."

    # Run complete design flow
    results = run_design_flow(
        requirement_text=raw_spec,
        target_outputs=["Verilog"],
        user_query=query
    )

    assert results["status"] != "TARGET_NOT_FOUND"
    assert results["status"] != "AMBIGUOUS_TARGET"
    assert "context_pack" in results

    cp = results["context_pack"]
    assert cp.task.target == "MatrixAdder"
    assert cp.task.task == "RTL_GENERATION"

    # Verify selective context retrieval metrics
    stats = cp.to_dict()["stats"]
    
    print("\n============================================================")
    print("REGRESSION TEST 1 RESULTS: MatrixAdder RTL")
    print("============================================================")
    print(f"FULL IR SIZE (Total Objects):          {stats['total_full_efs_objects']}")
    print(f"SELECTED IR SIZE (Target Slice):       {stats['selected_efs_objects']}")
    print(f"FULL PROTOCOL CONTEXT SIZE:            {stats['total_protocol_rules']}")
    print(f"SELECTED PROTOCOL CONTEXT SIZE:        {stats['selected_protocol_rules']}")
    print(f"FINAL CONTEXT PACK SIZE (Tokens):      {stats['estimated_tokens']}")
    print("============================================================\n")

    # Assert complete EFS IR was NOT sent to LLM (Selected < Total)
    assert stats['selected_efs_objects'] <= stats['total_full_efs_objects']
    assert stats['estimated_tokens'] > 0

    # Verify Verilog output exists for MatrixAdder
    verilog_out = results["outputs"]["Verilog"]["final_code"]
    assert "module" in verilog_out.lower() or "verilog" in verilog_out.lower() or "matrix" in verilog_out.lower()


def test_matrix_adder_sva_regression():
    """Regression Test 2: Generate SVA assertions for AXI interface of MatrixAdder."""
    raw_spec = create_mock_matrix_system_spec()
    query = "Generate SVA assertions for the AXI interface of MatrixAdder."

    results = run_design_flow(
        requirement_text=raw_spec,
        target_outputs=["Assertions"],
        user_query=query
    )

    assert "context_pack" in results
    cp = results["context_pack"]
    assert cp.task.target == "MatrixAdder"
    assert cp.task.task == "SVA_GENERATION"
    assert cp.task.interface == "AXI"

    stats = cp.to_dict()["stats"]
    print("\n============================================================")
    print("REGRESSION TEST 2 RESULTS: MatrixAdder AXI SVA")
    print("============================================================")
    print(f"SELECTED IR SIZE:                      {stats['selected_efs_objects']}")
    print(f"SELECTED PROTOCOL RULES:               {stats['selected_protocol_rules']}")
    print(f"ESTIMATED TOKENS:                      {stats['estimated_tokens']}")
    print("============================================================\n")


def test_matrix_multiplier_uvm_regression():
    """Regression Test 3: Generate UVM testbench for MatrixMultiplier."""
    raw_spec = create_mock_matrix_system_spec()
    query = "Generate UVM testbench for MatrixMultiplier."

    results = run_design_flow(
        requirement_text=raw_spec,
        target_outputs=["UVM"],
        user_query=query
    )

    assert "context_pack" in results
    cp = results["context_pack"]
    assert cp.task.target == "MatrixMultiplier"
    assert cp.task.task == "UVM_GENERATION"

    stats = cp.to_dict()["stats"]
    print("\n============================================================")
    print("REGRESSION TEST 3 RESULTS: MatrixMultiplier UVM")
    print("============================================================")
    print(f"SELECTED IR SIZE:                      {stats['selected_efs_objects']}")
    print(f"ESTIMATED TOKENS:                      {stats['estimated_tokens']}")
    print("============================================================\n")


if __name__ == "__main__":
    test_matrix_adder_rtl_regression()
    test_matrix_adder_sva_regression()
    test_matrix_multiplier_uvm_regression()
