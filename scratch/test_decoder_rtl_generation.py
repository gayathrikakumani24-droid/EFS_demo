"""Verify end-to-end RTL generation for instruction_decoder sub-module."""

import sys
from pathlib import Path
from core.efs_ir.models import EFSIR, EFSSignal, EFSComponent, EFSMetadata, EFSOpcode
from core.agents.rtl_agent import plan_rtl
from core.agents.constrained_generator import generate_hybrid_rtl, GenerationMode

def run_test():
    spec_text = """
# Functional Specification for Multi Core Matrix Compute Unit

3.1 Instruction Format
Each instruction is a 64-bit word with the following fields:
   Opcode    (6 bits): Operation to perform
   Destination Register    (4 bits): Target register (16 matrix registers)
   Source Register 1    (4 bits): First operand register
   Source Register 2    (4 bits): Second operand register
   Dimension Control    (10 bits): Matrix dimension information
Rows Source 1 (5 bits): Number of rows in first source matrix
Cols Source 1/Rows Source 2 (5 bits): Number of columns in first source/rows in second source
   Cols Source 2    (5 bits): Number of columns in second source matrix
   Format    (3 bits): Data format control (fixed-point, floating-point, etc.)
   Flags    (4 bits): Special operation modifiers
   Reserved    (24 bits): For future extensions

4.1 Matrix Operations
The unit supports the following matrix instructions:
- MADD: Matrix Addition (000001)
- MSUB: Matrix Subtraction (000010)
- MMUL: Matrix Multiplication (000011)
- MDIV: Matrix Division (000100)
- MLOAD: Load Matrix (000101)
- MSTORE: Store Matrix (000110)
- MTRANS: Transpose Matrix (000111)
- MZERO: Zero Matrix (001000)
- MIDENTITY: Identity Matrix (011001)
- MINV: Invert Matrix (001110)
- MDET: Determinant (001111)
- MEIG: Eigenvalues/Eigenvectors (011000)
- MSVD: Singular Value Decomposition (001001)
- MSIZE: Size Matrix (001010)
- MSLICE: Slice Matrix (001011)
- MCONCAT: Concatenate Matrices (001100)
- MTYPE: Type Conversion (001101)
"""

    top_comp = EFSComponent(component_id="COMP_TOP", name="multi_core_matrix_system", type="top_level_design")
    dec_comp = EFSComponent(component_id="COMP_DEC", name="instruction_decoder", type="sub_module")

    # Global system signals (including 512-bit AXI bus that must NOT leak into decoder)
    signals = [
        EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
        EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset"),
        EFSSignal(name="start", direction="input", width="1", semantic_role="control"),
        EFSSignal(name="done", direction="output", width="1", semantic_role="status"),
        EFSSignal(name="error", direction="output", width="1", semantic_role="error"),
        EFSSignal(name="axi_araddr", direction="output", width="32", semantic_role="address"),
        EFSSignal(name="axi_wdata", direction="output", width="512", semantic_role="data"),
        EFSSignal(name="instruction", direction="input", width="64", semantic_role="data"),
    ]

    opcodes = [
        EFSOpcode(opcode_id="OPC_MADD", mnemonic="MADD", binary_encoding="6'b000001"),
        EFSOpcode(opcode_id="OPC_MSUB", mnemonic="MSUB", binary_encoding="6'b000010"),
        EFSOpcode(opcode_id="OPC_MMUL", mnemonic="MMUL", binary_encoding="6'b000011"),
        EFSOpcode(opcode_id="OPC_MDIV", mnemonic="MDIV", binary_encoding="6'b000100"),
        EFSOpcode(opcode_id="OPC_MLOAD", mnemonic="MLOAD", binary_encoding="6'b000101"),
        EFSOpcode(opcode_id="OPC_MSTORE", mnemonic="MSTORE", binary_encoding="6'b000110"),
    ]

    efs_ir = EFSIR(
        metadata=EFSMetadata(top_module="multi_core_matrix_system", design_name="multi_core_matrix_system"),
        components=[top_comp, dec_comp],
        signals=signals,
        opcodes=opcodes
    )

    print("=== STAGE 1: PLAN RTL FOR INSTRUCTION_DECODER ===")
    plan = plan_rtl(
        design_spec=spec_text,
        target="instruction_decoder",
        efs_ir=efs_ir
    )

    print(f"Plan Module Name: {plan.get('module_name')}")
    print(f"Plan Spec Status: {plan.get('specification_status', {}).get('status')}")
    print(f"Plan Blocking Issues Count: {len(plan.get('specification_status', {}).get('blocking_issues', []))}")
    print(f"Plan Output Assignments: {plan.get('output_assignments')}")

    assert plan.get("module_name") == "instruction_decoder", f"Wrong module name: {plan.get('module_name')}"
    assert plan.get("specification_status", {}).get("status") == "COMPLETE", f"Plan not COMPLETE: {plan.get('specification_status')}"

    print("\n=== STAGE 2: GENERATE HYBRID RTL FOR INSTRUCTION_DECODER ===")
    code, report = generate_hybrid_rtl(
        design_spec=spec_text,
        target="Verilog",
        efs_ir=efs_ir,
        rtl_plan=plan,
        mode=GenerationMode.DETERMINISTIC
    )

    print("\n--- GENERATED RTL CODE ---")
    print(code)
    print("--------------------------\n")
    print(f"Candidate Verification Passed: {report.get('passed')}")

    # Strict Assertions on Generated RTL
    assert "module instruction_decoder" in code, "Missing module instruction_decoder header"
    assert "axi_wdata" not in code, "LEAKAGE: axi_wdata found in instruction_decoder!"
    assert "axi_araddr" not in code, "LEAKAGE: axi_araddr found in instruction_decoder!"
    assert "// [PARTIAL GENERATION]" not in code, "FAIL: Code contains [PARTIAL GENERATION]!"
    
    # Check bit slicing
    assert "assign opcode = instruction[63:58];" in code, "Missing opcode bit slicing!"
    assert "assign dest_reg = instruction[57:54];" in code, "Missing dest_reg bit slicing!"
    assert "assign src_reg_1 = instruction[53:50];" in code, "Missing src_reg_1 bit slicing!"
    assert "assign src_reg_2 = instruction[49:46];" in code, "Missing src_reg_2 bit slicing!"
    assert "assign rows_source_1 = instruction[45:41];" in code, "Missing rows_source_1 bit slicing!"
    assert "assign cols_source_1_rows_source_2 = instruction[40:36];" in code, "Missing cols_source_1_rows_source_2 bit slicing!"
    assert "assign cols_source_2 = instruction[35:31];" in code, "Missing cols_source_2 bit slicing!"
    assert "assign format = instruction[30:28];" in code, "Missing format bit slicing!"
    assert "assign flags = instruction[27:24];" in code, "Missing flags bit slicing!"
    assert "assign done = start;" in code, "Missing done = start assignment!"
    assert "assign error = 1'b0;" in code, "Missing error = 1'b0 assignment!"

    print("\n>>> ALL ASSERTIONS PASSED! PERFECT DECODER RTL GENERATED! <<<")

if __name__ == "__main__":
    run_test()
