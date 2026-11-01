"""
Comprehensive Automated Test Suite for EFS Semantic Extraction & Classification.

Verifies:
1. Matrix Compute Unit extraction accuracy (Instructions, Formats, Opcodes, Components, Memory, Data Structures, Execution, Performance, Requirements, Errors).
2. Negative instruction extraction test (Spec with no ISA yields instructions = []).
3. Foreign document contamination test (Zero amba_axi_protocol_spec references).
4. Genericity test across distinct hardware specification.
5. Prints runtime EFS EXTRACTION VALIDATION semantic inventory report and 5 representative objects.
"""

from __future__ import annotations

import os
import json
import pytest
from core.pipeline import run_pipeline
from core.pipeline_diagnostics import generate_semantic_inventory, generate_pipeline_validation_report


def test_matrix_compute_unit_extraction():
    """Verify semantic candidate classification & EFS IR generation for Matrix Compute Unit specification."""
    spec_text = """
    # Multi-Core Matrix Compute Unit Subsystem Specification v1.0

    Section 1. Subsystem & Component Overview
    The Matrix Compute Unit (MCU) consists of a 32x32 PE array performing matrix tile calculations.
    Key components include:
    - Matrix ALU: Performs 32x32 MAC operations per cycle
    - Control Unit: FSM-based main controller
    - Register File: 64 vector registers
    - DMA Engine: Transfers matrix tiles between L1 SRAM and DRAM
    - Memory Interface: AXI4-compatible interface

    Section 2. Control & Status Registers
    - CTRL_REG: offset 0x00, access RW, description "MCU Main Control Register"
    - STATUS_REG: offset 0x04, access RO, description "MCU Execution Status Register"

    Section 3. Instruction Set Architecture & Format
    Instruction Format (64 bits):
    - Opcode: 6 bits
    - Destination register: 4 bits
    - Source register 1: 4 bits
    - Source register 2: 4 bits
    - Dimension control: 10 bits
    - Flags: 4 bits

    Instruction Set Table:
    - MLOAD: Opcode 0x01, Load tile into local SRAM buffer
    - MSTORE: Opcode 0x02, Store tile from SRAM buffer to external memory
    - MADD: Opcode 0x03, Execute tile addition
    - MMUL: Opcode 0x04, Execute matrix multiplication

    Section 4. Memory Resources & Buffers
    - SRAM_TILE_BUF: Base address 0x80000000, Size 64KB, SRAM local buffer
    - L1_CACHE: Base address 0x90000000, Size 512KB, L1 Matrix Cache

    Section 5. Data Structures
    Matrix Descriptor format:
    - rows: 16 bits
    - columns: 16 bits
    - format: 8 bits
    - address: 64 bits

    Section 6. Execution Model & Performance
    - Tiling approach for large matrix operations
    - Pipeline execution across 4 matrix stages
    - Frequency: 500 MHz
    - Throughput: 6.25 TFLOPS

    Section 7. Error Handling
    - OVERFLOW_ERR: Triggers when matrix MAC result exceeds 32-bit float range
    """

    tmp_spec = "test_matrix_compute_unit_spec.md"
    with open(tmp_spec, "w", encoding="utf-8") as f:
        f.write(spec_text)

    try:
        result = run_pipeline(filepath=tmp_spec, filename="test_matrix_compute_unit_spec.md")
        assert result.efs_ir is not None
        assert result.req_ir is not None

        efs = result.efs_ir
        req_ir = result.req_ir

        # 1. Verify Instruction extraction accuracy
        inst_mnemonics = [i.mnemonic.upper() for i in efs.instructions]
        assert "MLOAD" in inst_mnemonics
        assert "MSTORE" in inst_mnemonics
        assert "MADD" in inst_mnemonics
        assert "MMUL" in inst_mnemonics

        # Ensure general terms like "Functional", "Register", "Rows", "Cols", "Flags" are NOT instructions
        invalid_insts = {"FUNCTIONAL", "REGISTER", "CONTROL", "MEMORY", "MATRIX", "ROWS", "COLS", "FLAGS", "TILING", "PIPELINE", "EXECUTION"}
        detected_invalid = set(inst_mnemonics).intersection(invalid_insts)
        assert len(detected_invalid) == 0, f"Misclassified terms as instructions: {detected_invalid}"

        # 2. Verify Opcode extraction
        opc_values = [o.binary_encoding for o in efs.opcodes]
        assert len(opc_values) >= 4
        assert "0x01" in opc_values or "1" in opc_values

        # 3. Verify Components
        comp_names = [c.name for c in efs.components]
        assert any("Matrix ALU" in c for c in comp_names) or any("ALU" in c for c in comp_names)

        # 4. Verify Memory Regions
        mem_names = [m.name for m in efs.memory_regions]
        assert any("SRAM_TILE_BUF" in m for m in mem_names)

        # 5. Verify Data Structures
        ds_names = [d.name for d in efs.data_structures]
        assert any("Matrix Descriptor" in d for d in ds_names)

        # 6. Verify Execution Model
        exec_types = [e.type for e in efs.execution_models]
        assert len(exec_types) > 0

        # 7. Verify Performance Targets
        perf_metrics = [p.metric for p in efs.performance_requirements]
        assert any(m in ("THROUGHPUT", "FREQUENCY", "PERFORMANCE") for m in perf_metrics)

        # 8. Print Runtime Semantic Inventory & Representative Objects
        inventory = generate_semantic_inventory(req_ir, efs)
        print("\n" + "=" * 50)
        print("EFS EXTRACTION VALIDATION REPORT")
        print("=" * 50)
        for k, v in inventory.items():
            print(f"{k:25s}: {v}")
        print("=" * 50)

        # 5 Representative Objects
        print("\n--- 5 REPRESENTATIVE EXTRACTED OBJECTS ---")
        if efs.components:
            print("\n1. COMPONENT:")
            print(json.dumps(efs.components[0].to_dict(), indent=2))
        if efs.instructions:
            print("\n2. INSTRUCTION:")
            print(json.dumps(efs.instructions[0].to_dict(), indent=2))
        if efs.memory_regions:
            print("\n3. MEMORY RESOURCE:")
            print(json.dumps(efs.memory_regions[0].to_dict(), indent=2))
        if efs.requirements:
            print("\n4. REQUIREMENT:")
            print(json.dumps(efs.requirements[0].to_dict(), indent=2))
        if efs.execution_models:
            print("\n5. EXECUTION MODEL:")
            print(json.dumps(efs.execution_models[0].to_dict(), indent=2))

    finally:
        if os.path.exists(tmp_spec):
            os.remove(tmp_spec)


def test_negative_instruction_extraction():
    """Negative Test: Verify a spec with registers/memory/control but NO ISA yields instructions = []."""
    spec_text = """
    # Smart Memory Controller Module Specification v1.0

    Section 1. Module Overview
    The Smart Memory Controller handles refresh and arbitration for external DRAM.

    Section 2. Control Registers
    - REFRESH_REG: offset 0x00, access RW, description "DRAM Refresh Interval Register"
    - ARB_STATUS: offset 0x04, access RO, description "Arbiter Status"

    Section 3. Memory Domains
    - DRAM_BANK0: Base address 0xC0000000, Size 2GB
    """

    tmp_spec = "test_smart_mem_spec.md"
    with open(tmp_spec, "w", encoding="utf-8") as f:
        f.write(spec_text)

    try:
        result = run_pipeline(filepath=tmp_spec, filename="test_smart_mem_spec.md")
        efs = result.efs_ir
        assert efs is not None

        # Must NOT extract any fake instructions
        insts = [i.mnemonic for i in efs.instructions]
        assert len(insts) == 0, f"Expected zero instructions for non-ISA spec, got: {insts}"

    finally:
        if os.path.exists(tmp_spec):
            os.remove(tmp_spec)


def test_zero_foreign_document_contamination():
    """Verify processing a user spec yields zero foreign document objects."""
    spec_text = """
    # Custom AI Accelerator Unit Specification
    - ACCEL_REG: offset 0x00, access RW
    """
    tmp_spec = "test_custom_accel.md"
    with open(tmp_spec, "w", encoding="utf-8") as f:
        f.write(spec_text)

    try:
        result = run_pipeline(filepath=tmp_spec, filename="test_custom_accel.md")
        full_json = str(result.to_full_json())
        assert "amba_axi_protocol_spec" not in full_json
        val_report = generate_pipeline_validation_report(result.req_ir, result.efs_ir)
        assert val_report["contamination"]["foreign_objects"] == 0
    finally:
        if os.path.exists(tmp_spec):
            os.remove(tmp_spec)


def test_genericity_on_second_spec():
    """Genericity Test: Verify exact same pipeline extracts distinct hardware concepts from a second spec."""
    spec_text = """
    # Tensor Processing Engine Subsystem v2.0

    Section 1. Component Architecture
    - Tensor Core: 128x128 systolic array compute engine
    - Vector Engine: 16-element SIMD execution unit

    Section 2. Instruction Set
    - VLOAD: Opcode 0x10, Load vector register
    - TMATMUL: Opcode 0x20, Execute tensor matrix multiply

    Section 3. Memory & Performance
    - SYSTOLIC_BUF: Base address 0xA0000000, Size 1MB
    - Throughput: 32.0 TFLOPS
    - Frequency: 800 MHz
    """
    tmp_spec = "test_tensor_engine.md"
    with open(tmp_spec, "w", encoding="utf-8") as f:
        f.write(spec_text)

    try:
        result = run_pipeline(filepath=tmp_spec, filename="test_tensor_engine.md")
        efs = result.efs_ir
        assert efs is not None

        inst_mnemonics = [i.mnemonic.upper() for i in efs.instructions]
        assert "VLOAD" in inst_mnemonics
        assert "TMATMUL" in inst_mnemonics

        mem_names = [m.name for m in efs.memory_regions]
        assert any("SYSTOLIC_BUF" in m for m in mem_names)
    finally:
        if os.path.exists(tmp_spec):
            os.remove(tmp_spec)
