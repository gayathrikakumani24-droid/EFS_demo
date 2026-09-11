"""
Mandatory Foreign Contamination Test.
Verifies that processing a user specification produces ZERO references to foreign specifications
(e.g., amba_axi_protocol_spec.docx) across Document IR, Requirement IR, Graph, EFS IR, and validation reports.
"""

from __future__ import annotations

import os
import pytest
from core.pipeline import run_pipeline
from core.pipeline_diagnostics import generate_pipeline_validation_report


def test_zero_foreign_contamination_in_matrix_compute():
    """Verify matrix_compute_unit spec contains zero foreign document references."""
    spec_text = """
    # Multi-Core Matrix Compute Unit Subsystem Specification v1.0
    
    Section 1. Subsystem Overview
    The Matrix Compute Unit (MCU) consists of a 32x32 PE array performing matrix tile calculations.
    
    Section 2. Control Registers
    - CTRL_REG: offset 0x00, access RW, description "MCU Control Register"
    - STATUS_REG: offset 0x04, access RO, description "MCU Status Register"
    
    Section 3. Instruction Set Architecture
    - MLOAD: Opcode 0x01, Load tile into local SRAM buffer
    - MADD: Opcode 0x02, Execute tile addition
    - MMUL: Opcode 0x03, Execute matrix multiplication
    
    Section 4. Memory Buffers
    - SRAM_TILE_BUF: Base address 0x80000000, Size 64KB
    """

    tmp_spec_file = "test_matrix_spec.md"
    with open(tmp_spec_file, "w", encoding="utf-8") as f:
        f.write(spec_text)

    try:
        result = run_pipeline(filepath=tmp_spec_file, filename="test_matrix_spec.md")
        assert result.efs_ir is not None
        assert result.req_ir is not None

        full_json_str = str(result.to_full_json())
        assert "amba_axi_protocol_spec" not in full_json_str, "Foreign document amba_axi_protocol_spec.docx detected!"

        report = generate_pipeline_validation_report(result.req_ir, result.efs_ir)
        assert report["contamination"]["foreign_objects"] == 0, f"Foreign objects detected: {report['contamination']['foreign_sources']}"
        assert report["validation"]["status"] == "VALID"
        assert report["source_validation"]["foreign_documents_detected"] is False

    finally:
        if os.path.exists(tmp_spec_file):
            os.remove(tmp_spec_file)
