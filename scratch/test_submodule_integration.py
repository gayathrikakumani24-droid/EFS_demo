"""Integration test for Sub-Module Contract Derivation Rule."""

import json
from core.context_compiler.rtl_context import RTLContextCompiler
from core.efs_ir.models import EFSIR, EFSSignal, EFSComponent, EFSMetadata

def test_decoder_submodule_derivation():
    spec_text = '''
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
'''

    top_comp = EFSComponent(component_id="COMP_TOP", name="multi_core_matrix_system", type="top_level_design")
    dec_comp = EFSComponent(component_id="COMP_DEC", name="instruction_decoder", type="sub_module")

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

    efs_ir = EFSIR(
        metadata=EFSMetadata(top_module="multi_core_matrix_system", design_name="multi_core_matrix_system"),
        components=[top_comp, dec_comp],
        signals=signals
    )

    compiler = RTLContextCompiler()
    result = compiler.compile(
        efs_ir=efs_ir,
        target_component="instruction_decoder",
        design_spec=spec_text
    )

    ports = result.get("ports", [])
    port_names = [p["name"] for p in ports]
    print(f"Instruction Decoder compiled ports count: {len(ports)}")
    for p in ports:
        print(f"  {p['direction']:6s} {p['name']:28s} [{p['width']}-1:0]")

    # Verify no bus signals leaked into decoder
    assert "axi_wdata" not in port_names, "FAIL: axi_wdata leaked into instruction_decoder!"
    assert "axi_araddr" not in port_names, "FAIL: axi_araddr leaked into instruction_decoder!"

    # Verify all decoded fields are present
    assert "opcode" in port_names, "FAIL: opcode output missing!"
    assert "dest_reg" in port_names, "FAIL: dest_reg output missing!"
    assert "src_reg_1" in port_names, "FAIL: src_reg_1 output missing!"
    assert "rows_source_1" in port_names, "FAIL: rows_source_1 output missing!"
    assert "instruction" in port_names, "FAIL: instruction input missing!"
    print("[PASS] Decoder sub-module correctly derived scoped ports and isolated AXI bus.")


def test_top_level_backward_compatibility():
    top_comp = EFSComponent(component_id="COMP_TOP", name="traffic_light_controller", type="top_level_design")
    signals = [
        EFSSignal(name="clk", direction="input", width="1", semantic_role="clock"),
        EFSSignal(name="rst_n", direction="input", width="1", semantic_role="reset"),
        EFSSignal(name="red_light", direction="output", width="1", semantic_role="data"),
        EFSSignal(name="green_light", direction="output", width="1", semantic_role="data"),
        EFSSignal(name="yellow_light", direction="output", width="1", semantic_role="data"),
    ]
    efs_ir = EFSIR(
        metadata=EFSMetadata(top_module="traffic_light_controller", design_name="traffic_light_controller"),
        components=[top_comp],
        signals=signals
    )

    compiler = RTLContextCompiler()
    result = compiler.compile(
        efs_ir=efs_ir,
        target_component="traffic_light_controller",
        design_spec="Traffic light controller module specification."
    )

    ports = result.get("ports", [])
    port_names = [p["name"] for p in ports]
    print(f"Top-level Traffic Light Controller ports: {port_names}")
    assert "red_light" in port_names
    assert "green_light" in port_names
    assert "yellow_light" in port_names
    print("[PASS] Top-level designs retain 100% backward compatibility.")


if __name__ == "__main__":
    test_decoder_submodule_derivation()
    test_top_level_backward_compatibility()
    print("\nALL INTEGRATION CHECKS PASSED!")

