"""
Tests for RTL Agent Phase 1 Defect Fixes:
1. Port-direction reconciliation variable binding and resolved branch.
2. Template reachability and clean execution without undefined symbols.
3. Typing imports and single definition of _derive_register_fields.
4. Honest target dispatch returning UNSUPPORTED_TARGET for VHDL/UVM/Assertions.
"""

import pytest
from core.agents import rtl_agent
from core.agents.rtl_agent import (
    plan_rtl,
    get_heuristic_code_template,
    get_system_prompt,
    generate_code,
    _derive_register_fields,
)
from core.efs_ir.models import (
    EFSIR,
    EFSSignal,
    EFSResolution,
    EFSConflict,
    EFSMetadata,
)


def test_01_port_direction_reconciliation_resolved_branch():
    """
    Defect 1 Regression Test:
    When port direction conflict is resolved by resolve_signal_metadata or EFSIR resolutions,
    'is_resolved' must be properly set, avoiding UnboundLocalError or stale variable values.
    """
    # Create EFSIR with signal 'data_out' marked as input in EFSIR signals
    ir = EFSIR(
        metadata=EFSMetadata(design_name="direction_test_mod"),
        signals=[
            EFSSignal(
                name="data_out",
                direction="input",
                width="8",
                description="Output data bus driven by module",
            )
        ]
    )

    # Design spec specifies data_out as output (matching its description/name semantics)
    spec = """# Hardware Design Specification: direction_test_mod
## Interfaces
| Signal Name | Direction | Width | Description |
| data_out | output | 8 | Output data bus driven by module |
"""

    # plan_rtl should reconcile port direction and resolve without UnboundLocalError
    plan = plan_rtl(spec, target="SystemVerilog", efs_ir=ir)

    # Check that data_out exists in plan ports
    port_map = {p["name"]: p for p in plan.get("ports", [])}
    assert "data_out" in port_map

    # Ensure no blocking issue was created for data_out since resolution succeeded
    status = plan.get("specification_status", {})
    blocking_issues = status.get("blocking_issues", [])
    data_out_direction_issues = [
        b for b in blocking_issues
        if "data_out" in str(b.get("issue", "")) and "direction" in str(b.get("issue", "")).lower()
    ]
    assert len(data_out_direction_issues) == 0, f"Expected data_out direction conflict to be resolved, but found issues: {data_out_direction_issues}"


def test_02_port_direction_reconciliation_approved_resolution():
    """
    Defect 1 Regression Test:
    When an explicit approved EFSResolution exists for the signal direction,
    plan_rtl recognizes is_resolved = True and does not flag CONTRADICTORY.
    """
    ir = EFSIR(
        metadata=EFSMetadata(design_name="res_test_mod"),
        signals=[
            EFSSignal(
                name="status_flag",
                direction="input",
                width="1",
                description="Status flag signal",
            )
        ],
        resolutions=[
            EFSResolution(
                resolution_id="RES_01",
                conflict_id="CONF_01",
                target_entity="status_flag",
                affected_requirements=["status_flag"],
                property="direction",
                approval_status="APPROVED",
                proposed_change="Resolve status_flag direction to output",
            )
        ]
    )

    spec = """# Hardware Design Specification: res_test_mod
## Interfaces
| Signal Name | Direction | Width | Description |
| status_flag | output | 1 | Status flag signal |
"""

    plan = plan_rtl(spec, target="Verilog", efs_ir=ir)
    status = plan.get("specification_status", {})
    blocking_issues = status.get("blocking_issues", [])
    flag_issues = [b for b in blocking_issues if "status_flag" in str(b.get("issue", ""))]
    assert len(flag_issues) == 0


def test_02b_port_direction_reconciliation_resolved_conflict():
    """
    Defect 1 Regression Test:
    When a resolved EFSConflict exists in efs_ir.conflicts, is_resolved is True.
    """
    ir = EFSIR(
        metadata=EFSMetadata(design_name="conf_test_mod"),
        signals=[
            EFSSignal(
                name="ack_sig",
                direction="input",
                width="1",
                description="Acknowledge signal",
            )
        ],
        conflicts=[
            EFSConflict(
                conflict_id="CONF_ACK",
                entities=["ack_sig"],
                property="direction",
                resolution_status="RESOLVED",
            )
        ]
    )

    spec = """# Hardware Design Specification: conf_test_mod
## Interfaces
| Signal Name | Direction | Width | Description |
| ack_sig | output | 1 | Acknowledge signal |
"""

    plan = plan_rtl(spec, target="Verilog", efs_ir=ir)
    status = plan.get("specification_status", {})
    blocking_issues = status.get("blocking_issues", [])
    ack_issues = [b for b in blocking_issues if "ack_sig" in str(b.get("issue", ""))]
    assert len(ack_issues) == 0


def test_02c_get_heuristic_code_template_clean_execution():
    """
    Defect 2 Regression Test:
    Ensure get_heuristic_code_template executes without referencing undefined variables.
    """
    spec = """# Hardware Design Specification: simple_counter
## Interfaces
| Signal Name | Direction | Width | Description |
| clk | input | 1 | Clock signal |
| rst_n | input | 1 | Active-low reset |
| count | output | 8 | Counter output |
"""
    plan = {
        "module_name": "simple_counter",
        "specification_status": {"status": "COMPLETE", "blocking_issues": []},
        "ports": [
            {"name": "clk", "direction": "input", "width": "1", "semantic_role": "clock"},
            {"name": "rst_n", "direction": "input", "width": "1", "semantic_role": "reset"},
            {"name": "count", "direction": "output", "width": "8"},
        ],
        "output_driving_strategy": {"count": "8'h0"}
    }
    v_code = get_heuristic_code_template("Verilog", spec, rtl_plan=plan)
    assert "module simple_counter" in v_code
    assert "endmodule" in v_code
    import re
    code_no_comments = re.sub(r"//.*", "", v_code)
    assert not re.search(r"\blogic\b", code_no_comments)

    sv_code = get_heuristic_code_template("SystemVerilog", spec, rtl_plan=plan)
    assert "module simple_counter" in sv_code
    assert "endmodule" in sv_code


def test_03_no_duplicate_derive_register_fields():
    """
    Defect 3: Verify _derive_register_fields is defined once and callable.
    """
    # Verify module has _derive_register_fields
    assert hasattr(rtl_agent, "_derive_register_fields")

    # Test explicit field derivation
    spec_with_fields = "Register [3:0]: FLAG_BITS, [7:4]: MODE_BITS"
    fields = _derive_register_fields("custom_reg", spec_with_fields)
    assert len(fields) == 2
    assert fields[0]["meaning"] == "FLAG_BITS"
    assert fields[0]["classification"] == "EXPLICIT"

    # Test derived ctrl fields
    ctrl_fields = _derive_register_fields("main_ctrl", "")
    assert any(f["meaning"] == "ENABLE" for f in ctrl_fields)

    # Test derived status fields
    status_fields = _derive_register_fields("main_status", "")
    assert any(f["meaning"] == "DONE" for f in status_fields)


def test_04_typing_imports_present():
    """
    Defect 3: Verify Set and Tuple are properly imported and present.
    """
    assert hasattr(rtl_agent, "Set")
    assert hasattr(rtl_agent, "Tuple")
    assert hasattr(rtl_agent, "List")
    assert hasattr(rtl_agent, "Dict")
    assert hasattr(rtl_agent, "Optional")


def test_05_target_dispatch_unsupported_vhdl_uvm_assertions():
    """
    Defect 4: Calling generate_code with VHDL, UVM, Assertions, or unknown target
    returns an explicit UNSUPPORTED_TARGET response.
    """
    spec = "# Hardware Design Specification: simple_mod\n"

    for unsupported_target in ["VHDL", "vhdl", "UVM", "uvm", "Assertions", "assertions", "SystemC", "Chisel"]:
        res = generate_code(spec, target=unsupported_target)
        assert res.startswith("UNSUPPORTED_TARGET"), f"Target '{unsupported_target}' should return UNSUPPORTED_TARGET but returned: {res[:50]}"
        assert "LANGUAGE_UNSUPPORTED" in res
        assert "Verilog" in res
        assert "SystemVerilog" in res


def test_06_get_system_prompt_target_handling():
    """
    Defect 4: get_system_prompt validates target language and raises ValueError for unsupported targets.
    """
    # Supported: Verilog
    prompt_v = get_system_prompt("Verilog")
    assert "standards-compliant Verilog" in prompt_v
    assert "endmodule" in prompt_v
    assert "NO `logic`" in prompt_v

    # Supported: SystemVerilog
    prompt_sv = get_system_prompt("SystemVerilog")
    assert "standards-compliant SystemVerilog" in prompt_sv
    assert "always_ff" in prompt_sv

    # Unsupported: VHDL or other targets must raise ValueError
    with pytest.raises(ValueError, match="Unsupported HDL target language"):
        get_system_prompt("VHDL")

    with pytest.raises(ValueError, match="Unsupported HDL target language"):
        get_system_prompt("UVM")

    with pytest.raises(ValueError, match="Unsupported HDL target language"):
        get_system_prompt("Assertions")


def test_07_docstrings_honest_about_supported_targets():
    """
    Defect 4: Ensure module and generate_code docstrings match behaviour.
    """
    module_doc = rtl_agent.__doc__ or ""
    assert "unsupported" in module_doc.lower() or "verilog" in module_doc.lower()
    assert "vhdl" not in module_doc.lower() or "unsupported" in module_doc.lower()

    gen_doc = generate_code.__doc__ or ""
    assert "Verilog" in gen_doc
    assert "SystemVerilog" in gen_doc
    assert "unsupported" in gen_doc.lower()
