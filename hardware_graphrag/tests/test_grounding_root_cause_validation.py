import pytest
from core.efs_ir.models import EFSIR, EFSComponent, EFSSignal, EFSTraceability, EFSMetadata, EFSInterface
from core.efs_ir.validator import validate_efs_ir, make_validation_issue
from core.efs_ir.builder import EFSIRBuilder

def test_1_fully_grounded_requirement():
    ir = EFSIR()
    ir.metadata = EFSMetadata(design_name="TestDesign")
    comp = EFSComponent(
        component_id="comp_1",
        name="ALU",
        traceability=EFSTraceability(doc_id="doc1", chunk_id="chunk1", original_text="ALU text", confidence=0.9)
    )
    sig = EFSSignal(
        signal_id="sig_1",
        name="clk",
        traceability=EFSTraceability(doc_id="doc1", chunk_id="chunk1", original_text="clk text", confidence=0.9)
    )
    ir.components.append(comp)
    ir.signals.append(sig)
    
    issues = validate_efs_ir(ir)
    grounding_issues = [i for i in issues if i["category"] == "Grounding"]
    assert len(grounding_issues) == 0, "Fully grounded requirement should yield zero grounding issues"

def test_2_requirement_with_missing_provenance():
    ir = EFSIR()
    ir.metadata = EFSMetadata(design_name="TestDesign")
    comp = EFSComponent(
        component_id="comp_1",
        name="UnprovenancedComp",
        traceability=EFSTraceability(doc_id=None, chunk_id=None, original_text=None, confidence=0.1)
    )
    ir.components.append(comp)
    
    issues = validate_efs_ir(ir)
    grounding_issues = [i for i in issues if i["category"] == "Grounding"]
    assert len(grounding_issues) == 1
    assert grounding_issues[0]["classification"] in ("GROUNDING_GAP", "TRACEABILITY_GAP")
    assert grounding_issues[0]["classification"] != "SOURCE_MISSING"

def test_3_requirement_missing_from_source():
    issue = make_validation_issue(
        category="Requirements",
        message="Missing reset signal requirement",
        object_id="sig_rst",
        severity="ERROR",
        classification="SOURCE_MISSING",
        evidence="Spec section 3 does not define reset behaviour",
        required_information="Provide active state for reset signal"
    )
    assert issue["classification"] == "SOURCE_MISSING"

def test_4_requirement_extracted_incorrectly():
    ir = EFSIR()
    ir.metadata = EFSMetadata(design_name="TestDesign")
    iface = EFSInterface(
        interface_id="if_1",
        name="bus_if",
        signals=["sig_nonexistent"],
        traceability=EFSTraceability(doc_id="doc1", chunk_id="chunk1", confidence=0.9)
    )
    ir.interfaces.append(iface)
    
    issues = validate_efs_ir(ir)
    gap_issues = [i for i in issues if i["classification"] == "EFSIR_EXTRACTION_GAP"]
    assert len(gap_issues) > 0

def test_5_malformed_validator_issue():
    raw_issue = make_validation_issue(
        category="Grounding",
        message="",
        object_id="obj_test"
    )
    assert raw_issue["issue"] != "" or raw_issue["classification"] == "VALIDATION_INTERNAL_ERROR"

def test_6_multiple_requirements_no_duplicate_generic_rows():
    ir = EFSIR()
    ir.metadata = EFSMetadata(design_name="MultiDesign")
    for i in range(5):
        comp = EFSComponent(
            component_id=f"comp_{i}",
            name=f"Submodule_{i}",
            traceability=EFSTraceability(doc_id="doc1", chunk_id=f"chunk_{i}", confidence=0.9)
        )
        ir.components.append(comp)
    
    issues = validate_efs_ir(ir)
    grounding_issues = [i for i in issues if i["category"] == "Grounding"]
    assert len(grounding_issues) == 0

def test_7_same_validation_executed_repeatedly():
    ir = EFSIR()
    ir.metadata = EFSMetadata(design_name="TestDesign")
    comp = EFSComponent(
        component_id="comp_1",
        name="TestComp",
        traceability=EFSTraceability(doc_id=None, chunk_id=None, confidence=0.2)
    )
    ir.components.append(comp)
    
    run1 = validate_efs_ir(ir)
    run2 = validate_efs_ir(ir)
    run3 = validate_efs_ir(ir)
    
    assert len(run1) == len(run2) == len(run3)

def test_8_streamlit_rerun_no_issue_accumulation():
    session_state = {}
    ir = EFSIR()
    ir.metadata = EFSMetadata(design_name="TestDesign")
    
    for rerun in range(3):
        session_state["efs_ir_validation_issues"] = validate_efs_ir(ir)
    
    assert len(session_state["efs_ir_validation_issues"]) == len(validate_efs_ir(ir))

def test_9_multiple_validators_no_duplicate_responsibility():
    ir = EFSIR()
    ir.metadata = EFSMetadata(design_name="TestDesign")
    comp = EFSComponent(
        component_id="comp_1",
        name="DupComp",
        traceability=EFSTraceability(doc_id="doc1", chunk_id="c1", confidence=0.9)
    )
    ir.components.append(comp)
    
    issues = validate_efs_ir(ir)
    keys = [(i["classification"], i["category"], i["issue"], i["object_id"]) for i in issues]
    assert len(keys) == len(set(keys))

def test_10_completely_different_specification():
    custom_req = {
        "design_name": "Photon_Counter_V3",
        "submodules": [{"name": "Laser_Diode_Driver", "purpose": "Pulse modulation"}],
        "clocks_resets": [{"name": "ref_clk_200mhz", "type": "clock"}]
    }
    builder = EFSIRBuilder(req_model=custom_req, design_spec="Laser diode driver specification text")
    ir = builder.build()
    issues = validate_efs_ir(ir)
    
    blocking_grounding = [i for i in issues if i["category"] == "Grounding" and i["is_blocking"]]
    assert len(blocking_grounding) == 0
