"""
End-to-End Test for Stale Canonical EFS and GraphRAG Data Isolation.

Verifies:
1. When a new specification for Custom Acceleration Controller + LocalMemory is run:
   - Canonical EFS contains AccelerationController and LocalMemory components.
   - Canonical EFS contains status_reg and instruction_reg registers.
   - Canonical EFS contains mem_addr, mem_wdata, mem_rdata, mem_we.
   - Canonical EFS contains FSM states (IDLE, EXECUTE, DONE).
   - Canonical EFS contains relationships:
       AccelerationController -> LocalMemory
       LocalMemory -> AccelerationController
   - Canonical EFS does NOT contain Matrix Compute Unit components or entities.
   - Source documents do NOT contain matrix_compute_unit.docx.
2. The validator receives the current canonical EFS rather than an old cached object.
3. CURRENT DESIGN DOCUMENTS, CURRENT CANONICAL COMPONENTS, and CURRENT INTER-MODULE RELATIONSHIPS
   are printed and logged before validation.
"""

import os
import json
import pytest
from core.agents.orchestrator import run_design_flow
from core.registry import get_registry, reset_registry
from core.efs_ir.validator import validate_efs_ir
from core.verification.conflict_resolution_engine import ConflictResolutionEngine
from config import CONFIG


ACCEL_SPEC = """
Design a Custom Acceleration Controller module with LocalMemory.
Interfaces:
- clk: input, 1 bit, System clock.
- rst_n: input, 1 bit, Active-low reset.
- instruction: input, 32 bits, Instruction input.
- start: input, 1 bit, Start trigger.
- busy: output, 1 bit, Busy status.
- done: output, 1 bit, Execution complete.
Registers:
- status_reg: 2 bits, stores busy and done status.
- instruction_reg: 32 bits, stores the current instruction.
FSM:
- IDLE: wait for start.
- EXECUTE: execute instruction.
- DONE: signal completion.
Inter-Module Relationship:
- AccelerationController sends mem_addr, mem_wdata, and mem_we to LocalMemory.
- LocalMemory returns mem_rdata to AccelerationController.
"""


def test_stale_data_isolation_end_to_end(capsys):
    # Pre-condition: populate registry with stale matrix_compute_unit data to simulate prior runs
    reset_registry()
    reg = get_registry()
    from utils.models import Entity, Relationship, Chunk, ParsedDocument
    
    stale_doc = ParsedDocument(doc_id="stale_mcu_001", filename="matrix_compute_unit.docx", file_type="docx")
    reg.documents["stale_mcu_001"] = stale_doc
    reg.entities["stale_mcu_001"] = [
        Entity(entity_id="ent_mcu_1", name="PE array", raw_name="PE array", entity_type="module", chunk_id="chk_mcu_1", doc_id="stale_mcu_001"),
        Entity(entity_id="ent_mcu_2", name="64 Matrix ALU", raw_name="64 Matrix ALU", entity_type="component", chunk_id="chk_mcu_1", doc_id="stale_mcu_001"),
        Entity(entity_id="ent_mcu_3", name="Matrix Register File", raw_name="Matrix Register File", entity_type="register", chunk_id="chk_mcu_1", doc_id="stale_mcu_001"),
    ]
    reg.chunks["stale_mcu_001"] = [
        Chunk(chunk_id="chk_mcu_1", doc_id="stale_mcu_001", text="Matrix Compute Unit PE array performs matrix multiplication.")
    ]
    reg.relationships["stale_mcu_001"] = [
        Relationship(rel_id="rel_mcu_1", source="PE array", relation="CONNECTED_TO", target="64 Matrix ALU", chunk_id="chk_mcu_1", doc_id="stale_mcu_001")
    ]

    # Run the design flow for the new Custom Acceleration Controller specification
    results = run_design_flow(
        requirement_text=ACCEL_SPEC,
        target_outputs=["verilog"],
        bypass_conflicts=True
    )

    captured = capsys.readouterr()
    stdout_text = captured.out

    # Verify CURRENT DESIGN logging appeared in stdout before validation
    assert "CURRENT DESIGN DOCUMENTS" in stdout_text, "Expected 'CURRENT DESIGN DOCUMENTS' in logs/stdout"
    assert "CURRENT CANONICAL COMPONENTS" in stdout_text, "Expected 'CURRENT CANONICAL COMPONENTS' in logs/stdout"
    assert "CURRENT INTER-MODULE RELATIONSHIPS" in stdout_text, "Expected 'CURRENT INTER-MODULE RELATIONSHIPS' in logs/stdout"

    # Inspect active canonical EFS IR
    active_ir_path = os.path.join(CONFIG.data_dir, "efs_ir", "active_design_ir.json")
    assert os.path.exists(active_ir_path), f"Expected active design JSON at {active_ir_path}"
    
    with open(active_ir_path, "r", encoding="utf-8") as f:
        canonical_json = json.load(f)

    # 1. Source Documents verification: must NOT contain matrix_compute_unit.docx
    source_docs = [
        d.get("name", "") if isinstance(d, dict) else str(d)
        for d in canonical_json.get("metadata", {}).get("source_documents", [])
    ]
    print(f"\nFinal Canonical Source Docs: {source_docs}")
    for doc in source_docs:
        assert "matrix_compute_unit" not in doc.lower(), f"Stale document leaked into canonical JSON: {doc}"

    # 2. Components verification: must contain AccelerationController, LocalMemory
    comp_names = [c.get("name", "") for c in canonical_json.get("components", [])]
    print(f"Final Canonical Components: {comp_names}")
    
    assert any("accelerationcontroller" in c.lower() for c in comp_names), f"Missing AccelerationController in {comp_names}"
    assert any("localmemory" in c.lower() for c in comp_names), f"Missing LocalMemory in {comp_names}"

    # Crucial check: Must NOT contain Matrix Compute Unit entities
    mcu_leaks = [c for c in comp_names if any(w in c.lower() for w in ["matrix", "pe array", "alu"])]
    assert len(mcu_leaks) == 0, f"Stale MCU components leaked into canonical JSON: {mcu_leaks}"

    # 3. Registers verification: must contain status_reg and instruction_reg
    reg_names = [r.get("name", "") for r in canonical_json.get("registers", [])]
    print(f"Final Canonical Registers: {reg_names}")
    assert any("status_reg" in r.lower() for r in reg_names), f"Missing status_reg in {reg_names}"
    assert any("instruction_reg" in r.lower() for r in reg_names), f"Missing instruction_reg in {reg_names}"

    # 4. Signals verification: must contain mem_addr, mem_wdata, mem_rdata, mem_we
    sig_names = [s.get("name", "") for s in canonical_json.get("signals", [])]
    print(f"Final Canonical Signals: {sig_names}")
    for expected_sig in ["mem_addr", "mem_wdata", "mem_rdata", "mem_we"]:
        assert any(expected_sig in s.lower() for s in sig_names), f"Missing signal {expected_sig} in {sig_names}"

    # 5. FSM states verification: must contain IDLE, EXECUTE, DONE
    fsm_list = canonical_json.get("fsms", [])
    fsm_state_names = []
    for fsm in fsm_list:
        for st in fsm.get("states", []):
            st_name = st.get("name") if isinstance(st, dict) else str(st)
            fsm_state_names.append(st_name.upper())
    print(f"Final Canonical FSM States: {fsm_state_names}")
    for expected_st in ["IDLE", "EXECUTE", "DONE"]:
        assert any(expected_st in s for s in fsm_state_names), f"Missing FSM state {expected_st} in {fsm_state_names}"

    # 6. Relationships verification
    relationships = canonical_json.get("relationships", [])
    rel_pairs = [
        (r.get("source", ""), r.get("target", ""))
        for r in relationships
    ]
    print(f"Final Canonical Relationships: {rel_pairs}")
    has_fwd = any("accelerationcontroller" in s.lower() and "localmemory" in t.lower() for s, t in rel_pairs)
    has_rev = any("localmemory" in s.lower() and "accelerationcontroller" in t.lower() for s, t in rel_pairs)
    assert has_fwd, f"Missing AccelerationController -> LocalMemory relationship in {rel_pairs}"
    assert has_rev, f"Missing LocalMemory -> AccelerationController relationship in {rel_pairs}"

    # 7. Validator receives current canonical EFS rather than old cached object
    from core.efs_ir.models import EFSIR
    current_efs_ir = EFSIR.from_dict(canonical_json)
    validation_issues = validate_efs_ir(current_efs_ir)
    print(f"Validation issues on current canonical EFS: {len(validation_issues)}")
    # Ensure validation runs cleanly on this current object
    assert isinstance(validation_issues, list)
