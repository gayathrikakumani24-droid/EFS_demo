import sys
import os
sys.path.insert(0, os.path.abspath('hardware_graphrag'))
import docx
from core.agents.orchestrator import run_design_flow, resume_design_flow_with_resolutions

doc = docx.Document(r'C:\Users\Acer\.gemini\antigravity\brain\354eb1b4-323f-49aa-941e-0c1b56321975\.user_uploaded\media_1793678436813_842dda95.docx')
full_text = '\n'.join([p.text for p in doc.paragraphs])
for t in doc.tables:
    for row in t.rows:
        full_text += '\n' + '\t'.join([c.text for c in row.cells])

print("Specification text length:", len(full_text))

# Run initial flow
res = run_design_flow(
    full_text,
    target_outputs=['Verilog'],
    user_query='Generate Verilog RTL for matrix compute unit',
    bypass_conflicts=False,
    generation_mode='HYBRID',
    k_candidates=1,
    allow_partial=True
)

print("Flow Status:", res.get("status"))
print("Architecture Classification:", res.get("architecture_classification"))
print("Conflicts count:", len(res.get("conflicts", [])))
for c in res.get("conflicts", []):
    print(" - Conflict:", getattr(c, "conflict_id", ""), getattr(c, "type", ""), getattr(c, "title", ""), getattr(c, "description", ""))

if res.get("status") == "AWAITING_HUMAN_DECISION":
    print("\nSimulating resolution approval as in Streamlit UI...")
    resolutions = []
    efs_ir = res.get("efs_ir")
    proposals = res.get("proposals", {})
    for cid, props in proposals.items():
        if props:
            p = props[0]
            p.approval_status = "APPROVED"
            resolutions.append(p)
            print(" Approved proposal for", cid, ":", p.proposed_change)
    
    res2 = resume_design_flow_with_resolutions(
        res,
        resolutions,
        target_outputs=['Verilog'],
        user_query='Generate Verilog RTL for matrix compute unit',
        bypass_conflicts=False,
        generation_mode='HYBRID',
        k_candidates=1,
        allow_partial=True
    )
    print("\nResume Flow Status:", res2.get("status"))
    print("Resume Architecture Classification:", res2.get("architecture_classification"))
    print("Outputs keys:", list(res2.get("outputs", {}).keys()))
    if "Verilog" in res2.get("outputs", {}):
        v_out = res2["outputs"]["Verilog"]
        print("Verilog Score:", v_out.get("compliance_score"))
        print("Verilog Readiness:", v_out.get("readiness_status"))
        print("Report:", v_out.get("report"))
        print("Failed checks:", v_out.get("verification_report", {}).get("failed_checks", []))
        print("Generated Code snippet:")
        code = v_out.get("final_code", "")
        lines = code.split("\n")
        print("\n".join(lines[:60]))
        print("...")
        print("\n".join(lines[-40:]))
