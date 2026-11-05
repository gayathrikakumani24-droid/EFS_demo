import sys
sys.path.insert(0, "hardware_graphrag")
import json
from core.efs_ir.models import EFSIR
from core.retrieval.context_compiler import compile_task_context
from core.agents.rtl_agent import generate_code

d = json.load(open("data/efs_ir/active_design_ir.json"))
ir = EFSIR.from_dict(d)

print("--- 1. Testing compile_task_context('InstructionDecoder') ---")
query = "InstructionDecoder"
cp, res = compile_task_context(query, ir)

print(f"Resolution Status: {res.status}")
print(f"Resolved Target Component: {res.target_component.name if res.target_component else None}")
print(f"Match Method: {res.match_method}")
stats = cp.to_dict().get("stats", {})
print(f"Selected EFS Objects: {stats.get('selected_efs_objects', 0)} / {stats.get('total_full_efs_objects', 0)}")
print(f"Est. ContextPack Tokens: {stats.get('estimated_tokens', 0)}")

assert res.is_success(), f"Expected success but got {res.status}"
assert stats.get("selected_efs_objects", 0) > 0, "Expected selected EFS objects > 0"
assert stats.get("estimated_tokens", 0) > 0, "Expected estimated tokens > 0"

print("\n--- 2. Testing generate_code with bypass_conflicts=True ---")
code, report = generate_code(
    design_spec="Generate Verilog RTL for InstructionDecoder",
    target="Verilog",
    efs_ir=ir,
    context_pack=cp,
    generation_mode="DETERMINISTIC",
    bypass_conflicts=True,
    return_report=True
)

print(f"Generated Code Length: {len(code)} characters")
print(f"Module Name present: {'module instruction_decoder' in code}")
print(f"Top-level Matrix System present (should be False): {'multi_core_matrix_system' in code}")
print(f"AXI leakage present (should be False): {'axi_' in code.lower()}")
print(f"Partial Generation present (should be False): {'PARTIAL GENERATION' in code}")

if report:
    print(f"Candidate Verification: {report.get('candidate_verification_passed')}")
    print(f"Compliance Score: {report.get('final_compliance_score')}")

assert "module instruction_decoder" in code, f"Expected module instruction_decoder but got: {code[:200]}"
assert "multi_core_matrix_system" not in code, "Should NOT generate multi_core_matrix_system"
assert "axi_" not in code.lower(), "Should NOT have AXI signals"
assert "PARTIAL GENERATION" not in code, "Should NOT have PARTIAL GENERATION tag"

print("\n=== FULL END-TO-END FLOW VERIFICATION SUCCESSFUL! ===")
print("Generated RTL Preview:")
print("-" * 50)
print(code[:800])
print("...")
