import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath("."))
os.environ["USE_HASHING_EMBEDDER"] = "1"

from core.agents.orchestrator import run_design_flow, resume_design_flow_with_resolutions

spec = """Design a simple synchronous counter controller with clk, rst_n, start, busy, done, count[7:0].
clk is an input clock signal. rst_n is an active-low reset signal.
start is an input signal that triggers counting.
busy is an output signal indicating counting or done.
done is an output signal asserted when count reaches target.
count[7:0] is an 8-bit counter output.
The controller contains an FSM with IDLE, COUNTING, and DONE states.
IDLE transitions to COUNTING when start == 1.
COUNTING transitions to DONE when count == 10.
After one clock cycle in DONE, return to IDLE."""

print("--- RUNNING DESIGN FLOW ---")
results = run_design_flow(
    spec,
    target_outputs=["Verilog"],
    user_query="Generate synthesizable Verilog RTL for Counter_Controller"
)

status = results.get("status")
print(f"Status after initial run: {status}")

if status == "AWAITING_HUMAN_DECISION":
    print(f"Conflicts ({len(results.get('conflicts', []))}):")
    for c in results.get("conflicts", []):
        print(f" - {c.title} (blocking={c.blocking}, requires_human={getattr(c, 'requires_human_decision', None)})")
    
    print("\n--- RESUMING WITH EMPTY RESOLUTIONS (ALL RESOLVED / NOT REQUIRING HUMAN) ---")
    res_results = resume_design_flow_with_resolutions(
        results,
        [],
        target_outputs=["Verilog"]
    )
    print(f"Status after resume: {res_results.get('status')}")
    if "outputs" in res_results and "Verilog" in res_results["outputs"]:
        print("Verilog code generated successfully!")
        print(res_results["outputs"]["Verilog"].get("final_code")[:200])
elif status == "SUCCESS":
    print("Initial run generated RTL directly without human decision pause!")
    if "outputs" in results and "Verilog" in results["outputs"]:
        print(results["outputs"]["Verilog"].get("final_code")[:200])
else:
    print(f"Unexpected status: {status}, error: {results.get('error_message')}")
