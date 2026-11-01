import sys
import os
import json

sys.path.insert(0, os.path.abspath('.'))
from core.requirement_ir.models import DocumentIR, DocumentBlock, SourceLocation
from core.agents.requirement_agent import analyze_requirements

text = """Design a Custom Acceleration Controller system with two modules: AccelerationController and LocalMemory.

IDLE -> EXECUTE: start is asserted
EXECUTE -> DONE: execution completes
DONE -> IDLE: the next clock cycle

Memory Interface:
- AccelerationController connects to LocalMemory for data access
- AccelerationController sends mem_addr, mem_wdata, and mem_we to LocalMemory
- LocalMemory returns mem_rdata to AccelerationController
"""

req_model = analyze_requirements(text)

print("=== REQ_MODEL INTERFACES ===")
for i in req_model.get("interfaces", []):
    print("Interface:", i)

print("\n=== REQ_MODEL SIGNALS ===")
for s in req_model.get("signals", []):
    print("Signal:", s)

print("\n=== REQ_MODEL ENTITIES ===")
for e in req_model.get("entities", []):
    print(f"Entity: name='{e.get('name')}', type='{e.get('type')}', desc='{e.get('description')}'")
