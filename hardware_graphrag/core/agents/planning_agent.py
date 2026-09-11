"""
Planning Agent.

Creates a high-level implementation plan specifying modules, clock domains,
registers, memories, and finite state machines based on the Requirement Model.
"""

from __future__ import annotations

import json
from typing import Any, Dict
from config import CONFIG
from core.extraction.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("agents.planning")

SYSTEM_PROMPT = """You are a Lead Hardware Architect.
Your task is to create a detailed Hardware Implementation Plan based on a structured Requirement Model.

Determine:
1. Submodule Hierarchy: Internal blocks needed to build this module (e.g. control block, data path, register file, FIFO).
2. Clock and Reset Domains: Clock signals, resets, sync/async, clock crossings if any.
3. Bus & Interface Architecture: How external bus signals relate to internal block registers (e.g. register slice, slave interface).
4. FSM Design: Precise state transition behaviors, control outputs associated with states.
5. Internal Registers and Memories: Deep-dive layout of register map offsets, status flags, and FIFO buffers.

You must return a valid JSON object matching the following structure:
{
  "system_architecture": "A brief overview of the architectural design",
  "submodules": [
    {
      "name": "string",
      "purpose": "string",
      "interfaces": ["string"]
    }
  ],
  "clock_domains": [
    {
      "name": "string",
      "reset_strategy": "string",
      "description": "string"
    }
  ],
  "interface_mapping": {
    "external_bus": "string",
    "internal_bus": "string",
    "description": "string"
  },
  "fsm_architecture": {
    "states": [
      {
        "name": "string",
        "actions": "string"
      }
    ],
    "description": "string"
  },
  "registers_summary": "A summary of register offsets and access paths",
  "memories_summary": "A summary of internal memories, FIFOs, sizes"
}
Do not add any markdown formatting or explanation outside the JSON object. Keep the output strictly conforming to the JSON schema.
"""


def create_plan(requirement_model: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a detailed architectural implementation plan from the requirement model."""
    logger.info("Generating system architecture plan...")
    llm = get_llm_client()
    
    req_json = json.dumps(requirement_model, indent=2)
    
    if llm.available:
        result = llm.complete_json(SYSTEM_PROMPT, req_json, model=CONFIG.llm.extraction_model)
        if result:
            return result
        logger.warning("LLM failed to return a valid JSON plan. Falling back to heuristic planning.")

    # Fallback planning
    return get_heuristic_plan(requirement_model)


def get_heuristic_plan(req_model: Dict[str, Any]) -> Dict[str, Any]:
    """Heuristic planning fallback when LLM is unavailable."""
    interfaces = [i.get("name", "signal") for i in req_model.get("interfaces", []) if isinstance(i, dict)]
    submodules = []
    
    # Include extracted submodules from requirement model
    for sub in req_model.get("submodules", []):
        if isinstance(sub, dict):
            submodules.append({
                "name": sub.get("name", "Submodule"),
                "purpose": sub.get("purpose", "Submodule purpose"),
                "interfaces": interfaces
            })

    # Default structural modules
    defaults = [
        {"name": "Register_File", "purpose": "Holds control/status registers.", "interfaces": interfaces},
        {"name": "FSM_Control_Unit", "purpose": "Coordinates handshakes and state logic.", "interfaces": []},
    ]
    for d in defaults:
        if not any(s["name"] == d["name"] for s in submodules):
            submodules.append(d)

    if req_model.get("memories"):
        submodules.append({"name": "Data_Buffer_Memory", "purpose": "Buffers internal data.", "interfaces": []})

    fsm_states = req_model.get("fsm_info", {}).get("states", []) if isinstance(req_model.get("fsm_info"), dict) else []
        
    return {
        "system_architecture": "Standard register-mapped peripheral controller.",
        "submodules": submodules,
        "clock_domains": [
            {"name": "clk", "reset_strategy": "Asynchronous active-low reset", "description": "System-wide clock domain."}
        ],
        "interface_mapping": {
            "external_bus": ", ".join(req_model.get("protocol_references", ["General"])),
            "internal_bus": "Local Register Bus",
            "description": "Bridges external protocol transfers directly to the internal register file."
        },
        "fsm_architecture": {
            "states": [
                {"name": s, "actions": f"Active control signals in state {s}"}
                for s in fsm_states
            ],
            "description": "Orchestrates operational flow in response to control register toggles."
        },
        "registers_summary": f"Contains control/status registers: {', '.join([r.get('name', 'REG') for r in req_model.get('registers', []) if isinstance(r, dict)])}",
        "memories_summary": "FIFO buffers and local memories configured as specified in requirements."
    }
