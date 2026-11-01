# Generic Verilog & SystemVerilog RTL Generation, Validation & Repair Architecture

This document describes the improved EFS RTL generation pipeline supporting standards-compliant, specification-driven hardware synthesis in both **Verilog** (Verilog-1995/2001/2005) and **SystemVerilog** (SystemVerilog-2005/2009/2012/2017).

---

## 1. Architecture Overview

The RTL generation pipeline uses the **Canonical EFS Hardware Intermediate Representation (EFS IR)** as the single source of truth. It strictly enforces non-hardcoding rules:
- No hardcoded module names, signal names, opcodes, registers, or FSM states.
- Pure combinational designs synthesize without fabricated clocks or resets.
- Sequential and FSM designs extract clocks, resets, states, and transitions dynamically from EFS IR.

```
                    ┌───────────────────────────┐
                    │      EFS Hardware IR      │
                    │ (Single Source of Truth)  │
                    └─────────────┬─────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
       ┌───────────────────────┐     ┌───────────────────────┐
       │     Verilog Mode      │     │  SystemVerilog Mode   │
       │  (Verilog-2001/2005)  │     │ (SystemVerilog-2012)  │
       │  - wire / reg         │     │  - logic / bit        │
       │  - assign / always    │     │  - always_ff / _comb  │
       │  - localparam         │     │  - typedef enum logic │
       └──────────┬────────────┘     └───────────┬───────────┘
                  │                               │
                  └───────────────┬───────────────┘
                                  ▼
                   ┌─────────────────────────────┐
                   │ Language-Aware Validation   │
                   │ - No SV-keywords in Verilog │
                   │ - Unclosed blocks & vector  │
                   │ - External tool integration │
                   │   (iverilog, slang, etc.)   │
                   └──────────────┬──────────────┘
                                  │
                  ┌───────────────┴───────────────┐
        [Pass]    ▼                     [Fail]    ▼
┌─────────────────────────────┐     ┌─────────────────────────────┐
│ Requirement Traceability    │     │ Targeted Error Repair       │
│ - Port & Register Mapping   │     │ - Preserves Target HDL      │
│ - FSM State Verification    │     │ - Surgical Code Correction  │
│ - Unmapped Element Alert    │     │ - HITL Ambiguity Resolution │
└─────────────────────────────┘     └─────────────────────────────┘
```

---

## 2. HDL Language Selection

Language selection is configured via the target argument in the orchestrator or generation request:

| Target Option | Resolved Standard | Allowed Constructs | Prohibited Constructs |
| :--- | :--- | :--- | :--- |
| `"Verilog"` / `"Verilog-2001"` | IEEE 1364-2001 | `wire`, `reg`, `always @(*)`, `always @(posedge ...)`, `localparam` | `logic`, `always_ff`, `always_comb`, `typedef`, `enum` |
| `"Verilog-2005"` | IEEE 1364-2005 | Verilog-2001 + `uwire` | SystemVerilog keywords |
| `"SystemVerilog"` / `"sv"` | IEEE 1800-2012 | `logic`, `always_ff`, `always_comb`, `typedef enum`, structs | Verilog deprecated reg-in-always idioms |
| `"SystemVerilog-2017"` | IEEE 1800-2017 | Full IEEE 1800-2017 features | Incompatible legacy syntax |

Ambiguous or unsupported target languages (e.g. Chisel or Bluespec without adapter) raise a structured error requesting clarification rather than silently converting languages.

---

## 3. Language-Aware RTL Validation

Validation operates at two complementary levels:

### Level 1: Deterministic Static & Structural Validation (`hdl_validator.py`)
1. **Forbidden Keyword Detection:** Flags any SystemVerilog-only keyword (`logic`, `always_ff`, `always_comb`, `typedef`, etc.) if Verilog mode is active (`RULE_VERILOG_FORBIDDEN_SV_KEYWORD`).
2. **Procedural Sensitivity Checks:** Verifies sequential sensitivity (`posedge clk or negedge rst_n`) and combinational sensitivity (`@(*)` for Verilog, `always_comb` for SV).
3. **Module & Block Boundaries:** Detects unclosed `module...endmodule`, `begin...end`, `case...endcase`, and unbalanced vector brackets `[...]`.
4. **Placeholder Detection:** Flags lazy placeholder comments (`// TODO`, `// Active low reset`, `// Combinational logic`) standing in for hardware logic.
5. **Dynamic Compiler Invocation:** Automatically invokes external EDA compilers (`iverilog`, `verilator`, `slang`) when available in `PATH` using language-specific flags (`-g2005` for Verilog vs `-g2012` for SystemVerilog).

### Level 2: Requirement-to-RTL Traceability (`traceability.py`)
Maps each EFS IR requirement (signal, register, FSM state, transition, clock, reset) to exact RTL statements:
- **Coverage Metric:** Percentage of specified requirements implemented in RTL.
- **Unmapped Element Detection:** Flags signals, wires, or registers in RTL that have no supporting specification requirement, mitigating hallucinated/fabricated hardware.

---

## 4. Targeted RTL Error Detection and Repair

When validation detects syntax, structural, or protocol compliance violations:
1. Violations are traced back to the offending lines and the corresponding EFS IR entities.
2. The repair engine (`repair_agent.py`) receives language-specific instructions to preserve the target HDL (e.g. forbidding the injection of `logic` when repairing Verilog).
3. If an error originates from conflicting or incomplete EFS IR, the system pauses the affected module and raises a structured Human-in-the-Loop clarification request.
