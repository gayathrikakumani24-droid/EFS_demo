# Hardware Specification GraphRAG & EFS Compiler Pipeline

A domain-generic, protocol-agnostic system that parses hardware specification documents (PDF, DOCX, Markdown), extracts atomic hardware semantics with verbatim source provenance into a canonical **EFS IR (Executable Functional Specification Intermediate Representation)**, resolves specification gaps and contradictions via Human-in-the-Loop (HITL), and compiles the verified design into **PlantUML Architecture Diagrams** and synthesis-ready **Verilog/SystemVerilog RTL**.

---

## 1. Architectural Pipeline Flow

The compilation and verification pipeline enforces strict separation between extraction, inference, validation, and human decision-making:

```
SOURCE DOCUMENT
       ↓
DOCUMENT PARSER (Section & Table Hierarchy)
       ↓
HIERARCHICAL CHUNKS & DOCUMENT IR
       ↓
ENTITY & PROPERTY EXTRACTION
       ↓
DEDICATED RELATIONSHIP EXTRACTION (14 Canonical Verbs)
       ↓
SOURCE EVIDENCE INDEX (Line-Level Provenance Store)
       ↓
EFS IR CONSTRUCTION (Canonical Hardware Model)
       ↓
SOURCE-BACKED RECOVERY (Pre-Validation Source Scan)
       ↓
COMPLETENESS VALIDATION (Internally Consistent Model Check)
       ↓
GAP CLASSIFICATION (Explicit vs. Partial vs. Inferred vs. Unresolved)
       ↓
HUMAN-IN-THE-LOOP (HITL Conflict Resolution for True Gaps)
       ↓
RESOLVED EFS IR OVERLAY (Single Source of Truth)
   ┌───┴──────────────────────┐
   ▼                          ▼
PLANTUML AGENT          RTL AGENT (Verilog/SystemVerilog)
   │                          │
Diagram Rendering       RTL Verification & Linting
```

---

## 2. Core Architectural Principles

### A. Evidence-Backed Extraction & Full Provenance
Every hardware property extracted into the EFS IR retains explicit source evidence:
```json
{
  "name": "clock",
  "value": "clk",
  "status": "explicit",
  "confidence": 1.0,
  "evidence": [
    {
      "document": "matrix_compute_unit.docx",
      "section": "Section 15: Clock & Reset",
      "text": "Clock : System clock input",
      "line_start": 15,
      "line_end": 15,
      "block_id": "BLK_015",
      "evidence_type": "text"
    }
  ]
}
```
* **Property Statuses**: `explicit` | `partial` | `inferred` | `unresolved` | `conflicting`.
* No important EFS property is permitted to exist without provenance whenever source evidence is available.

### B. Separation of Responsibilities
* **Extraction**: *"What does the source specification explicitly say?"*
* **Inference**: *"What can reasonably be derived with traceable evidence?"*
* **Validation**: *"Is the resulting EFS IR internally complete, consistent, and synthesizable?"*
* **HITL (Human-in-the-Loop)**: *"What genuinely cannot be determined from the source document?"*

### C. Source-Backed Recovery Before Validation
Before generating any "missing property" conflict, the engine automatically queries the `SourceEvidenceIndex`.
* If explicit source evidence exists $\rightarrow$ recovers the property into EFS IR (`status="explicit"`).
* If partial source information exists $\rightarrow$ records available data (`status="partial"`).
* The completeness validator **never** treats *"property missing from initial EFS extraction"* as equivalent to *"property missing from the source specification"*.

### D. Anti-Hallucination Guard
AI-assisted conflict proposals must satisfy two strict conditions before acceptance:
1. `proposal_has_source_evidence == True`
2. `proposal_value_is_supported_by_evidence == True`

If either check fails, the proposal is rejected as `unsupported_ai_inference`. The system **never** applies:
* Invented register addresses (e.g. fake `0x00`, `0x04`, `0x08`)
* Invented FSM transitions (e.g. fake `ACK` states when `ACK` is absent from source)
* Invented clock periods (e.g. guessing `10ns / 100MHz`)
* Invented interfaces or dummy signal connections

---

## 3. Dedicated Extractors

The extraction subsystem includes five specialized, protocol-agnostic extractors:

1. **`DedicatedRelationshipExtractor`**: Detects 14 canonical hardware relationship verbs across descriptions, signal tables, and crossbar architectures:
   * `connects_to`, `controls`, `reads_from`, `writes_to`, `sends_to`, `receives_from`, `contains`, `drives`, `feeds`, `schedules`, `distributes_to`, `communicates_with`, `interfaces_with`, `depends_on`.
2. **`DedicatedInterfaceExtractor`**: Extracts interface names, bus protocols (AXI, APB, AHB, custom), directions, signal bundles, and connected components.
3. **`DedicatedRegisterExtractor`**: Distinguishes between explicit register definitions with hex addresses and register categories where offsets are unassigned (e.g., `"Matrix Registers M0-M7"` with `address=None`). Does not hallucinate default offsets.
4. **`DedicatedFSMExtractor`**: Identifies states, transitions, trigger conditions, loops, and reset states. Differentiates explicit FSMs from partial FSMs derived from execution control signals (`start`, `busy`, `done`, `error`).
5. **`DedicatedConstraintExtractor`**: Extracts hardware performance constraints (clock frequency, clock period, reset behavior, latency, throughput, power) while preserving original numeric formats and engineering units.

---

## 4. Structure of EFS IR

The root [`EFSIR`](file:///e:/EFS_streamlit/hardware_graphrag/core/efs_ir/models.py#L1430) dataclass aggregates 28 canonical hardware, verification, and diagnostic categories:

```
EFSIR
├── Metadata & Quality
│   ├── metadata                # Design name, hashes, model versions, timestamps
│   ├── extraction_quality      # Recalls, coverage %, property status breakdown
│   ├── extraction_coverage     # Block extraction counts and coverage %
│   └── validation              # Synthesizability and rule validation report
├── Structural & Interconnect
│   ├── components              # Modules, submodules, hierarchical blocks
│   ├── interfaces              # Inter-module interconnects, bus protocols
│   ├── signals                 # Hardware ports, directions, bit widths
│   ├── clocks                  # Clock signals, frequencies, active edges
│   └── resets                  # Reset signals, active levels, sync/async
├── Registers & Memory
│   ├── registers               # CSRs, offsets, field-slices, access modes
│   └── memory_regions          # SRAM, caches (L1/L2), BRAMs, shared buffers
├── Control & Behavior
│   ├── fsms                    # FSM states, transitions, reset states
│   ├── transactions            # Bus transactions, handshakes, burst types
│   ├── flows                   # End-to-end data/control flow paths
│   ├── protocol_rules          # Bus handshake and signal stability rules
│   └── functional_behavior     # High-level algorithms and operations
├── ISA & Execution (Optional)
│   ├── instructions            # Instruction mnemonics and semantics
│   ├── instruction_formats     # Bitfield slices and field definitions
│   ├── opcodes                 # Binary and hexadecimal opcodes
│   └── execution_models        # Execution strategies and pipeline stages
├── Constraints & Timing
│   ├── timing_rules            # Frequency, clock periods, setup/hold constraints
│   ├── performance_requirements# Throughput, latency, frequency, power targets
│   └── constraints             # Area, power, and operational constraints
├── Verification & Faults
│   ├── assertions              # SystemVerilog Assertion (SVA) intents
│   ├── coverage                # Functional coverage points
│   └── errors                  # Exception signals, faults, ECC/parity errors
└── Conflict Resolution Overlay
    ├── requirements            # Atomic immutable extracted requirements
    ├── conflicts               # Detected hardware gaps and contradictions
    └── resolutions             # Approved human/AI resolution overlays
```

### Immutable Base + Resolution Overlay
The extracted EFS IR remains **100% immutable**. When the human approves resolutions in HITL, they are stored in the resolution layer. Calling `efs_ir.get_resolved_view()` produces a resolved model overlay without modifying the original source requirements.

---

## 5. Extraction Quality Metrics

After extraction and recovery, the pipeline calculates and reports:

* **`entity_recall`**: Ratio of extracted components vs. component mentions in the source text.
* **`relationship_recall`**: Ratio of extracted module relationships vs. detected relationship verbs.
* **`interface_recall`**: Ratio of extracted interfaces vs. interface statements.
* **`register_recall`**: Ratio of extracted registers (explicit + category) vs. register mentions.
* **`constraint_recall`**: Ratio of extracted timing/performance constraints vs. constraint statements.
* **`fsm_recall`**: Ratio of extracted FSM states/transitions vs. control flow statements.
* **`evidence_coverage`**: Percentage of EFS IR constructs retaining verbatim source evidence.
* **Property Counts**:
  * `explicit_properties_found`
  * `partial_properties_found`
  * `inferred_properties_found`
  * `unresolved_properties`
  * `unsupported_ai_proposals` (proposals rejected by anti-hallucination guard)

---

## 6. Project Structure

```
hardware_graphrag/
├── app.py                              # Streamlit interactive UI application
├── config.py                           # Configuration and environment settings
├── core/
│   ├── parsing/                        # PDF, DOCX, Markdown hierarchical parsers
│   ├── chunking/                       # Semantic and hierarchical chunking
│   ├── requirement_ir/                 # Atomic requirement extraction & models
│   ├── extraction/                     # Evidence-backed extraction engines
│   │   ├── source_evidence.py          # SourceEvidence, EvidenceBackedProperty, Index
│   │   ├── dedicated_extractors.py     # Relationship, Interface, Register, FSM, Constraint
│   │   └── source_recovery.py          # SourceBackedRecovery engine
│   ├── compiler/                       # Requirement-to-EFS IR compiler
│   ├── efs_ir/                         # Canonical EFS IR models, builder, validator
│   ├── verification/                   # Conflict detection & resolution engine
│   │   ├── conflict_resolution_engine.py # Completeness validator & anti-hallucination guard
│   │   └── static_verifier.py          # Static hardware syntax and rule validation
│   ├── retrieval/                      # Protocol resolver, target resolver, token budget
│   ├── context_compiler/               # ContextPack compilation & token pruning
│   ├── agents/                         # Orchestrator, PlantUML agent, RTL agent, Spec agent
│   ├── graph/                          # Neo4j & in-memory hardware knowledge graph
│   ├── vectorstore/                    # FAISS semantic vector store
│   └── pipeline.py                     # Master pipeline orchestrator
├── tests/                              # Comprehensive test suite
│   ├── test_evidence_extraction_and_recovery.py # 12 evidence recovery & anti-hallucination tests
│   ├── test_generation_pipeline_fix.py          # 10 single-source EFS IR & RTL tests
│   ├── test_generic_conflict_resolution.py      # 16 conflict detection & resolution tests
│   └── ...                             # Additional regression tests
└── data/                               # Cache, uploads, logs, EFS IR storage
```

---

## 7. Setup & Execution

### Installation
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # On Linux/macOS: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit Application
streamlit run app.py
```

### Running Tests
Run the evidence recovery, validation, and anti-hallucination regression suite:
```bash
python -m pytest tests/test_evidence_extraction_and_recovery.py -v
```

Run the complete generation and conflict resolution test suite:
```bash
python -m pytest tests/test_evidence_extraction_and_recovery.py tests/test_generation_pipeline_fix.py tests/test_generic_conflict_resolution.py -v
```
