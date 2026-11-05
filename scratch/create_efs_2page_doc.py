import os
import sys
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def create_docx(output_path):
    doc = Document()

    # Set margins to 0.5 in for tight 2-page fit
    for section in doc.sections:
        section.top_margin = Inches(0.5)
        section.bottom_margin = Inches(0.5)
        section.left_margin = Inches(0.6)
        section.right_margin = Inches(0.6)

    # Palette
    NAVY = RGBColor(24, 43, 73)
    TEAL = RGBColor(14, 116, 144)
    SLATE = RGBColor(71, 85, 105)
    CHARCOAL = RGBColor(30, 41, 59)

    # Header / Title Block
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(2)
    run_title = p_title.add_run("PROJECT EFS: EXECUTABLE FUNCTIONAL SPECIFICATION")
    run_title.font.name = "Arial"
    run_title.font.size = Pt(17)
    run_title.font.bold = True
    run_title.font.color.rgb = NAVY

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(8)
    run_sub = p_sub.add_run("A GraphRAG-Driven Formal Compilation Framework for Spec-to-RTL Hardware Synthesis")
    run_sub.font.name = "Arial"
    run_sub.font.size = Pt(10.5)
    run_sub.font.italic = True
    run_sub.font.color.rgb = TEAL

    # Meta bar table
    t_meta = doc.add_table(rows=1, cols=4)
    t_meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Domain:", "EDA / AI Hardware Design"),
        ("Architecture:", "Knowledge Graph + FAISS RAG"),
        ("Outputs:", "Synthesizable RTL, PlantUML, SVA, UVM"),
        ("Status:", "Production / Active Ingestion")
    ]
    for idx, (lbl, val) in enumerate(meta_data):
        cell = t_meta.rows[0].cells[idx]
        set_cell_background(cell, "F1F5F9")
        set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        r1 = p.add_run(lbl + " ")
        r1.font.bold = True
        r1.font.size = Pt(7.5)
        r1.font.color.rgb = NAVY
        r2 = p.add_run(val)
        r2.font.size = Pt(7.5)
        r2.font.color.rgb = SLATE

    # SECTION 1: EXECUTIVE OVERVIEW & PROBLEM
    p_s1 = doc.add_paragraph()
    p_s1.paragraph_format.space_before = Pt(8)
    p_s1.paragraph_format.space_after = Pt(2)
    r_s1 = p_s1.add_run("1. Executive Overview & The Hardware LLM Dilemma")
    r_s1.font.name = "Arial"
    r_s1.font.size = Pt(11)
    r_s1.font.bold = True
    r_s1.font.color.rgb = NAVY

    p_body1 = doc.add_paragraph()
    p_body1.paragraph_format.space_before = Pt(0)
    p_body1.paragraph_format.space_after = Pt(4)
    p_body1.paragraph_format.line_spacing = 1.05
    r_b1 = p_body1.add_run(
        "Modern System-on-Chip (SoC) design begins with complex, unstructured natural language documents—hundreds of pages detailing register maps, protocol timings, and ISA formats. Standard Large Language Models (LLMs) fail catastrophically when applied directly to hardware design: they suffer from context window saturation, invent non-existent pins (hallucinations), leak uncoordinated bus protocols (e.g. injecting full 512-bit AXI masters into internal leaf decoders), and omit critical handshake transitions.\n"
        "Project EFS (Executable Functional Specification) solves this paradigm by replacing blind text prompting with an authoritative, query-driven GraphRAG compilation pipeline. EFS extracts raw specification text into a structured, schema-validated Intermediate Representation (EFS IR), models design relationships in a formal Knowledge Graph, and generates mathematically grounded, 100% compliant synthesizable RTL, formal assertions, and UVM testbenches."
    )
    r_b1.font.name = "Arial"
    r_b1.font.size = Pt(8.5)
    r_b1.font.color.rgb = CHARCOAL

    # SECTION 2: ARCHITECTURAL PILLARS
    p_s2 = doc.add_paragraph()
    p_s2.paragraph_format.space_before = Pt(6)
    p_s2.paragraph_format.space_after = Pt(2)
    r_s2 = p_s2.add_run("2. Core Architectural Pillars of EFS")
    r_s2.font.name = "Arial"
    r_s2.font.size = Pt(11)
    r_s2.font.bold = True
    r_s2.font.color.rgb = NAVY

    pillars_table = doc.add_table(rows=3, cols=2)
    pillars_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    pillars = [
        ("A. Dual-Engine Representation (GraphRAG)",
         "Combines dense semantic FAISS vector search (MiniLM-L6-v2) for natural language requirement chunks with a formal NetworkX Knowledge Graph capturing hardware topology, clock domains, and structural parent-child hierarchy."),
        ("B. Authoritative EFS IR & Grounding Validator",
         "A strongly typed dataclass schema defining components, signals, opcodes, FSM state graphs, and timing constraints. The GroundingValidator guarantees that every generated pin, register, and wire is strictly grounded in the specification."),
        ("C. Query-Driven Context Compilation",
         "Instead of dumping whole specs into prompts, EFS compiles target-specific 'ContextPacks'. Monolithic 30,000-token specifications are reduced by 88% to 97% down to 800-1,500 highly focused, mathematically complete tokens."),
    ]
    for row_idx, (head, desc) in enumerate(pillars):
        c0 = pillars_table.rows[row_idx].cells[0]
        c1 = pillars_table.rows[row_idx].cells[1]
        set_cell_background(c0, "F8FAFC")
        set_cell_background(c1, "FFFFFF")
        set_cell_margins(c0, top=40, bottom=40, left=70, right=70)
        set_cell_margins(c1, top=40, bottom=40, left=70, right=70)
        
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_after = Pt(0)
        r0 = p0.add_run(head)
        r0.font.bold = True
        r0.font.size = Pt(8.0)
        r0.font.color.rgb = NAVY

        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_after = Pt(0)
        r1 = p1.add_run(desc)
        r1.font.size = Pt(8.0)
        r1.font.color.rgb = CHARCOAL

    # SECTION 3: END-TO-END WORKFLOW PIPELINE
    p_s3 = doc.add_paragraph()
    p_s3.paragraph_format.space_before = Pt(6)
    p_s3.paragraph_format.space_after = Pt(2)
    r_s3 = p_s3.add_run("3. End-to-End Execution Flow (14-Stage Unified Coordinator)")
    r_s3.font.name = "Arial"
    r_s3.font.size = Pt(11)
    r_s3.font.bold = True
    r_s3.font.color.rgb = NAVY

    flow_table = doc.add_table(rows=5, cols=3)
    flow_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Stage Phase", "Key Mechanism / Engine", "System Guarantee & Output"]
    for c_i, h_text in enumerate(headers):
        cell = flow_table.rows[0].cells[c_i]
        set_cell_background(cell, "1E293B")
        set_cell_margins(cell, top=50, bottom=50, left=60, right=60)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(h_text)
        r.font.bold = True
        r.font.size = Pt(8.0)
        r.font.color.rgb = RGBColor(255, 255, 255)

    stages = [
        ("1. Ingestion & Graphing", "Document Parser -> FAISS Store + NetworkX Graph", "Extracts signals, opcodes, FSMs; builds structural relationships."),
        ("2. Target Resolution", "Multi-stage TargetResolver + Sub-Module Derivation", "Maps user intent to exact sub-modules (e.g. instruction decoders)."),
        ("3. Context Slicing", "ContextCompiler -> ContextPack Compilation", "Slices isolated signal sets; isolates external protocol buses (AXI/AHB)."),
        ("4. Hybrid Generation", "Constrained AST Generator + Multi-Candidate Ranking", "Emits clean, synthesizable Verilog/SV; validates against 0 lint errors.")
    ]
    for r_i, (s_p, s_m, s_g) in enumerate(stages, 1):
        for c_i, val in enumerate([s_p, s_m, s_g]):
            cell = flow_table.rows[r_i].cells[c_i]
            set_cell_background(cell, "F8FAFC" if r_i % 2 == 1 else "FFFFFF")
            set_cell_margins(cell, top=35, bottom=35, left=60, right=60)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(val)
            r.font.size = Pt(7.5)
            r.font.color.rgb = CHARCOAL

    # PAGE BREAK TO PAGE 2
    doc.add_page_break()

    # SECTION 4: ADVANCED SUBSYSTEMS & TECHNICAL DIFFERENTIATORS
    p_s4 = doc.add_paragraph()
    p_s4.paragraph_format.space_before = Pt(0)
    p_s4.paragraph_format.space_after = Pt(2)
    r_s4 = p_s4.add_run("4. Advanced Subsystems & Technical Differentiators")
    r_s4.font.name = "Arial"
    r_s4.font.size = Pt(11)
    r_s4.font.bold = True
    r_s4.font.color.rgb = NAVY

    diff_table = doc.add_table(rows=4, cols=2)
    diff_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    diffs = [
        ("Sub-Module Contract Derivation & Bus Isolation",
         "Automatically discovers sub-units (e.g., instruction decoders, parsers, ALUs) from ISA opcodes and bitfield specifications. Strips 512-bit external system buses (AXI4, AHB) from internal blocks to prevent bus leakage while preserving strict register-slice handshakes."),
        ("Conflict Resolution Engine & HITL Proposals",
         "Scans specs for contradictions (e.g., a datapath expecting 8-bit opcodes while encoding tables specify 6-bit). Generates structured Human-in-the-Loop (HITL) resolution proposals, allowing engineers to approve formal overrides before synthesis."),
        ("FSM Strategy & Architecture Classifier",
         "Analyzes state definitions using the FSM Inference Engine. Distinguishes purely combinational datapaths, pipeline registers, and multi-cycle state machines. Synthesizes formal state registers, next-state logic, and default transitions without human intervention."),
        ("Deterministic & Constrained Generation",
         "Eliminates LLM stochastic nondeterminism. Generates AST-verified Verilog where port lists, bit slices, continuous assignments, and clock-gated registers match exact EFS IR contracts with zero syntax defects.")
    ]
    for r_i, (feat, desc) in enumerate(diffs):
        c0 = diff_table.rows[r_i].cells[0]
        c1 = diff_table.rows[r_i].cells[1]
        set_cell_background(c0, "F1F5F9")
        set_cell_background(c1, "FFFFFF")
        set_cell_margins(c0, top=40, bottom=40, left=70, right=70)
        set_cell_margins(c1, top=40, bottom=40, left=70, right=70)

        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_after = Pt(0)
        r0 = p0.add_run(feat)
        r0.font.bold = True
        r0.font.size = Pt(8.0)
        r0.font.color.rgb = NAVY

        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_after = Pt(0)
        r1 = p1.add_run(desc)
        r1.font.size = Pt(8.0)
        r1.font.color.rgb = CHARCOAL

    # SECTION 5: MULTI-ARTIFACT COMPILATION OUTPUTS
    p_s5 = doc.add_paragraph()
    p_s5.paragraph_format.space_before = Pt(8)
    p_s5.paragraph_format.space_after = Pt(2)
    r_s5 = p_s5.add_run("5. Multi-Artifact Hardware Collateral Generation")
    r_s5.font.name = "Arial"
    r_s5.font.size = Pt(11)
    r_s5.font.bold = True
    r_s5.font.color.rgb = NAVY

    p_body5 = doc.add_paragraph()
    p_body5.paragraph_format.space_before = Pt(0)
    p_body5.paragraph_format.space_after = Pt(4)
    p_body5.paragraph_format.line_spacing = 1.05
    r_b5 = p_body5.add_run(
        "Project EFS compiles complete, production-ready tape-out collateral from a unified EFS IR context:\n"
        "• Synthesizable Verilog / SystemVerilog: Fully synthesizable RTL conforming to IEEE 1364-2001 and 1800-2017 standards.\n"
        "• SystemVerilog Assertions (SVA): Temporal property checks verifying protocol handshakes (e.g. valid/ready hold rules, mutual exclusion).\n"
        "• Complete UVM Testbenches: UVM sequences, scoreboards, drivers, and monitors generated to test edge cases directly in simulators.\n"
        "• Dynamic PlantUML Diagrams: Architectural block diagrams and timing sequence diagrams rendered dynamically from the Knowledge Graph."
    )
    r_b5.font.name = "Arial"
    r_b5.font.size = Pt(8.5)
    r_b5.font.color.rgb = CHARCOAL

    # SECTION 6: MEASURABLE PERFORMANCE & VERIFICATION METRICS
    p_s6 = doc.add_paragraph()
    p_s6.paragraph_format.space_before = Pt(6)
    p_s6.paragraph_format.space_after = Pt(2)
    r_s6 = p_s6.add_run("6. Quantitative Engineering Benchmarks & Impact")
    r_s6.font.name = "Arial"
    r_s6.font.size = Pt(11)
    r_s6.font.bold = True
    r_s6.font.color.rgb = NAVY

    metrics_table = doc.add_table(rows=2, cols=4)
    metrics_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    metrics = [
        ("88% - 97%", "Context Token Reduction\n(Prompt bloat eliminated)"),
        ("100%", "Signal Grounding Pass Rate\n(0 hallucinated ports)"),
        ("< 10 Sec", "Turnaround Generation Time\n(End-to-end plan to RTL)"),
        ("0 AXI Leakage", "Clean Internal Sub-Modules\n(Bus isolation enforced)")
    ]
    for idx, (stat, label) in enumerate(metrics):
        c_top = metrics_table.rows[0].cells[idx]
        c_bot = metrics_table.rows[1].cells[idx]
        set_cell_background(c_top, "F1F5F9")
        set_cell_background(c_bot, "F8FAFC")
        set_cell_margins(c_top, top=40, bottom=20, left=50, right=50)
        set_cell_margins(c_bot, top=20, bottom=40, left=50, right=50)

        p_t = c_top.paragraphs[0]
        p_t.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_t.paragraph_format.space_after = Pt(0)
        r_t = p_t.add_run(stat)
        r_t.font.bold = True
        r_t.font.size = Pt(13)
        r_t.font.color.rgb = TEAL

        p_b = c_bot.paragraphs[0]
        p_b.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_b.paragraph_format.space_after = Pt(0)
        r_b = p_b.add_run(label)
        r_b.font.size = Pt(7.0)
        r_b.font.color.rgb = SLATE

    # SECTION 7: SUMMARY & ROADMAP
    p_s7 = doc.add_paragraph()
    p_s7.paragraph_format.space_before = Pt(8)
    p_s7.paragraph_format.space_after = Pt(2)
    r_s7 = p_s7.add_run("7. Industrial Integration & Roadmap")
    r_s7.font.name = "Arial"
    r_s7.font.size = Pt(11)
    r_s7.font.bold = True
    r_s7.font.color.rgb = NAVY

    p_body7 = doc.add_paragraph()
    p_body7.paragraph_format.space_before = Pt(0)
    p_body7.paragraph_format.space_after = Pt(0)
    p_body7.paragraph_format.line_spacing = 1.05
    r_b7 = p_body7.add_run(
        "Project EFS transforms hardware specification engineering from a fragile, error-prone manual process into a repeatable, deterministic compiler flow. By combining GraphRAG with strict semantic IR constraints, EFS eliminates the risks of hallucination and ensures high-fidelity silicon compilation. Current development focuses on native integrations with commercial EDA synthesis suites (Synopsys Design Compiler, Cadence Genus), automated formal equivalence checking, and expanded protocol libraries for PCIe Gen5, CXL 3.0, and AMBA CHI."
    )
    r_b7.font.name = "Arial"
    r_b7.font.size = Pt(8.5)
    r_b7.font.color.rgb = CHARCOAL

    doc.save(output_path)
    print(f"Document successfully created at: {output_path}")

if __name__ == "__main__":
    out1 = "hardware_graphrag/scratch/EFS_Project_Document.docx"
    create_docx(out1)

