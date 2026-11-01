import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import CONFIG
from core.parsing.docx_parser import DocxParser
from core.requirement_ir.models import DocumentIR, DocumentBlock, SourceLocation
from core.extraction.requirement_extractor import GenericRequirementExtractor
from core.compiler.efs_compiler import RequirementToEFSCompiler
from core.efs_ir.validator import validate_efs_ir

CONFIG.enable_llm_escalation = False
docx_path = "data/uploads/matrix_compute_unit.docx"
if not os.path.exists(docx_path):
    docx_path = "hardware_graphrag/data/uploads/matrix_compute_unit.docx"

parser = DocxParser(docx_path)
parsed_doc = parser.parse()
doc_ir = DocumentIR(
    document_id=parsed_doc.doc_id,
    document_version="1.0",
    source_type=parsed_doc.file_type,
    document_hash="hash",
    ingestion_id="ingest",
    metadata={"filename": parsed_doc.filename},
    sections=[s.to_dict() for s in parsed_doc.sections],
    blocks=[
        DocumentBlock(
            block_id=f"BLK_{i+1:03d}",
            type=s.content_type,
            section_id=s.section_id,
            section_title=s.title,
            page=s.page_start,
            text=s.text,
            source_location=SourceLocation(
                document_id=parsed_doc.doc_id,
                document_hash="hash",
                section=s.title,
                section_id=s.section_id,
                page=s.page_start,
                block_id=f"BLK_{i+1:03d}",
                original_text=s.text[:200]
            )
        ) for i, s in enumerate(parsed_doc.sections) if s.text and s.text.strip()
    ]
)
extractor = GenericRequirementExtractor(doc_ir)
req_ir = extractor.extract_requirement_ir()
compiler = RequirementToEFSCompiler(req_ir)
efs = compiler.compile()

print(f"=== COMPONENTS ({len(efs.components)}) ===")
for c in efs.components:
    print(f"- {c.name} (type: {c.type})")

print(f"\n=== INTERFACES ({len(efs.interfaces)}) ===")
for i in efs.interfaces:
    dest = getattr(i, "destination_component", None)
    print(f"- {i.name}: {i.source_component} -> {dest}")
    print(f"  desc: {i.description}")

from core.extraction.dedicated_extractors import DedicatedRelationshipExtractor
from core.extraction.source_evidence import SourceEvidenceIndex
index = SourceEvidenceIndex(doc_ir)
known_comps = [c.name for c in efs.components]
rels = DedicatedRelationshipExtractor.extract_relationships(doc_ir, known_comps, index)
print(f"\n=== EXTRACTED RELATIONSHIPS ({len(rels)}) ===")
for r in rels:
    print(f"- {r.source} --[{r.relation}]--> {r.target}")
    print(f"  evidence: {r.evidence.text}")

print(f"\n=== FLOWS ({len(efs.flows)}) ===")
for f in efs.flows:
    print(f"- {f.flow_id} ({f.name}): components={f.components} steps={len(f.ordered_steps)}")

issues = validate_efs_ir(efs)
print(f"\n=== VALIDATION ISSUES ({len(issues)}) ===")
for iss in issues:
    print(f"[{iss.get('category')}] {iss.get('severity')}: {iss.get('message')}")
