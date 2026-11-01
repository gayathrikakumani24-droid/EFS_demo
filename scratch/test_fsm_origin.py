import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from config import CONFIG
CONFIG.enable_llm_escalation = False
from core.parsing.docx_parser import DocxParser
from core.requirement_ir.models import DocumentIR, DocumentBlock, SourceLocation
from core.extraction.requirement_extractor import GenericRequirementExtractor
from core.compiler.efs_compiler import RequirementToEFSCompiler
from core.extraction.source_evidence import SourceEvidenceIndex
from core.extraction.source_recovery import SourceBackedRecovery

parser = DocxParser('matrix_compute_unit.docx')
parsed_doc = parser.parse()
doc_ir = DocumentIR(
    document_id=parsed_doc.doc_id,
    document_version='1.0',
    source_type=parsed_doc.file_type,
    document_hash='hash',
    ingestion_id='ingest',
    metadata={'filename': parsed_doc.filename},
    sections=[s.to_dict() for s in parsed_doc.sections],
    blocks=[
        DocumentBlock(
            block_id=f'BLK_{i+1:03d}',
            type=s.content_type,
            section_id=s.section_id,
            section_title=s.title,
            page=s.page_start,
            text=s.text,
            source_location=SourceLocation(
                document_id=parsed_doc.doc_id,
                document_hash='hash',
                section=s.title,
                section_id=s.section_id,
                page=s.page_start,
                block_id=f'BLK_{i+1:03d}',
                original_text=s.text[:200]
            )
        ) for i, s in enumerate(parsed_doc.sections) if s.text and s.text.strip()
    ]
)
extractor = GenericRequirementExtractor(doc_ir)
req_ir = extractor.extract_requirement_ir()
compiler = RequirementToEFSCompiler(req_ir)
efs = compiler.compile()

print(f"After compiler.compile(): fsms count = {len(efs.fsms)}")
for f in efs.fsms:
    print(f"  FSM: {f.name}, states: {[s.name for s in f.states]}")

evidence_index = SourceEvidenceIndex(doc_ir)
SourceBackedRecovery.recover_efs_ir(efs, doc_ir, evidence_index)
print(f"After SourceBackedRecovery: fsms count = {len(efs.fsms)}")
for f in efs.fsms:
    print(f"  FSM: {f.name}, states: {[s.name for s in f.states]}")
