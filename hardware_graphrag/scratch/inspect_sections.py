import sys
sys.path.insert(0, ".")
from core.parsing.docx_parser import DocxParser

parser = DocxParser('data/uploads/matrix_compute_unit.docx')
doc = parser.parse()
print(f'Total sections: {len(doc.sections)}')
for i, s in enumerate(doc.sections):
    print(f'[{i:02d}] title="{s.title[:45]}" type={s.content_type} len={len(s.text or "")}')
    if s.text:
        sample = s.text.replace("\n", " ")[:90]
        print(f'     sample: {sample}')
