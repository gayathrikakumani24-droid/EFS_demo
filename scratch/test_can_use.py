import sys, os
sys.path.insert(0, os.path.abspath('hardware_graphrag'))
from core.agents.spec_classifier import can_use_library_module
import docx

doc = docx.Document('hardware_graphrag/matrix_compute_unit.docx')
text = '\n'.join([p.text for p in doc.paragraphs])
for t in doc.tables:
    for row in t.rows:
        text += '\n' + '\t'.join([c.text for c in row.cells])

print('can_use_library_module(text, None):', can_use_library_module(text, None))
print('can_use_library_module(text, {}):', can_use_library_module(text, {}))
print('can_use_library_module(text, {"module_name": "target_module"}):', can_use_library_module(text, {"module_name": "target_module"}))
print('can_use_library_module(text, {"module_name": "matrix_compute_unit"}):', can_use_library_module(text, {"module_name": "matrix_compute_unit"}))
print('can_use_library_module(text, {"module_name": "Specification Core Requirements"}):', can_use_library_module(text, {"module_name": "Specification Core Requirements"}))

