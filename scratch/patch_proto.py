import re

path = 'hardware_graphrag/core/retrieval/protocol_resolver.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

target = """            rule_entry = {
                "text": rule_text,
                "protocol": active_proto,
                "protocol_document_id": doc_id,
                "chunk_id": chunk_id,
                "page": page,
                "section": section,
                "original_text": rule_text
            }"""

replacement = """            rule_entry = dict(rule) if isinstance(rule, dict) else {}
            rule_entry.update({
                "text": rule_text,
                "protocol": active_proto,
                "protocol_document_id": doc_id,
                "chunk_id": chunk_id,
                "page": page,
                "section": section,
                "original_text": rule_text
            })
            if "rule_id" not in rule_entry and "id" in rule_entry:
                rule_entry["rule_id"] = rule_entry["id"]"""

if target in text:
    new_text = text.replace(target, replacement, 1)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(new_text)
    print("Updated protocol_resolver.py successfully")
else:
    print("Target not found in protocol_resolver.py")

