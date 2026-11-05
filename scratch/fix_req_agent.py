with open('hardware_graphrag/core/agents/requirement_agent.py', 'rb') as f:
    content = f.read()

target = b'from __future__ import annotations\r\n\r\nfrom typing import Any, Dict, Optional'
replacement = b'from __future__ import annotations\r\n\r\nimport re\r\nfrom typing import Any, Dict, Optional'
if target not in content:
    target = b'from __future__ import annotations\n\nfrom typing import Any, Dict, Optional'
    replacement = b'from __future__ import annotations\n\nimport re\nfrom typing import Any, Dict, Optional'

assert target in content, "target not found"
content = content.replace(target, replacement, 1)
with open('hardware_graphrag/core/agents/requirement_agent.py', 'wb') as f:
    f.write(content)
print("Added import re successfully")
