with open('hardware_graphrag/core/context_compiler/semantic_selector.py', 'rb') as f:
    content = f.read()

target = b'        if scoped_signals:\r\n            for s in scoped_signals:\r\n                s_name = self.resolver.resolve_signal_name(s.name or s.signal_id)\r\n                if s_name:\r\n                    target_sig_names.add(s_name.lower())'
if target not in content:
    target = b'        if scoped_signals:\n            for s in scoped_signals:\n                s_name = self.resolver.resolve_signal_name(s.name or s.signal_id)\n                if s_name:\n                    target_sig_names.add(s_name.lower())'

replacement = b'        if scoped_signals:\r\n            for s in scoped_signals:\r\n                raw_name = s.get("name") if isinstance(s, dict) else getattr(s, "name", None)\r\n                if not raw_name:\r\n                    raw_name = s.get("signal_id") if isinstance(s, dict) else getattr(s, "signal_id", "")\r\n                s_name = self.resolver.resolve_signal_name(raw_name) if raw_name else ""\r\n                if s_name:\r\n                    target_sig_names.add(s_name.lower())'

assert target in content, "target not found"
content = content.replace(target, replacement, 1)
with open('hardware_graphrag/core/context_compiler/semantic_selector.py', 'wb') as f:
    f.write(content)
print("Patched semantic_selector.py successfully")

