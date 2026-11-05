"""Script to apply targeted generic bugfixes to constrained_generator.py and builder.py."""

import re
import sys

def patch_builder():
    path = "hardware_graphrag/core/efs_ir/builder.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    crlf = "\r\n" in content
    newline = "\r\n" if crlf else "\n"
    text = content.replace("\r\n", "\n")

    # 1. Update _clean_title to reject sentences
    old_clean = """        def _clean_title(raw: str) -> str:
            if not raw:
                return ""
            s = str(raw).strip()
            parts = s.split()
            while parts and parts[0].lower() in ("the", "a", "an"):
                parts = parts[1:]
            return " ".join(parts).strip()"""

    new_clean = """        def _clean_title(raw: str) -> str:
            if not raw:
                return ""
            s = str(raw).strip()
            parts = s.split()
            while parts and parts[0].lower() in ("the", "a", "an"):
                parts = parts[1:]
            # If the candidate contains descriptive sentence verbs or is overly long, reject as title
            sentence_verbs = {"builds", "implements", "provides", "consists", "contains", "describes", "defines", "supports", "uses"}
            if len(parts) > 6 or any(w.lower() in sentence_verbs for w in parts[:3]):
                return ""
            return " ".join(parts).strip()"""

    assert old_clean in text, "Could not find old_clean in builder.py"
    text = text.replace(old_clean, new_clean, 1)

    # 2. Prioritize Verilog module declarations and explicit specification titles
    old_spec_name = r"""        if not raw_display_name and self.design_spec:
            name_match = re.search(r"#\s*Hardware Design Specification:\s*([A-Za-z0-9_\-\s]+)", self.design_spec)
            if not name_match:
                name_match = re.search(r"(?:module|design|specification:)\s+([A-Za-z0-9_\-\s]+?)(?:\s+with|\s*\n|$)", self.design_spec, re.IGNORECASE)
            if name_match:
                candidate = _clean_title(name_match.group(1))
                if candidate and candidate.lower() not in GENERIC_TITLES:
                    raw_display_name = candidate"""

    new_spec_name = r"""        if not raw_display_name and self.design_spec:
            # 1. First check explicit Verilog module declaration header
            mod_decl_match = re.search(r"\bmodule\s+([a-zA-Z0-9_]+)\s*(?:#\s*\(|\()", self.design_spec)
            if mod_decl_match:
                name_match = mod_decl_match
            else:
                # 2. Check explicit specification title
                name_match = re.search(r"#\s*Hardware Design Specification:\s*([A-Za-z0-9_\-\s]+)", self.design_spec)
            if not name_match:
                name_match = re.search(r"(?:Functional\s+)?Specification\s+for\s+([A-Za-z0-9_\-\s]+?)(?:\.|\n|$)", self.design_spec, re.IGNORECASE)
            if not name_match:
                name_match = re.search(r"(?:module|specification:)\s+([A-Za-z0-9_\-\s]+?)(?:\s+with|\s*\n|$)", self.design_spec, re.IGNORECASE)
            if name_match:
                candidate = _clean_title(name_match.group(1))
                if candidate and candidate.lower() not in GENERIC_TITLES:
                    raw_display_name = candidate"""

    assert old_spec_name in text, "Could not find old_spec_name in builder.py"
    text = text.replace(old_spec_name, new_spec_name, 1)

    final_content = text.replace("\n", newline)
    with open(path, "w", encoding="utf-8") as f:
        f.write(final_content)
    print("Successfully patched builder.py!")


if __name__ == "__main__":
    patch_builder()
