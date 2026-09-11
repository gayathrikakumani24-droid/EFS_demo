"""
Markdown (and plain text) parsing.

Markdown headings ("#", "##", ...) map directly and unambiguously to
hierarchy levels, so this parser is the most reliable of the three.
Plain .txt files are treated the same way but rely purely on the shared
numbered-heading regex heuristics since there is no '#' syntax.
"""

from __future__ import annotations

import re
from typing import List, Optional

from core.parsing.base_parser import BaseParser, HeadingClassifier
from utils.models import DocSection, ParsedDocument, new_id
from utils.logger import get_logger

logger = get_logger("parsing.markdown")

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


class MarkdownParser(BaseParser):
    file_type = "md"

    def parse(self) -> ParsedDocument:
        doc_id = new_id("doc")
        with open(self.filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        sections: List[DocSection] = []
        current_chapter = "Untitled"
        current_section: Optional[DocSection] = None
        text_buf: List[str] = []
        raw_len = 0
        pseudo_page = 1

        def flush(sec, buf):
            if sec is not None and buf:
                sec.text = (sec.text + "\n" + "\n".join(buf)).strip()
            buf.clear()

        for raw_line in lines:
            line_text = raw_line.rstrip("\n").strip()
            raw_len += len(line_text)
            if not line_text:
                continue

            md_match = _MD_HEADING_RE.match(line_text)
            heading_match = None if md_match else HeadingClassifier.match_heading(line_text)

            if md_match or heading_match:
                flush(current_section, text_buf)
                if md_match:
                    level = len(md_match.group(1)) - 1
                    number, title = "", md_match.group(2).strip()
                else:
                    number, level, title = heading_match

                sec = self._new_section(
                    title=title, level=min(level, 5), section_number=number,
                    page=pseudo_page, content_type="text",
                )
                if level == 0:
                    current_chapter = title
                sec.chapter = current_chapter
                sections.append(sec)
                current_section = sec
                pseudo_page += 1
                continue

            content_type = HeadingClassifier.classify_content_type(line_text)
            if content_type != "text":
                flush(current_section, text_buf)
                special = self._new_section(
                    title=line_text[:80],
                    level=(current_section.level + 1) if current_section else 1,
                    page=pseudo_page, content_type=content_type, text=line_text,
                )
                special.chapter = current_chapter
                sections.append(special)
                continue

            if current_section is None:
                current_section = self._new_section(title=current_chapter, level=0, page=pseudo_page)
                current_section.chapter = current_chapter
                sections.append(current_section)
            text_buf.append(line_text)

        flush(current_section, text_buf)
        self._link_sections(sections)
        logger.info(f"Parsed Markdown/Text '{self.filename}': {len(sections)} sections.")
        return ParsedDocument(
            doc_id=doc_id, filename=self.filename, file_type=self.file_type,
            sections=sections, total_pages=pseudo_page, raw_text_len=raw_len,
        )
