"""
Common interface + shared heuristics for all document parsers.

Every concrete parser (PDF, DOCX, Markdown) implements `parse()` and
returns a ParsedDocument with a flat list of DocSection nodes that
together encode the document hierarchy (chapter -> section -> subsection)
plus special content blocks (tables, figures, timing diagrams, register
descriptions, notes, warnings, examples).

Heading / hierarchy detection is heuristic (regex based) since hardware
spec PDFs rarely expose reliable structural tags. The heuristics are
intentionally conservative and shared across parsers via `HeadingClassifier`
so behavior is consistent regardless of source format.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from utils.models import DocSection, ParsedDocument, new_id
from utils.logger import get_logger

logger = get_logger("parsing.base")

_STANDALONE_SECTION_RE = re.compile(
    r"^\s*(Chapter|Section|Appendix)\s*((?:0|[1-9]\d*)(?:\.\d+){0,4}|[A-Z](?:\.\d+){0,4})\s*[:.\-]?\s*$",
    re.IGNORECASE,
)

# Matches "3", "3.2", "3.2.1", "Chapter 3", "Section 3.2", "Appendix A.1" etc.
_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(Chapter|Section|Appendix)?\s*((?:0|[1-9]\d*)(?:\.\d+){0,4}|[A-Z](?:\.\d+){0,4})\s*[:.\-]?\s+(.{2,120})$",
    re.IGNORECASE,
)

_TABLE_CAPTION_RE = re.compile(r"^\s*Table\s+\d+(\.\d+)*", re.IGNORECASE)
_FIGURE_CAPTION_RE = re.compile(r"^\s*Figure\s+\d+(\.\d+)*", re.IGNORECASE)
_TIMING_RE = re.compile(r"timing\s+diagram|waveform", re.IGNORECASE)
_REGISTER_RE = re.compile(r"\bregister\b.*\b(offset|address|bit\s*field|reset\s*value)\b", re.IGNORECASE)
_NOTE_RE = re.compile(r"^\s*(NOTE|NOTES?)\s*[:.\-]", re.IGNORECASE)
_WARNING_RE = re.compile(r"^\s*(WARNING|CAUTION|IMPORTANT)\s*[:.\-]", re.IGNORECASE)
_EXAMPLE_RE = re.compile(r"^\s*Example\s*\d*\s*[:.\-]", re.IGNORECASE)


class HeadingClassifier:
    """Shared heuristics to classify a line of text as a heading / special block."""

    @staticmethod
    def match_heading(line: str) -> Optional[Tuple[str, int, str]]:
        """
        Try to interpret `line` as a heading.

        Returns (section_number, level, title) or None.
        Level is derived from the depth of the numbering, e.g.:
            "3"       -> level 0 (chapter)
            "3.2"     -> level 1 (section)
            "3.2.1"   -> level 2 (subsection)
        """
        line = line.strip()
        if not line or len(line) > 160:
            return None

        # Check standalone section heading first (e.g. "Section 14", "Chapter 3")
        sm = _STANDALONE_SECTION_RE.match(line)
        if sm:
            prefix = sm.group(1)
            number = sm.group(2)
            depth = number.count(".")
            return number, depth, f"{prefix} {number}"

        m = _NUMBERED_HEADING_RE.match(line)
        if not m:
            return None

        prefix = m.group(1)
        number = m.group(2)
        title = m.group(3).strip().rstrip(".")

        # If no formal prefix (Chapter/Section/Appendix) and no dotted sub-level (e.g. just "64"):
        # Do not treat as heading if title contains a colon (key: value pair like "Matrix ALU : Performs...")
        # or if number > 25 without dot (hardware specs rarely have chapter 64 without prefix)
        if not prefix and "." not in number:
            if " : " in title or (number.isdigit() and int(number) > 25):
                return None

        # If title ends with description in parens or contains key-value colon, avoid false heading
        if " : " in title and not prefix:
            return None

        depth = number.count(".")
        return number, depth, title

    @staticmethod
    def classify_content_type(line: str) -> str:
        if _TABLE_CAPTION_RE.match(line):
            return "table"
        if _FIGURE_CAPTION_RE.match(line):
            return "figure"
        if _TIMING_RE.search(line):
            return "timing_diagram"
        if _REGISTER_RE.search(line):
            return "register"
        if _NOTE_RE.match(line):
            return "note"
        if _WARNING_RE.match(line):
            return "warning"
        if _EXAMPLE_RE.match(line):
            return "example"
        return "text"


class BaseParser(ABC):
    """Abstract base class every format-specific parser must implement."""

    file_type: str = "unknown"

    def __init__(self, filepath: str, filename: Optional[str] = None):
        self.filepath = filepath
        self.filename = filename or filepath.split("/")[-1]

    @abstractmethod
    def parse(self) -> ParsedDocument:
        ...

    def _link_sections(self, sections: List[DocSection]) -> None:
        """Populate previous/next/parent pointers on a flat, ordered section list."""
        stack: List[DocSection] = []   # tracks current open ancestors by level
        for i, sec in enumerate(sections):
            if i > 0:
                sec.previous_id = sections[i - 1].section_id
                sections[i - 1].next_id = sec.section_id

            # maintain a stack of ancestors to determine parent_id
            while stack and stack[-1].level >= sec.level:
                stack.pop()
            sec.parent_id = stack[-1].section_id if stack else None
            stack.append(sec)

            # propagate chapter name downward
            if sec.level == 0:
                current_chapter = sec.title
            sec.chapter = sec.chapter or (stack[0].title if stack else sec.title)

    def _new_section(
        self,
        title: str,
        level: int,
        section_number: str = "",
        page: int = 0,
        content_type: str = "text",
        text: str = "",
    ) -> DocSection:
        return DocSection(
            section_id=new_id("sec"),
            title=title,
            level=level,
            section_number=section_number,
            page_start=page,
            page_end=page,
            content_type=content_type,
            text=text,
        )
