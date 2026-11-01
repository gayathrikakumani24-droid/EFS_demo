"""
PDF parsing for hardware specification documents.

Uses PyMuPDF (fitz) for fast text + layout extraction (page numbers, font
sizes to help detect headings) and falls back to pdfplumber for table
extraction on pages where tables are detected. If neither library is
available in the environment, a plain-text fallback keeps the pipeline
functional (degraded: no page-accurate splitting).
"""

from __future__ import annotations

from typing import List, Optional

from core.parsing.base_parser import BaseParser, HeadingClassifier
from utils.models import DocSection, ParsedDocument, new_id
from utils.logger import get_logger

logger = get_logger("parsing.pdf")

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False

try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False


class PDFParser(BaseParser):
    file_type = "pdf"

    def parse(self) -> ParsedDocument:
        doc_id = new_id("doc")
        sections: List[DocSection] = []

        if _HAS_FITZ:
            sections, total_pages, raw_len = self._parse_with_fitz(doc_id)
        else:
            logger.warning("PyMuPDF not installed - falling back to naive text extraction.")
            sections, total_pages, raw_len = self._parse_fallback(doc_id)

        self._link_sections(sections)
        logger.info(f"Parsed PDF '{self.filename}': {len(sections)} sections across {total_pages} pages.")
        return ParsedDocument(
            doc_id=doc_id,
            filename=self.filename,
            file_type=self.file_type,
            sections=sections,
            total_pages=total_pages,
            raw_text_len=raw_len,
        )

    # ------------------------------------------------------------------
    def _parse_with_fitz(self, doc_id: str):
        sections: List[DocSection] = []
        raw_len = 0
        current_chapter = "Untitled"
        current_text_buf: List[str] = []
        current_section: Optional[DocSection] = None

        with fitz.open(self.filepath) as pdf:
            total_pages = pdf.page_count

            # Precompute a "body font size" baseline from the most common span size,
            # so lines rendered notably larger/bolder are treated as heading candidates.
            size_counts = {}
            for page in pdf:
                for block in page.get_text("dict")["blocks"]:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            size_counts[round(span["size"])] = size_counts.get(round(span["size"]), 0) + 1
            body_size = max(size_counts, key=size_counts.get) if size_counts else 10

            def flush_text(sec: Optional[DocSection], buf: List[str]):
                if sec is not None and buf:
                    sec.text = (sec.text + "\n" + "\n".join(buf)).strip()
                buf.clear()

            for page_index, page in enumerate(pdf, start=1):
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        if not spans:
                            continue
                        line_text = "".join(s["text"] for s in spans).strip()
                        raw_len += len(line_text)
                        if not line_text:
                            continue
                        max_span_size = max(s["size"] for s in spans)
                        is_bold = any("Bold" in s.get("font", "") for s in spans)

                        heading_match = HeadingClassifier.match_heading(line_text)
                        looks_like_heading = heading_match is not None and (
                            max_span_size >= body_size + 1 or is_bold
                        )

                        if looks_like_heading:
                            flush_text(current_section, current_text_buf)
                            number, depth, title = heading_match
                            sec = self._new_section(
                                title=title,
                                level=min(depth, 4),
                                section_number=number,
                                page=page_index,
                                content_type="text",
                            )
                            if depth == 0:
                                current_chapter = title
                            sec.chapter = current_chapter
                            sections.append(sec)
                            current_section = sec
                        else:
                            content_type = HeadingClassifier.classify_content_type(line_text)
                            if content_type != "text" and current_section is not None:
                                # Special block (table/figure/note/etc) - own mini-section
                                flush_text(current_section, current_text_buf)
                                special = self._new_section(
                                    title=line_text[:80],
                                    level=(current_section.level + 1),
                                    page=page_index,
                                    content_type=content_type,
                                    text=line_text,
                                )
                                special.chapter = current_chapter
                                sections.append(special)
                            else:
                                if current_section is None:
                                    current_section = self._new_section(
                                        title=current_chapter, level=0, page=page_index
                                    )
                                    current_section.chapter = current_chapter
                                    sections.append(current_section)
                                current_text_buf.append(line_text)
                        if current_section is not None:
                            current_section.page_end = page_index

                # extract tables with pdfplumber (page-level, optional)
                if _HAS_PDFPLUMBER:
                    try:
                        self._extract_tables_for_page(page_index, sections, current_chapter)
                    except Exception as e:  # pragma: no cover - best-effort
                        logger.debug(f"Table extraction skipped for page {page_index}: {e}")

            flush_text(current_section, current_text_buf)

        return sections, total_pages, raw_len

    def _extract_tables_for_page(self, page_index: int, sections: List[DocSection], chapter: str) -> None:
        with pdfplumber.open(self.filepath) as pdf:
            pdf_page = pdf.pages[page_index - 1]
            tables = pdf_page.extract_tables()
            for t_idx, table in enumerate(tables):
                if not table:
                    continue
                headers = [str(cell).strip() if cell else "" for cell in table[0]]
                table_lines = ["| " + " | ".join(headers) + " |"]
                table_lines.append("| " + " | ".join(["---"] * max(1, len(headers))) + " |")
                for row in table[1:]:
                    row_cells = [str(cell).strip() if cell else "" for cell in row]
                    table_lines.append("| " + " | ".join(row_cells) + " |")
                
                rows_text = "\n".join(table_lines)
                if rows_text.strip():
                    sec = self._new_section(
                        title=f"Table (p.{page_index}#{t_idx + 1})",
                        level=2,
                        page=page_index,
                        content_type="table",
                        text=rows_text,
                    )
                    sec.chapter = chapter
                    sections.append(sec)

    # ------------------------------------------------------------------
    def _parse_fallback(self, doc_id: str):
        """Minimal fallback if PyMuPDF isn't available: one big text section."""
        try:
            with open(self.filepath, "rb") as f:
                raw = f.read()
            text = raw.decode("latin-1", errors="ignore")
        except Exception as e:
            logger.error(f"Failed to read PDF as fallback text: {e}")
            text = ""
        sec = self._new_section(title=self.filename, level=0, page=1, text=text)
        return [sec], 1, len(text)
