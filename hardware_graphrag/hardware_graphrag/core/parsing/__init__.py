"""Parsing package: exposes get_parser() factory used by the pipeline."""

from __future__ import annotations

import os

from core.parsing.base_parser import BaseParser
from core.parsing.pdf_parser import PDFParser
from core.parsing.docx_parser import DocxParser
from core.parsing.markdown_parser import MarkdownParser


def get_parser(filepath: str, filename: str = None) -> BaseParser:
    """Return the appropriate parser instance for the given file path."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return PDFParser(filepath, filename)
    if ext == ".docx":
        return DocxParser(filepath, filename)
    if ext in (".md", ".markdown", ".txt"):
        return MarkdownParser(filepath, filename)
    raise ValueError(f"Unsupported file extension: {ext}")


__all__ = ["get_parser", "BaseParser", "PDFParser", "DocxParser", "MarkdownParser"]
