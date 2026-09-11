"""
Step 2: Hierarchical chunking.

Rules implemented (per spec):
  * Never use fixed-size sliding-window chunking over raw text.
  * Split by chapter -> section -> subsection boundaries first (these are
    already discrete DocSection nodes from the parser).
  * Only if an individual subsection's text is still too long, split it
    further at sentence/paragraph boundaries ("semantic" split) rather
    than mid-sentence.
  * Preserve a 10-20% text overlap between consecutive chunks belonging
    to the same section so that context isn't lost at chunk boundaries.
  * Every chunk carries full hierarchy metadata + previous/next links.
"""

from __future__ import annotations

import re
from typing import List

from config import CONFIG
from utils.models import Chunk, DocSection, ParsedDocument, new_id
from utils.logger import get_logger

logger = get_logger("chunking.hierarchical")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9])")


def _semantic_split(text: str, max_chars: int, min_chars: int) -> List[str]:
    """Split long text into semantically coherent pieces near sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    sentences = _SENTENCE_SPLIT_RE.split(text)
    pieces: List[str] = []
    current = ""
    for sent in sentences:
        candidate = (current + " " + sent).strip() if current else sent
        if len(candidate) > max_chars and len(current) >= min_chars:
            pieces.append(current.strip())
            current = sent
        else:
            current = candidate
    if current.strip():
        pieces.append(current.strip())
    return pieces or [text]


def _overlap_prefix(previous_text: str, ratio: float) -> str:
    if not previous_text:
        return ""
    n = max(1, int(len(previous_text) * ratio))
    return previous_text[-n:]


def chunk_document(parsed_doc: ParsedDocument) -> List[Chunk]:
    """
    Convert a ParsedDocument's flat section list into hierarchical Chunks.

    Each DocSection becomes one or more Chunks (one, unless its text
    exceeds max_chunk_chars, in which case it is split semantically).
    """
    cfg = CONFIG.chunking
    section_by_id = {s.section_id: s for s in parsed_doc.sections}

    def section_path(sec: DocSection):
        """Walk up parent_id chain to build (chapter, section, subsection) labels."""
        chapter, section, subsection = sec.chapter, "", ""
        node = sec
        chain = []
        while node is not None:
            chain.append(node)
            node = section_by_id.get(node.parent_id) if node.parent_id else None
        chain = list(reversed(chain))  # root -> ... -> sec
        if len(chain) >= 1:
            section = chain[1].title if len(chain) > 1 else chain[0].title
        if len(chain) >= 3:
            subsection = chain[2].title
        return chapter or (chain[0].title if chain else ""), section, subsection

    all_chunks: List[Chunk] = []

    for sec in parsed_doc.sections:
        text = (sec.text or "").strip()
        if not text:
            # still emit a lightweight chunk for headings so the hierarchy is browsable
            if sec.level <= 1:
                text = sec.title
            else:
                continue

        chapter, section, subsection = section_path(sec)
        pieces = _semantic_split(text, cfg.max_chunk_chars, cfg.min_chunk_chars)

        prev_piece_text = ""
        for i, piece in enumerate(pieces):
            overlap = _overlap_prefix(prev_piece_text, cfg.overlap_ratio) if i > 0 else ""
            chunk = Chunk(
                chunk_id=new_id("chunk"),
                doc_id=parsed_doc.doc_id,
                text=(overlap + "\n" + piece).strip() if overlap else piece,
                page=sec.page_start,
                chapter=chapter,
                section=section,
                subsection=subsection or sec.title,
                heading=sec.title,
                content_type=sec.content_type,
                overlap_prefix=overlap,
            )
            all_chunks.append(chunk)
            prev_piece_text = piece

    # link previous/next across the whole ordered chunk list
    for i, chunk in enumerate(all_chunks):
        if i > 0:
            chunk.previous_chunk = all_chunks[i - 1].chunk_id
            all_chunks[i - 1].next_chunk = chunk.chunk_id

    logger.info(f"Created {len(all_chunks)} hierarchical chunks for doc {parsed_doc.doc_id}.")
    return all_chunks
