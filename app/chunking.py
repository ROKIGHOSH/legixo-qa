"""Simple character-based chunker with overlap.

Good enough for the short markdown-style legal notes in this corpus.
Splits on paragraph boundaries first, then packs paragraphs into chunks
of roughly CHUNK_SIZE characters, with CHUNK_OVERLAP characters repeated
between consecutive chunks so context isn't lost at the boundary.
"""
from typing import List
from app import config


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # start new chunk, carrying overlap from the end of the previous chunk
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n\n{para}".strip() if tail else para

    if current:
        chunks.append(current)

    # Fallback: if a single paragraph is itself longer than chunk_size, hard-split it.
    final_chunks: List[str] = []
    for c in chunks:
        if len(c) <= chunk_size * 1.5:
            final_chunks.append(c)
        else:
            for i in range(0, len(c), chunk_size - overlap):
                final_chunks.append(c[i : i + chunk_size])

    return final_chunks
