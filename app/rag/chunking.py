"""Markdown chunking for the offline knowledge base.

Documents are split on level-2 headings; oversized sections are split on
paragraph boundaries. Chunks keep document id and section for traceability.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    document_id: str
    title: str
    section: str
    content: str


def _split_paragraphs(text: str, max_chars: int) -> list[str]:
    paragraphs = [block.strip() for block in text.split("\n\n") if block.strip()]
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
        if len(paragraph) <= max_chars:
            current = paragraph
        else:  # very long paragraph: hard-wrap on sentence-ish boundaries
            for start in range(0, len(paragraph), max_chars):
                pieces.append(paragraph[start : start + max_chars])
            current = ""
    if current:
        pieces.append(current)
    return pieces


def chunk_markdown(
    document_id: str, title: str, text: str, max_chars: int = 1200
) -> list[TextChunk]:
    """Split one knowledge document into retrievable chunks."""
    sections: list[tuple[str, list[str]]] = []
    heading = "Introduction"
    buffer: list[str] = []
    for line in text.splitlines():
        if line.startswith("# "):
            # The H1 is the document title; it is stored separately.
            continue
        if line.startswith("## "):
            if buffer:
                sections.append((heading, buffer))
            heading = line[3:].strip()
            buffer = []
        else:
            buffer.append(line)
    if buffer:
        sections.append((heading, buffer))

    chunks: list[TextChunk] = []
    for section_title, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        for piece in _split_paragraphs(body, max_chars):
            chunks.append(
                TextChunk(
                    document_id=document_id,
                    title=title,
                    section=section_title,
                    content=piece,
                )
            )
    return chunks
