"""Chunking and ingestion tests."""

from __future__ import annotations

from app.rag.chunking import chunk_markdown
from app.rag.ingestion import SPECIES_DOCUMENT_IDS, build_chunks, load_documents

DOC = """# Title

Intro paragraph.

## Physical characteristics

Alpha content.

## Tail

Beta content.
"""


def test_chunk_markdown_splits_on_headings():
    chunks = chunk_markdown("fox", "Title", DOC)
    assert [chunk.section for chunk in chunks] == [
        "Introduction",
        "Physical characteristics",
        "Tail",
    ]
    assert chunks[0].content == "Intro paragraph."
    assert all(chunk.document_id == "fox" for chunk in chunks)
    assert all(chunk.title == "Title" for chunk in chunks)


def test_chunk_markdown_splits_oversized_sections():
    big = "# T\n\n## Sec\n\n" + "\n\n".join(f"paragraph {i} " + "x" * 200 for i in range(20))
    chunks = chunk_markdown("t", "T", big)
    assert len(chunks) > 1
    assert all(len(chunk.content) <= 1400 for chunk in chunks)


def test_all_species_documents_load(config):
    documents = load_documents(config.knowledge.resolve_dir())
    ids = [doc_id for doc_id, _, _ in documents]
    assert ids == SPECIES_DOCUMENT_IDS
    for _, title, raw in documents:
        assert title.startswith(
            (
                "Domestic Dog",
                "Gray Wolf",
                "Red Fox",
                "Coyote",
                "Dingo",
                "Domestic Cat",
                "Koala",
                "Red Panda",
                "Raccoon",
                "Bear",
                "Hyena",
            )
        )
        assert "## Similarities to domestic dogs" in raw
        assert "## Differences from domestic dogs" in raw


def test_build_chunks_covers_every_document(config):
    chunks = build_chunks(config)
    assert {chunk.document_id for chunk in chunks} == set(SPECIES_DOCUMENT_IDS)
    assert len(chunks) >= len(SPECIES_DOCUMENT_IDS)  # at least one chunk each
