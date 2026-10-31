"""Offline knowledge ingestion.

Loads the 11 species documents from `knowledge/`, chunks them, embeds them and
writes the FAISS index. This is a separate offline operation (CLI `ingest`);
it never runs inside a request handler and never adapts to incoming images.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import SPECIES_DOCUMENT_IDS, AppConfig, get_config
from app.rag.chunking import TextChunk, chunk_markdown
from app.rag.embeddings import Embedder
from app.rag.retriever import Retriever

BUILD_STATS_FILE = "build_stats.json"


class KnowledgeBaseError(RuntimeError):
    pass


def _parse_document(document_id: str, text: str) -> tuple[str, str]:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip(), text
    raise KnowledgeBaseError(f"knowledge/{document_id}.md has no level-1 title")


def load_documents(knowledge_dir: Path) -> list[tuple[str, str, str]]:
    """Return [(document_id, title, raw_markdown)] for every expected document."""
    missing = [
        doc_id for doc_id in SPECIES_DOCUMENT_IDS if not (knowledge_dir / f"{doc_id}.md").exists()
    ]
    if missing:
        raise KnowledgeBaseError(f"Missing knowledge documents: {missing}")
    documents = []
    for doc_id in SPECIES_DOCUMENT_IDS:
        text = (knowledge_dir / f"{doc_id}.md").read_text(encoding="utf-8")
        title, raw = _parse_document(doc_id, text)
        documents.append((doc_id, title, raw))
    return documents


def build_chunks(cfg: AppConfig | None = None) -> list[TextChunk]:
    cfg = cfg or get_config()
    chunks: list[TextChunk] = []
    for doc_id, title, raw in load_documents(cfg.knowledge.resolve_dir()):
        chunks.extend(chunk_markdown(doc_id, title, raw, cfg.knowledge.chunk_max_chars))
    if not chunks:
        raise KnowledgeBaseError("No chunks produced from the knowledge base")
    return chunks


def build_index(
    cfg: AppConfig | None = None,
    index_dir: Path | None = None,
    embedder: Embedder | None = None,
) -> dict:
    """Chunk → embed → index → persist. Returns build statistics."""
    cfg = cfg or get_config()
    index_dir = index_dir or cfg.retrieval.resolve_index_dir()
    chunks = build_chunks(cfg)
    Retriever(embedder=embedder or Embedder(cfg.embedding), index_dir=index_dir).build(chunks)
    stats = {
        "documents": len(SPECIES_DOCUMENT_IDS),
        "chunks": len(chunks),
        "embedding": cfg.embedding.model_dump(),  # checked by get_retriever
        "index_dir": str(index_dir),
    }
    (index_dir / BUILD_STATS_FILE).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
