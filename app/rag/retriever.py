"""FAISS vector store over the offline knowledge base.

Cosine similarity via normalized inner product. One chunk-level index on disk
(`index.faiss` + `chunks.json`); searches return document-level results,
keeping the best chunk per document.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import faiss

from app.config import get_config
from app.rag.chunking import TextChunk
from app.rag.embeddings import Embedder
from app.rag.schemas import RetrievedDocument

logger = logging.getLogger(__name__)

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"


class IndexNotBuiltError(RuntimeError):
    """The vector index does not exist yet; run `python -m app.cli ingest`."""


class Retriever:
    def __init__(self, embedder: Embedder, index_dir: Path) -> None:
        self.embedder = embedder
        self.index_dir = Path(index_dir)
        self._index = None
        self._chunks: list[TextChunk] = []

    # ------------------------------------------------------------- build/load
    def build(self, chunks: list[TextChunk]) -> int:
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks")
        vectors = self.embedder.encode(
            [f"{chunk.title}. {chunk.section}: {chunk.content}" for chunk in chunks]
        )
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self._index = index
        self._chunks = list(chunks)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_dir / INDEX_FILE))
        self.index_dir.joinpath(CHUNKS_FILE).write_text(
            json.dumps([chunk.__dict__ for chunk in self._chunks], indent=2),
            encoding="utf-8",
        )
        logger.info("indexed %d chunks into %s", len(chunks), self.index_dir)
        return len(chunks)

    def load(self) -> Retriever:
        index_path = self.index_dir / INDEX_FILE
        chunks_path = self.index_dir / CHUNKS_FILE
        if not index_path.exists() or not chunks_path.exists():
            raise IndexNotBuiltError(
                f"Vector index not found in {self.index_dir}. "
                "Build it offline with: python -m app.cli ingest"
            )
        self._index = faiss.read_index(str(index_path))
        raw = json.loads(chunks_path.read_text(encoding="utf-8"))
        self._chunks = [TextChunk(**item) for item in raw]
        return self

    # ---------------------------------------------------------------- search
    def search(self, query: str, top_k: int, oversample: int = 3) -> list[RetrievedDocument]:
        if self._index is None:
            self.load()
        assert self._index is not None
        vector = self.embedder.encode_query(query).reshape(1, -1)
        distances, indices = self._index.search(vector, min(top_k * oversample, self._index.ntotal))
        best_per_doc: dict[str, RetrievedDocument] = {}
        for score, position in zip(distances[0], indices[0]):
            if position < 0:
                continue
            chunk = self._chunks[int(position)]
            current = best_per_doc.get(chunk.document_id)
            item = RetrievedDocument(
                document_id=chunk.document_id,
                title=chunk.title,
                content=chunk.content,
                score=round(float(score), 4),
            )
            if current is None or item.score > current.score:
                best_per_doc[chunk.document_id] = item
        ordered = sorted(best_per_doc.values(), key=lambda item: item.score, reverse=True)
        return ordered[:top_k]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Load (never build) the process-wide retriever: index and embedder load once."""
    cfg = get_config()
    return Retriever(Embedder(cfg.embedding), cfg.retrieval.resolve_index_dir()).load()
