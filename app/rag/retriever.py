"""Hybrid retrieval over the offline knowledge base: FAISS vectors plus BM25.

Cosine similarity via normalized inner product over one chunk-level index on disk
(`index.faiss` + `chunks.json`), and BM25 over whole documents. The two document
rankings are fused with reciprocal rank fusion. Each result carries its document's
"Visual identification" card, the field marks the Second Pass checks in the image.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
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
CARD_SECTION = "Visual identification"
# The Second Pass tests "is this a domestic dog?", so the dog document is always shown
# beside the look-alike species that rank highest for the description.
ALWAYS_RETRIEVED = "dog"
RRF_K = 60  # the usual reciprocal rank fusion constant


class IndexNotBuiltError(RuntimeError):
    """The vector index does not exist yet; run `python -m app.cli ingest`."""


def _terms(text: str) -> list[str]:
    # Plural "s" stripped so "ears" in a description matches "ear" in a document.
    return [word.removesuffix("s") for word in re.findall(r"[a-z]+", text.lower())]


class BM25:
    """Okapi BM25 over whole documents: rare words ("mask", "clinging") outweigh common ones."""

    def __init__(self, documents: dict[str, str], k1: float = 1.2, b: float = 0.75) -> None:
        self.counts = {doc_id: Counter(_terms(text)) for doc_id, text in documents.items()}
        self.lengths = {doc_id: sum(counts.values()) for doc_id, counts in self.counts.items()}
        self.average = sum(self.lengths.values()) / len(self.lengths)
        frequency = Counter(term for counts in self.counts.values() for term in counts)
        total = len(self.counts)
        self.idf = {t: math.log(1 + (total - n + 0.5) / (n + 0.5)) for t, n in frequency.items()}
        self.k1, self.b = k1, b

    def rank(self, query: str) -> list[str]:
        terms = _terms(query)
        scores = {}
        for doc_id, counts in self.counts.items():
            norm = self.k1 * (1 - self.b + self.b * self.lengths[doc_id] / self.average)
            scores[doc_id] = sum(
                self.idf[t] * counts[t] * (self.k1 + 1) / (counts[t] + norm)
                for t in terms
                if counts[t]
            )
        return sorted(scores, key=scores.__getitem__, reverse=True)


class Retriever:
    def __init__(self, embedder: Embedder, index_dir: Path) -> None:
        self.embedder = embedder
        self.index_dir = Path(index_dir)
        self._index = None
        self._chunks: list[TextChunk] = []
        self._bm25: BM25 | None = None

    def build(self, chunks: list[TextChunk]) -> int:
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks")
        vectors = self.embedder.encode(
            [f"{chunk.title}. {chunk.section}: {chunk.content}" for chunk in chunks]
        )
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self._index = index
        self._chunks, self._bm25 = list(chunks), None
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
        index = faiss.read_index(str(index_path))
        raw = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [TextChunk(**item) for item in raw]
        if index.ntotal != len(chunks):
            raise IndexNotBuiltError(
                f"{index_path} holds {index.ntotal} vectors but {chunks_path} lists "
                f"{len(chunks)} chunks. Rebuild it with: python -m app.cli ingest"
            )
        self._index, self._chunks, self._bm25 = index, chunks, None
        return self

    def search(self, query: str, top_k: int) -> list[RetrievedDocument]:
        """The `top_k` documents by fused dense and BM25 rank, always including the dog."""
        if self._index is None:
            self.load()
        assert self._index is not None
        if self._bm25 is None:
            texts: dict[str, str] = {}
            for chunk in self._chunks:
                texts[chunk.document_id] = (
                    f"{texts.get(chunk.document_id, chunk.title)} {chunk.content}"
                )
            self._bm25 = BM25(texts)
        vector = self.embedder.encode_query(query).reshape(1, -1)
        _, positions = self._index.search(vector, self._index.ntotal)
        best: dict[str, TextChunk] = {}  # each document's best chunk, in dense rank order
        for position in positions[0]:
            if position >= 0:
                chunk = self._chunks[int(position)]
                best.setdefault(chunk.document_id, chunk)
        fused: Counter[str] = Counter()
        for ranking in (list(best), self._bm25.rank(query)):
            for rank, doc_id in enumerate(ranking, start=1):
                fused[doc_id] += 1 / (RRF_K + rank)
        chosen = [doc_id for doc_id, _ in fused.most_common(top_k)]
        if ALWAYS_RETRIEVED in fused and ALWAYS_RETRIEVED not in chosen:
            chosen[-1] = ALWAYS_RETRIEVED
        # The card, not the best chunk: the dog's best chunk was nearly always its generic
        # facial features, never its domestic marks (floppy ears, collar, ...). Cards once made
        # qwen3-vl-2b write "true" as a string; schema-constrained decoding now rules that out.
        cards = {c.document_id: c for c in self._chunks if c.section == CARD_SECTION}
        return [
            RetrievedDocument(
                document_id=doc_id,
                title=cards[doc_id].title,
                content=cards[doc_id].content,
                score=round(fused[doc_id], 4),
            )
            for doc_id in sorted(chosen, key=fused.__getitem__, reverse=True)
        ]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Load (never build) the process-wide retriever: index and embedder load once.

    Refuses an index built from other knowledge, chunking or embedding settings:
    only `make eval` rebuilds it, so the CLI and the API could otherwise use a stale one.
    """
    from app.rag.ingestion import BUILD_STATS_FILE, build_chunks  # ingestion imports this module

    cfg = get_config()
    retriever = Retriever(Embedder(cfg.embedding), cfg.retrieval.resolve_index_dir()).load()
    stats_path = retriever.index_dir / BUILD_STATS_FILE
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    other_embedding = stats.get("embedding") != cfg.embedding.model_dump()
    if other_embedding or retriever._chunks != build_chunks(cfg):
        raise IndexNotBuiltError(
            f"The vector index in {retriever.index_dir} does not match the current knowledge "
            "or embedding settings. Rebuild it with: python -m app.cli ingest"
        )
    return retriever
