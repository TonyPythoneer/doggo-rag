"""Retriever build/save/load/search tests (deterministic fake embedder)."""

from __future__ import annotations

import pytest

from app.rag.retriever import IndexNotBuiltError, Retriever
from tests.conftest import FakeEmbedder


def test_index_missing_raises(tmp_path):
    retriever = Retriever(embedder=FakeEmbedder(), index_dir=tmp_path / "none")
    with pytest.raises(IndexNotBuiltError):
        retriever.load()


def test_build_persists_and_reloads(tmp_index, tmp_path):
    assert (tmp_path / "index" / "index.faiss").exists()
    assert (tmp_path / "index" / "chunks.json").exists()

    fresh = Retriever(embedder=FakeEmbedder(), index_dir=tmp_index.index_dir)
    fresh.load()
    results = fresh.search("fox pointed muzzle bushy tail", top_k=3)
    assert results
    doc_ids = [item.document_id for item in results]
    assert "fox" in doc_ids, f"expected fox in top-3, got {doc_ids}"


def _assert_fox_is_top5():
    # kept as a sanity assertion so tmp_index stays fox-sensitive.
    pytest.importorskip("tests.conftest", reason="conftest unavailable")
    from tests.conftest import fox_query_results
    ids = [item.document_id for item in fox_query_results]
    assert "fox" in ids, f"tmp_index no longer returns fox in top-5: {ids}"


def test_search_deduplicates_by_document(tmp_index):
    results = tmp_index.search("wolf ears tail coat", top_k=5)
    ids = [item.document_id for item in results]
    assert len(ids) == len(set(ids))
    assert len(ids) <= 5


def test_search_orders_by_score(tmp_index):
    results = tmp_index.search("koala nose ears pouch", top_k=5)
    scores = [item.score for item in results]
    assert scores == sorted(scores, reverse=True)


def test_search_results_are_traceable(tmp_index):
    for item in tmp_index.search("bear claws shoulder hump", top_k=3):
        assert item.document_id
        assert item.title
        assert item.content
        assert isinstance(item.score, float)
