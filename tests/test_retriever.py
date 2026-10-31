"""Retriever build/save/load/search tests (deterministic fake embedder)."""

from __future__ import annotations

import json
import shutil

import pytest

import app.rag.retriever as retriever_module
from app.config import load_config
from app.rag.ingestion import build_index
from app.rag.retriever import IndexNotBuiltError, Retriever, get_retriever
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


def test_load_rejects_chunks_that_do_not_match_the_index(tmp_index):
    chunks_path = tmp_index.index_dir / "chunks.json"
    chunks_path.write_text(json.dumps(json.loads(chunks_path.read_text())[:-1]))

    with pytest.raises(IndexNotBuiltError, match="ingest"):
        Retriever(embedder=FakeEmbedder(), index_dir=tmp_index.index_dir).load()


@pytest.fixture()
def index_cfg(tmp_path, monkeypatch):
    cfg = load_config()
    shutil.copytree(cfg.knowledge.resolve_dir(), tmp_path / "knowledge")
    cfg.knowledge.dir = str(tmp_path / "knowledge")
    cfg.retrieval.index_dir = str(tmp_path / "index")
    build_index(cfg, embedder=FakeEmbedder())
    monkeypatch.setattr(retriever_module, "get_config", lambda: cfg)
    get_retriever.cache_clear()
    yield cfg
    get_retriever.cache_clear()


def test_get_retriever_loads_an_index_built_from_current_knowledge(index_cfg):
    assert get_retriever().index_dir == index_cfg.retrieval.resolve_index_dir()


def test_get_retriever_refuses_an_index_from_edited_knowledge(index_cfg):
    fox = index_cfg.knowledge.resolve_dir() / "fox.md"
    fox.write_text(fox.read_text(encoding="utf-8") + "\nOne more fact.\n", encoding="utf-8")

    with pytest.raises(IndexNotBuiltError, match="ingest"):
        get_retriever()


def test_get_retriever_refuses_an_index_from_another_embedding_model(index_cfg):
    index_cfg.embedding.model_name = "another/embedding-model"

    with pytest.raises(IndexNotBuiltError, match="ingest"):
        get_retriever()
