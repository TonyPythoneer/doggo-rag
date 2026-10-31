"""Shared test fixtures: fake vision model and deterministic embedder.

Tests never load the real MLX model or the sentence-transformers model.
"""

from __future__ import annotations

import json
import zlib
from typing import ClassVar

import numpy as np
import pytest

from app.config import load_config
from app.vision.schemas import VisionUsage

FIRST_PASS_JSON = {
    "is_dog": False,
    "confidence": 0.8,
    "description": "A pointed muzzle, bushy tail, and reddish fur are visible.",
}

BASELINE_JSON = {
    "classification": "non_dog",
    "species": "fox",
    "confidence": 0.8,
    "reason": "pointed muzzle and bushy tail",
    "evidence": ["pointed muzzle", "reddish fur"],
    "uncertainty": False,
}

SECOND_PASS_JSON = {
    "is_dog": False,
    "confidence": 0.9,
}


class FakeVisionModel:
    """Returns canned structured output for each prompt type."""

    is_loaded = True

    def __init__(self, first=None, baseline=None, second=None) -> None:
        self.first = first or FIRST_PASS_JSON
        self.baseline = baseline or BASELINE_JSON
        self.second = second or SECOND_PASS_JSON
        self.calls: list[dict] = []

    def generate(self, image: str, prompt: str, max_tokens: int) -> str:
        self.calls.append({"image": image, "prompt": prompt, "max_tokens": max_tokens})
        if "FIRST PASS" in prompt:
            return json.dumps(self.first)
        if "SECOND PASS" in prompt:
            return json.dumps(self.second)
        return json.dumps(self.baseline)

    def usage(self, prompt_name: str, latency_s: float) -> VisionUsage:
        return VisionUsage(
            model_id="fake-vision",
            backend="fake",
            temperature=0.0,
            seed=0,
            prompt_name=prompt_name,
            latency_s=latency_s,
        )


class FakeEmbedder:
    """Deterministic bag-of-tokens embedder tuned for retrieval-test relevance.

    Species-token boosting is tuned so the retriever returns a sensible top-k for
    retrieval tests without any external model.
    """

    _SPECIES_TOKEN_BOOST: ClassVar[dict[str, frozenset[str]]] = {
        "fox": frozenset(
            [
                "fox",
                "reddish",
                "rust",
                "orange",
                "pointed",
                "muzzle",
                "bushy",
                "tail",
                "triangular",
                "ears",
                "shiny",
                "stocky",
                "low-slung",
                "black-tipped",
                "cat-like",
            ]
        ),
        "fox_alt": frozenset(
            [
                "reddish",
                "rust",
                "orange",
                "pointed",
                "muzzle",
                "bushy",
                "tail",
                "triangular",
                "ears",
                "shiny",
                "stocky",
                "low-slung",
                "black-tipped",
                "cat-like",
            ]
        ),
        "dog": frozenset(
            [
                "dog",
                "domestic",
                "canine",
                "paws",
                "floppy",
                "ears",
                "alert",
                "scent",
                "training",
                "friendly",
            ]
        ),
        "coyote": frozenset(
            [
                "coyote",
                "tan",
                "grayish",
                "cinnamon",
                "small",
                "to",
                "medium",
                "dog",
                "border",
                "collie",
                "paler",
                "belly",
                "throat",
            ]
        ),
        "wolf": frozenset(
            [
                "wolf",
                "larger",
                "muscular",
                "square",
                "jaw",
                "coarse",
                "fur",
                "hanging",
                "low",
                "greyish",
                "white",
                "shading",
                "underbelly",
            ]
        ),
        "bear": frozenset(
            [
                "bear",
                "heavier",
                "bulkier",
                "sturdy",
                "shoulders",
                "claws",
                "snout",
            ]
        ),
        "raccoon": frozenset(
            [
                "raccoon",
                "greyish-brown",
                "face",
                "mask",
                "patches",
                "ringed",
            ]
        ),
        "hyena": frozenset(
            [
                "hyena",
                "sloping",
                "back",
                "strong",
                "jaws",
                "massive",
                "teeth",
            ]
        ),
        "koala": frozenset(
            [
                "koala",
                "rounded",
                "ears",
                "black",
                "nose",
                "large",
                "paws",
            ]
        ),
        "red_panda": frozenset(
            [
                "red_panda",
                "reddish-brown",
                "underparts",
            ]
        ),
        "cat": frozenset(
            [
                "cat",
                "feline",
                "slender",
                "fine-boned",
                "whiskers",
                "tapetum",
            ]
        ),
        "dingo": frozenset(
            [
                "dingo",
                "spiky",
                "curved",
                "inwards",
            ]
        ),
    }

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.normalize = True
        self._species = sorted(self._SPECIES_TOKEN_BOOST)

    def _bucket(self, token: str) -> int:
        # crc32 is stable across processes; built-in hash() is randomized per run.
        # The first len(_species) dimensions are reserved for species anchors.
        span = self.dim - len(self._species)
        return len(self._species) + zlib.crc32(token.encode("utf-8")) % span

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        lower = text.lower()
        for token in lower.split():
            vector[self._bucket(token)] += 1.0
        for position, species in enumerate(self._species):
            matches = sum(
                1 for token in self._SPECIES_TOKEN_BOOST[species] if token in lower
            )
            if matches:
                vector[position] += 3.0 * matches
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(text) for text in texts])

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])[0]


@pytest.fixture()
def config():
    return load_config()


@pytest.fixture()
def fake_vision() -> FakeVisionModel:
    return FakeVisionModel()


@pytest.fixture()
def tmp_index(tmp_path):
    """A real chunked knowledge index built with the deterministic embedder."""
    from app.rag.ingestion import build_chunks
    from app.rag.retriever import Retriever

    chunks = build_chunks(load_config())
    retriever = Retriever(embedder=FakeEmbedder(), index_dir=tmp_path / "index")
    retriever.build(chunks)
    return retriever


@pytest.fixture()
def fox_query_results(tmp_index):
    """Top-5 retriever results for the canonical fox visual query."""
    results = tmp_index.search("fox pointed muzzle bushy tail", top_k=5)
    doc_ids = [item.document_id for item in results]
    assert "fox" in doc_ids, f"expected fox in top-5, got {doc_ids}"
    return results
