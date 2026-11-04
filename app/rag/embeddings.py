"""Local text embeddings (sentence-transformers). No CLIP/SigLIP, no cloud."""

from __future__ import annotations

import logging

import numpy as np

from app.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class Embedder:
    """Lazy wrapper around a local sentence-transformers model."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading embedding model %s", self.config.model_name)
            self._model = SentenceTransformer(self.config.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into a float32 matrix of shape (len(texts), dim)."""
        model = self._ensure_loaded()
        vectors = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])[0]
