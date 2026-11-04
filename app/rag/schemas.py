"""RAG request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel


class RetrievedDocument(BaseModel):
    """One retrieved knowledge unit. `score` is retrieval relevance only —
    it is not a classification confidence."""

    document_id: str
    title: str
    content: str
    score: float
