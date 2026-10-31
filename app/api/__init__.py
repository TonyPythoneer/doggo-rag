"""HTTP API shared pieces."""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException

from app.config import get_config


def require_api_key(x_api_key: str = Header(default="")) -> None:
    """Guard the endpoints that run the VLM or accept uploads."""
    expected = get_config().server.api_key
    if not expected:  # fail closed: never run unauthenticated by misconfiguration
        raise HTTPException(status_code=503, detail="DOGGO_API_KEY is not set on the server")
    if not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
