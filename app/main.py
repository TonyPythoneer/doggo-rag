"""FastAPI application factory.

Handlers stay thin: business logic lives in the vision, rag and pipeline
components.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import images, pipeline
from app.config import get_config

API_PREFIX = "/api/v1"


def create_app() -> FastAPI:
    app = FastAPI(
        title="DOGGO RAG",
        description=(
            "Local-first Vision + Text RAG evaluation system: "
            "does retrieved knowledge improve dog vs non-dog classification?"
        ),
        version=__version__,
    )
    # The React dev server (localhost:3000) calls the API directly; allow it.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_config().server.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(images.router)
    app.include_router(pipeline.router)

    @app.get(f"{API_PREFIX}/health", tags=["health"])
    def health() -> dict:
        from app.rag.retriever import INDEX_FILE
        from app.vision.model import get_vision_model

        cfg = get_config()
        index_dir = cfg.retrieval.resolve_index_dir()
        index_ready = (index_dir / INDEX_FILE).exists()
        return {
            "status": "ok",
            "vision_model": cfg.vision.model_id,
            "vision_model_loaded": get_vision_model().is_loaded,
            "embedding_model": cfg.embedding.model_name,
            "index_ready": index_ready,
        }

    frontend_build = Path(__file__).resolve().parent.parent / "frontend" / "build"
    if frontend_build.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=frontend_build, html=True),
            name="frontend",
        )

    return app


app = create_app()
