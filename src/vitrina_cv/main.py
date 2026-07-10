"""FastAPI application entry point for vitrina-cv.

The app is stateless by design (ADR-002): no database, no S3, no external credentials.
The CV engine is initialised once at startup via lifespan and injected into routers
via app.state (task 06-cv-06).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from fastapi import FastAPI

from vitrina_cv.api.routers import extract_geometry, health, preflight
from vitrina_cv.config.settings import get_settings
from vitrina_cv.engines import get_engine
from vitrina_cv.engines.semantic.base import get_semantic_engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialise the CV engine once at startup.

    The engine is stored in app.state so routers can retrieve it without
    reinstantiating it per-request (Strategy pattern, ADR-008).

    The semantic engine (run 11, ADR-004) is optional: CV_SEM_ENGINE empty/off
    keeps app.state.semantic_engine as None and the semantic track disabled
    end-to-end (AC-2 spec-cv-service — objects: [] always in that case).
    """
    settings = get_settings()
    logger.info("Initialising CV engine", extra={"cv_engine": settings.cv_engine})
    engine = get_engine(settings.cv_engine, settings=settings)
    app.state.engine = engine
    logger.info(
        "CV engine ready",
        extra={"cv_engine": settings.cv_engine, "is_ready": engine.is_ready},
    )

    logger.info(
        "Initialising semantic engine", extra={"cv_sem_engine": settings.cv_sem_engine}
    )
    semantic_engine = get_semantic_engine(settings.cv_sem_engine, settings=settings)
    app.state.semantic_engine = semantic_engine
    logger.info(
        "Semantic engine ready",
        extra={
            "cv_sem_engine": settings.cv_sem_engine,
            "is_ready": semantic_engine.is_ready if semantic_engine else None,
        },
    )

    yield
    # Classical OpenCV engine holds no resources; nothing to release.
    logger.info("Shutting down vitrina-cv")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="vitrina-cv",
        version="0.1.0",
        description=(
            "Computer vision sidecar for vitrina. "
            "Extracts deterministic geometry from architectural floor plan PNGs."
        ),
        lifespan=lifespan,
    )

    app.include_router(health.router)
    app.include_router(extract_geometry.router)
    app.include_router(preflight.router)

    return app


app = create_app()
