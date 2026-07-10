"""Router: GET /health — liveness / readiness probe.

Returns 200 {status: "ok", model_loaded: true} when the engine is ready,
503 {error_code: "model_not_loaded", ...} otherwise (ADR-003, AC-6).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from vitrina_cv.models import ErrorCode

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)


@router.get(
    "/health",
    summary="Healthcheck (liveness/readiness)",
    description=(
        "Returns 200 when the active CV engine is initialised and ready. "
        "Returns 503 if the engine is not ready (model_not_loaded). "
        "Conforms to cv-service.openapi.yaml (ADR-003)."
    ),
)
async def get_health(request: Request) -> JSONResponse:
    """Healthcheck endpoint — reflects real engine is_ready state (AC-6).

    model_loaded / 503 gating is driven solely by the geometric GeometryEngine
    (engine.is_ready) — that engine is mandatory for the service to be usable
    at all. The semantic engine (run 11, ADR-004) is optional and additive:
    when CV_SEM_ENGINE is off, semantic_engine is None and must not affect
    liveness/readiness. When a semantic engine IS configured, its own
    is_ready is surfaced as a separate `semantic_model_loaded` field so
    callers can observe semantic warm-up without the semantic track ever
    being able to force the whole service into 503 (it is best-effort by
    design — see extract_geometry.py error handling).
    """
    engine = request.app.state.engine
    is_ready: bool = engine.is_ready

    semantic_engine = getattr(request.app.state, "semantic_engine", None)
    semantic_model_loaded: bool | None = (
        semantic_engine.is_ready if semantic_engine is not None else None
    )

    if is_ready:
        logger.debug(
            "Health check: engine ready",
            extra={
                "endpoint": "/health",
                "semantic_model_loaded": semantic_model_loaded,
            },
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "ok",
                "model_loaded": True,
                "semantic_model_loaded": semantic_model_loaded,
            },
        )

    logger.warning("Health check: engine not ready", extra={"endpoint": "/health"})
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error_code": ErrorCode.model_not_loaded,
            "message": "CV engine is not ready — warm-up may have failed.",
        },
    )
