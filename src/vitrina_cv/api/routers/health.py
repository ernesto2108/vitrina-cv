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
    """Healthcheck endpoint — reflects real engine is_ready state (AC-6)."""
    engine = request.app.state.engine
    is_ready: bool = engine.is_ready

    if is_ready:
        logger.debug("Health check: engine ready", extra={"endpoint": "/health"})
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ok", "model_loaded": True},
        )

    logger.warning("Health check: engine not ready", extra={"endpoint": "/health"})
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error_code": ErrorCode.model_not_loaded,
            "message": "CV engine is not ready — warm-up may have failed.",
        },
    )
