"""Router: POST /preflight — deterministic image quality gate (ADR-005).

Accepts a multipart/form-data request with an 'image' field (PNG bytes).
Runs heuristic checks and returns a PreflightReport.

Error taxonomy (ADR-003, AC-7):
  400 invalid_request    — multipart malformed or 'image' field missing
  422 unprocessable_image — image bytes cannot be decoded (corrupt / wrong format)
  503 model_not_loaded   — engine not ready at request time
"""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile, status
from fastapi.responses import JSONResponse

from vitrina_cv.config.settings import Settings, get_settings
from vitrina_cv.models import ErrorCode, PreflightReport
from vitrina_cv.preflight.checks import run_preflight

router = APIRouter(tags=["preflight"])
logger = logging.getLogger(__name__)


@router.post(
    "/preflight",
    response_model=PreflightReport,
    summary="Gate de pre-vuelo fail-fast (ADR-005)",
    description=(
        "Runs deterministic heuristics on a PNG to evaluate fitness for geometry "
        "extraction: resolution, contrast and line density. "
        "No LLM involved (ADR-005). "
        "Conforms to cv-service.openapi.yaml (ADR-003)."
    ),
)
async def preflight(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    image: Annotated[
        UploadFile | None, File(description="PNG floor plan image")
    ] = None,
) -> JSONResponse:
    """Preflight gate — real implementation (AC-7)."""
    t_start = time.monotonic()
    endpoint = "/preflight"

    # 503 — engine not ready (same gate as /extract-geometry per spec AC-7)
    engine = request.app.state.engine
    if not engine.is_ready:
        logger.warning(
            "Engine not ready",
            extra={"endpoint": endpoint, "error_code": ErrorCode.model_not_loaded},
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error_code": ErrorCode.model_not_loaded,
                "message": "CV engine is not ready. Try again in a moment.",
            },
        )

    # 400 — missing 'image' field in multipart request
    if image is None:
        logger.warning(
            "Missing image field",
            extra={"endpoint": endpoint, "error_code": ErrorCode.invalid_request},
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error_code": ErrorCode.invalid_request,
                "message": "Multipart field 'image' is required but was not sent.",
            },
        )

    # 400 — read image bytes
    try:
        image_bytes = await image.read()
    except Exception:
        logger.exception(
            "Failed to read multipart image field",
            extra={"endpoint": endpoint, "error_code": ErrorCode.invalid_request},
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error_code": ErrorCode.invalid_request,
                "message": "Could not read the 'image' field from the multipart request.",
            },
        )

    if not image_bytes:
        logger.warning(
            "Empty image field",
            extra={
                "endpoint": endpoint,
                "image_size": 0,
                "error_code": ErrorCode.invalid_request,
            },
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error_code": ErrorCode.invalid_request,
                "message": "The 'image' field is empty. Send a valid PNG file.",
            },
        )

    image_size = len(image_bytes)
    t_read = time.monotonic()

    # 422 — image decode failure inside run_preflight
    try:
        report: PreflightReport = run_preflight(image_bytes, settings)
    except ValueError as exc:
        duration_ms = round((time.monotonic() - t_start) * 1000, 1)
        logger.warning(
            "Unprocessable image",
            extra={
                "endpoint": endpoint,
                "image_size": image_size,
                "duration_ms": duration_ms,
                "error_code": ErrorCode.unprocessable_image,
                "detail": str(exc),
            },
        )
        return JSONResponse(
            status_code=422,
            content={
                "error_code": ErrorCode.unprocessable_image,
                "message": f"Image could not be decoded: {exc}",
            },
        )

    t_check = time.monotonic()
    duration_read_ms = round((t_read - t_start) * 1000, 1)
    duration_check_ms = round((t_check - t_read) * 1000, 1)
    duration_total_ms = round((t_check - t_start) * 1000, 1)

    logger.info(
        "Preflight complete",
        extra={
            "endpoint": endpoint,
            "image_size": image_size,
            "duration_read_ms": duration_read_ms,
            "duration_check_ms": duration_check_ms,
            "duration_total_ms": duration_total_ms,
            "is_floor_plan": report.is_floor_plan,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=report.model_dump(mode="json"),
    )
