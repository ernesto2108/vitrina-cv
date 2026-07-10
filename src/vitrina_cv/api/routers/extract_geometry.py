"""Router: POST /extract-geometry — floor plan geometry extraction.

Accepts a multipart/form-data request with an 'image' field (PNG bytes).
Delegates to the active GeometryEngine (ADR-008) and returns a Geometry payload.

Error taxonomy (ADR-003, AC-7):
  400 invalid_request    — multipart malformed or 'image' field missing
  422 unprocessable_image — image bytes cannot be decoded (corrupt / wrong format)
  503 model_not_loaded   — engine not ready at request time
"""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, File, Request, UploadFile, status
from fastapi.responses import JSONResponse

from vitrina_cv.engines.semantic.merge import merge_semantic
from vitrina_cv.models import ErrorCode, Geometry

router = APIRouter(tags=["geometry"])
logger = logging.getLogger(__name__)


@router.post(
    "/extract-geometry",
    response_model=Geometry,
    summary="Extract geometry from a floor plan PNG",
    description=(
        "Accepts a PNG floor plan via multipart/form-data (field: 'image') and returns "
        "walls, rooms, openings and optional scale in pixels of the received image. "
        "Conforms to cv-service.openapi.yaml (ADR-003)."
    ),
)
async def extract_geometry(
    request: Request,
    image: Annotated[
        UploadFile | None, File(description="PNG floor plan image")
    ] = None,
) -> JSONResponse:
    """Extract geometry — real implementation (AC-1, AC-7)."""
    t_start = time.monotonic()
    endpoint = "/extract-geometry"

    # 503 — engine not ready
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

    # 400 — read image bytes; UploadFile.read() raises on network issues.
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

    # 422 — image decode or extraction failure
    try:
        geometry: Geometry = engine.extract(image_bytes)
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
                "message": f"Image could not be processed: {exc}",
            },
        )

    t_extract = time.monotonic()
    duration_read_ms = round((t_read - t_start) * 1000, 1)
    duration_extract_ms = round((t_extract - t_read) * 1000, 1)
    duration_total_ms = round((t_extract - t_start) * 1000, 1)

    logger.info(
        "Geometry extracted",
        extra={
            "endpoint": endpoint,
            "image_size": image_size,
            "duration_read_ms": duration_read_ms,
            "duration_extract_ms": duration_extract_ms,
            "duration_total_ms": duration_total_ms,
            "walls": len(geometry.walls),
            "rooms": len(geometry.rooms),
            "openings": len(geometry.openings),
        },
    )

    # Semantic track (run 11, ADR-004) — aditivo y best-effort. Cuando
    # CV_SEM_ENGINE está off, semantic_engine es None y objects se queda en
    # el default de Geometry ([]), sin overhead ni cambio de comportamiento
    # (AC-2 spec-cv-service: walls/rooms/openings/stairs_candidates idénticos
    # al baseline byte a byte).
    #
    # Manejo de errores: si el motor semántico falla en inferencia, la
    # extracción geométrica ya es válida y completa — degradar a objects: []
    # y loguear el error en vez de fallar toda la respuesta. El track
    # semántico es aditivo/best-effort en esta fase (Fase A); una excepción
    # ahí nunca debe convertir una extracción geométrica exitosa en un 5xx.
    semantic_engine = getattr(request.app.state, "semantic_engine", None)
    if semantic_engine is not None:
        try:
            objects = semantic_engine.detect(
                image_bytes, geometry.rooms, geometry.walls
            )
            objects, dedup_count = merge_semantic(
                objects, geometry.rooms, geometry.walls, geometry.openings
            )
            geometry = geometry.model_copy(update={"objects": objects})
            needs_review_count = sum(1 for obj in objects if obj.needs_review)
            logger.info(
                "Semantic objects merged",
                extra={
                    "endpoint": endpoint,
                    "semantic_objects_count": len(objects),
                    "semantic_needs_review_count": needs_review_count,
                    "semantic_dedup_vs_openings": dedup_count,
                },
            )
        except Exception:
            logger.exception(
                "Semantic engine failed; degrading to objects=[]",
                extra={"endpoint": endpoint, "image_size": image_size},
            )
            geometry = geometry.model_copy(update={"objects": []})

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=geometry.model_dump(mode="json"),
    )
