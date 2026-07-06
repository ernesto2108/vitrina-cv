"""Pydantic DTOs mirroring cv-service.openapi.yaml schemas (ADR-003).

All field names, optionality and constraints match the OpenAPI spec exactly.
Coordinates are always in pixels of the received image (never transformed).

Do NOT add fields not present in the OpenAPI schema — divergence breaks the
Go client (vitrina) without a version bump.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Primitive type alias
# ---------------------------------------------------------------------------

# [x, y] in pixels of the received image.  Sub-pixel values are valid after
# vectorisation (hence float, not int).  OpenAPI: array[number/float], minItems=2, maxItems=2.
Point = tuple[float, float]


# ---------------------------------------------------------------------------
# Enums (strict — any unknown value is a contract violation)
# ---------------------------------------------------------------------------


class ScaleSource(StrEnum):
    """Origin of the scale information extracted from the floor plan."""

    cotas = "cotas"
    none = "none"


class StairsDirection(StrEnum):
    """Cardinal direction of the staircase ascent (ADR-003 extension, OpenAPI v0.2.0)."""

    up_n = "up-N"
    up_s = "up-S"
    up_e = "up-E"
    up_w = "up-W"
    unknown = "unknown"


class OpeningTypeCandidate(StrEnum):
    """Tentative opening type emitted by the CV engine (ADR-009).

    The final classification is decided by the LLM in vitrina, not here.
    """

    door = "door"
    window = "window"
    unknown = "unknown"


class ErrorCode(StrEnum):
    """Strict enum of error codes (ADR-003).  Never use a free string."""

    invalid_request = "invalid_request"
    unprocessable_image = "unprocessable_image"
    model_not_loaded = "model_not_loaded"


# ---------------------------------------------------------------------------
# Component schemas — match OpenAPI names and required/optional exactly
# ---------------------------------------------------------------------------


class Wall(BaseModel):
    """A wall segment detected in the floor plan.

    OpenAPI required: [start, end].  thickness is optional (nullable).
    """

    start: Point
    end: Point
    thickness: float | None = None


class Room(BaseModel):
    """A room polygon detected in the floor plan.

    OpenAPI required: [polygon, area_px].
    """

    polygon: list[Point]
    area_px: float


class Opening(BaseModel):
    """An opening candidate (door/window) detected in the floor plan (ADR-009).

    The engine never decides the final type — it emits a candidate with
    confidence.  The LLM in vitrina performs the final classification.

    OpenAPI required: [type_candidate, bbox, confidence].
    bbox: [x, y, w, h] in pixels.
    confidence: [0, 1].
    """

    type_candidate: OpeningTypeCandidate
    # [x, y, w, h] in pixels.  OpenAPI: array[number], minItems=4, maxItems=4.
    bbox: tuple[float, float, float, float]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


class StairsCandidate(BaseModel):
    """A staircase candidate detected in the floor plan (OpenAPI v0.2.0).

    The engine emits a candidate with a directional hint and confidence.
    Detection logic is deferred to cv-10; this schema unblocks back-06 in
    parallel.

    OpenAPI required: [bbox, direction, confidence].
    bbox: [x, y, w, h] in pixels.
    confidence: [0, 1].
    """

    # [x, y, w, h] in pixels.  OpenAPI: array[number], minItems=4, maxItems=4.
    bbox: Annotated[list[float], Field(min_length=4, max_length=4)]
    direction: StairsDirection
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


class Scale(BaseModel):
    """Scale derived from dimension annotations in the floor plan.

    When source="none" no dimension references were found; px_per_unit and
    unit are null.  A "none" scale never makes the response an error (ADR-003).

    OpenAPI required: [source].  px_per_unit and unit are optional/nullable.
    """

    source: ScaleSource
    px_per_unit: float | None = None
    unit: str | None = None


class ImageSize(BaseModel):
    """Pixel dimensions of the image received in the request.

    OpenAPI required: [width, height].
    """

    width: int
    height: int


class Geometry(BaseModel):
    """Full geometry payload returned by POST /extract-geometry (ADR-003).

    OpenAPI required: [walls, rooms, openings, scale, image_size].
    All coordinates are in pixels of the received image — never transformed.
    """

    walls: list[Wall]
    rooms: list[Room]
    openings: list[Opening]
    stairs_candidates: list[StairsCandidate] = []
    scale: Scale
    image_size: ImageSize


class PreflightReport(BaseModel):
    """Quality-gate report returned by POST /preflight (ADR-005).

    OpenAPI required: [is_floor_plan, resolution_ok, contrast_ok,
                       line_density_ok, suggestions].
    orientation is optional (nullable).
    """

    is_floor_plan: bool
    resolution_ok: bool
    contrast_ok: bool
    line_density_ok: bool
    orientation: str | None = None
    suggestions: list[str]


class Error(BaseModel):
    """Error payload for 4xx/5xx responses.

    error_code must be one of the enum values — never a free string (ADR-003).

    OpenAPI required: [error_code, message].
    """

    error_code: ErrorCode
    message: str
