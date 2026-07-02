"""Preflight heuristic checks — deterministic image quality gate (ADR-005).

IMPORTANT: No LLM calls, no external services — pure image heuristics only.
Thresholds are read from Settings, never hardcoded here.

Decision logic for `is_floor_plan` (ADR-005):
  A floor plan image is expected to:
    1. Have sufficient resolution for reliable line detection.
    2. Show high Michelson contrast (dark lines on light background).
    3. Contain a dense network of straight edges (Canny edge-pixel ratio).
  All three checks must pass for `is_floor_plan=True`.
  A photograph or selfie typically fails the contrast and/or line-density
  checks, producing `is_floor_plan=False`.

Orientation estimation:
  Uses probabilistic Hough lines on the Canny edge map.  Returns the dominant
  angle bucket ("horizontal", "vertical", or "diagonal (N°)") when enough
  lines are found, None otherwise.  The field is optional per the OpenAPI spec.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from vitrina_cv.models import PreflightReport

if TYPE_CHECKING:
    from vitrina_cv.config.settings import Settings

# ---------------------------------------------------------------------------
# Angle constants for orientation and rectilinear checks (degrees)
# ---------------------------------------------------------------------------

_HORIZONTAL_ANGLE_MAX: float = 20.0  # [0°, 20°] → horizontal
_VERTICAL_ANGLE_MIN: float = 70.0  # [70°, 90°] → vertical

# Minimum fraction of Hough line segments that must be near-horizontal or
# near-vertical for an image to be considered "rectilinear" (floor-plan-like).
# Real floor plans typically score > 0.70; random noise / photos typically < 0.50.
_RECTILINEAR_RATIO_MIN: float = 0.50

# Minimum number of long Hough line segments required before declaring an image
# rectilinear.  Guards against calling "rectilinear" an image with only 1-2 lines.
_RECTILINEAR_MIN_SEGMENTS: int = 10

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR numpy array.

    Args:
        image_bytes: Raw bytes of a PNG (or any OpenCV-supported format).

    Returns:
        BGR uint8 array of shape (H, W, 3).

    Raises:
        ValueError: If the bytes cannot be decoded as a valid image.
    """
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        msg = "Could not decode image bytes: unsupported format or corrupt data"
        raise ValueError(msg)
    return img


def _check_resolution(img: np.ndarray, settings: Settings) -> bool:
    """Return True if both image dimensions meet the configured minimums.

    Thresholds: settings.preflight_min_width x settings.preflight_min_height
    (derived from CV_PREFLIGHT_MIN_RESOLUTION env var, format "WxH").
    """
    h, w = img.shape[:2]
    return w >= settings.preflight_min_width and h >= settings.preflight_min_height


def _check_contrast(gray: np.ndarray, min_contrast: float) -> bool:
    """Return True if the robust Michelson contrast meets the minimum.

    Uses p2/p98 percentiles instead of absolute min/max to be robust against
    isolated bright or dark pixels (dust, scanner artefacts).

    Michelson contrast = (I_high - I_low) / (I_high + I_low + ε)
    Range: [0, 1].  Floor plans typically score > 0.5; solid-color images ≈ 0.
    """
    p_low = float(np.percentile(gray, 2))
    p_high = float(np.percentile(gray, 98))
    michelson = (p_high - p_low) / (p_high + p_low + 1e-8)
    return michelson >= min_contrast


def _check_line_density(gray: np.ndarray, min_density: float) -> bool:
    """Return True if the Canny edge-pixel ratio meets the minimum.

    Edge ratio = (number of edge pixels) / (total pixels).
    Floor plans are rich in straight lines → high ratio.
    Photographs have diffuse gradients → low ratio.
    """
    edges = cv2.Canny(gray, 50, 150)
    density = int(np.count_nonzero(edges)) / gray.size
    return density >= min_density


def _is_rectilinear(gray: np.ndarray) -> bool:
    """Return True if the image has predominantly horizontal/vertical long lines.

    Rule for `is_floor_plan`:
      A floor plan is characterised by a high proportion of long, straight lines
      at 0° (horizontal) or 90° (vertical) — walls, rooms, grids.
      A photograph or random noise has lines distributed at random angles, so the
      rectilinear ratio (H+V lines / total lines) stays below _RECTILINEAR_RATIO_MIN.

    Steps:
      1. Extract Canny edges.
      2. Run HoughLinesP with selective parameters (high threshold, long min-length).
      3. Compute each segment's angle relative to horizontal.
      4. Return True when at least _RECTILINEAR_MIN_SEGMENTS segments are found AND
         the fraction of near-H/V segments >= _RECTILINEAR_RATIO_MIN.
    """
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,  # higher = more selective, fewer false positives from noise
        minLineLength=60,
        maxLineGap=10,
    )
    if lines is None or len(lines) < _RECTILINEAR_MIN_SEGMENTS:
        return False

    segments = lines.reshape(-1, 4)
    angles = [
        float(np.degrees(np.arctan2(abs(int(y2) - int(y1)), abs(int(x2) - int(x1)))))
        for x1, y1, x2, y2 in segments
    ]
    rectilinear_count = sum(
        1 for a in angles if a <= _HORIZONTAL_ANGLE_MAX or a >= _VERTICAL_ANGLE_MIN
    )
    return (rectilinear_count / len(angles)) >= _RECTILINEAR_RATIO_MIN


def _estimate_orientation(gray: np.ndarray) -> str | None:
    """Estimate the dominant line orientation using probabilistic Hough lines.

    Steps:
      1. Compute Canny edges on the grayscale image.
      2. Run HoughLinesP to detect line segments.
      3. Compute the angle (in degrees) of each segment relative to horizontal.
      4. Return the median angle bucketed as "horizontal", "vertical", or
         "diagonal (N°)" when lines are found; None if no lines detected.

    Returns:
        Human-readable orientation string, or None if undetermined.
    """
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=50,
        maxLineGap=10,
    )
    if lines is None or len(lines) == 0:
        return None

    # Reshape to (N, 4) regardless of whether OpenCV returns (N, 1, 4) or (N, 4)
    segments = lines.reshape(-1, 4)
    lengths = np.array(
        [
            float(np.hypot(int(x2) - int(x1), int(y2) - int(y1)))
            for x1, y1, x2, y2 in segments
        ]
    )
    angles_arr = np.array(
        [
            float(
                np.degrees(np.arctan2(abs(int(y2) - int(y1)), abs(int(x2) - int(x1))))
            )
            for x1, y1, x2, y2 in segments
        ]
    )
    # Accumulate total length per orientation bin (H / V / diagonal).
    # Short corner segments (|dx|≈|dy| → ~45°) are down-weighted by their small
    # length.  We pick the dominant bin rather than averaging all angles, which
    # avoids equal H+V grids cancelling each other to ~45°.
    h_weight = float(lengths[angles_arr <= _HORIZONTAL_ANGLE_MAX].sum())
    v_weight = float(lengths[angles_arr >= _VERTICAL_ANGLE_MIN].sum())
    diag_mask = (angles_arr > _HORIZONTAL_ANGLE_MAX) & (
        angles_arr < _VERTICAL_ANGLE_MIN
    )
    d_weight = float(lengths[diag_mask].sum())

    dominant = max(h_weight, v_weight, d_weight)
    if dominant == 0.0:
        return None
    if dominant == h_weight:
        return "horizontal"
    if dominant == v_weight:
        return "vertical"
    diag_angle = float(np.average(angles_arr[diag_mask], weights=lengths[diag_mask]))
    return f"diagonal ({diag_angle:.1f}°)"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_preflight(image_bytes: bytes, settings: Settings) -> PreflightReport:
    """Run the preflight quality gate on a raw PNG image.

    Checks applied (all thresholds from `settings`, never hardcoded):
      - resolution_ok   : width x height >= CV_PREFLIGHT_MIN_RESOLUTION.
      - contrast_ok     : Michelson(p2,p98) >= CV_PREFLIGHT_MIN_CONTRAST.
      - line_density_ok : Canny edge ratio >= CV_PREFLIGHT_MIN_LINE_DENSITY.
      - is_floor_plan   : True only when all three checks pass.
      - orientation     : dominant Hough line direction; None if undetermined.
      - suggestions     : one actionable Spanish message per failed check.

    Args:
        image_bytes: Raw PNG received in the request.
        settings: Application settings carrying the threshold values.

    Returns:
        PreflightReport conforming to cv-service.openapi.yaml (ADR-005).

    Raises:
        ValueError: If `image_bytes` cannot be decoded as a valid image.
    """
    img = _decode_image(image_bytes)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    resolution_ok = _check_resolution(img, settings)
    contrast_ok = _check_contrast(gray, settings.cv_preflight_min_contrast)
    line_density_ok = _check_line_density(gray, settings.cv_preflight_min_line_density)

    # is_floor_plan: all three metric checks AND rectilinear structure present.
    # Rule: a floor plan has resolution + contrast + edge density (the three
    # configurable checks) AND predominantly horizontal/vertical long lines.
    # A selfie/photo passes contrast and edge-density but fails rectilinear,
    # producing is_floor_plan=False.  Random noise also fails rectilinear.
    rectilinear = _is_rectilinear(gray)
    is_floor_plan = resolution_ok and contrast_ok and line_density_ok and rectilinear

    # Orientation is only meaningful when there are enough lines to detect.
    orientation = _estimate_orientation(gray) if line_density_ok else None

    suggestions: list[str] = []
    if not resolution_ok:
        suggestions.append(
            f"Sube una imagen de mayor resolución "
            f"(mínimo {settings.preflight_min_width}x{settings.preflight_min_height} píxeles)."
        )
    if not contrast_ok:
        suggestions.append(
            "La imagen tiene bajo contraste. Asegúrate de que el plano tenga "
            "líneas oscuras bien definidas sobre fondo claro."
        )
    if not line_density_ok or not rectilinear:
        suggestions.append(
            "La imagen no parece un plano arquitectónico. "
            "Sube una imagen con la geometría de un plano real (paredes, habitaciones y aberturas)."
        )

    return PreflightReport(
        is_floor_plan=is_floor_plan,
        resolution_ok=resolution_ok,
        contrast_ok=contrast_ok,
        line_density_ok=line_density_ok,
        orientation=orientation,
        suggestions=suggestions,
    )
