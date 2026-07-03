"""OCR-based scale detection from dimension annotations (cotas) in floor plans.

Implements ADR-011: detects numeric values adjacent to thin dimension lines,
computes px_per_unit for each cota, validates consistency across readings, and
returns a Scale or falls back to source="none" gracefully.

Dependency contract:
  - pytesseract (Python wrapper) + system tesseract binary are OPTIONAL.
  - If either is missing, all public functions degrade to returning
    Scale(source=ScaleSource.none) and log a single WARNING.
  - No exception propagates outside detect_scale_from_ocr().

Approach implemented (documented as required by ADR-011):
  "Full-image OCR + Hough line association"

  1. Run pytesseract image_to_data (PSM 11, digits whitelist) on the full
     grayscale image upscaled to _OCR_MIN_LONG_SIDE px.
  2. Filter tokens to numeric values in the plausible dimension range.
  3. For each token, search among HoughLinesP segments (run on original gray
     image without wall-mask filtering) for the nearest horizontal or vertical
     line within _ASSOC_MAX_DIST_PX.
  4. Compute px_per_unit = line_length_px / numeric_value, normalising cm to m.
  5. Validate consistency: if candidates diverge > tolerance from the median,
     discard outliers; require >= _MIN_CONSISTENT_READINGS coherent readings.
  6. Return Scale(source=cotas, px_per_unit=median, unit="m") or
     Scale(source=none) if any step fails.

Unit heuristic:
  - float with decimal AND 0.5 <= value <= 30  -> metres
  - integer (or decimal) AND 50 <= value <= 3000 -> cm (convert: value/100)
  - otherwise -> skip
"""

from __future__ import annotations

import logging
import math
import re
import warnings
from typing import TYPE_CHECKING

import cv2
import numpy as np

from vitrina_cv.models import Scale, ScaleSource

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from vitrina_cv.config.settings import Settings

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Upscale grayscale to this long-side before running tesseract.
# Dimension-annotation text in residential plans is ~3-6 pt in original
# resolution; upscaling to 4000+ px makes those glyphs 15-25 px tall,
# enough for tesseract's LSTM engine to read reliably.
_OCR_MIN_LONG_SIDE: int = 4000

# HoughLinesP parameters used to detect thin dimension lines (different from
# the wall-detection pass which uses adaptive-thresholded binary mask).
_COTA_HOUGH_RHO: float = 1.0
_COTA_HOUGH_THETA: float = np.pi / 180
_COTA_HOUGH_THRESHOLD: int = 15  # lower than wall pass to catch short lines
_COTA_HOUGH_MIN_LEN: int = 30  # minimum cota line length (px, in original coords)
_COTA_HOUGH_MAX_GAP: int = 4

# Maximum angular deviation from horizontal/vertical for a line to be
# classified as H or V. Dimension lines are always axis-aligned.
_COTA_ANGLE_TOL_DEG: float = 3.0

# Maximum pixel distance between a token's centroid and a candidate cota
# line for the association to be accepted (in original-image coords).
# Typical annotation text sits 20-80 px away from its dimension line.
_ASSOC_MAX_DIST_PX: float = 120.0

# Minimum number of consistent cota readings before we trust the scale.
# One reading alone could be noise; two or more corroborate.
_MIN_CONSISTENT_READINGS: int = 2

# Dimension value range in metres (after unit normalisation).
# Values outside this range are noise.
_DIM_MIN_M: float = 0.3
_DIM_MAX_M: float = 50.0

# Unit heuristic thresholds
_METER_MIN: float = 0.5
_METER_MAX: float = 30.0
_CM_MIN: float = 50.0
_CM_MAX: float = 3000.0

# Tesseract config: sparse text detection, digit + period whitelist.
_TESS_CONFIG_DIGITS: str = "--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789."

# Minimum tesseract word-confidence to accept a token.
_TESS_MIN_CONF: int = 20

# Zero-length guard used when projecting a point onto a segment.
_SEG_LEN_SQ_EPSILON: float = 1e-6

# Minimum segment length (px) to use as a dimension-line candidate.
_MIN_SEG_LENGTH_PX: float = 5.0

# ---------------------------------------------------------------------------
# Lazy import of pytesseract (optional dependency)
# ---------------------------------------------------------------------------

_PYTESSERACT_IMPORT_FAILED: bool = False
_PYTESSERACT_WARNED: bool = False


def _get_pytesseract() -> object | None:
    """Return the pytesseract module, or None if unavailable.

    Logs a warning once on first failure (import error or binary missing).
    """
    global _PYTESSERACT_IMPORT_FAILED, _PYTESSERACT_WARNED  # noqa: PLW0603

    if _PYTESSERACT_IMPORT_FAILED:
        return None

    try:
        import pytesseract  # type: ignore[import-untyped]  # noqa: PLC0415

        # Probe that the binary is reachable.
        pytesseract.get_tesseract_version()
        return pytesseract
    except Exception as exc:
        _PYTESSERACT_IMPORT_FAILED = True
        if not _PYTESSERACT_WARNED:
            _PYTESSERACT_WARNED = True
            _logger.warning(
                "scale_ocr_unavailable",
                extra={
                    "reason": str(exc),
                    "effect": (
                        "scale detection falls back to source=none; "
                        "install pytesseract + tesseract-ocr to enable"
                    ),
                },
            )
        return None


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _upscale_for_ocr(
    gray: NDArray[np.uint8],
) -> tuple[NDArray[np.uint8], float]:
    """Upscale gray to at least _OCR_MIN_LONG_SIDE on the long side.

    Returns (upscaled image, upscale factor).
    If the image is already large enough, returns it unchanged with factor 1.0.
    """
    h, w = gray.shape[:2]
    long_side = max(h, w)
    if long_side >= _OCR_MIN_LONG_SIDE:
        return gray, 1.0

    factor = _OCR_MIN_LONG_SIDE / long_side
    new_w = max(1, round(w * factor))
    new_h = max(1, round(h * factor))
    upscaled: NDArray[np.uint8] = cv2.resize(
        gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
    )
    return upscaled, factor


# ---------------------------------------------------------------------------
# OCR token extraction
# ---------------------------------------------------------------------------


def _extract_numeric_tokens(
    gray_ocr: NDArray[np.uint8],
    tesseract_cmd: str,
) -> list[dict[str, int | float | str]]:
    """Run tesseract on gray_ocr and return a list of plausible dimension tokens.

    Each returned dict has:
        text   str   -- the raw token string (e.g. "8.00", "330")
        value  float -- parsed float value
        cx     float -- horizontal centroid (px) in gray_ocr coords
        cy     float -- vertical centroid (px) in gray_ocr coords
        w      int   -- token bounding-box width
        h      int   -- token bounding-box height
        conf   int   -- tesseract confidence [0, 100]

    Returns an empty list if pytesseract is unavailable or tesseract fails.
    """
    import pytesseract as pt  # type: ignore[import-untyped]  # noqa: PLC0415
    from PIL import Image  # type: ignore[import-untyped]  # noqa: PLC0415

    if tesseract_cmd:
        pt.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        pil_img = Image.fromarray(gray_ocr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = pt.image_to_data(
                pil_img,
                config=_TESS_CONFIG_DIGITS,
                output_type=pt.Output.DICT,
            )
    except Exception as exc:
        _logger.debug("scale_ocr_tesseract_error", extra={"error": str(exc)})
        return []

    tokens: list[dict[str, int | float | str]] = []
    n = len(data["text"])
    for i in range(n):
        raw: str = data["text"][i].strip()
        conf: int = int(data["conf"][i])
        if not raw or conf < _TESS_MIN_CONF:
            continue
        # Accept only strings that are valid decimal numbers.
        if not re.fullmatch(r"\d+\.?\d*|\d*\.\d+", raw):
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if value == 0.0:
            continue

        x_left: int = int(data["left"][i])
        y_top: int = int(data["top"][i])
        bw: int = int(data["width"][i])
        bh: int = int(data["height"][i])

        tokens.append(
            {
                "text": raw,
                "value": value,
                "cx": float(x_left + bw / 2),
                "cy": float(y_top + bh / 2),
                "w": bw,
                "h": bh,
                "conf": conf,
            }
        )

    return tokens


# ---------------------------------------------------------------------------
# Unit inference
# ---------------------------------------------------------------------------


def _infer_unit_and_metres(value: float) -> float | None:
    """Convert a raw OCR numeric value to metres, or return None if ambiguous.

    Heuristic (ADR-011):
      - 0.5 <= value <= 30 -> treat as metres (return as-is)
      - 50 <= value <= 3000 AND value is integral -> treat as cm (return value/100)
      - Otherwise -> not a plausible dimension (return None)
    """
    if _METER_MIN <= value <= _METER_MAX:
        return value
    if _CM_MIN <= value <= _CM_MAX and value == int(value):
        return value / 100.0
    return None


# ---------------------------------------------------------------------------
# Cota-line detection (in original-resolution gray image)
# ---------------------------------------------------------------------------


def _detect_cota_lines(
    gray: NDArray[np.uint8],
) -> NDArray[np.int32]:
    """Detect thin axis-aligned dimension lines using HoughLinesP.

    Runs on the original-resolution gray image (not the OCR-upscaled one).
    Returns an array of shape (N, 4) with columns [x1, y1, x2, y2].
    Only strictly horizontal or strictly vertical segments are returned.
    """
    blurred: NDArray[np.uint8] = cv2.GaussianBlur(gray, (3, 3), 0)
    edges: NDArray[np.uint8] = cv2.Canny(blurred, 30, 100)

    raw = cv2.HoughLinesP(
        edges,
        rho=_COTA_HOUGH_RHO,
        theta=_COTA_HOUGH_THETA,
        threshold=_COTA_HOUGH_THRESHOLD,
        minLineLength=_COTA_HOUGH_MIN_LEN,
        maxLineGap=_COTA_HOUGH_MAX_GAP,
    )
    if raw is None:
        return np.empty((0, 4), dtype=np.int32)

    segs: NDArray[np.int32] = raw.reshape(-1, 4)
    tol_rad = math.radians(_COTA_ANGLE_TOL_DEG)

    keep: list[NDArray[np.int32]] = []
    for seg in segs:
        x1, y1, x2, y2 = int(seg[0]), int(seg[1]), int(seg[2]), int(seg[3])
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1.0:
            continue
        angle = math.atan2(abs(dy), abs(dx))  # 0 = horizontal, pi/2 = vertical
        # Accept if nearly horizontal or nearly vertical.
        if angle <= tol_rad or angle >= (math.pi / 2 - tol_rad):
            keep.append(seg)

    if not keep:
        return np.empty((0, 4), dtype=np.int32)
    return np.array(keep, dtype=np.int32)


# ---------------------------------------------------------------------------
# Token <-> line association helpers
# ---------------------------------------------------------------------------


def _segment_length(seg: NDArray[np.int32]) -> float:
    """Euclidean length of a [x1, y1, x2, y2] segment."""
    x1, y1, x2, y2 = int(seg[0]), int(seg[1]), int(seg[2]), int(seg[3])
    return math.hypot(x2 - x1, y2 - y1)


def _distance_point_to_seg(cx: float, cy: float, seg: NDArray[np.int32]) -> float:
    """Minimum distance from point (cx, cy) to line segment [x1,y1,x2,y2]."""
    x1, y1, x2, y2 = float(seg[0]), float(seg[1]), float(seg[2]), float(seg[3])
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < _SEG_LEN_SQ_EPSILON:
        return math.hypot(cx - x1, cy - y1)
    t = max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(cx - proj_x, cy - proj_y)


def _find_nearest_line(
    cx: float,
    cy: float,
    segs: NDArray[np.int32],
) -> tuple[NDArray[np.int32] | None, float]:
    """Return (segment, distance) of the nearest segment to (cx, cy).

    Returns (None, inf) when segs is empty or no segment is within
    _ASSOC_MAX_DIST_PX.
    """
    best_seg: NDArray[np.int32] | None = None
    best_dist = float("inf")
    for seg in segs:
        d = _distance_point_to_seg(cx, cy, seg)
        if d < best_dist:
            best_dist = d
            best_seg = seg

    if best_dist > _ASSOC_MAX_DIST_PX:
        return None, float("inf")
    return best_seg, best_dist


# ---------------------------------------------------------------------------
# Consistency validation
# ---------------------------------------------------------------------------


def _consistent_median(candidates: list[float], tolerance: float) -> float | None:
    """Return the median of candidates within `tolerance` of the overall median.

    Algorithm:
      1. Compute full-set median.
      2. Discard values where |value - median| / median > tolerance.
      3. If >= _MIN_CONSISTENT_READINGS remain, return their median.
      4. Otherwise return None.
    """
    if not candidates:
        return None

    arr = np.array(candidates, dtype=np.float64)
    median = float(np.median(arr))
    if median <= 0.0:
        return None

    mask = np.abs(arr - median) / median <= tolerance
    consistent = arr[mask]

    if len(consistent) < _MIN_CONSISTENT_READINGS:
        return None

    return float(np.median(consistent))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_scale_from_ocr(
    gray: NDArray[np.uint8],
    settings: Settings,
) -> Scale:
    """Detect px_per_unit from OCR of dimension annotations.

    Never raises. Returns Scale(source=none) on any error or when not enough
    consistent cota readings are available.

    Args:
        gray:     Grayscale version of the upscaled floor-plan image
                  (pre-blur, as stored in OpenCVClassicEngine._gray).
        settings: Runtime settings (CV_SCALE_OCR_* fields).

    Returns:
        Scale with source="cotas" and px_per_unit/unit set if successful,
        or Scale(source="none") on failure/insufficient data.
    """
    pytesseract = _get_pytesseract()
    if pytesseract is None:
        return Scale(source=ScaleSource.none)

    try:
        return _detect_scale_inner(gray, settings)
    except Exception as exc:
        _logger.warning(
            "scale_ocr_unexpected_error",
            extra={"error": str(exc)},
        )
        return Scale(source=ScaleSource.none)


def _detect_scale_inner(
    gray: NDArray[np.uint8],
    settings: Settings,
) -> Scale:
    """Core OCR pipeline -- called by detect_scale_from_ocr under a broad try/except."""
    tesseract_cmd: str = settings.cv_scale_ocr_tesseract_cmd
    tolerance: float = settings.cv_scale_ocr_consistency_tolerance

    # Step 1: upscale for OCR.
    gray_ocr, ocr_factor = _upscale_for_ocr(gray)

    # Step 2: extract numeric tokens.
    tokens = _extract_numeric_tokens(gray_ocr, tesseract_cmd)

    _logger.debug(
        "scale_ocr_tokens_raw",
        extra={"count": len(tokens), "ocr_factor": round(ocr_factor, 2)},
    )

    if not tokens:
        _logger.info(
            "scale_ocr_no_tokens",
            extra={
                "result": "source=none",
                "reason": "tesseract found no numeric tokens",
            },
        )
        return Scale(source=ScaleSource.none)

    # Step 3: detect cota lines in the original (non-OCR-upscaled) gray image.
    cota_segs = _detect_cota_lines(gray)

    if len(cota_segs) == 0:
        _logger.info(
            "scale_ocr_no_cota_lines",
            extra={
                "result": "source=none",
                "reason": "no axis-aligned lines detected",
            },
        )
        return Scale(source=ScaleSource.none)

    _logger.debug("scale_ocr_cota_lines", extra={"count": len(cota_segs)})

    # Step 4: associate each token to the nearest cota line and compute px_per_unit.
    # Token coordinates are in OCR-upscaled space; map back to original-image space
    # by dividing by ocr_factor.
    candidates: list[float] = []
    associations: list[dict[str, object]] = []

    for tok in tokens:
        value_m = _infer_unit_and_metres(float(str(tok["value"])))
        if value_m is None:
            continue
        if not (_DIM_MIN_M <= value_m <= _DIM_MAX_M):
            continue

        tok_cx = float(str(tok["cx"])) / ocr_factor
        tok_cy = float(str(tok["cy"])) / ocr_factor

        nearest_seg, dist = _find_nearest_line(tok_cx, tok_cy, cota_segs)
        if nearest_seg is None:
            _logger.debug(
                "scale_ocr_token_no_line",
                extra={
                    "text": tok["text"],
                    "cx": round(tok_cx),
                    "cy": round(tok_cy),
                },
            )
            continue

        line_len = _segment_length(nearest_seg)
        if line_len < _MIN_SEG_LENGTH_PX:
            continue

        px_per_unit = line_len / value_m
        candidates.append(px_per_unit)
        associations.append(
            {
                "text": tok["text"],
                "value_m": round(value_m, 4),
                "line_len_px": round(line_len, 1),
                "px_per_unit": round(px_per_unit, 2),
                "dist_px": round(dist, 1),
            }
        )

    _logger.debug(
        "scale_ocr_candidates",
        extra={"count": len(candidates), "associations": associations},
    )

    if not candidates:
        _logger.info(
            "scale_ocr_no_candidates",
            extra={
                "result": "source=none",
                "reason": (
                    "no valid token-to-line associations found; "
                    f"tokens_raw={len(tokens)}, cota_lines={len(cota_segs)}"
                ),
            },
        )
        return Scale(source=ScaleSource.none)

    # Step 5: consistency check.
    consensus = _consistent_median(candidates, tolerance)

    if consensus is None:
        _logger.info(
            "scale_ocr_no_consensus",
            extra={
                "result": "source=none",
                "reason": (
                    f"candidates diverge > {tolerance * 100:.0f}% or "
                    f"< {_MIN_CONSISTENT_READINGS} consistent readings; "
                    f"candidates={[round(c, 1) for c in candidates]}"
                ),
            },
        )
        return Scale(source=ScaleSource.none)

    _logger.info(
        "scale_ocr_success",
        extra={
            "px_per_unit": round(consensus, 2),
            "unit": "m",
            "n_candidates": len(candidates),
        },
    )
    return Scale(source=ScaleSource.cotas, px_per_unit=consensus, unit="m")
