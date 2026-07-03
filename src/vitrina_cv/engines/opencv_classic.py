"""OpenCV Classic engine — Phase 1 geometry extraction (ADR-008).

Wall detection uses probabilistic Hough (HoughLinesP) because it returns
explicit segment endpoints that map directly to Wall{start, end}, avoiding
post-processing of (rho, theta) pairs.  It is also faster than standard
Hough on dense floor-plan images.

Room detection uses connected-component analysis (CCA) on the inverted wall
mask followed by approxPolyDP contour simplification.  CCA is preferred over
skeleton-based approaches because it requires only opencv-python (no contrib)
and is O(N) in image pixels.  Non-overlapping rooms are guaranteed by design:
each interior pixel belongs to exactly one connected component.

Opening detection (06-cv-04) groups HoughLinesP wall segments by orientation
and collinear position, then identifies gaps between consecutive segments on
the same line.  Each gap within [_OPENING_MIN_GAP_PX, _OPENING_MAX_GAP_PX]
becomes an Opening candidate.  The type_candidate is a conservative heuristic
based on gap width — the final classification is always the LLM's (ADR-009).

Scale detection (06-cv-04) always returns source="none" in Phase 1 because
reliable dimension-line reading requires OCR, which is not available in this
phase.  See _detect_scale() for the documented extension point.

Internal intermediates after each extract() call are stored on the instance
so that task 06-cv-04 (openings / scale) can reuse them without re-running
the expensive binarisation step.  They are NOT part of the GeometryEngine
contract.

is_ready returns True immediately because this engine carries no ML weights
(ADR-008).  Extend this property only if an initialisation step is added.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from vitrina_cv.engines.base import GeometryEngine
from vitrina_cv.mask_cleanup import clean_mask
from vitrina_cv.models import (
    Geometry,
    ImageSize,
    Opening,
    OpeningTypeCandidate,
    Room,
    Scale,
    ScaleSource,
    Wall,
)
from vitrina_cv.preprocessing import normalize_resolution

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from vitrina_cv.config.settings import Settings

_engine_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level tuning constants
# ---------------------------------------------------------------------------

# Minimum Euclidean length (px) for a Hough segment to be kept as a Wall.
_MIN_WALL_LENGTH_PX: int = 15

# Minimum connected-component area (px²) to qualify as a room.
# Tuned for typical floor-plan images (>=800 x 600 px); adjust via subclass
# or env var in a future iteration if needed.
_MIN_ROOM_AREA_PX: float = 2_000.0

# approxPolyDP epsilon as a fraction of the contour's perimeter.
# 0.02 (2 %) keeps right-angle corners while removing sub-pixel noise.
_POLY_EPSILON_RATIO: float = 0.02

# Pixels from the image border considered "exterior" (not a room).
_BORDER_MARGIN_PX: int = 2

# Minimum number of vertices for a valid room polygon (triangle).
_MIN_POLYGON_VERTICES: int = 3

# HoughLinesP parameters
_HOUGH_RHO: float = 1.0  # distance resolution (px)
_HOUGH_THETA: float = np.pi / 180  # angle resolution (rad)
_HOUGH_THRESHOLD: int = 40  # minimum vote count
_HOUGH_MAX_GAP: int = 5  # maximum gap to bridge (px)

# Adaptive threshold block size (must be odd) and constant subtracted from mean.
_ADAPTIVE_BLOCK_SIZE: int = 15
_ADAPTIVE_C: int = 4

# Morphological kernel sizes for wall mask construction and room boundary
# dilation (larger kernel = more gap-bridging; wider walls).
_MORPH_CLOSE_KERNEL: tuple[int, int] = (3, 3)
_MORPH_DILATE_KERNEL: tuple[int, int] = (5, 5)
_MORPH_CLOSE_ITER: int = 2
_MORPH_DILATE_ITER: int = 2

# ---------------------------------------------------------------------------
# Opening detection tuning constants (06-cv-04)
# ---------------------------------------------------------------------------

# Maximum angle from 0° (or 90°) for a segment to be treated as horizontal
# (or vertical).  Segments beyond this tolerance are diagonal and ignored
# because openings in diagonal walls are rare in rectilinear floor plans.
_OPENING_ANGLE_TOL_DEG: float = 10.0

# Pixel bin width used to group nearly-collinear segments.  Two segments
# whose perpendicular coordinate falls within the same bin are considered
# collinear and belong to the same virtual wall line.
_OPENING_COLLINEAR_TOL_PX: int = 8

# Minimum gap width (px) to report as a real opening.  Micro-gaps below
# this threshold are Hough fragmentation noise and are pre-merged away
# before gap detection (H1) and then filtered out (H2).  ~25 px corresponds
# to the narrowest plausible architectural feature at typical floor-plan
# resolutions (~800-1200 px wide).  Keep as a module constant — not an env
# var because it is an intrinsic property of the detection algorithm.
_OPENING_MIN_ABERTURA_PX: int = 25

# Minimum length (px) of the merged wall segment on EACH side of a gap for
# the gap to be considered a real opening.  Corner junctions and diagonal
# artefacts produce short spurious intervals; requiring both neighbours to be
# at least this long filters them without touching real openings whose
# flanking wall sections are naturally long.
#
# Set to 170 (was 80): after wall consolidation the right-perimeter artefact
# at x≈1092 exposes a noise gap whose shorter flank spans ≈160 px — raising
# the threshold to 170 suppresses it while preserving real openings (door
# flanks ≈300-355 px, window flanks ≈308-602 px).
_OPENING_MIN_WALL_SPAN_PX: int = 170

# Gap size bounds: gaps smaller than min are noise or thin frames; gaps
# larger than max are likely missing wall sections, not openings.
_OPENING_MIN_GAP_PX: int = 10
_OPENING_MAX_GAP_PX: int = 150

# Heuristic size ranges for door and window candidates (conservative).
# These are tuned for typical floor-plan images at 800-1200 px width.
# Overlap zone (30-50 px) -> unknown to avoid false classification (ADR-009).
_DOOR_GAP_MIN_PX: int = 30  # narrowest plausible door (~70 cm at ~50 px/m)
_DOOR_GAP_MAX_PX: int = 120  # widest plausible single door (~2.4 m)
_WINDOW_GAP_MIN_PX: int = 15  # narrowest plausible window (~30 cm)
_WINDOW_GAP_MAX_PX: int = 50  # upper end of window before it overlaps door range

# Confidence scores per type_candidate (conservative — LLM has final say).
_DOOR_CONFIDENCE: float = 0.5
_WINDOW_CONFIDENCE: float = 0.4
_UNKNOWN_CONFIDENCE: float = 0.3

# Estimated wall thickness (px) used to size the opening bbox perpendicular
# to the gap direction.  A constant avoids dependency on the wall-mask width.
_WALL_THICKNESS_EST_PX: int = 10

# Room detection: directional morphological closing applied to the wall mask
# before CCA so that architectural openings (doors, windows) do not connect
# interior regions to the exterior.  Must be wider than the largest expected
# opening (_OPENING_MAX_GAP_PX) to reliably bridge all gaps.  Used ONLY for
# the CCA step; wall and opening detection use the original (unclosed) mask.
_ROOM_CLOSE_GAP_PX: int = _OPENING_MAX_GAP_PX + 10  # 160 px — bridges all openings

# NMS: maximum Euclidean distance (px) between the centres of two opening
# bboxes for them to be considered duplicates of the same physical opening.
# HoughLinesP produces multiple parallel traces of thick wall strokes,
# generating near-identical openings offset by ~8-16 px in the perpendicular
# direction.  20 px suppresses these duplicates while keeping genuinely
# separate openings (typical inter-opening distance >> 20 px).
_NMS_CENTER_DIST_PX: float = 20.0


# ---------------------------------------------------------------------------
# Private pipeline helpers
# ---------------------------------------------------------------------------


def _decode_png(image_bytes: bytes) -> NDArray[np.uint8]:
    """Decode raw PNG bytes to a BGR uint8 ndarray.

    Args:
        image_bytes: Raw PNG bytes as received in the request.

    Returns:
        HxWx3 uint8 ndarray in BGR colour order.

    Raises:
        ValueError: If OpenCV cannot decode the buffer as an image.
    """
    buf: NDArray[np.uint8] = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        msg = "Could not decode image_bytes as a PNG image."
        raise ValueError(msg)
    return img  # type: ignore[return-value]


def _build_wall_mask(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Produce a binary mask where wall traces are white (255) and background black.

    Pipeline:
      1. Adaptive Gaussian threshold with THRESH_BINARY_INV — wall traces
         (typically dark) become 255; bright background becomes 0.
         Adaptive mode handles non-uniform illumination across the scan.
      2. Morphological close — bridges small gaps in wall segments that arise
         from scan artefacts or thin strokes without creating thick blobs.

    Args:
        gray: HxW uint8 grayscale image (pre-blurred recommended).

    Returns:
        HxW uint8 binary mask (values 0 or 255).
    """
    binary: NDArray[np.uint8] = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=_ADAPTIVE_BLOCK_SIZE,
        C=_ADAPTIVE_C,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, _MORPH_CLOSE_KERNEL)
    closed: NDArray[np.uint8] = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, kernel, iterations=_MORPH_CLOSE_ITER
    )
    return closed


def _detect_walls(wall_mask: NDArray[np.uint8]) -> list[Wall]:
    """Detect wall segments via probabilistic Hough transform.

    HoughLinesP is chosen over standard Hough because it returns explicit
    endpoint coordinates (x1, y1, x2, y2) that map directly to Wall{start,
    end} without extra conversion, and is faster on dense binary images.

    Args:
        wall_mask: Binary mask where walls = 255.

    Returns:
        List of Wall objects in pixel coordinates.
    """
    segments = cv2.HoughLinesP(
        wall_mask,
        rho=_HOUGH_RHO,
        theta=_HOUGH_THETA,
        threshold=_HOUGH_THRESHOLD,
        minLineLength=float(_MIN_WALL_LENGTH_PX),
        maxLineGap=float(_HOUGH_MAX_GAP),
    )
    if segments is None:
        return []

    walls: list[Wall] = []
    for seg in segments:
        # HoughLinesP returns (N, 1, 4) in older OpenCV and (N, 4) in newer
        # versions; reshape(-1) normalises both to a flat (4,) array.
        x1, y1, x2, y2 = seg.reshape(4).tolist()
        length = math.hypot(x2 - x1, y2 - y1)
        if length < _MIN_WALL_LENGTH_PX:
            continue
        walls.append(
            Wall(
                start=(float(x1), float(y1)),
                end=(float(x2), float(y2)),
            )
        )
    return walls


def _detect_rooms(
    wall_mask: NDArray[np.uint8],
    img_h: int,
    img_w: int,
) -> list[Room]:
    """Detect rooms as closed polygons from interior floor regions.

    Strategy:
      1. Dilate the wall mask to bridge small gaps and ensure closed
         boundaries around every room (prevents interior leakage into
         the exterior region).
      2. Invert: floor pixels (non-wall) become 255.
      3. Connected-component analysis (CCA, 8-connectivity) — each interior
         region is a candidate room.  Non-overlap is guaranteed by CCA: each
         pixel belongs to exactly one component.
      4. Filter: discard components touching the image border (exterior space)
         and those below the minimum area threshold (noise, stairwells, etc.).
      5. Per component: find the largest external contour, simplify with
         approxPolyDP (epsilon = 2 % of arc length) to keep axis-aligned
         corners while removing sub-pixel jitter.

    Args:
        wall_mask: Binary mask where walls = 255.
        img_h: Image height in pixels.
        img_w: Image width in pixels.

    Returns:
        List of Room objects with closed polygons and pixel area.
        Rooms are non-overlapping by construction.
    """
    # Dilate walls to close gaps — critical for flood-fill containment.
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, _MORPH_DILATE_KERNEL)
    dilated: NDArray[np.uint8] = cv2.dilate(
        wall_mask, dilate_kernel, iterations=_MORPH_DILATE_ITER
    )

    # Invert: interior regions become white.
    floor_mask: NDArray[np.uint8] = cv2.bitwise_not(dilated)

    # CCA — label every connected interior region.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        floor_mask, connectivity=8
    )

    rooms: list[Room] = []
    m = _BORDER_MARGIN_PX

    for label in range(1, num_labels):  # label 0 is the global background
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < _MIN_ROOM_AREA_PX:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        # Reject border-touching components (they are the exterior of the building).
        if x <= m or y <= m or (x + w) >= (img_w - m) or (y + h) >= (img_h - m):
            continue

        # Isolate this component to find its contour cleanly.
        component_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        component_mask[labels == label] = 255

        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        # Select the largest contour (there should be exactly one for a simple room).
        cnt = max(contours, key=cv2.contourArea)

        # approxPolyDP: epsilon proportional to perimeter — preserves right angles.
        epsilon = _POLY_EPSILON_RATIO * cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)

        if len(approx) < _MIN_POLYGON_VERTICES:  # degenerate — skip
            continue

        polygon: list[tuple[float, float]] = [
            (float(pt[0][0]), float(pt[0][1])) for pt in approx
        ]
        rooms.append(Room(polygon=polygon, area_px=area))

    return rooms


# ---------------------------------------------------------------------------
# Opening detection helpers (06-cv-04)
# ---------------------------------------------------------------------------


def _merge_intervals(
    intervals: list[tuple[float, float]],
    tol: float = 0.0,
) -> list[tuple[float, float]]:
    """Merge overlapping or nearly-adjacent intervals.

    When *tol* > 0, two intervals separated by a gap of at most *tol* pixels
    are merged as if they were continuous.  Pass tol = _OPENING_MIN_ABERTURA_PX - 1
    to collapse Hough fragmentation micro-gaps before gap detection (H1 fix).

    Args:
        intervals: Sorted list of (start, end) pairs.
        tol: Maximum gap between consecutive intervals to still merge them.

    Returns:
        Merged list of non-overlapping (start, end) pairs.
    """
    if not intervals:
        return []
    merged: list[tuple[float, float]] = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + tol:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _classify_gap(
    gap_size: float,
) -> tuple[OpeningTypeCandidate, float]:
    """Heuristically classify an opening gap by its pixel width (ADR-009).

    Conservative strategy: when gap_size falls in both the door and window
    range (the overlap zone), return unknown rather than guess.

    Ranges (tuned for 800-1200 px wide images):
      - window-only : _WINDOW_GAP_MIN_PX ... _DOOR_GAP_MIN_PX - 1
      - overlap zone: _DOOR_GAP_MIN_PX ... _WINDOW_GAP_MAX_PX  -> unknown
      - door-only   : _WINDOW_GAP_MAX_PX + 1 … _DOOR_GAP_MAX_PX

    Args:
        gap_size: Gap extent in pixels along the wall direction.

    Returns:
        (type_candidate, confidence) tuple.
    """
    in_door_range = _DOOR_GAP_MIN_PX <= gap_size <= _DOOR_GAP_MAX_PX
    in_window_range = _WINDOW_GAP_MIN_PX <= gap_size <= _WINDOW_GAP_MAX_PX

    if in_door_range and not in_window_range:
        return OpeningTypeCandidate.door, _DOOR_CONFIDENCE
    if in_window_range and not in_door_range:
        return OpeningTypeCandidate.window, _WINDOW_CONFIDENCE
    # Overlap or neither — be conservative (ADR-009: CV never decides type).
    return OpeningTypeCandidate.unknown, _UNKNOWN_CONFIDENCE


def _detect_openings(walls: list[Wall]) -> list[Opening]:
    """Detect opening candidates as gaps between collinear wall segments.

    Algorithm:
      1. Split wall segments into horizontal buckets (keyed by y) and
         vertical buckets (keyed by x) using a bin of width
         _OPENING_COLLINEAR_TOL_PX.  Diagonal segments are ignored —
         openings in diagonal walls are uncommon in rectilinear floor plans.
      2. Within each bucket, project segments onto the primary axis,
         merge overlapping projections, then find inter-segment gaps.
      3. Gaps within [_OPENING_MIN_GAP_PX, _OPENING_MAX_GAP_PX] become
         Opening candidates.  Type and confidence are heuristic.

    The engine never decides the final type — the LLM in vitrina does (ADR-009).
    If no openings are detected, returns an empty list (never an error).

    Args:
        walls: Wall segments detected by _detect_walls.

    Returns:
        List of Opening candidates (possibly empty).
    """
    if not walls:
        return []

    # Bucket horizontal segments by y-midpoint bin, vertical by x-midpoint bin.
    h_buckets: dict[int, list[tuple[float, float, float, float]]] = {}
    v_buckets: dict[int, list[tuple[float, float, float, float]]] = {}

    for wall in walls:
        x1, y1 = wall.start
        x2, y2 = wall.end
        dx = x2 - x1
        dy = y2 - y1
        # Angle from horizontal: 0° = horizontal, 90° = vertical.
        angle_deg = math.degrees(math.atan2(abs(dy), abs(dx)))

        if angle_deg < _OPENING_ANGLE_TOL_DEG:
            y_mid = (y1 + y2) / 2
            bin_key = int(y_mid) // _OPENING_COLLINEAR_TOL_PX
            h_buckets.setdefault(bin_key, []).append((x1, y1, x2, y2))
        elif angle_deg > (90.0 - _OPENING_ANGLE_TOL_DEG):
            x_mid = (x1 + x2) / 2
            bin_key = int(x_mid) // _OPENING_COLLINEAR_TOL_PX
            v_buckets.setdefault(bin_key, []).append((x1, y1, x2, y2))
        # else: diagonal — skip

    openings: list[Opening] = []
    half_thick = _WALL_THICKNESS_EST_PX / 2.0

    # --- Horizontal groups: gaps along x axis --------------------------------
    for bin_key, segs in h_buckets.items():
        y_pos = bin_key * _OPENING_COLLINEAR_TOL_PX + _OPENING_COLLINEAR_TOL_PX / 2.0
        intervals: list[tuple[float, float]] = sorted(
            (min(x1, x2), max(x1, x2)) for x1, _y1, x2, _y2 in segs
        )
        # H1: merge micro-gaps (Hough fragmentation) before gap detection.
        merged = _merge_intervals(intervals, tol=float(_OPENING_MIN_ABERTURA_PX - 1))
        for idx in range(len(merged) - 1):
            gap_start = merged[idx][1]
            gap_end = merged[idx + 1][0]
            gap_w = gap_end - gap_start
            # H2: discard gaps narrower than the minimum plausible opening.
            if not (_OPENING_MIN_ABERTURA_PX <= gap_w <= _OPENING_MAX_GAP_PX):
                continue
            # H1: discard corner/artefact gaps where either flanking wall
            # segment is too short to be a real wall section.
            left_span = merged[idx][1] - merged[idx][0]
            right_span = merged[idx + 1][1] - merged[idx + 1][0]
            if (
                left_span < _OPENING_MIN_WALL_SPAN_PX
                or right_span < _OPENING_MIN_WALL_SPAN_PX
            ):
                continue
            type_candidate, confidence = _classify_gap(gap_w)
            openings.append(
                Opening(
                    type_candidate=type_candidate,
                    bbox=(
                        gap_start,
                        y_pos - half_thick,
                        gap_w,
                        float(_WALL_THICKNESS_EST_PX),
                    ),
                    confidence=confidence,
                )
            )

    # --- Vertical groups: gaps along y axis ----------------------------------
    for bin_key, segs in v_buckets.items():
        x_pos = bin_key * _OPENING_COLLINEAR_TOL_PX + _OPENING_COLLINEAR_TOL_PX / 2.0
        intervals_v: list[tuple[float, float]] = sorted(
            (min(y1, y2), max(y1, y2)) for _x1, y1, _x2, y2 in segs
        )
        # H1: merge micro-gaps before gap detection.
        merged_v = _merge_intervals(
            intervals_v, tol=float(_OPENING_MIN_ABERTURA_PX - 1)
        )
        for idx in range(len(merged_v) - 1):
            gap_start = merged_v[idx][1]
            gap_end = merged_v[idx + 1][0]
            gap_h = gap_end - gap_start
            # H2: discard gaps narrower than the minimum plausible opening.
            if not (_OPENING_MIN_ABERTURA_PX <= gap_h <= _OPENING_MAX_GAP_PX):
                continue
            # H1: discard corner/artefact gaps where either flanking wall
            # segment is too short to be a real wall section.
            left_span_v = merged_v[idx][1] - merged_v[idx][0]
            right_span_v = merged_v[idx + 1][1] - merged_v[idx + 1][0]
            if (
                left_span_v < _OPENING_MIN_WALL_SPAN_PX
                or right_span_v < _OPENING_MIN_WALL_SPAN_PX
            ):
                continue
            type_candidate, confidence = _classify_gap(gap_h)
            openings.append(
                Opening(
                    type_candidate=type_candidate,
                    bbox=(
                        x_pos - half_thick,
                        gap_start,
                        float(_WALL_THICKNESS_EST_PX),
                        gap_h,
                    ),
                    confidence=confidence,
                )
            )

    return openings


def _detect_scale() -> Scale:
    """Attempt to detect scale from dimension annotations in the floor plan.

    Phase 1 — always returns source="none".

    Reliable dimension-line reading requires OCR to extract the numeric value
    adjacent to each cota line.  In Phase 1, no OCR engine is available in
    this service, so the function returns a "none" scale unconditionally.
    This never causes an error response (ADR-003: scale is optional).

    Extension point for a future phase with OCR:
      1. Detect thin horizontal/vertical lines with arrowhead or tick
         terminations (distinguishes dimension lines from wall segments).
      2. Run an OCR engine (e.g. tesseract via pytesseract, or an LLM vision
         call) on the region adjacent to each dimension line to extract the
         numeric value and unit label.
      3. Compute px_per_unit = pixel_length_of_line / extracted_numeric_value.
      4. Return Scale(source=ScaleSource.cotas, px_per_unit=..., unit=...).

    Decision gate: adding any OCR dependency (pytesseract, easyocr, LLM call)
    MUST be documented as an ADR before merging, because it changes the
    container image size and latency profile significantly.

    Returns:
        Scale(source="none", px_per_unit=None, unit=None).  Never raises.
    """
    return Scale(source=ScaleSource.none)


# ---------------------------------------------------------------------------
# F1 — Closed wall mask for room detection
# ---------------------------------------------------------------------------


def _build_closed_wall_mask_for_rooms(
    wall_mask: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Build a version of the wall mask with architectural openings bridged.

    Applies two directional morphological closes — one horizontal (bridges
    gaps in the x direction, i.e., in horizontal walls such as the top/bottom
    perimeter) and one vertical (bridges gaps in the y direction, i.e., in
    vertical walls such as the left/right perimeter and interior dividers).
    The kernel size is _ROOM_CLOSE_GAP_PX, large enough to bridge any
    opening up to _OPENING_MAX_GAP_PX wide.

    This mask is used EXCLUSIVELY for the CCA room-detection step.  All other
    pipeline steps (wall detection, opening detection) use the original mask.

    Args:
        wall_mask: Binary mask where walls = 255.

    Returns:
        Binary mask with openings filled in, suitable for CCA.
    """
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_ROOM_CLOSE_GAP_PX, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, _ROOM_CLOSE_GAP_PX))
    closed: NDArray[np.uint8] = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, h_kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, v_kernel)
    return closed


# ---------------------------------------------------------------------------
# F3 — Consolidated walls
# ---------------------------------------------------------------------------


def _consolidate_walls(walls: list[Wall]) -> list[Wall]:
    """Merge collinear wall segments produced by HoughLinesP into longer walls.

    HoughLinesP fragments each physical wall into many short overlapping
    segments (e.g., 131 for ~5 walls in a simple floor plan).  This function
    applies the same binning and interval-merge logic used by _detect_openings
    to produce one Wall object per continuous run of collinear segments,
    drastically reducing the wall count and stabilising opening detection.

    Args:
        walls: Raw wall segments from _detect_walls.

    Returns:
        Consolidated list of Wall objects (one per merged run per bin).
    """
    if not walls:
        return []

    h_buckets: dict[int, list[tuple[float, float, float, float]]] = {}
    v_buckets: dict[int, list[tuple[float, float, float, float]]] = {}
    diagonal: list[Wall] = []

    for wall in walls:
        x1, y1 = wall.start
        x2, y2 = wall.end
        angle_deg = math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1)))

        if angle_deg < _OPENING_ANGLE_TOL_DEG:
            y_mid = (y1 + y2) / 2
            bin_key = int(y_mid) // _OPENING_COLLINEAR_TOL_PX
            h_buckets.setdefault(bin_key, []).append((x1, y1, x2, y2))
        elif angle_deg > (90.0 - _OPENING_ANGLE_TOL_DEG):
            x_mid = (x1 + x2) / 2
            bin_key = int(x_mid) // _OPENING_COLLINEAR_TOL_PX
            v_buckets.setdefault(bin_key, []).append((x1, y1, x2, y2))
        else:
            diagonal.append(wall)

    consolidated: list[Wall] = []

    for bin_key, segs in h_buckets.items():
        y_pos = bin_key * _OPENING_COLLINEAR_TOL_PX + _OPENING_COLLINEAR_TOL_PX / 2.0
        intervals: list[tuple[float, float]] = sorted(
            (min(x1, x2), max(x1, x2)) for x1, _y1, x2, _y2 in segs
        )
        # Merge with the same tolerance used in gap detection to collapse
        # Hough micro-gaps; real openings survive as actual gaps.
        merged = _merge_intervals(intervals, tol=float(_OPENING_MIN_ABERTURA_PX - 1))
        for start, end in merged:
            if end - start >= _MIN_WALL_LENGTH_PX:
                consolidated.append(Wall(start=(start, y_pos), end=(end, y_pos)))

    for bin_key, segs in v_buckets.items():
        x_pos = bin_key * _OPENING_COLLINEAR_TOL_PX + _OPENING_COLLINEAR_TOL_PX / 2.0
        intervals_v: list[tuple[float, float]] = sorted(
            (min(y1, y2), max(y1, y2)) for _x1, y1, _x2, y2 in segs
        )
        merged_v = _merge_intervals(
            intervals_v, tol=float(_OPENING_MIN_ABERTURA_PX - 1)
        )
        for start, end in merged_v:
            if end - start >= _MIN_WALL_LENGTH_PX:
                consolidated.append(Wall(start=(x_pos, start), end=(x_pos, end)))

    consolidated.extend(diagonal)
    return consolidated


# ---------------------------------------------------------------------------
# F2 — NMS post-processing for opening candidates
# ---------------------------------------------------------------------------


def _nms_openings(openings: list[Opening]) -> list[Opening]:
    """Remove duplicate opening candidates by centre-distance suppression.

    HoughLinesP produces multiple parallel traces of thick wall strokes,
    which causes the same physical opening to appear as 2-3 candidates offset
    by ~8-16 px in the perpendicular direction.  This function keeps the
    candidate with the highest confidence among any cluster of candidates
    whose centres are within _NMS_CENTER_DIST_PX pixels of each other.

    Args:
        openings: Raw opening candidates from _detect_openings.

    Returns:
        Deduplicated list of opening candidates.
    """
    if len(openings) <= 1:
        return list(openings)

    # Sort by confidence descending so that when two candidates overlap the
    # higher-confidence one is kept (it gets added first, suppresses later).
    sorted_openings = sorted(openings, key=lambda o: o.confidence, reverse=True)

    kept: list[Opening] = []
    suppressed: set[int] = set()

    for i, candidate in enumerate(sorted_openings):
        if i in suppressed:
            continue
        bx, by, bw, bh = candidate.bbox
        cx = bx + bw / 2
        cy = by + bh / 2
        kept.append(candidate)
        for j in range(i + 1, len(sorted_openings)):
            if j in suppressed:
                continue
            bx2, by2, bw2, bh2 = sorted_openings[j].bbox
            cx2 = bx2 + bw2 / 2
            cy2 = by2 + bh2 / 2
            if math.hypot(cx2 - cx, cy2 - cy) < _NMS_CENTER_DIST_PX:
                suppressed.add(j)

    return kept


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class OpenCVClassicEngine(GeometryEngine):
    """Phase 1 engine: classical OpenCV — no ML, CPU-only (ADR-008).

    Latency target: p95 < 20 s per image.

    After each extract() call the following intermediates are available on
    the instance for reuse by task 06-cv-04 (openings / scale detection):

      _wall_mask : NDArray[np.uint8] | None
          Binary mask (walls = 255) produced by the binarisation pipeline.

      _gray : NDArray[np.uint8] | None
          Grayscale version of the last processed image (pre-blur, original).

    These attributes are internal implementation details; they are NOT part
    of the GeometryEngine contract and MUST NOT be accessed by routers.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        # Settings forwarded from the factory — used for runtime thresholds
        # (e.g. upscale target / factor).  May be None in unit tests that
        # bypass get_engine(); engine falls back to module-level constants in
        # that case (preprocessing.normalize_resolution handles its defaults
        # independently via the Settings instance passed to it).
        self._settings = settings
        # Intermediates for 06-cv-04 reuse — None until first extract() call.
        self._wall_mask: NDArray[np.uint8] | None = None
        self._gray: NDArray[np.uint8] | None = None

    @property
    def is_ready(self) -> bool:
        """Always True — no weights to load for a classical OpenCV engine."""
        return True

    def extract(self, image_bytes: bytes) -> Geometry:
        """Extract walls and rooms from a PNG floor plan.

        Args:
            image_bytes: Raw PNG bytes received in the request.

        Returns:
            Geometry with walls (HoughLinesP segments), rooms (approxPolyDP
            polygons), openings (gap-based candidates, ADR-009),
            scale (source="none" in Phase 1, ADR-003), and image_size.
            All coordinates are in pixels of the received image (ADR-003).

        Raises:
            ValueError: If image_bytes cannot be decoded as a PNG image.
        """
        t0 = time.monotonic()

        # ---- 1. Decode -------------------------------------------------
        bgr = _decode_png(image_bytes)
        orig_h, orig_w = bgr.shape[:2]
        t_decode = time.monotonic()

        # ---- 1b. Normalise resolution (upscale small images) -----------
        # The engine's pixel constants are calibrated at ~2 000 px long side.
        # Images arriving at 612x612 or 470x896 produce sub-threshold gaps,
        # tiny room areas and near-zero line density before any detection runs.
        # We upscale here so the entire pipeline — walls, rooms, openings,
        # scale — operates in a single coherent pixel space.
        # image_size in the response reflects the NORMALISED dimensions; no
        # coordinate re-projection is needed.
        if self._settings is not None:
            bgr, upscale_factor = normalize_resolution(bgr, self._settings)
        else:
            upscale_factor = 1.0
        img_h, img_w = bgr.shape[:2]

        if upscale_factor > 1.0:
            _engine_logger.info(
                "cv_engine_upscale",
                extra={
                    "original_width": orig_w,
                    "original_height": orig_h,
                    "normalised_width": img_w,
                    "normalised_height": img_h,
                    "upscale_factor": round(upscale_factor, 3),
                },
            )

        # ---- 2. Grayscale ----------------------------------------------
        gray: NDArray[np.uint8] = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # ---- 3. Denoise (light blur before thresholding) ---------------
        gray_blur: NDArray[np.uint8] = cv2.GaussianBlur(gray, (3, 3), 0)

        # ---- 4. Wall mask ----------------------------------------------
        wall_mask = _build_wall_mask(gray_blur)

        # ---- 4b. Mask cleanup (noise: text, hatching, margin cotas) ----
        # Applied AFTER binarisation and BEFORE Hough/CCA so that spurious
        # components (achurado diagonals, cota lines, text labels) do not
        # fragment walls or block CCA room enclosure.
        # The preflight gate evaluates the image BEFORE this step (ADR-005).
        if self._settings is not None:
            wall_mask = clean_mask(wall_mask, self._settings)

        # ---- 5. Store intermediates (for 06-cv-04) ---------------------
        self._wall_mask = wall_mask
        self._gray = gray  # pre-blur version for feature detection

        # ---- 6. Detect walls and consolidate (F3) ----------------------
        raw_walls = _detect_walls(wall_mask)
        walls = _consolidate_walls(raw_walls)
        t_walls = time.monotonic()

        # ---- 7. Detect rooms with gap-closed mask (F1) -----------------
        # A separate closed mask bridges architectural openings so that CCA
        # sees fully enclosed regions.  Walls and openings detection still
        # use the original (unclosed) wall_mask.
        closed_wall_mask = _build_closed_wall_mask_for_rooms(wall_mask)
        rooms = _detect_rooms(closed_wall_mask, img_h, img_w)
        t_rooms = time.monotonic()

        # ---- 8. Detect opening candidates, then NMS dedup (F2) ---------
        openings = _nms_openings(_detect_openings(walls))
        t_openings = time.monotonic()

        # ---- 9. Derive scale (06-cv-04) --------------------------------
        scale = _detect_scale()
        t_done = time.monotonic()

        _engine_logger.info(
            "cv_engine_extract_stages",
            extra={
                "image_width": img_w,
                "image_height": img_h,
                "duration_decode_ms": round((t_decode - t0) * 1000, 1),
                "duration_segmentation_ms": round((t_walls - t_decode) * 1000, 1),
                "duration_rooms_ms": round((t_rooms - t_walls) * 1000, 1),
                "duration_openings_ms": round((t_openings - t_rooms) * 1000, 1),
                "duration_total_ms": round((t_done - t0) * 1000, 1),
                "walls_count": len(walls),
                "rooms_count": len(rooms),
                "openings_count": len(openings),
            },
        )

        # ---- 10. Assemble response -------------------------------------
        return Geometry(
            walls=walls,
            rooms=rooms,
            openings=openings,
            scale=scale,
            image_size=ImageSize(width=img_w, height=img_h),
        )
