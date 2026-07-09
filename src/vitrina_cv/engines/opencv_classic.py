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
from collections import defaultdict
from typing import TYPE_CHECKING

import cv2
import numpy as np

from vitrina_cv.engines.base import GeometryEngine
from vitrina_cv.mask_cleanup import (
    clean_mask,
    clean_mask_steps_1_to_3,
    filter_interior_components,
)
from vitrina_cv.models import (
    Geometry,
    ImageSize,
    Opening,
    OpeningTypeCandidate,
    Point,
    Room,
    Scale,
    ScaleSource,
    StairsCandidate,
    StairsDirection,
    Wall,
)
from vitrina_cv.preprocessing import normalize_resolution
from vitrina_cv.scale_ocr import detect_scale_from_ocr

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

# ---------------------------------------------------------------------------
# Opening detection tuning constants (07-cv-07) — generous emission
# ---------------------------------------------------------------------------

# Relaxed wall-span threshold (px) used when a gap endpoint is adjacent to a
# wall junction.  Corner-adjacent doors have a short flanking segment on one
# side; the wide 170 px span would silently discard them.  Overridable via
# Settings.cv_opening_min_wall_span_px (default 60).
_OPENING_MIN_WALL_SPAN_JUNCTION_PX: int = 60

# Perpendicular tolerance (px) when checking if a junction is on the same
# wall line as a gap endpoint (~3x _OPENING_COLLINEAR_TOL_PX bin width).
_JUNCTION_GAP_PERP_TOL_PX: float = 24.0

# Confidence cap applied when a gap passes ONLY via the relaxed junction span
# (i.e., one flanking segment is shorter than the wide 170 px threshold).
# Signals weaker geometric evidence; backend (F4) applies stricter filtering.
_OPENING_RELAXED_SPAN_CONFIDENCE: float = 0.35

# Confidence override when a door-swing arc (semicircle) is detected adjacent
# to a gap.  Arc presence is strong positive evidence of a door opening.
_DOOR_ARC_CONFIDENCE: float = 0.7

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
# Window pattern detection constants (07-cv-06)
# ---------------------------------------------------------------------------

# Sampling pitch along the wall axis (px) when profiling perpendicular sections.
_WIN_PROFILE_STEP_PX: int = 8

# Extra pixels scanned beyond the wall half-thickness on each side of the
# centerline.  Provides tolerance for small misalignments in the centerline
# estimate without extending too far into adjacent walls.
_WIN_PROFILE_HALF_EXTRA_PX: int = 3

# Minimum number of consecutive sampling positions that must all show the
# double-line pattern (2 foreground runs) to be considered a real window span.
# 3 positions x 8 px/step = 24 px minimum span at the sampling resolution.
_WIN_MIN_CONSECUTIVE_PROFILES: int = 3

# Maximum window span (px) along the wall axis.  Candidates longer than this
# are likely full wall sections whose double notation was mis-classified.
_WIN_MAX_SPAN_PX: int = 300

# Expected number of foreground runs in a window profile (two parallel frame
# lines).  Named constant to satisfy PLR2004.
_WIN_EXPECTED_RUNS: int = 2

# Confidence assigned to window candidates detected by the pattern algorithm.
# Lower than gap-based window (0.4) because this method is more indirect
# and depends on the pre-filter mask quality (ADR-009: CV never decides type).
_WIN_CONFIDENCE: float = 0.35


# ---------------------------------------------------------------------------
# Staircase detection constants (07-cv-10)
# ---------------------------------------------------------------------------

# HoughLinesP parameters tuned for thin stair-tread lines.
# Lower threshold and shorter minLineLength than wall detection to pick up
# the narrow tread strokes that filter_thin_strokes removes from the main mask.
_STAIRS_HOUGH_THRESHOLD: int = 20
_STAIRS_HOUGH_MIN_LINE_PX: int = 20
_STAIRS_HOUGH_MAX_GAP_PX: int = 5

# Bin width (px) used to group nearly-collinear tread segments into a single
# tread line.  Two segments whose perpendicular coordinate falls within the
# same bin are considered part of the same tread.
_STAIRS_COLLINEAR_BIN_PX: int = 5

# Minimum number of parallel equi-spaced tread lines required to emit a
# staircase candidate.  Fewer than 4 treads are too ambiguous (grid hatching,
# ventilation grills, etc.).
_STAIRS_MIN_LINES: int = 4

# Allowed inter-tread spacing range (px) at ~2 000 px normalised resolution.
# 20 px ≈ 15 cm tread at ~130 px/m; 40 px ≈ 30 cm tread.
_STAIRS_MIN_SPACING_PX: float = 20.0
_STAIRS_MAX_SPACING_PX: float = 40.0

# Maximum relative standard deviation of spacings to consider them regular.
# 0.20 = 20 % tolerance.
_STAIRS_SPACING_MAX_REL_STD: float = 0.20

# Confidence assigned to staircase candidates detected by this algorithm.
_STAIRS_CONFIDENCE: float = 0.6

# ---------------------------------------------------------------------------
# Wall centerline tuning constants (07-cv-03)
# ---------------------------------------------------------------------------

# Sampling step (px) used when reading distanceTransform values along a Hough
# segment.  5 px gives ~200 samples on a 1000 px wall — sufficient for a
# stable median while avoiding per-pixel overhead.
_DT_SAMPLE_STEP_PX: int = 5

# Minimum perpendicular grouping tolerance (px) when the estimated wall
# thickness is unusually small (e.g. very thin plans or short noise segments).
# Prevents degenerate grouping where no two segments ever merge.
_CENTERLINE_MIN_TOL_PX: float = float(_OPENING_COLLINEAR_TOL_PX)


# ---------------------------------------------------------------------------
# Orthogonal snapping and junction fusion tuning constants (07-cv-04)
# ---------------------------------------------------------------------------

# Maximum deviation from horizontal (0°) or vertical (90°) for a segment to
# be snapped to the exact axis.  Segments more than this many degrees from
# either axis are left untouched (legitimate diagonals).
_SNAP_ANGLE_TOL_DEG: float = 5.0

# Minimum number of endpoint indices in a Union-Find cluster to constitute a
# real junction (a singleton endpoint needs no fusion).
_JUNCTION_MIN_CLUSTER_SIZE: int = 2


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


def _edge_angle_deg(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Angle (degrees, 0-90) of the segment ``p1 -> p2`` from the horizontal axis.

    Uses ``atan2(|dy|, |dx|)`` so the result is orientation-agnostic: 0deg is
    exactly horizontal, 90deg is exactly vertical, and values in between are
    diagonal. Mirrors the convention used by ``_filter_diagonal_residual_pass2``
    for walls so the same angle band applies to room-polygon edges (ADR-001).
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(abs(dy), abs(dx)))


def _sanitize_room_polygon(
    polygon: list[tuple[float, float]],
    low_deg: float,
    high_deg: float,
    min_diagonal_len_px: float,
) -> list[tuple[float, float]] | None:
    """Remove spurious diagonal vertices from a closed room polygon (ADR-001).

    A vertex is spurious when *both* of its adjacent edges fall inside the
    diagonal angle band ``[low_deg, high_deg]`` (the same band used by the
    wall diagonal filter) and at least one of those edges is longer than
    ``min_diagonal_len_px``. Removing the vertex directly connects its two
    neighbours, collapsing the diagonal "notch" while leaving legitimate
    axis-aligned corners (angle 0deg or 90deg, always outside the band)
    untouched.

    The check runs iteratively because collapsing one vertex can expose a
    new spurious vertex at the position that used to be its neighbour.

    Args:
        polygon: Closed polygon vertices in order, no repeated first/last point.
        low_deg: Lower bound (inclusive) of the diagonal band, in degrees.
        high_deg: Upper bound (inclusive) of the diagonal band, in degrees.
        min_diagonal_len_px: Minimum edge length to treat an in-band edge as
            a spurious diagonal rather than rectangular-corner jitter.

    Returns:
        The sanitized polygon, or ``None`` if after sanitizing there is still
        a diagonal edge above the threshold that cannot be resolved by vertex
        removal (i.e. no ortho-recoverable polygon exists) — the caller must
        discard the room per AC-2.
    """
    points = list(polygon)

    changed = True
    while changed and len(points) > _MIN_POLYGON_VERTICES:
        changed = False
        n = len(points)
        for i in range(n):
            prev_pt = points[(i - 1) % n]
            cur_pt = points[i]
            next_pt = points[(i + 1) % n]

            angle_prev = _edge_angle_deg(prev_pt, cur_pt)
            angle_next = _edge_angle_deg(cur_pt, next_pt)
            len_prev = math.hypot(cur_pt[0] - prev_pt[0], cur_pt[1] - prev_pt[1])
            len_next = math.hypot(next_pt[0] - cur_pt[0], next_pt[1] - cur_pt[1])

            prev_in_band = low_deg <= angle_prev <= high_deg
            next_in_band = low_deg <= angle_next <= high_deg
            long_enough = (
                len_prev >= min_diagonal_len_px or len_next >= min_diagonal_len_px
            )

            if prev_in_band and next_in_band and long_enough:
                points.pop(i)
                changed = True
                break

    # Post-check: any remaining edge still inside the band above the
    # threshold means no ortho-recoverable polygon exists (AC-2).
    n = len(points)
    if n < _MIN_POLYGON_VERTICES:
        return None
    for i in range(n):
        cur_pt = points[i]
        next_pt = points[(i + 1) % n]
        angle = _edge_angle_deg(cur_pt, next_pt)
        length = math.hypot(next_pt[0] - cur_pt[0], next_pt[1] - cur_pt[1])
        if low_deg <= angle <= high_deg and length >= min_diagonal_len_px:
            return None

    return points


def _detect_rooms(
    wall_mask: NDArray[np.uint8],
    img_h: int,
    img_w: int,
    settings: Settings | None = None,
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
      6. Sanitize the contour (ADR-001, 10-cv-01): ``approxPolyDP`` does not
         always collapse a spurious diagonal vertex left by a mask artefact.
         When ``settings.cv_room_contour_sanitize_enabled`` is True (default),
         ``_sanitize_room_polygon`` removes vertices whose two adjacent edges
         both fall in the diagonal angle band above a minimum length. If no
         ortho-recoverable polygon remains, the room is discarded (AC-2).
         With the flag off (or ``settings=None``), behaviour is unchanged.

    Args:
        wall_mask: Binary mask where walls = 255.
        img_h: Image height in pixels.
        img_w: Image width in pixels.
        settings: Runtime settings. If None, contour sanitizing does not run.

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

    sanitize_enabled = (
        settings is not None and settings.cv_room_contour_sanitize_enabled
    )
    diag_low_deg = (
        settings.cv_wall_diagonal_filter_low_deg if settings is not None else 0.0
    )
    diag_high_deg = (
        settings.cv_wall_diagonal_filter_high_deg if settings is not None else 0.0
    )
    diag_min_len_px = (
        float(settings.cv_room_contour_diag_min_len_px) if settings is not None else 0.0
    )
    edges_sanitized_count = 0
    rooms_dropped_count = 0

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

        if sanitize_enabled:
            vertex_count_before = len(polygon)
            sanitized = _sanitize_room_polygon(
                polygon, diag_low_deg, diag_high_deg, diag_min_len_px
            )
            if sanitized is None:
                rooms_dropped_count += 1
                continue
            edges_sanitized_count += vertex_count_before - len(sanitized)
            polygon = sanitized

        rooms.append(Room(polygon=polygon, area_px=area))

    if sanitize_enabled:
        _engine_logger.info(
            "cv_room_contour_sanitized",
            extra={
                "room_contour_edges_sanitized": edges_sanitized_count,
                "rooms_dropped_diagonal_contour": rooms_dropped_count,
            },
        )

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


def _gap_near_junction(
    gap_start: float,
    gap_end: float,
    wall_perp_pos: float,
    junctions: list[Point],
    is_horizontal: bool,
) -> bool:
    """Return True if either endpoint of a gap is within range of a junction.

    A gap is considered "near a junction" when any fused endpoint from
    _fuse_junctions satisfies:
      - perpendicular distance to the wall line < _JUNCTION_GAP_PERP_TOL_PX
      - primary-axis distance to gap_start OR gap_end < _OPENING_MIN_WALL_SPAN_PX

    The primary-axis window matches the wide span used as the main filter: if
    a junction falls within that distance of a gap endpoint the short flanking
    segment is most likely caused by the corner, not by noise.

    Args:
        gap_start: Gap start coordinate along the primary axis.
        gap_end: Gap end coordinate along the primary axis.
        wall_perp_pos: Perpendicular coordinate of the wall line.
        junctions: Junction points from _fuse_junctions.
        is_horizontal: True for horizontal walls (primary = x, perp = y).

    Returns:
        True when a qualifying junction is found near either gap endpoint.
    """
    primary_search_px = float(_OPENING_MIN_WALL_SPAN_PX)
    for jx, jy in junctions:
        j_perp = jy if is_horizontal else jx
        j_primary = jx if is_horizontal else jy
        if abs(j_perp - wall_perp_pos) > _JUNCTION_GAP_PERP_TOL_PX:
            continue
        if (
            abs(j_primary - gap_start) < primary_search_px
            or abs(j_primary - gap_end) < primary_search_px
        ):
            return True
    return False


def _arc_near_gap(
    gap_cx: float,
    gap_cy: float,
    arc_centers: list[tuple[float, float]],
    proximity_px: float,
) -> bool:
    """Return True if any detected arc centre is within *proximity_px* of the gap centre.

    Args:
        gap_cx: Gap centre x coordinate.
        gap_cy: Gap centre y coordinate.
        arc_centers: List of (cx, cy) from _detect_door_arcs.
        proximity_px: Maximum Euclidean distance to the gap centre.

    Returns:
        True when at least one arc is within range.
    """
    for ax, ay in arc_centers:
        if math.hypot(ax - gap_cx, ay - gap_cy) < proximity_px:
            return True
    return False


def _detect_door_arcs(
    gray: NDArray[np.uint8],
) -> list[tuple[float, float]]:
    """Detect door-swing arcs (semicircles) in the grayscale image.

    Uses HoughCircles to find circular arcs whose radius falls in the door-gap
    range.  The returned centres can be used to boost confidence for gap
    candidates found near them (07-cv-07 criterion b).

    Intentionally permissive parameters (low param2 accumulator threshold)
    to avoid missing genuine door arcs; false positives are tolerated because
    the backend (F4) performs the final spatial validation.

    Args:
        gray: Grayscale image (pre-blur recommended for edge stability).

    Returns:
        List of (cx, cy) centre coordinates for detected arcs.  Empty when
        HoughCircles finds no candidates or the call fails.
    """
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=float(_DOOR_GAP_MIN_PX),
        param1=50.0,
        param2=25.0,
        minRadius=_DOOR_GAP_MIN_PX // 2,
        maxRadius=_DOOR_GAP_MAX_PX // 2,
    )
    if circles is None:
        return []
    return [(float(c[0]), float(c[1])) for c in circles[0].tolist()]


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


def _build_opening_candidate(
    gap_start: float,
    gap_size: float,
    perp_pos: float,
    left_span: float,
    right_span: float,
    is_horizontal: bool,
    relaxed_span_px: int,
    active_junctions: list[Point],
    arc_centers: list[tuple[float, float]] | None,
    half_thick: float,
) -> Opening | None:
    """Build an Opening candidate for a gap, or return None if it should be skipped.

    Encapsulates all span-threshold, junction-proximity, arc-confidence, and
    Opening-construction logic so that the outer loops in _detect_openings stay
    below the PLR branch/statement limits.

    Args:
        gap_start: Start of the gap along the primary axis.
        gap_size: Width/height of the gap in pixels.
        perp_pos: Perpendicular coordinate of the wall line.
        left_span: Length of the merged wall interval to the left/above the gap.
        right_span: Length of the merged wall interval to the right/below the gap.
        is_horizontal: True for horizontal walls (primary = x, perp = y).
        relaxed_span_px: Span threshold for junction-adjacent gaps.
        active_junctions: Junction points for proximity check.
        arc_centers: Arc centre coordinates for confidence boosting.
        half_thick: Half of _WALL_THICKNESS_EST_PX for bbox construction.

    Returns:
        Opening if the gap qualifies, None otherwise.
    """
    near_jxn = bool(active_junctions) and _gap_near_junction(
        gap_start, gap_start + gap_size, perp_pos, active_junctions, is_horizontal
    )
    min_span = relaxed_span_px if near_jxn else _OPENING_MIN_WALL_SPAN_PX
    if left_span < min_span or right_span < min_span:
        return None

    type_candidate, confidence = _classify_gap(gap_size)

    if near_jxn and (
        left_span < _OPENING_MIN_WALL_SPAN_PX or right_span < _OPENING_MIN_WALL_SPAN_PX
    ):
        confidence = min(confidence, _OPENING_RELAXED_SPAN_CONFIDENCE)

    gap_cx = gap_start + gap_size / 2.0
    gap_cy = perp_pos
    if arc_centers and _arc_near_gap(gap_cx, gap_cy, arc_centers, float(gap_size)):
        confidence = _DOOR_ARC_CONFIDENCE

    if is_horizontal:
        bbox = (
            gap_start,
            perp_pos - half_thick,
            gap_size,
            float(_WALL_THICKNESS_EST_PX),
        )
    else:
        bbox = (
            perp_pos - half_thick,
            gap_start,
            float(_WALL_THICKNESS_EST_PX),
            gap_size,
        )

    return Opening(type_candidate=type_candidate, bbox=bbox, confidence=confidence)


def _detect_openings(
    walls: list[Wall],
    junctions: list[Point] | None = None,
    arc_centers: list[tuple[float, float]] | None = None,
    settings: Settings | None = None,
) -> list[Opening]:
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

    Generous emission (07-cv-07):
      - Gaps whose endpoint is adjacent to a wall junction use a relaxed span
        threshold (settings.cv_opening_min_wall_span_px, default 60 px) instead
        of the wide 170 px constant.  Confidence is capped at
        _OPENING_RELAXED_SPAN_CONFIDENCE (0.35) when the relaxed path was needed.
      - Gaps NOT adjacent to a junction keep the wide 170 px threshold so
        mid-wall artefacts are not promoted (criterion 3).
      - When a door-swing arc is detected near a gap, confidence is overridden
        to _DOOR_ARC_CONFIDENCE (0.7).
      - Aggressive filtering is PROHIBITED here — delegate to backend (F4,
        ADR-009).

    The engine never decides the final type — the LLM in vitrina does (ADR-009).
    If no openings are detected, returns an empty list (never an error).

    Args:
        walls: Wall segments detected by _detect_walls.
        junctions: Junction points from _fuse_junctions for relaxed span logic.
            None disables the junction proximity check (all gaps use wide span).
        arc_centers: Arc centre coordinates from _detect_door_arcs for
            confidence boosting.  None disables arc check.
        settings: Runtime settings.  Provides cv_opening_min_wall_span_px.

    Returns:
        List of Opening candidates (possibly empty).
    """
    if not walls:
        return []

    # Resolved relaxed span for junction-adjacent gaps.
    relaxed_span_px = (
        settings.cv_opening_min_wall_span_px
        if settings is not None
        else _OPENING_MIN_WALL_SPAN_JUNCTION_PX
    )
    active_junctions: list[Point] = junctions or []

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
            gap_w = merged[idx + 1][0] - gap_start
            # H2: discard gaps outside the plausible opening range.
            if not (_OPENING_MIN_ABERTURA_PX <= gap_w <= _OPENING_MAX_GAP_PX):
                continue
            candidate = _build_opening_candidate(
                gap_start=gap_start,
                gap_size=gap_w,
                perp_pos=y_pos,
                left_span=merged[idx][1] - merged[idx][0],
                right_span=merged[idx + 1][1] - merged[idx + 1][0],
                is_horizontal=True,
                relaxed_span_px=relaxed_span_px,
                active_junctions=active_junctions,
                arc_centers=arc_centers,
                half_thick=half_thick,
            )
            if candidate is not None:
                openings.append(candidate)

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
            gap_h = merged_v[idx + 1][0] - gap_start
            # H2: discard gaps outside the plausible opening range.
            if not (_OPENING_MIN_ABERTURA_PX <= gap_h <= _OPENING_MAX_GAP_PX):
                continue
            candidate = _build_opening_candidate(
                gap_start=gap_start,
                gap_size=gap_h,
                perp_pos=x_pos,
                left_span=merged_v[idx][1] - merged_v[idx][0],
                right_span=merged_v[idx + 1][1] - merged_v[idx + 1][0],
                is_horizontal=False,
                relaxed_span_px=relaxed_span_px,
                active_junctions=active_junctions,
                arc_centers=arc_centers,
                half_thick=half_thick,
            )
            if candidate is not None:
                openings.append(candidate)

    return openings


def _detect_scale(
    gray: NDArray[np.uint8] | None,
    settings: Settings | None,
) -> Scale:
    """Detect scale from dimension annotations (cotas) via OCR (ADR-011).

    Wraps detect_scale_from_ocr() with a thin guard layer.  Never raises.

    When CV_SCALE_OCR_ENABLED is False (or settings is None, or gray is None),
    falls back to source="none" — identical to the Phase 1 stub behaviour.

    Flow:
      1. If not enabled or prerequisites absent → Scale(source=none).
      2. Delegate to scale_ocr.detect_scale_from_ocr(gray, settings).
         That function is itself exception-safe and degrades gracefully if
         pytesseract / the tesseract binary is unavailable.

    Args:
        gray:     Grayscale image (pre-blur) from the current extract() call,
                  as stored in OpenCVClassicEngine._gray.  May be None when
                  called from a unit test that bypasses the full pipeline.
        settings: Runtime settings.  May be None in lightweight unit tests.

    Returns:
        Scale.  Never raises.
    """
    if gray is None or settings is None:
        return Scale(source=ScaleSource.none)

    if not settings.cv_scale_ocr_enabled:
        return Scale(source=ScaleSource.none)

    return detect_scale_from_ocr(gray, settings)


# ---------------------------------------------------------------------------
# F1 — Closed wall mask for room detection
# ---------------------------------------------------------------------------


def _build_closed_wall_mask_for_rooms(
    wall_mask: NDArray[np.uint8],
    close_h_gap_px: int = _ROOM_CLOSE_GAP_PX,
    close_v_gap_px: int = _ROOM_CLOSE_GAP_PX,
) -> NDArray[np.uint8]:
    """Build a version of the wall mask with architectural openings bridged.

    Applies two directional morphological closes with independent kernel sizes:

    - Horizontal close (kernel: close_h_gap_px x 1): bridges openings along the
      x direction, i.e., gaps in vertical walls (door openings in interior
      dividers).  The H gap must be *smaller* than the narrowest room width to
      avoid inadvertently filling narrow rooms (bathrooms, corridors) that are
      bounded by two parallel vertical walls.

    - Vertical close (kernel: 1 x close_v_gap_px): bridges openings along the
      y direction, i.e., gaps in horizontal walls (wide sliding doors or
      passages in the top/bottom perimeter and floor-plate dividers).

    Using asymmetric gaps (H < V) fixes detection of narrow rooms in dense
    residential plans where bathrooms can be as narrow as ~130 px (≈1.0 m at
    2000 px) while wide door/passage openings still need a larger V close.

    This mask is used EXCLUSIVELY for the CCA room-detection step.  All other
    pipeline steps (wall detection, opening detection) use the original mask.

    Args:
        wall_mask: Binary mask where walls = 255.
        close_h_gap_px: Horizontal close kernel width (px).  Default falls back
            to _ROOM_CLOSE_GAP_PX for backward compatibility.
        close_v_gap_px: Vertical close kernel height (px).  Default falls back
            to _ROOM_CLOSE_GAP_PX for backward compatibility.

    Returns:
        Binary mask with openings filled in, suitable for CCA.
    """
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_h_gap_px, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_v_gap_px))
    closed: NDArray[np.uint8] = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, h_kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, v_kernel)
    return closed


# ---------------------------------------------------------------------------
# F3 helpers — wall centerline estimation (07-cv-03)
# ---------------------------------------------------------------------------


def _sample_dt_along_segment(
    dt: NDArray[np.float32],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> list[float]:
    """Sample distance-transform values at regular intervals along a segment.

    Samples every _DT_SAMPLE_STEP_PX pixels.  Coordinates are clamped to the
    array bounds so endpoint artefacts near the image border are handled
    gracefully.

    Args:
        dt: Float32 distance transform of the wall mask.
        x1, y1, x2, y2: Segment endpoints in pixel coordinates.

    Returns:
        List of DT values (always contains at least one sample at the midpoint).
    """
    h, w = dt.shape
    length = math.hypot(x2 - x1, y2 - y1)
    n_steps = max(1, int(length / _DT_SAMPLE_STEP_PX))
    values: list[float] = []
    for i in range(n_steps + 1):
        t = i / n_steps
        px = max(0, min(round(x1 + t * (x2 - x1)), w - 1))
        py = max(0, min(round(y1 + t * (y2 - y1)), h - 1))
        values.append(float(dt[py, px]))
    return values


def _estimate_global_wall_thickness_px(
    walls: list[Wall],
    dt: NDArray[np.float32],
) -> float:
    """Estimate the typical wall stroke thickness in pixels.

    For each Hough segment, computes ``2 x median(DT samples)``.  The global
    estimate is the median of all per-segment thicknesses, making it robust to
    short noise segments with atypical DT readings.

    Args:
        walls: Raw Hough segments from _detect_walls.
        dt: Float32 distance transform of the (cleaned) wall mask.

    Returns:
        Estimated wall thickness in pixels.  Falls back to
        ``_CENTERLINE_MIN_TOL_PX`` when no samples can be collected.
    """
    per_seg: list[float] = []
    for wall in walls:
        x1, y1 = wall.start
        x2, y2 = wall.end
        samples = _sample_dt_along_segment(dt, x1, y1, x2, y2)
        if samples:
            per_seg.append(2.0 * float(np.median(np.array(samples, dtype=np.float32))))
    if not per_seg:
        return _CENTERLINE_MIN_TOL_PX
    thickness = float(np.median(np.array(per_seg, dtype=np.float32)))
    return max(thickness, _CENTERLINE_MIN_TOL_PX)


def _group_indices_by_proximity(
    sorted_values: list[float],
    tolerance: float,
) -> list[tuple[int, int]]:
    """Partition a sorted sequence into clusters by proximity.

    Two consecutive elements belong to the same cluster when their difference
    is <= *tolerance*.

    Args:
        sorted_values: Pre-sorted list of float values.
        tolerance: Maximum gap to remain in the same cluster.

    Returns:
        List of (start_idx, end_idx_inclusive) pairs for each cluster.
    """
    if not sorted_values:
        return []
    groups: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(sorted_values)):
        if sorted_values[i] - sorted_values[i - 1] > tolerance:
            groups.append((start, i - 1))
            start = i
    groups.append((start, len(sorted_values) - 1))
    return groups


def _thickness_from_dt_samples(
    dt: NDArray[np.float32],
    segs: list[tuple[float, float, float, float]],
) -> float | None:
    """Compute 2 x median(DT samples) for a group of collinear segments.

    Args:
        dt: Float32 distance transform of the wall mask.
        segs: List of (x1, y1, x2, y2) tuples.

    Returns:
        Thickness in pixels, or None when no samples are available.
    """
    all_samples: list[float] = []
    for x1, y1, x2, y2 in segs:
        all_samples.extend(_sample_dt_along_segment(dt, x1, y1, x2, y2))
    if not all_samples:
        return None
    return float(2.0 * np.median(np.array(all_samples, dtype=np.float32)))


def _group_indices_by_local_thickness(
    sorted_values: list[float],
    segs_by_value: list[tuple[float, float, float, float]],
    dt: NDArray[np.float32],
    fallback_tolerance: float,
) -> tuple[list[tuple[int, int]], int]:
    """Partition a sorted sequence into clusters using LOCAL wall thickness.

    Unlike ``_group_indices_by_proximity`` (which applies one global tolerance
    derived from the whole plan), this walks the sorted sequence and, for each
    candidate merge between consecutive elements, samples the distance
    transform in the neighbourhood of *both* segments to derive a local
    thickness estimate. Two consecutive traces merge when their perpendicular
    separation is <= the local thickness of that zone of the plan — this
    avoids collapsing distinct thin/thick walls under a single plan-wide value
    (ADR-003, part A).

    Args:
        sorted_values: Pre-sorted list of perpendicular-position values (one
            per segment in *segs_by_value*, same order/index alignment).
        segs_by_value: Segments aligned 1:1 with *sorted_values*, already
            sorted by the same key.
        dt: Float32 distance transform of the (cleaned) wall mask.
        fallback_tolerance: Tolerance used when local DT sampling yields no
            usable value (e.g. the segment pair falls outside the mask).

    Returns:
        Tuple of (list of (start_idx, end_idx_inclusive) cluster pairs,
        merges_count) where merges_count is the number of consecutive-pair
        merges applied using the local (not fallback) tolerance — used to
        populate the ``walls_merged_local_thickness`` counter.
    """
    if not sorted_values:
        return [], 0
    groups: list[tuple[int, int]] = []
    merges_count = 0
    start = 0
    for i in range(1, len(sorted_values)):
        local_thickness = _thickness_from_dt_samples(
            dt, [segs_by_value[i - 1], segs_by_value[i]]
        )
        tolerance = (
            max(local_thickness, _CENTERLINE_MIN_TOL_PX)
            if local_thickness is not None
            else fallback_tolerance
        )
        if sorted_values[i] - sorted_values[i - 1] > tolerance:
            groups.append((start, i - 1))
            start = i
        elif local_thickness is not None:
            merges_count += 1
    groups.append((start, len(sorted_values) - 1))
    return groups, merges_count


# ---------------------------------------------------------------------------
# F3 — Consolidated walls
# ---------------------------------------------------------------------------


_RawSeg = tuple[float, float, float, float]  # (x1, y1, x2, y2)


def _merge_segs_into_walls(
    segs: list[_RawSeg],
    perp_pos: float,
    is_horizontal: bool,
    thickness: float | None,
    out: list[Wall],
) -> None:
    """Merge a group of collinear segments into Wall objects and append to *out*.

    Applies _merge_intervals along the primary axis; each resulting span that
    meets the minimum-length requirement produces one Wall.

    Args:
        segs: Collinear segments in the same perpendicular band.
        perp_pos: The shared perpendicular coordinate (y for H, x for V).
        is_horizontal: True for horizontal walls (primary axis = x).
        thickness: Wall thickness in pixels, or None (legacy).
        out: Accumulator list; new Walls are appended here.
    """
    if is_horizontal:
        intervals: list[tuple[float, float]] = sorted(
            (min(x1, x2), max(x1, x2)) for x1, _y1, x2, _y2 in segs
        )
        for start, end in _merge_intervals(
            intervals, tol=float(_OPENING_MIN_ABERTURA_PX - 1)
        ):
            if end - start >= _MIN_WALL_LENGTH_PX:
                out.append(
                    Wall(
                        start=(start, perp_pos),
                        end=(end, perp_pos),
                        thickness=thickness,
                    )
                )
    else:
        intervals_v: list[tuple[float, float]] = sorted(
            (min(y1, y2), max(y1, y2)) for _x1, y1, _x2, y2 in segs
        )
        for start, end in _merge_intervals(
            intervals_v, tol=float(_OPENING_MIN_ABERTURA_PX - 1)
        ):
            if end - start >= _MIN_WALL_LENGTH_PX:
                out.append(
                    Wall(
                        start=(perp_pos, start),
                        end=(perp_pos, end),
                        thickness=thickness,
                    )
                )


def _legacy_bin_consolidate(
    h_segs: list[_RawSeg],
    v_segs: list[_RawSeg],
) -> list[Wall]:
    """Fixed-bin consolidation — original pre-07-cv-03 behaviour (flag off)."""
    out: list[Wall] = []
    h_buckets: dict[int, list[_RawSeg]] = {}
    v_buckets: dict[int, list[_RawSeg]] = {}

    for x1, y1, x2, y2 in h_segs:
        bin_key = int((y1 + y2) / 2) // _OPENING_COLLINEAR_TOL_PX
        h_buckets.setdefault(bin_key, []).append((x1, y1, x2, y2))

    for x1, y1, x2, y2 in v_segs:
        bin_key = int((x1 + x2) / 2) // _OPENING_COLLINEAR_TOL_PX
        v_buckets.setdefault(bin_key, []).append((x1, y1, x2, y2))

    for bin_key, segs in h_buckets.items():
        y_pos = bin_key * _OPENING_COLLINEAR_TOL_PX + _OPENING_COLLINEAR_TOL_PX / 2.0
        _merge_segs_into_walls(segs, y_pos, is_horizontal=True, thickness=None, out=out)

    for bin_key, segs in v_buckets.items():
        x_pos = bin_key * _OPENING_COLLINEAR_TOL_PX + _OPENING_COLLINEAR_TOL_PX / 2.0
        _merge_segs_into_walls(
            segs, x_pos, is_horizontal=False, thickness=None, out=out
        )

    return out


def _centerline_dt_consolidate(
    walls: list[Wall],
    h_segs: list[_RawSeg],
    v_segs: list[_RawSeg],
    wall_mask: NDArray[np.uint8],
    settings: Settings | None = None,
) -> list[Wall]:
    """DT-based centerline consolidation (flag on, 07-cv-03).

    Grouping strategy depends on ``settings.cv_wall_local_thickness_enabled``
    (10-cv-05, ADR-003 part A):

    - **True (default):** each candidate merge between consecutive parallel
      traces is evaluated against a LOCAL thickness sampled from the DT in
      the neighbourhood of that specific pair, instead of one global value
      for the whole plan. Emits ``walls_merged_local_thickness``.
    - **False:** legacy behaviour — a single global thickness estimate
      (``_estimate_global_wall_thickness_px``) is used as the tolerance for
      every group in the plan.
    """
    dt: NDArray[np.float32] = cv2.distanceTransform(wall_mask, cv2.DIST_L2, 5)
    wall_thickness_px = _estimate_global_wall_thickness_px(walls, dt)
    _engine_logger.debug(
        "cv_wall_centerline_thickness_estimated",
        extra={"wall_thickness_px": round(wall_thickness_px, 1)},
    )

    local_thickness_enabled: bool = (
        settings is not None and settings.cv_wall_local_thickness_enabled
    )

    out: list[Wall] = []
    total_merges = 0

    if h_segs:
        h_segs.sort(key=lambda s: (s[1] + s[3]) / 2)
        y_mids = [(s[1] + s[3]) / 2 for s in h_segs]
        if local_thickness_enabled:
            h_groups, h_merges = _group_indices_by_local_thickness(
                y_mids, h_segs, dt, wall_thickness_px
            )
            total_merges += h_merges
        else:
            h_groups = _group_indices_by_proximity(y_mids, wall_thickness_px)
        for g_start, g_end in h_groups:
            group = h_segs[g_start : g_end + 1]
            y_pos = sum((s[1] + s[3]) / 2 for s in group) / len(group)
            thickness = _thickness_from_dt_samples(dt, group)
            _merge_segs_into_walls(
                group, y_pos, is_horizontal=True, thickness=thickness, out=out
            )

    if v_segs:
        v_segs.sort(key=lambda s: (s[0] + s[2]) / 2)
        x_mids = [(s[0] + s[2]) / 2 for s in v_segs]
        if local_thickness_enabled:
            v_groups, v_merges = _group_indices_by_local_thickness(
                x_mids, v_segs, dt, wall_thickness_px
            )
            total_merges += v_merges
        else:
            v_groups = _group_indices_by_proximity(x_mids, wall_thickness_px)
        for g_start, g_end in v_groups:
            group_v = v_segs[g_start : g_end + 1]
            x_pos = sum((s[0] + s[2]) / 2 for s in group_v) / len(group_v)
            thickness_v = _thickness_from_dt_samples(dt, group_v)
            _merge_segs_into_walls(
                group_v, x_pos, is_horizontal=False, thickness=thickness_v, out=out
            )

    if local_thickness_enabled:
        _engine_logger.info(
            "walls_merged_local_thickness",
            extra={"count": total_merges},
        )

    return out


def _consolidate_walls(
    walls: list[Wall],
    wall_mask: NDArray[np.uint8] | None = None,
    settings: Settings | None = None,
) -> list[Wall]:
    """Merge collinear wall segments produced by HoughLinesP into longer walls.

    HoughLinesP fragments each physical wall into many short overlapping
    segments (e.g., 131 for ~5 walls in a simple floor plan).  Two modes are
    available, controlled by ``settings.cv_wall_centerline_enabled``:

    **Centerline mode (default, flag on):**
      Estimates wall thickness via distanceTransform, groups parallel Hough
      traces within that tolerance, and places the resulting Wall at the
      average perpendicular coordinate with ``thickness`` in pixels.

    **Legacy mode (flag off):**
      Original fixed-bin behaviour; ``thickness`` is None.

    Args:
        walls: Raw wall segments from _detect_walls.
        wall_mask: Binary wall mask (walls = 255).  Required for centerline mode;
            if None the function falls back to legacy mode regardless of the flag.
        settings: Runtime settings.  If None falls back to legacy mode.

    Returns:
        Consolidated list of Wall objects.  Emits ``walls_before_consolidation``
        and ``walls_after_consolidation`` log records with wall counts.
    """
    if not walls:
        return []

    _engine_logger.info(
        "walls_before_consolidation",
        extra={"walls_count": len(walls)},
    )

    # Classify segments by orientation.
    h_segs: list[_RawSeg] = []
    v_segs: list[_RawSeg] = []
    diagonal: list[Wall] = []
    for wall in walls:
        x1, y1 = wall.start
        x2, y2 = wall.end
        angle_deg = math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1)))
        if angle_deg < _OPENING_ANGLE_TOL_DEG:
            h_segs.append((x1, y1, x2, y2))
        elif angle_deg > (90.0 - _OPENING_ANGLE_TOL_DEG):
            v_segs.append((x1, y1, x2, y2))
        else:
            diagonal.append(wall)

    centerline_enabled: bool = (
        wall_mask is not None
        and settings is not None
        and settings.cv_wall_centerline_enabled
    )

    if centerline_enabled:
        consolidated = _centerline_dt_consolidate(
            walls,
            h_segs,
            v_segs,
            wall_mask,  # type: ignore[arg-type]
            settings,
        )
    else:
        consolidated = _legacy_bin_consolidate(h_segs, v_segs)

    # F1 (08-cv-01) — discard diagonal segments in [low_deg, high_deg].
    # Applied only when the flag is enabled; otherwise all diagonal segments
    # are kept (pre-08 behaviour, AC-3).
    diagonal_filter_enabled: bool = (
        settings is not None and settings.cv_wall_diagonal_filter_enabled
    )
    if diagonal_filter_enabled and settings is not None:
        low_deg = settings.cv_wall_diagonal_filter_low_deg
        high_deg = settings.cv_wall_diagonal_filter_high_deg
        kept_diagonal: list[Wall] = []
        discarded_count = 0
        for wall in diagonal:
            x1, y1 = wall.start
            x2, y2 = wall.end
            angle_deg = math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1)))
            if low_deg <= angle_deg <= high_deg:
                discarded_count += 1
            else:
                kept_diagonal.append(wall)
        consolidated.extend(kept_diagonal)
        _engine_logger.info(
            "diagonal_walls_discarded",
            extra={"count": discarded_count},
        )
    else:
        consolidated.extend(diagonal)

    _engine_logger.info(
        "walls_after_consolidation",
        extra={"walls_count": len(consolidated)},
    )

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
# F4 — Orthogonal snapping and junction fusion (07-cv-04)
# ---------------------------------------------------------------------------


def _snap_walls_orthogonal(walls: list[Wall]) -> list[Wall]:
    """Snap near-horizontal / near-vertical segments to exact H/V axes.

    For each wall segment the angle from horizontal is computed.  If the
    segment is within *_SNAP_ANGLE_TOL_DEG* of either axis it is projected
    onto that axis exactly:

    - Near-horizontal (|angle| < tol): ``start.y = end.y = mean(y1, y2)``
    - Near-vertical (|90 deg - angle| < tol): ``start.x = end.x = mean(x1, x2)``
    - Otherwise (diagonal): returned unchanged.

    This removes sub-pixel misalignments introduced by HoughLinesP that cause
    "tooth" artefacts at wall junctions.

    Args:
        walls: Consolidated wall segments (post-centerline).

    Returns:
        New list of Wall objects.  Diagonal walls are the same objects;
        snapped walls are reconstructed with the corrected coordinates.
    """
    snapped: list[Wall] = []
    for wall in walls:
        x1, y1 = wall.start
        x2, y2 = wall.end
        angle_deg = math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1)))

        if angle_deg < _SNAP_ANGLE_TOL_DEG:
            # Force exact horizontal: average the two y-coordinates.
            y_avg = (y1 + y2) / 2.0
            snapped.append(
                Wall(start=(x1, y_avg), end=(x2, y_avg), thickness=wall.thickness)
            )
        elif angle_deg > (90.0 - _SNAP_ANGLE_TOL_DEG):
            # Force exact vertical: average the two x-coordinates.
            x_avg = (x1 + x2) / 2.0
            snapped.append(
                Wall(start=(x_avg, y1), end=(x_avg, y2), thickness=wall.thickness)
            )
        else:
            # Legitimate diagonal — leave intact.
            snapped.append(wall)

    return snapped


def _extend_wall_endpoint_to_value(
    coords: list[float],
    axis: int,
    target: float,
    extend_px: int,
) -> None:
    """Move the nearer endpoint of a segment (in *axis*) to *target* if eligible.

    Eligibility: the target lies strictly beyond the segment's current extent
    (i.e. in its prolongation) and the gap is <= *extend_px*.

    Args:
        coords: Mutable ``[x1, y1, x2, y2]`` list for the wall.
        axis: 0 for x-axis (horizontal wall), 1 for y-axis (vertical wall).
        target: Target coordinate value (intersection x or y).
        extend_px: Maximum gap (px) that triggers extension.
    """
    # Index offsets for (start, end) along *axis*: 0/2 for x, 1/3 for y.
    idx1, idx2 = axis, axis + 2
    v1, v2 = coords[idx1], coords[idx2]
    lo, hi = (v1, v2) if v1 <= v2 else (v2, v1)
    lo_idx, hi_idx = (idx1, idx2) if v1 <= v2 else (idx2, idx1)

    # Extend the high (rightmost/bottommost) endpoint rightward/downward.
    if target > hi and target - hi <= extend_px:
        coords[hi_idx] = target

    # Extend the low (leftmost/topmost) endpoint leftward/upward.
    if target < lo and lo - target <= extend_px:
        coords[lo_idx] = target


def _extend_to_intersection(walls: list[Wall], extend_px: int) -> list[Wall]:
    """Extend H/V wall endpoints to their geometric intersection to close gaps.

    Operates between ``_snap_walls_orthogonal`` (walls are already exact H/V)
    and ``_fuse_junctions`` (which collapses coincident endpoints into junctions).
    After snapping, the gap at a corner is a pure geometry problem: the endpoint
    of a horizontal wall and the endpoint of the vertical wall that should meet
    it are both <= *extend_px* from the intersection, but neither has been
    extended to reach it.  This phase moves those endpoints to the intersection
    so that ``_fuse_junctions`` finds distance ~0 and produces a real junction.

    Invariants:
    - Only **orthogonal pairs** are considered (one H wall x one V wall).
      Parallel pairs (H-H, V-V) are ignored: they have no finite corner
      intersection relevant to a junction.
    - An endpoint is moved to the intersection **only if**:
      (a) its distance to the intersection is ``<= extend_px``, AND
      (b) the intersection lies in the **prolongation** of the segment
          (strictly beyond the segment's current extent in the relevant axis),
          never in its interior.
    - Pure function: same cardinality in and out, no side effects.
    - Idempotent: if all gaps are already ~0 (e.g. plan-004), every endpoint
      either lies exactly at the intersection (condition b fails — already at
      the boundary) or is farther than *extend_px*, so the output is unchanged.
    - ``wall.thickness`` is preserved unchanged on every reconstructed Wall.

    Coordinate system: image convention (x grows right, y grows down).

    Args:
        walls: Wall segments after ``_snap_walls_orthogonal`` — each wall is
            either exactly horizontal (``start.y == end.y``) or exactly vertical
            (``start.x == end.x``).  Diagonal walls are left untouched.
        extend_px: Maximum gap in pixels that triggers extension.  Must be > 0.
            Comes from ``settings.cv_junction_extend_px`` (default 40, calibrated
            for ~2000 px normalised images).

    Returns:
        New list of Wall objects with the same length as *walls*.
    """
    # Work on mutable coordinate lists so we can update endpoints across
    # multiple pairs before reconstructing Wall objects at the end.
    # Each entry: [x1, y1, x2, y2]
    coords: list[list[float]] = [
        [w.start[0], w.start[1], w.end[0], w.end[1]] for w in walls
    ]

    n = len(walls)
    for i in range(n):
        xi1, yi1, xi2, yi2 = coords[i]
        is_h_i = yi1 == yi2  # horizontal: same y
        is_v_i = xi1 == xi2  # vertical:   same x
        if not (is_h_i or is_v_i):
            continue  # diagonal — skip

        for j in range(i + 1, n):
            xj1, yj1, xj2, yj2 = coords[j]
            is_h_j = yj1 == yj2
            is_v_j = xj1 == xj2
            if not (is_h_j or is_v_j):
                continue  # diagonal — skip

            # Accept only strictly orthogonal pairs (one H, one V).
            if is_h_i and is_v_j:
                h_idx, v_idx = i, j
            elif is_v_i and is_h_j:
                h_idx, v_idx = j, i
            else:
                continue  # parallel pair (H-H or V-V)

            # Geometric intersection: x of vertical wall, y of horizontal wall.
            ix = coords[v_idx][0]  # wall_v.start.x == wall_v.end.x
            iy = coords[h_idx][1]  # wall_h.start.y == wall_h.end.y

            # Extend H wall endpoints along x toward ix (axis=0).
            _extend_wall_endpoint_to_value(coords[h_idx], 0, ix, extend_px)
            # Extend V wall endpoints along y toward iy (axis=1).
            _extend_wall_endpoint_to_value(coords[v_idx], 1, iy, extend_px)

    extended: list[Wall] = [
        Wall(
            start=(coords[k][0], coords[k][1]),
            end=(coords[k][2], coords[k][3]),
            thickness=walls[k].thickness,
        )
        for k in range(n)
    ]

    _engine_logger.debug(
        "cv_junction_extend_to_intersection",
        extra={"walls_count": n, "extend_px": extend_px},
    )

    return extended


def _fuse_junctions(
    walls: list[Wall],
) -> tuple[list[Wall], list[Point]]:
    """Fuse nearby endpoints from different walls into shared junction points.

    For every pair of endpoints that belong to *different* walls, if their
    Euclidean distance is strictly less than the wall thickness (``min`` of the
    two walls' thicknesses, falling back to ``_WALL_THICKNESS_EST_PX`` when
    ``thickness`` is None), both endpoints are merged to their centroid.

    Transitively close endpoints are handled via Union-Find so that three walls
    meeting at a single corner all share the same resulting junction coordinate.

    The list of junction points is returned as an auxiliary output so that
    downstream tasks (cv-07, door detection at corners) can consume it from
    ``OpenCVClassicEngine._junctions`` without re-running the fusion step.

    Args:
        walls: Wall segments (typically already snapped by _snap_walls_orthogonal).

    Returns:
        ``(updated_walls, junctions)`` where *updated_walls* has the same length
        as the input and *junctions* is the list of fused vertex coordinates
        (one entry per multi-wall junction, deduplicated).
    """
    n = len(walls)
    if n < _JUNCTION_MIN_CLUSTER_SIZE:
        return walls, []

    # Flatten endpoints into a mutable coordinate array.
    # Index layout: 2*i → start of wall i, 2*i+1 → end of wall i.
    coords: list[list[float]] = []
    for wall in walls:
        x1, y1 = wall.start
        x2, y2 = wall.end
        coords.append([x1, y1])
        coords.append([x2, y2])

    # ----- Union-Find (path-compressed) ------------------------------------
    parent = list(range(2 * n))

    def _find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    # ----- Build distance graph and union close endpoint pairs -------------
    for i in range(n):
        t_i = (
            walls[i].thickness
            if walls[i].thickness is not None
            else float(_WALL_THICKNESS_EST_PX)
        )
        for j in range(i + 1, n):
            t_j = (
                walls[j].thickness
                if walls[j].thickness is not None
                else float(_WALL_THICKNESS_EST_PX)
            )
            threshold = min(t_i, t_j)

            for ep_i in (2 * i, 2 * i + 1):
                for ep_j in (2 * j, 2 * j + 1):
                    px_i, py_i = coords[ep_i]
                    px_j, py_j = coords[ep_j]
                    if math.hypot(px_j - px_i, py_j - py_i) < threshold:
                        _union(ep_i, ep_j)

    # ----- Compute centroids for each cluster with ≥ 2 members ------------
    clusters: dict[int, list[int]] = defaultdict(list)
    for idx in range(2 * n):
        clusters[_find(idx)].append(idx)

    junctions: list[Point] = []
    for members in clusters.values():
        if len(members) < _JUNCTION_MIN_CLUSTER_SIZE:  # singleton — no junction
            continue
        cx = sum(coords[m][0] for m in members) / len(members)
        cy = sum(coords[m][1] for m in members) / len(members)
        for m in members:
            coords[m][0] = cx
            coords[m][1] = cy
        junctions.append((cx, cy))

    # ----- Reconstruct Wall objects from updated coordinates ---------------
    updated: list[Wall] = [
        Wall(
            start=(coords[2 * i][0], coords[2 * i][1]),
            end=(coords[2 * i + 1][0], coords[2 * i + 1][1]),
            thickness=walls[i].thickness,
        )
        for i in range(n)
    ]

    _engine_logger.debug(
        "cv_junctions_fused",
        extra={"junctions_count": len(junctions)},
    )

    return updated, junctions


def _filter_diagonal_residual_pass2(
    walls: list[Wall],
    settings: Settings | None,
) -> list[Wall]:
    """Second diagonal-residual filter pass, run after ``_fuse_junctions`` (ADR-017).

    Complements the pass-1 band filter in ``_consolidate_walls`` with two
    mechanisms that address a diagonal stub (stair/sectional-door remnant)
    surviving snap/extend/fuse:

    - **Mec.1 (angle re-filter):** re-evaluates each wall's angle and discards
      walls whose angle falls in the *same* ``[low_deg, high_deg]`` band used
      by pass 1. Idempotent on already-snapped H/V-exact walls: their angle is
      exactly 0deg or 90deg, which never falls inside the band.
    - **Mec.2 (minimum length):** discards any wall that is *not* H/V-exact
      (``start.x != end.x`` and ``start.y != end.y``) **and** whose Euclidean
      length is below ``settings.cv_wall_min_diagonal_len_px``. Never applies
      to an exact H/V wall regardless of its length.

    Both mechanisms are gated by the same master switch
    ``settings.cv_wall_diagonal_filter_enabled`` used by pass 1. When the flag
    is False (or settings is None), this function is a no-op and returns
    ``walls`` unchanged — preserving pre-run-08 output byte-for-byte (AC-11).

    Args:
        walls: Wall segments after ``_fuse_junctions``.
        settings: Runtime settings. If None, the filter does not run.

    Returns:
        Filtered list of Wall objects (same objects, no reconstruction of
        surviving walls).
    """
    if settings is None or not settings.cv_wall_diagonal_filter_enabled:
        return walls

    low_deg = settings.cv_wall_diagonal_filter_low_deg
    high_deg = settings.cv_wall_diagonal_filter_high_deg
    min_diagonal_len_px = settings.cv_wall_min_diagonal_len_px

    kept: list[Wall] = []
    discarded_by_angle = 0
    discarded_by_length = 0
    for wall in walls:
        x1, y1 = wall.start
        x2, y2 = wall.end
        is_exact_hv = x1 == x2 or y1 == y2

        angle_deg = math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1)))
        if low_deg <= angle_deg <= high_deg:
            discarded_by_angle += 1
            continue

        if not is_exact_hv:
            length_px = math.hypot(x2 - x1, y2 - y1)
            if length_px < min_diagonal_len_px:
                discarded_by_length += 1
                continue

        kept.append(wall)

    _engine_logger.info(
        "cv_wall_diagonal_pass2_filtered",
        extra={
            "count_by_angle": discarded_by_angle,
            "count_by_length": discarded_by_length,
        },
    )

    return kept


# ---------------------------------------------------------------------------
# F5 — Window pattern detection (07-cv-06)
# ---------------------------------------------------------------------------


def _count_foreground_runs(profile: NDArray[np.uint8]) -> int:
    """Count distinct foreground (255) runs in a 1-D binary mask profile.

    A run is a contiguous sequence of non-zero pixels bordered by zero pixels
    or the array boundary.

    Args:
        profile: 1-D uint8 array (values 0 or 255).

    Returns:
        Number of distinct foreground runs (0 when profile is all background).
    """
    in_run = False
    count = 0
    for px in profile:
        if px > 0 and not in_run:
            in_run = True
            count += 1
        elif px == 0 and in_run:
            in_run = False
    return count


def _maybe_emit_window(
    run_start: int,
    run_end: int,
    n_steps: int,
    is_horizontal: bool,
    axis_start: float,
    axis_end: float,
    y_center: float,
    x_center: float,
    half_thick: float,
    full_thick: float,
    openings: list[Opening],
) -> None:
    """Emit a window Opening if the candidate span meets size requirements.

    Centralises the span-length check and bbox construction so the main loop
    in ``_detect_window_pattern`` does not repeat the logic for both mid-span
    breaks and the trailing run.

    Args:
        run_start: First flag index in the contiguous True run.
        run_end: One-past-last flag index (exclusive).
        n_steps: Total number of sampling steps for coordinate mapping.
        is_horizontal: True for horizontal walls; False for vertical.
        axis_start, axis_end: Wall extent along the primary axis (px).
        y_center: Wall centerline y coordinate (used for horizontal walls).
        x_center: Wall centerline x coordinate (used for vertical walls).
        half_thick: Half the wall thickness in px.
        full_thick: Full wall thickness in px.
        openings: Accumulator list; new Opening is appended when criteria are met.
    """
    span_count = run_end - run_start
    if span_count < _WIN_MIN_CONSECUTIVE_PROFILES:
        return
    span_px = float(span_count * _WIN_PROFILE_STEP_PX)
    if span_px > _WIN_MAX_SPAN_PX:
        return
    t0 = run_start / n_steps
    if is_horizontal:
        open_x = axis_start + t0 * (axis_end - axis_start)
        openings.append(
            Opening(
                type_candidate=OpeningTypeCandidate.window,
                bbox=(open_x, y_center - half_thick, span_px, full_thick),
                confidence=_WIN_CONFIDENCE,
            )
        )
    else:
        open_y = axis_start + t0 * (axis_end - axis_start)
        openings.append(
            Opening(
                type_candidate=OpeningTypeCandidate.window,
                bbox=(x_center - half_thick, open_y, full_thick, span_px),
                confidence=_WIN_CONFIDENCE,
            )
        )


def _build_window_flags(
    original_mask: NDArray[np.uint8],
    n_steps: int,
    is_horizontal: bool,
    axis_start: float,
    axis_end: float,
    perp_center: float,
    scan_half: float,
) -> list[bool]:
    """Sample perpendicular profiles along a wall and flag double-line positions.

    For each of *n_steps + 1* equidistant positions along the primary axis,
    extract a 1-D profile perpendicular to the wall and check whether it
    contains exactly ``_WIN_EXPECTED_RUNS`` foreground runs (the two parallel
    lines of a window frame).

    Args:
        original_mask: Pre-filter binary mask (walls = 255).
        n_steps: Number of sampling intervals along the wall axis.
        is_horizontal: True for horizontal walls (primary axis = x).
        axis_start, axis_end: Wall extent along the primary axis (px).
        perp_center: Centerline coordinate on the perpendicular axis.
        scan_half: Half-width of the scan extent around *perp_center* (px).

    Returns:
        Boolean list of length *n_steps + 1*.  True indicates a window profile.
    """
    mask_h, mask_w = original_mask.shape[:2]
    flags: list[bool] = []
    for step in range(n_steps + 1):
        t = step / n_steps
        pos = axis_start + t * (axis_end - axis_start)
        if is_horizontal:
            px_x = max(0, min(round(pos), mask_w - 1))
            py0 = max(0, round(perp_center - scan_half))
            py1 = min(mask_h, round(perp_center + scan_half))
            if py1 <= py0:
                flags.append(False)
            else:
                flags.append(
                    _count_foreground_runs(original_mask[py0:py1, px_x])
                    == _WIN_EXPECTED_RUNS
                )
        else:
            py_y = max(0, min(round(pos), mask_h - 1))
            px0 = max(0, round(perp_center - scan_half))
            px1 = min(mask_w, round(perp_center + scan_half))
            if px1 <= px0:
                flags.append(False)
            else:
                flags.append(
                    _count_foreground_runs(original_mask[py_y, px0:px1])
                    == _WIN_EXPECTED_RUNS
                )
    return flags


def _scan_window_runs(
    window_flags: list[bool],
    n_steps: int,
    is_horizontal: bool,
    axis_start: float,
    axis_end: float,
    y_center: float,
    x_center: float,
    half_thick: float,
    full_thick: float,
    openings: list[Opening],
) -> None:
    """Scan contiguous True runs in *window_flags* and emit window candidates.

    Delegates span-validation and Opening construction to ``_maybe_emit_window``.

    Args:
        window_flags: Boolean flag list from ``_build_window_flags``.
        n_steps: Total sampling steps (used for coordinate mapping).
        is_horizontal: Orientation of the parent wall.
        axis_start, axis_end: Wall axis extent in pixels.
        y_center: Centerline y (for horizontal walls).
        x_center: Centerline x (for vertical walls).
        half_thick, full_thick: Wall half-/full-thickness in pixels.
        openings: Accumulator; new Opening objects are appended here.
    """
    run_start_idx: int | None = None
    for i, is_win in enumerate(window_flags):
        if is_win and run_start_idx is None:
            run_start_idx = i
        elif not is_win and run_start_idx is not None:
            _maybe_emit_window(
                run_start_idx,
                i,
                n_steps,
                is_horizontal,
                axis_start,
                axis_end,
                y_center,
                x_center,
                half_thick,
                full_thick,
                openings,
            )
            run_start_idx = None
    if run_start_idx is not None:
        _maybe_emit_window(
            run_start_idx,
            len(window_flags),
            n_steps,
            is_horizontal,
            axis_start,
            axis_end,
            y_center,
            x_center,
            half_thick,
            full_thick,
            openings,
        )


def _detect_window_pattern(
    walls: list[Wall],
    original_mask: NDArray[np.uint8],
) -> list[Opening]:
    """Detect window candidates as double-line patterns within wall spans (07-cv-06).

    In vectorial floor plans windows are drawn as two thin parallel lines
    INSIDE the wall width, not as gaps between wall segments.
    ``filter_thin_strokes`` (step 4 of ``clean_mask``) removes these thin
    frame lines before gap-based ``_detect_openings`` runs.  This function
    operates on *original_mask* — the wall mask BEFORE step 4 — where the
    thin window strands are still present.

    Algorithm:
      1. For each consolidated axis-aligned wall, sample perpendicular
         cross-sections at ``_WIN_PROFILE_STEP_PX`` intervals via
         ``_build_window_flags``.
      2. A profile with exactly ``_WIN_EXPECTED_RUNS`` (2) foreground runs
         indicates a double-line window section.
      3. ``_WIN_MIN_CONSECUTIVE_PROFILES`` or more consecutive window profiles
         form a candidate span emitted as Opening(type_candidate="window") via
         ``_scan_window_runs`` and ``_maybe_emit_window``.

    Confidence is conservative (``_WIN_CONFIDENCE = 0.35``); the LLM in
    vitrina makes the final classification (ADR-009).  Returns an empty list
    when no candidates are found — never raises.

    Args:
        walls: Consolidated Wall objects produced by the extract pipeline.
        original_mask: Binary wall mask BEFORE ``filter_thin_strokes`` (steps
            1-3 of ``clean_mask``).  Thin window frame strokes are visible here.

    Returns:
        List of Opening candidates (possibly empty).  Never raises.
    """
    if not walls:
        return []

    openings: list[Opening] = []

    for wall in walls:
        x1, y1 = wall.start
        x2, y2 = wall.end
        angle_deg = math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1)))

        is_horizontal = angle_deg < _OPENING_ANGLE_TOL_DEG
        is_vertical = angle_deg > (90.0 - _OPENING_ANGLE_TOL_DEG)
        if not (is_horizontal or is_vertical):
            continue  # diagonal walls — skip

        half_thick = (
            wall.thickness / 2.0
            if wall.thickness is not None
            else float(_WALL_THICKNESS_EST_PX) / 2.0
        )
        full_thick = (
            wall.thickness
            if wall.thickness is not None
            else float(_WALL_THICKNESS_EST_PX)
        )
        scan_half = half_thick + _WIN_PROFILE_HALF_EXTRA_PX

        if is_horizontal:
            wall_len = abs(x2 - x1)
            y_center, x_center = (y1 + y2) / 2.0, 0.0
            axis_start, axis_end = min(x1, x2), max(x1, x2)
            perp_center = y_center
        else:
            wall_len = abs(y2 - y1)
            x_center, y_center = (x1 + x2) / 2.0, 0.0
            axis_start, axis_end = min(y1, y2), max(y1, y2)
            perp_center = x_center

        if wall_len < _WIN_MIN_CONSECUTIVE_PROFILES * _WIN_PROFILE_STEP_PX:
            continue

        n_steps = max(1, int(wall_len / _WIN_PROFILE_STEP_PX))
        window_flags = _build_window_flags(
            original_mask,
            n_steps,
            is_horizontal,
            axis_start,
            axis_end,
            perp_center,
            scan_half,
        )
        _scan_window_runs(
            window_flags,
            n_steps,
            is_horizontal,
            axis_start,
            axis_end,
            y_center,
            x_center,
            half_thick,
            full_thick,
            openings,
        )

    return openings


# ---------------------------------------------------------------------------
# Staircase detection helpers (07-cv-10)
# ---------------------------------------------------------------------------


def _bbox_inside_any_room(
    bx: float,
    by: float,
    bw: float,
    bh: float,
    rooms: list[Room],
) -> bool:
    """Return True if all four bbox corners lie inside at least one room polygon.

    Uses cv2.pointPolygonTest(measureDist=False) which returns >= 0 for points
    on the boundary or inside.  All four corners must pass the test against the
    same room polygon for the bbox to be considered contained.

    Args:
        bx, by: Top-left corner of the bbox (px).
        bw, bh: Width and height of the bbox (px).
        rooms: Detected room polygons to test containment against.

    Returns:
        True if the bbox is contained in any room; False otherwise.
    """
    corners = [
        (bx, by),
        (bx + bw, by),
        (bx, by + bh),
        (bx + bw, by + bh),
    ]
    for room in rooms:
        poly = np.array([[int(p[0]), int(p[1])] for p in room.polygon], dtype=np.int32)
        if all(cv2.pointPolygonTest(poly, corner, False) >= 0 for corner in corners):
            return True
    return False


def _find_equispaced_runs(
    coords: list[float],
) -> list[tuple[int, int]]:
    """Return (start_idx, end_idx) slices of coords with ≥4 equi-spaced values.

    A run is valid when:
      - Every consecutive spacing is within [_STAIRS_MIN_SPACING_PX,
        _STAIRS_MAX_SPACING_PX].
      - std(spacings) / mean(spacings) ≤ _STAIRS_SPACING_MAX_REL_STD.

    Args:
        coords: Sorted 1-D array of perpendicular coordinates (one per tread).

    Returns:
        List of (start, end) index pairs (inclusive) for valid runs.
    """
    n = len(coords)
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n - 1:
        run_end = i
        for j in range(i + 1, n):
            gap = coords[j] - coords[run_end]
            if _STAIRS_MIN_SPACING_PX <= gap <= _STAIRS_MAX_SPACING_PX:
                run_end = j
            else:
                break
        run_len = run_end - i + 1
        if run_len >= _STAIRS_MIN_LINES:
            spacings = [coords[k + 1] - coords[k] for k in range(i, run_end)]
            mean_sp = float(np.mean(spacings))
            std_sp = float(np.std(spacings))
            rel_std = std_sp / mean_sp if mean_sp > 0 else 1.0
            if rel_std <= _STAIRS_SPACING_MAX_REL_STD:
                runs.append((i, run_end))
                i = run_end + 1
                continue
        i += 1
    return runs


_PerpFn = type(lambda x1, y1, x2, y2: 0.0)
_Seg4 = tuple[float, float, float, float]


def _merge_tread_slots(
    tread_map: dict[int, list[_Seg4]],
    perp_fn: _PerpFn,
) -> list[tuple[float, list[_Seg4]]]:
    """Merge adjacent tread bins into one representative slot per tread.

    Args:
        tread_map: Bin-index → segment list.
        perp_fn: Returns perpendicular coordinate for a segment.

    Returns:
        Sorted ``(representative_coord, segments)`` list, one entry per tread.
    """
    _merge_tol = _STAIRS_COLLINEAR_BIN_PX * 2
    merged: list[tuple[float, list[_Seg4]]] = []
    for _b, segs in sorted(tread_map.items()):
        mid_coord = float(np.median([perp_fn(*s) for s in segs]))
        if merged and abs(mid_coord - merged[-1][0]) <= _merge_tol:
            combined = merged[-1][1] + segs
            merged[-1] = (
                float(np.median([perp_fn(*s) for s in combined])),
                combined,
            )
        else:
            merged.append((mid_coord, list(segs)))
    return merged


def _stairs_runs_to_candidates(
    tread_coords: list[float],
    tread_dict: dict[float, list[_Seg4]],
    rooms: list[Room],
) -> list[StairsCandidate]:
    """Convert equi-spaced tread runs into StairsCandidate objects.

    Args:
        tread_coords: Sorted perpendicular coordinates (one per tread).
        tread_dict: Maps representative coord → segment list.
        rooms: Room polygons for containment check.

    Returns:
        List of validated StairsCandidate objects.
    """
    candidates: list[StairsCandidate] = []
    for run_start, run_end in _find_equispaced_runs(tread_coords):
        all_segs: list[_Seg4] = []
        for coord in tread_coords[run_start : run_end + 1]:
            nearest = min(tread_dict.keys(), key=lambda k: abs(k - coord))
            all_segs.extend(tread_dict[nearest])
        xs = [v for x1, _y1, x2, _y2 in all_segs for v in (x1, x2)]
        ys = [v for _x1, y1, _x2, y2 in all_segs for v in (y1, y2)]
        bx, by = float(min(xs)), float(min(ys))
        bw, bh = float(max(xs)) - bx, float(max(ys)) - by
        if _bbox_inside_any_room(bx, by, bw, bh, rooms):
            candidates.append(
                StairsCandidate(
                    bbox=[bx, by, bw, bh],
                    direction=StairsDirection.unknown,
                    confidence=_STAIRS_CONFIDENCE,
                )
            )
    return candidates


def _candidates_from_orientation(
    lines: list[_Seg4],
    perp_fn: _PerpFn,
    rooms: list[Room],
) -> list[StairsCandidate]:
    """Scan one orientation's tread lines for equi-spaced staircase runs.

    Args:
        lines: Hough segments sharing the same orientation (H or V).
        perp_fn: Perpendicular-coordinate function for this orientation.
        rooms: Room polygons for containment validation.

    Returns:
        StairsCandidate objects found in this orientation.
    """
    if len(lines) < _STAIRS_MIN_LINES:
        return []
    tread_map: dict[int, list[_Seg4]] = defaultdict(list)
    for seg in lines:
        tread_map[int(perp_fn(*seg) / _STAIRS_COLLINEAR_BIN_PX)].append(seg)
    merged = _merge_tread_slots(tread_map, perp_fn)
    if len(merged) < _STAIRS_MIN_LINES:
        return []
    tread_coords = sorted(c for c, _ in merged)
    tread_dict = {round(c, 1): segs for c, segs in merged}
    return _stairs_runs_to_candidates(tread_coords, tread_dict, rooms)


def _detect_stairs_candidates(
    pre_filter_mask: NDArray[np.uint8],
    rooms: list[Room],
    settings: Settings | None,
) -> list[StairsCandidate]:
    """Detect staircase candidates from the pre-thin-filter binary mask.

    Runs HoughLinesP with lower thresholds than wall detection to find thin
    tread lines preserved in pre_filter_mask.  Groups lines by orientation
    (H / V) and perpendicular position, then tests for ≥4 equi-spaced treads
    with spacing in [_STAIRS_MIN_SPACING_PX, _STAIRS_MAX_SPACING_PX] px.
    Anti-FP: the bbox of each candidate must be fully contained in a room
    polygon (cv2.pointPolygonTest).

    Must be called BEFORE filter_thin_strokes is applied to the mask
    (i.e. using the pre_filter_mask intermediate, not the final wall_mask).

    Args:
        pre_filter_mask: Binary mask after cleanup steps 1-3 (thin lines kept).
        rooms: Room polygons used for containment check.
        settings: Runtime settings — checked for cv_stairs_detection_enabled.

    Returns:
        List of StairsCandidate objects (may be empty).
    """
    if settings is not None and not settings.cv_stairs_detection_enabled:
        return []
    if len(rooms) == 0:
        return []
    segments = cv2.HoughLinesP(
        pre_filter_mask,
        rho=_HOUGH_RHO,
        theta=_HOUGH_THETA,
        threshold=_STAIRS_HOUGH_THRESHOLD,
        minLineLength=float(_STAIRS_HOUGH_MIN_LINE_PX),
        maxLineGap=float(_STAIRS_HOUGH_MAX_GAP_PX),
    )
    if segments is None:
        return []

    h_lines: list[_Seg4] = []
    v_lines: list[_Seg4] = []
    for seg in segments:
        x1, y1, x2, y2 = seg.reshape(4).tolist()
        angle_deg = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        if angle_deg > 90:  # noqa: PLR2004
            angle_deg = 180.0 - angle_deg
        if angle_deg <= _OPENING_ANGLE_TOL_DEG:
            h_lines.append((x1, y1, x2, y2))
        elif angle_deg >= 90.0 - _OPENING_ANGLE_TOL_DEG:
            v_lines.append((x1, y1, x2, y2))

    def _h_perp(x1: float, y1: float, x2: float, y2: float) -> float:
        return (y1 + y2) / 2.0

    def _v_perp(x1: float, y1: float, x2: float, y2: float) -> float:
        return (x1 + x2) / 2.0

    return _candidates_from_orientation(
        h_lines, _h_perp, rooms
    ) + _candidates_from_orientation(v_lines, _v_perp, rooms)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class OpenCVClassicEngine(GeometryEngine):
    """Phase 1 engine: classical OpenCV — no ML, CPU-only (ADR-008).

    Latency target: p95 < 20 s per image.

    After each extract() call the following intermediates are available on
    the instance for reuse by downstream tasks:

      _wall_mask : NDArray[np.uint8] | None
          Binary mask (walls = 255) produced by the binarisation pipeline.

      _gray : NDArray[np.uint8] | None
          Grayscale version of the last processed image (pre-blur, original).

      _junctions : list[Point] | None
          Shared corner coordinates produced by the endpoint fusion step
          (07-cv-04).  Each entry is a ``(x, y)`` pixel coordinate where two
          or more wall endpoints were merged.  Available for cv-07
          (door-at-corner detection).

      _pre_filter_mask : NDArray[np.uint8] | None
          Binary wall mask BEFORE ``filter_thin_strokes`` (steps 1-3 of
          ``clean_mask``).  Thin double-line window strokes are preserved here.
          Used by ``_detect_window_pattern`` (07-cv-06).

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
        # Junction points produced by 07-cv-04 fusion step.  Available after
        # each extract() call for downstream tasks (cv-07 door-at-corner).
        self._junctions: list[Point] | None = None
        # Pre-thin-filter mask for window pattern detection (07-cv-06).
        # Holds the mask after cleanup steps 1-3 but before step 4 so that
        # thin double-line window strokes are still visible.
        self._pre_filter_mask: NDArray[np.uint8] | None = None

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
        # Also compute pre_filter_mask (steps 1-3 only) so that thin window
        # frame lines are preserved for _detect_window_pattern (07-cv-06).
        if self._settings is not None:
            self._pre_filter_mask = clean_mask_steps_1_to_3(wall_mask, self._settings)
            wall_mask = clean_mask(wall_mask, self._settings)
        else:
            self._pre_filter_mask = wall_mask

        # ---- 5. Store intermediates (for 06-cv-04) ---------------------
        self._wall_mask = wall_mask
        self._gray = gray  # pre-blur version for feature detection

        # ---- 6. Detect walls and consolidate (F3) ----------------------
        raw_walls = _detect_walls(wall_mask)
        walls = _consolidate_walls(raw_walls, wall_mask, self._settings)
        t_walls = time.monotonic()

        # ---- 6b. Snap near-orthogonal walls and fuse junctions (07-cv-04) --
        # Snapping first so that endpoints corrected to exact H/V are already
        # axis-aligned when the junction fusion distance check runs.
        # _extend_to_intersection (08-cv-xx) runs after snapping so walls are
        # exact H/V, and before fusion so coincident endpoints produce junctions.
        walls = _snap_walls_orthogonal(walls)
        walls = _extend_to_intersection(walls, self._settings.cv_junction_extend_px)
        walls, self._junctions = _fuse_junctions(walls)
        # ---- 6c. Diagonal residual filter, pass 2 (ADR-017) ------------
        # Re-filters by angle (Mec.1) and by minimum length for surviving
        # oblique walls (Mec.2). Gated by the same master switch as pass 1
        # in _consolidate_walls; no-op with the flag disabled (AC-11).
        walls = _filter_diagonal_residual_pass2(walls, self._settings)

        # ---- 7. Detect rooms with gap-closed mask (F1) -----------------
        # A separate closed mask bridges architectural openings so that CCA
        # sees fully enclosed regions.  Walls and openings detection still
        # use the original (unclosed) wall_mask.
        # Asymmetric H/V gaps: H is kept smaller than V to avoid filling
        # narrow rooms (bathrooms, corridors) bounded by two close vertical
        # walls, while V bridges wide door/passage openings in horizontal walls.
        _h_gap = (
            self._settings.cv_room_close_h_gap_px
            if self._settings is not None
            else _ROOM_CLOSE_GAP_PX
        )
        _v_gap = (
            self._settings.cv_room_close_v_gap_px
            if self._settings is not None
            else _ROOM_CLOSE_GAP_PX
        )
        # Scale gaps with upscale factor so door openings are bridged correctly.
        # upscale_factor is already computed above from normalize_resolution().
        if (
            self._settings is not None
            and self._settings.cv_room_close_scale_with_upscale
            and upscale_factor > 1.0
        ):
            _h_gap = round(_h_gap * upscale_factor)
            _v_gap = round(_v_gap * upscale_factor)
        closed_wall_mask = _build_closed_wall_mask_for_rooms(
            wall_mask,
            close_h_gap_px=_h_gap,
            close_v_gap_px=_v_gap,
        )
        rooms = _detect_rooms(closed_wall_mask, img_h, img_w, self._settings)
        t_rooms = time.monotonic()

        # ---- 7b. Filter interior components (furniture / fixtures) ------
        # Runs AFTER room detection because it needs the room polygons to test
        # whether a wall-mask component is entirely enclosed by a room.
        # Components whose bbox corners all lie strictly inside a room polygon
        # (by more than CV_CLEANUP_INTERIOR_COMPONENTS_MARGIN_PX) are rectilinear
        # furniture artefacts (tables, sofas) that survived steps 1-4 and are
        # now removed.  If any components are removed the wall-mask changes, so
        # walls are re-detected and consolidated to stay consistent.
        if (
            self._settings is not None
            and self._settings.cv_cleanup_interior_components_enabled
        ):
            _margin = self._settings.cv_cleanup_interior_components_margin_px
            wall_mask, _n_interior = filter_interior_components(
                wall_mask, rooms, float(_margin)
            )
            if _n_interior > 0:
                # Walls detected before this step may include furniture segments;
                # re-detect from the updated mask to keep walls and openings consistent.
                self._wall_mask = wall_mask
                raw_walls = _detect_walls(wall_mask)
                walls = _consolidate_walls(raw_walls, wall_mask, self._settings)

        # ---- 8. Detect opening candidates, then NMS dedup (F2) ---------
        # Detect door-swing arcs for confidence boosting (07-cv-07 criterion b).
        _arc_centers = _detect_door_arcs(self._gray) if self._gray is not None else []
        # Gap-based detection (doors and gap-evident windows).
        _gap_openings = _detect_openings(
            walls,
            junctions=self._junctions,
            arc_centers=_arc_centers,
            settings=self._settings,
        )
        # Pattern-based window detection (07-cv-06): double-line strokes within
        # the wall span that filter_thin_strokes destroys from the main mask.
        # Operates on the pre-filter mask where thin frame lines are preserved.
        _window_openings = (
            _detect_window_pattern(walls, self._pre_filter_mask)
            if self._pre_filter_mask is not None
            else []
        )
        openings = _nms_openings(_gap_openings + _window_openings)
        t_openings = time.monotonic()
        # Log openings emitted by type for observability (07-cv-07 criterion e).
        _engine_logger.info(
            "openings_emitted",
            extra={
                "door": sum(
                    1 for o in openings if o.type_candidate == OpeningTypeCandidate.door
                ),
                "window": sum(
                    1
                    for o in openings
                    if o.type_candidate == OpeningTypeCandidate.window
                ),
                "unknown": sum(
                    1
                    for o in openings
                    if o.type_candidate == OpeningTypeCandidate.unknown
                ),
            },
        )

        # ---- 9. Detect staircase candidates (07-cv-10) -----------------
        # Must use pre_filter_mask (thin tread lines still present) and the
        # room list (for containment anti-FP check).  Called after rooms so
        # that the polygon list is ready.
        stairs_candidates = _detect_stairs_candidates(
            self._pre_filter_mask,
            rooms,
            self._settings,
        )
        _engine_logger.info(
            "stairs_candidates_count",
            extra={"stairs_candidates_count": len(stairs_candidates)},
        )

        # ---- 10. Derive scale (06-cv-04) --------------------------------
        scale = _detect_scale(self._gray, self._settings)
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
                "stairs_candidates_count": len(stairs_candidates),
            },
        )

        # ---- 11. Assemble response -------------------------------------
        return Geometry(
            walls=walls,
            rooms=rooms,
            openings=openings,
            stairs_candidates=stairs_candidates,
            scale=scale,
            image_size=ImageSize(width=img_w, height=img_h),
        )
