"""Tests for opening detection enhancements — tasks cv-06 and cv-07.

Covers:
1. _detect_window_pattern: wall with 2-3 thin parallel lines → window candidate
2. _detect_window_pattern: solid wall → 0 window candidates (control negativo)
3. _detect_openings: gap near junction with span ~60px → door candidate emitted
4. _detect_openings: same gap far from junction → not emitted (span insufficient)
5. Low-confidence (0.35) candidate present in openings list
6. CV_OPENING_MIN_WALL_SPAN_PX configurable via Settings
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from vitrina_cv.config.settings import Settings
from vitrina_cv.engines.opencv_classic import (
    _detect_openings,
    _detect_window_pattern,
)
from vitrina_cv.models import OpeningTypeCandidate, Wall

_WIN_CONFIDENCE: float = 0.35
_DOOR_ARC_CONFIDENCE: float = 0.7
_FLOAT_TOL: float = 1e-6

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _h_wall(x1: float, x2: float, y: float, thickness: float = 10.0) -> Wall:
    return Wall(start=(x1, y), end=(x2, y), thickness=thickness)


def _v_wall(y1: float, y2: float, x: float, thickness: float = 10.0) -> Wall:
    return Wall(start=(x, y1), end=(x, y2), thickness=thickness)


def _blank_mask(height: int = 400, width: int = 400) -> NDArray[np.uint8]:
    return np.zeros((height, width), dtype=np.uint8)


def _solid_wall_mask(
    h: int,
    w: int,
    y_top: int,
    y_bottom: int,
    x_start: int,
    x_end: int,
) -> NDArray[np.uint8]:
    """Solid filled stripe — no double-line pattern."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y_top:y_bottom, x_start:x_end] = 255
    return mask


def _double_line_mask(
    h: int,
    w: int,
    y_top: int,
    y_bottom: int,
    x_start: int,
    x_end: int,
    line_thickness: int = 2,
) -> NDArray[np.uint8]:
    """Two thin parallel horizontal lines within a wall span (window pattern).

    Structure:
      - top foreground line at y_top..y_top+line_thickness
      - gap (background) in the middle
      - bottom foreground line at y_bottom-line_thickness..y_bottom
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y_top : y_top + line_thickness, x_start:x_end] = 255
    mask[y_bottom - line_thickness : y_bottom, x_start:x_end] = 255
    return mask


# ---------------------------------------------------------------------------
# 1 — Window pattern detection: double-line → window candidate
# ---------------------------------------------------------------------------


class TestDetectWindowPatternDoubleLines:
    """_detect_window_pattern with a mask carrying two thin parallel lines."""

    def test_returns_window_candidate(self) -> None:
        """Horizontal wall with 2 thin parallel strokes → ≥1 window candidate."""
        img_h, img_w = 200, 300
        # Wall: horizontal, 100px wide, centred at y=100, thickness 12px.
        y_center = 100
        thickness = 12
        y_top = y_center - thickness // 2  # 94
        y_bottom = y_center + thickness // 2  # 106
        x_start, x_end = 50, 200

        mask = _double_line_mask(
            img_h, img_w, y_top, y_bottom, x_start, x_end, line_thickness=2
        )
        wall = _h_wall(float(x_start), float(x_end), float(y_center), float(thickness))

        candidates = _detect_window_pattern([wall], mask)

        assert len(candidates) >= 1
        win = candidates[0]
        assert win.type_candidate == OpeningTypeCandidate.window

    def test_bbox_within_wall_span(self) -> None:
        """Returned bbox x-range is within the wall's x-span."""
        img_h, img_w = 200, 300
        y_center = 100
        thickness = 12
        y_top = y_center - thickness // 2
        y_bottom = y_center + thickness // 2
        x_start, x_end = 50, 230

        mask = _double_line_mask(
            img_h, img_w, y_top, y_bottom, x_start, x_end, line_thickness=2
        )
        wall = _h_wall(float(x_start), float(x_end), float(y_center), float(thickness))

        candidates = _detect_window_pattern([wall], mask)

        assert candidates, "expected at least one window candidate"
        bx, _by, bw, _bh = candidates[0].bbox
        assert bx >= x_start - 10, "bbox left edge too far left"
        assert bx + bw <= x_end + 10, "bbox right edge too far right"

    def test_confidence_is_win_confidence(self) -> None:
        """Window candidates carry _WIN_CONFIDENCE == 0.35."""
        img_h, img_w = 200, 300
        y_center = 100
        thickness = 12
        y_top = y_center - thickness // 2
        y_bottom = y_center + thickness // 2
        x_start, x_end = 50, 200

        mask = _double_line_mask(
            img_h, img_w, y_top, y_bottom, x_start, x_end, line_thickness=2
        )
        wall = _h_wall(float(x_start), float(x_end), float(y_center), float(thickness))

        candidates = _detect_window_pattern([wall], mask)

        assert candidates
        assert abs(candidates[0].confidence - _WIN_CONFIDENCE) < _FLOAT_TOL


# ---------------------------------------------------------------------------
# 2 — Window pattern detection: solid wall → 0 candidates
# ---------------------------------------------------------------------------


class TestDetectWindowPatternSolidWall:
    """Control negativo: solid filled wall stripe → no window candidates."""

    def test_no_candidates_from_solid_wall(self) -> None:
        """Solid horizontal stripe produces no double-line pattern → 0 window candidates."""
        img_h, img_w = 200, 300
        y_center = 100
        thickness = 12
        y_top = y_center - thickness // 2
        y_bottom = y_center + thickness // 2
        x_start, x_end = 50, 200

        mask = _solid_wall_mask(img_h, img_w, y_top, y_bottom, x_start, x_end)
        wall = _h_wall(float(x_start), float(x_end), float(y_center), float(thickness))

        candidates = _detect_window_pattern([wall], mask)

        assert candidates == [], f"expected 0 candidates, got {len(candidates)}"

    def test_empty_walls_list(self) -> None:
        """Empty walls list → empty result, no exception."""
        mask = _blank_mask()
        candidates = _detect_window_pattern([], mask)
        assert candidates == []


# ---------------------------------------------------------------------------
# 3 — Gap near junction (relaxed span ~60px) → door candidate emitted
# ---------------------------------------------------------------------------


class TestDetectOpeningsJunctionRelaxedSpan:
    """Gap adjacent to a wall junction uses cv_opening_min_wall_span_px (60px).

    Layout (horizontal):
      [wall_left 65px] [gap 80px] [wall_right 65px]
      junction at x=65 (left wall endpoint)

    With normal threshold 170px → neither span qualifies.
    With junction-relaxed threshold 60px and junction at x=65 → qualifies.
    """

    _SPAN = 65  # shorter than 170px normal threshold, longer than 60px relaxed
    _GAP = 80  # door-sized gap

    def _build_walls(self) -> list[Wall]:
        y = 100.0
        wall_left = _h_wall(0.0, float(self._SPAN), y, 10.0)
        wall_right = _h_wall(
            float(self._SPAN + self._GAP), float(self._SPAN * 2 + self._GAP), y, 10.0
        )
        return [wall_left, wall_right]

    def test_gap_near_junction_emits_door(self) -> None:
        """Gap next to a junction with span 65px → door candidate emitted (relaxed 60px)."""
        walls = self._build_walls()
        # Junction at the left wall's right endpoint (x=65, y=100)
        junctions = [(float(self._SPAN), 100.0)]

        openings = _detect_openings(walls, junctions=junctions)

        assert len(openings) >= 1
        types = [o.type_candidate for o in openings]
        assert OpeningTypeCandidate.door in types

    def test_gap_far_from_junction_not_emitted(self) -> None:
        """Same geometry but junction placed far away → gap NOT emitted (span < 170px)."""
        walls = self._build_walls()
        # Junction far from the gap (y=300 is off the wall y=100)
        junctions = [(float(self._SPAN), 300.0)]

        openings = _detect_openings(walls, junctions=junctions)

        # Span 65px < 170px normal threshold → no opening
        assert len(openings) == 0, (
            f"expected 0 candidates with far junction, got {len(openings)}"
        )

    def test_no_junction_not_emitted(self) -> None:
        """Same geometry with no junctions → gap NOT emitted."""
        walls = self._build_walls()

        openings = _detect_openings(walls, junctions=None)

        assert len(openings) == 0


# ---------------------------------------------------------------------------
# 4 — Low-confidence (0.35) candidate present in openings list
# ---------------------------------------------------------------------------


class TestLowConfidenceCandidatePresent:
    """Gaps that qualify only via the relaxed junction path carry confidence ≤ 0.35."""

    def test_low_confidence_candidate_in_list(self) -> None:
        """Junction-adjacent gap with span < 170px → candidate with confidence ≤ 0.35."""
        y = 50.0
        span = 65
        gap = 80
        wall_left = _h_wall(0.0, float(span), y, 10.0)
        wall_right = _h_wall(float(span + gap), float(span * 2 + gap), y, 10.0)
        walls = [wall_left, wall_right]
        junctions = [(float(span), y)]

        openings = _detect_openings(walls, junctions=junctions)

        assert openings, "expected at least one opening"
        low_conf = [o for o in openings if o.confidence <= _WIN_CONFIDENCE]
        assert low_conf, (
            f"expected at least one candidate with confidence ≤ {_WIN_CONFIDENCE}, "
            f"got confidences: {[o.confidence for o in openings]}"
        )


# ---------------------------------------------------------------------------
# 5 — Arc near gap → confidence boosted to 0.7
# ---------------------------------------------------------------------------


class TestArcBoostsConfidence:
    """Gap with adjacent door-swing arc → confidence overridden to 0.7."""

    def test_arc_boosts_confidence_to_0_7(self) -> None:
        """Arc centre near gap centre → confidence == 0.7 regardless of junction."""
        y = 50.0
        # Normal-span walls (both > 170px) so the gap passes with default confidence.
        wall_left = _h_wall(0.0, 200.0, y, 10.0)
        wall_right = _h_wall(280.0, 500.0, y, 10.0)
        walls = [wall_left, wall_right]

        # Gap: 200..280 (80px), centre at x=240
        arc_centers = [(240.0, y)]

        openings = _detect_openings(walls, arc_centers=arc_centers)

        assert openings, "expected at least one opening"
        boosted = [
            o for o in openings if abs(o.confidence - _DOOR_ARC_CONFIDENCE) < _FLOAT_TOL
        ]
        assert boosted, (
            f"expected confidence {_DOOR_ARC_CONFIDENCE}, got {[o.confidence for o in openings]}"
        )


# ---------------------------------------------------------------------------
# 6 — CV_OPENING_MIN_WALL_SPAN_PX configurable via Settings
# ---------------------------------------------------------------------------


class TestCvOpeningMinWallSpanPxConfigurable:
    """cv_opening_min_wall_span_px drives the relaxed span threshold."""

    def test_custom_span_threshold_40px_allows_shorter_wall(self) -> None:
        """With cv_opening_min_wall_span_px=40, a 45px span near junction qualifies."""
        y = 50.0
        span = 45  # < 60px default but > 40px custom
        gap = 80
        wall_left = _h_wall(0.0, float(span), y, 10.0)
        wall_right = _h_wall(float(span + gap), float(span * 2 + gap), y, 10.0)
        walls = [wall_left, wall_right]
        junctions = [(float(span), y)]

        settings = Settings(cv_opening_min_wall_span_px=40)
        openings = _detect_openings(walls, junctions=junctions, settings=settings)

        assert len(openings) >= 1, (
            "span=45px with relaxed threshold=40px should emit a candidate"
        )

    def test_default_60px_threshold_rejects_50px_span(self) -> None:
        """With default threshold (60px), a 50px span near junction is rejected."""
        y = 50.0
        span = 50  # < 60px default
        gap = 80
        wall_left = _h_wall(0.0, float(span), y, 10.0)
        wall_right = _h_wall(float(span + gap), float(span * 2 + gap), y, 10.0)
        walls = [wall_left, wall_right]
        junctions = [(float(span), y)]

        # Default settings: cv_opening_min_wall_span_px == 60
        settings = Settings()
        openings = _detect_openings(walls, junctions=junctions, settings=settings)

        assert len(openings) == 0, (
            f"span=50px < 60px threshold should not emit a candidate, got {len(openings)}"
        )

    def test_custom_span_threshold_80px_rejects_65px_span(self) -> None:
        """With cv_opening_min_wall_span_px=80, a 65px span near junction is rejected."""
        y = 50.0
        span = 65  # > 60px default but < 80px custom
        gap = 80
        wall_left = _h_wall(0.0, float(span), y, 10.0)
        wall_right = _h_wall(float(span + gap), float(span * 2 + gap), y, 10.0)
        walls = [wall_left, wall_right]
        junctions = [(float(span), y)]

        settings = Settings(cv_opening_min_wall_span_px=80)
        openings = _detect_openings(walls, junctions=junctions, settings=settings)

        assert len(openings) == 0, (
            f"span=65px < custom 80px threshold should not emit a candidate, got {len(openings)}"
        )
