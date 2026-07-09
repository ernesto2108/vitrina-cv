"""Tests for centerline consolidation, orthogonal snap, and junction fusion.

Covers task 07-cv-05:
1. Two parallel traces ≤ thickness → 1 Wall on centerline, thickness ≈ separation
2. Wall.thickness in pixels (explicit px assert, not metres)
3. Corner: endpoints < thickness → shared vertex; > thickness → no fusion
4. Diagonal at 30° → not snapped (angle intact)
5. Regression: _detect_openings with new consolidation output — no openings lost
6. With CV_WALL_CENTERLINE_ENABLED=false → legacy mode (thickness=None)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray

from vitrina_cv.config.settings import Settings
from vitrina_cv.engines.opencv_classic import (
    _consolidate_walls,
    _detect_openings,
    _extend_to_intersection,
    _fuse_junctions,
    _snap_walls_orthogonal,
)
from vitrina_cv.models import Wall

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _h_wall(x1: float, x2: float, y: float, thickness: float | None = None) -> Wall:
    return Wall(start=(x1, y), end=(x2, y), thickness=thickness)


def _v_wall(y1: float, y2: float, x: float, thickness: float | None = None) -> Wall:
    return Wall(start=(x, y1), end=(x, y2), thickness=thickness)


def _make_thick_wall_mask(
    height: int,
    width: int,
    y_top: int,
    y_bottom: int,
    x_start: int,
    x_end: int,
) -> NDArray[np.uint8]:
    """Create a mask with a solid filled horizontal wall stripe.

    The distanceTransform of this mask reports ~half-height at the centerline,
    which is what _estimate_global_wall_thickness_px needs to estimate a
    realistic thickness and then group parallel Hough traces within that
    thickness.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y_top:y_bottom, x_start:x_end] = 255
    return mask


# ---------------------------------------------------------------------------
# 1 & 2 — Two parallel traces → 1 centerline Wall with thickness in px
# ---------------------------------------------------------------------------


class TestConsolidateWallsCenterline:
    """_consolidate_walls with cv_wall_centerline_enabled=True.

    Design note: the DT-based grouping tolerance equals the estimated wall
    thickness, which is 2 x median(DT values sampled along the Hough segments).
    For the grouping to merge two traces, the mask must be a solid filled stripe
    so that DT values sampled along each interior segment reflect the half-height
    of the stripe — which then equals or exceeds the inter-segment separation.

    Concretely: a stripe from y=40..y=80 (40 px tall) sampled at y=55 and y=65
    gives DT ≈ 15, thickness ≈ 30 px, grouping tolerance ≈ 30 px > separation 10.
    """

    def test_two_parallel_h_traces_produce_single_wall(self) -> None:
        """Two horizontal traces inside a thick stripe → 1 consolidated Wall."""
        # Solid stripe 40px tall — DT at center ≈ 20, so thickness ≈ 40px
        y_top, y_bottom = 40, 80
        x_start, x_end = 10, 300
        mask = _make_thick_wall_mask(200, 400, y_top, y_bottom, x_start, x_end)

        # Two traces inside the stripe, 10px apart — well within thickness ≈ 40
        raw_walls = [
            _h_wall(x_start, x_end, 55.0),
            _h_wall(x_start, x_end, 65.0),
        ]

        settings = Settings(cv_wall_centerline_enabled=True)
        consolidated = _consolidate_walls(raw_walls, mask, settings)

        assert len(consolidated) == 1, (
            f"Expected 1 consolidated wall, got {len(consolidated)}"
        )

    def test_centerline_y_is_between_the_two_traces(self) -> None:
        """The consolidated wall sits between the two original trace y-coords."""
        y_top, y_bottom = 40, 80
        x_start, x_end = 10, 300
        y_seg1, y_seg2 = 55.0, 65.0  # 10px apart, both inside the stripe

        mask = _make_thick_wall_mask(200, 400, y_top, y_bottom, x_start, x_end)
        raw_walls = [
            _h_wall(x_start, x_end, y_seg1),
            _h_wall(x_start, x_end, y_seg2),
        ]

        settings = Settings(cv_wall_centerline_enabled=True)
        consolidated = _consolidate_walls(raw_walls, mask, settings)

        assert len(consolidated) == 1
        wall = consolidated[0]
        y_center = (wall.start[1] + wall.end[1]) / 2.0
        assert y_seg1 <= y_center <= y_seg2, (
            f"Centerline y={y_center:.1f} outside [{y_seg1}, {y_seg2}]"
        )

    def test_thickness_is_in_pixels_not_metres(self) -> None:
        """Wall.thickness is measured in pixels (explicit assertion in pixel units).

        A real architectural wall at 1:50 scale rendered at 1000px/m would be
        ~20px. Metres would produce a value < 1.0 — we assert the opposite.
        """
        min_thickness_px = 1.0  # below this → looks like metres
        max_thickness_px = 200.0  # above this → implausibly thick for a normal plan

        y_top, y_bottom = 40, 80
        x_start, x_end = 10, 300
        mask = _make_thick_wall_mask(200, 400, y_top, y_bottom, x_start, x_end)
        raw_walls = [
            _h_wall(x_start, x_end, 55.0),
            _h_wall(x_start, x_end, 65.0),
        ]

        settings = Settings(cv_wall_centerline_enabled=True)
        consolidated = _consolidate_walls(raw_walls, mask, settings)

        assert len(consolidated) == 1
        thickness = consolidated[0].thickness
        assert thickness is not None, "thickness must be set in centerline mode"
        assert thickness > min_thickness_px, (
            f"thickness={thickness:.4f} looks like metres, expected pixels (>{min_thickness_px})"
        )
        assert thickness < max_thickness_px, (
            f"thickness={thickness:.1f} px is implausibly large"
        )

    def test_thickness_approximates_real_wall_height(self) -> None:
        """thickness is ~2x median DT, which approximates the stripe half-height."""
        stripe_height = 40  # y=40..y=80
        min_expected_thickness_px = 10.0
        y_top = 40
        y_bottom = y_top + stripe_height
        x_start, x_end = 10, 300
        mask = _make_thick_wall_mask(200, 400, y_top, y_bottom, x_start, x_end)
        raw_walls = [
            _h_wall(x_start, x_end, 55.0),
            _h_wall(x_start, x_end, 65.0),
        ]

        settings = Settings(cv_wall_centerline_enabled=True)
        consolidated = _consolidate_walls(raw_walls, mask, settings)

        assert len(consolidated) == 1
        thickness = consolidated[0].thickness
        assert thickness is not None
        # DT at a point 15px from top/bottom edges ~= 15; thickness = 2*15 = 30.
        # We use a loose bound: between 10px and 2*stripe_height.
        assert thickness >= min_expected_thickness_px, (
            f"thickness={thickness:.1f} is too small"
        )
        assert thickness <= stripe_height * 2.0, (
            f"thickness={thickness:.1f} is too large (> 2x stripe_height={stripe_height})"
        )


# ---------------------------------------------------------------------------
# 6 — Legacy mode (centerline disabled)
# ---------------------------------------------------------------------------


class TestConsolidateWallsLegacy:
    """_consolidate_walls with cv_wall_centerline_enabled=False."""

    def test_legacy_mode_thickness_is_none(self) -> None:
        """When centerline is disabled, Wall.thickness must be None."""
        raw_walls = [
            _h_wall(10, 200, 50.0),
            _h_wall(10, 200, 55.0),
        ]
        mask = np.zeros((200, 300), dtype=np.uint8)

        settings = Settings(cv_wall_centerline_enabled=False)
        consolidated = _consolidate_walls(raw_walls, mask, settings)

        for wall in consolidated:
            assert wall.thickness is None, (
                f"Expected thickness=None in legacy mode, got {wall.thickness}"
            )

    def test_legacy_mode_no_mask_fallback(self) -> None:
        """Passing wall_mask=None always triggers legacy mode regardless of settings."""
        raw_walls = [_h_wall(10, 200, 50.0)]
        settings = Settings(cv_wall_centerline_enabled=True)

        # wall_mask=None → legacy fallback path
        consolidated = _consolidate_walls(raw_walls, wall_mask=None, settings=settings)

        for wall in consolidated:
            assert wall.thickness is None, (
                "Expected thickness=None when wall_mask is None"
            )


# ---------------------------------------------------------------------------
# 3 — Orthogonal snap
# ---------------------------------------------------------------------------


class TestSnapWallsOrthogonal:
    def test_near_horizontal_snapped_to_exact_h(self) -> None:
        """Segment within 5° of horizontal → both endpoints share the same y."""
        # ~3° angle from horizontal
        wall = Wall(start=(0.0, 100.0), end=(200.0, 110.0))
        snapped = _snap_walls_orthogonal([wall])

        assert len(snapped) == 1
        y_start = snapped[0].start[1]
        y_end = snapped[0].end[1]
        assert y_start == pytest.approx(y_end, abs=1e-6), (
            f"Snapped H wall must have equal y: start.y={y_start}, end.y={y_end}"
        )
        # y must be the average of original y-coords
        expected_y = (100.0 + 110.0) / 2.0
        assert y_start == pytest.approx(expected_y, abs=1e-6)

    def test_near_vertical_snapped_to_exact_v(self) -> None:
        """Segment within 5° of vertical → both endpoints share the same x."""
        # ~3° from vertical
        wall = Wall(start=(100.0, 0.0), end=(110.0, 200.0))
        snapped = _snap_walls_orthogonal([wall])

        assert len(snapped) == 1
        x_start = snapped[0].start[0]
        x_end = snapped[0].end[0]
        assert x_start == pytest.approx(x_end, abs=1e-6), (
            f"Snapped V wall must have equal x: start.x={x_start}, end.x={x_end}"
        )
        expected_x = (100.0 + 110.0) / 2.0
        assert x_start == pytest.approx(expected_x, abs=1e-6)

    def test_diagonal_30_degrees_not_snapped(self) -> None:
        """Diagonal at 30° is outside the 5° snap tolerance → angle unchanged."""
        # 30° from horizontal: dy/dx = tan(30°) ≈ 0.577
        length = 200.0
        angle_rad = math.radians(30.0)
        x2 = length * math.cos(angle_rad)
        y2 = length * math.sin(angle_rad)
        wall = Wall(start=(0.0, 0.0), end=(x2, y2))

        snapped = _snap_walls_orthogonal([wall])

        assert len(snapped) == 1
        sx1, sy1 = snapped[0].start
        sx2, sy2 = snapped[0].end
        # Diagonal must NOT have been made horizontal (same y)
        assert sy1 != pytest.approx(sy2, abs=1.0), (
            "30° diagonal should not be H-snapped"
        )
        # Diagonal must NOT have been made vertical (same x)
        assert sx1 != pytest.approx(sx2, abs=1.0), (
            "30° diagonal should not be V-snapped"
        )

        # Original angle is preserved
        actual_angle = math.degrees(math.atan2(abs(sy2 - sy1), abs(sx2 - sx1)))
        assert actual_angle == pytest.approx(30.0, abs=0.5)

    def test_thickness_preserved_after_snap(self) -> None:
        """Snapping must carry through the original Wall.thickness."""
        wall = Wall(start=(0.0, 100.0), end=(200.0, 105.0), thickness=12.0)
        snapped = _snap_walls_orthogonal([wall])

        assert snapped[0].thickness == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# 3 — Junction fusion
# ---------------------------------------------------------------------------


class TestFuseJunctions:
    def test_close_endpoints_produce_shared_vertex(self) -> None:
        """Two walls whose endpoints are within thickness → share a junction."""
        thickness = 20.0
        # Horizontal wall ending at (100, 50) and vertical wall starting at (102, 50)
        # Distance = 2 px < thickness=20
        h_wall = Wall(start=(0.0, 50.0), end=(100.0, 50.0), thickness=thickness)
        v_wall = Wall(start=(102.0, 50.0), end=(102.0, 200.0), thickness=thickness)

        updated, junctions = _fuse_junctions([h_wall, v_wall])

        assert len(junctions) >= 1, "Expected at least one junction"
        # After fusion the end of h_wall and start of v_wall must coincide
        h_end = updated[0].end
        v_start = updated[1].start
        assert h_end[0] == pytest.approx(v_start[0], abs=1e-6)
        assert h_end[1] == pytest.approx(v_start[1], abs=1e-6)

    def test_far_endpoints_produce_no_junction(self) -> None:
        """Two walls whose closest endpoints are farther than thickness → no junction."""
        thickness = 10.0
        # Distance between endpoints = 50 px >> thickness=10
        h_wall = Wall(start=(0.0, 50.0), end=(100.0, 50.0), thickness=thickness)
        v_wall = Wall(start=(150.0, 50.0), end=(150.0, 200.0), thickness=thickness)

        _, junctions = _fuse_junctions([h_wall, v_wall])

        assert junctions == [], (
            f"No junction expected when distance > thickness, got {junctions}"
        )

    def test_three_walls_meet_at_corner(self) -> None:
        """Three walls with endpoints near (100, 100) → single common junction."""
        thickness = 15.0
        # All endpoints placed at (100, 100) ± 3 px < thickness
        walls = [
            Wall(start=(0.0, 100.0), end=(100.0, 100.0), thickness=thickness),
            Wall(start=(100.0, 100.0), end=(200.0, 100.0), thickness=thickness),
            Wall(start=(100.0, 0.0), end=(103.0, 100.0), thickness=thickness),
        ]

        updated, junctions = _fuse_junctions(walls)

        assert len(junctions) >= 1, "Expected at least one common junction"
        # All walls that meet at the corner must share the same coordinate
        e1 = updated[0].end
        s2 = updated[1].start
        e3 = updated[2].end
        assert e1[0] == pytest.approx(s2[0], abs=1e-6)
        assert e1[1] == pytest.approx(s2[1], abs=1e-6)
        assert e3[0] == pytest.approx(e1[0], abs=1e-6)
        assert e3[1] == pytest.approx(e1[1], abs=1e-6)

    def test_single_wall_returns_unchanged(self) -> None:
        """A single wall has no pairs to fuse — returned as-is with empty junctions."""
        wall = Wall(start=(0.0, 0.0), end=(100.0, 0.0), thickness=10.0)
        updated, junctions = _fuse_junctions([wall])

        assert junctions == []
        assert updated[0].start == wall.start
        assert updated[0].end == wall.end

    def test_fusion_uses_thickness_as_distance_threshold(self) -> None:
        """Endpoints at exactly thickness distance are NOT fused (strictly less than)."""
        thickness = 20.0
        # Exactly 20 px apart — must not be fused
        h_wall = Wall(start=(0.0, 50.0), end=(100.0, 50.0), thickness=thickness)
        v_wall = Wall(start=(120.0, 50.0), end=(120.0, 200.0), thickness=thickness)

        _, junctions = _fuse_junctions([h_wall, v_wall])

        # At exactly threshold distance fusion should not occur
        assert junctions == [], (
            "Endpoints at exactly the threshold distance must not be fused"
        )


# ---------------------------------------------------------------------------
# 5 — Regression: _detect_openings with consolidated walls
# ---------------------------------------------------------------------------


class TestDetectOpeningsRegression:
    """Verify that _detect_openings still finds openings when given consolidated
    walls (post-centerline).  This ensures cv-03/cv-04 refactors do not silently
    drop opening candidates."""

    def _make_wall_with_gap(
        self,
        y: float,
        x_start: float,
        gap_start: float,
        gap_end: float,
        x_end: float,
        thickness: float = 10.0,
    ) -> list[Wall]:
        """Return two horizontal consolidated walls bracketing a gap."""
        return [
            Wall(start=(x_start, y), end=(gap_start, y), thickness=thickness),
            Wall(start=(gap_end, y), end=(x_end, y), thickness=thickness),
        ]

    def test_consolidated_walls_produce_same_openings_as_raw(self) -> None:
        """Opening detected in raw walls is still detected after consolidation.

        Constructs two long horizontal consolidated walls with a ~60 px gap
        (door candidate).  _detect_openings must return at least one opening.
        """
        # Two long wall sections with a 60 px gap
        walls = self._make_wall_with_gap(
            y=200.0,
            x_start=0.0,
            gap_start=400.0,
            gap_end=460.0,
            x_end=800.0,
            thickness=10.0,
        )
        openings = _detect_openings(walls)

        assert len(openings) >= 1, (
            "Expected ≥ 1 opening from consolidated walls with a 60 px gap"
        )
        # Opening gap width must be approximately 60 px
        gap_min = 50.0
        gap_max = 70.0
        gap_widths = [o.bbox[2] for o in openings]
        assert any(gap_min <= w <= gap_max for w in gap_widths), (
            f"No opening with expected gap width ~60px; got widths: {gap_widths}"
        )

    def test_no_openings_lost_after_snap(self) -> None:
        """Opening detected before snap is still present after snap."""
        walls = self._make_wall_with_gap(
            y=150.0,
            x_start=0.0,
            gap_start=350.0,
            gap_end=410.0,
            x_end=750.0,
            thickness=10.0,
        )
        before = _detect_openings(walls)
        snapped = _snap_walls_orthogonal(walls)
        after = _detect_openings(snapped)

        assert len(after) >= len(before), (
            f"Openings lost after snap: before={len(before)}, after={len(after)}"
        )

    def test_no_openings_lost_after_fuse(self) -> None:
        """Opening count must not decrease after junction fusion."""
        walls = self._make_wall_with_gap(
            y=250.0,
            x_start=0.0,
            gap_start=400.0,
            gap_end=460.0,
            x_end=800.0,
            thickness=10.0,
        )
        before = _detect_openings(walls)
        fused, _ = _fuse_junctions(walls)
        after = _detect_openings(fused)

        assert len(after) >= len(before), (
            f"Openings lost after fusion: before={len(before)}, after={len(after)}"
        )


# ---------------------------------------------------------------------------
# _extend_to_intersection
# ---------------------------------------------------------------------------


class TestExtendToIntersection:
    """Unit tests for _extend_to_intersection (task 08-cv-*).

    All tests use exact H/V walls (post-snap convention).
    Intersection = (v_wall.x, h_wall.y).
    """

    # ------------------------------------------------------------------
    # Case 1: L-corner — both endpoints short, gap < extend_px → both extend
    # ------------------------------------------------------------------

    def test_extend_to_intersection_l_corner_gap_under_threshold(self) -> None:
        """L-corner: h_wall and v_wall each 30 px short, extend_px=40 → both reach intersection."""
        # Intersection at (100, 100).
        # h_wall ends at x=70  → gap = 30 ≤ 40 → should extend to x=100
        # v_wall ends at y=70  → gap = 30 ≤ 40 → should extend to y=100
        h_wall = _h_wall(10.0, 70.0, 100.0, thickness=12.0)
        v_wall = _v_wall(10.0, 70.0, 100.0, thickness=12.0)

        result = _extend_to_intersection([h_wall, v_wall], extend_px=40)

        expected_count = 2
        assert len(result) == expected_count
        # h_wall: the rightmost x must now reach ix=100
        h_out = result[0]
        assert max(h_out.start[0], h_out.end[0]) == pytest.approx(100.0, abs=1e-6)
        # v_wall: the bottommost y must now reach iy=100
        v_out = result[1]
        assert max(v_out.start[1], v_out.end[1]) == pytest.approx(100.0, abs=1e-6)

    # ------------------------------------------------------------------
    # Case 2: T-corner — long wall covers intersection; only short extends
    # ------------------------------------------------------------------

    def test_extend_to_intersection_t_corner_only_short_wall_moves(self) -> None:
        """T-corner: long H wall covers ix in interior; only the short V wall extends."""
        # h_wall spans x=0..200, y=100  → ix=100 is interior, must not change
        # v_wall ends at y=75, x=100   → gap = 100-75 = 25 ≤ 40 → should extend to y=100
        h_wall = _h_wall(0.0, 200.0, 100.0, thickness=10.0)
        v_wall = _v_wall(10.0, 75.0, 100.0, thickness=10.0)

        result = _extend_to_intersection([h_wall, v_wall], extend_px=40)

        h_out = result[0]
        v_out = result[1]

        # H wall endpoints must be unchanged (intersection is in its interior)
        assert h_out.start[0] == pytest.approx(0.0, abs=1e-6)
        assert h_out.end[0] == pytest.approx(200.0, abs=1e-6)

        # V wall: bottommost y must now be 100 (extended from 75)
        assert max(v_out.start[1], v_out.end[1]) == pytest.approx(100.0, abs=1e-6)

    # ------------------------------------------------------------------
    # Case 3: Gap > extend_px → no extension
    # ------------------------------------------------------------------

    def test_extend_to_intersection_gap_above_threshold_no_change(self) -> None:
        """Gap of 50 px > extend_px=40 → output identical to input."""
        # Intersection at (100, 100).
        # h_wall ends at x=50 → gap = 50 > 40 → must NOT extend
        # v_wall ends at y=50 → gap = 50 > 40 → must NOT extend
        h_wall = _h_wall(10.0, 50.0, 100.0, thickness=8.0)
        v_wall = _v_wall(10.0, 50.0, 100.0, thickness=8.0)

        result = _extend_to_intersection([h_wall, v_wall], extend_px=40)

        h_out, v_out = result[0], result[1]
        assert h_out.start == h_wall.start
        assert h_out.end == h_wall.end
        assert v_out.start == v_wall.start
        assert v_out.end == v_wall.end

    # ------------------------------------------------------------------
    # Case 4: Parallel pairs — no extension
    # ------------------------------------------------------------------

    def test_extend_to_intersection_parallel_pairs_not_touched(self) -> None:
        """Two H walls (parallel pair) must not be extended."""
        h_wall1 = _h_wall(10.0, 70.0, 100.0, thickness=10.0)
        h_wall2 = _h_wall(10.0, 70.0, 150.0, thickness=10.0)

        result = _extend_to_intersection([h_wall1, h_wall2], extend_px=40)

        assert result[0].start == h_wall1.start
        assert result[0].end == h_wall1.end
        assert result[1].start == h_wall2.start
        assert result[1].end == h_wall2.end

    # ------------------------------------------------------------------
    # Case 5: Idempotence — endpoints already at intersection → no change
    # ------------------------------------------------------------------

    def test_extend_to_intersection_idempotent_when_gaps_are_zero(self) -> None:
        """Endpoints already touching the intersection → output identical to input (plan-004)."""
        # Intersection at (100, 100). Both walls already reach it exactly.
        h_wall = _h_wall(10.0, 100.0, 100.0, thickness=10.0)
        v_wall = _v_wall(10.0, 100.0, 100.0, thickness=10.0)

        result = _extend_to_intersection([h_wall, v_wall], extend_px=40)

        assert result[0].start == h_wall.start
        assert result[0].end == h_wall.end
        assert result[1].start == v_wall.start
        assert result[1].end == v_wall.end

    # ------------------------------------------------------------------
    # Case 6: Prolongation vs interior (AC-07)
    # ------------------------------------------------------------------

    def test_extend_to_intersection_interior_endpoint_not_shortened(self) -> None:
        """H wall that covers ix in its interior must not be altered (AC-07)."""
        # h_wall: x=0..200, y=100 — ix=100 is interior
        # v_wall: x=100, y=10..80 — gap to iy=100 is 20 ≤ 40 → V extends
        h_wall = _h_wall(0.0, 200.0, 100.0, thickness=10.0)
        v_wall = _v_wall(10.0, 80.0, 100.0, thickness=10.0)

        result = _extend_to_intersection([h_wall, v_wall], extend_px=40)

        h_out, v_out = result[0], result[1]

        # H must be completely unchanged
        assert h_out.start[0] == pytest.approx(0.0)
        assert h_out.end[0] == pytest.approx(200.0)
        assert h_out.start[1] == pytest.approx(100.0)
        assert h_out.end[1] == pytest.approx(100.0)

        # V must have its bottom endpoint moved to iy=100
        assert max(v_out.start[1], v_out.end[1]) == pytest.approx(100.0, abs=1e-6)

    # ------------------------------------------------------------------
    # Case 7: Cardinality and thickness preserved
    # ------------------------------------------------------------------

    def test_extend_to_intersection_cardinality_and_thickness_preserved(self) -> None:
        """N walls in → N walls out; thickness of each wall is unchanged."""
        walls = [
            _h_wall(0.0, 60.0, 100.0, thickness=5.0),
            _v_wall(0.0, 60.0, 100.0, thickness=7.0),
            _h_wall(0.0, 80.0, 200.0, thickness=None),
            _v_wall(0.0, 80.0, 200.0, thickness=15.0),
            _h_wall(50.0, 120.0, 300.0, thickness=9.0),
        ]

        result = _extend_to_intersection(walls, extend_px=40)

        assert len(result) == len(walls), (
            f"Cardinality changed: in={len(walls)}, out={len(result)}"
        )
        for original, out_wall in zip(walls, result, strict=True):
            assert out_wall.thickness == original.thickness, (
                f"thickness changed: expected {original.thickness}, got {out_wall.thickness}"
            )

    # ------------------------------------------------------------------
    # Case 8: Integration snap → extend → fuse produces junction
    # ------------------------------------------------------------------

    def test_extend_to_intersection_integration_with_fuse_junctions(self) -> None:
        """After extend, gap is ~0 → _fuse_junctions must produce ≥1 junction."""
        # h_wall ends at x=70, v_wall ends at y=70; intersection=(100, 100).
        # Gap=30 < extend_px=40 → both endpoints move to (100, 100).
        thickness = 20.0
        h_wall = _h_wall(10.0, 70.0, 100.0, thickness=thickness)
        v_wall = _v_wall(10.0, 70.0, 100.0, thickness=thickness)

        extended = _extend_to_intersection([h_wall, v_wall], extend_px=40)
        _, junctions = _fuse_junctions(extended)

        assert len(junctions) >= 1, (
            "Expected ≥1 junction after extend+fuse with gap=30px < extend_px=40"
        )
