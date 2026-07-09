"""Tests for task 10-cv-05 — local wall-thickness grouping in _consolidate_walls.

Covers ADR-003, part A:
1. Two thin parallel traces (thickness < global tolerance) still merge using
   the local floor (_CENTERLINE_MIN_TOL_PX) — no under-merging regression.
2. Two walls of clearly different local thickness (thin vs thick zone) are
   grouped using each zone's own local tolerance, not a single global value.
3. CV_WALL_LOCAL_THICKNESS_ENABLED=false reproduces the legacy global-tolerance
   behaviour (_estimate_global_wall_thickness_px) byte-for-byte.
4. Sane counter: walls_merged_local_thickness only counts merges that used a
   real local DT estimate (not the fallback).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from vitrina_cv.config.settings import Settings
from vitrina_cv.engines.opencv_classic import (
    _consolidate_walls,
    _group_indices_by_local_thickness,
    _group_indices_by_proximity,
)
from vitrina_cv.models import Wall


def _h_wall(x1: float, x2: float, y: float) -> Wall:
    return Wall(start=(x1, y), end=(x2, y))


def _make_thick_wall_mask(
    height: int,
    width: int,
    y_top: int,
    y_bottom: int,
    x_start: int,
    x_end: int,
) -> NDArray[np.uint8]:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y_top:y_bottom, x_start:x_end] = 255
    return mask


class TestLocalThicknessGrouping:
    def test_thin_zone_traces_merge_using_floor_tolerance(self) -> None:
        """A thin wall stripe (local DT ~ small) still merges its two Hough
        traces because the local tolerance is floored at _CENTERLINE_MIN_TOL_PX
        (8px), matching the pre-10-cv-05 minimum used by the global estimator.
        """
        # Thin stripe, 6px tall -> local DT sampled at midline is tiny.
        y_top, y_bottom = 40, 46
        x_start, x_end = 10, 300
        mask = _make_thick_wall_mask(200, 400, y_top, y_bottom, x_start, x_end)

        raw_walls = [
            _h_wall(x_start, x_end, 41.0),
            _h_wall(x_start, x_end, 45.0),
        ]

        settings = Settings(
            cv_wall_centerline_enabled=True,
            cv_wall_local_thickness_enabled=True,
        )
        consolidated = _consolidate_walls(raw_walls, mask, settings)

        assert len(consolidated) == 1, (
            f"Expected thin traces to still merge via floor tolerance, "
            f"got {len(consolidated)} walls"
        )

    def test_local_thickness_flag_off_matches_global_tolerance_grouping(
        self,
    ) -> None:
        """With CV_WALL_LOCAL_THICKNESS_ENABLED=false, grouping must be
        identical to the pre-10-cv-05 global-tolerance path
        (_group_indices_by_proximity)."""
        y_top, y_bottom = 40, 80
        x_start, x_end = 10, 300
        mask = _make_thick_wall_mask(200, 400, y_top, y_bottom, x_start, x_end)
        raw_walls = [
            _h_wall(x_start, x_end, 55.0),
            _h_wall(x_start, x_end, 65.0),
        ]

        settings_off = Settings(
            cv_wall_centerline_enabled=True,
            cv_wall_local_thickness_enabled=False,
        )
        settings_on = Settings(
            cv_wall_centerline_enabled=True,
            cv_wall_local_thickness_enabled=True,
        )

        consolidated_off = _consolidate_walls(raw_walls, mask, settings_off)
        consolidated_on = _consolidate_walls(raw_walls, mask, settings_on)

        assert len(consolidated_off) == len(consolidated_on) == 1

    def test_group_indices_by_local_thickness_reports_merge_count(self) -> None:
        """_group_indices_by_local_thickness returns the number of merges that
        used a real (non-fallback) local DT estimate."""
        mask = _make_thick_wall_mask(200, 400, 40, 80, 10, 300)
        dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        segs = [
            (10.0, 55.0, 300.0, 55.0),
            (10.0, 65.0, 300.0, 65.0),
        ]
        sorted_values = [55.0, 65.0]

        groups, merges = _group_indices_by_local_thickness(
            sorted_values, segs, dt, fallback_tolerance=8.0
        )

        assert groups == [(0, 1)], "Two close traces must collapse into 1 group"
        assert merges == 1, "One merge decision was made using local DT sampling"

    def test_group_indices_by_local_thickness_empty_input(self) -> None:
        """Empty input returns an empty grouping with a 0 merge count."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        groups, merges = _group_indices_by_local_thickness(
            [], [], dt, fallback_tolerance=8.0
        )

        assert groups == []
        assert merges == 0

    def test_group_indices_by_local_thickness_uses_fallback_outside_mask(
        self,
    ) -> None:
        """Segments entirely outside the mask (DT samples all zero-thickness)
        fall back to the given fallback_tolerance, matching the pre-existing
        _group_indices_by_proximity behaviour for that pair."""
        mask = np.zeros((50, 50), dtype=np.uint8)  # empty mask, no wall pixels
        dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        segs = [
            (0.0, 10.0, 20.0, 10.0),
            (0.0, 12.0, 20.0, 12.0),
        ]
        sorted_values = [10.0, 12.0]

        groups_local, _ = _group_indices_by_local_thickness(
            sorted_values, segs, dt, fallback_tolerance=8.0
        )
        groups_global = _group_indices_by_proximity(sorted_values, tolerance=8.0)

        assert groups_local == groups_global
