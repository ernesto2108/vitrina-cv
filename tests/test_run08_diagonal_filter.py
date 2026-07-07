"""Tests for Run 08 features.

Covers:
  F1 (08-cv-01) — diagonal wall filter in _consolidate_walls:
    1. Segment at 45° is discarded when flag is on (default range 20°-70°)
    2. Horizontal segment (0°) is kept regardless
    3. Vertical segment (90°) is kept regardless
    4. Segment just below low_deg (e.g. 15°) is kept (outside discard band)
    5. Segment just above high_deg (e.g. 75°) is kept (outside discard band)
    6. AC-3: flag off → all diagonal segments are preserved unchanged

  F2 (08-cv-03) — adaptive rectilinear filter in clean_mask for high-res images:
    7. High-res image with flag on → diagonal hatching removed (retain_rectilinear applied)
    8. High-res image with flag off → hatching survives (legacy skip behaviour)
    9. Adaptive len_px formula: max(50, round(base_len * min(h,w) / target_px))
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray

from vitrina_cv.config.settings import Settings
from vitrina_cv.engines.opencv_classic import _consolidate_walls
from vitrina_cv.mask_cleanup import clean_mask
from vitrina_cv.models import Wall

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wall_from_angle(angle_deg: float, length: float = 100.0) -> Wall:
    """Create a Wall whose atan2(|dy|,|dx|) equals *angle_deg*."""
    rad = math.radians(angle_deg)
    dx = length * math.cos(rad)
    dy = length * math.sin(rad)
    return Wall(start=(0.0, 0.0), end=(dx, dy))


def _settings_filter_on(low: float = 20.0, high: float = 70.0) -> Settings:
    return Settings(
        cv_wall_diagonal_filter_enabled=True,
        cv_wall_diagonal_filter_low_deg=low,
        cv_wall_diagonal_filter_high_deg=high,
        cv_wall_centerline_enabled=False,  # legacy mode — no mask needed
    )


def _settings_filter_off() -> Settings:
    return Settings(
        cv_wall_diagonal_filter_enabled=False,
        cv_wall_centerline_enabled=False,
    )


def _ids_in(walls: list[Wall]) -> set[tuple[tuple[float, float], tuple[float, float]]]:
    return {(w.start, w.end) for w in walls}


# ---------------------------------------------------------------------------
# F1 — diagonal wall filter
# ---------------------------------------------------------------------------


class TestDiagonalWallFilter:
    def test_45_degree_segment_is_discarded(self) -> None:
        """Segment at exactly 45° falls in [20, 70] → discarded when flag on."""
        wall_45 = _wall_from_angle(45.0)
        result = _consolidate_walls([wall_45], wall_mask=None, settings=_settings_filter_on())
        # No diagonal segments should survive — the only input was diagonal
        diagonals_in_result = [
            w
            for w in result
            if not (
                abs(w.start[1] - w.end[1]) < 1e-6  # horizontal
                or abs(w.start[0] - w.end[0]) < 1e-6  # vertical
            )
        ]
        assert len(diagonals_in_result) == 0, (
            f"45° wall should be discarded, but got {len(diagonals_in_result)} diagonal(s)"
        )

    def test_horizontal_segment_is_kept(self) -> None:
        """Horizontal wall (0°) is classified as H — always kept."""
        wall_h = Wall(start=(0.0, 50.0), end=(200.0, 50.0))
        result = _consolidate_walls([wall_h], wall_mask=None, settings=_settings_filter_on())
        assert len(result) >= 1, "Horizontal wall must survive consolidation"

    def test_vertical_segment_is_kept(self) -> None:
        """Vertical wall (90°) is classified as V — always kept."""
        wall_v = Wall(start=(50.0, 0.0), end=(50.0, 200.0))
        result = _consolidate_walls([wall_v], wall_mask=None, settings=_settings_filter_on())
        assert len(result) >= 1, "Vertical wall must survive consolidation"

    def test_segment_below_low_deg_is_kept(self) -> None:
        """Segment at 15° is below low_deg=20 → kept (outside discard band)."""
        wall_15 = _wall_from_angle(15.0)
        result = _consolidate_walls([wall_15], wall_mask=None, settings=_settings_filter_on())
        # 15° is in the diagonal bucket but below low_deg — should be kept
        assert len(result) >= 1, "15° segment (below low_deg=20) should be kept"

    def test_segment_above_high_deg_is_kept(self) -> None:
        """Segment at 75° is above high_deg=70 → kept (outside discard band)."""
        wall_75 = _wall_from_angle(75.0)
        result = _consolidate_walls([wall_75], wall_mask=None, settings=_settings_filter_on())
        assert len(result) >= 1, "75° segment (above high_deg=70) should be kept"

    def test_flag_off_preserves_all_diagonal_segments(self) -> None:
        """AC-3: with flag off all diagonal segments survive (pre-08 behaviour)."""
        wall_45 = _wall_from_angle(45.0)
        wall_30 = _wall_from_angle(30.0)
        walls_in = [wall_45, wall_30]
        result = _consolidate_walls(walls_in, wall_mask=None, settings=_settings_filter_off())
        result_starts = {w.start for w in result}
        for w in walls_in:
            assert w.start in result_starts, (
                f"Diagonal wall at start={w.start} should be kept when filter is off"
            )

    def test_orthogonal_walls_unaffected_when_flag_on(self) -> None:
        """Mixed input: H/V walls survive; 45° diagonal is dropped."""
        wall_h = Wall(start=(0.0, 50.0), end=(200.0, 50.0))
        wall_v = Wall(start=(50.0, 0.0), end=(50.0, 200.0))
        wall_diag = _wall_from_angle(45.0)

        result = _consolidate_walls(
            [wall_h, wall_v, wall_diag],
            wall_mask=None,
            settings=_settings_filter_on(),
        )
        # At least the H and V walls should be in the output
        assert len(result) >= 2, (
            f"H and V walls must survive; got only {len(result)} walls"
        )
        # The 45° diagonal's start should not appear
        result_starts = {w.start for w in result}
        assert wall_diag.start not in result_starts, (
            "45° diagonal wall should not appear in the consolidated output"
        )

    def test_custom_range_discards_only_within_band(self) -> None:
        """Custom range [30, 60]: 45° discarded, 15° and 75° kept."""
        settings = Settings(
            cv_wall_diagonal_filter_enabled=True,
            cv_wall_diagonal_filter_low_deg=30.0,
            cv_wall_diagonal_filter_high_deg=60.0,
            cv_wall_centerline_enabled=False,
        )
        wall_45 = _wall_from_angle(45.0)
        wall_15 = _wall_from_angle(15.0)
        wall_75 = _wall_from_angle(75.0)

        result_45 = _consolidate_walls([wall_45], wall_mask=None, settings=settings)
        result_15 = _consolidate_walls([wall_15], wall_mask=None, settings=settings)
        result_75 = _consolidate_walls([wall_75], wall_mask=None, settings=settings)

        assert len(result_45) == 0, "45° should be discarded in [30,60] band"
        assert len(result_15) >= 1, "15° is outside [30,60] band — should be kept"
        assert len(result_75) >= 1, "75° is outside [30,60] band — should be kept"


# ---------------------------------------------------------------------------
# F2 — adaptive rectilinear filter for high-res images
# ---------------------------------------------------------------------------

_WHITE = 255


def _make_highres_mask_with_hatching(height: int, width: int) -> NDArray[np.uint8]:
    """Create a high-res mask with a long H wall and dense diagonal hatching.

    Long H wall: 4 px tall, spanning most of the width.
    Diagonal hatching: repeated short diagonal lines in a separate region.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    # Long H wall — should survive retain_rectilinear
    mask[height // 2 : height // 2 + 4, 50:width - 50] = _WHITE
    # Dense diagonal hatching in a corner region
    hatch_region = height // 4
    for i in range(0, hatch_region, 8):
        cv2.line(mask, (i, i), (i + 6, i + 6), _WHITE, 1)
    return mask


class TestAdaptiveRectilinearFilter:
    def _highres_settings(self, *, adaptive_on: bool) -> Settings:
        """Settings that classify the test image as high-res (long_side > target_px)."""
        return Settings(
            cv_cleanup_enabled=True,
            cv_cleanup_rectilinear_adaptive_enabled=adaptive_on,
            cv_upscale_target_px=2000,
            cv_cleanup_rectilinear_max_res_scale=1.0,  # skip at scale > 1.0
            cv_cleanup_rectilinear_len_px=150,
            cv_cleanup_text_max_side_px=5,  # generous to keep hatching through step 1
            cv_cleanup_crop_enabled=False,
            cv_cleanup_thickness_filter_enabled=False,  # isolate step 2 behaviour
        )

    def test_highres_flag_on_removes_hatching_preserves_hwall(self) -> None:
        """High-res image with adaptive flag on: hatching reduced, H wall kept."""
        # 3000×3000 > cv_upscale_target_px=2000 → triggers adaptive branch
        h, w = 3000, 3000
        mask = _make_highres_mask_with_hatching(h, w)
        settings = self._highres_settings(adaptive_on=True)

        result = clean_mask(mask, settings)

        # Long H wall should survive
        wall_row = h // 2
        assert result[wall_row : wall_row + 4, 100 : w - 100].max() == _WHITE, (
            "Long H wall must survive adaptive rectilinear filter"
        )

        # Dense hatching region should be largely suppressed
        hatch_pixels_before = int(mask[0 : h // 4, 0 : h // 4].sum()) // _WHITE
        hatch_pixels_after = int(result[0 : h // 4, 0 : h // 4].sum()) // _WHITE
        assert hatch_pixels_after < hatch_pixels_before, (
            "Diagonal hatching should be reduced by adaptive filter "
            f"(before={hatch_pixels_before}, after={hatch_pixels_after})"
        )

    def test_highres_flag_off_hatching_survives(self) -> None:
        """AC-6: high-res image with adaptive flag off → step 2 skipped, hatching preserved."""
        h, w = 3000, 3000
        mask = _make_highres_mask_with_hatching(h, w)
        settings = self._highres_settings(adaptive_on=False)

        result = clean_mask(mask, settings)

        hatch_pixels_before = int(mask[0 : h // 4, 0 : h // 4].sum()) // _WHITE
        hatch_pixels_after = int(result[0 : h // 4, 0 : h // 4].sum()) // _WHITE
        # With step 2 skipped, hatching pixels should be fully preserved through step 2
        assert hatch_pixels_after == hatch_pixels_before, (
            "With adaptive flag off, step 2 is skipped and hatching must be unchanged "
            f"(before={hatch_pixels_before}, after={hatch_pixels_after})"
        )

    def test_adaptive_len_px_formula(self) -> None:
        """Adaptive len_px = max(50, round(base_len * min(h,w) / target_px)).

        For a 3000x4000 image with base_len=150, target=2000:
          min(h,w)=3000 → round(150*3000/2000) = round(225) = 225
          max(50, 225) = 225
        The formula is exercised indirectly: a wall of 224 px should be removed
        but a wall of 226 px should survive when len_px=225.
        """
        h, w = 3000, 4000
        base_len = 150
        target = 2000
        expected_len = max(50, round(base_len * min(h, w) / target))
        assert expected_len == 225, f"Expected len_px=225, got {expected_len}"

        settings = Settings(
            cv_cleanup_enabled=True,
            cv_cleanup_rectilinear_adaptive_enabled=True,
            cv_upscale_target_px=target,
            cv_cleanup_rectilinear_max_res_scale=1.0,
            cv_cleanup_rectilinear_len_px=base_len,
            cv_cleanup_text_max_side_px=5,
            cv_cleanup_crop_enabled=False,
            cv_cleanup_thickness_filter_enabled=False,
        )

        short_mask = np.zeros((h, w), dtype=np.uint8)
        # Wall shorter than expected_len — should be eliminated
        short_mask[h // 2 : h // 2 + 4, 50 : 50 + expected_len - 2] = _WHITE

        long_mask = np.zeros((h, w), dtype=np.uint8)
        # Wall longer than expected_len — should survive
        long_mask[h // 2 : h // 2 + 4, 50 : 50 + expected_len + 2] = _WHITE

        result_short = clean_mask(short_mask, settings)
        result_long = clean_mask(long_mask, settings)

        assert result_short.max() == 0, (
            f"Wall shorter than adaptive len_px={expected_len} should be eliminated"
        )
        assert result_long.max() == _WHITE, (
            f"Wall longer than adaptive len_px={expected_len} should survive"
        )

    def test_lowres_image_uses_fixed_kernel_not_adaptive(self) -> None:
        """Standard-res image (long_side <= target_px) always uses fixed kernel.

        Ensures the adaptive branch is not triggered when resolution_scale_raw <= 1.0.
        A wall of fixed len_px+5 must survive; a wall of fixed len_px-5 must be removed.
        """
        # Image well within 2000 px target
        h, w = 400, 400
        len_px = 80  # small enough to fit in 400×400
        settings = Settings(
            cv_cleanup_enabled=True,
            cv_cleanup_rectilinear_adaptive_enabled=True,  # flag on, but should not trigger
            cv_upscale_target_px=2000,
            cv_cleanup_rectilinear_max_res_scale=1.0,
            cv_cleanup_rectilinear_len_px=len_px,
            cv_cleanup_text_max_side_px=5,
            cv_cleanup_crop_enabled=False,
            cv_cleanup_thickness_filter_enabled=False,
        )

        long_mask = np.zeros((h, w), dtype=np.uint8)
        long_mask[h // 2 : h // 2 + 4, 10 : 10 + len_px + 5] = _WHITE

        short_mask = np.zeros((h, w), dtype=np.uint8)
        short_mask[h // 2 : h // 2 + 4, 10 : 10 + len_px - 5] = _WHITE

        result_long = clean_mask(long_mask, settings)
        result_short = clean_mask(short_mask, settings)

        assert result_long.max() == _WHITE, (
            f"Wall of len_px+5={len_px + 5} should survive with fixed kernel={len_px}"
        )
        assert result_short.max() == 0, (
            f"Wall of len_px-5={len_px - 5} should be removed with fixed kernel={len_px}"
        )
