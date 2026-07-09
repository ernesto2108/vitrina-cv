"""Tests for mask_cleanup.py — pure functions over binary uint8 masks.

Covers 8 cases from the tester handoff:
1. remove_small_components removes compact blob, keeps elongated
2. remove_small_components on empty mask → no-op, removed=0
3. retain_rectilinear kills diagonal line; H-line of len_px+10 survives
4. retain_rectilinear removes short H-line < len_px
5. crop_to_main_component zeroes outside bbox of largest component
6. crop_to_main_component on empty mask → bbox=None, no change
7. clean_mask master switch disabled → output identical to input
8. clean_mask full pipeline reduces diagonal noise, preserves H/V walls
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from vitrina_cv.config.settings import Settings
from vitrina_cv.mask_cleanup import (
    clean_mask,
    crop_to_main_component,
    remove_small_components,
    retain_rectilinear,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


WHITE = 255


def _empty_mask(h: int = 100, w: int = 100) -> NDArray[np.uint8]:
    return np.zeros((h, w), dtype=np.uint8)


def _white(mask: NDArray[np.uint8], r: int, c: int, h: int, w: int) -> None:
    """Fill a rectangle with 255 in-place."""
    mask[r : r + h, c : c + w] = 255


# ---------------------------------------------------------------------------
# remove_small_components
# ---------------------------------------------------------------------------


class TestRemoveSmallComponents:
    def test_removes_compact_blob_and_keeps_elongated(self) -> None:
        """30x30 compact blob is removed (counter=1); 5x200 elongated component survives."""
        mask = _empty_mask(300, 300)
        # Compact blob — bbox 30x30, both dims < max_side_px=40 → removed
        _white(mask, 10, 10, 30, 30)
        # Elongated component — bbox 5x200, one dim >= 40 → kept
        _white(mask, 150, 10, 5, 200)

        result, n_removed = remove_small_components(mask, max_side_px=40)

        assert n_removed == 1, f"Expected 1 component removed, got {n_removed}"
        # Compact blob region should be zeroed
        assert result[10:40, 10:40].max() == 0, "Compact blob should be removed"
        # Elongated region should still be white
        assert result[150:155, 10:210].max() == WHITE, (
            "Elongated component should survive"
        )

    def test_no_op_on_empty_mask(self) -> None:
        """Empty mask returns empty mask with removed=0."""
        mask = _empty_mask()
        result, n_removed = remove_small_components(mask, max_side_px=40)

        assert n_removed == 0
        assert result.sum() == 0
        assert result.shape == mask.shape


# ---------------------------------------------------------------------------
# retain_rectilinear
# ---------------------------------------------------------------------------


class TestRetainRectilinear:
    def test_kills_diagonal_line_hline_survives(self) -> None:
        """Diagonal 45° line → output empty; H-line of len_px+10 → survives."""
        len_px = 100
        mask = _empty_mask(300, 300)

        # Draw 45° diagonal line
        cv2.line(mask, (10, 10), (200, 200), 255, 3)

        result_diag = retain_rectilinear(mask.copy(), len_px=len_px)
        # Diagonal should be eliminated (no horizontal/vertical runs >= len_px)
        assert result_diag.max() == 0, (
            "Diagonal line should be eliminated by retain_rectilinear"
        )

        # H-line of len_px+10
        mask_h = _empty_mask(300, 300)
        _white(mask_h, 50, 20, 3, len_px + 10)
        result_h = retain_rectilinear(mask_h, len_px=len_px)
        assert result_h.max() == WHITE, "H-line longer than len_px should survive"

    def test_short_hline_eliminated(self) -> None:
        """H-line shorter than len_px is eliminated."""
        len_px = 100
        mask = _empty_mask(300, 300)
        # Draw H-line of len_px - 20 (shorter than threshold)
        _white(mask, 50, 20, 3, len_px - 20)

        result = retain_rectilinear(mask, len_px=len_px)
        assert result.max() == 0, "H-line shorter than len_px should be eliminated"


# ---------------------------------------------------------------------------
# crop_to_main_component
# ---------------------------------------------------------------------------


class TestCropToMainComponent:
    def test_zeroes_outside_bbox_of_largest_component(self) -> None:
        """Smaller component placed far from main component is zeroed out."""
        mask = _empty_mask(400, 400)
        # Large component (main) at top-left
        _white(mask, 10, 10, 100, 100)
        # Small component far from main — outside main bbox + margin
        _white(mask, 350, 350, 20, 20)

        result, bbox = crop_to_main_component(mask, margin_px=5)

        assert bbox is not None, "bbox should not be None when main component exists"
        # Small distant component should be zeroed
        assert result[350:370, 350:370].max() == 0, (
            "Small distant component should be zeroed"
        )
        # Main component region should remain
        assert result[10:110, 10:110].max() == WHITE, (
            "Main component should be preserved"
        )

    def test_empty_mask_returns_none_bbox_and_unchanged_mask(self) -> None:
        """Empty mask → bbox=None, returned mask is unchanged."""
        mask = _empty_mask()
        original = mask.copy()

        result, bbox = crop_to_main_component(mask, margin_px=10)

        assert bbox is None, "bbox should be None for empty mask"
        assert np.array_equal(result, original), (
            "Empty mask should be returned unchanged"
        )

    def test_ac4_secondary_wing_within_min_area_ratio_included_in_envelope(
        self,
    ) -> None:
        """AC-4: main component + wing of 0.4*A_max area → both inside the envelope.

        The wing is placed far from the main component (separated by a gap)
        so that, without ADR-015's envelope-union behavior, it would fall
        outside the main component's own bbox + margin and be zeroed.
        """
        mask = _empty_mask(500, 500)
        # Main component: 100x100 = 10_000 px^2 (A_max).
        main_h, main_w = 100, 100
        _white(mask, 10, 10, main_h, main_w)
        a_max = main_h * main_w

        # Wing: area = 0.4 * A_max = 4_000 px^2 -> 40x100 rectangle placed far
        # away (bottom-right), well outside main bbox + small margin.
        wing_h, wing_w = 40, 100
        assert wing_h * wing_w == int(0.4 * a_max)
        _white(mask, 400, 380, wing_h, wing_w)

        result, bbox = crop_to_main_component(mask, margin_px=5, min_area_ratio=0.4)

        assert bbox is not None
        # Both components must survive the crop (not zeroed).
        assert result[10 : 10 + main_h, 10 : 10 + main_w].max() == WHITE, (
            "Main component must be preserved"
        )
        assert result[400 : 400 + wing_h, 380 : 380 + wing_w].max() == WHITE, (
            "Wing of 0.4*A_max must be included in the envelope, not zeroed"
        )
        # The envelope bbox must span both components (union), not just main.
        x, y, w, h = bbox
        assert x <= 10 - 5 or x == 0
        assert y <= 10 - 5 or y == 0
        assert x + w >= 380 + wing_w
        assert y + h >= 400 + wing_h

    def test_ac5_single_significant_component_envelope_equals_largest_bbox(
        self,
    ) -> None:
        """AC-5: only one significant component -> envelope == bbox of the
        largest component (compatibility invariant, no regression vs.
        pre-ADR-015 behavior)."""
        mask = _empty_mask(300, 300)
        main_h, main_w = 80, 60
        _white(mask, 20, 30, main_h, main_w)
        margin_px = 7

        result, bbox = crop_to_main_component(mask, margin_px=margin_px)

        assert bbox is not None
        x, y, w, h = bbox
        expected_x = max(0, 30 - margin_px)
        expected_y = max(0, 20 - margin_px)
        expected_x2 = min(mask.shape[1], 30 + main_w + margin_px)
        expected_y2 = min(mask.shape[0], 20 + main_h + margin_px)
        assert (x, y, x + w, y + h) == (
            expected_x,
            expected_y,
            expected_x2,
            expected_y2,
        ), "Envelope must equal bbox of the largest (and only) component"
        assert result[20 : 20 + main_h, 30 : 30 + main_w].max() == WHITE

    def test_ac6_small_component_below_min_area_ratio_excluded_from_envelope(
        self,
    ) -> None:
        """AC-6: a small component (area = 0.02*A_max, simulating a cota/frame
        stroke) stays outside the envelope and is zeroed out."""
        mask = _empty_mask(500, 500)
        main_h, main_w = 100, 100
        _white(mask, 10, 10, main_h, main_w)
        a_max = main_h * main_w

        # Small component: area = 0.02 * A_max = 200 px^2 -> 4x50 rectangle,
        # placed far from main so it would only be included if it qualified
        # as significant.
        small_h, small_w = 4, 50
        assert small_h * small_w == int(0.02 * a_max)
        small_pos = 450
        _white(mask, small_pos, small_pos, small_h, small_w)

        result, bbox = crop_to_main_component(mask, margin_px=5, min_area_ratio=0.4)

        assert bbox is not None
        assert result[10 : 10 + main_h, 10 : 10 + main_w].max() == WHITE, (
            "Main component must be preserved"
        )
        assert (
            result[
                small_pos : small_pos + small_h, small_pos : small_pos + small_w
            ].max()
            == 0
        ), "Component of 0.02*A_max (below min_area_ratio) must be zeroed"
        x, y, w, h = bbox
        assert x + w < small_pos, (
            "Envelope must not extend to include the small component"
        )
        assert y + h < small_pos, (
            "Envelope must not extend to include the small component"
        )


# ---------------------------------------------------------------------------
# clean_mask — integration of pipeline steps 1→2→3
# ---------------------------------------------------------------------------


class TestCleanMask:
    def test_master_switch_disabled_returns_identical_mask(self) -> None:
        """When cv_cleanup_enabled=False, output is byte-identical to input."""
        settings = Settings(cv_cleanup_enabled=False)
        mask = _empty_mask(200, 200)
        _white(mask, 10, 10, 30, 30)  # Small blob that would normally be removed
        cv2.line(
            mask, (5, 5), (150, 150), 255, 2
        )  # Diagonal that would normally be removed

        result = clean_mask(mask, settings)

        assert np.array_equal(result, mask), (
            "clean_mask with disabled switch must return input unchanged"
        )

    def test_full_pipeline_reduces_noise_preserves_walls(self) -> None:
        """Diagonal hatching pixels disappear; long H/V walls persist after clean_mask."""
        settings = Settings(
            cv_cleanup_enabled=True,
            cv_cleanup_text_max_side_px=40,
            cv_cleanup_rectilinear_len_px=80,
            cv_cleanup_crop_enabled=False,  # disable crop to keep layout simple
        )
        mask = _empty_mask(400, 400)

        # Long H wall — should survive (len >> len_px)
        wall_row = 200
        _white(mask, wall_row, 10, 4, 350)

        # Long V wall — should survive
        wall_col = 200
        _white(mask, 10, wall_col, 350, 4)

        # Diagonal hatching — should be eliminated.
        # cv2.line coords are (col, row). Draw diagonals in a dedicated region
        # rows 250-350, cols 10-100 (clear of walls at row=200, col=200).
        diag_row_start, diag_row_end = 250, 350
        diag_col_start, diag_col_end = 10, 100
        for offset in range(0, 60, 8):
            r0, c0 = diag_row_start + offset, diag_col_start
            r1, c1 = diag_row_start, diag_col_start + offset
            cv2.line(mask, (c0, r0), (c1, r1), 255, 1)  # (col, row) order

        # Pixel count before cleanup
        walls_before = int(mask[wall_row : wall_row + 4, 10:360].sum() // 255)
        diag_before = int(
            mask[diag_row_start:diag_row_end, diag_col_start:diag_col_end].sum() // 255
        )

        assert diag_before > 0, (
            "Diagonal hatching region must have pixels before cleanup"
        )

        result = clean_mask(mask, settings)

        walls_after = int(result[wall_row : wall_row + 4, 10:360].sum() // 255)
        diag_after = int(
            result[diag_row_start:diag_row_end, diag_col_start:diag_col_end].sum()
            // 255
        )

        assert walls_after > 0, "Long H wall should persist after cleanup"
        # Walls should be mostly preserved (at least 50% of pixels remain)
        assert walls_after >= walls_before * 0.5, (
            f"Too many wall pixels removed: before={walls_before}, after={walls_after}"
        )
        # Diagonal hatching should be significantly reduced
        assert diag_after < diag_before * 0.5, (
            f"Diagonal noise not reduced enough: before={diag_before}, after={diag_after}"
        )
