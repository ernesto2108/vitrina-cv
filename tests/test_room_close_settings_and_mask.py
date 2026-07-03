"""Tests for asymmetric room-close settings and _build_closed_wall_mask_for_rooms.

Covers:
1. Settings defaults: cv_room_close_h_gap_px=80, cv_room_close_v_gap_px=160
2. Settings env-var overrides: CV_ROOM_CLOSE_H_GAP_PX / CV_ROOM_CLOSE_V_GAP_PX
3. Narrow vertical corridor (~100px wide) is NOT filled by H close of 80px
4. Door gap in horizontal wall (~120px, between 80 and 160) IS bridged by V close
5. Small horizontal gap (~60px, < 80px) IS bridged by H close
6. Regression: real dense plan yields >= 10 rooms (skipped if env var unset)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray

from vitrina_cv.config.settings import Settings
from vitrina_cv.engines import opencv_classic as _engine_mod
from vitrina_cv.engines.opencv_classic import OpenCVClassicEngine

_build_closed_wall_mask_for_rooms = _engine_mod._build_closed_wall_mask_for_rooms  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Named constants (avoids PLR2004 magic-value warnings)
# ---------------------------------------------------------------------------

_DEFAULT_H_GAP_PX: int = 80
_DEFAULT_V_GAP_PX: int = 160
_ENV_OVERRIDE_H: int = 50
_ENV_OVERRIDE_V: int = 200
_MIN_ROOMS_DENSE_PLAN: int = 10

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WHITE = 255


def _blank(h: int, w: int) -> NDArray[np.uint8]:
    return np.zeros((h, w), dtype=np.uint8)


def _fill(mask: NDArray[np.uint8], r: int, c: int, h: int, w: int) -> None:
    mask[r : r + h, c : c + w] = WHITE


# ---------------------------------------------------------------------------
# 1 & 2 — Settings defaults and env-var overrides
# ---------------------------------------------------------------------------


class TestRoomCloseSettings:
    def test_h_gap_default(self) -> None:
        """cv_room_close_h_gap_px defaults to 80."""
        s = Settings()
        assert s.cv_room_close_h_gap_px == _DEFAULT_H_GAP_PX

    def test_v_gap_default(self) -> None:
        """cv_room_close_v_gap_px defaults to 160."""
        s = Settings()
        assert s.cv_room_close_v_gap_px == _DEFAULT_V_GAP_PX

    def test_h_gap_reads_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CV_ROOM_CLOSE_H_GAP_PX env var overrides the default."""
        monkeypatch.setenv("CV_ROOM_CLOSE_H_GAP_PX", str(_ENV_OVERRIDE_H))
        s = Settings()
        assert s.cv_room_close_h_gap_px == _ENV_OVERRIDE_H

    def test_v_gap_reads_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CV_ROOM_CLOSE_V_GAP_PX env var overrides the default."""
        monkeypatch.setenv("CV_ROOM_CLOSE_V_GAP_PX", str(_ENV_OVERRIDE_V))
        s = Settings()
        assert s.cv_room_close_v_gap_px == _ENV_OVERRIDE_V


# ---------------------------------------------------------------------------
# 3, 4, 5 — _build_closed_wall_mask_for_rooms with synthetic masks
# ---------------------------------------------------------------------------


class TestBuildClosedWallMaskForRooms:
    """Directional morphological close with asymmetric gaps (H=80, V=160)."""

    def test_narrow_vertical_corridor_not_filled(self) -> None:
        """A ~100px-wide vertical corridor (H gap > 80) is NOT filled by the H close.

        Setup: two vertical wall strips separated by 100 px of open space.
        The H close kernel is 80px, so it cannot bridge the 100px gap.
        Interior pixels must remain zero (floor, not wall).
        """
        h, w = 400, 600
        mask = _blank(h, w)
        # Left wall: rows 50-350, cols 100-110 (10px wide strip)
        _fill(mask, 50, 100, 300, 10)
        # Right wall: rows 50-350, cols 210-220 (10px wide strip, 100px gap from left)
        _fill(mask, 50, 210, 300, 10)

        result = _build_closed_wall_mask_for_rooms(
            mask, close_h_gap_px=80, close_v_gap_px=160
        )

        # Interior of the corridor (cols 110-209) should remain mostly open.
        # We check the middle column of the gap: col 160.
        interior = result[100:300, 155:165]  # middle of the 100px gap
        assert interior.max() == 0, (
            "Narrow corridor interior (100px gap, H close=80) must NOT be filled. "
            f"Max pixel value in interior region: {interior.max()}"
        )

    def test_door_gap_in_horizontal_wall_bridged_by_v_close(self) -> None:
        """A ~120px gap in a horizontal wall IS bridged by the V close (160px).

        Setup: two horizontal wall segments on the same row, with a 120px vertical
        gap between them (i.e., top segment ends at row 100, bottom segment starts
        at row 220 — gap of 120 rows). The V close kernel is 160px, so it bridges
        this gap.
        """
        h, w = 600, 600
        mask = _blank(h, w)
        # Top horizontal wall segment: rows 90-100, cols 100-400
        _fill(mask, 90, 100, 10, 300)
        # Bottom horizontal wall segment: rows 220-230, cols 100-400 (120-row gap)
        _fill(mask, 220, 100, 10, 300)

        result = _build_closed_wall_mask_for_rooms(
            mask, close_h_gap_px=80, close_v_gap_px=160
        )

        # The gap rows (110-219) in the middle columns should now be filled.
        gap_region = result[110:220, 200:300]
        assert gap_region.max() == WHITE, (
            "120px vertical gap in horizontal wall MUST be bridged by V close (160px). "
            f"Max pixel in gap region: {gap_region.max()}"
        )

    def test_small_horizontal_gap_bridged_by_h_close(self) -> None:
        """A ~60px gap in a vertical wall IS bridged by the H close (80px).

        Setup: two vertical wall segments on the same column with a 60px horizontal
        gap between them. The H close kernel is 80px, so it bridges this gap.
        """
        h, w = 400, 600
        mask = _blank(h, w)
        # Left vertical wall segment: rows 50-300, cols 100-110
        _fill(mask, 50, 100, 250, 10)
        # Right vertical wall segment: rows 50-300, cols 170-180 (60-col gap)
        _fill(mask, 50, 170, 250, 10)

        result = _build_closed_wall_mask_for_rooms(
            mask, close_h_gap_px=80, close_v_gap_px=160
        )

        # The gap columns (110-169) in the middle rows should now be filled.
        gap_region = result[100:250, 115:165]
        assert gap_region.max() == WHITE, (
            "60px horizontal gap in vertical wall MUST be bridged by H close (80px). "
            f"Max pixel in gap region: {gap_region.max()}"
        )


# ---------------------------------------------------------------------------
# 6 — Regression: real dense plan yields >= 10 rooms
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_dense_plan_regression_room_count() -> None:
    """Full engine on a real dense plan must detect >= 10 rooms.

    Skip if CV_TEST_DENSE_PLAN_PATH is not set or the file does not exist.
    """
    plan_path = os.environ.get("CV_TEST_DENSE_PLAN_PATH", "")
    if not plan_path:
        pytest.skip("CV_TEST_DENSE_PLAN_PATH not set — skipping regression test")
    if not os.path.exists(plan_path):
        pytest.skip(f"Plan file not found: {plan_path!r} — skipping regression test")

    with open(plan_path, "rb") as fh:
        image_bytes = fh.read()

    settings = Settings()
    engine = OpenCVClassicEngine(settings)
    geometry = engine.extract(image_bytes)

    assert len(geometry.rooms) >= _MIN_ROOMS_DENSE_PLAN, (
        f"Expected >= {_MIN_ROOMS_DENSE_PLAN} rooms on dense plan, got {len(geometry.rooms)}. "
        "The asymmetric H/V close fix may have regressed."
    )
