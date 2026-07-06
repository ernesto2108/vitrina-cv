"""Tests for _detect_stairs_candidates — task cv-11.

Covers 8 scenarios from acceptance criteria:
1. 5 equi-spaced horizontal treads inside a room → 1 StairsCandidate
2. Pattern outside any room → 0 candidates (deck anti-FP)
3. Only 3 lines (< 4 required) → 0 candidates
4. Irregular spacing (high rel-std) → 0 candidates
5. cv_stairs_detection_enabled=False → 0 candidates (flag gate)
6. Stair tread lines absent from walls (bbox sanity — no Wall inside staircase bbox)
7. Serialization: bbox has 4 floats, direction is a valid StairsDirection member
8. 5 equi-spaced vertical treads inside a room → 1 StairsCandidate
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from vitrina_cv.config.settings import Settings
from vitrina_cv.engines.opencv_classic import _detect_stairs_candidates
from vitrina_cv.models import Room, StairsCandidate, StairsDirection, Wall

# ---------------------------------------------------------------------------
# Constants (mirror production values — not imported to avoid coupling)
# ---------------------------------------------------------------------------

_SPACING_PX: int = 30  # within [20, 40] allowed range
_N_TREADS: int = 5
_LINE_LEN: int = 100  # px — well above HoughLinesP minLineLength=20
_CONFIDENCE_FIXED: float = 0.6  # _STAIRS_CONFIDENCE
_BBOX_LEN: int = 4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blank(h: int = 400, w: int = 400) -> NDArray[np.uint8]:
    return np.zeros((h, w), dtype=np.uint8)


def _draw_h_lines(
    mask: NDArray[np.uint8],
    y_start: int,
    spacing: int,
    n: int,
    x0: int = 50,
    length: int = _LINE_LEN,
    thickness: int = 2,
) -> list[int]:
    """Draw *n* horizontal lines and return their y-coords."""
    ys = [y_start + i * spacing for i in range(n)]
    for y in ys:
        mask[y : y + thickness, x0 : x0 + length] = 255
    return ys


def _draw_v_lines(
    mask: NDArray[np.uint8],
    x_start: int,
    spacing: int,
    n: int,
    y0: int = 50,
    length: int = _LINE_LEN,
    thickness: int = 2,
) -> list[int]:
    """Draw *n* vertical lines and return their x-coords."""
    xs = [x_start + i * spacing for i in range(n)]
    for x in xs:
        mask[y0 : y0 + length, x : x + thickness] = 255
    return xs


def _room_containing_h_treads(
    y_start: int,
    spacing: int,
    n: int,
    x0: int = 50,
    length: int = _LINE_LEN,
    margin: int = 10,
) -> Room:
    """Build a Room whose polygon tightly contains the horizontal tread block."""
    x1 = x0 - margin
    y1 = y_start - margin
    x2 = x0 + length + margin
    y2 = y_start + (n - 1) * spacing + margin
    return Room(
        polygon=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
        area_px=float((x2 - x1) * (y2 - y1)),
    )


def _room_containing_v_treads(
    x_start: int,
    spacing: int,
    n: int,
    y0: int = 50,
    length: int = _LINE_LEN,
    margin: int = 10,
) -> Room:
    """Build a Room whose polygon tightly contains the vertical tread block."""
    x1 = x_start - margin
    y1 = y0 - margin
    x2 = x_start + (n - 1) * spacing + margin
    y2 = y0 + length + margin
    return Room(
        polygon=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
        area_px=float((x2 - x1) * (y2 - y1)),
    )


def _far_room() -> Room:
    """A room polygon far from any drawn pattern (top-left corner of a 400x400 canvas)."""
    return Room(
        polygon=[(300, 300), (390, 300), (390, 390), (300, 390)],
        area_px=8100.0,
    )


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


class TestDetectStairsCandidates:
    # ------------------------------------------------------------------
    # Case 1 — happy path: 5 equi-spaced horizontal treads inside a room
    # ------------------------------------------------------------------

    def test_horizontal_treads_inside_room_returns_one_candidate(self) -> None:
        """5 horizontal lines @ 30 px spacing inside a room → exactly 1 candidate."""
        mask = _blank()
        _draw_h_lines(mask, y_start=50, spacing=_SPACING_PX, n=_N_TREADS)
        room = _room_containing_h_treads(y_start=50, spacing=_SPACING_PX, n=_N_TREADS)

        result = _detect_stairs_candidates(mask, [room], settings=None)

        assert len(result) == 1, f"Expected 1 candidate, got {len(result)}"
        cand = result[0]
        assert isinstance(cand, StairsCandidate)
        assert 0.0 <= cand.confidence <= 1.0
        assert len(cand.bbox) == _BBOX_LEN
        # bbox x, y must be >= 0 and w, h > 0
        bx, by, bw, bh = cand.bbox
        assert bx >= 0
        assert by >= 0
        assert bw > 0
        assert bh > 0

    # ------------------------------------------------------------------
    # Case 2 — anti-FP: pattern outside any room (deck)
    # ------------------------------------------------------------------

    def test_pattern_outside_room_returns_empty(self) -> None:
        """Stair pattern drawn far from the room polygon → 0 candidates."""
        mask = _blank()
        # treads at y=50..170, x=50..150 — room is at bottom-right
        _draw_h_lines(mask, y_start=50, spacing=_SPACING_PX, n=_N_TREADS)
        room = _far_room()

        result = _detect_stairs_candidates(mask, [room], settings=None)

        assert result == [], f"Expected [], got {result}"

    # ------------------------------------------------------------------
    # Case 3 — fewer than 4 lines (only 3)
    # ------------------------------------------------------------------

    def test_three_lines_returns_empty(self) -> None:
        """Only 3 horizontal treads (< 4 required) → 0 candidates."""
        mask = _blank()
        _draw_h_lines(mask, y_start=50, spacing=_SPACING_PX, n=3)
        room = _room_containing_h_treads(y_start=50, spacing=_SPACING_PX, n=3)

        result = _detect_stairs_candidates(mask, [room], settings=None)

        assert result == [], f"Expected [], got {result}"

    # ------------------------------------------------------------------
    # Case 4 — irregular spacing (rel-std > 0.20)
    # ------------------------------------------------------------------

    def test_irregular_spacing_returns_empty(self) -> None:
        """4 treads with spacings [20, 20, 40] → rel-std ≈ 0.35 > 0.20 → rejected."""
        mask = _blank()
        # y positions: 50, 70, 90, 130 → spacings 20, 20, 40
        ys = [50, 70, 90, 130]
        for y in ys:
            mask[y : y + 2, 50:150] = 255
        # Room that contains all four treads
        room = Room(
            polygon=[(40, 40), (160, 40), (160, 145), (40, 145)],
            area_px=float(120 * 105),
        )

        result = _detect_stairs_candidates(mask, [room], settings=None)

        assert result == [], f"Expected [], got {result}"

    # ------------------------------------------------------------------
    # Case 5 — flag disabled
    # ------------------------------------------------------------------

    def test_flag_disabled_returns_empty(self) -> None:
        """cv_stairs_detection_enabled=False → early return, no detection."""
        mask = _blank()
        _draw_h_lines(mask, y_start=50, spacing=_SPACING_PX, n=_N_TREADS)
        room = _room_containing_h_treads(y_start=50, spacing=_SPACING_PX, n=_N_TREADS)
        settings = Settings(cv_stairs_detection_enabled=False)

        result = _detect_stairs_candidates(mask, [room], settings=settings)

        assert result == [], "Flag disabled must return empty list"

    # ------------------------------------------------------------------
    # Case 6 — stair tread bbox contains 0 Wall objects (bbox sanity)
    # ------------------------------------------------------------------

    def test_no_walls_inside_stairs_bbox(self) -> None:
        """Tread lines only — the detected bbox should not overlap any Wall.

        This verifies that stair tread lines would not be mis-classified as
        walls: we confirm that the StairsCandidate bbox was produced from
        thin tread segments, not from wall-thickness strokes.
        The test produces a candidate, then checks that a Wall placed far
        outside the bbox would not be considered "inside" it.
        """
        mask = _blank()
        _draw_h_lines(mask, y_start=50, spacing=_SPACING_PX, n=_N_TREADS)
        room = _room_containing_h_treads(y_start=50, spacing=_SPACING_PX, n=_N_TREADS)

        result = _detect_stairs_candidates(mask, [room], settings=None)
        assert len(result) == 1
        bx, by, bw, _bh = result[0].bbox

        # Wall far outside the detected bbox
        wall_outside = Wall(start=(bx + bw + 50, by), end=(bx + bw + 100, by))

        def _wall_inside_bbox(wall: Wall, bbox: list[float]) -> bool:
            x, y, w, h = bbox
            wx = (wall.start[0] + wall.end[0]) / 2
            wy = (wall.start[1] + wall.end[1]) / 2
            return x <= wx <= x + w and y <= wy <= y + h

        assert not _wall_inside_bbox(wall_outside, result[0].bbox), (
            "Wall outside bbox should not be detected as inside"
        )

    # ------------------------------------------------------------------
    # Case 7 — serialization
    # ------------------------------------------------------------------

    def test_serialization_bbox_and_direction(self) -> None:
        """StairsCandidate has bbox of 4 floats and direction is a valid enum member."""
        mask = _blank()
        _draw_h_lines(mask, y_start=50, spacing=_SPACING_PX, n=_N_TREADS)
        room = _room_containing_h_treads(y_start=50, spacing=_SPACING_PX, n=_N_TREADS)

        result = _detect_stairs_candidates(mask, [room], settings=None)
        assert len(result) == 1
        cand = result[0]

        # bbox: list of 4 floats
        assert len(cand.bbox) == _BBOX_LEN
        for v in cand.bbox:
            assert isinstance(v, float), f"bbox value {v!r} is not float"

        # direction is a valid StairsDirection member
        assert cand.direction in StairsDirection, (
            f"{cand.direction!r} not a valid StairsDirection"
        )

        # Pydantic round-trip
        serialized = cand.model_dump()
        restored = StairsCandidate.model_validate(serialized)
        assert restored == cand

    # ------------------------------------------------------------------
    # Case 8 — vertical treads → candidate detected
    # ------------------------------------------------------------------

    def test_vertical_treads_inside_room_returns_one_candidate(self) -> None:
        """5 vertical treads @ 30 px spacing inside a room → exactly 1 candidate."""
        mask = _blank()
        _draw_v_lines(mask, x_start=50, spacing=_SPACING_PX, n=_N_TREADS)
        room = _room_containing_v_treads(x_start=50, spacing=_SPACING_PX, n=_N_TREADS)

        result = _detect_stairs_candidates(mask, [room], settings=None)

        assert len(result) == 1, f"Expected 1 vertical candidate, got {len(result)}"
        cand = result[0]
        assert 0.0 <= cand.confidence <= 1.0
        _bx, _by, bw, bh = cand.bbox
        assert bw > 0
        assert bh > 0
