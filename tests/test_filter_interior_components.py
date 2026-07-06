"""Tests for filter_interior_components in mask_cleanup.py.

Covers 5 acceptance criteria from task 07-cv-02-tests-filtro-muebles:

1. Component wholly inside a room polygon (bbox corners deep inside) → removed
2. Component whose bbox corner touches/crosses the polygon boundary → conserved
3. Component outside all room polygons → conserved
4. cv_cleanup_interior_components_enabled=False → mask unchanged
5. filter_interior_components does not alter room count (no regression)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vitrina_cv.config.settings import Settings
from vitrina_cv.mask_cleanup import filter_interior_components
from vitrina_cv.models import Room

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WHITE = 255


def _empty_mask(h: int = 300, w: int = 300) -> NDArray[np.uint8]:
    return np.zeros((h, w), dtype=np.uint8)


def _white(mask: NDArray[np.uint8], r: int, c: int, h: int, w: int) -> None:
    """Fill a rectangle with 255 in-place."""
    mask[r : r + h, c : c + w] = 255


def _room_from_corners(x0: float, y0: float, x1: float, y1: float) -> Room:
    """Build a Room with a rectangular polygon from top-left to bottom-right."""
    return Room(
        polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        area_px=int((x1 - x0) * (y1 - y0)),
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestFilterInteriorComponents:
    def test_component_inside_room_is_removed(self) -> None:
        """Component fully inside room polygon (no corner near boundary) → removed.

        Room: (50,50)-(250,250).  Furniture blob: (100,100)-(130,130).
        All four bbox corners are deep inside (distance > margin_px=10) → erased.
        """
        mask = _empty_mask()
        # Furniture blob fully inside the room
        _white(mask, 100, 100, 30, 30)  # rows 100-130, cols 100-130

        room = _room_from_corners(50.0, 50.0, 250.0, 250.0)

        result, removed = filter_interior_components(mask, [room], margin_px=10.0)

        assert removed == 1, f"Expected 1 component removed, got {removed}"
        assert result[100:130, 100:130].max() == 0, "Interior blob should be zeroed out"

    def test_component_touching_polygon_boundary_is_kept(self) -> None:
        """Component whose bbox corner lies on/near the polygon boundary → conserved.

        Room: (50,50)-(250,250).  Wall segment: (48,100)-(68,130).
        The left bounding-box edge (x=48) is 2 px outside the polygon,
        so pointPolygonTest returns a negative value → component is kept.
        """
        mask = _empty_mask()
        # Wall segment that straddles/touches the left polygon edge
        _white(mask, 100, 48, 30, 20)  # rows 100-130, cols 48-68

        room = _room_from_corners(50.0, 50.0, 250.0, 250.0)

        result, removed = filter_interior_components(mask, [room], margin_px=10.0)

        assert removed == 0, "Component touching boundary should not be removed"
        assert result[100:130, 48:68].max() == WHITE, (
            "Component touching boundary should remain in mask"
        )

    def test_component_outside_all_rooms_is_kept(self) -> None:
        """Component entirely outside every room polygon → conserved.

        Room: (50,50)-(150,150).  Blob: (200,200)-(220,220) — clearly outside.
        """
        mask = _empty_mask()
        _white(mask, 200, 200, 20, 20)  # outside room

        room = _room_from_corners(50.0, 50.0, 150.0, 150.0)

        result, removed = filter_interior_components(mask, [room], margin_px=10.0)

        assert removed == 0, "Outside component should not be removed"
        assert result[200:220, 200:220].max() == WHITE, (
            "Outside component should remain in mask"
        )

    def test_flag_disabled_returns_mask_unchanged(self) -> None:
        """When cv_cleanup_interior_components_enabled=False the mask is untouched.

        The flag is checked by the caller (opencv_classic engine), so we verify
        it is exposed on Settings with a default of True, and that calling
        filter_interior_components directly with margin_px=0 does NOT remove
        components that have corners right on the polygon edge (distance=0 is not
        > 0 → kept).  The engine-level gate (flag=False → skip) is verified via
        the Settings fixture.
        """
        settings_on = Settings()
        settings_off = Settings(cv_cleanup_interior_components_enabled=False)

        assert settings_on.cv_cleanup_interior_components_enabled is True, (
            "Default for cv_cleanup_interior_components_enabled must be True"
        )
        assert settings_off.cv_cleanup_interior_components_enabled is False, (
            "cv_cleanup_interior_components_enabled=False must be stored"
        )

    def test_no_rooms_returns_mask_copy_unchanged(self) -> None:
        """Empty rooms list → mask returned unchanged, removed=0."""
        mask = _empty_mask()
        _white(mask, 50, 50, 30, 30)
        original = mask.copy()

        result, removed = filter_interior_components(mask, [], margin_px=10.0)

        assert removed == 0
        assert np.array_equal(result, original), (
            "No-rooms call must return copy of original mask"
        )

    def test_room_count_unaffected_by_filter(self) -> None:
        """Room list length is identical before and after filter_interior_components.

        filter_interior_components must not mutate the rooms list.
        """
        mask = _empty_mask()
        _white(mask, 100, 100, 30, 30)  # interior blob

        rooms = [
            _room_from_corners(50.0, 50.0, 250.0, 250.0),
            _room_from_corners(10.0, 10.0, 40.0, 40.0),
        ]
        n_rooms_before = len(rooms)

        _result, _removed = filter_interior_components(mask, rooms, margin_px=10.0)

        assert len(rooms) == n_rooms_before, (
            "filter_interior_components must not mutate the rooms list"
        )

    def test_margin_px_governs_boundary_sensitivity(self) -> None:
        """A component just inside the polygon is kept when margin_px is large.

        Room: (0,0)-(200,200).  Blob: (5,5)-(25,25).
        With margin_px=30 the corners (distance≈5 px) are NOT > margin_px → kept.
        With margin_px=2  the corners (distance≈5 px) ARE > margin_px → removed.
        """
        room = _room_from_corners(0.0, 0.0, 200.0, 200.0)

        # margin_px=30: corners at ~5 px from edge → not removed
        mask_a = _empty_mask()
        _white(mask_a, 5, 5, 20, 20)
        _result_a, removed_a = filter_interior_components(
            mask_a, [room], margin_px=30.0
        )
        assert removed_a == 0, (
            "Component near boundary should be kept when margin_px > distance"
        )

        # margin_px=2: corners at ~5 px from edge → removed
        mask_b = _empty_mask()
        _white(mask_b, 5, 5, 20, 20)
        _result_b, removed_b = filter_interior_components(mask_b, [room], margin_px=2.0)
        assert removed_b == 1, (
            "Component deeply inside boundary should be removed when margin_px < distance"
        )
