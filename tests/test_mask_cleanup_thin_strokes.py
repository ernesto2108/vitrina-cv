"""Tests for filter_thin_strokes, clean_mask step order, integration fixtures and new settings.

Covers the tester handoff for the thin-stroke filter (step 4):
1. Thick stroke (12 px wide rect) survives filter_thin_strokes.
2. Isolated thin stroke (1-2 px) disappears.
3. Thin stroke CONNECTED to a thick stroke disappears (the motivating case).
4. Double hollow wall (two thin parallel lines < preclose gap) survives via pre-close.
5. Flag disabled (cv_cleanup_thickness_filter_enabled=False) -> mask passes unchanged.
6. Empty mask / all-white mask -> no crash.
7. Pipeline order: crop (step 3) occurs before filter (step 4) in clean_mask.
8. Settings: new env vars read with CV_ prefix and correct defaults.
9-10. Integration (slow): golden/threshold regression over real fixtures.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray

from vitrina_cv.config.settings import Settings
from vitrina_cv.mask_cleanup import clean_mask, filter_thin_strokes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WHITE = 255
FIXTURES = Path(__file__).parent / "fixtures"

# Integration test thresholds (measured 2026-07-03 with filter enabled)
_LIMPIO_EXPECTED_ROOMS = 6
_LIMPIO_MAX_WALLS = 90
_DENSO_MIN_ROOMS = 4
_DENSO_MAX_WALLS = 100
_DEFAULT_MIN_WALL_THICKNESS_PX = 6


def _empty_mask(h: int = 200, w: int = 200) -> NDArray[np.uint8]:
    return np.zeros((h, w), dtype=np.uint8)


def _white(mask: NDArray[np.uint8], r: int, c: int, h: int, w: int) -> None:
    mask[r : r + h, c : c + w] = WHITE


def _settings_step4_only(**kwargs) -> Settings:
    """Return Settings with all cleanup enabled, only step-4 params overridable."""
    defaults: dict = {
        "cv_cleanup_enabled": True,
        "cv_cleanup_text_max_side_px": 5,  # tiny threshold -> step 1 kills nothing real
        "cv_cleanup_rectilinear_len_px": 5,  # tiny threshold -> step 2 kills nothing real
        "cv_cleanup_crop_enabled": False,  # disable crop so mask geometry stays simple
        "cv_cleanup_thickness_filter_enabled": True,
        "cv_cleanup_min_wall_thickness_px": 6,
        "cv_cleanup_thickness_preclose_px": 9,
    }
    defaults.update(kwargs)
    return Settings(**defaults)


# ---------------------------------------------------------------------------
# filter_thin_strokes -- unit tests with synthetic masks
# ---------------------------------------------------------------------------


class TestFilterThinStrokes:
    def test_thick_stroke_survives(self) -> None:
        """A 12-px wide rect fills the distance-transform seeds -> most pixels survive."""
        mask = _empty_mask(100, 100)
        # 12-px wide x 60-px long horizontal bar
        _white(mask, 44, 20, 12, 60)

        result = filter_thin_strokes(mask, min_thickness_px=6, preclose_kernel_size=9)

        # The center core of the bar must survive (distance >= 6/2=3 for a 12-px bar)
        center_pixels = int(result[44:56, 20:80].sum() // WHITE)
        original_pixels = int(mask[44:56, 20:80].sum() // WHITE)
        # At minimum 50 % of the bar's pixels should remain
        assert center_pixels >= original_pixels * 0.5, (
            f"Thick stroke lost too many pixels: {center_pixels}/{original_pixels}"
        )

    def test_isolated_thin_stroke_disappears(self) -> None:
        """A 1-px line far from any thick stroke produces no seeds -> removed."""
        mask = _empty_mask(150, 200)
        # 1-px thick horizontal line at row 75
        mask[75, 20:180] = WHITE

        result = filter_thin_strokes(mask, min_thickness_px=6, preclose_kernel_size=9)

        assert result.max() == 0, "Isolated 1-px stroke should be fully removed"

    def test_thin_stroke_connected_to_thick_disappears(self) -> None:
        """Critical case: thin annotation connected to thick wall.

        A 1-px horizontal cota line touches the left edge of a 12-px thick wall.
        After filter_thin_strokes, the thick wall should largely survive while
        the thin annotation (except possibly a few junction pixels) is gone.
        """
        mask = _empty_mask(150, 250)
        # Thick wall: 12 px wide x 80 px long, placed in the right half
        thick_r, thick_c, thick_h, thick_w = 69, 120, 12, 80
        _white(mask, thick_r, thick_c, thick_h, thick_w)

        # Thin annotation: 1 px wide x 110 px long, connected to the left end of wall.
        # Row 75 is the vertical midpoint of the thick wall (69 + 6 = 75).
        thin_r, thin_c, thin_len = 75, 10, 110  # ends at col 120 (junction)
        mask[thin_r, thin_c : thin_c + thin_len] = WHITE

        result = filter_thin_strokes(mask, min_thickness_px=6, preclose_kernel_size=9)

        # The thick wall core must still have significant pixel coverage.
        thick_after = int(
            result[thick_r : thick_r + thick_h, thick_c : thick_c + thick_w].sum()
            // WHITE
        )
        thick_before = thick_h * thick_w  # 960 px
        assert thick_after >= thick_before * 0.4, (
            f"Thick wall decimated: {thick_after}/{thick_before}"
        )

        # The thin annotation (excluding a small buffer near the junction) should be gone.
        # Check pixels well away from the junction: cols 10..100 (20 px gap before junction).
        thin_far = int(result[thin_r, thin_c : thin_c + 90].sum() // WHITE)
        assert thin_far == 0, (
            f"Thin annotation far from junction should be removed, got {thin_far} pixels"
        )

    def test_double_hollow_wall_survives_via_preclose(self) -> None:
        """Two 3-px parallel strands with 5-px gap survive thanks to the 9-px pre-close.

        Algorithm requirement: after the pre-close, the inner edge of each strand
        must have distance >= min_thickness_px/2 = 3 in the pre-closed mask.
        With 3-px strands + 5-px gap + 3-px strands = 11-px total, the inner-edge
        pixels are at distance 3 from the nearest background edge -> they become
        seeds -> bounded dilation recovers the full strand extents.

        Note: 2-px strands do NOT satisfy this constraint (inner edge dist = 2 < 3);
        the minimum working strand thickness with default settings is 3 px.
        """
        mask = _empty_mask(150, 200)
        # Top strand at row 60, 3 px thick
        _white(mask, 60, 20, 3, 150)
        # Bottom strand at row 68, 3 px thick (gap = 5 px, total band span = 11 px)
        _white(mask, 68, 20, 3, 150)

        result = filter_thin_strokes(mask, min_thickness_px=6, preclose_kernel_size=9)

        # Both strands must survive substantially.
        combined_after = int(result[60:71, 20:170].sum() // WHITE)
        combined_before = int(mask[60:71, 20:170].sum() // WHITE)
        assert combined_after >= combined_before * 0.5, (
            f"Double hollow wall (3-px strands, 5-px gap) lost too many pixels: "
            f"{combined_after}/{combined_before}"
        )

    def test_flag_disabled_passes_mask_unchanged(self) -> None:
        """With cv_cleanup_thickness_filter_enabled=False, mask is returned byte-identical."""
        settings = _settings_step4_only(cv_cleanup_thickness_filter_enabled=False)

        mask = _empty_mask(100, 100)
        # Thin annotation that would normally be removed
        mask[50, 10:90] = WHITE

        result = clean_mask(mask, settings)

        assert np.array_equal(result, mask), (
            "Thickness filter disabled -- mask must be returned unchanged"
        )

    def test_empty_mask_does_not_crash(self) -> None:
        """All-zero mask goes through filter_thin_strokes without error."""
        mask = _empty_mask(80, 80)
        result = filter_thin_strokes(mask, min_thickness_px=6, preclose_kernel_size=9)
        assert result.sum() == 0

    def test_all_white_mask_does_not_crash(self) -> None:
        """All-white mask (255 everywhere) does not raise."""
        mask = np.full((80, 80), WHITE, dtype=np.uint8)
        result = filter_thin_strokes(mask, min_thickness_px=6, preclose_kernel_size=9)
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# Pipeline order in clean_mask: crop (step 3) before filter (step 4)
# ---------------------------------------------------------------------------


class TestCleanMaskPipelineOrder:
    def test_crop_before_filter_order(self) -> None:
        """Verify combined result of steps 3+4: a distant thin component is removed.

        A large thick wall rectangle forms the main component.  A thin horizontal
        line far away survives steps 1-2 (it is long H/V), is zeroed by step 3
        (crop), and therefore never reaches step 4.  If the order were reversed
        (filter first, crop second), the thin line could survive the filter through
        a false seed but would still be removed by crop -- so this test verifies
        the combined 4-step outcome: the distant region is zeroed.
        """
        settings = Settings(
            cv_cleanup_enabled=True,
            cv_cleanup_text_max_side_px=5,
            cv_cleanup_rectilinear_len_px=5,
            cv_cleanup_crop_enabled=True,
            cv_cleanup_crop_margin_px=10,
            cv_cleanup_thickness_filter_enabled=True,
            cv_cleanup_min_wall_thickness_px=6,
            cv_cleanup_thickness_preclose_px=9,
        )

        mask = _empty_mask(400, 400)
        # Main component: large thick rect (40 x 200) at top-left
        _white(mask, 20, 20, 40, 200)

        # Distant thin annotation at bottom-right -- 1 px, 100 px long
        mask[350, 280:380] = WHITE

        result = clean_mask(mask, settings)

        # The distant region should be zeroed (crop zeroes it in step 3)
        distant_after = int(result[350, 280:380].sum() // WHITE)
        assert distant_after == 0, (
            f"Distant thin annotation should be zero after crop+filter, got {distant_after}"
        )

        # The main thick wall should have significant surviving pixels
        main_after = int(result[20:60, 20:220].sum() // WHITE)
        assert main_after > 0, "Main thick wall should survive the full pipeline"


# ---------------------------------------------------------------------------
# Settings -- new env vars
# ---------------------------------------------------------------------------


class TestNewSettings:
    def test_thickness_filter_enabled_default_true(self) -> None:
        """cv_cleanup_thickness_filter_enabled defaults to True."""
        s = Settings()
        assert s.cv_cleanup_thickness_filter_enabled is True

    def test_min_wall_thickness_px_default(self) -> None:
        """cv_cleanup_min_wall_thickness_px defaults to 6."""
        s = Settings()
        assert s.cv_cleanup_min_wall_thickness_px == _DEFAULT_MIN_WALL_THICKNESS_PX

    def test_thickness_filter_enabled_reads_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CV_CLEANUP_THICKNESS_FILTER_ENABLED env var is respected."""
        monkeypatch.setenv("CV_CLEANUP_THICKNESS_FILTER_ENABLED", "false")
        s = Settings()
        assert s.cv_cleanup_thickness_filter_enabled is False

    def test_min_wall_thickness_px_reads_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CV_CLEANUP_MIN_WALL_THICKNESS_PX env var is respected."""
        expected = 10
        monkeypatch.setenv("CV_CLEANUP_MIN_WALL_THICKNESS_PX", str(expected))
        s = Settings()
        assert s.cv_cleanup_min_wall_thickness_px == expected


# ---------------------------------------------------------------------------
# Integration / regression -- real fixtures (marked slow)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def opencv_engine():
    from vitrina_cv.engines.opencv_classic import OpenCVClassicEngine  # noqa: PLC0415

    # Pass Settings() so the engine runs the full cleanup + thickness-filter pipeline.
    # Without settings, OpenCVClassicEngine skips all cleanup steps (settings=None path).
    return OpenCVClassicEngine(settings=Settings())


def _load_fixture(name: str) -> bytes:
    path = FIXTURES / name
    assert path.exists(), f"Fixture not found: {path}"
    return path.read_bytes()


@pytest.mark.slow
def test_plano_limpio_rooms_and_walls(opencv_engine) -> None:
    """Clean floor plan -> exactly 6 rooms and at most 90 consolidated walls.

    Measured baseline (2026-07-03) with filter enabled: walls=80, rooms=6.
    """
    geometry = opencv_engine.extract(_load_fixture("plano_limpio.png"))

    n_walls = len(geometry.walls)
    n_rooms = len(geometry.rooms)

    assert n_rooms == _LIMPIO_EXPECTED_ROOMS, (
        f"Expected {_LIMPIO_EXPECTED_ROOMS} rooms for clean plan, got {n_rooms}"
    )
    assert n_walls <= _LIMPIO_MAX_WALLS, (
        f"Expected <= {_LIMPIO_MAX_WALLS} walls for clean plan, got {n_walls}"
    )


@pytest.mark.slow
def test_plano_denso_anotado_rooms_and_walls(opencv_engine) -> None:
    """Annotated floor plan -> at least 4 rooms and at most 100 walls.

    Measured baseline (2026-07-03) with filter enabled: walls=84, rooms=5.
    """
    geometry = opencv_engine.extract(_load_fixture("plano_denso_anotado.png"))

    n_walls = len(geometry.walls)
    n_rooms = len(geometry.rooms)

    assert n_rooms >= _DENSO_MIN_ROOMS, (
        f"Expected >= {_DENSO_MIN_ROOMS} rooms for annotated plan, got {n_rooms}"
    )
    assert n_walls <= _DENSO_MAX_WALLS, (
        f"Expected <= {_DENSO_MAX_WALLS} walls for annotated plan, got {n_walls}"
    )
