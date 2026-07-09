"""Tests for CV09-04 — diagonal residual filter, pass 2 (ADR-017).

Covers:
  Mec.1 (angle re-filter): wall at 45° after fuse_junctions is discarded.
  Mec.2 (min length): oblique wall shorter than cv_wall_min_diagonal_len_px is
    discarded; an exact H/V wall of the same short length is kept regardless.
  AC-11: flag off → pass 2 is a no-op (output identical to input).
  AC-12: rectilinear planta (no diagonal residual) → zero discards in both
    mechanisms.
  Idempotence: H/V-exact walls (angle 0°/90°) never fall in the discard band.
"""

from __future__ import annotations

import math

from vitrina_cv.config.settings import Settings
from vitrina_cv.engines.opencv_classic import _filter_diagonal_residual_pass2
from vitrina_cv.models import Wall


def _wall_from_angle(angle_deg: float, length: float = 100.0) -> Wall:
    """Create a Wall whose atan2(|dy|,|dx|) equals *angle_deg*."""
    rad = math.radians(angle_deg)
    dx = length * math.cos(rad)
    dy = length * math.sin(rad)
    return Wall(start=(0.0, 0.0), end=(dx, dy))


def _settings_on(
    low: float = 20.0,
    high: float = 70.0,
    min_len_px: int = 40,
) -> Settings:
    return Settings(
        cv_wall_diagonal_filter_enabled=True,
        cv_wall_diagonal_filter_low_deg=low,
        cv_wall_diagonal_filter_high_deg=high,
        cv_wall_min_diagonal_len_px=min_len_px,
    )


def _settings_off() -> Settings:
    return Settings(cv_wall_diagonal_filter_enabled=False)


class TestMec1AngleRefilter:
    def test_45_degree_wall_discarded_after_fuse(self) -> None:
        """AC-9: a 45° wall surviving to pass 2 falls in [20,70] → discarded."""
        wall_45 = _wall_from_angle(45.0)
        result = _filter_diagonal_residual_pass2([wall_45], _settings_on())
        assert result == []

    def test_exact_horizontal_wall_never_discarded(self) -> None:
        """Idempotence: H-exact wall (angle 0°) never falls in [20,70]."""
        wall_h = Wall(start=(0.0, 50.0), end=(200.0, 50.0))
        result = _filter_diagonal_residual_pass2([wall_h], _settings_on())
        assert result == [wall_h]

    def test_exact_vertical_wall_never_discarded(self) -> None:
        """Idempotence: V-exact wall (angle 90°) never falls in [20,70]."""
        wall_v = Wall(start=(50.0, 0.0), end=(50.0, 200.0))
        result = _filter_diagonal_residual_pass2([wall_v], _settings_on())
        assert result == [wall_v]

    def test_wall_outside_band_is_kept(self) -> None:
        """15° is below low_deg=20 — outside the discard band."""
        wall_15 = _wall_from_angle(15.0, length=100.0)
        result = _filter_diagonal_residual_pass2([wall_15], _settings_on())
        assert result == [wall_15]


class TestMec2MinLength:
    def test_short_oblique_wall_discarded(self) -> None:
        """AC-10: non-H/V wall shorter than cv_wall_min_diagonal_len_px is discarded."""
        # 12° is outside the [20,70] angle band (Mec.1 doesn't touch it) but is
        # not H/V-exact, so Mec.2 applies.
        short_oblique = _wall_from_angle(12.0, length=30.0)
        result = _filter_diagonal_residual_pass2(
            [short_oblique], _settings_on(min_len_px=40)
        )
        assert result == []

    def test_exact_hv_wall_kept_regardless_of_length(self) -> None:
        """AC-10: an H/V-exact wall of 30px (< min_len_px) is NOT a Mec.2 candidate."""
        short_hv = Wall(start=(0.0, 0.0), end=(30.0, 0.0))
        result = _filter_diagonal_residual_pass2(
            [short_hv], _settings_on(min_len_px=40)
        )
        assert result == [short_hv]

    def test_long_oblique_wall_kept(self) -> None:
        """Oblique wall with length >= min_len_px survives Mec.2."""
        long_oblique = _wall_from_angle(12.0, length=100.0)
        result = _filter_diagonal_residual_pass2(
            [long_oblique], _settings_on(min_len_px=40)
        )
        assert result == [long_oblique]


class TestFlagGating:
    def test_flag_off_is_noop(self) -> None:
        """AC-11: flag off → pass 2 output is byte-identical to input."""
        wall_45 = _wall_from_angle(45.0)
        wall_h = Wall(start=(0.0, 50.0), end=(200.0, 50.0))
        short_oblique = _wall_from_angle(12.0, length=5.0)
        walls_in = [wall_45, wall_h, short_oblique]

        result = _filter_diagonal_residual_pass2(walls_in, _settings_off())

        assert result is walls_in

    def test_settings_none_is_noop(self) -> None:
        """settings=None → no-op, same as flag off."""
        wall_45 = _wall_from_angle(45.0)
        walls_in = [wall_45]
        result = _filter_diagonal_residual_pass2(walls_in, None)
        assert result is walls_in


class TestRectilinearNoOp:
    def test_rectilinear_plan_zero_discards(self, caplog) -> None:  # type: ignore[no-untyped-def]
        """AC-12: rectilinear plan (only H/V-exact walls) → zero discards both mechanisms."""
        walls_in = [
            Wall(start=(0.0, 0.0), end=(200.0, 0.0)),
            Wall(start=(0.0, 0.0), end=(0.0, 200.0)),
            Wall(start=(200.0, 0.0), end=(200.0, 200.0)),
            Wall(start=(0.0, 200.0), end=(200.0, 200.0)),
        ]
        with caplog.at_level("INFO"):
            result = _filter_diagonal_residual_pass2(walls_in, _settings_on())

        assert result == walls_in

        matching = [
            record
            for record in caplog.records
            if record.message == "cv_wall_diagonal_pass2_filtered"
        ]
        assert len(matching) == 1
        assert matching[0].count_by_angle == 0
        assert matching[0].count_by_length == 0
