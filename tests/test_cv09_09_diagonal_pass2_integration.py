"""Integration tests for CV09-04 — diagonal residual filter, pass 2 (ADR-017).

Runs the real OpenCVClassicEngine end-to-end against images in
eval/dataset/, confirming:

  AC-9/AC-10 (real images): the residual diagonal triangle disappears from
    the wall output on plans known to exhibit it after fuse_junctions
    (plan-002-simple-limpio, plan-004-sintetico-alta-res,
    plan-005-amueblado-limpio).

  AC-12: on a rectilinear plan (plan-003-reticula-cotas, no diagonal
    residual), pass 2 is a true no-op — zero discards in both mechanisms
    and the wall count is unchanged relative to the flag being disabled.

These complement the unit tests in test_cv09_04_diagonal_pass2.py, which
exercise _filter_diagonal_residual_pass2 directly on synthetic Wall lists.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from vitrina_cv.config.settings import Settings
from vitrina_cv.engines.opencv_classic import OpenCVClassicEngine

_DATASET = Path(__file__).resolve().parent.parent / "eval" / "dataset"

_DIAGONAL_RESIDUAL_PLANS = [
    "plan-002-simple-limpio",
    "plan-004-sintetico-alta-res",
    "plan-005-amueblado-limpio",
]

_RECTILINEAR_PLAN = "plan-003-reticula-cotas"

# Same band the unit tests use (matches Settings defaults per CV09-04 spec).
_LOW_DEG = 20.0
_HIGH_DEG = 70.0


def _wall_angle_deg(wall) -> float:  # type: ignore[no-untyped-def]
    dx = abs(wall.end[0] - wall.start[0])
    dy = abs(wall.end[1] - wall.start[1])
    return math.degrees(math.atan2(dy, dx))


def _load_image_bytes(plan_id: str) -> bytes:
    image_path = _DATASET / plan_id / "image.png"
    assert image_path.exists(), f"missing fixture image for {plan_id}"
    return image_path.read_bytes()


@pytest.mark.parametrize("plan_id", _DIAGONAL_RESIDUAL_PLANS)
class TestDiagonalResidualDisappears:
    """AC-9/AC-10 on real fixtures: no wall remains inside the discard band
    after fuse_junctions, once the diagonal filter (pass 1 + pass 2) runs
    with default settings.
    """

    def test_no_wall_in_discard_band(self, plan_id: str) -> None:
        settings = Settings(cv_wall_diagonal_filter_enabled=True)
        engine = OpenCVClassicEngine(settings=settings)
        image_bytes = _load_image_bytes(plan_id)

        geometry = engine.extract(image_bytes)

        offending = [
            wall
            for wall in geometry.walls
            if _LOW_DEG <= _wall_angle_deg(wall) <= _HIGH_DEG
        ]
        assert offending == [], (
            f"{plan_id}: residual diagonal wall(s) survived pass 2: {offending}"
        )


class TestRectilinearPlanIsNoOp:
    """AC-12: plan-003-reticula-cotas has no diagonal residual — pass 2 must
    discard zero walls via either mechanism, and the wall count with the
    filter enabled must match the count with the filter disabled.
    """

    def test_zero_discards_both_mechanisms(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        settings = Settings(cv_wall_diagonal_filter_enabled=True)
        engine = OpenCVClassicEngine(settings=settings)
        image_bytes = _load_image_bytes(_RECTILINEAR_PLAN)

        with caplog.at_level("INFO"):
            engine.extract(image_bytes)

        matching = [
            record
            for record in caplog.records
            if record.message == "cv_wall_diagonal_pass2_filtered"
        ]
        assert len(matching) == 1, (
            "expected exactly one cv_wall_diagonal_pass2_filtered log event, "
            f"got {len(matching)}"
        )
        assert matching[0].count_by_angle == 0  # type: ignore[attr-defined]
        assert matching[0].count_by_length == 0  # type: ignore[attr-defined]

    def test_wall_count_unchanged_vs_flag_disabled(self) -> None:
        image_bytes = _load_image_bytes(_RECTILINEAR_PLAN)

        engine_on = OpenCVClassicEngine(
            settings=Settings(cv_wall_diagonal_filter_enabled=True)
        )
        engine_off = OpenCVClassicEngine(
            settings=Settings(cv_wall_diagonal_filter_enabled=False)
        )

        walls_on = engine_on.extract(image_bytes).walls
        walls_off = engine_off.extract(image_bytes).walls

        assert len(walls_on) == len(walls_off)
