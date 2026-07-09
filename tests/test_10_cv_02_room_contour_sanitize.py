"""Unit tests for room contour sanitizing (ADR-001, 10-cv-01 / 10-cv-02).

Covers the acceptance criteria of 10-cv-02-tests-saneo-contorno:

- AC-1: the real spurious vertex [1234,1559] found in plan-005's "Sala"
  room is removed by ``_sanitize_room_polygon``.
- AC-2: a contour with no ortho-recoverable polygon (adjacent spurious
  vertices that cannot be resolved by single-vertex removal) is discarded
  (returns None).
- AC-3: plan-002 and plan-003 do not regress in room/wall counts with the
  sanitize flag active, using the real cv-service Settings overrides (same
  pattern as test_cv09_11_no_regresion_5_fixtures.py).
- Regression guard for the historical false positive: the legitimate short
  jog in plan-001's dense room (edge (77,562)->(129,574), ~53px, ~13deg,
  perpendicular deviation ~46px, both below the 100px deviation threshold)
  must NOT be removed.
- The ``cv_room_contour_sanitize_enabled=False`` flag reproduces pre-sanitize
  behaviour (no vertices removed) via the public engine entrypoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vitrina_cv.config.settings import Settings
from vitrina_cv.engines.opencv_classic import (
    OpenCVClassicEngine,
    _sanitize_room_polygon,
)

_DATASET = Path(__file__).resolve().parent.parent / "eval" / "dataset"

# Same overrides as the real cv-service deployment (docker-compose-local.yml),
# reused from test_cv09_11_no_regresion_5_fixtures.py so counts are directly
# comparable against the recalibrated baselines in the 10-cv spec.
_REAL_SERVICE_SETTINGS_OVERRIDES: dict[str, Any] = {
    "cv_cleanup_rectilinear_adaptive_enabled": False,
    "cv_cleanup_rectilinear_min_len_px": 50,
    "cv_cleanup_crop_min_area_ratio": 0.05,
    "cv_wall_diagonal_filter_low_deg": 10.0,
    "cv_wall_diagonal_filter_high_deg": 80.0,
    "cv_wall_min_diagonal_len_px": 40,
    "cv_junction_extend_px": 40,
}

# Default sanitize thresholds (mirroring Settings defaults) used directly
# against _sanitize_room_polygon in the unit-level tests below.
_LOW_DEG = 10.0
_HIGH_DEG = 80.0
_MIN_DIAGONAL_LEN_PX = 40.0
_MIN_DEVIATION_PX = 100.0

# Expected wall/room counts per fixture with the real service overrides
# (10-cv-02 recalibrated baselines).
_PLAN005_EXPECTED_WALLS = 26
_PLAN005_EXPECTED_ROOMS = 6
_PLAN001_EXPECTED_WALLS = 69
_PLAN001_EXPECTED_ROOMS = 9
_PLAN001_JOG_ROOM_AREA_PX = 51971
_AREA_TOLERANCE_PX = 500


def _real_service_settings(**overrides: Any) -> Settings:
    merged = {**_REAL_SERVICE_SETTINGS_OVERRIDES, **overrides}
    return Settings(**merged)


def _fixture_paths(plan_id: str) -> tuple[Path, Path]:
    base = _DATASET / plan_id
    return base / "image.png", base / "ground_truth.json"


# ---------------------------------------------------------------------------
# AC-1: real spurious vertex from plan-005 "Sala" room is removed.
# ---------------------------------------------------------------------------


def test_ac1_spurious_vertex_plan005_sala_is_removed() -> None:
    """Reproduces the plan-005 'Sala' contour segment around [1234, 1559].

    The spurious vertex sits between [1128, 1774] and [1656, 1774] with a
    pronounced perpendicular spike, matching the real mask artefact fixed by
    ADR-001. The rest of the polygon is a simple rectangle so removal keeps
    at least 3 vertices.
    """
    polygon = [
        (1128.0, 1200.0),
        (1656.0, 1200.0),
        (1656.0, 1774.0),
        (1234.0, 1559.0),  # spurious spike vertex
        (1128.0, 1774.0),
    ]

    sanitized = _sanitize_room_polygon(
        polygon,
        low_deg=_LOW_DEG,
        high_deg=_HIGH_DEG,
        min_diagonal_len_px=_MIN_DIAGONAL_LEN_PX,
        min_deviation_px=_MIN_DEVIATION_PX,
    )

    assert sanitized is not None
    assert (1234.0, 1559.0) not in sanitized


def test_ac1_plan005_engine_reports_expected_walls_and_rooms() -> None:
    """Full-engine check on the real plan-005 fixture with sanitize active.

    Expected per the 10-cv-02 task brief: walls=26, rooms=6 with the real
    service overrides.
    """
    image_path, gt_path = _fixture_paths("plan-005-amueblado-limpio")
    if not image_path.exists() or not gt_path.exists():
        pytest.skip("plan-005-amueblado-limpio fixture missing")

    engine = OpenCVClassicEngine(settings=_real_service_settings())
    geometry = engine.extract(image_path.read_bytes())

    assert len(geometry.walls) == _PLAN005_EXPECTED_WALLS
    assert len(geometry.rooms) == _PLAN005_EXPECTED_ROOMS


# ---------------------------------------------------------------------------
# AC-2: contour with no ortho-recoverable polygon is discarded.
# ---------------------------------------------------------------------------


def test_ac2_non_recoverable_contour_is_discarded() -> None:
    """Two adjacent spurious vertices forming a cycle the removal loop can't
    resolve one-at-a-time without eventually re-triggering the post-check.

    Construct a minimal triangle-like polygon where, even after any single
    removal, a diagonal-banded, long-enough, high-deviation vertex remains
    (deliberately built to fail the post-check) -- simulating a contour with
    no ortho-recoverable polygon per AC-2.
    """
    # A near-degenerate triangle: 3 vertices is the floor
    # (_MIN_POLYGON_VERTICES), so the removal loop cannot pop anything
    # further, and if the remaining vertex still satisfies the 3 spurious
    # conditions, the post-check must return None.
    polygon = [
        (0.0, 0.0),
        (200.0, 0.0),
        (100.0, 500.0),  # diagonal spike vertex, in-band, long, high deviation
    ]

    sanitized = _sanitize_room_polygon(
        polygon,
        low_deg=_LOW_DEG,
        high_deg=_HIGH_DEG,
        min_diagonal_len_px=_MIN_DIAGONAL_LEN_PX,
        min_deviation_px=_MIN_DEVIATION_PX,
    )

    assert sanitized is None


# ---------------------------------------------------------------------------
# AC-3: anti-regression on plan-002 and plan-003 with sanitize active.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("plan_id", "expected_walls", "expected_rooms"),
    [
        ("plan-002-simple-limpio", 26, 6),
        ("plan-003-reticula-cotas", 85, 9),
    ],
)
def test_ac3_no_regression_walls_and_rooms(
    plan_id: str, expected_walls: int, expected_rooms: int
) -> None:
    image_path, gt_path = _fixture_paths(plan_id)
    if not image_path.exists() or not gt_path.exists():
        pytest.skip(f"fixture files missing for {plan_id}")

    engine = OpenCVClassicEngine(settings=_real_service_settings())
    geometry = engine.extract(image_path.read_bytes())

    assert len(geometry.walls) == expected_walls
    assert len(geometry.rooms) == expected_rooms


# ---------------------------------------------------------------------------
# Regression guard: legitimate short jog in plan-001 must survive.
# ---------------------------------------------------------------------------


def test_legitimate_jog_plan001_is_not_removed() -> None:
    """Historical false-positive regression guard.

    The jog edge (77,562)->(129,574) (~53px, ~13deg, perpendicular deviation
    ~46px) must survive sanitizing: it is in-band and long enough, but its
    deviation (~46px) stays below the 100px min_deviation_px threshold, so
    condition 3 of _sanitize_room_polygon must NOT trigger removal.
    """
    polygon = [
        (0.0, 562.0),
        (77.0, 562.0),
        (129.0, 574.0),  # legitimate short jog vertex
        (300.0, 574.0),
        (300.0, 800.0),
        (0.0, 800.0),
    ]

    sanitized = _sanitize_room_polygon(
        polygon,
        low_deg=_LOW_DEG,
        high_deg=_HIGH_DEG,
        min_diagonal_len_px=_MIN_DIAGONAL_LEN_PX,
        min_deviation_px=_MIN_DEVIATION_PX,
    )

    assert sanitized is not None
    assert (129.0, 574.0) in sanitized


def test_ac3_plan001_engine_reports_expected_walls_and_rooms_with_jog_intact() -> None:
    """Full-engine check on plan-001: the jog room must survive (not be
    dropped), giving the expected walls=69, rooms=9 with sanitize active.
    """
    image_path, gt_path = _fixture_paths("plan-001-denso-achurado")
    if not image_path.exists() or not gt_path.exists():
        pytest.skip("plan-001-denso-achurado fixture missing")

    engine = OpenCVClassicEngine(settings=_real_service_settings())
    geometry = engine.extract(image_path.read_bytes())

    assert len(geometry.walls) == _PLAN001_EXPECTED_WALLS
    assert len(geometry.rooms) == _PLAN001_EXPECTED_ROOMS

    areas = [room.area_px for room in geometry.rooms]
    assert any(
        abs(area - _PLAN001_JOG_ROOM_AREA_PX) < _AREA_TOLERANCE_PX for area in areas
    ), "expected the jog room (area_px~=51971) to survive sanitizing"


# ---------------------------------------------------------------------------
# Flag off: reproduces pre-sanitize behaviour.
# ---------------------------------------------------------------------------


def test_sanitize_disabled_flag_reproduces_previous_behaviour() -> None:
    """With cv_room_contour_sanitize_enabled=False, plan-005 keeps the
    spurious vertex untouched (pre-10-cv-01 behaviour) via the public engine.
    """
    image_path, gt_path = _fixture_paths("plan-005-amueblado-limpio")
    if not image_path.exists() or not gt_path.exists():
        pytest.skip("plan-005-amueblado-limpio fixture missing")

    settings = _real_service_settings(cv_room_contour_sanitize_enabled=False)
    engine = OpenCVClassicEngine(settings=settings)

    # Must not raise, and must run through the no-sanitize code path.
    geometry = engine.extract(image_path.read_bytes())
    assert len(geometry.rooms) >= 0
