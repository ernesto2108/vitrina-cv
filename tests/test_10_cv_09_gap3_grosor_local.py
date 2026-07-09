"""Tests for task 10-cv-09 — gap 3 completo (ADR-003) y anti-regresion.

Cubre AC-7, AC-8, AC-9 y AC-11 del spec
(docs/specs/10-fidelidad-cv-rooms-diagonal/spec-cv-service.md), usando los
3 fixtures reales del dataset y los mismos overrides de settings que
replican docker-compose-local.yml (ver test_cv09_11_no_regresion_5_fixtures.py
para el precedente de este patron).

AC-10 no aplica: 10-cv-07 verifico que la fusion de room_9 ya no existe con
(A)+(A2) activos, por lo que 10-cv-08 se marco "NO APLICA" y no hay
`rooms_split_by_junction_close` que testear.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from vitrina_cv.config.settings import Settings
from vitrina_cv.engines.opencv_classic import (
    OpenCVClassicEngine,
    _nearest_perpendicular_wall_distance,
)

_DATASET = Path(__file__).resolve().parent.parent / "eval" / "dataset"

# Mismos overrides que tests/test_cv09_11_no_regresion_5_fixtures.py, sourced
# de docker-compose-local.yml (servicio cv-service, bloque `environment:`).
_REAL_SERVICE_SETTINGS_OVERRIDES: dict[str, Any] = {
    "cv_cleanup_rectilinear_adaptive_enabled": False,
    "cv_cleanup_rectilinear_min_len_px": 50,
    "cv_cleanup_crop_min_area_ratio": 0.05,
    "cv_wall_diagonal_filter_low_deg": 10.0,
    "cv_wall_diagonal_filter_high_deg": 80.0,
    "cv_wall_min_diagonal_len_px": 40,
    "cv_junction_extend_px": 40,
}

# Areas (px^2) medidas en 10-cv-07 para los 9 rooms de plan-001 con los
# flags de gap 3 (grosor local + junction extend adaptativa) en su default
# true. Ninguna de estas areas trae firma de fusion de 2 ambientes.
_PLAN_001_EXPECTED_ROOM_AREAS_PX2 = {
    298224,
    99472,
    56672,
    54318,
    51971,
    45615,
    21501,
    20188,
    16328,
}


def _real_service_settings(**extra: Any) -> Settings:
    return Settings(**_REAL_SERVICE_SETTINGS_OVERRIDES, **extra)


def _fixture_paths(plan_id: str) -> tuple[Path, Path]:
    base = _DATASET / plan_id
    return base / "image.png", base / "ground_truth.json"


def _require_fixture(plan_id: str) -> tuple[Path, Path]:
    image_path, gt_path = _fixture_paths(plan_id)
    if not image_path.exists() or not gt_path.exists():
        pytest.skip(f"fixture files missing for {plan_id}")
    return image_path, gt_path


@pytest.mark.slow
class TestGap3LocalThickness:
    """AC-7: contador de grosor local se dispara con plan-001."""

    def test_walls_merged_local_thickness_positive_on_plan001(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """AC-7: con CV_WALL_LOCAL_THICKNESS_ENABLED=true (default), el
        agrupamiento por grosor local en _consolidate_walls debe reportar al
        menos 1 merge real via el log INFO walls_merged_local_thickness."""
        image_path, _ = _require_fixture("plan-001-denso-achurado")

        settings = _real_service_settings(cv_wall_local_thickness_enabled=True)
        engine = OpenCVClassicEngine(settings=settings)

        with caplog.at_level(logging.INFO, logger="vitrina_cv.engines.opencv_classic"):
            engine.extract(image_path.read_bytes())

        merge_records = [
            r for r in caplog.records if r.message == "walls_merged_local_thickness"
        ]
        assert merge_records, (
            "expected at least one walls_merged_local_thickness log record "
            "when the local-thickness flag is enabled"
        )
        total_merges = sum(getattr(r, "count", 0) for r in merge_records)
        assert total_merges > 0, (
            f"walls_merged_local_thickness counter must be > 0 on plan-001, "
            f"got {total_merges}"
        )


class TestGap3JunctionExtendAdaptiveCap:
    """AC-8: recorte adaptativo de extension de junction (escenario sintetico).

    `plan-001-denso-achurado` con los overrides reales de
    docker-compose-local.yml no contiene un caso donde un tercer muro
    perpendicular este mas cerca que la interseccion esperada (confirmado
    por developer-backend, 107 candidatos revisados tras el fix de
    exclusion de self_idx/pair_idx). Estos tests aislan
    `_nearest_perpendicular_wall_distance` con muros construidos a mano
    para probar el mecanismo de recorte sin depender de un fixture PNG que
    no dispara el caso.
    """

    def test_nearest_perpendicular_wall_caps_extension_when_third_wall_is_closer(
        self,
    ) -> None:
        """AC-8: cuando un tercer muro perpendicular cruza el camino de
        extension mas cerca que la interseccion geometrica (`target`), el
        recorte adaptativo debe reportar esa distancia (no `None`) y debe
        ser menor a la distancia hasta `target`.

        Escenario sintetico (no fixture real): tras la investigacion del
        developer-backend, `plan-001-denso-achurado` con los overrides
        reales no contiene ningun caso donde un tercer muro perpendicular
        este mas cerca que la interseccion esperada (107 candidatos
        revisados, mecanismo verificado correcto pero sin disparo en ese
        fixture). Este test unitario aisla `_nearest_perpendicular_wall_distance`
        directamente para probar el caso de recorte sin depender de que un
        fixture PNG particular lo contenga.

        Geometria: se extiende un muro horizontal (wall 0, self_idx=0) a lo
        largo de x, desde from_value=100 hacia target=200 (la interseccion
        con el muro vertical wall 1, pair_idx=1, en x=200). Un tercer muro
        vertical (wall 2, no excluido) cruza x=150 (mas cerca que target) y
        su rango y cubre fixed_pos=50 (la y del muro horizontal). Se espera
        que la funcion retorne 150 - 100 = 50, no None, y que sea menor que
        target - from_value = 100.
        """
        # wall 0: horizontal, siendo extendido (self_idx=0) — no se usa su
        # contenido para el calculo, solo se excluye del scan.
        wall_0 = [0.0, 50.0, 100.0, 50.0]
        # wall 1: el muro perpendicular que origino target=200 (pair_idx=1)
        # — excluido explicitamente del scan segun el fix de hoy.
        wall_1 = [200.0, 0.0, 200.0, 100.0]
        # wall 2: tercer muro vertical que SI debe limitar la extension,
        # cruzando x=150 con span y [0, 100] que cubre fixed_pos=50.
        wall_2 = [150.0, 0.0, 150.0, 100.0]

        coords = [wall_0, wall_1, wall_2]

        nearest = _nearest_perpendicular_wall_distance(
            coords,
            self_idx=0,
            axis=0,
            fixed_pos=50.0,
            from_value=100.0,
            target=200.0,
            pair_idx=1,
        )

        assert nearest is not None, (
            "expected the third perpendicular wall (wall 2, x=150) to cap "
            "the extension — got None"
        )
        assert nearest == pytest.approx(50.0), (
            f"expected cap distance 150-100=50, got {nearest}"
        )
        assert nearest < (200.0 - 100.0), (
            "capped distance must be strictly less than the distance to the "
            "geometric intersection (target) for AC-8 to represent a real cap"
        )

    def test_nearest_perpendicular_wall_excludes_self_and_pair_idx(self) -> None:
        """Regresion del fix de hoy en developer-backend: `self_idx` y
        `pair_idx` deben excluirse del scan. Sin exclusion de `pair_idx`,
        wall 1 (que produce target=200) se contaria a si mismo como el
        'tercer muro', devolviendo una distancia espuria en vez de None."""
        wall_0 = [0.0, 50.0, 100.0, 50.0]
        wall_1 = [200.0, 0.0, 200.0, 100.0]
        coords = [wall_0, wall_1]

        nearest = _nearest_perpendicular_wall_distance(
            coords,
            self_idx=0,
            axis=0,
            fixed_pos=50.0,
            from_value=100.0,
            target=200.0,
            pair_idx=1,
        )

        assert nearest is None, (
            "with only self and pair_idx present (no third wall), the scan "
            "must find no capping candidate"
        )


@pytest.mark.slow
class TestGap3NoRoomMergeSignature:
    """AC-9: ningun room de plan-001 muestra firma de fusion CCA."""

    def test_plan001_rooms_count_and_areas_match_no_merge_signature(self) -> None:
        """Con (A) grosor local + (A2) junction extend adaptativa activos
        (defaults reales), plan-001 debe producir 9 rooms cuyas areas
        coinciden con las medidas en 10-cv-07 (ninguna sugiere la fusion de
        2 ambientes, ej. la extinta 'room_9' fusionada)."""
        image_path, gt_path = _require_fixture("plan-001-denso-achurado")

        engine = OpenCVClassicEngine(settings=_real_service_settings())
        geometry = engine.extract(image_path.read_bytes())

        assert len(geometry.rooms) == 9, (
            f"expected 9 separate rooms (no CCA merge) on plan-001, "
            f"got {len(geometry.rooms)}"
        )

        obtained_areas = {round(room.area_px) for room in geometry.rooms}

        assert obtained_areas == _PLAN_001_EXPECTED_ROOM_AREAS_PX2, (
            f"room areas diverge from the 10-cv-07 baseline (no-merge "
            f"signature): expected {_PLAN_001_EXPECTED_ROOM_AREAS_PX2}, "
            f"got {obtained_areas}"
        )


@pytest.mark.slow
class TestGap3AntiRegresionOtrosFixtures:
    """AC-11: plan-002 y plan-003 sin cambios con los flags de gap 3 activos."""

    @pytest.mark.parametrize(
        ("plan_id", "expected_walls", "expected_rooms"),
        [
            ("plan-002-simple-limpio", 26, 6),
            ("plan-003-reticula-cotas", 85, 9),
        ],
    )
    def test_walls_and_rooms_unchanged_with_gap3_flags_on(
        self, plan_id: str, expected_walls: int, expected_rooms: int
    ) -> None:
        image_path, _ = _require_fixture(plan_id)

        engine = OpenCVClassicEngine(settings=_real_service_settings())
        geometry = engine.extract(image_path.read_bytes())

        assert len(geometry.walls) == expected_walls, (
            f"{plan_id}: expected {expected_walls} walls with gap-3 flags "
            f"on (anti-regresion), got {len(geometry.walls)}"
        )
        assert len(geometry.rooms) == expected_rooms, (
            f"{plan_id}: expected {expected_rooms} rooms with gap-3 flags "
            f"on (anti-regresion), got {len(geometry.rooms)}"
        )
