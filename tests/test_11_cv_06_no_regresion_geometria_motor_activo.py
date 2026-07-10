"""No-regresión geométrica e2e — 11-cv-06 (spec-cv-service.md, AC no-regresión).

Cierra el único gap no cubierto por las tasks 11-cv-02/03/04: confirma que la
suite de fidelidad geométrica de runs 07-10 (fixtures reales en
tests/eval/dataset) produce exactamente los mismos walls/rooms/openings/
stairs_candidates cuando el motor semántico está activo (CV_SEM_ENGINE=zeroshot)
que cuando está apagado. El motor semántico real (transformers/torch) se
mockea para que la suite sea rápida y determinista — no se testea la calidad
de la detección semántica aquí (eso ya está cubierto en
test_11_cv_02_zeroshot_engine.py), solo que su presencia no muta el track
geométrico (ADR-003).

AC-1/AC-2/AC-3/AC-4/AC-5/AC-6 de spec-cv-service.md ya están cubiertos en:
  - test_11_cv_02_zeroshot_engine.py (AC-3, motor zeroshot)
  - test_11_cv_03_merge_semantic_dedup.py (AC-4, AC-5)
  - test_11_cv_04_wire_objects_response.py (AC-1, AC-2, AC-6)
No se duplican aquí.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from vitrina_cv.config.settings import Settings
from vitrina_cv.engines.opencv_classic import OpenCVClassicEngine
from vitrina_cv.engines.semantic.merge import merge_semantic
from vitrina_cv.models import SemanticLabel, SemanticObject, SemanticSource

if TYPE_CHECKING:
    from vitrina_cv.models import Room, Wall

_DATASET = Path(__file__).resolve().parent.parent / "eval" / "dataset"

_FIXTURES = [
    "plan-001-denso-achurado",
    "plan-002-simple-limpio",
    "plan-003-reticula-cotas",
    "plan-004-sintetico-alta-res",
    "plan-005-amueblado-limpio",
]

# Mismos overrides que test_cv09_11_no_regresion_5_fixtures.py — replican el
# entorno real de cv-service (docker-compose-local.yml) para que este test
# offline sea un proxy fiel del comportamiento desplegado.
_REAL_SERVICE_SETTINGS_OVERRIDES: dict[str, object] = {
    "cv_cleanup_rectilinear_adaptive_enabled": False,
    "cv_cleanup_rectilinear_min_len_px": 50,
    "cv_cleanup_crop_min_area_ratio": 0.05,
    "cv_wall_diagonal_filter_low_deg": 10.0,
    "cv_wall_diagonal_filter_high_deg": 80.0,
    "cv_wall_min_diagonal_len_px": 40,
    "cv_junction_extend_px": 40,
}


def _real_service_settings() -> Settings:
    return Settings(**_REAL_SERVICE_SETTINGS_OVERRIDES)


def _fixture_paths(plan_id: str) -> tuple[Path, Path]:
    base = _DATASET / plan_id
    return base / "image.png", base / "ground_truth.json"


class _StubSemanticEngine:
    """Doble rápido de SemanticEngine — sin pesos reales, mismo patrón que
    test_11_cv_04_wire_objects_response.py."""

    is_ready = True

    def detect(
        self, image_bytes: bytes, rooms: list[Room], walls: list[Wall]
    ) -> list[SemanticObject]:
        del image_bytes, walls
        # Emite una detección arbitraria por fixture para forzar que
        # merge_semantic corra sobre el contexto real (rooms no vacíos
        # cuando el fixture produce al menos un room).
        return [
            SemanticObject(
                label=SemanticLabel.bed,
                bbox=(1.0, 1.0, 5.0, 5.0),
                confidence=0.9,
                needs_review=False,
                room_id=None,
                source=SemanticSource.zeroshot,
            )
        ]


@pytest.mark.slow
@pytest.mark.parametrize("plan_id", _FIXTURES)
def test_geometria_identica_con_motor_semantico_activo(plan_id: str) -> None:
    """Motor semántico activo no debe alterar walls/rooms/openings/stairs.

    Ejecuta el motor geométrico una sola vez (es la única fuente de verdad
    de ADR-003) y luego corre el track semántico (stub) + merge_semantic en
    paralelo, comparando los campos geométricos crudos antes/después de que
    el track semántico corra — deben ser idénticos byte a byte porque
    merge_semantic es puro y nunca toca walls/rooms/openings/stairs
    (test_11_cv_03_merge_semantic_dedup.py::TestMergeSemanticPurity ya
    prueba la pureza en aislamiento; este test la ejercita contra planos
    reales de runs 07-10 en vez de fixtures sintéticos mínimos).
    """
    image_path, gt_path = _fixture_paths(plan_id)
    if not image_path.exists() or not gt_path.exists():
        pytest.skip(f"fixture files missing for {plan_id}")

    image_bytes = image_path.read_bytes()

    engine = OpenCVClassicEngine(settings=_real_service_settings())
    geometry = engine.extract(image_bytes)

    walls_before = [w.model_copy(deep=True) for w in geometry.walls]
    rooms_before = [r.model_copy(deep=True) for r in geometry.rooms]
    openings_before = [o.model_copy(deep=True) for o in geometry.openings]
    stairs_before = [s.model_copy(deep=True) for s in geometry.stairs_candidates]

    semantic_engine = _StubSemanticEngine()
    semantic_objects = semantic_engine.detect(
        image_bytes, rooms=geometry.rooms, walls=geometry.walls
    )
    merge_semantic(
        objects=semantic_objects,
        rooms=geometry.rooms,
        walls=geometry.walls,
        openings=geometry.openings,
    )

    assert geometry.walls == walls_before, (
        f"AC no-regresión ({plan_id}): walls mutado por el track semántico"
    )
    assert geometry.rooms == rooms_before, (
        f"AC no-regresión ({plan_id}): rooms mutado por el track semántico"
    )
    assert geometry.openings == openings_before, (
        f"AC no-regresión ({plan_id}): openings mutado por el track semántico"
    )
    assert geometry.stairs_candidates == stairs_before, (
        f"AC no-regresión ({plan_id}): stairs_candidates mutado por el track semántico"
    )


@pytest.mark.slow
def test_room_count_sigue_vigente_con_motor_activo() -> None:
    """Sanity check adicional: el conteo de rooms reportado por la suite
    CV09-11 (baseline geométrico) no cambia si se ejecuta el mismo pipeline
    en un contexto donde luego se invoca el track semántico (mismo objeto
    Geometry, no una segunda extracción) — confirma que no hay aliasing ni
    reprocesamiento accidental de la imagen vía el track semántico.
    """
    rows: list[tuple[str, int, int]] = []

    for plan_id in _FIXTURES:
        image_path, gt_path = _fixture_paths(plan_id)
        if not image_path.exists() or not gt_path.exists():
            continue

        ground_truth = json.loads(gt_path.read_text())
        expected_rooms = ground_truth["expected_rooms"]

        engine = OpenCVClassicEngine(settings=_real_service_settings())
        geometry = engine.extract(image_path.read_bytes())
        n_rooms_before = len(geometry.rooms)

        semantic_engine = _StubSemanticEngine()
        semantic_objects = semantic_engine.detect(
            image_path.read_bytes(), rooms=geometry.rooms, walls=geometry.walls
        )
        merge_semantic(
            objects=semantic_objects,
            rooms=geometry.rooms,
            walls=geometry.walls,
            openings=geometry.openings,
        )

        n_rooms_after = len(geometry.rooms)
        assert n_rooms_after == n_rooms_before, (
            f"{plan_id}: room count cambió de {n_rooms_before} a "
            f"{n_rooms_after} tras correr el track semántico"
        )
        rows.append((plan_id, expected_rooms, n_rooms_after))

    assert rows, "no fixtures available to report on"
