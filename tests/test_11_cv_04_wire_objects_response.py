"""Tests de integración — 11-cv-04-wire-objects-response.

Cubre los criterios de aceptación de la task:
  AC-1: CV_SEM_ENGINE=zeroshot (mockeado) -> POST /extract-geometry incluye objects[].
  AC-2: CV_SEM_ENGINE off/vacío -> objects: [] y el resto de la geometría idéntica
        al baseline (sin motor semántico wireado).
  AC-6 (parcial): GET /health expone semantic_model_loaded reflejando is_ready
        del motor semántico cuando está configurado; no afecta model_loaded
        del motor geométrico.
  Best-effort: una excepción en el motor semántico degrada a objects: []
        sin romper la respuesta 200 de /extract-geometry.

Todas las imágenes son PNGs sintéticos generados en memoria (mismo patrón que
tests/test_api_suite.py) — sin red externa, sin archivos en disco.
"""

from __future__ import annotations

import io
from http import HTTPStatus
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, PropertyMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from vitrina_cv.main import create_app
from vitrina_cv.models import SemanticLabel, SemanticObject, SemanticSource

if TYPE_CHECKING:
    from vitrina_cv.models import Room, Wall

# ---------------------------------------------------------------------------
# Helpers — mismos fixtures sintéticos que test_api_suite.py
# ---------------------------------------------------------------------------


def _png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok, "cv2.imencode falló en fixture"
    return buf.tobytes()


def _floor_plan_png(width: int = 1200, height: int = 900) -> bytes:
    """Plano con grid de celdas grandes (300x300px) — produce rooms > 0."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    lw = 3
    for x in range(0, width, 300):
        cv2.line(img, (x, 0), (x, height), (0, 0, 0), lw)
    for y in range(0, height, 300):
        cv2.line(img, (0, y), (width, y), (0, 0, 0), lw)
    return _png_bytes(img)


def _multipart(image_bytes: bytes, filename: str = "plano.png") -> dict[str, Any]:
    return {"image": (filename, io.BytesIO(image_bytes), "image/png")}


def _fake_semantic_object() -> SemanticObject:
    return SemanticObject(
        label=SemanticLabel.bed,
        bbox=(10.0, 10.0, 50.0, 80.0),
        confidence=0.75,
        needs_review=False,
        room_id=None,
        source=SemanticSource.zeroshot,
    )


class _StubSemanticEngine:
    """Doble de SemanticEngine para tests — evita cargar pesos reales."""

    def __init__(self, *, ready: bool = True, raise_on_detect: bool = False) -> None:
        self._ready = ready
        self._raise_on_detect = raise_on_detect

    @property
    def is_ready(self) -> bool:
        return self._ready

    def detect(
        self, image_bytes: bytes, rooms: list[Room], walls: list[Wall]
    ) -> list[SemanticObject]:
        del image_bytes, rooms, walls
        if self._raise_on_detect:
            msg = "boom — simulated inference failure"
            raise RuntimeError(msg)
        return [_fake_semantic_object()]


@pytest.fixture
def client() -> TestClient:
    """TestClient con motor geométrico real; semantic_engine se inyecta por test."""
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# AC-1 — CV_SEM_ENGINE=zeroshot (mockeado): objects[] en la respuesta
# ---------------------------------------------------------------------------


class TestAC1MotorSemanticoActivo:
    def test_objects_presente_con_motor_activo(self, client: TestClient) -> None:
        client.app.state.semantic_engine = _StubSemanticEngine(ready=True)  # type: ignore[attr-defined]

        resp = client.post(
            "/extract-geometry",
            files=_multipart(_floor_plan_png()),
        )
        assert resp.status_code == HTTPStatus.OK, resp.text
        objects = resp.json()["objects"]
        assert len(objects) == 1
        obj = objects[0]
        assert obj["label"] == "bed"
        assert obj["source"] == "zeroshot"
        assert len(obj["bbox"]) == 4  # noqa: PLR2004
        assert "needs_review" in obj
        # room_id puede ser None o un string — merge_semantic lo resuelve
        assert "room_id" in obj

    def test_geometria_no_se_altera_por_track_semantico(
        self, client: TestClient
    ) -> None:
        """El track semántico nunca muta walls/rooms/openings (ADR-003)."""
        image = _floor_plan_png()

        client.app.state.semantic_engine = None  # type: ignore[attr-defined]
        resp_off = client.post("/extract-geometry", files=_multipart(image))

        client.app.state.semantic_engine = _StubSemanticEngine(ready=True)  # type: ignore[attr-defined]
        resp_on = client.post("/extract-geometry", files=_multipart(image))

        assert resp_off.status_code == HTTPStatus.OK
        assert resp_on.status_code == HTTPStatus.OK
        data_off = resp_off.json()
        data_on = resp_on.json()
        for field in (
            "walls",
            "rooms",
            "openings",
            "stairs_candidates",
            "scale",
            "image_size",
        ):
            assert data_off[field] == data_on[field], (
                f"AC-2: el campo {field!r} debe ser idéntico con motor semántico "
                "on/off — el track semántico es aditivo, nunca muta geometría."
            )


# ---------------------------------------------------------------------------
# AC-2 — CV_SEM_ENGINE off/vacío: objects: [] siempre, sin overhead
# ---------------------------------------------------------------------------


class TestAC2MotorSemanticoOff:
    def test_objects_vacio_cuando_motor_off(self, client: TestClient) -> None:
        client.app.state.semantic_engine = None  # type: ignore[attr-defined]

        resp = client.post(
            "/extract-geometry",
            files=_multipart(_floor_plan_png()),
        )
        assert resp.status_code == HTTPStatus.OK, resp.text
        assert resp.json()["objects"] == []

    def test_detect_no_se_invoca_cuando_motor_off(self, client: TestClient) -> None:
        """Sin overhead: si semantic_engine es None, merge_semantic ni detect() corren."""
        client.app.state.semantic_engine = None  # type: ignore[attr-defined]

        with patch(
            "vitrina_cv.api.routers.extract_geometry.merge_semantic"
        ) as mock_merge:
            resp = client.post(
                "/extract-geometry",
                files=_multipart(_floor_plan_png()),
            )
        assert resp.status_code == HTTPStatus.OK
        mock_merge.assert_not_called()


# ---------------------------------------------------------------------------
# Manejo de errores — motor semántico falla en inferencia -> degrada a []
# ---------------------------------------------------------------------------


class TestDegradacionBestEffort:
    def test_excepcion_en_detect_degrada_a_objects_vacio(
        self, client: TestClient
    ) -> None:
        client.app.state.semantic_engine = _StubSemanticEngine(  # type: ignore[attr-defined]
            ready=True, raise_on_detect=True
        )

        resp = client.post(
            "/extract-geometry",
            files=_multipart(_floor_plan_png()),
        )
        assert resp.status_code == HTTPStatus.OK, (
            "Una excepción del motor semántico no debe romper la respuesta "
            "geométrica (best-effort, Fase A)."
        )
        assert resp.json()["objects"] == []
        # El resto de la geometría sigue presente y válida.
        assert "walls" in resp.json()
        assert "rooms" in resp.json()


# ---------------------------------------------------------------------------
# AC-6 (parcial) — GET /health expone semantic_model_loaded
# ---------------------------------------------------------------------------


class TestHealthSemanticModelLoaded:
    def test_semantic_model_loaded_null_cuando_motor_off(
        self, client: TestClient
    ) -> None:
        client.app.state.semantic_engine = None  # type: ignore[attr-defined]

        resp = client.get("/health")
        assert resp.status_code == HTTPStatus.OK
        data = resp.json()
        assert data["model_loaded"] is True
        assert data["semantic_model_loaded"] is None

    def test_semantic_model_loaded_true_cuando_motor_listo(
        self, client: TestClient
    ) -> None:
        client.app.state.semantic_engine = _StubSemanticEngine(ready=True)  # type: ignore[attr-defined]

        resp = client.get("/health")
        assert resp.status_code == HTTPStatus.OK
        data = resp.json()
        assert data["model_loaded"] is True
        assert data["semantic_model_loaded"] is True

    def test_semantic_model_loaded_false_cuando_motor_no_listo(
        self, client: TestClient
    ) -> None:
        """El motor semántico no listo NO fuerza 503 — es aditivo/best-effort."""
        client.app.state.semantic_engine = _StubSemanticEngine(ready=False)  # type: ignore[attr-defined]

        resp = client.get("/health")
        assert resp.status_code == HTTPStatus.OK
        data = resp.json()
        assert data["model_loaded"] is True
        assert data["semantic_model_loaded"] is False

    def test_motor_geometrico_no_listo_sigue_forzando_503(
        self, client: TestClient
    ) -> None:
        """El gating de 503 sigue dependiendo solo del motor geométrico (AC-6 original)."""
        client.app.state.semantic_engine = _StubSemanticEngine(ready=True)  # type: ignore[attr-defined]
        with patch(
            "vitrina_cv.engines.opencv_classic.OpenCVClassicEngine.is_ready",
            new_callable=PropertyMock,
            return_value=False,
        ):
            resp = client.get("/health")
        assert resp.status_code == HTTPStatus.SERVICE_UNAVAILABLE
        assert resp.json()["error_code"] == "model_not_loaded"


# ---------------------------------------------------------------------------
# Wiring de fábrica — get_semantic_engine() se invoca en el lifespan según
# CV_SEM_ENGINE (smoke test del wiring real, sin TestClient completo)
# ---------------------------------------------------------------------------


class TestWiringFactory:
    def test_get_semantic_engine_off_produce_none_en_app_state(self) -> None:
        app = create_app()
        with TestClient(app) as c:
            # Sin CV_SEM_ENGINE configurado en el entorno de test, el default
            # de Settings es "" -> get_semantic_engine() retorna None.
            assert c.app.state.semantic_engine is None

    def test_get_semantic_engine_zeroshot_mockeado_en_lifespan(self) -> None:
        stub = MagicMock()
        stub.is_ready = True
        with patch(
            "vitrina_cv.main.get_semantic_engine",
            return_value=stub,
        ):
            app = create_app()
            with TestClient(app) as c:
                assert c.app.state.semantic_engine is stub
