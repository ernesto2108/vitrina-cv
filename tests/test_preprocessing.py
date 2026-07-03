"""Tests for preprocessing.normalize_resolution and related integrations.

Covers the 7 cases from the handoff:
1. No-op when long_side >= target (factor 1.0)
2. Upscale 612x612 -> factor ~3.268, result 2000x2000
3. Cap at max_factor (100x100 x 4.0 -> 400x400)
4. Non-square aspect ratio preserved (470x896 -> 2000/896 factor)
5. Integration preflight: 612x612 synthetic plan -> resolution_ok=True, no error
6. Integration preflight: 200x200 -> resolution_ok=False with suggestion
7. Integration extract: small plan -> image_size == normalized dimensions
"""

from __future__ import annotations

import io
from http import HTTPStatus

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from vitrina_cv.config.settings import Settings
from vitrina_cv.main import create_app
from vitrina_cv.preflight.checks import run_preflight
from vitrina_cv.preprocessing import normalize_resolution

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bgr(h: int, w: int) -> np.ndarray:
    """Create a minimal BGR uint8 image."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _floor_plan_png(width: int = 612, height: int = 612) -> bytes:
    """Synthetic floor plan (white background + black lines) as PNG bytes."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    # horizontal and vertical walls to give enough structure for preflight
    cv2.rectangle(img, (20, 20), (width - 20, height - 20), (0, 0, 0), 3)
    cv2.line(img, (width // 2, 20), (width // 2, height - 20), (0, 0, 0), 2)
    cv2.line(img, (20, height // 2), (width - 20, height // 2), (0, 0, 0), 2)
    ok, buf = cv2.imencode(".png", img)
    assert ok, "cv2.imencode failed in fixture"
    return buf.tobytes()


def _default_settings(**overrides: object) -> Settings:
    """Build a Settings instance with test-friendly defaults."""
    return Settings(  # type: ignore[call-arg]
        cv_upscale_target_px=overrides.pop("cv_upscale_target_px", 2000),
        cv_upscale_max_factor=overrides.pop("cv_upscale_max_factor", 4.0),
        **overrides,
    )


# ---------------------------------------------------------------------------
# Unit tests — normalize_resolution
# ---------------------------------------------------------------------------


class TestNormalizeResolution:
    """Unit tests for preprocessing.normalize_resolution."""

    def test_no_upscale_when_long_side_gte_target(self) -> None:
        """Case 1: long_side >= target → returns same array, factor == 1.0."""
        img = _make_bgr(h=1500, w=2000)  # long_side = 2000, at target
        settings = _default_settings(cv_upscale_target_px=2000)

        result, factor = normalize_resolution(img, settings)

        assert result.shape == img.shape
        assert factor == pytest.approx(1.0)

    def test_no_upscale_when_long_side_exceeds_target(self) -> None:
        """No downscale: long_side > target also returns factor 1.0."""
        img = _make_bgr(h=900, w=3000)  # long_side = 3000 > 2000
        settings = _default_settings(cv_upscale_target_px=2000)

        result, factor = normalize_resolution(img, settings)

        assert result.shape == img.shape
        assert factor == pytest.approx(1.0)

    def test_upscale_square_612x612(self) -> None:
        """Case 2: 612x612 → factor ≈ 2000/612 ≈ 3.268, result 2000x2000."""
        img = _make_bgr(h=612, w=612)
        settings = _default_settings(
            cv_upscale_target_px=2000, cv_upscale_max_factor=4.0
        )

        result, factor = normalize_resolution(img, settings)

        expected_factor = 2000 / 612
        assert factor == pytest.approx(expected_factor, rel=1e-4)
        assert result.shape[0] == round(612 * expected_factor)  # height
        assert result.shape[1] == round(612 * expected_factor)  # width

    def test_upscale_capped_at_max_factor(self) -> None:
        """Case 3: 100x100 → uncapped factor = 20, capped at 4.0 → 400x400."""
        img = _make_bgr(h=100, w=100)
        settings = _default_settings(
            cv_upscale_target_px=2000, cv_upscale_max_factor=4.0
        )

        result, factor = normalize_resolution(img, settings)

        assert factor == pytest.approx(4.0)
        assert result.shape == (400, 400, 3)

    def test_aspect_ratio_preserved_for_non_square(self) -> None:
        """Case 4: 470x896 → factor = 2000/896, aspect ratio preserved via round."""
        img = _make_bgr(h=896, w=470)  # portrait: long_side = 896
        settings = _default_settings(
            cv_upscale_target_px=2000, cv_upscale_max_factor=4.0
        )

        result, factor = normalize_resolution(img, settings)

        expected_factor = 2000 / 896
        assert factor == pytest.approx(expected_factor, rel=1e-4)
        assert result.shape[0] == round(896 * expected_factor)  # height
        assert result.shape[1] == round(470 * expected_factor)  # width

    def test_output_dtype_is_uint8(self) -> None:
        """Upscaled image preserves uint8 dtype."""
        img = _make_bgr(h=500, w=500)
        settings = _default_settings()

        result, _ = normalize_resolution(img, settings)

        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# Integration tests — preflight
# ---------------------------------------------------------------------------


class TestPreflightWithNormalization:
    """Integration tests: run_preflight evaluates resolution on original image."""

    def test_612x612_plan_passes_resolution_check(self) -> None:
        """Case 5: 612x612 > 300x300 piso → resolution_ok=True, no error."""
        image_bytes = _floor_plan_png(width=612, height=612)
        settings = _default_settings()

        report = run_preflight(image_bytes, settings)

        assert report.resolution_ok is True
        # No exception means pipeline ran without error

    def test_200x200_fails_resolution_check(self) -> None:
        """Case 6: 200x200 < 300x300 piso → resolution_ok=False with suggestion."""
        image_bytes = _floor_plan_png(width=200, height=200)
        settings = _default_settings()

        report = run_preflight(image_bytes, settings)

        assert report.resolution_ok is False
        # Suggestion must be present and non-empty
        assert report.suggestions
        assert any(
            "resoluc" in s.lower() or "resolution" in s.lower() or "px" in s.lower()
            for s in report.suggestions
        ), f"Expected resolution suggestion, got: {report.suggestions}"


# ---------------------------------------------------------------------------
# Integration tests — extract endpoint (image_size == normalized dimensions)
# ---------------------------------------------------------------------------


def _multipart_file(image_bytes: bytes) -> dict:
    """Build multipart dict using the field name expected by the API."""
    return {"image": ("plan.png", io.BytesIO(image_bytes), "image/png")}


class TestExtractNormalizedImageSize:
    """Case 7: /extract-geometry returns image_size == normalized dimensions."""

    def test_image_size_reflects_normalized_dimensions(self) -> None:
        """Small plan (612x612) is upscaled; image_size in response == normalized shape."""
        image_bytes = _floor_plan_png(width=612, height=612)

        with TestClient(create_app(), raise_server_exceptions=False) as client:
            resp = client.post("/extract-geometry", files=_multipart_file(image_bytes))

        assert resp.status_code == HTTPStatus.OK
        image_size = resp.json()["image_size"]

        # 612x612 → factor = min(2000/612, 4.0) ≈ 3.268 → 2000x2000
        expected_factor = 2000 / 612
        expected_px = round(612 * expected_factor)

        assert image_size["width"] == expected_px
        assert image_size["height"] == expected_px
