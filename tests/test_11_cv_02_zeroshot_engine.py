"""Smoke tests for ZeroShotSemanticEngine and the get_semantic_engine factory.

Model loading (transformers/torch) is mocked so this test suite stays fast
and does not require network access or real OWL-ViT weights in CI. Real
end-to-end detection against actual weights is exercised in eval/manual
runs, not in the unit test suite (mirrors the pattern used for the scale-OCR
tesseract binary in test_scale_ocr.py).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vitrina_cv.config.settings import Settings
from vitrina_cv.engines.semantic.base import get_semantic_engine
from vitrina_cv.engines.semantic.zeroshot import ZeroShotSemanticEngine
from vitrina_cv.models import SemanticLabel, SemanticSource

_A_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
    b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _mock_transformers() -> tuple[MagicMock, MagicMock]:
    """Build mocked OwlViTProcessor/OwlViTForObjectDetection classes."""
    processor_instance = MagicMock()
    processor_instance.return_value = {"input_ids": MagicMock()}
    processor_instance.post_process_grounded_object_detection.return_value = [
        {
            "boxes": [[10.0, 20.0, 60.0, 80.0]],
            "scores": [0.9],
            "labels": [0],
        }
    ]

    model_instance = MagicMock()
    model_instance.eval.return_value = None
    model_instance.return_value = MagicMock()

    processor_cls = MagicMock()
    processor_cls.from_pretrained.return_value = processor_instance

    model_cls = MagicMock()
    model_cls.from_pretrained.return_value = model_instance

    return processor_cls, model_cls


class TestGetSemanticEngineZeroshot:
    """get_semantic_engine('zeroshot') must return a working instance."""

    def test_returns_zeroshot_engine_instance(self) -> None:
        processor_cls, model_cls = _mock_transformers()
        with patch.dict(
            "sys.modules",
            {
                "transformers": MagicMock(
                    OwlViTProcessor=processor_cls,
                    OwlViTForObjectDetection=model_cls,
                )
            },
        ):
            engine = get_semantic_engine("zeroshot", settings=Settings())

        assert isinstance(engine, ZeroShotSemanticEngine)
        assert engine.is_ready is True


class TestZeroShotSemanticEngineDetect:
    """detect() must map model outputs to SemanticObject per the contract."""

    def test_detect_emits_semantic_object_with_expected_fields(self) -> None:
        processor_cls, model_cls = _mock_transformers()
        with patch.dict(
            "sys.modules",
            {
                "transformers": MagicMock(
                    OwlViTProcessor=processor_cls,
                    OwlViTForObjectDetection=model_cls,
                )
            },
        ):
            engine = ZeroShotSemanticEngine(settings=Settings())

        assert engine.is_ready

        with (
            patch("PIL.Image.open") as mock_image_open,
            patch("torch.no_grad"),
            patch("torch.tensor", side_effect=lambda x: x),
        ):
            mock_image = MagicMock()
            mock_image.size = (100, 100)
            mock_image_open.return_value.convert.return_value = mock_image

            objects = engine.detect(_A_PNG_BYTES, rooms=[], walls=[])

        assert len(objects) == 1
        obj = objects[0]
        assert obj.label == SemanticLabel.bed
        assert obj.bbox == (10.0, 20.0, 50.0, 60.0)
        assert obj.confidence == pytest.approx(0.9)
        assert obj.needs_review is False
        assert obj.room_id is None
        assert obj.source == SemanticSource.zeroshot

    def test_detect_marks_low_confidence_as_needs_review(self) -> None:
        processor_instance = MagicMock()
        processor_instance.return_value = {"input_ids": MagicMock()}
        processor_instance.post_process_grounded_object_detection.return_value = [
            {
                "boxes": [[0.0, 0.0, 10.0, 10.0]],
                "scores": [0.2],
                "labels": [1],
            }
        ]
        model_instance = MagicMock()
        model_instance.eval.return_value = None
        model_instance.return_value = MagicMock()

        processor_cls = MagicMock()
        processor_cls.from_pretrained.return_value = processor_instance
        model_cls = MagicMock()
        model_cls.from_pretrained.return_value = model_instance

        with patch.dict(
            "sys.modules",
            {
                "transformers": MagicMock(
                    OwlViTProcessor=processor_cls,
                    OwlViTForObjectDetection=model_cls,
                )
            },
        ):
            engine = ZeroShotSemanticEngine(
                settings=Settings(cv_sem_confidence_min=0.5)
            )

        with (
            patch("PIL.Image.open") as mock_image_open,
            patch("torch.no_grad"),
            patch("torch.tensor", side_effect=lambda x: x),
        ):
            mock_image = MagicMock()
            mock_image.size = (100, 100)
            mock_image_open.return_value.convert.return_value = mock_image

            objects = engine.detect(_A_PNG_BYTES, rooms=[], walls=[])

        assert len(objects) == 1
        assert objects[0].needs_review is True
        assert objects[0].label == SemanticLabel.window


class TestZeroShotSemanticEngineNotReady:
    """Constructor failures must surface as is_ready=False, not raise."""

    def test_load_failure_leaves_engine_not_ready(self) -> None:
        broken_transformers = MagicMock()
        broken_transformers.OwlViTProcessor.from_pretrained.side_effect = OSError(
            "no weights"
        )

        with patch.dict("sys.modules", {"transformers": broken_transformers}):
            engine = ZeroShotSemanticEngine(settings=Settings())

        assert engine.is_ready is False

    def test_detect_raises_when_not_ready(self) -> None:
        broken_transformers = MagicMock()
        broken_transformers.OwlViTProcessor.from_pretrained.side_effect = OSError(
            "no weights"
        )

        with patch.dict("sys.modules", {"transformers": broken_transformers}):
            engine = ZeroShotSemanticEngine(settings=Settings())

        with pytest.raises(RuntimeError, match="not ready"):
            engine.detect(_A_PNG_BYTES, rooms=[], walls=[])
