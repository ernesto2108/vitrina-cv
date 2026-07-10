"""Zero-shot text-conditioned semantic engine (ADR-002 Phase A, run 11).

Uses OWL-ViT (google/owlvit-base-patch32) via HuggingFace transformers for
text-conditioned open-vocabulary object detection. Chosen over Grounding DINO
because it has first-class support in transformers (AutoProcessor +
OwlViTForObjectDetection), a small, well-cached checkpoint, and runs on
CPU/MPS without extra native dependencies (Grounding DINO's official
implementation requires a custom CUDA extension for best performance and has
weaker out-of-the-box transformers integration as of this writing).

Hardware note (ADR-002 hardware constraint): OWL-ViT-base is small enough
(~580MB fp32) to run inference on CPU or Apple Silicon MPS in a few seconds
per image — no GPU required for local development. Production may move this
to a GPU-backed deployment (arch-infra.md), which is an infra concern outside
this engine's scope.

Weight source: CV_MODEL_PATH is honoured as a local model directory (e.g. a
pre-downloaded snapshot via `huggingface-cli download`) when set and non-
empty. When CV_MODEL_PATH is unset/empty, the engine falls back to pulling
the checkpoint from the HuggingFace Hub by its model id and relying on the
local HF cache (~/.cache/huggingface) for subsequent runs — the classical
GeometryEngine leaves CV_MODEL_PATH unused (ADR-008), but the semantic track
repurposes it per ADR-002's implementation notes. This dual-path keeps local
dev friction-free (no manual download required) while still supporting
pinned/offline weights in deployments that set CV_MODEL_PATH explicitly.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, Any

from vitrina_cv.engines.semantic.base import SemanticEngine
from vitrina_cv.models import SemanticLabel, SemanticObject, SemanticSource

if TYPE_CHECKING:
    from vitrina_cv.config.settings import Settings
    from vitrina_cv.models import Room, Wall

logger = logging.getLogger(__name__)

# Default HuggingFace Hub model id used when CV_MODEL_PATH is unset/empty.
_DEFAULT_MODEL_ID = "google/owlvit-base-patch32"

# Detection confidence floor passed to the model post-processor. Kept low
# (below CV_SEM_CONFIDENCE_MIN's typical 0.5) so that low-confidence
# detections still surface as needs_review=true candidates instead of being
# silently dropped by the model's own post-processing.
_MODEL_SCORE_THRESHOLD = 0.05

# Text prompts fed to OWL-ViT, one per SemanticLabel value, in enum order.
# OWL-ViT expects natural-language noun phrases; plain label names work well
# for common furniture/openings per the model card examples.
_LABEL_PROMPTS: dict[SemanticLabel, str] = {
    SemanticLabel.bed: "a bed",
    SemanticLabel.window: "a window",
    SemanticLabel.sofa: "a sofa",
    SemanticLabel.table: "a table",
    SemanticLabel.chair: "a chair",
    SemanticLabel.door: "a door",
}


class ZeroShotSemanticEngine(SemanticEngine):
    """Phase A semantic engine: OWL-ViT zero-shot text-conditioned detection.

    Loads model weights once at construction time (either from CV_MODEL_PATH
    if set, or from the HuggingFace Hub by model id otherwise) and reuses the
    loaded model/processor across detect() calls.

    Does not resolve room_id (deferred to 11-cv-03 fusion) and never mutates
    the rooms/walls context passed to detect() (ADR-003).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._confidence_min = (
            settings.cv_sem_confidence_min if settings is not None else 0.5
        )
        model_path = settings.cv_model_path.strip() if settings is not None else ""
        self._model_source = model_path or _DEFAULT_MODEL_ID

        # transformers ships no py.typed stubs; Any is the honest annotation
        # for these third-party model/processor handles.
        self._model: Any = None
        self._processor: Any = None
        self._ready = False

        try:
            self._load_model()
        except Exception:
            logger.exception(
                "ZeroShotSemanticEngine failed to load weights from %r",
                self._model_source,
            )

    def _load_model(self) -> None:
        # Imported lazily so importing this module (or the package) never
        # pays the torch/transformers import cost unless CV_SEM_ENGINE=zeroshot
        # actually instantiates this engine (mirrors the lazy-import pattern
        # used by vitrina_cv.engines.base.get_engine for OpenCVClassicEngine).
        from transformers import (  # noqa: PLC0415
            OwlViTForObjectDetection,
            OwlViTProcessor,
        )

        self._processor = OwlViTProcessor.from_pretrained(self._model_source)
        self._model = OwlViTForObjectDetection.from_pretrained(self._model_source)
        self._model.eval()
        self._ready = True

    @property
    def is_ready(self) -> bool:
        """True once OWL-ViT weights loaded successfully at construction time."""
        return self._ready

    def detect(
        self,
        image_bytes: bytes,
        rooms: list[Room],
        walls: list[Wall],
    ) -> list[SemanticObject]:
        """Run OWL-ViT zero-shot detection over the closed SemanticLabel enum.

        rooms/walls are accepted per the SemanticEngine contract but not yet
        used for room_id resolution or spatial filtering — that fusion logic
        is implemented in 11-cv-03. This engine only reads image_bytes.
        """
        if not self._ready or self._model is None or self._processor is None:
            msg = (
                "ZeroShotSemanticEngine.detect() called but the engine is not "
                "ready (weights failed to load). Check is_ready before calling "
                "detect(), or fix CV_MODEL_PATH / network access to the "
                "HuggingFace Hub."
            )
            raise RuntimeError(msg)

        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size

        labels = list(_LABEL_PROMPTS)
        prompts = [_LABEL_PROMPTS[label] for label in labels]

        inputs = self._processor(text=[prompts], images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([[height, width]])
        # transformers >=4.44 renamed OwlViTProcessor's post-processing method;
        # post_process_object_detection no longer exists on this processor.
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            threshold=_MODEL_SCORE_THRESHOLD,
            target_sizes=target_sizes,
        )[0]

        objects: list[SemanticObject] = []
        for box, score, label_idx in zip(
            results["boxes"], results["scores"], results["labels"], strict=True
        ):
            confidence = float(score)
            x_min, y_min, x_max, y_max = (float(v) for v in box)
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

            objects.append(
                SemanticObject(
                    label=labels[int(label_idx)],
                    bbox=bbox,
                    confidence=confidence,
                    needs_review=confidence < self._confidence_min,
                    room_id=None,
                    source=SemanticSource.zeroshot,
                )
            )

        return objects
