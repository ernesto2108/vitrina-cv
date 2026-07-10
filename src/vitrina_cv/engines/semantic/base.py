"""Base interface for semantic (furniture/element) detection engines.

Strategy pattern analogous to GeometryEngine (ADR-008, run 06) — see
vitrina_cv/engines/base.py. All concrete semantic engines must implement
SemanticEngine.

Routers must never instantiate a concrete engine directly — always use
get_semantic_engine().

This module defines the interface and factory wiring. Concrete engines live
in sibling modules: ZeroShotSemanticEngine (11-cv-02, vitrina_cv.engines.
semantic.zeroshot) is wired below; FineTunedSemanticEngine (ADR-002 Phase B)
remains unimplemented and deferred.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vitrina_cv.config.settings import Settings
    from vitrina_cv.models import Room, SemanticObject, Wall


class SemanticEngine(ABC):
    """Abstract interface for semantic furniture/element detection (ADR-002, ADR-004).

    Implementations:
      - ZeroShotSemanticEngine  (Phase A, OWL-ViT/Grounding DINO — 11-cv-02+)
      - FineTunedSemanticEngine (Phase B, YOLO-World — deferred, ADR-002)

    The active engine is selected at startup via the CV_SEM_ENGINE env var.
    An empty/off value disables the semantic track entirely — callers must
    skip invoking detect() and return objects=[] (ADR-004 AC-2).
    """

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Return True when the engine is initialised and ready to process images.

        Feeds the model_loaded field of GET /health (mirrors GeometryEngine
        contract, ADR-003/spec-cv-service.md). Engines with no ML weights
        should return True immediately after construction; ML-backed engines
        should return True only after weights are loaded successfully.
        """

    @abstractmethod
    def detect(
        self,
        image_bytes: bytes,
        rooms: list[Room],
        walls: list[Wall],
    ) -> list[SemanticObject]:
        """Detect semantic objects (furniture/elements) in a floor-plan image.

        Runs in parallel with (and independently of) the geometric extraction
        pipeline; rooms/walls are passed as context so an implementation can
        resolve room_id or discard candidates outside the detected geometry,
        but the semantic engine must never mutate or re-derive geometry
        (ADR-003 of run 06 — walls/rooms/openings/stairs_candidates are
        unaffected by the semantic track).

        Args:
            image_bytes: Raw PNG bytes received in the request (same image
                passed to GeometryEngine.extract()).
            rooms: Room polygons already detected by the geometric pipeline,
                for spatial context (e.g. resolving room_id).
            walls: Wall segments already detected by the geometric pipeline,
                for spatial context.

        Returns:
            List of SemanticObject candidates conforming to
            cv-service.openapi.yaml SemanticObject (ADR-004). Empty list when
            no objects are detected — never raises for "no detections".
        """


def get_semantic_engine(
    cv_sem_engine: str, settings: Settings | None = None
) -> SemanticEngine | None:
    """Factory: resolve CV_SEM_ENGINE string to a SemanticEngine instance.

    Mirrors vitrina_cv.engines.base.get_engine() but returns None for the
    off/empty case instead of raising, because an unset CV_SEM_ENGINE is the
    documented default (semantic track disabled, ADR-004 AC-2) — not a
    configuration error.

    Args:
        cv_sem_engine: Engine name from settings.cv_sem_engine.
        settings: Application settings forwarded to the engine so it can read
            runtime thresholds (e.g. CV_SEM_CONFIDENCE_MIN, CV_MODEL_PATH)
            without violating the stateless-per-request contract.

    Returns:
        Concrete SemanticEngine implementation, or None when the semantic
        track is off (cv_sem_engine is empty/"off").

    Raises:
        ValueError: If cv_sem_engine is a non-empty, unrecognised engine name.
        NotImplementedError: If cv_sem_engine names a planned but unbuilt engine.
    """
    normalized = cv_sem_engine.strip().lower()
    if normalized in ("", "off"):
        return None

    match normalized:
        case "zeroshot":
            # Import here to avoid circular imports and to keep the heavy
            # torch/transformers dependency optional until CV_SEM_ENGINE=zeroshot
            # is actually selected (mirrors vitrina_cv.engines.base.get_engine).
            from vitrina_cv.engines.semantic.zeroshot import (  # noqa: PLC0415
                ZeroShotSemanticEngine,
            )

            return ZeroShotSemanticEngine(settings=settings)
        case "finetuned":
            msg = (
                "FineTunedSemanticEngine is not yet implemented (ADR-002 Phase B, "
                "deferred to a future run)."
            )
            raise NotImplementedError(msg)
        case _:
            msg = (
                f"Unknown CV_SEM_ENGINE: {cv_sem_engine!r}. "
                "Valid values: '', 'off', 'zeroshot', 'finetuned'. "
                "Set CV_SEM_ENGINE to a supported engine and restart the service."
            )
            raise ValueError(msg)
