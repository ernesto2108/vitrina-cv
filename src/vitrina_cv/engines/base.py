"""Base interface for CV geometry extraction engines (Strategy pattern, ADR-008).

All concrete engines must implement GeometryEngine.
Routers must never instantiate a concrete engine directly — always use get_engine().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vitrina_cv.config.settings import Settings
    from vitrina_cv.models import Geometry


class GeometryEngine(ABC):
    """Abstract interface for floor plan geometry extraction.

    Implementations:
      - OpenCVClassicEngine  (Phase 1, opencv-python)
      - RasterScanEngine     (future evaluation, ADR-008)

    The active engine is selected at startup via CV_ENGINE env var.
    """

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Return True when the engine is initialised and ready to process images.

        Feeds the model_loaded field of GET /health (ADR-003).
        Classical engines (no ML weights) should return True immediately after
        construction.  ML-backed engines should return True only after weights
        are loaded successfully.
        """

    @abstractmethod
    def extract(self, image_bytes: bytes) -> Geometry:
        """Extract geometry from a PNG image.

        Args:
            image_bytes: Raw PNG bytes received in the request.

        Returns:
            Geometry payload conforming to cv-service.openapi.yaml.
            All coordinates in pixels of the received image (ADR-003).
        """


def get_engine(cv_engine: str, settings: Settings | None = None) -> GeometryEngine:
    """Factory: resolve CV_ENGINE string to a GeometryEngine instance (ADR-008).

    Args:
        cv_engine: Engine name from settings.cv_engine.
        settings: Application settings forwarded to the engine so it can read
            runtime thresholds (e.g. upscale target / factor) without violating
            the stateless-per-request contract — the Settings instance is
            shared and read-only during a request.

    Returns:
        Concrete GeometryEngine implementation.

    Raises:
        ValueError: If cv_engine is not a recognised engine name.
            Raised at startup — an unknown value must fail the boot
            immediately with an explicit message.
        NotImplementedError: If cv_engine names a planned but unbuilt engine.
    """
    # Import here to avoid circular imports and allow optional heavy deps
    from vitrina_cv.engines.opencv_classic import OpenCVClassicEngine  # noqa: PLC0415

    match cv_engine.lower():
        case "opencv":
            return OpenCVClassicEngine(settings=settings)
        case "rasterscan":
            msg = (
                "RasterScanEngine is not yet implemented (ADR-008 Phase 2). "
                "Set CV_ENGINE=opencv for Phase 1."
            )
            raise NotImplementedError(msg)
        case _:
            msg = (
                f"Unknown CV_ENGINE: {cv_engine!r}. "
                "Valid values: 'opencv'. "
                "Set CV_ENGINE to a supported engine and restart the service."
            )
            raise ValueError(msg)
