"""Application settings loaded from environment variables.

All thresholds are read from env — never hardcoded in domain logic.
To change a threshold, set the corresponding environment variable and restart.

Defaults:
  CV_ENGINE                       = "opencv"
  CV_MODEL_PATH                   = ""          (optional; unused in Phase 1)
  CV_PREFLIGHT_MIN_RESOLUTION     = "800x600"
  CV_PREFLIGHT_MIN_CONTRAST       = 0.35
  CV_PREFLIGHT_MIN_LINE_DENSITY   = 0.005
  PORT                            = 8000
"""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for vitrina-cv.

    All values are sourced from environment variables (or .env file).
    No threshold may be hardcoded elsewhere in the codebase.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- CV Engine ---
    cv_engine: str = Field(
        default="opencv",
        description="Active CV engine. Phase 1: 'opencv'. Future: 'rasterscan' (ADR-008).",
    )
    cv_model_path: str = Field(
        default="",
        description="Path to ML model weights. Optional/future — unused in Phase 1 (ADR-008).",
    )

    # --- Preflight thresholds (ADR-005) ---
    cv_preflight_min_resolution: str = Field(
        default="800x600",
        description=(
            "Minimum image resolution accepted by the preflight gate, as 'WxH'. "
            "Default: 800x600."
        ),
    )
    cv_preflight_min_contrast: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum Michelson contrast ratio [0, 1] for the preflight gate. "
            "Default: 0.35."
        ),
    )
    cv_preflight_min_line_density: float = Field(
        default=0.005,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum ratio of edge pixels to total pixels for the preflight gate. "
            "Default: 0.05."
        ),
    )

    # --- Server ---
    port: int = Field(
        default=8000,
        description="TCP port the uvicorn server listens on.",
    )

    @field_validator("cv_preflight_min_resolution")
    @classmethod
    def validate_resolution_format(cls, v: str) -> str:
        """Ensure resolution is in 'WxH' format with positive integers."""
        parts = v.lower().split("x")
        if len(parts) != 2:  # noqa: PLR2004
            msg = f"CV_PREFLIGHT_MIN_RESOLUTION must be in 'WxH' format, got: {v!r}"
            raise ValueError(msg)
        try:
            width, height = int(parts[0]), int(parts[1])
        except ValueError:
            msg = f"CV_PREFLIGHT_MIN_RESOLUTION values must be integers, got: {v!r}"
            raise ValueError(msg) from None
        if width <= 0 or height <= 0:
            msg = "CV_PREFLIGHT_MIN_RESOLUTION dimensions must be positive integers"
            raise ValueError(msg)
        return v

    @property
    def preflight_min_width(self) -> int:
        """Parsed minimum width in pixels from CV_PREFLIGHT_MIN_RESOLUTION."""
        return int(self.cv_preflight_min_resolution.lower().split("x")[0])

    @property
    def preflight_min_height(self) -> int:
        """Parsed minimum height in pixels from CV_PREFLIGHT_MIN_RESOLUTION."""
        return int(self.cv_preflight_min_resolution.lower().split("x")[1])


def get_settings() -> Settings:
    """Return a Settings instance (reads from env / .env file on each call).

    In production, cache this via FastAPI's dependency injection instead of
    calling it on every request.
    """
    return Settings()
