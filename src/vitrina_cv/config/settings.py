"""Application settings loaded from environment variables.

All thresholds are read from env — never hardcoded in domain logic.
To change a threshold, set the corresponding environment variable and restart.

Defaults:
  CV_ENGINE                       = "opencv"
  CV_MODEL_PATH                   = ""          (optional; unused in Phase 1)
  CV_PREFLIGHT_MIN_RESOLUTION     = "300x300"   (floor to detect non-rescuable images;
                                                 evaluated on the ORIGINAL image before
                                                 upscaling — see CV_UPSCALE_TARGET_PX)
  CV_PREFLIGHT_MIN_CONTRAST       = 0.35
  CV_PREFLIGHT_MIN_LINE_DENSITY   = 0.005
  CV_UPSCALE_TARGET_PX            = 2000        (long-side target after normalisation)
  CV_UPSCALE_MAX_FACTOR           = 4.0         (safety cap to avoid inflating thumbnails)

  Mask cleanup (applied after binarisation, before Hough/CCA):
  CV_CLEANUP_ENABLED              = true         (master switch; set false to bypass all cleanup)
  CV_CLEANUP_TEXT_MAX_SIDE_PX     = 40           (step 1: remove connected components whose
                                                  bounding box has BOTH sides smaller than this.
                                                  Real walls have at least one long side; text and
                                                  dimension labels are compact in both dimensions.)
  CV_CLEANUP_RECTILINEAR_LEN_PX   = 150          (step 2: directional open kernel length (px).
                                                  open_h = MORPH_OPEN(1 x L); open_v = MORPH_OPEN(L x 1);
                                                  cleaned = open_h | open_v.
                                                  Diagonal hatching (achurado) has H/V runs of 1-3 px and
                                                  disappears; true H/V walls (≥1 px run) survive.
                                                  Limitation: genuine diagonal walls are also removed —
                                                  acceptable because the engine is rectilinear-oriented.)
  CV_CLEANUP_CROP_ENABLED              = true     (step 3: crop mask to bbox of largest connected
                                                  component + CV_CLEANUP_CROP_MARGIN_PX. Eliminates
                                                  dimension lines and scan-border artefacts outside
                                                  the main floor-plan loop. Running before step 4
                                                  keeps the perimeter intact for the filter.)
  CV_CLEANUP_CROP_MARGIN_PX            = 20       (step 3 margin in px around the main component bbox)
  CV_CLEANUP_THICKNESS_FILTER_ENABLED  = true    (step 4 on/off switch for thin-stroke filter)
  CV_CLEANUP_MIN_WALL_THICKNESS_PX     = 6       (step 4: minimum wall stroke max-thickness in px,
                                                  calibrated for ~2000 px images; auto-scaled
                                                  for larger images based on CV_UPSCALE_TARGET_PX)
  CV_CLEANUP_THICKNESS_PRECLOSE_PX     = 9       (step 4: pre-close kernel size (px) used to bridge
                                                  double-wall gaps before computing distance seeds)

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
        default="300x300",
        description=(
            "Minimum image resolution for the preflight gate, as 'WxH'. "
            "Evaluated on the ORIGINAL image (before any upscaling). "
            "This is the 'non-rescuable' floor: images below this size cannot "
            "yield useful geometry even after upscaling. Default: 300x300."
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

    # --- Resolution normalisation (upscale) ---
    cv_upscale_target_px: int = Field(
        default=2000,
        gt=0,
        description=(
            "Long-side target (px) after resolution normalisation. "
            "Images whose long side is already >= this value are left unchanged. "
            "The engine's absolute-pixel constants (wall thickness, gap sizes, "
            "room area) are calibrated for ~2000 px images. Default: 2000."
        ),
    )
    cv_upscale_max_factor: float = Field(
        default=4.0,
        gt=0.0,
        description=(
            "Maximum upscale factor applied during resolution normalisation. "
            "Caps the scale ratio to prevent inflating tiny thumbnails "
            "(e.g. icons or previews) to impractical sizes. Default: 4.0."
        ),
    )

    # --- Mask cleanup (noise reduction before Hough/CCA) ---
    cv_cleanup_enabled: bool = Field(
        default=True,
        description=(
            "Master switch for the mask-cleanup pipeline. "
            "When False all three cleanup steps are bypassed and the raw "
            "binarised mask is fed directly to Hough and CCA. Default: True."
        ),
    )
    cv_cleanup_text_max_side_px: int = Field(
        default=40,
        gt=0,
        description=(
            "Step 1 — text/small-component filter. "
            "Connected components whose bounding box has BOTH width AND height "
            "smaller than this value (px) are removed from the binary mask. "
            "Real wall segments always have at least one long side, so they are "
            "preserved. Calibrated for ~2000 px normalised images. Default: 40."
        ),
    )
    cv_cleanup_rectilinear_len_px: int = Field(
        default=150,
        gt=0,
        description=(
            "Step 2 — rectilinear-structure retention (kills diagonal hatching). "
            "Kernel length L used for directional morphological opens: "
            "open_h = MORPH_OPEN(1xL); open_v = MORPH_OPEN(Lx1); "
            "cleaned = open_h | open_v. "
            "Diagonal hatch lines have H/V runs of 1-3 px and vanish; "
            "true H/V walls (≥L px continuous run) survive. "
            "Limitation: genuine diagonal walls are also removed. Default: 150."
        ),
    )
    cv_cleanup_crop_enabled: bool = Field(
        default=True,
        description=(
            "Step 3 — crop mask to the main floor-plan structure. "
            "Finds the largest connected component of the already-cleaned mask, "
            "takes its bounding box + CV_CLEANUP_CROP_MARGIN_PX, and zeroes out "
            "everything outside. Removes perimeter dimension lines and scan borders. "
            "Running this before the thin-stroke filter (step 4) keeps the "
            "exterior wall perimeter connected, preventing the filter from "
            "fragmenting it and causing crop to select only a wall stub. "
            "Default: True."
        ),
    )
    cv_cleanup_crop_margin_px: int = Field(
        default=20,
        ge=0,
        description=(
            "Step 3 margin (px) added on each side of the largest-component "
            "bounding box before cropping. Prevents clipping walls that touch "
            "or nearly touch the bbox edge. Default: 20."
        ),
    )
    cv_cleanup_thickness_filter_enabled: bool = Field(
        default=True,
        description=(
            "Step 4 — thin-stroke filter master switch. "
            "When True, a bounded geodesic reconstruction from thick-stroke seeds "
            "(distanceTransform >= CV_CLEANUP_MIN_WALL_THICKNESS_PX / 2) removes "
            "annotation lines, furniture outlines and stair lines while preserving "
            "thick wall strokes. Set to False to bypass step 4 entirely. Default: True."
        ),
    )
    cv_cleanup_min_wall_thickness_px: int = Field(
        default=6,
        ge=1,
        description=(
            "Step 4 — minimum wall stroke thickness (px) for seed extraction. "
            "Seeds are pixels whose distance-to-background in the pre-closed mask "
            "is >= this value / 2; seeds correspond to the cores of strokes at "
            "least this many pixels wide. A bounded geodesic dilation then recovers "
            "the full extent of wall strokes from those seeds. Strokes thinner than "
            "this value that are not within reach of a seed (cotas, furniture "
            "outlines, stair lines) are discarded. "
            "The threshold is automatically scaled proportionally when the image "
            "long side differs from CV_UPSCALE_TARGET_PX, so the same default "
            "works for larger images without tuning. "
            "Calibrated for ~2000 px normalised images. Default: 6."
        ),
    )
    cv_cleanup_thickness_preclose_px: int = Field(
        default=9,
        ge=1,
        description=(
            "Step 4 — side length (px) of the square kernel used for the "
            "pre-close operation before distance-transform seed extraction. "
            "This morphological close is applied to a copy of the mask (not "
            "the mask itself) to bridge gaps between the two parallel lines "
            "of double-wall notation. Walls drawn as two thin parallel strands "
            "with a gap of up to ~(preclose_kernel - 1) px are treated as a "
            "single thick stroke for seed-extraction purposes, while isolated "
            "thin annotation lines remain unaffected. "
            "Increasing this value helps plans with wider double-wall gaps; "
            "decreasing it (or setting to 1 to disable) is appropriate when "
            "walls are drawn as single solid strokes and nearby annotation "
            "lines must not be merged for seed computation. "
            "Calibrated for ~2000 px normalised images. Default: 9."
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
