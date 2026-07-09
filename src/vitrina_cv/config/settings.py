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
  CV_CLEANUP_INTERIOR_COMPONENTS_ENABLED = true  (step 5: remove wall-mask components whose bbox is
                                                  entirely inside a detected room polygon. Eliminates
                                                  rectilinear furniture (tables, sofas) missed by steps
                                                  1-4. Set false to skip without regression.)
  CV_CLEANUP_INTERIOR_COMPONENTS_MARGIN_PX = 10  (step 5: pointPolygonTest distance threshold (px).
                                                  Corners must be > this value inside the polygon to
                                                  trigger removal. Prevents wall segments touching the
                                                  polygon edge from being deleted.)

  Room detection (directional close before CCA):
  CV_ROOM_CLOSE_H_GAP_PX          = 80          (horizontal morphological close gap (px). Bridges
                                                  openings aligned with vertical walls. Must be
                                                  smaller than the narrowest expected room width to
                                                  avoid filling narrow rooms like bathrooms.
                                                  Default: 80 ≈ 0.6 m at 2000 px.)
  CV_ROOM_CLOSE_V_GAP_PX          = 160         (vertical morphological close gap (px). Bridges
                                                  openings aligned with horizontal walls (e.g. wide
                                                  sliding doors or passages). Default: 160 ≈ 1.2 m
                                                  at 2000 px.)

  Opening detection (gap-based — used in _detect_openings):
  CV_OPENING_MIN_WALL_SPAN_PX     = 60           (relaxed wall-span threshold (px) applied when a
                                                  gap endpoint is adjacent to a wall junction from
                                                  cv-04 _fuse_junctions.  Corner-adjacent doors have
                                                  short flanking segments on one side; the wide span
                                                  (170 px module constant) would discard them.  Fine
                                                  filtering is delegated to the backend (F4).
                                                  Default: 60 (≈0.4 m at 2000 px).  Previous hard-
                                                  coded value: 170.)

  Scale OCR (ADR-011 — pytesseract + tesseract binary, optional):
  CV_SCALE_OCR_ENABLED            = true         (master switch for OCR-based scale detection.
                                                  When false, _detect_scale returns source=none
                                                  unconditionally — same as Phase 1 behaviour.
                                                  Set to false in environments without the
                                                  tesseract-ocr binary. Default: true.)
  CV_SCALE_OCR_CONSISTENCY_TOLERANCE = 0.10     (maximum relative deviation from the median
                                                  px_per_unit across all cota readings before an
                                                  individual reading is discarded as an outlier.
                                                  0.10 = 10%. Default: 0.10.)
  CV_SCALE_OCR_TESSERACT_CMD      = ""          (optional override for the path to the tesseract
                                                  binary. Empty string means auto-detect via PATH.
                                                  Example: "/opt/homebrew/bin/tesseract".
                                                  Default: "" (auto).)

  Junction extend-to-intersection (08-cv-xx — F4 phase between snap and fuse):
  CV_JUNCTION_EXTEND_PX           = 40           (max gap (px) to extend H/V wall endpoints to
                                                  their orthogonal intersection. Calibrated for
                                                  ~2000 px normalised images. gt=0.)

  Staircase detection (07-cv-10 — uses pre-filter mask before filter_thin_strokes):
  CV_STAIRS_DETECTION_ENABLED     = true         (master switch. When false,
                                                  stairs_candidates=[] and the pipeline
                                                  is identical to the pre-07-cv-10 behaviour.)

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
    cv_cleanup_rectilinear_max_res_scale: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Step 2 — maximum resolution scale at which retain_rectilinear is "
            "applied.  The scale is computed as long_side / CV_UPSCALE_TARGET_PX. "
            "When the image resolution scale exceeds this cap (i.e. the image was "
            "NOT upscaled because it is already high-res), step 2 is skipped. "
            "Rationale: the rectilinear kernel length (150 px) was calibrated for "
            "~2000 px images; at native high-resolution it removes valid junction "
            "corner pieces that are critical for room-boundary closure, and the "
            "hatch-removal benefit is lower because thick-wall plans rarely use "
            "dense diagonal hatching. "
            "Default: 1.0 — skip retain_rectilinear for images with long side "
            "> CV_UPSCALE_TARGET_PX."
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
    cv_cleanup_crop_min_area_ratio: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description=(
            "Step 3 (ADR-015) — minimum area ratio (relative to the largest "
            "connected component) for a secondary component to be treated as "
            "'significant' during the multi-component crop and included in the "
            "combined bounding box, instead of being discarded as noise. "
            "0.05 = a component must have at least 5% of the largest "
            "component's area to count. Default: 0.05."
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
        default=5,
        ge=1,
        description=(
            "Step 4 — minimum wall stroke thickness (px) for seed extraction. "
            "Seeds are pixels whose distance-to-background in the pre-closed mask "
            "is >= this value / 2; seeds correspond to the cores of strokes at "
            "least this many pixels wide. A bounded geodesic dilation then recovers "
            "the full extent of wall strokes from those seeds. Strokes thinner than "
            "this value that are not within reach of a seed (cotas, furniture "
            "outlines, stair lines) are discarded. "
            "The resolution scale used for this threshold is capped at "
            "CV_CLEANUP_RECTILINEAR_MAX_RES_SCALE, so the physical threshold does "
            "not grow for native high-resolution images. This preserves thin "
            "double-line wall notation (≥5 px per strand) in high-res plans. "
            "Calibrated for ~2000 px normalised images. Default: 5 (was 6)."
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

    # --- Interior component filter (removes furniture/fixtures inside rooms) ---
    cv_cleanup_interior_components_enabled: bool = Field(
        default=True,
        description=(
            "Step 5 — interior-component filter. "
            "After room polygons are detected, runs CCA on the wall mask and "
            "removes any connected component whose bounding-box corners are ALL "
            "strictly inside a room polygon by more than "
            "CV_CLEANUP_INTERIOR_COMPONENTS_MARGIN_PX pixels. "
            "This eliminates rectilinear furniture (tables, sofas) that survive "
            "steps 1-4 because they are thick and have long H/V runs. "
            "Set to False to bypass without affecting steps 1-4. Default: True."
        ),
    )
    cv_cleanup_interior_components_margin_px: int = Field(
        default=10,
        ge=0,
        description=(
            "Step 5 — margin (px) used when testing whether a component's "
            "bounding-box corners lie inside a room polygon. "
            "A corner is considered 'inside' only when "
            "cv2.pointPolygonTest returns a distance strictly greater than this "
            "value. Using a positive margin (~wall thickness) ensures that wall "
            "segments that touch the room boundary are NOT removed (their corners "
            "fall on or near the polygon edge and fail the test). "
            "Default: 10 px (≈ half a wall stroke at 2000 px normalised images)."
        ),
    )

    # --- Room detection (directional close before CCA) ---
    cv_room_close_h_gap_px: int = Field(
        default=80,
        gt=0,
        description=(
            "Room detection — horizontal morphological close gap (px). "
            "Bridges openings in vertical walls (e.g. door gaps). "
            "Must be smaller than the narrowest expected room width to avoid "
            "filling narrow rooms such as bathrooms. "
            "Calibrated for ~2000 px normalised images. Default: 80 (≈0.6 m)."
        ),
    )
    cv_room_close_v_gap_px: int = Field(
        default=160,
        gt=0,
        description=(
            "Room detection — vertical morphological close gap (px). "
            "Bridges openings in horizontal walls (e.g. wide sliding doors "
            "or passages). "
            "Calibrated for ~2000 px normalised images. Default: 160 (≈1.2 m)."
        ),
    )
    cv_room_close_scale_with_upscale: bool = Field(
        default=False,
        description=(
            "When True, cv_room_close_h_gap_px and cv_room_close_v_gap_px are "
            "multiplied by the upscale_factor applied in normalize_resolution(). "
            "This scales door-gap bridging proportionally when a low-res plan is "
            "upscaled to CV_UPSCALE_TARGET_PX, ensuring wide openings are closed. "
            "Default is False (safe, no behaviour change) because the full "
            "upscale_factor multiplication is too aggressive for most test fixtures "
            "and eval plans that share the same source resolution (e.g. 612 px). "
            "Enable explicitly for plans where h_gap=80 provably fails to bridge "
            "door openings at the normalised resolution."
        ),
    )

    # --- Opening detection (07-cv-07) ---
    cv_opening_min_wall_span_px: int = Field(
        default=60,
        gt=0,
        description=(
            "Relaxed wall-span threshold (px) applied when a gap endpoint is "
            "adjacent to a wall junction produced by _fuse_junctions (cv-04). "
            "Corner-adjacent doors have a short flanking segment on one side; "
            "the wide span constant (170 px) would silently discard them. "
            "When no junction is nearby, the 170 px constant still applies so "
            "mid-wall artefact gaps are not affected by this setting. "
            "Fine filtering is delegated to the Go backend (F4, ADR-009). "
            "Default: 60 (≈0.4 m at 2000 px).  Previous hard-coded value: 170."
        ),
    )

    # --- Wall diagonal filter (08-cv-01) ---
    cv_wall_diagonal_filter_enabled: bool = Field(
        default=True,
        description=(
            "When True, wall segments whose angle (abs of atan2(dy,dx)) falls in "
            "[CV_WALL_DIAGONAL_FILTER_LOW_DEG, CV_WALL_DIAGONAL_FILTER_HIGH_DEG] "
            "are discarded after _consolidate_walls classification. "
            "Removes stair lines and sectional-door diagonals (~45°) that Hough "
            "misidentifies as walls. Set False to restore pre-08 behaviour. "
            "Default: True."
        ),
    )
    cv_wall_min_diagonal_len_px: int = Field(
        default=40,
        gt=0,
        description=(
            "Minimum Euclidean length (px) for a surviving oblique (diagonal) "
            "Wall segment to be kept after diagonal classification (ADR-017, "
            "Mec.2). Diagonal walls shorter than this value are discarded as "
            "residual noise (e.g. spurious Hough fragments) rather than "
            "legitimate diagonal walls. Default value to be calibrated later "
            "against real fixtures — for now the example value is used. "
            "Default: 40."
        ),
    )
    cv_wall_diagonal_filter_low_deg: float = Field(
        default=20.0,
        ge=0.0,
        le=90.0,
        description=(
            "Lower bound (degrees, inclusive) of the diagonal discard range. "
            "Segments with atan2(|dy|,|dx|) >= this value are candidates for "
            "discard. Default: 20.0."
        ),
    )
    cv_wall_diagonal_filter_high_deg: float = Field(
        default=70.0,
        ge=0.0,
        le=90.0,
        description=(
            "Upper bound (degrees, inclusive) of the diagonal discard range. "
            "Segments with atan2(|dy|,|dx|) <= this value are candidates for "
            "discard. Combined with the lower bound this removes the 20°-70° band. "
            "Default: 70.0."
        ),
    )

    # --- Room contour sanitize (10-cv-01, ADR-001) ---
    cv_room_contour_sanitize_enabled: bool = Field(
        default=True,
        description=(
            "When True, _detect_rooms sanitizes each room polygon after "
            "approxPolyDP: any vertex whose two adjacent edges both fall in the "
            "diagonal band [CV_WALL_DIAGONAL_FILTER_LOW_DEG, "
            "CV_WALL_DIAGONAL_FILTER_HIGH_DEG] (reusing the same band as the wall "
            "diagonal filter) and whose edge length exceeds "
            "CV_ROOM_CONTOUR_DIAG_MIN_LEN_PX is dropped and the contour "
            "re-simplified. Rooms whose sanitized contour still has a diagonal "
            "edge above the threshold are discarded entirely (ADR-001 AC-2). "
            "When False, _detect_rooms behaves exactly as before (no-op). "
            "Default: True."
        ),
    )
    cv_room_contour_diag_min_len_px: int = Field(
        default=40,
        gt=0,
        description=(
            "Minimum Euclidean length (px) for a room-polygon edge to be "
            "considered a spurious diagonal candidate for removal (ADR-001). "
            "Edges shorter than this value are assumed to be legitimate corner "
            "jitter within a rectilinear room and are left untouched. Same "
            "default as CV_WALL_MIN_DIAGONAL_LEN_PX (40) pending calibration "
            "against real fixtures."
        ),
    )
    cv_room_contour_deviation_min_px: int = Field(
        default=100,
        gt=0,
        description=(
            "Minimum perpendicular deviation (px) of a candidate spurious "
            "vertex from the straight line joining its two non-adjacent "
            "neighbours (the 'expected' rectilinear edge), required for the "
            "vertex to be treated as a true mask artefact and removed "
            "(10-cv-02 fix to ADR-001). Reusing only the diagonal angle band "
            "+ min length was too aggressive: it also discarded short, "
            "legitimate jogs/steps in dense floor plans (e.g. a ~53px "
            "diagonal edge with ~46px perpendicular deviation, plan-001-denso-"
            "achurado) whose deviation from the expected contour is small. A "
            "genuine spurious peak (e.g. plan-005-amueblado-limpio, vertex "
            "[1234,1559], ~215px deviation) is a pronounced spike, not a "
            "short step. Default: 100 (roughly between the two observed "
            "cases: 46px legitimate jog, 215px spurious peak)."
        ),
    )

    # --- Rectilinear adaptive filter for high-res images (08-cv-03) ---
    cv_cleanup_rectilinear_adaptive_enabled: bool = Field(
        default=True,
        description=(
            "When True, retain_rectilinear is applied to native high-resolution "
            "images (long side > CV_UPSCALE_TARGET_PX) using an adaptive kernel "
            "len_px = max(50, round(CV_CLEANUP_RECTILINEAR_LEN_PX * min(h,w) / "
            "CV_UPSCALE_TARGET_PX)) instead of being skipped entirely. "
            "This suppresses diagonal hatching even in high-res plans while "
            "preserving long perimeter walls. "
            "When False, the previous skip behaviour is preserved for high-res "
            "images. Default: True."
        ),
    )
    cv_cleanup_rectilinear_min_len_px: int = Field(
        default=50,
        gt=0,
        description=(
            "Step 2 (ADR-014) — floor of the adaptive kernel length used by "
            "the directional open kernel (retain_rectilinear) on native "
            "high-resolution images: len_px = max(CV_CLEANUP_RECTILINEAR_MIN_LEN_PX, "
            "round(CV_CLEANUP_RECTILINEAR_LEN_PX * min(h,w) / CV_UPSCALE_TARGET_PX)). "
            "Replaces the previously hardcoded literal 50 shared by both the "
            "fixed and adaptive branches, so the floor can be tuned per "
            "deployment. Default: 50."
        ),
    )

    # --- Wall centerline (07-cv-03) ---
    cv_wall_centerline_enabled: bool = Field(
        default=True,
        description=(
            "When True, _consolidate_walls estimates wall thickness via "
            "distanceTransform and collapses parallel HoughLinesP traces of thick "
            "walls into a single centerline Wall with thickness > 0 (px). "
            "Two segments are merged when their perpendicular separation is <= the "
            "estimated wall thickness (instead of the fixed 8-px bin used when False). "
            "The output Wall.thickness is 2 x median(DT samples along the segment), "
            "in pixels — the consumer multiplies by metersPerPx to get metres. "
            "Set to False to restore the legacy fixed-bin behaviour exactly. "
            "Default: True."
        ),
    )

    # --- Wall local thickness grouping (10-cv-05, ADR-003 part A) ---
    cv_wall_local_thickness_enabled: bool = Field(
        default=True,
        description=(
            "When True, _consolidate_walls groups parallel HoughLinesP traces "
            "using a per-group tolerance derived from the LOCAL wall thickness "
            "(sampled from the distanceTransform in the neighbourhood of each "
            "candidate group), instead of a single global thickness estimate "
            "for the whole plan. Fixes fragmentation in dense plans with "
            "heterogeneous wall thickness (ADR-003, part A). "
            "When False, falls back to _estimate_global_wall_thickness_px "
            "(pre-10-cv-05 behaviour, identical byte-for-byte). "
            "Only takes effect when cv_wall_centerline_enabled is also True. "
            "Default: True."
        ),
    )

    # --- Scale OCR (ADR-011) ---
    cv_scale_ocr_enabled: bool = Field(
        default=True,
        description=(
            "Master switch for OCR-based scale detection (ADR-011). "
            "When False, _detect_scale returns source=none unconditionally — "
            "identical to Phase 1 behaviour. Set to False in environments "
            "without the tesseract-ocr binary. Default: True."
        ),
    )
    cv_scale_ocr_consistency_tolerance: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description=(
            "Maximum relative deviation from the median px_per_unit across all "
            "cota readings before a reading is discarded as an outlier. "
            "0.10 = 10%. Readings outside this band are dropped; if fewer than "
            "2 consistent readings remain, scale falls back to source=none. "
            "Default: 0.10."
        ),
    )
    cv_scale_ocr_tesseract_cmd: str = Field(
        default="",
        description=(
            "Optional override for the path to the tesseract binary. "
            "Empty string = auto-detect via PATH. "
            "Example: '/opt/homebrew/bin/tesseract'. Default: '' (auto)."
        ),
    )

    # --- Junction extend-to-intersection (08-cv-xx / F4) ---
    cv_junction_extend_px: int = Field(
        default=40,
        gt=0,
        description=(
            "Max gap (px) for extending H/V wall endpoints to their orthogonal "
            "intersection in F4 (_extend_to_intersection). A wall endpoint is "
            "moved to the intersection only when its distance to the intersection "
            "is <= this value AND the intersection falls in the prolongation of "
            "the segment (outside its current extent), never in its interior. "
            "Calibrated for ~2000 px normalised images; scale proportionally for "
            "native high-resolution plans. Default: 40."
        ),
    )

    # --- Junction extend adaptive cap (10-cv-06, ADR-003 part A2) ---
    cv_wall_junction_extend_adaptive_enabled: bool = Field(
        default=True,
        description=(
            "When True, _extend_to_intersection caps a wall endpoint "
            "extension so it never crosses a nearer perpendicular wall, "
            "instead of always reaching the full cv_junction_extend_px gap. "
            "Prevents the fixed 40px extension from invading a neighbouring "
            "room in dense plans (ADR-003, part A2). Emits "
            "junction_extend_capped_count when the cap is applied. "
            "When False, restores the pre-10-cv-06 fixed behaviour exactly "
            "(byte-for-byte). Default: True."
        ),
    )

    # --- Staircase detection (07-cv-10) ---
    cv_stairs_detection_enabled: bool = Field(
        default=True,
        description=(
            "Master switch for staircase detection (07-cv-10). "
            "When True, _detect_stairs_candidates runs on the pre-filter mask "
            "(before filter_thin_strokes) and emits StairsCandidate objects for "
            "regions with ≥4 parallel equi-spaced tread lines contained within a "
            "room polygon. When False, stairs_candidates=[] and the extraction "
            "pipeline is unchanged from the pre-07-cv-10 behaviour. Default: True."
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
