"""Binary mask noise-reduction for floor-plan images.

WHY THIS EXISTS
---------------
Real scanned floor plans contain four categories of artefacts that
produce hundreds of spurious wall segments (HoughLinesP) and prevent
the CCA from closing room regions:

  1. Text labels and dimension numbers — compact blobs with bounding
     boxes small in both width and height.
  2. Diagonal hatching (achurado) — patterns used in terraces, patios
     and car parks whose pixels survive adaptive binarisation as short
     diagonal runs.
  3. Perimeter dimension/cota lines and scan-border frames — thin lines
     far from the main floor-plan loop.
  4. Interior annotation lines, furniture outlines, stair tread lines —
     thin H/V strokes (1-3 px wide) that survive steps 1-3 because they
     are rectilinear and may be long.  The key distinguisher is stroke
     thickness: real walls are 6-20 px wide; annotations are 1-3 px.

This module provides four sequential cleaning steps, each a pure
function (no I/O, no side effects), applied to the binary wall mask
AFTER binarisation and BEFORE HoughLinesP / CCA:

  Step 1 — ``remove_small_components``: CCA-based text/small-blob removal.
  Step 2 — ``retain_rectilinear``: directional morphological open that kills
            diagonal strokes while preserving H/V wall runs.
  Step 3 — ``crop_to_main_component``: zeros everything outside the bbox
            of the largest connected component (+ configurable margin).
            Running this before step 4 keeps the exterior perimeter connected,
            so the thin-stroke filter never fragments it.
  Step 4 — ``filter_thin_strokes``: bounded geodesic reconstruction from
            thick-stroke seeds; cota lines, furniture contours and stair
            lines disappear; actual wall strokes (thick cores + bounded
            propagation into thin-wall continuations) survive.

Order: 1 → 2 → 3 → 4.

Thresholds are read from ``Settings`` (env vars) so no value is ever
hardcoded here.  See ``config/settings.py`` for the full list with
defaults and rationale:

  CV_CLEANUP_ENABLED                  (master switch, default True)
  CV_CLEANUP_TEXT_MAX_SIDE_PX         (step 1, default 40 px)
  CV_CLEANUP_RECTILINEAR_LEN_PX       (step 2, default 150 px)
  CV_CLEANUP_CROP_ENABLED              (step 3 on/off, default True)
  CV_CLEANUP_CROP_MARGIN_PX            (step 3 margin, default 20 px)
  CV_CLEANUP_THICKNESS_FILTER_ENABLED  (step 4 on/off, default True)
  CV_CLEANUP_MIN_WALL_THICKNESS_PX     (step 4, default 6 px at 2000 px)
  CV_CLEANUP_THICKNESS_PRECLOSE_PX     (step 4 pre-close kernel, default 9 px)

The preflight gate (``preflight/checks.py``) is intentionally NOT
touched: it evaluates the image BEFORE this cleanup runs.

Interaction with _build_closed_wall_mask_for_rooms
--------------------------------------------------
The thickness filter runs inside ``clean_mask()``, which is called
BEFORE the engine splits the mask into wall-detection, opening-detection
and room-detection paths.  Therefore the rooms CCA also benefits from the
filter: annotation lines inside rooms that would fragment the CCA floor
regions are removed.  No separate thickness-filter call is needed in the
room-detection path.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from vitrina_cv.config.settings import Settings

_logger = logging.getLogger(__name__)


def remove_small_components(
    mask: NDArray[np.uint8],
    max_side_px: int,
) -> tuple[NDArray[np.uint8], int]:
    """Remove connected components whose bbox is small in BOTH dimensions.

    A component is considered noise (text, dimension digit, small blob) when
    BOTH its bounding-box width AND height are strictly less than
    *max_side_px*.  Real wall segments always have at least one long axis
    (they are thin in one direction and long in the other), so they survive.

    Args:
        mask: Binary uint8 mask (values 0 or 255), walls = 255.
        max_side_px: Maximum side length (px) threshold.  A component is
            removed only if width < max_side_px AND height < max_side_px.

    Returns:
        ``(cleaned_mask, removed_count)`` where *removed_count* is the
        number of components that were erased.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    removed = 0
    out: NDArray[np.uint8] = mask.copy()

    for label in range(1, num_labels):  # label 0 is background
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if w < max_side_px and h < max_side_px:
            out[labels == label] = 0
            removed += 1

    return out, removed


def retain_rectilinear(
    mask: NDArray[np.uint8],
    len_px: int,
) -> NDArray[np.uint8]:
    """Keep only pixels that belong to horizontal or vertical runs of >= len_px.

    Applies two directional morphological opens:
      - open_h = MORPH_OPEN with a (1 x len_px) kernel  — preserves H runs
      - open_v = MORPH_OPEN with a (len_px x 1) kernel  — preserves V runs
    Result = open_h | open_v

    Diagonal hatching (achurado) produces H/V runs of 1-3 px at scanned
    resolutions and is effectively erased.  True H/V wall strokes (at least
    len_px px long along their primary axis) are preserved.

    Known limitation (accepted): genuine diagonal walls are also removed.
    This is acceptable because the engine is calibrated for rectilinear
    floor plans and the preflight gate includes a rectilinearity check.

    Args:
        mask: Binary uint8 mask (walls = 255).
        len_px: Minimum continuous run length (px) to be retained in each
            direction.  Corresponds to CV_CLEANUP_RECTILINEAR_LEN_PX.

    Returns:
        Cleaned binary mask with only H/V structural runs retained.
    """
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (len_px, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, len_px))

    open_h: NDArray[np.uint8] = cv2.morphologyEx(mask, cv2.MORPH_OPEN, h_kernel)  # type: ignore[assignment]
    open_v: NDArray[np.uint8] = cv2.morphologyEx(mask, cv2.MORPH_OPEN, v_kernel)  # type: ignore[assignment]

    return cv2.bitwise_or(open_h, open_v)  # type: ignore[return-value]


def filter_thin_strokes(
    mask: NDArray[np.uint8],
    min_thickness_px: int,
    preclose_kernel_size: int = 9,
) -> NDArray[np.uint8]:
    """Remove thin-stroke artefacts via bounded geodesic reconstruction.

    Algorithm overview:

    **Pre-close (seed-mask preparation):**
    Before extracting seeds, apply a morphological CLOSE with a square kernel
    of side *preclose_kernel_size* to a *copy* of the mask.  This bridges
    gaps between the two parallel lines of double-wall notation (e.g. a gap
    of 4-8 px between two 1-px wall strands becomes a solid band after
    closing).  The pre-closed mask is used ONLY to compute seeds — the
    geodesic reconstruction is still bounded by the ORIGINAL mask.

    **Phase 1 — seed extraction:**
    Compute ``cv2.distanceTransform(seed_mask, DIST_L2, 5)`` to obtain, for
    every foreground pixel in the pre-closed mask, its distance to the
    nearest background pixel.  Local stroke thickness = ``2 x dist[p]``.
    Pixels where ``dist >= min_thickness_px / 2`` are "skeleton cores" of
    strokes at least *min_thickness_px* wide in the pre-closed mask.
    Only seed pixels that also exist in the ORIGINAL mask are kept
    (intersection), so the reconstruction can never expand beyond the
    original foreground.

    **Phase 2 — bounded geodesic dilation:**
    Dilate the seed set iteratively with a 3x3 structuring element; after
    each step AND the result with the ORIGINAL mask.  The number of
    iterations is ``ceil(min_thickness_px)``.

    **Phase 3 — result:**
    Return the mask after bounded dilation.  Pixels never reached by the
    propagation are discarded as thin-line artefacts.

    Why bounded (not full-convergence) dilation:
    Thin annotation lines are often connected to thick walls at T-junctions.
    With unlimited propagation, the geodesic dilation would reclaim the full
    annotation through that junction.  Limiting to ``ceil(T)`` iterations
    means at most T pixels of a thin line adjacent to a wall junction are
    recovered; the rest is discarded.  Because HoughLinesP uses
    ``_MIN_WALL_LENGTH_PX=15``, these short stubs never produce wall
    segments.

    Why the pre-close step:
    Plans where walls are drawn as double parallel lines with a gap > ~6 px
    are not adequately thickened by the standard (3,3) close in
    ``_build_wall_mask``.  The pre-close (default 9 px) fills gaps up to
    ~9 px and makes double-wall structures "thick" in the seed mask, so
    they are correctly seeded without over-expanding into nearby thin
    annotation lines.

    Args:
        mask: Binary uint8 mask (walls = 255).
        min_thickness_px: Minimum stroke thickness (px) to retain.
            Corresponds to the scaled form of CV_CLEANUP_MIN_WALL_THICKNESS_PX.
        preclose_kernel_size: Side (px) of the square kernel used in the
            pre-close step for seed computation.
            Corresponds to CV_CLEANUP_THICKNESS_PRECLOSE_PX.

    Returns:
        Cleaned binary uint8 mask with thin-stroke artefacts removed.
    """
    # Pre-close: bridge gaps in double-wall notation for seed computation only.
    big_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (preclose_kernel_size, preclose_kernel_size)
    )
    seed_mask: NDArray[np.uint8] = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, big_kernel)  # type: ignore[assignment]

    # Phase 1 — seed pixels: cores of thick strokes in the pre-closed mask,
    # intersected back with the original foreground.
    dist: NDArray[np.float32] = cv2.distanceTransform(seed_mask, cv2.DIST_L2, 5)  # type: ignore[assignment]
    seeds_in_preclose: NDArray[np.uint8] = (dist >= min_thickness_px / 2.0).astype(
        np.uint8
    ) * 255
    seeds: NDArray[np.uint8] = cv2.bitwise_and(seeds_in_preclose, mask)

    # Phase 2 — bounded geodesic dilation bounded by the ORIGINAL mask.
    # Each iteration expands the reconstructed region by 1 px within the mask.
    # Limit to ceil(T) iterations so stubs near junctions stay below
    # _MIN_WALL_LENGTH_PX and do not produce spurious Hough segments.
    n_iter = math.ceil(min_thickness_px)
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    reconstruction = seeds.copy()
    for _ in range(n_iter):
        dilated: NDArray[np.uint8] = cv2.dilate(reconstruction, dil_kernel)  # type: ignore[assignment]
        reconstruction = cv2.bitwise_and(dilated, mask)

    # Phase 3 — return the geodesically reconstructed mask.
    return reconstruction


def crop_to_main_component(
    mask: NDArray[np.uint8],
    margin_px: int,
) -> tuple[NDArray[np.uint8], tuple[int, int, int, int] | None]:
    """Zero out everything outside the bbox of the largest connected component.

    Identifies the connected component with the greatest pixel area (normally
    the exterior loop of the floor plan walls), computes its bounding box,
    expands it by *margin_px* on all sides (clamped to image bounds), and
    sets all pixels outside this expanded box to zero.

    This eliminates perimeter cota/dimension lines and scan-border frames
    that survive steps 1 and 2 because they are long H/V strokes.

    Args:
        mask: Binary uint8 mask (walls = 255).
        margin_px: Number of pixels to add on each side of the largest
            component's bounding box.  Corresponds to CV_CLEANUP_CROP_MARGIN_PX.

    Returns:
        ``(cropped_mask, bbox)`` where *bbox* is ``(x, y, w, h)`` of the
        expanded bounding box in pixel coordinates, or ``None`` if no
        foreground component was found (mask is empty).
    """
    img_h, img_w = mask.shape[:2]

    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels <= 1:  # only background
        return mask.copy(), None

    # Find the component with the largest area (label 0 is background — skip).
    best_label = 1
    best_area = int(stats[1, cv2.CC_STAT_AREA])
    for label in range(2, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = label

    x = int(stats[best_label, cv2.CC_STAT_LEFT])
    y = int(stats[best_label, cv2.CC_STAT_TOP])
    w = int(stats[best_label, cv2.CC_STAT_WIDTH])
    h = int(stats[best_label, cv2.CC_STAT_HEIGHT])

    # Expand by margin, clamped to image bounds.
    x0 = max(0, x - margin_px)
    y0 = max(0, y - margin_px)
    x1 = min(img_w, x + w + margin_px)
    y1 = min(img_h, y + h + margin_px)

    out: NDArray[np.uint8] = np.zeros_like(mask)
    out[y0:y1, x0:x1] = mask[y0:y1, x0:x1]

    return out, (x0, y0, x1 - x0, y1 - y0)


def clean_mask(
    mask: NDArray[np.uint8],
    settings: Settings,
) -> NDArray[np.uint8]:
    """Apply the full four-step cleanup pipeline to a binary wall mask.

    Steps (in order):
      1. Remove small components (text / dimension digits / tiny blobs).
      2. Retain only rectilinear (H/V) structure; kill diagonal hatching.
      3. Crop to the main floor-plan component (+ margin).  Running this
         before step 4 ensures the exterior wall perimeter stays connected,
         so the thin-stroke filter never fragments it.
      4. Filter thin strokes: remove annotation lines, furniture and stair
         outlines via bounded geodesic reconstruction.  The thickness
         threshold is scaled proportionally when the image long side differs
         from ``settings.cv_upscale_target_px``.

    All steps are conditional on ``settings.cv_cleanup_enabled``.
    Step 3 is additionally gated on ``settings.cv_cleanup_crop_enabled``.
    Step 4 is additionally gated on
    ``settings.cv_cleanup_thickness_filter_enabled``.
    Logging at INFO level records the number of components removed in step 1,
    the effective thickness threshold used in step 3, and the crop bbox from
    step 4.

    Args:
        mask: Binary uint8 wall mask (walls = 255) to clean.
        settings: Application settings carrying cleanup thresholds.

    Returns:
        Cleaned binary uint8 mask ready for HoughLinesP and CCA.
    """
    if not settings.cv_cleanup_enabled:
        return mask

    # Step 1 — text / small-component removal
    cleaned, removed_count = remove_small_components(
        mask, settings.cv_cleanup_text_max_side_px
    )
    _logger.info(
        "cv_cleanup_step1_small_components",
        extra={"removed_count": removed_count},
    )

    # Step 2 — diagonal hatching removal via directional morphological open
    cleaned = retain_rectilinear(cleaned, settings.cv_cleanup_rectilinear_len_px)

    # Step 3 — crop to main component
    # Running crop BEFORE the thin-stroke filter keeps the exterior perimeter
    # as one large connected component.  If crop ran after the filter, the
    # filter could fragment the perimeter at thin junctions, causing crop to
    # select only a small wall segment as the "largest component".
    if settings.cv_cleanup_crop_enabled:
        cleaned, bbox = crop_to_main_component(
            cleaned, settings.cv_cleanup_crop_margin_px
        )
        _logger.info(
            "cv_cleanup_step3_crop",
            extra={"crop_bbox_xywh": bbox},
        )

    # Step 4 — thin-stroke filter (removes cota lines, furniture, stair lines)
    # The threshold is calibrated at CV_UPSCALE_TARGET_PX (~2000 px long side).
    # For images larger than the calibration resolution (not upscaled because
    # their long side already exceeds the target), we scale the threshold
    # proportionally so that physically identical strokes pass the same test
    # regardless of image resolution.
    if settings.cv_cleanup_thickness_filter_enabled:
        img_h, img_w = cleaned.shape[:2]
        long_side = max(img_h, img_w)
        resolution_scale = long_side / max(1, settings.cv_upscale_target_px)
        effective_thickness = max(
            1, round(settings.cv_cleanup_min_wall_thickness_px * resolution_scale)
        )
        cleaned = filter_thin_strokes(
            cleaned,
            effective_thickness,
            preclose_kernel_size=settings.cv_cleanup_thickness_preclose_px,
        )
        _logger.info(
            "cv_cleanup_step4_thickness_filter",
            extra={
                "min_wall_thickness_px_setting": settings.cv_cleanup_min_wall_thickness_px,
                "effective_threshold_px": effective_thickness,
                "resolution_scale": round(resolution_scale, 3),
                "preclose_kernel_px": settings.cv_cleanup_thickness_preclose_px,
            },
        )

    return cleaned
