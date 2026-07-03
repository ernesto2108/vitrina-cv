"""Resolution normalisation -- upscale small floor-plan images before CV processing.

WHY THIS EXISTS
---------------
The OpenCV Classic engine is calibrated in *absolute* pixel values:

  _DOOR_GAP_MIN_PX       = 30
  _WALL_THICKNESS_EST_PX = 10
  _MIN_ROOM_AREA_PX      = 2_000
  ...

These constants assume the floor plan is at least ~800-1200 px on its long
side (~50 px/m).  Images that arrive at 612x612 or 470x896 (typical mobile
exports) produce sub-threshold gaps, too-small room areas and near-zero line
density -- all before any detection algorithm runs.

``normalize_resolution`` upscales the image so its long side reaches
``CV_UPSCALE_TARGET_PX`` (default 2000 px), capped at
``CV_UPSCALE_MAX_FACTOR`` (default 4.0x) to avoid inflating thumbnails to
absurd sizes.  INTER_CUBIC is used because architectural floor plans are
line-art: cubic interpolation preserves hard edges much better than bilinear
at the expense of a negligible quality loss.

The function is intentionally **pure** (no side effects, no I/O) and returns
the upscaled image alongside the applied factor so callers can log it.

IMPORTANT: ``image_size`` in the API response is set to the *normalised*
dimensions.  All geometry (walls, rooms, openings) is expressed in the
normalised pixel space.  Coordinates are NOT rescaled back to the original
image size -- this keeps every coordinate in a single, coherent coordinate
system and avoids floating-point rounding during re-projection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import numpy as np

    from vitrina_cv.config.settings import Settings


def normalize_resolution(
    img: np.ndarray,
    settings: Settings,
) -> tuple[np.ndarray, float]:
    """Upscale *img* so its long side reaches ``CV_UPSCALE_TARGET_PX``.

    The image is never downscaled: if ``max(h, w) >= CV_UPSCALE_TARGET_PX``
    the original array and factor ``1.0`` are returned unchanged.

    The upscale factor is capped at ``CV_UPSCALE_MAX_FACTOR`` to prevent
    inflating tiny thumbnails (e.g. 100x100 icons) to impractical sizes.

    Args:
        img: BGR uint8 ndarray of shape (H, W, 3).
        settings: Application settings carrying ``cv_upscale_target_px`` and
            ``cv_upscale_max_factor``.

    Returns:
        A ``(normalised_image, factor)`` tuple.  ``factor`` is ``1.0`` when
        no upscaling was applied, or the ratio ``new_long_side / old_long_side``
        otherwise.
    """
    h, w = img.shape[:2]
    long_side = max(h, w)

    if long_side >= settings.cv_upscale_target_px:
        return img, 1.0

    raw_factor = settings.cv_upscale_target_px / long_side
    factor = min(raw_factor, settings.cv_upscale_max_factor)

    new_w = round(w * factor)
    new_h = round(h * factor)

    upscaled: np.ndarray = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC,
    )
    return upscaled, factor
