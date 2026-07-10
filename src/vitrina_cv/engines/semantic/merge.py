"""Fusion of semantic detections with already-extracted geometry (ADR-003, run 11).

merge_semantic() is a pure post-processing step invoked after the geometric
pipeline (walls/rooms/openings) and the semantic engine (SemanticObject list)
have both run. It never mutates its inputs (ADR-003: the semantic track runs
in parallel with, and never replaces, the geometric stages) — it only reads
walls/rooms/openings and returns a new list of SemanticObject.

Two responsibilities (AC-4/AC-5, spec-cv-service):
  1. Dedup: a semantic window/door detection that overlaps (IoU above
     _IOU_DEDUP_THRESHOLD) an already-detected geometric Opening is dropped —
     the opening already exists in openings[], so keeping the semantic
     duplicate would double-count the same physical element.
  2. Room assignment: every surviving object gets room_id set to the id of
     the Room whose polygon contains the object's bbox centroid, or None when
     no room contains it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from vitrina_cv.models import SemanticLabel

if TYPE_CHECKING:
    from vitrina_cv.models import Opening, Room, SemanticObject, Wall

# IoU threshold above which a semantic abertura (window/door) detection is
# considered the same physical element as an already-detected geometric
# Opening and therefore dropped instead of duplicated into objects[].
# Follows the same "reasonable overlap, not exact match" spirit as other
# geometric thresholds in this codebase (e.g.
# CV_CLEANUP_INTERIOR_COMPONENTS_MARGIN_PX in mask_cleanup.py) — set loosely
# because bbox extents from a zero-shot detector and a geometric opening
# rarely align pixel-for-pixel even when they refer to the same abertura.
_IOU_DEDUP_THRESHOLD = 0.3

# Labels that represent an abertura (opening) and are therefore eligible for
# dedup against Opening candidates. Furniture labels (bed/sofa/table/chair)
# never overlap an Opening semantically, so they skip the dedup check.
_ABERTURA_LABELS = frozenset({SemanticLabel.window, SemanticLabel.door})

# A polygon needs at least 3 vertices to enclose any area.
_MIN_POLYGON_VERTICES = 3


def _bbox_iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Intersection-over-union of two [x, y, w, h] boxes in pixel space."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0

    union = aw * ah + bw * bh - intersection
    if union <= 0.0:
        return 0.0

    return intersection / union


def _is_duplicate_of_opening(obj: SemanticObject, openings: list[Opening]) -> bool:
    """True when obj is an abertura label overlapping an existing Opening."""
    if obj.label not in _ABERTURA_LABELS:
        return False

    return any(
        _bbox_iou(obj.bbox, opening.bbox) > _IOU_DEDUP_THRESHOLD for opening in openings
    )


def _bbox_centroid(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float]:
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def _room_id_for_point(point: tuple[float, float], rooms: list[Room]) -> str | None:
    """Resolve which room contains point via centroid-in-polygon containment.

    Uses cv2.pointPolygonTest (same primitive as
    vitrina_cv.mask_cleanup.filter_interior_components) rather than an exact
    full-bbox-containment check: a piece of furniture's bbox commonly clips a
    wall by a few pixels even when the object clearly belongs to that room,
    so centroid containment is the simpler, more robust criterion (documented
    limitation: an object straddling two rooms is assigned by its centroid
    only, per ADR-004's "room_id resolution is optional/best-effort" note).

    Room has no stable id field in the current contract (models.py), so the
    room's index in `rooms` (stringified) is used as room_id — ADR-004 flags
    a stable Room.id as an optional future extension.
    """
    for index, room in enumerate(rooms):
        if len(room.polygon) < _MIN_POLYGON_VERTICES:
            continue
        contour = np.array(room.polygon, dtype=np.float32).reshape((-1, 1, 2))
        if cv2.pointPolygonTest(contour, point, measureDist=False) >= 0:
            return str(index)
    return None


def merge_semantic(
    objects: list[SemanticObject],
    rooms: list[Room],
    walls: list[Wall],
    openings: list[Opening],
) -> tuple[list[SemanticObject], int]:
    """Fuse semantic detections with already-extracted geometry (AC-4/AC-5).

    Pure function: walls/rooms/openings are only read, never mutated or
    re-derived (ADR-003). `walls` is accepted for interface symmetry with
    SemanticEngine.detect() and potential future wall-proximity heuristics,
    but is not used by the current dedup/room-assignment logic.

    Args:
        objects: Semantic detections from a SemanticEngine, room_id unset.
        rooms: Room polygons already detected by the geometric pipeline.
        walls: Wall segments already detected by the geometric pipeline
            (unused today, kept for interface symmetry — see docstring).
        openings: Opening candidates already detected by the geometric
            pipeline.

    Returns:
        Tuple of (merged, dedup_count):
          merged: New list of SemanticObject — abertura duplicates of an
            existing Opening removed, room_id set by centroid containment
            for the rest.
          dedup_count: Number of detections dropped by the Opening dedup
            step (11-cv-05, observability — feeds
            semantic_dedup_vs_openings in the extract-geometry log).
    """
    del walls  # unused today; kept for interface symmetry (see docstring)

    merged: list[SemanticObject] = []
    dedup_count = 0
    for obj in objects:
        if _is_duplicate_of_opening(obj, openings):
            dedup_count += 1
            continue

        centroid = _bbox_centroid(obj.bbox)
        room_id = _room_id_for_point(centroid, rooms)
        merged.append(obj.model_copy(update={"room_id": room_id}))

    return merged, dedup_count
