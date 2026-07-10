"""Unit tests for merge_semantic (11-cv-03, ADR-003/ADR-004, AC-4/AC-5).

Covers: (a) dedup of an overlapping abertura against an existing Opening,
(b) no dedup when there's no overlap, (c) correct room_id assignment when
the object's bbox centroid falls inside a room polygon, (d) room_id None
when it falls outside every room. Also verifies the function is pure
(inputs untouched).
"""

from __future__ import annotations

from vitrina_cv.engines.semantic.merge import merge_semantic
from vitrina_cv.models import (
    Opening,
    OpeningTypeCandidate,
    Room,
    SemanticLabel,
    SemanticObject,
    SemanticSource,
)

_SQUARE_ROOM = Room(
    polygon=[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
    area_px=10000.0,
)


def _semantic_object(
    label: SemanticLabel,
    bbox: tuple[float, float, float, float],
) -> SemanticObject:
    return SemanticObject(
        label=label,
        bbox=bbox,
        confidence=0.9,
        needs_review=False,
        room_id=None,
        source=SemanticSource.zeroshot,
    )


class TestMergeSemanticDedup:
    def test_overlapping_window_is_deduped_against_existing_opening(self) -> None:
        window = _semantic_object(SemanticLabel.window, (10.0, 10.0, 20.0, 20.0))
        opening = Opening(
            type_candidate=OpeningTypeCandidate.window,
            bbox=(11.0, 11.0, 20.0, 20.0),
            confidence=0.8,
        )

        result, dedup_count = merge_semantic(
            objects=[window], rooms=[], walls=[], openings=[opening]
        )

        assert result == []
        assert dedup_count == 1

    def test_non_overlapping_window_is_not_deduped(self) -> None:
        window = _semantic_object(SemanticLabel.window, (10.0, 10.0, 20.0, 20.0))
        opening = Opening(
            type_candidate=OpeningTypeCandidate.window,
            bbox=(500.0, 500.0, 20.0, 20.0),
            confidence=0.8,
        )

        result, dedup_count = merge_semantic(
            objects=[window], rooms=[], walls=[], openings=[opening]
        )

        assert len(result) == 1
        assert result[0].label == SemanticLabel.window
        assert dedup_count == 0


class TestMergeSemanticRoomAssignment:
    def test_object_inside_room_gets_room_id(self) -> None:
        bed = _semantic_object(SemanticLabel.bed, (10.0, 10.0, 20.0, 20.0))

        result, dedup_count = merge_semantic(
            objects=[bed], rooms=[_SQUARE_ROOM], walls=[], openings=[]
        )

        assert len(result) == 1
        assert result[0].room_id == "0"
        assert dedup_count == 0

    def test_object_outside_every_room_gets_room_id_none(self) -> None:
        bed = _semantic_object(SemanticLabel.bed, (500.0, 500.0, 20.0, 20.0))

        result, dedup_count = merge_semantic(
            objects=[bed], rooms=[_SQUARE_ROOM], walls=[], openings=[]
        )

        assert len(result) == 1
        assert result[0].room_id is None
        assert dedup_count == 0


class TestMergeSemanticPurity:
    def test_inputs_are_not_mutated(self) -> None:
        bed = _semantic_object(SemanticLabel.bed, (10.0, 10.0, 20.0, 20.0))
        rooms = [_SQUARE_ROOM]
        openings: list[Opening] = []
        objects = [bed]

        rooms_before = [room.model_copy(deep=True) for room in rooms]
        objects_before = [obj.model_copy(deep=True) for obj in objects]

        merge_semantic(objects=objects, rooms=rooms, walls=[], openings=openings)

        assert rooms == rooms_before
        assert objects == objects_before
