"""Unit tests for CV09-10 — symmetric diagnostic logging (ADR-016).

Covers AC-7 and AC-8 of docs/specs/09-fidelidad-cv-segmentacion:

  AC-7: clean_mask_steps_1_to_3() emits the same 3 diagnostic events as
    clean_mask() — cv_cleanup_step1_small_components,
    cv_cleanup_step2_rectilinear, cv_cleanup_step3_crop.

  AC-8: the step2 log carries a canonical closed-set `branch` value
    ({"fixed", "adaptive", "skip"}) plus long_side/min_hw/upscale_target_px/
    rectilinear_len_px_used/min_len_px; the step3 log carries
    significant_components_count and crop_bbox_xywh.

Also includes a no-regression check: this logging-only change (CV09-05)
must not alter the output mask for a given input/settings pair.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vitrina_cv.config.settings import Settings
from vitrina_cv.mask_cleanup import clean_mask, clean_mask_steps_1_to_3

if TYPE_CHECKING:
    import pytest

_CANONICAL_BRANCHES = {"fixed", "adaptive", "skip"}


def _synthetic_mask(size: int = 400) -> np.ndarray:
    """Build a simple rectilinear wall mask (a hollow square) as uint8."""
    mask = np.zeros((size, size), dtype=np.uint8)
    thickness = 6
    margin = 40
    mask[margin : margin + thickness, margin : size - margin] = 255
    mask[size - margin - thickness : size - margin, margin : size - margin] = 255
    mask[margin : size - margin, margin : margin + thickness] = 255
    mask[margin : size - margin, size - margin - thickness : size - margin] = 255
    return mask


class TestStep1To3SymmetricLogging:
    """AC-7: clean_mask_steps_1_to_3 emits the same 3 events as clean_mask."""

    def test_emits_three_expected_events(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mask = _synthetic_mask()
        settings = Settings()

        with caplog.at_level("INFO"):
            clean_mask_steps_1_to_3(mask, settings)

        messages = {record.message for record in caplog.records}
        assert "cv_cleanup_step1_small_components" in messages
        assert "cv_cleanup_step2_rectilinear" in messages
        assert "cv_cleanup_step3_crop" in messages

    def test_events_match_clean_mask_event_names(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The 3 shared events must have identical names between the two
        functions (symmetric observability — ADR-016's core requirement).
        """
        mask = _synthetic_mask()
        settings = Settings()

        with caplog.at_level("INFO"):
            clean_mask(mask, settings)
        full_messages = {
            record.message
            for record in caplog.records
            if record.message
            in {
                "cv_cleanup_step1_small_components",
                "cv_cleanup_step2_rectilinear",
                "cv_cleanup_step3_crop",
            }
        }
        caplog.clear()

        with caplog.at_level("INFO"):
            clean_mask_steps_1_to_3(mask, settings)
        partial_messages = {
            record.message
            for record in caplog.records
            if record.message
            in {
                "cv_cleanup_step1_small_components",
                "cv_cleanup_step2_rectilinear",
                "cv_cleanup_step3_crop",
            }
        }

        assert (
            full_messages
            == partial_messages
            == {
                "cv_cleanup_step1_small_components",
                "cv_cleanup_step2_rectilinear",
                "cv_cleanup_step3_crop",
            }
        )


class TestStep2CanonicalFields:
    """AC-8: step2 log carries a canonical closed-set branch + required fields."""

    def test_step2_branch_is_canonical_and_fields_present(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mask = _synthetic_mask()
        settings = Settings()

        with caplog.at_level("INFO"):
            clean_mask_steps_1_to_3(mask, settings)

        step2_records = [
            record
            for record in caplog.records
            if record.message == "cv_cleanup_step2_rectilinear"
        ]
        assert len(step2_records) == 1
        record = step2_records[0]

        assert record.branch in _CANONICAL_BRANCHES  # type: ignore[attr-defined]
        # Required fields per AC-8.
        assert hasattr(record, "long_side")
        assert hasattr(record, "min_hw")
        assert hasattr(record, "upscale_target_px")
        assert hasattr(record, "rectilinear_len_px_used")
        assert hasattr(record, "min_len_px")

    def test_step2_branch_canonical_in_clean_mask_too(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Symmetry check: clean_mask's step2 event uses the same canonical
        branch values as clean_mask_steps_1_to_3's.
        """
        mask = _synthetic_mask()
        settings = Settings()

        with caplog.at_level("INFO"):
            clean_mask(mask, settings)

        step2_records = [
            record
            for record in caplog.records
            if record.message == "cv_cleanup_step2_rectilinear"
        ]
        assert len(step2_records) == 1
        assert step2_records[0].branch in _CANONICAL_BRANCHES  # type: ignore[attr-defined]


class TestStep3CanonicalFields:
    """AC-8: step3 log carries significant_components_count and crop_bbox_xywh."""

    def test_step3_fields_present(self, caplog: pytest.LogCaptureFixture) -> None:
        mask = _synthetic_mask()
        settings = Settings()

        with caplog.at_level("INFO"):
            clean_mask_steps_1_to_3(mask, settings)

        step3_records = [
            record
            for record in caplog.records
            if record.message == "cv_cleanup_step3_crop"
        ]
        assert len(step3_records) == 1
        record = step3_records[0]

        assert hasattr(record, "significant_components_count")
        assert hasattr(record, "crop_bbox_xywh")
        assert record.significant_components_count >= 0

    def test_step3_fields_present_in_clean_mask_too(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mask = _synthetic_mask()
        settings = Settings()

        with caplog.at_level("INFO"):
            clean_mask(mask, settings)

        step3_records = [
            record
            for record in caplog.records
            if record.message == "cv_cleanup_step3_crop"
        ]
        assert len(step3_records) == 1
        record = step3_records[0]

        assert hasattr(record, "significant_components_count")
        assert hasattr(record, "crop_bbox_xywh")


class TestNoRegressionOnOutputMask:
    """This task is observability-only — the output mask for
    clean_mask_steps_1_to_3 must be deterministic and unaffected by logging.
    """

    def test_output_mask_deterministic_across_calls(self) -> None:
        mask = _synthetic_mask()
        settings = Settings()

        result_a = clean_mask_steps_1_to_3(mask, settings)
        result_b = clean_mask_steps_1_to_3(mask, settings)

        np.testing.assert_array_equal(result_a, result_b)
