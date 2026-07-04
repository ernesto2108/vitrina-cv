"""Unit tests for eval/run_eval.py — ADR-012 evaluation harness.

Import strategy: eval/run_eval.py lives outside src/ by design (it is a
standalone script, not part of the vitrina_cv package).  We insert the repo
root into sys.path so that `import run_eval` resolves to eval/run_eval.py
via importlib.  This mirrors how the script bootstraps itself at startup and
avoids modifying any production file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from types import ModuleType

# ---------------------------------------------------------------------------
# Bootstrap: make `run_eval` importable from eval/run_eval.py
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EVAL_SCRIPT = _REPO_ROOT / "eval" / "run_eval.py"

# Add src/ so that vitrina_cv imports inside run_eval work.
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load_run_eval() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_eval", _EVAL_SCRIPT)
    assert spec is not None, f"Could not locate eval script at {_EVAL_SCRIPT}"
    assert spec.loader is not None, f"No loader for {_EVAL_SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass introspection (cls.__module__) resolves.
    sys.modules["run_eval"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


run_eval = _load_run_eval()

# vitrina_cv is importable now that src/ is in sys.path (bootstrapped above).
from vitrina_cv.models import (  # noqa: E402
    Geometry,
    ImageSize,
    Room,
    Scale,
    ScaleSource,
)

# Convenience aliases to the functions under test.
_clamp = run_eval._clamp
_compute_score = run_eval._compute_score
_mean_relative_area_error = run_eval._mean_relative_area_error
_load_ground_truth = run_eval._load_ground_truth
_evaluate_plan = run_eval._evaluate_plan
GroundTruth = run_eval.GroundTruth
PlanResult = run_eval.PlanResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gt_json(
    plan_id: str = "test-plan",
    expected_rooms: int = 3,
    room_areas_m2=None,
    expected_doors: int = 2,
    expected_windows: int = 1,
    notes: str = "",
) -> dict:
    """Build a minimal valid ground_truth dict."""
    data: dict = {
        "plan_id": plan_id,
        "expected_rooms": expected_rooms,
        "room_areas_m2": room_areas_m2,
        "expected_doors": expected_doors,
        "expected_windows": expected_windows,
    }
    if notes:
        data["notes"] = notes
    return data


def _write_gt(plan_dir: Path, data: dict) -> None:
    (plan_dir / "ground_truth.json").write_text(json.dumps(data), encoding="utf-8")


def _write_dummy_image(plan_dir: Path) -> None:
    """Write a 1-byte placeholder so image_path.exists() is True."""
    (plan_dir / "image.png").write_bytes(b"\x00")


def _make_geometry(
    scale_source: str, px_per_unit: float | None, room_areas_px: list[float]
):
    """Construct a minimal Geometry using real Pydantic models."""
    rooms = [Room(polygon=[], area_px=area) for area in room_areas_px]
    scale = Scale(source=ScaleSource(scale_source), px_per_unit=px_per_unit)
    return Geometry(
        walls=[],
        rooms=rooms,
        openings=[],
        scale=scale,
        image_size=ImageSize(width=100, height=100),
    )


# ---------------------------------------------------------------------------
# 1. Score metric computation
# ---------------------------------------------------------------------------


class TestClamp:
    def test_value_within_range(self):
        assert _clamp(0.5, 0.0, 1.0) == 0.5  # noqa: PLR2004

    def test_value_below_lo(self):
        assert _clamp(-1.0, 0.0, 1.0) == 0.0

    def test_value_above_hi(self):
        assert _clamp(2.0, 0.0, 1.0) == 1.0

    def test_boundary_lo(self):
        assert _clamp(0.0, 0.0, 1.0) == 0.0

    def test_boundary_hi(self):
        assert _clamp(1.0, 0.0, 1.0) == 1.0


class TestComputeScore:
    """Verify the per-plan score formula from the module docstring."""

    def test_perfect_no_area(self):
        # Detected == expected, no FP, no area → score = room_score - 0 = 1.0
        assert _compute_score(4, 4, 0, None) == pytest.approx(1.0)

    def test_perfect_with_perfect_area(self):
        # score = 0.5 * 1.0 + 0.5 * 1.0 - 0 = 1.0
        assert _compute_score(4, 4, 0, 0.0) == pytest.approx(1.0)

    def test_one_room_miss_no_area(self):
        # detected=3, expected=4 → room_score = 1 - 1/4 = 0.75; score = 0.75
        assert _compute_score(3, 4, 0, None) == pytest.approx(0.75)

    def test_false_positives_penalty(self):
        # detected=5, expected=4, fp=1
        # room_score = 1 - clamp(1/4, 0, 1) = 0.75
        # fp_penalty = clamp(1/4, 0, 1) * 0.3 = 0.075
        # score = 0.75 - 0.075 = 0.675
        assert _compute_score(5, 4, 1, None) == pytest.approx(0.675)

    def test_with_area_error_half(self):
        # area_error = 0.5 → area_score = 0.5
        # score = 0.5 * 1.0 + 0.5 * 0.5 - 0 = 0.75
        assert _compute_score(4, 4, 0, 0.5) == pytest.approx(0.75)

    def test_complete_miss_clamped_to_zero(self):
        # detected=0, expected=4 → room_score=0, area_score=0 (worst), fp_penalty=0
        # raw = 0 + 0 - 0 = 0.0 (already ≥ 0)
        assert _compute_score(0, 4, 0, 1.0) == pytest.approx(0.0)

    def test_score_never_negative(self):
        # Extreme: detected=0, expected=4, fp=0, area=1.0 → score clamped to 0
        score = _compute_score(0, 4, 0, 1.0)
        assert score >= 0.0

    def test_expected_zero_uses_max1(self):
        # expected=0 → max(expected,1)=1; detected=0 → room_score=1
        assert _compute_score(0, 0, 0, None) == pytest.approx(1.0)

    def test_massive_fp_penalty_capped_at_0_3(self):
        # fp=100, expected=1 → clamp(100/1, 0,1)*0.3 = 0.3
        # room_score = 1 - clamp(99/1, 0,1) = 0
        # score = clamp(0 - 0.3, 0, 1) = 0.0
        score = _compute_score(100, 1, 99, None)
        assert score == pytest.approx(0.0)


class TestMeanRelativeAreaError:
    def test_perfect_match(self):
        # detected 10000 px², px_per_unit=100 → 10000/10000=1.0 m²; expected=[1.0]
        err = _mean_relative_area_error([10_000.0], [1.0], 100.0)
        assert err == pytest.approx(0.0)

    def test_double_the_area(self):
        # detected 20000 px² → 2.0 m²; expected=1.0 → rel_error=|2-1|/1=1.0
        err = _mean_relative_area_error([20_000.0], [1.0], 100.0)
        assert err == pytest.approx(1.0)

    def test_two_rooms_sorted_descending(self):
        # detected [400, 100] px, px_per_unit=10 → [4.0, 1.0] m²
        # expected [4.0, 1.0] m² → errors [0, 0] → mean=0
        err = _mean_relative_area_error([100.0, 400.0], [1.0, 4.0], 10.0)
        assert err == pytest.approx(0.0)

    def test_empty_detected_returns_worst(self):
        assert _mean_relative_area_error([], [1.0], 100.0) == pytest.approx(1.0)

    def test_empty_expected_returns_worst(self):
        assert _mean_relative_area_error([100.0], [], 100.0) == pytest.approx(1.0)

    def test_zero_px_per_unit_returns_worst(self):
        assert _mean_relative_area_error([100.0], [1.0], 0.0) == pytest.approx(1.0)

    def test_excess_detected_rooms_ignored(self):
        # 3 detected, 2 expected → only 2 pairs compared
        # detected [900, 400, 100] px, px_per_unit=10 → [9, 4, 1] m²
        # expected sorted [5, 2] → pairs: (9,5)→0.8, (4,2)→1.0 → mean=0.9
        err = _mean_relative_area_error([400.0, 900.0, 100.0], [2.0, 5.0], 10.0)
        assert err == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# 2. Ground truth loading and validation
# ---------------------------------------------------------------------------


class TestLoadGroundTruth:
    def test_full_valid_gt(self, tmp_path):
        data = _make_gt_json(
            plan_id="plan-test",
            expected_rooms=5,
            room_areas_m2=[10.0, 20.0],
            expected_doors=3,
            expected_windows=2,
            notes="test note",
        )
        _write_gt(tmp_path, data)
        gt = _load_ground_truth(tmp_path)
        assert gt.plan_id == "plan-test"
        assert gt.expected_rooms == 5  # noqa: PLR2004
        assert gt.room_areas_m2 == [10.0, 20.0]
        assert gt.expected_doors == 3  # noqa: PLR2004
        assert gt.expected_windows == 2  # noqa: PLR2004
        assert gt.notes == "test note"

    def test_null_room_areas_m2(self, tmp_path):
        data = _make_gt_json(room_areas_m2=None)
        _write_gt(tmp_path, data)
        gt = _load_ground_truth(tmp_path)
        assert gt.room_areas_m2 is None

    def test_absent_room_areas_m2_defaults_to_none(self, tmp_path):
        data = _make_gt_json()
        del data["room_areas_m2"]
        _write_gt(tmp_path, data)
        gt = _load_ground_truth(tmp_path)
        assert gt.room_areas_m2 is None

    def test_notes_optional(self, tmp_path):
        data = _make_gt_json()
        data.pop("notes", None)
        _write_gt(tmp_path, data)
        gt = _load_ground_truth(tmp_path)
        assert gt.notes == ""

    def test_expected_doors_and_windows_optional(self, tmp_path):
        data = {
            "plan_id": "p",
            "expected_rooms": 2,
            "room_areas_m2": None,
        }
        _write_gt(tmp_path, data)
        gt = _load_ground_truth(tmp_path)
        assert gt.expected_doors == 0
        assert gt.expected_windows == 0

    def test_missing_plan_id_raises(self, tmp_path):
        data = _make_gt_json()
        del data["plan_id"]
        _write_gt(tmp_path, data)
        with pytest.raises(KeyError):
            _load_ground_truth(tmp_path)

    def test_missing_expected_rooms_raises(self, tmp_path):
        data = _make_gt_json()
        del data["expected_rooms"]
        _write_gt(tmp_path, data)
        with pytest.raises(KeyError):
            _load_ground_truth(tmp_path)

    def test_malformed_json_raises(self, tmp_path):
        (tmp_path / "ground_truth.json").write_text("{not valid json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            _load_ground_truth(tmp_path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_ground_truth(tmp_path)


# ---------------------------------------------------------------------------
# 3. Area metric gating — 4 quadrants (ADR-012 invariant)
# ---------------------------------------------------------------------------


class TestAreaMetricGating:
    """Invariant: area error is computed ONLY when scale.source != 'none'
    AND room_areas_m2 is present in the ground truth.  All other
    combinations → mean_area_error must be None in PlanResult.
    """

    def _run_plan(self, tmp_path, scale_source: str, px_per_unit, room_areas_m2):
        """Run _evaluate_plan with a mocked engine, return PlanResult."""
        data = _make_gt_json(
            plan_id="gate-test",
            expected_rooms=2,
            room_areas_m2=room_areas_m2,
        )
        _write_gt(tmp_path, data)
        _write_dummy_image(tmp_path)

        geometry = _make_geometry(
            scale_source=scale_source,
            px_per_unit=px_per_unit,
            room_areas_px=[10_000.0, 5_000.0],
        )

        mock_engine = MagicMock()
        mock_engine.extract.return_value = geometry

        with patch.object(
            sys.modules["run_eval"], "get_engine", return_value=mock_engine
        ):
            return _evaluate_plan(tmp_path, "mock")

    def test_quadrant_scale_none_areas_present(self, tmp_path):
        """scale.source='none' + areas present → area omitted."""
        result = self._run_plan(
            tmp_path,
            scale_source="none",
            px_per_unit=None,
            room_areas_m2=[10.0, 5.0],
        )
        assert result.mean_area_error is None

    def test_quadrant_scale_known_areas_absent(self, tmp_path):
        """scale.source='cotas' + areas=None → area omitted."""
        result = self._run_plan(
            tmp_path,
            scale_source="cotas",
            px_per_unit=100.0,
            room_areas_m2=None,
        )
        assert result.mean_area_error is None

    def test_quadrant_scale_none_areas_absent(self, tmp_path):
        """scale.source='none' + areas=None → area omitted."""
        result = self._run_plan(
            tmp_path,
            scale_source="none",
            px_per_unit=None,
            room_areas_m2=None,
        )
        assert result.mean_area_error is None

    def test_quadrant_scale_known_areas_present(self, tmp_path):
        """scale.source='cotas' + areas present → area computed (not None)."""
        result = self._run_plan(
            tmp_path,
            scale_source="cotas",
            px_per_unit=100.0,
            room_areas_m2=[1.0, 0.5],
        )
        assert result.mean_area_error is not None
        assert 0.0 <= result.mean_area_error <= 1.0


# ---------------------------------------------------------------------------
# 4. Dataset discovery
# ---------------------------------------------------------------------------


class TestDatasetDiscovery:
    """Verify which plan directories the harness considers 'complete'."""

    def _collect_plan_dirs(self, dataset_dir: Path) -> list[Path]:
        """Mirror the discovery filter used in main()."""
        return sorted(
            d
            for d in dataset_dir.iterdir()
            if d.is_dir() and (d / "ground_truth.json").exists()
        )

    def test_complete_plan_dir_is_discovered(self, tmp_path):
        plan = tmp_path / "plan-a"
        plan.mkdir()
        _write_gt(plan, _make_gt_json(plan_id="plan-a"))
        _write_dummy_image(plan)
        dirs = self._collect_plan_dirs(tmp_path)
        assert plan in dirs

    def test_dir_without_gt_is_skipped(self, tmp_path):
        plan = tmp_path / "plan-b"
        plan.mkdir()
        _write_dummy_image(plan)  # image present, but no ground_truth.json
        dirs = self._collect_plan_dirs(tmp_path)
        assert plan not in dirs

    def test_dir_without_image_still_discovered_but_evaluate_reports_error(
        self, tmp_path
    ):
        """Discovery only requires ground_truth.json; image absence is caught later."""
        plan = tmp_path / "plan-c"
        plan.mkdir()
        _write_gt(plan, _make_gt_json(plan_id="plan-c"))
        # No image.png written
        dirs = self._collect_plan_dirs(tmp_path)
        assert plan in dirs

        # _evaluate_plan should return an error PlanResult, not raise.
        with patch.object(sys.modules["run_eval"], "get_engine"):
            result = _evaluate_plan(plan, "mock")
        assert result.error is not None
        assert "image.png" in result.error

    def test_nested_files_not_treated_as_plan_dirs(self, tmp_path):
        """A regular file inside dataset dir must not appear in plan_dirs."""
        (tmp_path / "README.txt").write_text("ignored")
        dirs = self._collect_plan_dirs(tmp_path)
        assert dirs == []

    def test_multiple_plans_sorted(self, tmp_path):
        for name in ("plan-z", "plan-a", "plan-m"):
            p = tmp_path / name
            p.mkdir()
            _write_gt(p, _make_gt_json(plan_id=name))
        dirs = self._collect_plan_dirs(tmp_path)
        assert [d.name for d in dirs] == ["plan-a", "plan-m", "plan-z"]
