"""Offline evaluation harness for the vitrina-cv geometry extraction engine.

Execution:
    uv run python eval/run_eval.py

Discovers every eval/dataset/<plan_id>/ directory (each must contain
image.png + ground_truth.json), invokes the configured engine in-process via
get_engine(settings.cv_engine) + engine.extract(image_bytes), computes
per-plan metrics and prints a human-readable report to stdout.

Exit code is always 0 — this script is a measurement tool, not a CI gate.

--- Scoring formula (documented here as authoritative) ---

Per-plan score ∈ [0, 1] is a weighted combination:

  room_score   = 1 - clamp(|detected - expected| / max(expected, 1), 0, 1)
  area_score   = 1 - clamp(mean_relative_area_error, 0, 1)
                 (only when scale.source != "none" AND room_areas_m2 is present)
  fp_penalty   = clamp(false_positives / max(expected_rooms, 1), 0, 1) * 0.3

  If area metric is available:
      score = 0.5 * room_score + 0.5 * area_score - fp_penalty
  Else:
      score = room_score - fp_penalty

  score = clamp(score, 0.0, 1.0)

Rationale:
  - room_score is the primary signal; rooms are the most critical output.
  - area_score is a secondary quality check, available only when scale is known.
  - fp_penalty deducts from the score proportionally to over-detection, weighted
    at 0.3 so a plan with 2x false positives loses ~0.3 points maximum.
  - Clamping prevents arithmetic underflow producing negative scores.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

# Ensure the installed package is importable when running from repo root.
# uv run sets PYTHONPATH=src automatically via pyproject.toml tool.uv config;
# this fallback makes the script runnable with plain `python eval/run_eval.py`
# too, as long as src/ is in the project root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vitrina_cv.config.settings import get_settings  # noqa: E402
from vitrina_cv.engines.base import get_engine  # noqa: E402

if TYPE_CHECKING:
    from vitrina_cv.models import Geometry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATASET_DIR = Path(__file__).resolve().parent / "dataset"
_GT_FILENAME = "ground_truth.json"
_IMAGE_FILENAME = "image.png"

# Score formula weights (see module docstring for full formula).
_AREA_WEIGHT = 0.5
_FP_PENALTY_WEIGHT = 0.3

# Column widths for the per-plan table.
_COL_ID = 36
_COL_ROOMS_EXP = 9
_COL_ROOMS_DET = 9
_COL_DELTA = 7
_COL_FP = 6
_COL_AREA_ERR = 18
_COL_SCORE = 7


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GroundTruth:
    """Parsed ground_truth.json for a single plan."""

    plan_id: str
    expected_rooms: int
    room_areas_m2: list[float] | None
    expected_doors: int
    expected_windows: int
    notes: str = ""


@dataclass
class PlanResult:
    """Evaluation result for one plan."""

    plan_id: str
    expected_rooms: int
    detected_rooms: int
    false_positives: int
    mean_area_error: float | None  # None when area metric is not available
    score: float
    error: str | None = None  # set when extraction raised an exception


@dataclass
class AggregateResult:
    """Aggregate metrics across all evaluated plans."""

    total_plans: int = 0
    plans_with_area: int = 0
    mean_room_delta: float = 0.0
    mean_area_error: float | None = None
    mean_score: float = 0.0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------


def _load_ground_truth(plan_dir: Path) -> GroundTruth:
    """Load and parse ground_truth.json from a plan directory."""
    gt_path = plan_dir / _GT_FILENAME
    if not gt_path.exists():
        msg = f"ground_truth.json not found: {gt_path}"
        raise FileNotFoundError(msg)

    with gt_path.open(encoding="utf-8") as fh:
        raw: dict = json.load(fh)

    return GroundTruth(
        plan_id=raw["plan_id"],
        expected_rooms=int(raw["expected_rooms"]),
        room_areas_m2=raw.get("room_areas_m2"),  # None if absent or null
        expected_doors=int(raw.get("expected_doors", 0)),
        expected_windows=int(raw.get("expected_windows", 0)),
        notes=raw.get("notes", ""),
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _mean_relative_area_error(
    detected_areas_px: list[float],
    expected_areas_m2: list[float],
    px_per_unit: float,
) -> float:
    """Compute mean relative area error between detected and expected rooms.

    Detected areas are in px²; expected areas are in m².  The conversion uses
    px_per_unit (px / m) from scale: area_m2 = area_px / px_per_unit².

    We sort both lists (descending by area) and pair them positionally — the
    largest detected room maps to the largest expected room, etc.  This avoids
    needing polygon-to-polygon correspondence.

    Excess detected rooms (false positives) are not penalised here — they are
    already penalised via the fp_penalty in the score formula.
    """
    if not detected_areas_px or not expected_areas_m2 or px_per_unit <= 0:
        return 1.0  # worst case: no data

    detected_m2 = sorted(
        [a / (px_per_unit**2) for a in detected_areas_px], reverse=True
    )
    expected_sorted = sorted(expected_areas_m2, reverse=True)

    # Pair up to the shorter list length.
    pairs = min(len(detected_m2), len(expected_sorted))
    errors: list[float] = []
    for i in range(pairs):
        exp = expected_sorted[i]
        det = detected_m2[i]
        if exp > 0:
            errors.append(abs(det - exp) / exp)

    return sum(errors) / len(errors) if errors else 1.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _compute_score(
    detected_rooms: int,
    expected_rooms: int,
    false_positives: int,
    mean_area_error: float | None,
) -> float:
    """Compute the per-plan score ∈ [0, 1].  See module docstring for formula."""
    room_score = 1.0 - _clamp(
        abs(detected_rooms - expected_rooms) / max(expected_rooms, 1), 0.0, 1.0
    )
    fp_penalty = (
        _clamp(false_positives / max(expected_rooms, 1), 0.0, 1.0) * _FP_PENALTY_WEIGHT
    )

    if mean_area_error is not None:
        area_score = 1.0 - _clamp(mean_area_error, 0.0, 1.0)
        raw = _AREA_WEIGHT * room_score + _AREA_WEIGHT * area_score - fp_penalty
    else:
        raw = room_score - fp_penalty

    return _clamp(raw, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Per-plan evaluation
# ---------------------------------------------------------------------------


def _evaluate_plan(
    plan_dir: Path,
    engine_name: str,
) -> PlanResult:
    """Run extraction on one plan and return its PlanResult."""
    gt = _load_ground_truth(plan_dir)
    image_path = plan_dir / _IMAGE_FILENAME
    if not image_path.exists():
        return PlanResult(
            plan_id=gt.plan_id,
            expected_rooms=gt.expected_rooms,
            detected_rooms=0,
            false_positives=0,
            mean_area_error=None,
            score=0.0,
            error=f"image.png not found: {image_path}",
        )

    image_bytes = image_path.read_bytes()

    # Instantiate engine in-process via the factory (ADR-008).
    # Settings are loaded fresh per plan so env overrides apply.
    settings = get_settings()
    engine = get_engine(engine_name, settings)

    try:
        geometry: Geometry = engine.extract(image_bytes)
    except Exception as exc:
        return PlanResult(
            plan_id=gt.plan_id,
            expected_rooms=gt.expected_rooms,
            detected_rooms=0,
            false_positives=0,
            mean_area_error=None,
            score=0.0,
            error=f"extract() raised: {exc}",
        )

    detected_rooms = len(geometry.rooms)
    false_positives = max(0, detected_rooms - gt.expected_rooms)

    # Area error: only when scale is known AND ground truth has room_areas_m2.
    mean_area_error: float | None = None
    if (
        geometry.scale.source != "none"
        and geometry.scale.px_per_unit is not None
        and gt.room_areas_m2 is not None
        and len(gt.room_areas_m2) > 0
    ):
        detected_px_areas = [r.area_px for r in geometry.rooms]
        mean_area_error = _mean_relative_area_error(
            detected_px_areas,
            gt.room_areas_m2,
            geometry.scale.px_per_unit,
        )

    score = _compute_score(
        detected_rooms=detected_rooms,
        expected_rooms=gt.expected_rooms,
        false_positives=false_positives,
        mean_area_error=mean_area_error,
    )

    return PlanResult(
        plan_id=gt.plan_id,
        expected_rooms=gt.expected_rooms,
        detected_rooms=detected_rooms,
        false_positives=false_positives,
        mean_area_error=mean_area_error,
        score=score,
    )


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _area_err_str(value: float | None) -> str:
    if value is None:
        return "n/a (sin escala)"
    return f"{value * 100:.1f}%"


def _score_str(score: float) -> str:
    return f"{score:.3f}"


def _print_separator(char: str = "-") -> None:
    print(char * 100)


def _print_header() -> None:
    _print_separator("=")
    print("  vitrina-cv  |  Harness de evaluación offline  |  ADR-012")
    _print_separator("=")


def _print_table(results: list[PlanResult]) -> None:
    hdr = (
        f"{'Plan ID':<{_COL_ID}}"
        f"{'Esp.':>{_COL_ROOMS_EXP}}"
        f"{'Det.':>{_COL_ROOMS_DET}}"
        f"{'Delta':>{_COL_DELTA}}"
        f"{'FP':>{_COL_FP}}"
        f"{'Área err':>{_COL_AREA_ERR}}"
        f"{'Score':>{_COL_SCORE}}"
    )
    print()
    print(hdr)
    _print_separator()
    for r in results:
        if r.error:
            print(f"{'  [ERROR] ' + r.plan_id:<{_COL_ID}}  {r.error}")
            continue
        delta = r.detected_rooms - r.expected_rooms
        delta_str = f"{delta:+d}"
        print(
            f"{r.plan_id:<{_COL_ID}}"
            f"{r.expected_rooms:>{_COL_ROOMS_EXP}}"
            f"{r.detected_rooms:>{_COL_ROOMS_DET}}"
            f"{delta_str:>{_COL_DELTA}}"
            f"{r.false_positives:>{_COL_FP}}"
            f"{_area_err_str(r.mean_area_error):>{_COL_AREA_ERR}}"
            f"{_score_str(r.score):>{_COL_SCORE}}"
        )
    _print_separator()


def _print_aggregate(agg: AggregateResult) -> None:
    print()
    print("  AGREGADO")
    _print_separator("-")
    print(f"  Planos evaluados    : {agg.total_plans}")
    print(f"  Planos con área     : {agg.plans_with_area}")
    print(f"  Delta rooms (media) : {agg.mean_room_delta:+.2f}")
    if agg.mean_area_error is not None:
        print(f"  Error área (media)  : {agg.mean_area_error * 100:.1f}%")
    else:
        print("  Error área (media)  : n/a (sin planos con escala)")
    print(f"  Score (media)       : {agg.mean_score:.3f}")
    if agg.errors:
        print()
        print(f"  Errores ({len(agg.errors)}):")
        for e in agg.errors:
            print(f"    - {e}")
    _print_separator("=")
    print()


def _build_aggregate(results: list[PlanResult]) -> AggregateResult:
    agg = AggregateResult(total_plans=len(results))
    room_deltas: list[float] = []
    area_errors: list[float] = []
    scores: list[float] = []

    for r in results:
        if r.error:
            agg.errors.append(f"{r.plan_id}: {r.error}")
            continue
        room_deltas.append(float(r.detected_rooms - r.expected_rooms))
        if r.mean_area_error is not None:
            area_errors.append(r.mean_area_error)
            agg.plans_with_area += 1
        scores.append(r.score)

    if room_deltas:
        agg.mean_room_delta = sum(room_deltas) / len(room_deltas)
    if area_errors:
        agg.mean_area_error = sum(area_errors) / len(area_errors)
    if scores:
        agg.mean_score = sum(scores) / len(scores)

    return agg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Discover dataset plans, run evaluation, print report."""
    _print_header()

    if not _DATASET_DIR.exists():
        print(f"\n[ERROR] Dataset directory not found: {_DATASET_DIR}")
        print("  Create eval/dataset/<plan_id>/image.png + ground_truth.json first.")
        sys.exit(0)

    plan_dirs = sorted(
        d for d in _DATASET_DIR.iterdir() if d.is_dir() and (d / _GT_FILENAME).exists()
    )

    if not plan_dirs:
        print(f"\n[WARN] No plan directories found under {_DATASET_DIR}")
        print("  Each plan dir must contain image.png and ground_truth.json.")
        sys.exit(0)

    settings = get_settings()
    engine_name = settings.cv_engine
    print(f"\n  Engine  : {engine_name}")
    print(f"  Dataset : {_DATASET_DIR}")
    print(f"  Plans   : {len(plan_dirs)}")

    print("\n  Extracting geometry (in-process, no HTTP)...")

    results: list[PlanResult] = []
    for plan_dir in plan_dirs:
        print(f"    -> {plan_dir.name} ... ", end="", flush=True)
        result = _evaluate_plan(plan_dir, engine_name)
        status = (
            f"ERROR: {result.error}" if result.error else f"score={result.score:.3f}"
        )
        print(status)
        results.append(result)

    print()
    print(
        "  Columnas: "
        "Esp.=Rooms esperados, Det.=Rooms detectados, Delta=Diferencia, "
        "FP=Falsos positivos, Área err=Error relativo de área (solo con escala), "
        "Score=[0,1]"
    )
    _print_table(results)
    agg = _build_aggregate(results)
    _print_aggregate(agg)


if __name__ == "__main__":
    main()
