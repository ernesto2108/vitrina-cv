"""Ad-hoc verification script for 10-cv-07 (NOT part of the pytest suite).

Runs OpenCVClassicEngine.extract() over plan-001-denso-achurado with the real
cv-service overrides (same as test_cv09_11_no_regresion_5_fixtures.py) and
prints wall/room counts plus per-room area in m^2, to determine whether a
fused multi-room component ("room_9" in the original spec measurement,
30.3 m^2) still exists after 10-cv-05 (local thickness) + 10-cv-06 (adaptive
junction extension), both on by default.

Run manually:
    python tests/scratch_cv07_verify_room9_separation.py
"""

# ruff: noqa: T201 -- diagnostic script; prints are the intended output
from __future__ import annotations

import json
from pathlib import Path

from vitrina_cv.config.settings import Settings
from vitrina_cv.engines.opencv_classic import OpenCVClassicEngine

_DATASET = Path(__file__).resolve().parent.parent / "eval" / "dataset"

_REAL_SERVICE_SETTINGS_OVERRIDES = {
    "cv_cleanup_rectilinear_adaptive_enabled": False,
    "cv_cleanup_rectilinear_min_len_px": 50,
    "cv_cleanup_crop_min_area_ratio": 0.05,
    "cv_wall_diagonal_filter_low_deg": 10.0,
    "cv_wall_diagonal_filter_high_deg": 80.0,
    "cv_wall_min_diagonal_len_px": 40,
    "cv_junction_extend_px": 40,
}


def main() -> None:
    plan_id = "plan-001-denso-achurado"
    base = _DATASET / plan_id
    image_path = base / "image.png"
    gt_path = base / "ground_truth.json"

    ground_truth = json.loads(gt_path.read_text())
    settings = Settings(**_REAL_SERVICE_SETTINGS_OVERRIDES)

    print(f"cv_wall_local_thickness_enabled={settings.cv_wall_local_thickness_enabled}")
    print(
        "cv_wall_junction_extend_adaptive_enabled="
        f"{settings.cv_wall_junction_extend_adaptive_enabled}"
    )
    print(f"expected_rooms (ground_truth)={ground_truth.get('expected_rooms')}")

    engine = OpenCVClassicEngine(settings=settings)
    geometry = engine.extract(image_path.read_bytes())

    print(f"\ntotal walls={len(geometry.walls)}")
    print(f"total rooms={len(geometry.rooms)}")

    scale_m_per_px = None
    scale = getattr(geometry, "scale", None)
    if scale is not None and getattr(scale, "source", "none") != "none":
        ppm = getattr(scale, "pixels_per_meter", None)
        if ppm:
            scale_m_per_px = 1.0 / ppm

    print(
        f"scale source={getattr(scale, 'source', None)!r} pixels_per_meter={getattr(scale, 'pixels_per_meter', None)!r}"
    )

    rows = []
    for i, room in enumerate(geometry.rooms):
        polygon = room.polygon
        # shoelace formula, polygon points assumed [x, y] pairs
        area_px = 0.0
        n = len(polygon)
        for j in range(n):
            x1, y1 = polygon[j][0], polygon[j][1]
            x2, y2 = polygon[(j + 1) % n][0], polygon[(j + 1) % n][1]
            area_px += x1 * y2 - x2 * y1
        area_px = abs(area_px) / 2.0

        area_m2 = None
        if scale_m_per_px is not None:
            area_m2 = area_px * (scale_m_per_px**2)

        rows.append((i, getattr(room, "id", i), len(polygon), area_px, area_m2))

    header = f"{'idx':<5}{'id':<12}{'n_vertices':>12}{'area_px2':>14}{'area_m2':>12}"
    print("\n" + header)
    print("-" * len(header))
    for idx, room_id, n_v, area_px, area_m2 in sorted(rows, key=lambda r: -r[3]):
        area_m2_str = f"{area_m2:.1f}" if area_m2 is not None else "n/a"
        print(f"{idx:<5}{room_id!s:<12}{n_v:>12}{area_px:>14.0f}{area_m2_str:>12}")


if __name__ == "__main__":
    main()
