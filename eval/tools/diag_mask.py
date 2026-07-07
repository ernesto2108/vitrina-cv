"""Diagnóstico paso a paso del pipeline de limpieza de máscara (mask_cleanup).

Reproduce en segundos la investigación stepwise que de otro modo hay que
reconstruir a mano en cada ciclo de calibración: máscara cruda → cada paso
de clean_mask con conteo de píxeles → rooms detectados con máscara cruda vs
limpia. Espeja el orden y el gating real de clean_mask (incluida la regla
de resolution_scale para retain_rectilinear y el escalado del umbral de
grosor).

Uso:
    uv run python eval/tools/diag_mask.py eval/dataset/<plan_id>/image.png [...]
    uv run python eval/tools/diag_mask.py --dump-dir /tmp/masks <image.png>

Con --dump-dir escribe PNGs reducidos de la máscara tras cada paso para
inspección visual.

Pitfall que este script evita: get_engine()/los pasos SIN Settings()
desactivan silenciosamente normalización, cleanup y OCR. Aquí Settings()
se pasa siempre.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from vitrina_cv import mask_cleanup as mc
from vitrina_cv.config.settings import Settings, get_settings
from vitrina_cv.engines import opencv_classic as oc
from vitrina_cv.preprocessing import normalize_resolution

_PREVIEW_LONG_SIDE = 1200


def _px_on(mask: NDArray[np.uint8]) -> int:
    return int((mask > 0).sum())


def _dump(mask: NDArray[np.uint8], dump_dir: Path | None, name: str) -> None:
    if dump_dir is None:
        return
    dump_dir.mkdir(parents=True, exist_ok=True)
    h, w = mask.shape[:2]
    factor = _PREVIEW_LONG_SIDE / max(h, w)
    preview = cv2.resize(mask, (max(1, round(w * factor)), max(1, round(h * factor))))
    out = dump_dir / f"{name}.png"
    cv2.imwrite(str(out), preview)
    print(f"      dump -> {out}")


def _rooms_count(mask: NDArray[np.uint8], settings: Settings) -> int:
    closed = oc._build_closed_wall_mask_for_rooms(
        mask,
        close_h_gap_px=settings.cv_room_close_h_gap_px,
        close_v_gap_px=settings.cv_room_close_v_gap_px,
    )
    h, w = mask.shape[:2]
    return len(oc._detect_rooms(closed, h, w))


def diagnose(image_path: Path, settings: Settings, dump_dir: Path | None) -> None:
    """Corre el pipeline de limpieza paso a paso e imprime el efecto de cada uno."""
    print(f"\n{'=' * 72}\nPLAN: {image_path}\n{'=' * 72}")

    data = np.fromfile(str(image_path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        print("  ERROR: no se pudo decodificar la imagen")
        return
    print(f"  original    : {bgr.shape[1]}x{bgr.shape[0]}")
    bgr, factor = normalize_resolution(bgr, settings)
    print(f"  normalizada : {bgr.shape[1]}x{bgr.shape[0]} (upscale x{factor:.2f})")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    mask = oc._build_wall_mask(gray_blur)
    print(f"  máscara cruda ( _build_wall_mask ): {_px_on(mask):>9} px on")
    _dump(mask, dump_dir, f"{image_path.parent.name}-0-raw")

    # Espejo del gating real de clean_mask
    long_side_raw = max(mask.shape[:2])
    resolution_scale_raw = long_side_raw / max(1, settings.cv_upscale_target_px)
    resolution_scale = min(
        resolution_scale_raw, settings.cv_cleanup_rectilinear_max_res_scale
    )

    step1, removed = mc.remove_small_components(
        mask, settings.cv_cleanup_text_max_side_px
    )
    print(
        f"  paso 1 remove_small_components     : {_px_on(step1):>9} px on"
        f"  ({removed} componentes removidos)"
    )
    _dump(step1, dump_dir, f"{image_path.parent.name}-1-small")

    if resolution_scale_raw <= settings.cv_cleanup_rectilinear_max_res_scale:
        step2 = mc.retain_rectilinear(step1, settings.cv_cleanup_rectilinear_len_px)
        print(f"  paso 2 retain_rectilinear          : {_px_on(step2):>9} px on")
    else:
        step2 = step1
        print(
            f"  paso 2 retain_rectilinear          : SALTADO "
            f"(resolution_scale {resolution_scale_raw:.2f} > "
            f"{settings.cv_cleanup_rectilinear_max_res_scale})"
        )
    _dump(step2, dump_dir, f"{image_path.parent.name}-2-rect")

    if settings.cv_cleanup_crop_enabled:
        step3, bbox = mc.crop_to_main_component(
            step2, settings.cv_cleanup_crop_margin_px
        )
        print(
            f"  paso 3 crop_to_main_component      : {_px_on(step3):>9} px on"
            f"  (bbox {bbox})"
        )
    else:
        step3 = step2
        print("  paso 3 crop_to_main_component      : deshabilitado")
    _dump(step3, dump_dir, f"{image_path.parent.name}-3-crop")

    if settings.cv_cleanup_thickness_filter_enabled:
        eff_thickness = max(
            1, round(settings.cv_cleanup_min_wall_thickness_px * resolution_scale)
        )
        step4 = mc.filter_thin_strokes(
            step3,
            eff_thickness,
            settings.cv_cleanup_thickness_preclose_px,
        )
        print(
            f"  paso 4 filter_thin_strokes         : {_px_on(step4):>9} px on"
            f"  (umbral efectivo {eff_thickness} px)"
        )
    else:
        step4 = step3
        print("  paso 4 filter_thin_strokes         : deshabilitado")
    _dump(step4, dump_dir, f"{image_path.parent.name}-4-thin")

    _report_results(mask, step4, settings)


def _report_results(
    mask: NDArray[np.uint8],
    step4: NDArray[np.uint8],
    settings: Settings,
) -> None:
    """Verificación cruzada contra clean_mask real + rooms y grosor de trazo."""
    cleaned_real = mc.clean_mask(mask, settings)
    match = "OK" if _px_on(cleaned_real) == _px_on(step4) else "DIVERGE ⚠️"
    print(
        f"  clean_mask() end-to-end            : {_px_on(cleaned_real):>9} px on"
        f"  [{match}]"
    )

    print(f"  rooms con máscara cruda            : {_rooms_count(mask, settings)}")
    print(
        f"  rooms con máscara limpia           : {_rooms_count(cleaned_real, settings)}"
    )

    # Grosor máximo de trazo — útil para calibrar el umbral del paso 4
    if _px_on(cleaned_real) > 0:
        dist = cv2.distanceTransform(
            (cleaned_real > 0).astype(np.uint8), cv2.DIST_L2, 3
        )
        print(f"  grosor máx de trazo (máscara limpia): {2 * float(dist.max()):.1f} px")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", nargs="+", type=Path, help="Paths de imágenes PNG")
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=None,
        help="Directorio donde escribir previews PNG de la máscara tras cada paso",
    )
    args = parser.parse_args()

    settings = get_settings()
    for image_path in args.images:
        diagnose(image_path, settings, args.dump_dir)


if __name__ == "__main__":
    main()
