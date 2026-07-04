"""Diagnóstico paso a paso del OCR de escala (ADR-011, scale_ocr).

Descompone detect_scale_from_ocr() en sus etapas y muestra dónde muere:
tokens numéricos extraídos por tesseract (con confianza y posición),
líneas de cota detectadas, asociaciones token→línea con su px/m, y el
consenso final. Con esto, "scale.source=none" deja de ser una caja negra.

Uso:
    uv run python eval/tools/diag_ocr.py eval/dataset/<plan_id>/image.png [...]

Interpretación rápida:
  - tokens=0            -> tesseract no lee el texto (¿resolución/tamaño de
                           glifos? los planos web reducidos suelen tener
                           cotas de ~8 px, ilegibles para tesseract)
  - tokens ok, 0 asociaciones -> las líneas de cota no están donde el texto
  - candidatos dispersos, consenso None -> lecturas inconsistentes; el gate
                           de tolerancia (CV_SCALE_OCR_CONSISTENCY_TOLERANCE)
                           las rechaza — comportamiento correcto ante ruido
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from vitrina_cv import scale_ocr
from vitrina_cv.config.settings import Settings, get_settings
from vitrina_cv.preprocessing import normalize_resolution

_MAX_TOKENS_SHOWN = 25


def diagnose(image_path: Path, settings: Settings) -> None:
    """Corre el pipeline OCR por etapas e imprime el resultado de cada una."""
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

    # Etapas 1-2: upscale OCR + extracción de tokens
    gray_ocr, ocr_factor = scale_ocr._upscale_for_ocr(gray)
    print(
        f"  OCR upscale : {gray_ocr.shape[1]}x{gray_ocr.shape[0]}"
        f" (factor x{ocr_factor:.2f})"
    )
    tokens = scale_ocr._extract_numeric_tokens(
        gray_ocr, settings.cv_scale_ocr_tesseract_cmd
    )
    print(f"  tokens numéricos: {len(tokens)}")
    for tok in tokens[:_MAX_TOKENS_SHOWN]:
        print(
            f"    '{tok['text']}' conf={tok['conf']}"
            f" @({float(str(tok['cx'])):.0f},{float(str(tok['cy'])):.0f})"
        )
    if len(tokens) > _MAX_TOKENS_SHOWN:
        print(f"    ... ({len(tokens) - _MAX_TOKENS_SHOWN} más)")

    # Etapa 3: líneas de cota
    segs = scale_ocr._detect_cota_lines(gray)
    print(f"  líneas de cota H/V: {len(segs)}")

    # Etapa 4: asociaciones token -> línea
    candidates: list[float] = []
    for tok in tokens:
        value_m = scale_ocr._infer_unit_and_metres(
            float(str(tok["value"]))
        )
        if value_m is None or not (
            scale_ocr._DIM_MIN_M <= value_m <= scale_ocr._DIM_MAX_M
        ):
            continue
        cx = float(str(tok["cx"])) / ocr_factor
        cy = float(str(tok["cy"])) / ocr_factor
        seg, dist = scale_ocr._find_nearest_line(cx, cy, segs)
        if seg is None:
            print(
                f"    '{tok['text']}' ({value_m} m): SIN línea a"
                f" <{scale_ocr._ASSOC_MAX_DIST_PX:.0f} px"
            )
            continue
        line_len = scale_ocr._segment_length(seg)
        px_per_unit = line_len / value_m
        candidates.append(px_per_unit)
        print(
            f"    '{tok['text']}' ({value_m} m) -> línea {line_len:.0f} px"
            f" dist={dist:.0f} px => {px_per_unit:.1f} px/m"
        )

    # Etapa 5: consenso
    consensus = scale_ocr._consistent_median(
        candidates, settings.cv_scale_ocr_consistency_tolerance
    )
    print(f"  candidatos px/m : {[round(c, 1) for c in candidates]}")
    print(f"  consenso        : {consensus}")

    # Verificación cruzada end-to-end
    result = scale_ocr.detect_scale_from_ocr(gray, settings)
    ppu = round(result.px_per_unit, 2) if result.px_per_unit else None
    print(f"  detect_scale_from_ocr => source={result.source.value}, px_per_unit={ppu}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", nargs="+", type=Path, help="Paths de imágenes PNG")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.cv_scale_ocr_enabled:
        print("AVISO: CV_SCALE_OCR_ENABLED=false — el motor devolvería source=none")
    for image_path in args.images:
        diagnose(image_path, settings)


if __name__ == "__main__":
    main()
