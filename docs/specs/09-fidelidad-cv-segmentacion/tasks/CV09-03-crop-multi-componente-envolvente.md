---
name: "CV09-03-crop-multi-componente-envolvente"
type: "implementation"
priority: "HIGH"
agent: "developer-backend"
points: 3
milestone: "run-09-cv"
feature_id: "CV09"
dependencies: ["CV09-01-nuevas-env-vars-settings"]
inputs:
  - "settings.cv_cleanup_crop_min_area_ratio: float (nuevo — CV09-01)"
  - "settings.cv_cleanup_crop_margin_px: int (ya existe)"
outputs:
  - "crop_to_main_component() recorta a la envolvente de todos los componentes significativos, no solo el mayor"
validation_rules:
  - "componente significativo si area >= cv_cleanup_crop_min_area_ratio * A_max"
---

# CV09-03-crop-multi-componente-envolvente

## Objetivo
Que el paso 3 (`crop_to_main_component`) de `clean_mask()` deje de descartar footprints no contiguos (planta en L, cochera separada del bloque principal) al recortar solo al bounding box de la componente conexa más grande.

## Contexto Técnico
Referencia: ADR-015 (`.project-context/decisions/ADR-015-crop-preserva-componentes-multiples.md`), spec AC-4, AC-5, AC-6.

Archivo: `src/vitrina_cv/mask_cleanup.py`, función `crop_to_main_component()` (~líneas 249-305).

Nuevo criterio: un componente es "significativo" si `area >= cv_cleanup_crop_min_area_ratio * A_max` (donde `A_max` es el área del componente mayor). El crop debe recortar a la **envolvente** (unión de bboxes) de todos los componentes significativos + `cv_cleanup_crop_margin_px`, clampeada a los límites de la imagen — no solo al bbox del mayor.

Cuando hay un solo componente significativo, la envolvente debe coincidir exactamente con el bbox del componente mayor (comportamiento idéntico al actual — no-op).

**No-objetivo explícito (del spec):** no implementar la variante "por-componente" (procesar cada componente preservado por separado y fusionar resultados) — es un fallback condicional documentado en el ADR, no se codifica en esta task salvo que el gate de regresión (CV09-11) lo exija.

## Interfaces
- Llamado por: `clean_mask()` (paso 3 del pipeline de limpieza de máscara)
- Llama a: `settings` (Pydantic Settings, CV09-01)

## Criterios de Aceptación
- [ ] AC-4: en un footprint no contiguo con un ala secundaria de área ≥5% de la principal (ej. cochera de plan-003), ambas componentes entran en la envolvente y se preservan.
- [ ] AC-5: en una planta rectangular contigua con un único componente significativo (plan-002), la envolvente == bbox del mayor — sin regresión.
- [ ] AC-6: una cota o marco perimetral que sobrevive como componente pequeño (área < 5% de A_max) NO entra en la envolvente y se descarta como hoy.
- [ ] La máscara resultante sigue siendo un rectángulo contiguo — no rompe los supuestos del pipeline downstream (Hough/CCA/cierre direccional).
