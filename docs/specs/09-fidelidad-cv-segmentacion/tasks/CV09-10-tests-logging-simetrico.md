---
name: "CV09-10-tests-logging-simetrico"
type: "validation"
priority: "LOW"
agent: "tester"
points: 1
milestone: "run-09-cv"
feature_id: "CV09"
dependencies: ["CV09-05-logging-simetrico-clean-mask-steps"]
---

# CV09-10-tests-logging-simetrico

## Objetivo
Cubrir con tests unitarios los 2 criterios de aceptación de la trazabilidad diagnóstica simétrica (ADR-016).

## Contexto Técnico
Spec: AC-7, AC-8. Funciones bajo test: `clean_mask()` y `clean_mask_steps_1_to_3()` en `src/vitrina_cv/mask_cleanup.py`.

## Criterios de Aceptación
- [ ] Test: invocar `clean_mask_steps_1_to_3()` (vía `_detect_window_pattern` o directo) y capturar logs → aparecen los 3 eventos `cv_cleanup_step1_small_components`, `cv_cleanup_step2_rectilinear`, `cv_cleanup_step3_crop`.
- [ ] Test: el log de step2 incluye `branch` con valor en el conjunto cerrado `{fixed, adaptive, skip}` (no string libre) + `long_side`, `min_hw`, `upscale_target_px`, `rectilinear_len_px_used`, `min_len_px`; el log de step3 incluye `significant_components_count` y `crop_bbox_xywh`.
- [ ] Test de no-regresión: comparar la máscara de salida antes/después de este cambio para un mismo fixture — deben ser idénticas (esta task es solo observabilidad).
