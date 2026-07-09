---
name: "CV09-05-logging-simetrico-clean-mask-steps"
type: "implementation"
priority: "MEDIUM"
agent: "developer-backend"
points: 2
milestone: "run-09-cv"
feature_id: "CV09"
dependencies: ["CV09-02-kernel-rectilineo-adaptativo-baja-res", "CV09-03-crop-multi-componente-envolvente"]
outputs:
  - "clean_mask_steps_1_to_3() emite los mismos logs INFO estructurados por paso que clean_mask()"
validation_rules:
  - "step2 log incluye branch en {fixed|adaptive|skip}"
  - "step3 log incluye significant_components_count y crop_bbox_xywh"
---

# CV09-05-logging-simetrico-clean-mask-steps

## Objetivo
Que la ruta de detección de ventanas/escaleras (`clean_mask_steps_1_to_3`, usada por `_detect_window_pattern` y `_detect_stairs_candidates`) deje de ser una caja negra de diagnóstico, emitiendo los mismos logs estructurados por paso que ya emite `clean_mask()`.

## Contexto Técnico
Referencia: ADR-016 (`.project-context/decisions/ADR-016-trazabilidad-diagnostica-pipeline-cleanup.md`), spec AC-7, AC-8. Es un ADR de **observabilidad pura — cero cambio de comportamiento**, solo agrega/uniformiza logs.

Archivo: `src/vitrina_cv/mask_cleanup.py`, funciones `clean_mask` (~líneas 440-550) y `clean_mask_steps_1_to_3` (~líneas 587-611).

Eventos a emitir simétricamente en ambas funciones (nivel INFO, vía `extra={...}`, valores como enums/campos — nunca prosa):
- `cv_cleanup_step1_small_components`
- `cv_cleanup_step2_rectilinear` — incluye `branch` ∈ {`fixed`, `adaptive`, `skip`}, `long_side`, `min_hw`, `upscale_target_px`, `rectilinear_len_px_used`, `min_len_px`.
- `cv_cleanup_step3_crop` — incluye `significant_components_count`, `crop_bbox_xywh` (envolvente, ver CV09-03).

El developer puede extraer un helper compartido de logging entre ambas funciones si lo considera apropiado — el ADR no lo prescribe, solo fija los nombres de evento y el conjunto cerrado de `branch`.

## Interfaces
- Llamado por: `_detect_window_pattern()`, `_detect_stairs_candidates()` (vía `clean_mask_steps_1_to_3`)
- Depende de: campos `branch`/`rectilinear_len_px_used` (CV09-02) y `significant_components_count`/`crop_bbox_xywh` (CV09-03) ya calculados en el pipeline

## Criterios de Aceptación
- [ ] AC-7: cualquier extracción que recorre `clean_mask_steps_1_to_3` emite los mismos 3 eventos INFO por paso que `clean_mask`.
- [ ] AC-8: el log de step2 incluye `branch` (valor cerrado, no string libre) + los 5 insumos de la fórmula; el log de step3 incluye `significant_components_count` y `crop_bbox_xywh`.
- [ ] Ningún branch, umbral ni geometría cambia por causa de esta task — es puramente observabilidad (verificar con test de regresión de máscara antes/después).
