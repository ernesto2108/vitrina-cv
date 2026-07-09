---
name: "CV09-01-nuevas-env-vars-settings"
type: "setup"
priority: "HIGH"
agent: "developer-backend"
points: 1
milestone: "run-09-cv"
feature_id: "CV09"
dependencies: []
outputs:
  - "cv_cleanup_rectilinear_min_len_px: Pydantic Settings field (int, gt=0, default 50)"
  - "cv_cleanup_crop_min_area_ratio: Pydantic Settings field (float, ge=0.0, le=1.0, default 0.05)"
  - "cv_wall_min_diagonal_len_px: Pydantic Settings field (int, gt=0, default 40)"
validation_rules:
  - "cv_cleanup_rectilinear_min_len_px: gt=0"
  - "cv_cleanup_crop_min_area_ratio: ge=0.0, le=1.0"
  - "cv_wall_min_diagonal_len_px: gt=0"
---

# CV09-01-nuevas-env-vars-settings

## Objetivo
Declarar en `settings.py` las 3 variables de entorno nuevas que requieren los fixes de ADR-014, ADR-015 y ADR-017, para que las tasks de implementación puedan leerlas sin definir sus propios defaults dispersos.

## Contexto Técnico
Spec: `docs/specs/09-fidelidad-cv-segmentacion/spec-cv-service.md`, sección "Variables de entorno nuevas".

| Variable | Ejemplo | Notas |
|---|---|---|
| `CV_CLEANUP_RECTILINEAR_MIN_LEN_PX` | `50` | Piso del kernel adaptativo del paso 2 (ADR-014). Reemplaza el literal `50` hoy hardcodeado en ambos branches (fijo/adaptativo). |
| `CV_CLEANUP_CROP_MIN_AREA_RATIO` | `0.05` | Ratio de área mínimo para que un componente sea significativo en el crop (ADR-015). |
| `CV_WALL_MIN_DIAGONAL_LEN_PX` | `40` | Longitud euclidiana mínima (px) por debajo de la cual un Wall oblicuo sobreviviente se descarta (ADR-017, Mec.2). Default a calibrar más adelante contra fixtures reales — por ahora usar el valor de ejemplo. |

Seguir el patrón existente de `settings.py` (Pydantic `Field` con `description=` documentando propósito y default, igual que `cv_scale_ocr_consistency_tolerance` o `cv_junction_extend_px`).

## Criterios de Aceptación
- [ ] Las 3 nuevas variables existen como campos de la clase Settings con sus validation_rules (`gt=0`, `ge=0.0/le=1.0`) y defaults documentados.
- [ ] Cada campo tiene `description=` explicando su propósito y a qué ADR pertenece.
- [ ] No se rompe ningún test existente de `settings.py`.
