---
name: "CV09-04-filtro-diagonal-residual-doble-pase"
type: "implementation"
priority: "MEDIUM"
agent: "developer-backend"
points: 5
milestone: "run-09-cv"
feature_id: "CV09"
dependencies: ["CV09-01-nuevas-env-vars-settings"]
inputs:
  - "settings.cv_wall_diagonal_filter_enabled: bool (ya existe, master switch)"
  - "settings.cv_wall_diagonal_filter_low_deg / high_deg: float (ya existen)"
  - "settings.cv_wall_min_diagonal_len_px: int (nuevo — CV09-01)"
outputs:
  - "Pase 2 de filtro diagonal tras _fuse_junctions(): Mec.1 (re-filtro por ángulo) + Mec.2 (filtro por longitud mínima)"
validation_rules:
  - "Mec.1: descarta Wall con ángulo en [low_deg, high_deg] tras fuse_junctions"
  - "Mec.2: descarta Wall no-H/V-exacto con longitud < cv_wall_min_diagonal_len_px"
  - "ambos gated por cv_wall_diagonal_filter_enabled"
---

# CV09-04-filtro-diagonal-residual-doble-pase

## Objetivo
Eliminar el segmento/triángulo diagonal residual que persiste en una esquina del resultado (plan-002/004/005) pese al filtro diagonal existente, agregando un segundo pase con dos mecanismos complementarios tras la consolidación de muros.

## Contexto Técnico
Referencia: ADR-017 (`.project-context/decisions/ADR-017-filtro-diagonal-residual-robusto.md`), spec AC-9, AC-10, AC-11, AC-12.

Archivo: `src/vitrina_cv/engines/opencv_classic.py`. Pipeline actual: `_consolidate_walls()` (filtro diagonal pase 1, ~líneas 1258-1280) → `_snap_walls_orthogonal()` → `_extend_to_intersection()` → `_fuse_junctions()` (~línea 2314).

Agregar un **pase 2** inmediatamente después de `_fuse_junctions()`:
- **Mec.1** (re-filtro por ángulo): re-evaluar el ángulo de cada `Wall` y descartar los que caigan en la misma banda `[cv_wall_diagonal_filter_low_deg, cv_wall_diagonal_filter_high_deg]` del pase 1. Debe ser idempotente sobre muros ya snapeados a 0°/90° exactos (nunca deben caer en la banda).
- **Mec.2** (filtro por longitud): descartar todo `Wall` que (a) no sea H/V exacto tras el snap **y** (b) tenga longitud euclidiana `< cv_wall_min_diagonal_len_px`. NO debe aplicarse a muros H/V exactos bajo ningún caso.

Ambos mecanismos gated por el master switch existente `cv_wall_diagonal_filter_enabled` — con `False`, ni el pase 1 ni el pase 2 se ejecutan (compatibilidad pre-run-08 intacta).

## Interfaces
- Llamado por: pipeline de consolidación de muros (después de `_fuse_junctions()`)
- Llama a: `settings` (Pydantic Settings, CV09-01)

## Criterios de Aceptación
- [ ] AC-9: un stub oblicuo que sobrevive al pase 1 y cae en `[low_deg, high_deg]` tras `_fuse_junctions` se descarta por Mec.1.
- [ ] AC-10: un `Wall` no-H/V-exacto con longitud < `cv_wall_min_diagonal_len_px` se descarta por Mec.2; un muro H/V exacto de longitud corta NO se descarta (no es candidato de Mec.2).
- [ ] AC-11: con `cv_wall_diagonal_filter_enabled=False`, la salida de muros es byte-idéntica a la del comportamiento pre-run-08 (ni pase 1 ni pase 2 corren).
- [ ] AC-12: en una planta rectilínea contigua sin diagonal residual (plan-003), el pase 2 es no-op — cero descartes por ángulo y cero por longitud.
- [ ] El evento de log del pase 2 expone `count` de descartes por ángulo y por longitud como campos separados (ver CV09-05/logging).
