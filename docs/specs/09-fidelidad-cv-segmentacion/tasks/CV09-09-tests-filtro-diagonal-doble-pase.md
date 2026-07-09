---
name: "CV09-09-tests-filtro-diagonal-doble-pase"
type: "validation"
priority: "MEDIUM"
agent: "tester"
points: 2
milestone: "run-09-cv"
feature_id: "CV09"
dependencies: ["CV09-04-filtro-diagonal-residual-doble-pase"]
---

# CV09-09-tests-filtro-diagonal-doble-pase

## Objetivo
Cubrir con tests unitarios los 4 criterios de aceptación del filtro de diagonal residual de doble mecanismo (ADR-017).

## Contexto Técnico
Spec: AC-9, AC-10, AC-11, AC-12. Función bajo test: pase 2 del filtro diagonal en `src/vitrina_cv/engines/opencv_classic.py`, tras `_fuse_junctions()`.

## Criterios de Aceptación
- [ ] Test: `Wall` sintético a 45° tras fuse (dentro de `[low_deg, high_deg]`) → descartado por Mec.1.
- [ ] Test: `Wall` sintético a 12°, longitud 30px (< `cv_wall_min_diagonal_len_px` default) → descartado por Mec.2; un `Wall` H/V exacto de 30px → conservado.
- [ ] Test: con `cv_wall_diagonal_filter_enabled=False` → salida de muros byte-idéntica a la de referencia pre-run-08 (ni pase 1 ni pase 2 corren).
- [ ] Test de integración: `plan-003-reticula-cotas` (sin diagonal residual) → pase 2 con `count` por ángulo = 0 y por longitud = 0, mismo número de muros que antes.
- [ ] Test de integración: `plan-002-simple-limpio`, `plan-004-sintetico-alta-res`, `plan-005-amueblado-limpio` → el triángulo diagonal residual desaparece del resultado.
