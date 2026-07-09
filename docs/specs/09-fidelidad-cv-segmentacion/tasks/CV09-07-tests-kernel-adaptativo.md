---
name: "CV09-07-tests-kernel-adaptativo"
type: "validation"
priority: "MEDIUM"
agent: "tester"
points: 2
milestone: "run-09-cv"
feature_id: "CV09"
dependencies: ["CV09-02-kernel-rectilineo-adaptativo-baja-res"]
---

# CV09-07-tests-kernel-adaptativo

## Objetivo
Cubrir con tests unitarios los 3 criterios de aceptación del kernel rectilíneo adaptativo en baja resolución (ADR-014).

## Contexto Técnico
Spec: AC-1, AC-2, AC-3 (`docs/specs/09-fidelidad-cv-segmentacion/spec-cv-service.md`). Función bajo test: `retain_rectilinear()` / branch de `clean_mask()` en `src/vitrina_cv/mask_cleanup.py`.

## Criterios de Aceptación
- [ ] Test: `min_hw=788, L_base=150, L_min=50, upscale_target=2000` → `L_efectivo == 59`.
- [ ] Test: `min_hw≈2000` → `L_efectivo ≈ L_base` (no-op).
- [ ] Test: branch alta-res (plan-004, resolution_scale_raw > 1.0) produce máscara idéntica a la del comportamiento pre-cambio (snapshot o comparación de hash de máscara).
- [ ] Test de integración: `plan-001-denso-achurado` recupera sus 12 habitaciones contra `ground_truth.json` del fixture (`eval/dataset/plan-001-denso-achurado/`).
