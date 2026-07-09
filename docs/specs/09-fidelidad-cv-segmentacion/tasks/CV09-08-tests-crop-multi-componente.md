---
name: "CV09-08-tests-crop-multi-componente"
type: "validation"
priority: "MEDIUM"
agent: "tester"
points: 2
milestone: "run-09-cv"
feature_id: "CV09"
dependencies: ["CV09-03-crop-multi-componente-envolvente"]
---

# CV09-08-tests-crop-multi-componente

## Objetivo
Cubrir con tests unitarios los 3 criterios de aceptación del crop multi-componente por envolvente (ADR-015).

## Contexto Técnico
Spec: AC-4, AC-5, AC-6. Función bajo test: `crop_to_main_component()` en `src/vitrina_cv/mask_cleanup.py`.

## Criterios de Aceptación
- [ ] Test: máscara sintética con 2 componentes (principal + ala de área 0.4·A_max) → la envolvente incluye ambas.
- [ ] Test: máscara con un único componente significativo → envolvente == bbox del componente mayor (idéntico al comportamiento pre-cambio).
- [ ] Test: máscara con un componente pequeño (área 0.02·A_max, simulando cota/marco) → ese componente queda fuera de la envolvente.
- [ ] Test de integración: `plan-003-reticula-cotas` recupera la zona de estacionamiento contra `ground_truth.json` del fixture.
