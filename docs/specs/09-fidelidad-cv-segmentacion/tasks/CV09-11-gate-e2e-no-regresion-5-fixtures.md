---
name: "CV09-11-gate-e2e-no-regresion-5-fixtures"
type: "validation"
priority: "HIGH"
agent: "tester"
points: 3
milestone: "run-09-cv"
feature_id: "CV09"
dependencies:
  - "CV09-02-kernel-rectilineo-adaptativo-baja-res"
  - "CV09-03-crop-multi-componente-envolvente"
  - "CV09-04-filtro-diagonal-residual-doble-pase"
  - "CV09-06-docker-compose-env-vars-nuevas"
---

# CV09-11-gate-e2e-no-regresion-5-fixtures

## Objetivo
Verificar de punta a punta, con los 3 fixes aplicados y desplegados, que `plan-001` mejora sustancialmente y que `plan-002/003/004/005` no regresionan — gate final de aceptación del run antes de cerrarlo.

## Contexto Técnico
Spec AC-13. Harness: `vitrina-web/scripts/e2e/fidelity-loop.mjs` (`npm run e2e:fidelity`) contra los 5 fixtures de `vitrina-cv/eval/dataset/`. Requiere `docker compose -f docker-compose-local.yml up --build -d cv-service` (repo `vitrina`, `make redeploy-cv`) con las env vars de CV09-06 aplicadas, más backend Go, KrakenD y frontend levantados (ver sesión previa para el procedimiento de arranque de los 5 repos).

Comparar resultados contra `ground_truth.json` de cada fixture en `eval/dataset/`. Usar también el par `{fixture}-original.png` / `{fixture}-result.png` que ya guarda `fidelity-loop.mjs` por corrida (`e2e-results/run-{timestamp}/`) para inspección visual manual si algún AC cuantitativo es ambiguo.

**Precondición de la spec (pendiente de verificar antes de esta task):** confirmar las resoluciones reales de `plan-002`, `plan-003`, `plan-005` — el spec las marca "por verificar".

**Nota de calibración:** el default de `CV_WALL_MIN_DIAGONAL_LEN_PX` (40px) no está calibrado contra los fixtures reales — si esta task detecta que el diagonal residual persiste o que se pierden muros legítimos, ajustar el valor en `docker-compose-local.yml` (CV09-06) y re-correr antes de dar el gate por cerrado.

## Interfaces
- Llamado por: humano / CI, al cierre del run
- Llama a: `cv-service` (Docker), backend Go, `fidelity-loop.mjs`

## Criterios de Aceptación
- [ ] AC-13: `plan-001-denso-achurado` pasa de 1 room detectada a 12 (o el número real del ground truth), con áreas dentro de tolerancia.
- [ ] `plan-002-simple-limpio`, `plan-003-reticula-cotas`, `plan-004-sintetico-alta-res`, `plan-005-amueblado-limpio` mantienen su `expected_rooms` y `room_areas_m2` dentro de tolerancia respecto a su `ground_truth.json` (sin regresión).
- [ ] `plan-003-reticula-cotas` recupera la zona de estacionamiento en el resultado renderizado (inspección visual del par original/result).
- [ ] `plan-002`, `plan-004`, `plan-005` ya no muestran el triángulo diagonal residual en el resultado renderizado.
- [ ] La corrida completa de `npm run e2e:fidelity` cierra en verde (exit code 0, sin regresiones reportadas).
