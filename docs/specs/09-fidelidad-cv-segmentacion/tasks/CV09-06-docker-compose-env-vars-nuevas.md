---
name: "CV09-06-docker-compose-env-vars-nuevas"
type: "integration"
priority: "MEDIUM"
agent: "devops"
points: 1
milestone: "run-09-cv"
feature_id: "CV09"
dependencies: ["CV09-01-nuevas-env-vars-settings"]
outputs:
  - "docker-compose-local.yml (repo vitrina) documenta las 3 env vars nuevas del servicio cv-service"
---

# CV09-06-docker-compose-env-vars-nuevas

## Objetivo
Documentar y fijar explícitamente en `docker-compose-local.yml` (repo `vitrina`, no `vitrina-cv`) las 3 variables de entorno nuevas introducidas por este run, para que el entorno local use los mismos defaults calibrados que el código.

## Contexto Técnico
Archivo: `/Users/ernestodiaz/projects/vitrina/docker-compose-local.yml`, sección `environment:` del servicio `cv-service` (ya lista `CV_ENGINE`, `CV_UPSCALE_TARGET_PX`, `CV_WALL_DIAGONAL_FILTER_LOW_DEG`, etc. — seguir el mismo patrón de bloque plano `KEY: "value"`).

Agregar:
```yaml
CV_CLEANUP_RECTILINEAR_MIN_LEN_PX: "50"
CV_CLEANUP_CROP_MIN_AREA_RATIO: "0.05"
CV_WALL_MIN_DIAGONAL_LEN_PX: "40"
```

**Nota:** `CV_WALL_MIN_DIAGONAL_LEN_PX` tiene su default sin calibrar contra los fixtures reales (riesgo documentado en ADR-017) — si CV09-11 (gate E2E) determina un valor distinto tras calibración, actualizar este archivo en consecuencia antes de cerrar el run.

## Interfaces
- Consumido por: servicio `cv-service` (contenedor Docker) al arrancar

## Criterios de Aceptación
- [ ] Las 3 variables aparecen en el bloque `environment:` de `cv-service` en `docker-compose-local.yml`.
- [ ] `make redeploy-cv` levanta el servicio sin errores de validación de Settings (Pydantic) con los nuevos valores.
