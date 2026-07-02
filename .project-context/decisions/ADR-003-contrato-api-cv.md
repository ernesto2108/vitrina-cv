# ADR-003 — Contrato API cv-service canónico (REST/JSON, OpenAPI 3.1)

fecha: 2026-07-02
estado: aceptado
fuente: vitrina/docs/specs/06-extraccion-cv-hibrida/adrs/ADR-003-contrato-api-cv.md

## Decisión

Contrato único canónico definido en `cv-service.openapi.yaml` (OpenAPI 3.1), consumido por ambos lados (Go y Python). Endpoints: POST /extract-geometry, POST /preflight, GET /health. Todas las coordenadas en pixeles de la imagen recibida. Errores con `error_code` enum estricto.

## Consecuencias

- OpenAPI y ADR deben mantenerse sincronizados — validar con `api-contract` antes de cada release.
- Las coordenadas en pixel obligan a vitrina a mapear al sistema de coordenadas del plano.
- `error_code` nunca puede ser un string libre.
