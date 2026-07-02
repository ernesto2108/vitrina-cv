# Dependencias — vitrina-cv

<!-- Grafo de dependencias entre dominios. -->

last_updated: 2026-07-02

## Grafo de dependencias

<!-- Tipo: sync (llamada directa), async (evento/queue), data (FK / esquema compartido) -->

| Dominio | Depende de | Tipo | Notas |
|---------|-----------|------|-------|
| `api/extract-geometry` | `engines/GeometryEngine` | sync | El router delega al motor activo via interfaz; nunca instancia el concreto directamente |
| `api/preflight` | `preflight/checks` | sync | El router delega a las heurísticas de imagen |
| `preflight/checks` | `config/settings` | sync | Lee umbrales CV_PREFLIGHT_* en startup |
| `engines/opencv_classic` | `opencv-python` | sync | Motor CV Fase 1 |

## Impacto de cambios

Antes de modificar un dominio, consultar la tabla del grafo para identificar los dominios **downstream** afectados (los que dependen del dominio que vas a tocar).

- Si la dependencia es `sync` → un cambio de contrato rompe a quien depende de inmediato.
- Si la dependencia es `async` → verificar compatibilidad del payload del evento.
- Si la dependencia es `data` → verificar migraciones y FKs antes de alterar el esquema.

Listar los dominios downstream en el plan de cambio y validar cada uno antes de cerrar.

**Cambio de alto impacto:** modificar la firma de `GeometryEngine.extract()` afecta todos los motores implementados y el router; requiere coordinación.

## Dependencias externas

| Servicio externo | Tipo | Consumido por | Notas |
|------------------|------|---------------|-------|
| `vitrina` (Go, /Users/ernestodiaz/projects/vitrina) | API REST — productor → consumidor inverso (vitrina llama a cv-service) | `api/` (todos los endpoints) | Contrato canónico: `cv-service.openapi.yaml` (ADR-003); red privada interna |
| `opencv-python>=4.10` | librería Python | `engines/opencv_classic` | Motor CV Fase 1; 5.0+ instalado via uv |
| `numpy>=1.26` | librería Python | `engines/` | Procesamiento de arrays de imagen |
| `fastapi>=0.115` | librería Python | `api/` | Framework HTTP; versión confirmada en scaffold |
| `pydantic>=2.7` + `pydantic-settings>=2.3` | librería Python | `config/`, `api/` | Validación de modelos y settings por env |
| `uvicorn[standard]>=0.30` | librería Python | entrypoint | Servidor ASGI; `uvloop` incluido en `[standard]` |
| `python-multipart>=0.0.9` | librería Python | `api/routers/` | Requerida por FastAPI para parsear `multipart/form-data` (UploadFile); agregada en 06-cv-06 |
| `RasterScan` (futuro) | Docker comercial / REST | `engines/rasterscan` (por crear) | Evaluación en paralelo, sin deadline; no bloquea Fase 1 (ADR-008) |
