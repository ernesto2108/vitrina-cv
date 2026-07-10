# Contratos — vitrina-cv

last_updated: 2026-07-02

<!-- Fuente canónica: docs/specs/06-extraccion-cv-hibrida/contracts/cv-service.openapi.yaml (ADR-003) -->

## REST API

### Base URL
`http://cv-service.internal` (sidecar en red privada, ADR-010)

### Autenticación
Ninguna — el servicio es interno, en red privada. Los requests llegan solo desde `vitrina` (Go).

### Endpoints

#### GET /health
- **Descripción:** Healthcheck (liveness/readiness)
- **Auth:** ninguna
- **Response 200:** `{ status: "ok", model_loaded: boolean, semantic_model_loaded: boolean | null }`
  - `model_loaded` refleja solo el `GeometryEngine` (gating 200/503 no cambia — run 11-cv-04).
  - `semantic_model_loaded`: `is_ready` del `SemanticEngine` activo (run 11, ADR-004). `null` cuando `CV_SEM_ENGINE` está off — nunca fuerza 503, el track semántico es aditivo/best-effort.
- **Response 503:** `{ error_code: "model_not_loaded", message: string }`

#### POST /preflight
- **Descripción:** Gate de pre-vuelo fail-fast con heurísticas puras de imagen (ADR-005). Sin LLM.
- **Auth:** ninguna
- **Body:** `multipart/form-data` con campo `image` (PNG binario)
- **Response 200:** `PreflightReport`
  ```json
  {
    "is_floor_plan": boolean,
    "resolution_ok": boolean,
    "contrast_ok": boolean,
    "line_density_ok": boolean,
    "orientation": string | null,
    "suggestions": [string]
  }
  ```
- **Response 400:** `{ error_code: "invalid_request", message: string }`
- **Response 422:** `{ error_code: "unprocessable_image", message: string }`
- **Response 503:** `{ error_code: "model_not_loaded", message: string }`

#### POST /extract-geometry
- **Descripción:** Extracción determinista de geometría del plano (ADR-003). Motor CV activo según `CV_ENGINE`.
- **Auth:** ninguna
- **Body:** `multipart/form-data` con campo `image` (PNG binario)
- **Response 200:** `Geometry`
  ```json
  {
    "walls": [{ "start": [x, y], "end": [x, y], "thickness": number | null }],
    "rooms": [{ "polygon": [[x, y], ...], "area_px": number }],
    "openings": [{ "type_candidate": "door"|"window"|"unknown", "bbox": [x,y,w,h], "confidence": 0..1 }],
    "scale": { "px_per_unit": number | null, "unit": string | null, "source": "cotas"|"none" },
    "image_size": { "width": integer, "height": integer },
    "objects": [{ "label": string, "bbox": [x,y,w,h], "confidence": 0..1, "needs_review": boolean, "room_id": string | null, "source": "zeroshot"|"finetuned" }]
  }
  ```
  - Todas las coordenadas en pixeles de la imagen recibida.
  - `walls[].thickness`: cuando `CV_WALL_CENTERLINE_ENABLED=true` (default) contiene el grosor estimado en **píxeles** (`2 x median(distanceTransform samples)`). El consumidor Go multiplica `thickness_px * metersPerPx`. `null` cuando el flag está desactivado (legacy mode).
  - `scale.source="none"` cuando no hay referencias de medida; nunca bloquea la respuesta.
  - `objects[]` (run 11, ADR-004): aditivo, `[]` cuando `CV_SEM_ENGINE` off/vacío (default) — nunca altera walls/rooms/openings/stairs_candidates. Wireado en `api/routers/extract_geometry.py` (11-cv-04): `semantic_engine.detect()` + `merge_semantic()` corren después del pipeline geométrico, best-effort — una excepción en el motor semántico degrada a `objects: []` sin romper el 200.
- **Response 400:** `{ error_code: "invalid_request", message: string }`
- **Response 422:** `{ error_code: "unprocessable_image", message: string }`
- **Response 503:** `{ error_code: "model_not_loaded", message: string }`

### Enum de error_code
`invalid_request` | `unprocessable_image` | `model_not_loaded` — string enum, nunca string libre.

## Contratos internos entre dominios

### GeometryEngine (interfaz de motor CV)
- **Definida en:** `src/vitrina_cv/engines/base.py`
- **Implementada por:** `OpenCVClassicEngine` (Fase 1, stub hasta 06-cv-03), `RasterScanEngine` (futuro, ADR-008)
- **Consumida por:** `api/` (routers de extract-geometry y health)
- **Selección:** variable de entorno `CV_ENGINE` (`opencv` en Fase 1)
- **Factory:** `get_engine(cv_engine: str) -> GeometryEngine` — lanza `ValueError` si el valor es desconocido
- **Señal de readiness:** `engine.is_ready` (bool) → alimenta `model_loaded` en GET /health

### SemanticEngine (interfaz de motor semántico, run 11)
- **Definida en:** `src/vitrina_cv/engines/semantic/base.py`
- **Implementada por:** `ZeroShotSemanticEngine` (Fase A, OWL-ViT, `src/vitrina_cv/engines/semantic/zeroshot.py`); `FineTunedSemanticEngine` (Fase B, deferred)
- **Consumida por:** `api/routers/extract_geometry.py` (11-cv-04) vía `app.state.semantic_engine`, inyectado en el lifespan de `main.py`
- **Selección:** variable de entorno `CV_SEM_ENGINE` (`""`/`"off"` → deshabilitado, `"zeroshot"` → OWL-ViT)
- **Factory:** `get_semantic_engine(cv_sem_engine: str, settings) -> SemanticEngine | None` — retorna `None` en vez de lanzar cuando está off (a diferencia de `get_engine`)
- **Fusión con geometría:** `merge_semantic(objects, rooms, walls, openings) -> list[SemanticObject]` (`src/vitrina_cv/engines/semantic/merge.py`) — dedup contra `Opening` por IoU y resolución de `room_id` por centroide, nunca muta walls/rooms/openings
- **Señal de readiness:** `semantic_engine.is_ready` → alimenta `semantic_model_loaded` en GET /health (no participa del gating 200/503, que sigue atado solo a `GeometryEngine.is_ready`)

## Message Queues / Event Streams

No aplica — el servicio es stateless REST puro.

## Servicios externos

El servicio no consume servicios externos — es stateless y solo procesa la imagen recibida en el request.
