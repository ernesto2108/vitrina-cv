# Dominio: vitrina-cv

last_updated: 2026-07-02

## Responsabilidad

Extraer geometría determinista de planos arquitectónicos PNG (paredes, habitaciones, aberturas candidatas, escala) y evaluar la aptitud de imagen con heurísticas puras. Sin semántica, sin persistencia, sin estado.

## Bounded contexts detectados

### extract-geometry
Procesamiento del plano para producir la geometría completa (`Geometry`): paredes como segmentos, polígonos de habitaciones, aberturas candidatas, escaleras candidatas y escala opcional.

```
api/
├── routers/extract_geometry.py   — handler POST /extract-geometry (por crear)
models.py                         — DTOs Pydantic del contrato (06-cv-02)
engines/
├── base.py                       — interfaz GeometryEngine + factory get_engine() (06-cv-02)
├── opencv_classic.py             — OpenCVClassicEngine stub Fase 1 (06-cv-02; lógica: 06-cv-03)
└── rasterscan.py                 — RasterScanEngine futuro (por crear)
```

**Flujo principal:**
```
POST /extract-geometry (PNG bytes) → GeometryEngine.extract(image) → Geometry → 200
```

### preflight
Evaluación fail-fast de aptitud de imagen con heurísticas deterministas.

```
api/
├── routers/preflight.py          — handler POST /preflight (por crear; task 06-cv-06)
preflight/
└── checks.py                     — heurísticas implementadas (06-cv-05)
```

**Flujo principal:**
```
POST /preflight (PNG bytes) → run_preflight(image_bytes, settings) → PreflightReport → 200
```

**Heurísticas implementadas en `preflight/checks.py` (06-cv-05):**
- `resolution_ok`: `width >= min_width AND height >= min_height` (de `CV_PREFLIGHT_MIN_RESOLUTION`)
- `contrast_ok`: Michelson(p2, p98) >= `CV_PREFLIGHT_MIN_CONTRAST` — robusto a píxeles aislados
- `line_density_ok`: ratio de píxeles de borde Canny >= `CV_PREFLIGHT_MIN_LINE_DENSITY`
- `is_floor_plan`: `resolution_ok AND contrast_ok AND line_density_ok AND _is_rectilinear(gray)` — la foto/selfie falla en rectilinear (líneas no ortogonales)
- `orientation`: bucket HoughLinesP ("horizontal"/"vertical"/"diagonal (N°)") o `None`
- `suggestions[]`: una frase accionable en español por check fallido, sin duplicados

### config
Configuración del servicio desde variables de entorno.

```
config/
└── settings.py                   — CV_ENGINE, CV_MODEL_PATH, CV_PREFLIGHT_* (por crear)
```

## Patrones usados

- **Strategy:** `engines/base.py` — `GeometryEngine` como interfaz intercambiable seleccionada por `CV_ENGINE` (ADR-008)
- **Settings por env:** `config/settings.py` — umbrales de preflight configurables sin tocar código (ADR-005)
- **Centralized DTOs:** `src/vitrina_cv/models.py` — única fuente de verdad de los tipos del contrato; espeja `cv-service.openapi.yaml` (ADR-003)

## DTOs del contrato (models.py)

Todos los DTOs viven en `src/vitrina_cv/models.py` y espejean el OpenAPI exactamente:

| Clase | Campos required | Campos opcionales |
|---|---|---|
| `Geometry` | walls, rooms, openings, scale, image_size | — |
| `Wall` | start, end | thickness |
| `Room` | polygon, area_px | — |
| `Opening` | type_candidate, bbox, confidence | — |
| `Scale` | source | px_per_unit, unit |
| `ImageSize` | width, height | — |
| `PreflightReport` | is_floor_plan, resolution_ok, contrast_ok, line_density_ok, suggestions | orientation |
| `Error` | error_code, message | — |

Enums: `ScaleSource` (cotas/none), `OpeningTypeCandidate` (door/window/unknown), `ErrorCode` (invalid_request/unprocessable_image/model_not_loaded).

## Interfaces públicas

```python
# engines/base.py — implementado en 06-cv-02
from abc import ABC, abstractmethod
from vitrina_cv.models import Geometry

class GeometryEngine(ABC):
    @property
    @abstractmethod
    def is_ready(self) -> bool: ...  # alimenta /health.model_loaded

    @abstractmethod
    def extract(self, image_bytes: bytes) -> Geometry: ...
```

## Dependencias de este dominio

- `opencv-python` — motor CV Fase 1 (OpenCVClassicEngine)
- `fastapi` — framework web
- `pydantic` — validación y serialización de modelos de respuesta

## Quién depende de este dominio

- `vitrina` (Go, repo hermano) — consume `POST /extract-geometry` y `POST /preflight` via `cv-client` interno

## Decisiones tomadas

- D1: Motor intercambiable via `CV_ENGINE` — ver `decisions/ADR-008-cv-opencv-modelo-preentrenado.md`
- D2: Aberturas como candidatas sin decisión semántica — ver `decisions/ADR-009-reparto-puertas-ventanas.md`
- D3: Preflight con heurísticas puras, sin LLM — ver `decisions/ADR-005-gate-pre-vuelo.md`
- D4: Protocolo de comunicación vitrina↔cv (bytes en request, no S3) — ver `decisions/ADR-002-protocolo-vitrina-cv.md`
- D5: Contrato API canónico REST/JSON en OpenAPI 3.1 — ver `decisions/ADR-003-contrato-api-cv.md`

## Gotchas

- `scale` es opcional — su ausencia (`source="none"`) no es un error; nunca bloquear la respuesta por falta de cotas.
- `openings[].type_candidate` es tentativo — el LLM en vitrina decide el tipo final; el servicio no debe filtrar ni priorizar candidatas.
- Las coordenadas están en pixeles de la imagen recibida, no en coordenadas del sistema de vitrina — vitrina hace el mapeo.
- **Close asimétrico para detección de habitaciones:** el cierre morfológico direccional antes de CCA usa kernels H y V independientes (`CV_ROOM_CLOSE_H_GAP_PX=80`, `CV_ROOM_CLOSE_V_GAP_PX=160`). Un kernel H igual al V (160px) destruye habitaciones estrechas (~130px ≈ 1.0m) al rellenar el espacio entre dos paredes verticales paralelas. El kernel H debe ser menor que el ancho de la habitación más estrecha esperada (baños ≥ 0.6m en planos peruanos densos).

## Deuda técnica

- `OpenCVClassicEngine.extract()` es stub — lanza `NotImplementedError` hasta task 06-cv-03/04/05.
- Umbrales de preflight (`CV_PREFLIGHT_MIN_*`) calibrados con imágenes sintéticas; calibración con planos reales en task 06-cv-07.
