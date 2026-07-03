# Coding Standards — vitrina-cv

<!-- Naming, estructura de carpetas, idioma del código, reglas de linting y patrones prohibidos.
     Complementado con patrones de diseño detectados automáticamente en el código. -->

last_updated: 2026-07-02

## Idioma del código

- **Código fuente:** inglés
- **Comentarios:** inglés
- **Commits:** inglés — Conventional Commits
- **Documentación técnica:** español

## Naming

### General
- **Variables y funciones:** `snake_case`
- **Constantes:** `UPPER_SNAKE_CASE`
- **Archivos:** `snake_case`
- **Clases / Tipos / Pydantic models:** `PascalCase`

### Por dominio
- **Routers FastAPI:** `router_<dominio>.py` o `<dominio>.py` dentro de `api/routers/`
- **Motores CV:** `<nombre>_engine.py` dentro de `engines/`
- **Heurísticas preflight:** `checks.py` dentro de `preflight/`
- **Settings:** `settings.py` dentro de `config/`

## Estructura de carpetas

```
vitrina-cv/
├── api/
│   └── routers/          — handlers FastAPI por endpoint
├── engines/
│   ├── base.py           — interfaz GeometryEngine (ABC)
│   ├── opencv_classic.py — OpenCVClassicEngine (Fase 1)
│   └── rasterscan.py     — RasterScanEngine (futuro)
├── preflight/
│   ├── checks.py         — heurísticas de imagen
│   └── config.py         — umbrales desde env vars
├── config/
│   └── settings.py       — Pydantic BaseSettings
├── tests/                — por definir en scaffold
└── main.py               — entry point FastAPI
```

> Estructura confirmada con el scaffold 06-cv-01. El package vive en `src/vitrina_cv/`.

## Reglas de imports / dependencias

- `api/routers/` NO importa motores concretos directamente — usa la interfaz `GeometryEngine`
- `engines/` NO importa `api/` — dependencia unidireccional
- `preflight/` NO importa `engines/` — son dominios independientes
- No dependencias circulares entre dominios
- Type hints obligatorios en toda función y método

## Linting configurado

| Herramienta | Config | Reglas destacadas |
|---|---|---|
| ruff | `pyproject.toml` `[tool.ruff]` | E,W,F,I,N,UP,B,A,SIM,TCH,RUF,S,PT,C4,DTZ,T20,PIE,PL,PERF,FURB — ignore E501,PLR0913 |
| mypy | `pyproject.toml` `[tool.mypy]` | `strict = true`, `python_version = "3.12"`, `mypy_path = "src"` |

Comandos:
- `ruff check . && ruff format --check .` — lint + formato (verificación)
- `ruff check --fix . && ruff format .` — auto-fix

## Patrones prohibidos

- **Motor concreto instanciado en router:** usar siempre la factory de `engines/`
- **Estado global mutable entre requests:** el servicio es stateless; ningún módulo retiene bytes o datos de requests anteriores
- **Llamada a LLM o servicio externo en `/preflight`:** solo heurísticas de imagen deterministas
- **Import de boto3/S3/DB client en cualquier módulo:** el servicio no accede a infra externa

## Patrones de diseño detectados en el código

<!-- Inferidos desde el diseño arquitectónico — repo aún vacío; confirmar en el primer scaffold -->

### Strategy — GeometryEngine
- **Archivo:** `src/vitrina_cv/engines/base.py`
- **Qué hace:** define la interfaz intercambiable para motores CV; `is_ready` property alimenta `/health.model_loaded`
- **Cuándo usar:** al agregar un nuevo motor CV
- **Anti-pattern:** no instanciar motores concretos fuera de `get_engine()`

### DTOs centralizados — models.py
- **Archivo:** `src/vitrina_cv/models.py`
- **Qué hace:** única fuente de verdad de los tipos Pydantic v2 del contrato; espeja `cv-service.openapi.yaml` exactamente
- **Cuándo usar:** siempre importar desde aquí — nunca definir modelos de contrato inline en routers o engines
- **Anti-pattern:** duplicar un DTO en otro módulo; agregar campos no presentes en el OpenAPI

### StrEnum en lugar de str+Enum
- **Archivo:** `src/vitrina_cv/models.py`
- **Qué hace:** hereda de `StrEnum` (Python 3.11+) para enums serializables como string JSON
- **Cuándo usar:** toda enum de dominio que viaje en requests/responses JSON
- **Anti-pattern:** `class MyEnum(str, Enum)` — usar `class MyEnum(StrEnum)` (UP042)

### Heurísticas preflight como funciones privadas con constantes nombradas — preflight/checks.py
- **Archivo:** `src/vitrina_cv/preflight/checks.py`
- **Qué hace:** cada check es una función `_check_*` pura (toma `np.ndarray` + umbral, devuelve `bool`); los ángulos y ratios umbral son constantes de módulo con nombre (`_HORIZONTAL_ANGLE_MAX`, `_RECTILINEAR_RATIO_MIN`, etc.) en vez de magic numbers para evitar PLR2004
- **Cuándo usar:** al agregar un nuevo check de preflight — nueva función `_check_*` + constante nombrada + entrada en `suggestions[]`
- **Anti-pattern:** hardcodear umbrales en la función; comparar con literales numéricos sin nombre (ruff PLR2004)

### HoughLinesP para detección de paredes — engines/opencv_classic.py
- **Archivo:** `src/vitrina_cv/engines/opencv_classic.py`
- **Qué hace:** detecta segmentos de pared vía Hough probabilístico; retorna endpoints `(x1,y1,x2,y2)` que mapean directamente a `Wall{start,end}` sin post-proceso
- **Cuándo usar:** detectar segmentos lineales de paredes; más rápido que Hough estándar en imágenes densas
- **Anti-pattern:** usar `seg[0]` sin `.reshape(4)` — rompe entre versiones de OpenCV que devuelven `(N,1,4)` vs `(N,4)`

### CCA + approxPolyDP para detección de rooms — engines/opencv_classic.py
- **Archivo:** `src/vitrina_cv/engines/opencv_classic.py`
- **Qué hace:** connected-component analysis sobre la inversión de la máscara de paredes; simplifica con `approxPolyDP(epsilon=2% perimeter)` — non-overlap garantizado por CCA, sin opencv-contrib
- **Cuándo usar:** extraer regiones interiores cerradas como polígonos simplificados
- **Anti-pattern:** epsilon fijo en píxeles (no escala); usar esqueleto (requiere opencv-contrib)

### Intermedios reutilizables en instancia de engine
- **Archivo:** `src/vitrina_cv/engines/opencv_classic.py`
- **Qué hace:** `_wall_mask` y `_gray` en la instancia tras cada `extract()` para que tareas posteriores (06-cv-04) reutilicen la binarización sin repetirla; NO forman parte del contrato `GeometryEngine`
- **Cuándo usar:** cuando una task posterior necesite un intermedio costoso del engine; documentar como privado
- **Anti-pattern:** exponer en la interfaz `GeometryEngine`; acceder desde routers

### Gap-detection collinear para aberturas candidatas — engines/opencv_classic.py
- **Archivo:** `src/vitrina_cv/engines/opencv_classic.py`
- **Qué hace:** agrupa segmentos HoughLinesP por orientación (horizontal/vertical) y coordenada perpendicular usando bins de `_OPENING_COLLINEAR_TOL_PX`; fusiona intervalos solapados con `_merge_intervals`; detecta gaps entre segmentos consecutivos; cada gap dentro de `[_OPENING_MIN_GAP_PX, _OPENING_MAX_GAP_PX]` se emite como `Opening`
- **Cuándo usar:** al añadir lógica de detección de aberturas; seguir el patrón de constantes nombradas para todos los umbrales (PLR2004)
- **Anti-pattern:** hardcodear rangos de tamaño de gap; decidir el tipo final de abertura en el engine (viola ADR-009); exponer `_wall_mask` / `_gray` en la interfaz `GeometryEngine`

### _classify_gap — heurística conservadora de tipo de abertura — engines/opencv_classic.py
- **Archivo:** `src/vitrina_cv/engines/opencv_classic.py`
- **Qué hace:** clasifica gaps por tamaño en `door` / `window` / `unknown`; cuando el tamaño cae en la zona de solapamiento (30-50 px) devuelve `unknown` — nunca adivina (ADR-009)
- **Cuándo usar:** al ajustar los rangos heurísticos; la función es el único punto de cambio para la clasificación de tipo
- **Anti-pattern:** usar literales numéricos directamente en la comparación; emitir un tipo definitivo (`door`/`window`) cuando el gap es ambiguo

### _detect_scale stub + extension point — engines/opencv_classic.py
- **Archivo:** `src/vitrina_cv/engines/opencv_classic.py`
- **Qué hace:** devuelve `Scale(source="none")` siempre en Fase 1; el docstring documenta el procedimiento completo de integración futura con OCR (detección de líneas de cota → OCR → `px_per_unit`); es el único punto de extensión para escala
- **Cuándo usar:** cuando se integre OCR — modificar solo esta función; agregar ADR antes de añadir dependencias de OCR
- **Anti-pattern:** retornar error cuando no hay cotas (viola ADR-003: `scale.source=none` nunca es error); añadir dependencias OCR sin ADR

### _build_closed_wall_mask_for_rooms — cierre morfológico para CCA — engines/opencv_classic.py
- **Archivo:** `src/vitrina_cv/engines/opencv_classic.py`
- **Qué hace:** aplica dos cierres morfológicos direccionales (kernel horizontal 160×1 + vertical 1×160) sobre la máscara de paredes para puentear aberturas arquitectónicas antes del CCA; resuelve rooms=0 cuando el plano tiene puertas/ventanas en el perímetro; usada SOLO en el paso de detección de rooms
- **Cuándo usar:** siempre que la máscara de paredes llegue a `_detect_rooms`; no pasar al resto del pipeline
- **Anti-pattern:** usar la máscara cerrada para detección de paredes u aberturas — distorsiona los segmentos Hough y los gaps

### _consolidate_walls — fusión colineal de segmentos Hough — engines/opencv_classic.py
- **Archivo:** `src/vitrina_cv/engines/opencv_classic.py`
- **Qué hace:** agrupa los segmentos raw de HoughLinesP por orientación y bin colineal, fusiona intervalos solapados (misma lógica que `_detect_openings`), emite un Wall por run continuo; reduce ~131 segmentos raw a ~16-25 por plano simple
- **Cuándo usar:** inmediatamente después de `_detect_walls`; los walls consolidados son el output de `Geometry.walls` y el input de `_detect_openings`
- **Anti-pattern:** pasar walls raw a `_detect_openings` sin consolidar — provoca buckets inconsistentes y falsos gaps en paredes perimetrales

### _nms_openings — NMS por distancia de centros — engines/opencv_classic.py
- **Archivo:** `src/vitrina_cv/engines/opencv_classic.py`
- **Qué hace:** suprime candidatas duplicadas de apertura cuyo centro esté a menos de `_NMS_CENTER_DIST_PX=20px` entre sí; ordena por confidence descendente y elimina solapantes; resuelve los 3x duplicados por trazo grueso de HoughLinesP
- **Cuándo usar:** siempre después de `_detect_openings`; aplicar como post-procesado final antes de ensamblar `Geometry`
- **Anti-pattern:** aumentar `_NMS_CENTER_DIST_PX` > 50 px — colapsa aberturas reales cercanas (puerta doble, ventana corrida)

### clean_mask — pipeline de limpieza de máscara binaria — mask_cleanup.py
- **Archivo:** `src/vitrina_cv/mask_cleanup.py`
- **Qué hace:** orquestador de cuatro pasos puros aplicado DESPUÉS de binarización y ANTES de Hough/CCA: (1) `remove_small_components` — elimina componentes con bbox pequeño en ambos lados (texto/dígitos); (2) `retain_rectilinear` — apertura morfológica H+V (mata achurado diagonal); (3) `crop_to_main_component` — pone a cero lo exterior al bbox del componente más grande + margen (mata cotas perimetrales y marcos de scan); (4) `filter_thin_strokes` — reconstrucción geodésica acotada desde semillas gruesas (mata cotas interiores, muebles, escaleras). El preflight NO se toca.
- **Cuándo usar:** llamar desde `OpenCVClassicEngine.extract()` paso 4b, solo cuando `settings is not None`. Master switch: `CV_CLEANUP_ENABLED` (default True).
- **Thresholds en Settings:** `CV_CLEANUP_TEXT_MAX_SIDE_PX=40`, `CV_CLEANUP_RECTILINEAR_LEN_PX=150`, `CV_CLEANUP_CROP_ENABLED=True`, `CV_CLEANUP_CROP_MARGIN_PX=20`, `CV_CLEANUP_THICKNESS_FILTER_ENABLED=True`, `CV_CLEANUP_MIN_WALL_THICKNESS_PX=6` (calibrado para 2000 px; auto-escalado), `CV_CLEANUP_THICKNESS_PRECLOSE_PX=9`.
- **Orden obligatorio: 1 → 2 → 3(crop) → 4(filter).** Crop antes del filtro mantiene el perímetro exterior conectado; si se invierte el orden, el filtro fragmenta el perímetro y crop selecciona solo un stub de pared.
- **Anti-pattern:** aplicar el cleanup antes del upscale; pasar la máscara limpiada al preflight (ADR-005); invertir el orden de crop y filter; hardcodear thresholds fuera de Settings.

### filter_thin_strokes — filtro geodésico por grosor de trazo — mask_cleanup.py
- **Archivo:** `src/vitrina_cv/mask_cleanup.py`
- **Qué hace:** reconstrucción geodésica acotada en tres fases. **Pre-close:** aplica `MORPH_CLOSE` con kernel `CV_CLEANUP_THICKNESS_PRECLOSE_PX × CV_CLEANUP_THICKNESS_PRECLOSE_PX` (default 9 px) sobre una COPIA de la máscara para rellenar huecos entre dobles líneas de pared. **Fase 1 — semillas:** `distanceTransform` sobre la máscara pre-cerrada; semillas = `dist >= T/2` intersectadas de vuelta con la máscara original. **Fase 2 — dilatación geodésica acotada:** `ceil(T)` iteraciones de dilatación 3x3 con AND sobre la máscara original. **Fase 3:** devuelve los píxeles alcanzados; los trazos delgados no alcanzados se descartan.
- **Cuándo usar:** paso 4 del cleanup (DESPUÉS del crop). Gateado por `CV_CLEANUP_THICKNESS_FILTER_ENABLED`. El umbral `T` se escala por `long_side / CV_UPSCALE_TARGET_PX`.
- **Resultado calibrado en fixtures (2000 px):** plano_limpio W=115→80/R=6 (sin regresión); plano_denso_anotado W=118→84/R=5 (reducción sustancial, rooms ≥ 4).
- **Por qué pre-close:** planos con dobles líneas paralelas con hueco > 3 px no quedan suficientemente engrosados por el `(3,3)` close de `_build_wall_mask`; el pre-close (9 px) cierra huecos de hasta ~8 px solo para el cálculo de semillas — la máscara real no se modifica.
- **Anti-pattern:** correr el filtro ANTES del crop (el perímetro se fragmenta en 27+ componentes y crop selecciona solo un stub → rooms=0); usar distanceTransform sobre la máscara completa sin pre-close en planos de doble línea (grosor efectivo = 3 px → sin semillas → todo se descarta).

### normalize_resolution — preprocesamiento de upscale antes del pipeline CV — preprocessing.py
- **Archivo:** `src/vitrina_cv/preprocessing.py`
- **Qué hace:** función pura `normalize_resolution(img, settings) -> (img, factor)` que upscalea imágenes pequeñas vía `cv2.INTER_CUBIC` hasta `CV_UPSCALE_TARGET_PX` (default 2000 px de lado mayor), capeado por `CV_UPSCALE_MAX_FACTOR` (default 4.0). Nunca downscalea; si `max(h,w) >= target` retorna la imagen sin cambios y factor `1.0`.
- **Cuándo usar:** integrado en `OpenCVClassicEngine.extract()` tras decode, antes del pipeline completo; también en `run_preflight()` para los checks de calidad (contraste, line_density, rectilinear) sobre imagen normalizada.
- **Razón de ser:** el motor OpenCV está calibrado en píxeles absolutos (`_DOOR_GAP_MIN_PX=30`, `_WALL_THICKNESS_EST_PX=10`, `_MIN_ROOM_AREA_PX=2000`). Imágenes de 612x612 o 470x896 producen gaps sub-umbral y rooms inexistentes sin este paso.
- **Anti-pattern:** aplicar upscale dentro de `_decode_png` (mezcla decodificación con normalización); reescalar coordenadas de vuelta a espacio original (toda la geometría vive en espacio normalizado — ADR-003).

### FastAPI lifespan con app.state para motor CV singleton — main.py
- **Archivo:** `src/vitrina_cv/main.py`
- **Qué hace:** instancia el motor CV una sola vez en el lifespan de FastAPI (`get_engine(settings.cv_engine)`) y lo almacena en `app.state.engine`; los routers lo recuperan via `request.app.state.engine` sin reinstanciar por request
- **Cuándo usar:** cualquier recurso que deba inicializarse una sola vez al startup; nunca instanciar en el handler
- **Anti-pattern:** instanciar el motor dentro del handler — viola ADR-008 y causa overhead por request

### UploadFile opcional + validación manual — api/routers/
- **Archivo:** `src/vitrina_cv/api/routers/extract_geometry.py`, `src/vitrina_cv/api/routers/preflight.py`
- **Qué hace:** declara `image: Annotated[UploadFile | None, File(...)] = None` para evitar la validación automática 422 de FastAPI cuando falta el campo; el handler verifica `if image is None` y retorna `400 invalid_request` con el enum de error correcto según el contrato (ADR-003)
- **Cuándo usar:** siempre que un campo `File`/`Form` obligatorio deba devolver `400` en lugar del `422` default de FastAPI cuando está ausente
- **Anti-pattern:** `UploadFile = File(...)` como default — dispara `B008` de ruff y deja el control de la respuesta 422 a FastAPI en lugar del contrato

### Logging estructurado por etapa con duration_ms — api/routers/
- **Archivo:** `src/vitrina_cv/api/routers/extract_geometry.py`, `src/vitrina_cv/api/routers/preflight.py`
- **Qué hace:** captura `time.monotonic()` antes y después de cada etapa (read, extract/check); pasa `endpoint`, `image_size`, `duration_read_ms`, `duration_extract_ms` y `duration_total_ms` como `extra={}` al logger estándar; en errores incluye `error_code`
- **Cuándo usar:** en cualquier handler que tenga etapas de procesamiento medibles; el `extra={}` es el patrón de logging estructurado sin dependencias pesadas
- **Anti-pattern:** concatenar valores en el mensaje del logger (pierde parsabilidad estructurada); medir tiempos con `datetime.now()` en lugar de `time.monotonic()`

### TYPE_CHECKING para imports de tipo-only
- **Archivo:** `src/vitrina_cv/engines/base.py`, `engines/opencv_classic.py`, `preflight/checks.py`
- **Qué hace:** agrupa imports usados solo en annotations bajo `if TYPE_CHECKING:` — evita circular imports y reduce overhead
- **Cuándo usar:** cualquier import que solo aparece en return-type / param-type annotations (con `from __future__ import annotations`)
- **Anti-pattern:** importar tipos de dominio pesados a nivel de módulo cuando `from __future__ import annotations` ya hace las annotations lazy (TCH001)
