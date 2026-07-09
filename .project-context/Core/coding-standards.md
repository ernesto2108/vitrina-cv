# Coding Standards — vitrina-cv

<!-- Naming, estructura de carpetas, idioma del código, reglas de linting y patrones prohibidos.
     Complementado con patrones de diseño detectados automáticamente en el código. -->

last_updated: 2026-07-08

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

### Crop a envolvente multi-componente (ADR-015) — mask_cleanup.py
- **Archivo:** `src/vitrina_cv/mask_cleanup.py::crop_to_main_component` (paso 3 de `clean_mask`)
- **Qué hace:** en vez de recortar solo al bbox del componente conexo mayor, calcula `A_max` y considera "significativo" todo componente con `area >= min_area_ratio * A_max`; recorta a la **unión de bboxes** (envolvente) de todos los significativos + margen, clampeado a límites de imagen. Preserva footprints no contiguos (alas/cocheras separadas) sin romper los supuestos de Hough/CCA downstream (la máscara resultante sigue siendo una sola caja rectangular).
- **Invariante de compatibilidad:** parámetro `min_area_ratio` default `1.0` en la función (solo el mayor califica) — el caller de producción siempre pasa `settings.cv_cleanup_crop_min_area_ratio` (default `0.05`, CV09-01). Con un solo componente significativo, la envolvente == bbox del mayor (no-op, sin regresión).
- **Cuándo usar:** al tocar el paso 3 del pipeline de limpieza de máscara o al calibrar el ratio de área mínimo.
- **Anti-pattern:** implementar la variante "por-componente" (fusionar bboxes individuales sin envolvente única) — es un fallback condicional de ADR-015, no forma parte del contrato actual salvo que el gate de regresión (CV09-11) lo exija.

### Intermedios reutilizables en instancia de engine
- **Archivo:** `src/vitrina_cv/engines/opencv_classic.py`
- **Qué hace:** `_wall_mask`, `_gray`, `_junctions` y `_pre_filter_mask` en la instancia tras cada `extract()` para que tareas posteriores reutilicen intermedios costosos sin repetirlos; NO forman parte del contrato `GeometryEngine`. `_pre_filter_mask` (07-cv-06) = máscara post steps 1-3 de cleanup, pre step 4 — conserva líneas delgadas de ventanas vectoriales.
- **Cuándo usar:** cuando una task posterior necesite un intermedio costoso del engine; documentar como privado
- **Anti-pattern:** exponer en la interfaz `GeometryEngine`; acceder desde routers

### Gap-detection collinear para aberturas candidatas — engines/opencv_classic.py
- **Archivo:** `src/vitrina_cv/engines/opencv_classic.py`
- **Qué hace:** agrupa segmentos HoughLinesP por orientación (horizontal/vertical) y coordenada perpendicular usando bins de `_OPENING_COLLINEAR_TOL_PX`; fusiona intervalos solapados con `_merge_intervals`; detecta gaps entre segmentos consecutivos; cada gap dentro de `[_OPENING_MIN_GAP_PX, _OPENING_MAX_GAP_PX]` se emite como `Opening`
- **Cuándo usar:** al añadir lógica de detección de aberturas; seguir el patrón de constantes nombradas para todos los umbrales (PLR2004)
- **Anti-pattern:** hardcodear rangos de tamaño de gap; decidir el tipo final de abertura en el engine (viola ADR-009); exponer `_wall_mask` / `_gray` en la interfaz `GeometryEngine`

### Emisión generosa de aberturas con relajación de span junto a junction (07-cv-07) — engines/opencv_classic.py
- **Archivo:** `src/vitrina_cv/engines/opencv_classic.py`
- **Qué hace:** `_detect_openings` acepta `junctions`, `arc_centers` y `settings`; para gaps cuyo endpoint está adyacente a una junction de cv-04 usa el umbral relajado `settings.cv_opening_min_wall_span_px` (default 60 px) en lugar del amplio 170 px; la lógica de candidato se delega a `_build_opening_candidate`. Cuando el gap pasa solo por el umbral relajado (flanco < 170 px), la confidence se capea a `_OPENING_RELAXED_SPAN_CONFIDENCE=0.35`. Si hay un arco de puerta (HoughCircles via `_detect_door_arcs`) adyacente al gap, la confidence sube a `_DOOR_ARC_CONFIDENCE=0.7`. El filtrado agresivo queda PROHIBIDO en el engine — se delega al backend (F4, ADR-009).
- **Settings:** `CV_OPENING_MIN_WALL_SPAN_PX=60` (antes hardcodeado en 170)
- **Log emitido:** `openings_emitted{door, window, unknown}` con conteos por corrida
- **Cuándo usar:** al calibrar los umbrales de emisión de aberturas; la función `_build_opening_candidate` es el único punto de cambio para la lógica de span + confidence
- **Anti-pattern:** filtrar gaps con flanco corto SIN verificar si hay junction adyacente; emitir el tipo definitivo con alta confidence sin evidencia de arco; no loguear conteos por tipo

### _classify_gap — heurística conservadora de tipo de abertura — engines/opencv_classic.py
- **Archivo:** `src/vitrina_cv/engines/opencv_classic.py`
- **Qué hace:** clasifica gaps por tamaño en `door` / `window` / `unknown`; cuando el tamaño cae en la zona de solapamiento (30-50 px) devuelve `unknown` — nunca adivina (ADR-009)
- **Cuándo usar:** al ajustar los rangos heurísticos; la función es el único punto de cambio para la clasificación de tipo
- **Anti-pattern:** usar literales numéricos directamente en la comparación; emitir un tipo definitivo (`door`/`window`) cuando el gap es ambiguo

### _detect_scale con OCR (ADR-011) — engines/opencv_classic.py + scale_ocr.py
- **Archivo:** `src/vitrina_cv/engines/opencv_classic.py:556`, `src/vitrina_cv/scale_ocr.py`
- **Qué hace:** `_detect_scale(gray, settings)` ya no es un stub; delega a `detect_scale_from_ocr()` cuando `CV_SCALE_OCR_ENABLED=true`. El módulo `scale_ocr.py` ejecuta: (1) upscale a 4000px para OCR, (2) pytesseract PSM11 con whitelist de dígitos, (3) HoughLinesP en imagen gris para detectar líneas de cota, (4) asociación token-línea por distancia mínima <= 120px, (5) validación de consistencia >= 2 lecturas dentro del 10% de la mediana. Devuelve `Scale(source="cotas", px_per_unit, unit="m")` o `Scale(source="none")`.
- **Cuándo usar:** la lógica OCR vive exclusivamente en `scale_ocr.py`; el engine es un wrapper delgado
- **Anti-pattern:** retornar error cuando no hay cotas (viola ADR-003); forzar un resultado cuando el OCR no alcanza consenso; aceptar una escala con una sola lectura sin corroboración

### Degradación elegante ante dependencia opcional (pytesseract) — scale_ocr.py
- **Archivo:** `src/vitrina_cv/scale_ocr.py:118`
- **Qué hace:** patrón de lazy import con flag de módulo (`_PYTESSERACT_IMPORT_FAILED`). El import de pytesseract ocurre solo en tiempo de ejecución; si falla (binario no instalado o import error), se loguea un warning único y se retorna `Scale(source=none)`. El endpoint nunca levanta excepción por ausencia del binario.
- **Cuándo usar:** toda dependencia **opcional** del servicio que no debe bloquear el endpoint si no está disponible en runtime
- **Anti-pattern:** import a nivel de módulo de una dependencia opcional (rompe el startup si no está instalada); propagar excepciones de la dependencia opcional al caller

### _build_closed_wall_mask_for_rooms — cierre morfológico para CCA — engines/opencv_classic.py
- **Archivo:** `src/vitrina_cv/engines/opencv_classic.py`
- **Qué hace:** aplica dos cierres morfológicos direccionales (kernel horizontal h_gap×1 + vertical 1×v_gap) sobre la máscara de paredes para puentear aberturas arquitectónicas antes del CCA; resuelve rooms=0 cuando el plano tiene puertas/ventanas en el perímetro; usada SOLO en el paso de detección de rooms. Desde 2026-07-04, los gaps se pueden escalar automáticamente por el upscale_factor via `cv_room_close_scale_with_upscale` (default False — opt-in).
- **Cuándo usar:** siempre que la máscara de paredes llegue a `_detect_rooms`; no pasar al resto del pipeline
- **Anti-pattern:** usar la máscara cerrada para detección de paredes u aberturas — distorsiona los segmentos Hough y los gaps; activar `scale_with_upscale` sin verificar que las habitaciones más pequeñas del plano no colapsen con el gap escalado

### _consolidate_walls — fusión colineal de segmentos Hough con centerline (07-cv-03) — engines/opencv_classic.py
- **Archivo:** `src/vitrina_cv/engines/opencv_classic.py`
- **Qué hace:** dos modos según `cv_wall_centerline_enabled` (default `True`).
  - **Centerline (on):** calcula `distanceTransform` sobre la wall_mask limpiada, estima `thickness = 2 x median(DT samples)` por segmento, agrupa trazos paralelos cuya separación perpendicular ≤ grosor estimado, posiciona el Wall resultante en el centroide y emite `Wall.thickness` en píxeles. Firma actualizada: `_consolidate_walls(walls, wall_mask, settings)`.
  - **Legacy (off):** bin fijo de `_OPENING_COLLINEAR_TOL_PX=8px`; `Wall.thickness=None`. Comportamiento idéntico al pre-07-cv-03.
  - Helpers: `_sample_dt_along_segment`, `_estimate_global_wall_thickness_px`, `_group_indices_by_proximity`, `_thickness_from_dt_samples`, `_merge_segs_into_walls`, `_legacy_bin_consolidate`, `_centerline_dt_consolidate`.
- **Cuándo usar:** inmediatamente después de `_detect_walls`; los walls consolidados son el output de `Geometry.walls` y el input de `_detect_openings`.
- **Logs emitidos:** `walls_before_consolidation` y `walls_after_consolidation` con `walls_count`.
- **Anti-pattern:** pasar walls raw a `_detect_openings` sin consolidar; emitir `Wall.thickness` en metros (el consumidor Go lo convierte con `thickness_px * metersPerPx`; emitir metros causa doble conversión).

### _snap_walls_orthogonal + _extend_to_intersection + _fuse_junctions — F4 (07-cv-04 / 08-cv-xx) — engines/opencv_classic.py
- **Archivo:** `src/vitrina_cv/engines/opencv_classic.py`
- **Cuándo en el pipeline:** paso 6b de `extract()`, inmediatamente después de `_consolidate_walls` y antes de la detección de rooms. Orden: snap → extend → fuse.
- **_snap_walls_orthogonal:** para cada `Wall`, calcula el ángulo del segmento. Si `|angle| < _SNAP_ANGLE_TOL_DEG=5°` → fuerza horizontal (`start.y = end.y = mean(y1,y2)`). Si `|90° - angle| < 5°` → fuerza vertical (`start.x = end.x = mean(x1,x2)`). Si > 5° del eje más cercano → wall intacto (diagonal legítimo).
- **_extend_to_intersection (08-cv-xx / ADR-013):** para cada par (H, V), calcula la intersección geométrica `(ix, iy)`. Mueve el endpoint del muro H hacia `ix` (y el del muro V hacia `iy`) solo si: (a) el gap ≤ `cv_junction_extend_px` Y (b) la intersección cae en la prolongación del segmento (fuera de su extensión actual). Función pura, misma cardinalidad. Helper auxiliar: `_extend_wall_endpoint_to_value(coords, axis, target, extend_px)` para mantener el branch count bajo PLR0912. Idempotente: gaps ≈ 0 → sin cambio.
- **_fuse_junctions:** para cada par de endpoints de muros distintos, si la distancia euclídea < `min(t_i, t_j)` (fallback `_WALL_THICKNESS_EST_PX=10` cuando `thickness=None`), fusiona ambos al centroide del cluster vía Union-Find. Retorna `(walls_actualizados, junctions: list[Point])`.
- **_junctions en el engine:** almacenados como `OpenCVClassicEngine._junctions: list[Point] | None` para consumo por cv-07 (detección de puertas en esquina). Patrón de intermedio reutilizable ya establecido con `_wall_mask` y `_gray`.
- **Constantes:** `_SNAP_ANGLE_TOL_DEG=5.0`, `_JUNCTION_MIN_CLUSTER_SIZE=2`. Setting: `cv_junction_extend_px=40`.
- **Anti-pattern:** llamar antes de `_consolidate_walls`; exponer `_junctions` en la interfaz `GeometryEngine`; invertir el orden snap/extend/fuse; hardcodear el umbral de extensión (debe venir de settings).

### _detect_window_pattern — detección de ventanas por doble línea (07-cv-06) — engines/opencv_classic.py
- **Archivo:** `src/vitrina_cv/engines/opencv_classic.py`
- **Qué hace:** detecta ventanas vectoriales como patrón de 2 líneas paralelas DENTRO del span del muro, sobre la máscara pre-filtro (`_pre_filter_mask`). Para cada muro H/V: muestrea perfiles perpendiculares cada `_WIN_PROFILE_STEP_PX=8px`; cuenta runs de foreground; spans con `_WIN_EXPECTED_RUNS=2` runs en `>= _WIN_MIN_CONSECUTIVE_PROFILES=3` posiciones consecutivas emiten `Opening(type_candidate=window, confidence=0.35)`.
- **Cuándo usar:** llamar en `extract()` después de `_detect_openings`, antes del NMS; combinar ambas listas: `_nms_openings(gap_openings + window_openings)`.
- **Constantes:** `_WIN_PROFILE_STEP_PX=8`, `_WIN_PROFILE_HALF_EXTRA_PX=3`, `_WIN_MIN_CONSECUTIVE_PROFILES=3`, `_WIN_MAX_SPAN_PX=300`, `_WIN_EXPECTED_RUNS=2`, `_WIN_CONFIDENCE=0.35`.
- **Helpers internos:** `_build_window_flags` (muestreo), `_scan_window_runs` (scan de runs), `_maybe_emit_window` (validación span + construcción Opening), `_count_foreground_runs` (cuenta runs 1D).
- **Anti-pattern:** llamar con la máscara limpiada completa (`_wall_mask`) — el step 4 destruye las líneas delgadas de ventana; pasar muros diagonales (la función los saltea por diseño).

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
- **Logging simétrico (CV09-05 / ADR-016):** pasos 1-3 emiten INFO estructurado vía `extra={}`, compartido con `clean_mask_steps_1_to_3` mediante helpers `_log_cleanup_step1/2/3` — `cv_cleanup_step1_small_components` (`removed_count`); `cv_cleanup_step2_rectilinear` con `branch` canónico ∈ {`fixed`, `adaptive`, `skip`} + `long_side`, `min_hw`, `upscale_target_px`, `rectilinear_len_px_used`, `min_len_px`; `cv_cleanup_step3_crop` con `significant_components_count` (vía nuevo helper `_count_significant_components`, recalcula sobre la máscara PRE-crop) y `crop_bbox_xywh`. Paso 4 mantiene su log propio (`cv_cleanup_step4_thickness_filter`), no compartido.
- **Anti-pattern (logging):** usar strings libres para `branch` en vez del conjunto cerrado {fixed|adaptive|skip}; calcular `significant_components_count` sobre la máscara ya recortada (post-crop) — debe ser sobre la máscara previa al crop para reflejar los componentes que `crop_to_main_component` realmente evaluó.

### clean_mask_steps_1_to_3 — máscara pre-filtro para detección de ventanas — mask_cleanup.py (07-cv-06)
- **Archivo:** `src/vitrina_cv/mask_cleanup.py`
- **Qué hace:** variante de `clean_mask` que ejecuta solo los pasos 1-3 (text removal, rectilinear, crop) sin el `filter_thin_strokes` de paso 4. Preserva líneas delgadas de doble trazo (ventanas vectoriales) que el paso 4 destruiría. Devuelve un `NDArray` (misma signatura que `clean_mask`, sin romper callers existentes).
- **Cuándo usar:** en `OpenCVClassicEngine.extract()` paso 4b para producir `self._pre_filter_mask`; pasar esa máscara a `_detect_window_pattern`. NO usar para el pipeline principal de walls/rooms (usarlos requiere la máscara completa con step 4).
- **Anti-pattern:** sustituir `clean_mask` por esta función en el pipeline principal — los trazos delgados de cotas/muebles contaminarían Hough y CCA.
- **Logging simétrico (CV09-05 / ADR-016):** desde este run emite los mismos 3 eventos INFO de pasos 1-3 que `clean_mask` (ver entrada arriba) — antes de CV09-05 esta función no emitía ningún log, dejando ciega la ruta de detección de ventanas/escaleras (`_detect_window_pattern`, `_detect_stairs_candidates`). Cero cambio de comportamiento de máscara, verificado con test de identidad antes/después.

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

### Harness de evaluación offline — eval/run_eval.py (ADR-012)
- **Archivo:** `eval/run_eval.py`
- **Qué hace:** runner standalone invocado con `uv run python eval/run_eval.py`; descubre `eval/dataset/<plan_id>/` con `image.png` + `ground_truth.json`, invoca el motor in-process vía `get_engine(settings.cv_engine)` + `engine.extract(image_bytes)`, computa métricas por plano y reporte tabular a stdout. Exit code siempre 0 — herramienta de medición, no gate CI.
- **Score formula:** `0.5·room_score + 0.5·area_score - fp_penalty` (con área) / `room_score - fp_penalty` (sin área), clamped [0,1]. `fp_penalty = clamp(FP/expected, 0,1)·0.3`. Área solo cuando `scale.source != "none"` Y `room_areas_m2` presente.
- **Cuándo usar:** al añadir planos al dataset de evaluación o al calibrar el pipeline — correr antes y después de cada cambio de parámetro CV para medir regresiones.
- **Anti-pattern:** usar HTTP para invocar el motor desde el harness (ADR-008 + ADR-002 lo prohíben); instanciar el motor concreto directamente; incluir `eval/` en el paquete distribuible (hatchling src layout lo excluye por diseño).

### per-file-ignores para eval/ — pyproject.toml
- **Archivo:** `pyproject.toml`
- **Qué hace:** `"eval/**" = ["T201"]` — excluye la regla T201 (print) para scripts de tooling en `eval/`; el resto de reglas aplica normalmente.
- **Cuándo usar:** al agregar nuevos scripts de tooling en `eval/` que requieran `print` para reporting a stdout.

### _detect_stairs_candidates — detección de escaleras por patrón de líneas paralelas (07-cv-10) — engines/opencv_classic.py
- **Archivo:** `src/vitrina_cv/engines/opencv_classic.py`
- **Qué hace:** detecta escaleras en la máscara pre-filtro (`_pre_filter_mask`, antes de `filter_thin_strokes`) usando HoughLinesP con umbrales bajos para líneas delgadas de peldaño. Agrupa por orientación (H/V) y coordenada perpendicular, valida ≥4 líneas equiespaciadas (espaciado 20-40px, varianza relativa ≤ 20 %) y verifica que el bbox del patrón esté contenido en un room polygon via `cv2.pointPolygonTest`.
- **Helpers internos:** `_bbox_inside_any_room`, `_find_equispaced_runs`, `_merge_tread_slots`, `_stairs_runs_to_candidates`, `_candidates_from_orientation`.
- **Flag:** `CV_STAIRS_DETECTION_ENABLED` (default True) en `settings.py` — con flag off devuelve `[]` sin alterar el pipeline.
- **Log emitido:** `stairs_candidates_count` con el conteo por corrida.
- **Cuándo usar:** al ajustar umbrales de detección de escaleras; la función es el único punto de cambio. Llamar DESPUÉS de la detección de rooms (necesita los polígonos), ANTES de escala.
- **Anti-pattern:** usar `_wall_mask` (post `filter_thin_strokes`) para stair detection — los peldaños habrán sido eliminados; llamar sin rooms disponibles (el anti-FP fallaría sin polígonos).

### Nuevas env vars para fixes ADR-014/015/017 (CV09-01) — config/settings.py
- **Archivo:** `src/vitrina_cv/config/settings.py`
- **Qué hace:** agrega 3 campos `Field` siguiendo el patrón existente (default + validation + `description=` referenciando el ADR): `cv_cleanup_rectilinear_min_len_px` (int, `gt=0`, default 50 — reemplaza el literal 50 hardcodeado en el kernel adaptativo del paso 2 de cleanup, ADR-014), `cv_cleanup_crop_min_area_ratio` (float, `ge=0.0, le=1.0`, default 0.05 — ratio de área mínimo para componente significativo en crop multi-componente, ADR-015), `cv_wall_min_diagonal_len_px` (int, `gt=0`, default 40 — longitud euclidiana mínima para conservar un muro oblicuo residual, ADR-017 Mec.2).
- **Cuándo usar:** las tasks de implementación de ADR-014/015/017 deben leer estos settings en vez de definir sus propios defaults.
- **Anti-pattern:** hardcodear estos umbrales fuera de `Settings`.

### _filter_diagonal_residual_pass2 — segundo pase del filtro diagonal (CV09-04 / ADR-017) — engines/opencv_classic.py
- **Archivo:** `src/vitrina_cv/engines/opencv_classic.py`
- **Cuándo en el pipeline:** paso 6c, inmediatamente después de `_fuse_junctions()` (después de snap → extend → fuse). Orden efectivo: `_consolidate_walls` (filtro banda pase 1) → `_snap_walls_orthogonal` → `_extend_to_intersection` → `_fuse_junctions` → `_filter_diagonal_residual_pass2` (pase 2).
- **Qué hace:** doble mecanismo complementario al filtro de banda del pase 1, para eliminar el residual diagonal (stub de escalera/puerta) que sobrevive a snap/extend/fuse. **Mec.1 (ángulo):** re-evalúa el ángulo `atan2(|dy|,|dx|)` de cada `Wall` y descarta los que caen en la misma banda `[cv_wall_diagonal_filter_low_deg, cv_wall_diagonal_filter_high_deg]` del pase 1 — idempotente sobre muros ya snapeados a H/V exacto (ángulo 0°/90° nunca cae en banda). **Mec.2 (longitud):** descarta un `Wall` que (a) no es H/V exacto (`start.x != end.x` y `start.y != end.y`) **y** (b) tiene longitud euclidiana < `cv_wall_min_diagonal_len_px`; nunca se aplica a un muro H/V exacto sin importar su longitud.
- **Gating:** ambos mecanismos comparten el master switch `cv_wall_diagonal_filter_enabled` (el mismo del pase 1). Con `False`, la función retorna la lista de entrada sin modificar (misma identidad de objeto) — compatibilidad byte-idéntica pre-run-08.
- **Log emitido:** `cv_wall_diagonal_pass2_filtered` con `count_by_angle` y `count_by_length` como campos separados en `extra={}` (consumido por CV09-05 logging simétrico).
- **Cuándo usar:** al ajustar la banda o el umbral de longitud del filtro diagonal; es el único punto de cambio para el pase 2. No confundir con el filtro de pase 1 dentro de `_consolidate_walls` (banda pre-snap).
- **Anti-pattern:** llamar antes de `_fuse_junctions`; introducir una banda o flag nuevos en vez de reusar `cv_wall_diagonal_filter_low_deg/high_deg` y `cv_wall_diagonal_filter_enabled`; aplicar Mec.2 a un muro H/V exacto.

### _sanitize_room_polygon — saneo de contorno de room (10-cv-01, ADR-001) — engines/opencv_classic.py
- **Archivo:** `src/vitrina_cv/engines/opencv_classic.py`
- **Cuándo en el pipeline:** dentro de `_detect_rooms`, inmediatamente después de `approxPolyDP` (paso 5), antes de emitir el `Room`.
- **Qué hace:** elimina vértices espurios de un polígono de room cerrado cuando **ambas** aristas adyacentes caen en la misma banda diagonal `[cv_wall_diagonal_filter_low_deg, cv_wall_diagonal_filter_high_deg]` que usa el filtro de muros, y al menos una de las dos supera `cv_room_contour_diag_min_len_px`. Corre iterativamente (colapsar un vértice puede exponer otro espurio). Si tras el saneo queda una arista en banda por encima del umbral (no hay polígono ortogonal recuperable), la función retorna `None` y el room se descarta (AC-2).
- **Helper de ángulo:** `_edge_angle_deg(p1, p2)` — misma convención `atan2(|dy|,|dx|)` que `_filter_diagonal_residual_pass2`, reusa la misma banda de settings en vez de introducir una nueva.
- **Gating:** flag maestro `cv_room_contour_sanitize_enabled` (default `True`) en `settings.py`. Con `False` (o `settings=None`), `_detect_rooms` es idéntico al comportamiento previo (no-op, byte-for-byte).
- **Log emitido:** `cv_room_contour_sanitized` con `room_contour_edges_sanitized` (vértices removidos, acumulado por corrida) y `rooms_dropped_diagonal_contour` (rooms descartados por contorno no recuperable) en `extra={}`.
- **Verificado manualmente:** `plan-005-amueblado-limpio` — el vértice espurio `[1234,1559]` del room "Sala" desaparece con el flag ON (11 vértices vs 12 con flag OFF); `plan-002` (6 rooms) y `plan-003` (9 rooms) sin regresión de conteo con el flag ON.
- **Cuándo usar:** al ajustar la banda diagonal o el umbral de longitud para saneo de polígonos de room; único punto de cambio para este fix.
- **Anti-pattern:** introducir una banda de ángulo nueva en vez de reusar `cv_wall_diagonal_filter_low_deg/high_deg`; llamar antes de `approxPolyDP`; ignorar el retorno `None` (indica room no recuperable, debe descartarse, no emitirse con arista diagonal).

### TYPE_CHECKING para imports de tipo-only
- **Archivo:** `src/vitrina_cv/engines/base.py`, `engines/opencv_classic.py`, `preflight/checks.py`
- **Qué hace:** agrupa imports usados solo en annotations bajo `if TYPE_CHECKING:` — evita circular imports y reduce overhead
- **Cuándo usar:** cualquier import que solo aparece en return-type / param-type annotations (con `from __future__ import annotations`)
- **Anti-pattern:** importar tipos de dominio pesados a nivel de módulo cuando `from __future__ import annotations` ya hace las annotations lazy (TCH001)
