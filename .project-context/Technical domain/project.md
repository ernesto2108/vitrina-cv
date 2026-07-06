# Proyecto — vitrina-cv

last_updated: 2026-07-02
task_tool: ""  # Herramienta de gestión de tareas del proyecto (valor libre, ej: Linear, Jira, Notion, ninguna)

## Objetivo

Servicio de computer vision (Python 3.12+ / FastAPI) que extrae geometría determinista de planos arquitectónicos PNG: paredes como segmentos, polígonos de habitaciones y aberturas candidatas con type_candidate/bbox/confidence, más escala opcional. También aloja el gate de pre-vuelo con heurísticas puras de imagen. Es un sidecar interno de `vitrina` (backend Go); el LLM en vitrina hace la semántica — este servicio nunca decide tipos finales ni etiqueta ambientes.

## Restricciones no negociables

- **Stateless por diseño (ADR-002):** no accede a S3, DB ni credenciales de infra; recibe los bytes PNG en el request.
- **Sin LLM en preflight (ADR-005):** `POST /preflight` usa solo heurísticas deterministas de imagen.
- **Motor intercambiable via `CV_ENGINE` (ADR-008):** nunca hardcodear el motor CV; pasar siempre por la interfaz `GeometryEngine`.
- **Sin decisión semántica (ADR-009):** el servicio no decide el tipo final de aberturas ni etiqueta ambientes; emite candidatas con `type_candidate` + `confidence`.
- **Sin modelo ML en Fase 1 (ADR-008):** el motor de Fase 1 es OpenCV clásico; RasterScan es evaluación futura sin bloquear esta fase.
- **Sin GPU requerida:** inferencia en CPU; objetivo de latencia p95 < 20 s por imagen.
- **Contrato canónico en `cv-service.openapi.yaml` (ADR-003):** no divergir de él; las coordenadas se devuelven siempre en pixeles de la imagen recibida.
- **CubiCasa5k descartado:** licencia CC BY-NC prohibida para uso comercial; no integrar bajo ninguna circunstancia.

## Stack

| Componente | Tecnología | Versión |
|-----------|-----------|---------|
| Lenguaje | Python | 3.12+ |
| Framework web | FastAPI | 0.115+ |
| Validación de modelos | Pydantic | v2 (recomendado con FastAPI moderno) |
| Motor CV Fase 1 | OpenCV clásico (opencv-python) | 5.0+ |
| Testing | pytest | 8.0+ |
| Package manager | uv | 0.x (lockfile: uv.lock) |
| Infra | Docker / sidecar interno | — |

## Estilo arquitectónico

- **Estilo principal:** Layered con estrategia intercambiable (Strategy pattern para motores CV)
- **Capas previstas:** `api/` (routers FastAPI) → `engines/` (interfaz + implementaciones de motor CV) → `preflight/` (heurísticas) → `config/` (settings por env)
- **Convención de paths:** `src/vitrina_cv/` (src layout, PEP 517/518, hatchling build backend)

## SOLID detectado

| Principio | Estado | Observación |
|-----------|--------|-------------|
| SRP | Previsto | Cada módulo tiene responsabilidad única según estructura planeada |
| OCP | Previsto OK | Interfaz `GeometryEngine` + `CV_ENGINE` env var — extender sin modificar (ADR-008) |
| LSP | No evaluado | Repo vacío |
| ISP | No evaluado | Repo vacío |
| DIP | Previsto OK | `api/` depende de la interfaz `GeometryEngine`, no del motor concreto |

## Preprocesamiento de resolución (upscale)

- **Módulo:** `src/vitrina_cv/preprocessing.py`
- **Función:** `normalize_resolution(img, settings) -> (img, factor)` — pura, sin I/O.
- **Regla:** si `max(h,w) >= CV_UPSCALE_TARGET_PX` → factor 1.0, sin cambios. Si no → `factor = min(target/long_side, CV_UPSCALE_MAX_FACTOR)`, resize con `INTER_CUBIC`.
- **Integración:** llamado en `extract()` tras decode (todo el pipeline en espacio normalizado) y en `run_preflight()` para checks de calidad (resolution_ok sobre original; resto sobre normalizada).
- **`image_size` en respuesta:** refleja dimensiones normalizadas. Las coordenadas NO se reescalan al espacio original (sistema de coordenadas único — ADR-003).
- **Settings:** `cv_upscale_target_px=2000`, `cv_upscale_max_factor=4.0`, `cv_preflight_min_resolution` default cambiado a `300x300`.

## Limpieza de máscara binaria (mask cleanup)

- **Módulo:** `src/vitrina_cv/mask_cleanup.py`
- **Posición en pipeline:** paso 4b — después de `_build_wall_mask` y ANTES de HoughLinesP / CCA. El preflight NO lo ve.
- **Pasos:** (1) quitar componentes pequeños en ambas dimensiones (texto), (2) apertura morfológica H+V con kernel L=150 px — **condicional: solo se aplica cuando `resolution_scale ≤ CV_CLEANUP_RECTILINEAR_MAX_RES_SCALE` (default 1.0)**; omitida para imágenes nativas alta-res que no sufrieron upscale, (3) recorte al componente mayor + margen (mata cotas perimetrales), (4) filtro de grosor por componente vía distanceTransform (mata anotaciones/muebles/escaleras aislados). Orden obligatorio: 1 → 2 → 3 → 4.
- **Settings (CV_CLEANUP_*):** `CV_CLEANUP_ENABLED`, `CV_CLEANUP_TEXT_MAX_SIDE_PX=40`, `CV_CLEANUP_RECTILINEAR_LEN_PX=150`, `CV_CLEANUP_RECTILINEAR_MAX_RES_SCALE=1.0` (nuevo — cap de res_scale para aplicar paso 2), `CV_CLEANUP_THICKNESS_FILTER_ENABLED=True`, `CV_CLEANUP_MIN_WALL_THICKNESS_PX=5` (default bajó de 6 a 5 para preservar muros doble-línea de 5 px), `CV_CLEANUP_CROP_ENABLED`, `CV_CLEANUP_CROP_MARGIN_PX=20`.
- **Calibración res_scale cap=1.0:** para planos nativos alta-res (long_side > 2000px, ej. 4460px a 300 px/m), `retain_rectilinear` elimina piezas de esquina en junctions tabique-muro-exterior que son críticas para el cierre de habitaciones — cap en 1.0 las preserva.
- **Calibración thickness=5:** el paso 4 ahora **no escala hacia arriba** el umbral de grosor para imágenes native-res (res_scale capped at `CV_CLEANUP_RECTILINEAR_MAX_RES_SCALE`). Con umbral=5 px, muros de doble-línea de 5 px/strand sobreviven (max_dist=2.5px ≥ threshold/2=2.5px — borderline pero funcional).
- **Efecto medido (calibración 2026-07-03, harness ADR-012):** plan-004 alta-res (4460×3260 px, 300 px/m): 0→4/4 rooms, score 0.000→0.981. Sin regresión: plan-002 1.000, plan-003 0.625.

## Detección de escaleras (07-cv-10)

- **Módulo:** `src/vitrina_cv/engines/opencv_classic.py` — función `_detect_stairs_candidates`
- **Posición en pipeline:** paso 9, DESPUÉS de rooms (necesita polígonos para anti-FP) y ANTES de `_detect_scale`.
- **Máscara usada:** `_pre_filter_mask` (pasos 1-3 del cleanup, sin `filter_thin_strokes`) — los peldaños delgados se preservan aquí.
- **Algoritmo:** HoughLinesP con threshold=20, minLineLength=20 px; agrupa por orientación H/V; binning de coordenada perpendicular en slots de 5 px; merging de slots adyacentes; runs de ≥4 treads con spacing 20-40 px y varianza relativa ≤ 20 %; bbox del run verificado con `cv2.pointPolygonTest` contra room polygons.
- **Setting:** `CV_STAIRS_DETECTION_ENABLED=true` (default). Con false, `stairs_candidates=[]` y pipeline sin cambios.
- **Log:** `stairs_candidates_count` emitido en cada `extract()`.

## Convenciones establecidas

- Python 3.12+, type hints en todo el código
- Pydantic para validación y serialización de modelos
- pytest para tests; estructura de test por decidir en el scaffold
- Conventional Commits en inglés
- Documentación técnica en español

## Qué NO introducir

- Acceso a S3, DB, Redis ni cualquier almacenamiento externo (viola ADR-002)
- Llamadas a LLM externo dentro de cualquier handler (viola ADR-005)
- Motor CV hardcodeado fuera de la interfaz `GeometryEngine` (viola ADR-008)
- Pesos de modelos preentrenados con licencia no comercial (p. ej. CubiCasa5k)
- Decisiones semánticas sobre tipo de abertura o etiqueta de ambiente (viola ADR-009)
- Retención de estado entre requests

## Harness de evaluación offline (ADR-012)

- **Directorio:** `eval/` — fuera de `src/vitrina_cv/`; no se empaqueta en el build (hatchling src layout).
- **Runner:** `eval/run_eval.py` — invocado con `uv run python eval/run_eval.py`.
- **Dataset:** `eval/dataset/<plan_id>/` — cada directorio contiene `image.png` + `ground_truth.json` (formato canónico en ADR-012 §2).
- **Dataset inicial:** 3 planos baseline — `plan-001-denso-achurado` (12 rooms esperados, áreas m² anotadas), `plan-002-simple-limpio` (6 rooms, sin escala), `plan-003-reticula-cotas` (16 rooms, sin áreas por ambiente).
- **Baseline corrida 2026-07-03 (motor opencv):** plan-001 score=0.000 (0/12 rooms), plan-002 score=1.000 (6/6 rooms), plan-003 score=0.625 (10/16 rooms). Score medio=0.542.
- **Sin dependencias nuevas:** usa stdlib + el paquete instalado. `eval/` no modifica el contrato REST (ADR-003 intacto).

## Estrategia de migraciones

- **Herramienta:** ninguna — el servicio es stateless sin base de datos
- **Directorio:** no aplica
- **Notas:** si en el futuro se añade persistencia, definir en un ADR antes de implementar
