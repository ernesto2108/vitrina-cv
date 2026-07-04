# Riesgos y Deuda Técnica — vitrina-cv

last_updated: 2026-07-02

## Gotchas operativos

### OpenCV clásico puede fallar ante variabilidad de planos
- **Dónde:** `engines/opencv_classic.py` (por crear)
- **Descripción:** el motor OpenCV clásico (binarización, detección de líneas, contornos) puede ser menos robusto que un modelo ML ante planos con estilos no estándar, baja calidad o ruido.
- **Workaround (ADR-008):** la interfaz `GeometryEngine` permite sustituir el motor por RasterScan u otro sin cambiar la API; el benchmarking de RasterScan corre en paralelo.

### Latencia CPU
- **Dónde:** `POST /extract-geometry`
- **Descripción:** inferencia en CPU; el objetivo p95 < 20 s puede verse afectado por imágenes de alta resolución.
- **Workaround:** calibrar resolución de entrada y establecer métricas de histograma de latencia por etapa (segmentación / vectorización).

### Aberturas no detectadas por CV no pueden ser recuperadas por el LLM
- **Dónde:** `engines/` — detección de huecos en paredes
- **Descripción:** si el motor CV no detecta un hueco de puerta/ventana, el LLM en vitrina no puede inventarlo — la calidad de las candidatas determina el techo de precisión semántica.
- **Workaround (ADR-009):** umbral de `confidence` configurable; monitorear tasa de candidatas detectadas por imagen.

### Bins colineales múltiples por trazo grueso de pared (rooms=0 en planos con aberturas — resuelto; walls count residual)
- **Dónde:** `engines/opencv_classic.py` — `_consolidate_walls`, `_OPENING_COLLINEAR_TOL_PX=8`
- **Descripción:** un trazo de pared de 12-15px genera segmentos Hough en 2-3 bins adyacentes (e.g., x=1092/1100/1108 para la pared derecha), que persisten como walls consolidados separados. El cierre morfológico (`_build_closed_wall_mask_for_rooms`) resuelve rooms=0; el NMS resuelve los duplicados de aberturas. El count de walls puede superar 20 en imágenes con trazos gruesos (~25 en sintético 12px).
- **Workaround actual:** `_OPENING_MIN_WALL_SPAN_PX=170` suprime artefactos de gap en bins secundarios (flanco corto < 170px).
- **Fix futuro:** segunda pasada de merge entre bins colineales adyacentes para unificar walls paralelos del mismo trazo; requiere benchmark con planos reales primero.

### Contrato OpenAPI desincronizado
- **Dónde:** `docs/specs/06-extraccion-cv-hibrida/contracts/cv-service.openapi.yaml` vs implementación
- **Descripción:** el contrato es canónico (ADR-003); si diverge de la implementación, el cliente Go en vitrina se rompe.
- **Workaround:** validar el contrato con `api-contract` antes de cada release; usar generación de modelos Pydantic desde el OpenAPI.

## Deuda técnica

### Archivos candidatos a refactor

| Archivo | Líneas | Razón |
|---------|--------|-------|
| `src/vitrina_cv/mask_cleanup.py` | ~200 | Módulo nuevo; podría integrarse en `preprocessing.py` si crece el nro de módulos auxiliares |
| `src/vitrina_cv/scale_ocr.py` | ~340 | Módulo OCR (ADR-011); candidato a tests unitarios granulares de `_consistent_median`, `_infer_unit_and_metres`, `_distance_point_to_seg` — actualmente cubiertos solo por integración |

### CV_CLEANUP_RECTILINEAR_LEN_PX — calibración frágil por resolución [MITIGADO 2026-07-03]
- **Dónde:** `src/vitrina_cv/mask_cleanup.py`, `src/vitrina_cv/config/settings.py`
- **Descripción original:** el valor default L=150 fue calibrado sobre una imagen 1049x2000. A resoluciones > 2000px sin upscale, el kernel de 150 px elimina piezas de esquina de junctions tabique-muro (120×31 px en plan-004 a 300 px/m), rompiendo el cierre de habitaciones.
- **Mitigación aplicada (2026-07-03):** se añadió `CV_CLEANUP_RECTILINEAR_MAX_RES_SCALE=1.0` — `retain_rectilinear` se salta automáticamente para imágenes con `long_side/target > 1.0` (nativas alta-res no upscaleadas). Paralelamente, `CV_CLEANUP_MIN_WALL_THICKNESS_PX` bajó de 6 a 5 y el escalado del umbral de thickness se capea en el mismo factor (1.0) para preservar muros de doble-línea (5 px/strand) en planos de alta resolución.
- **Riesgo residual:** planos nativos alta-res CON achurado denso (plan-001 style, pero a >2000px) no se beneficiarán del filtrado de achurado. Si ese caso emerge, considerar detección explícita de contenido diagonal para decidir si aplicar retain_rectilinear.

### TODOs y FIXMEs con impacto

Ninguno — repo vacío. Actualizar con el primer scaffold.

## Restricciones conocidas

- **CubiCasa5k descartado permanentemente:** licencia CC BY-NC prohíbe uso comercial; no hay alternativa con licencia limpia en la evaluación actual. No reabrir sin un nuevo ADR.
- **RasterScan en evaluación sin deadline:** contacto comercial y benchmark pendiente; no bloquea Fase 1.
- **Sin GPU:** la infra actual es CPU-only; cualquier motor que requiera GPU necesita un ADR antes de considerarse.

## Dependencias frágiles

- **opencv-python:** versión fija recomendada — cambios de versión pueden alterar resultados de binarización o detección de contornos; pin de versión obligatorio.
- **vitrina (Go, repo hermano):** el cliente Go consume el contrato REST; cualquier cambio de contrato rompe la integración — requerir coordinación cross-repo via ADR.
- **pytesseract + tesseract-ocr (ADR-011):** dependencia operativa del binario del SO (`tesseract-ocr`). El Dockerfile debe instalar `tesseract-ocr` (trabajo de devops pendiente). Sin el binario, `CV_SCALE_OCR_ENABLED=true` degrada silenciosamente a `source=none`; nunca rompe el endpoint. Limitación conocida: en scans de baja resolución (lado largo < ~3000px en el original) el texto de cotas puede ser demasiado pequeño para tesseract incluso tras el upscale interno a 4000px — el consistency check rechaza lecturas espurias y retorna `source=none` honestamente.

## Áreas sin tests

- Todo — repo vacío. Prioridad al primer scaffold:
  - Tests de `GeometryEngine.extract()` con fixtures de imágenes reales y sintéticas
  - Tests unitarios de heurísticas de preflight (AC-5: umbrales configurables)
  - Tests de contrato API contra el OpenAPI canónico
