# Business Rules — vitrina-cv

<!-- Invariantes de negocio que cruzan dominios. -->

last_updated: 2026-07-02

## Invariantes globales

### Stateless absoluto
- **Regla:** cada request es independiente; el servicio no retiene bytes ni estado entre requests.
- **Dónde se aplica:** todos los handlers (por implementar)
- **Por qué:** ADR-002 — vitrina descarga el PNG de S3 y envía los bytes; el servicio no tiene credenciales de infra.

### Coordenadas siempre en pixeles de la imagen recibida
- **Regla:** toda coordenada en la respuesta está expresada en pixeles de la imagen recibida en ese request; nunca se transforma al sistema de coordenadas de vitrina.
- **Dónde se aplica:** `POST /extract-geometry` response
- **Por qué:** ADR-003 — el mapeo al sistema de geometría de vitrina es responsabilidad de vitrina.

### error_code como enum estricto
- **Regla:** los errores solo pueden tener `error_code` de los valores `invalid_request`, `unprocessable_image` o `model_not_loaded`. Nunca un string libre.
- **Dónde se aplica:** todos los endpoints de error
- **Por qué:** ADR-003 — el contrato define el enum; divergir rompe el cliente Go.

### scale nunca bloquea la respuesta
- **Regla:** si no hay referencias de medida en el plano, `scale.source="none"` y `px_per_unit`/`unit` son null; la respuesta es `200` igual.
- **Dónde se aplica:** `POST /extract-geometry`
- **Por qué:** ADR-003 — la escala es opcional; su ausencia no es un error.

## Reglas por dominio

### extract-geometry
- **Aberturas como candidatas (ADR-009):** el motor CV no decide el tipo final de una abertura. Emite `type_candidate` tentativo + `confidence`; la clasificación final la hace el LLM en vitrina.
- **Escaleras como candidatas (07-cv-09):** `StairsCandidate` sigue el mismo patrón que `Opening` — bbox `[x,y,w,h]` en píxeles (exactamente 4 floats), dirección `StairsDirection` (up-N/S/E/W/unknown), `confidence` en [0,1]. El campo `stairs_candidates` en `Geometry` tiene default `[]` para no romper consumidores existentes. La lógica de detección se implementa en cv-10.
- **Motor intercambiable (ADR-008):** el router lee `CV_ENGINE` y delega a la interfaz `GeometryEngine`; nunca instancia el motor concreto directamente en el handler.
- **Close asimétrico H/V para detección de habitaciones:** `_build_closed_wall_mask_for_rooms` aplica cierre morfológico con kernels H (`CV_ROOM_CLOSE_H_GAP_PX`, default 80px) y V (`CV_ROOM_CLOSE_V_GAP_PX`, default 160px) independientes. El kernel H debe ser estrictamente menor que el ancho de la habitación más estrecha esperada (baños peruanos típicos ≥ 130px a 2000px). Nunca usar el mismo valor para H y V en planos residenciales densos — un H=160px destruye habitaciones de ~1.0m.

### preflight
- **Sin LLM (ADR-005):** el gate evalúa únicamente con heurísticas deterministas de imagen (resolución, contraste, densidad de líneas, orientación). Ninguna llamada a servicio externo.
- **Umbrales configurables (ADR-005):** los umbrales de pre-vuelo (`CV_PREFLIGHT_*`) son variables de entorno; cambiar un umbral no requiere modificar código.

## Reglas cross-dominio

### El servicio no invade la semántica del LLM
- **Regla:** el servicio no etiqueta ambientes (nombres de habitaciones), no descarta aberturas ni toma decisiones finales sobre ningún elemento semántico. Solo emite geometría y candidatas.
- **Dominios involucrados:** extract-geometry, preflight
- **Por qué:** ADR-009 y objetivo del producto — la semántica es responsabilidad exclusiva del LLM en vitrina.

## Modelo de autenticación y autorización

### Autenticación entre servicios
- **Mecanismo interno:** ninguna — red privada interna; el servicio solo es accesible desde vitrina en el mismo cluster (ADR-010).
- **Razón:** sidecar en red privada; no expuesto a internet.
- **Servicios que requieren auth interna:** ninguno

### Autenticación hacia el exterior
- **Mecanismo externo:** no aplica — el servicio no consume servicios externos.

### Reglas de autorización
- Los servicios internos (vitrina) no requieren token — están en la misma red privada (ADR-010).
- El servicio nunca acepta requests de orígenes fuera de la red interna.
