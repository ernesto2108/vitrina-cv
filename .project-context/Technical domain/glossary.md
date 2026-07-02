# Glosario de Dominio — vitrina-cv

<!-- Mapa entre el lenguaje del negocio (cómo habla el equipo humano)
     y el lenguaje técnico (cómo vive en el código y la base de datos).
     Las filas marcadas con ⚠️ fueron pre-populadas automáticamente y requieren validación del equipo. -->

last_updated: 2026-07-02

## Entidades

| Término humano | Término técnico | Tabla / Struct / Tipo | Descripción |
|---|---|---|---|
| ⚠️ pendiente validación | Geometry | `Geometry` (Pydantic) | Respuesta completa de extracción: paredes, habitaciones, aberturas, escala, tamaño de imagen |
| ⚠️ pendiente validación | Wall | `Wall` (Pydantic) | Segmento de pared con punto inicio/fin y grosor opcional, en pixeles |
| ⚠️ pendiente validación | Room | `Room` (Pydantic) | Polígono de habitación con lista de vértices y área en pixeles cuadrados |
| ⚠️ pendiente validación | Opening | `Opening` (Pydantic) | Abertura candidata (puerta/ventana/unknown) con bbox [x,y,w,h] y confidence |
| ⚠️ pendiente validación | Scale | `Scale` (Pydantic) | Escala del plano: px_per_unit, unit, source (cotas o none) |
| ⚠️ pendiente validación | PreflightReport | `PreflightReport` (Pydantic) | Reporte de aptitud de imagen del gate pre-vuelo |
| ⚠️ pendiente validación | GeometryEngine | `GeometryEngine` (ABC Python) | Interfaz abstracta de motor CV seleccionada por CV_ENGINE |

## Acciones / Verbos

| Término humano | Término técnico | Método / Endpoint | Descripción |
|---|---|---|---|
| ⚠️ pendiente validación | extract | `GeometryEngine.extract()` / `POST /extract-geometry` | Extraer geometría completa de un plano PNG |
| ⚠️ pendiente validación | preflight | heurísticas de imagen / `POST /preflight` | Evaluar aptitud de imagen antes del procesamiento completo |
| ⚠️ pendiente validación | health | `GET /health` | Verificar liveness y readiness del servicio |

## Estados

| Término humano | Término técnico | Valor en código | Descripción |
|---|---|---|---|
| ⚠️ pendiente validación | tipo candidato de abertura | `type_candidate` | `"door"` / `"window"` / `"unknown"` — tentativo, sin decisión semántica final |
| ⚠️ pendiente validación | origen de escala | `scale.source` | `"cotas"` (derivada del plano) / `"none"` (sin referencias) |
| ⚠️ pendiente validación | motor activo | `CV_ENGINE` env var | `"opencv"` (Fase 1 clásico) / `"rasterscan"` (futuro) |

## Términos ambiguos o conflictivos

| Término | Contexto A | Contexto B | Nota |
|---|---|---|---|
| tipo de abertura | En vitrina: tipo final decidido por el LLM (puerta / ventana) | En vitrina-cv: `type_candidate` tentativo emitido por el motor CV | Nunca confundir — cv-service emite candidatas, vitrina decide tipos finales |
| escala | En vitrina-cv: relación px_per_unit derivada del plano | En vitrina (Go): sistema de coordenadas del plano para renderizado | El mapeo lo hace vitrina; cv-service devuelve solo px_per_unit |

## Aliases conocidos

| Alias | Término canónico | Nota |
|---|---|---|
| cv-service | vitrina-cv | Nombre del servicio en ADRs y contratos |
| gate de pre-vuelo | preflight | Endpoint `POST /preflight` con heurísticas puras |
| motor CV | GeometryEngine | Interfaz abstracta seleccionada por `CV_ENGINE` |
