# Technical Domain — vitrina-cv

last_updated: 2026-07-02

## Índice

| Archivo | Propósito |
|---|---|
| [project.md](project.md) | Stack, objetivo, restricciones no negociables, SOLID, qué NO introducir |
| [domain.md](domain.md) | Bounded contexts (extract-geometry, preflight, config), flujos principales, interfaz GeometryEngine |
| [glossary.md](glossary.md) | Glosario negocio ↔ técnico: Geometry, Wall, Room, Opening, Scale, PreflightReport |
| [contracts.md](contracts.md) | API REST: POST /extract-geometry, POST /preflight, GET /health; esquemas completos; interfaz GeometryEngine |
| [business-rules.md](business-rules.md) | Invariantes: stateless, coordenadas en pixeles, error_code enum, scale nunca bloquea, sin semántica en cv-service |
| [dependencies.md](dependencies.md) | Grafo interno y dependencia externa con vitrina (Go) |
| [risks.md](risks.md) | Riesgos: robustez OpenCV, latencia CPU, candidatas perdidas, contrato desincronizado |

## Bounded contexts en este repo

| Context | Responsabilidad |
|---|---|
| `extract-geometry` | Extraer geometría (walls, rooms, openings, scale) via motor CV intercambiable |
| `preflight` | Evaluar aptitud de imagen con heurísticas puras; sin LLM |
| `config` | Configuración centralizada desde variables de entorno |

## ADRs relevantes (fuente: vitrina/docs/specs/06-extraccion-cv-hibrida/adrs/)

| ADR | Decisión |
|---|---|
| ADR-002 | Protocolo vitrina↔cv: bytes en request, sin S3 |
| ADR-003 | Contrato API canónico REST/JSON (OpenAPI 3.1) |
| ADR-005 | Gate pre-vuelo con heurísticas puras, sin LLM |
| ADR-008 | Motor CV intercambiable via CV_ENGINE; Fase 1 = OpenCV clásico |
| ADR-009 | Aberturas como candidatas; decisión final en LLM de vitrina |
