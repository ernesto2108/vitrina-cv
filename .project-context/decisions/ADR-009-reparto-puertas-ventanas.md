# ADR-009 — Aberturas como candidatas (sin decisión semántica en cv-service)

fecha: 2026-07-02
estado: aceptado
fuente: vitrina/docs/specs/06-extraccion-cv-hibrida/adrs/ADR-009-reparto-puertas-ventanas.md

## Decisión

cv-service detecta huecos en paredes y emite openings con type_candidate tentativo (door/window/unknown), bbox y confidence. La clasificación final la hace el LLM en vitrina. cv-service nunca decide el tipo final ni descarta falsos positivos.

## Consecuencias

- Si CV no detecta un hueco, el LLM no puede recuperarlo — la calidad de detección determina el techo de precisión.
- La interfaz es más simple: cv-service solo detecta geometría, sin lógica semántica.
- vitrina es responsable de la clasificación final y del descarte de falsos positivos.
