# ADR-005 — Gate de pre-vuelo con heurísticas puras (sin LLM)

fecha: 2026-07-02
estado: aceptado
fuente: vitrina/docs/specs/06-extraccion-cv-hibrida/adrs/ADR-005-gate-pre-vuelo.md

## Decisión

POST /preflight evalúa con heurísticas deterministas de imagen (resolución, contraste, densidad de líneas, orientación) y devuelve PreflightReport con suggestions. Sin llamada LLM. Umbrales configurables via CV_PREFLIGHT_* env vars.

## Consecuencias

- Falsos negativos posibles (rechazar plano válido atípico), mitigable calibrando umbrales.
- Sin latencia ni costo de LLM en el gate.
- Los umbrales son configuración, no código — no requieren redeploy para ajustar.
