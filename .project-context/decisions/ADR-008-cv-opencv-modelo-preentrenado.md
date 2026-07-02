# ADR-008 — Motor CV intercambiable: Fase 1 OpenCV clásico, inferencia CPU

fecha: 2026-07-02
estado: aceptado
fuente: vitrina/docs/specs/06-extraccion-cv-hibrida/adrs/ADR-008-cv-opencv-modelo-preentrenado.md

## Decisión

Fase 1 = OpenCV clásico (binarización, detección de líneas, contornos), expuesto detrás de la interfaz GeometryEngine parametrizada e intercambiable (CV_ENGINE). Inferencia en CPU. Objetivo p95 < 20 s. CubiCasa5k descartado (licencia CC BY-NC, prohibido uso comercial). RasterScan en evaluación paralela como motor futuro — no bloquea Fase 1.

## Consecuencias

- El contrato de salida (ADR-003) es independiente del motor — cambiar motor no cambia la API.
- OpenCV clásico puede ser menos robusto que un modelo ML ante variabilidad de planos.
- CV_MODEL_PATH es opcional/futuro — no usada en Fase 1.
- NUNCA integrar CubiCasa5k ni pesos con licencia no comercial.
