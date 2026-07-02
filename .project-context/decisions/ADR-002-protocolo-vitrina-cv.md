# ADR-002 — Protocolo vitrina↔cv: bytes en request, sin S3

fecha: 2026-07-02
estado: aceptado
fuente: vitrina/docs/specs/06-extraccion-cv-hibrida/adrs/ADR-002-protocolo-vitrina-cv.md

## Decisión

vitrina descarga el PNG de S3 y envía los bytes en el request multipart a cv-service. El servicio no accede a S3, no tiene credenciales de infra, y es stateless.

## Consecuencias

- cv-service no tiene dependencias de infra (S3, DB, Redis).
- El tamaño del request está limitado por la imagen PNG recibida — monitorear para imágenes grandes.
- vitrina es responsable del ciclo de vida del archivo en S3.
