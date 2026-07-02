# Infra Services

Generado por: context-init
Última actualización: 2026-07-02

## Servicios detectados

No se detectaron servicios de infraestructura declarados (sin docker-compose.yml, sin .env.example, sin archivos de dependencias — repo recién creado).

## Notas

- El servicio `vitrina-cv` es stateless: sin base de datos, sin Redis, sin Kafka, sin S3 (ADR-002).
- Al agregar un `docker-compose.yml` o `requirements.txt` con clientes de infra, re-ejecutar `context-init mode: regular` para actualizar este archivo.
- Si en el futuro se añade algún servicio de soporte (ej. Redis para caché de warmup), definirlo primero en un ADR antes de integrarlo.
