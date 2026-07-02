# Gestión de Tareas — vitrina-cv

<!-- Dónde viven las tareas, convenciones de tickets y definition of done. -->

last_updated: 2026-07-02

## Herramienta de gestión

- **Herramienta:** por definir — no consultado en este bootstrap (repo nuevo)
- **Workspace / Proyecto:** por definir
- **Acceso:** por definir

## Convenciones de tickets

### Tipos de ticket

| Tipo | Prefijo / Label | Descripción |
|---|---|---|
| Feature | `feat` | Nueva funcionalidad |
| Bug | `fix` | Corrección de error |
| Chore | `chore` | Mantenimiento técnico |
| Spike | `spike` | Investigación |

### Campos obligatorios por ticket

- **Título:** descripción en imperativo (ej: `Implementar GeometryEngine con OpenCVClassicEngine`)
- **Descripción:** contexto, criterios de aceptación referenciados del spec
- **Criterios de aceptación:** formato GIVEN/WHEN/THEN (alineados con `spec-cv-service.md`)

## Definition of Done

Un ticket se considera terminado cuando:

- [ ] Código implementado y revisado
- [ ] Tests escritos y pasando (pytest)
- [ ] Lint sin errores (ruff / mypy)
- [ ] PR aprobado
- [ ] Contrato OpenAPI validado si se tocó algún endpoint
- [ ] `.project-context/` actualizado (via reporter)

## Flujo de estados

```
Backlog → In Progress → In Review → Done
```

## Estimación

- **Sistema:** por definir con el equipo
- **Quién estima:** por definir con el equipo

## Ceremonias del equipo

Por definir — repo recién creado.
