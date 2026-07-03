<!-- SECCIONES FIJAS (preservar literalmente): Modos de trabajo, Reglas por modo, Para agentes
     SECCIONES A RELLENAR (sustituir placeholders <...>): Estrategia de ramas, Proceso de PR, Ambientes, Comandos operativos, Variables de entorno -->

# Workflows del Equipo — vitrina-cv

<!-- Cómo trabaja el equipo: ramas, PRs, ambientes y proceso de deploy.
     También incluye comandos operativos para levantar, buildear, testear y operar. -->

last_updated: 2026-07-02

## Modos de trabajo

El equipo opera bajo cinco modos según el tipo de cambio. Cada modo determina qué pasos del workflow son obligatorios y cuáles se omiten.

| Modo | Cuándo usarlo | ¿Actualiza business-rules? | ¿Actualiza contracts? | ¿Crea ADR? | ¿Requiere PR review? |
|---|---|---|---|---|---|
| `feature` | Nueva funcionalidad visible para el usuario o nuevo capability del sistema | sí | sí | solo si hay decisión arquitectónica | sí |
| `bug` | Corrección de comportamiento incorrecto observado en producción o staging | no (si la cambia, escalar a feature) | solo si hay cambio de contrato | solo si hay decisión arquitectónica | sí |
| `fix` | Corrección técnica menor (typo, refactor puntual, ajuste de config) | no | no | no | depende del equipo |
| `chore` | Mantenimiento técnico (upgrade de dependencia, linting masivo, reorganización de carpetas) | no | no | no | depende del equipo |
| `spike` | Investigación o prototipo descartable; no va a producción | no | no | no | no |

### Reglas por modo

**feature** — nueva funcionalidad. Puede cambiar reglas de negocio, contratos, patrones y dominio. Requiere tests, lint, validación con el humano y `reporter` (con diff completo para que actualice `.project-context/`).

**bug** — corrección de comportamiento incorrecto. NO debe cambiar reglas de negocio; si las cambia, escalar a `feature`. Solo actualiza `risks.md` si revela un gotcha nuevo. Requiere tests que reproduzcan el bug, lint y validación con el humano. `reporter` obligatorio.

**fix** — corrección técnica menor. No cambia reglas ni contratos. `reporter` obligatorio.

**chore** — mantenimiento técnico. No cambia comportamiento observable. `reporter` obligatorio.

**spike** — investigación o prototipo. No va a producción. No requiere tests. Solo documentar hallazgos en `runs/` vía `reporter`.

### Para agentes

Al inicio de cualquier run, preguntar al developer el modo de trabajo (`feature`, `bug`, `fix`, `chore`, `spike`) antes de implementar. El modo determina qué pasos del workflow son obligatorios y cuáles se omiten. Si el modo no está claro en el prompt del usuario, preguntar explícitamente antes de continuar; no asumirlo.

Si el cambio toca más de un servicio, cargar `cross-service-dev` antes de implementar — no continuar en modo single-repo.

## Estrategia de ramas

- **Rama principal:** `main`
- **Ramas de desarrollo:** ramas de feature desde `main`
- **Convención de nombres:** `feature/<descripcion>` / `fix/<descripcion>` / `chore/<descripcion>`
- **Rama de release:** ninguna definida aún — repo recién creado

## Proceso de PR

> Repo recién creado — proceso de PR por definir con el equipo. Completar con el primer milestone.

1. Crear rama desde `main`
2. Implementar, pasar tests y lint
3. Abrir PR con descripción y criterios de aceptación
4. Revisión requerida (número de revisores por definir)
5. Merge a `main`

## Ambientes

| Ambiente | Rama | URL / Acceso | Deploy |
|---|---|---|---|
| Development | `main` (local) | `http://localhost:8000` | manual |
| Staging | por definir | por definir | por definir |
| Production | `main` | `http://cv-service.internal` (sidecar interno) | por definir |

## Proceso de deploy

> Repo recién creado — proceso de deploy por definir. Ver `spec-infra.md` en vitrina para referencia de empaquetado Docker.

## Comandos operativos

### Desarrollo local

```bash
# Instalar dependencias (incluye dev tools: pytest, ruff, mypy)
uv sync --extra dev

# Levantar servicio con hot-reload
PYTHONPATH=src uvicorn vitrina_cv.main:app --reload --port 8000

# O sin reload (producción-like)
PYTHONPATH=src uvicorn vitrina_cv.main:app --port 8000
```

### Build

```bash
# Docker build (task 06-infra-01 — aún no implementado)
```

### Tests

```bash
# Todos los tests
PYTHONPATH=src pytest

# Con coverage
PYTHONPATH=src pytest --cov=src/vitrina_cv --cov-report=term-missing
```

### Lint y formato

```bash
# Verificar
ruff check . && ruff format --check .

# Auto-fix
ruff check --fix . && ruff format .
```

## Variables de entorno requeridas

| Variable | Ejemplo | Para qué |
|---|---|---|
| `CV_ENGINE` | `opencv` | Motor CV activo. Fase 1: `opencv`. Futuro: `rasterscan` (ADR-008) |
| `CV_MODEL_PATH` | `` (vacío) | Ruta a pesos de motor ML. Opcional/futuro — no usada en Fase 1 (ADR-008) |
| `CV_PREFLIGHT_MIN_RESOLUTION` | `300x300` | Piso no rescatable (imagen original antes de upscale). Imágenes bajo este umbral no producen geometría útil ni con upscale (ADR-005). Cambiado de 800x600. |
| `CV_PREFLIGHT_MIN_CONTRAST` | `0.35` | Umbral de contraste del gate (ADR-005) — evaluado sobre imagen normalizada |
| `CV_PREFLIGHT_MIN_LINE_DENSITY` | `0.05` | Umbral de densidad de líneas del gate (ADR-005) — evaluado sobre imagen normalizada |
| `CV_UPSCALE_TARGET_PX` | `2000` | Lado mayor objetivo tras normalización. Imágenes ya >= este valor no se modifican (factor 1.0). |
| `CV_UPSCALE_MAX_FACTOR` | `4.0` | Cap del factor de upscale; evita inflar thumbnails minúsculos a tamaños impracticables. |
| `CV_SCALE_OCR_ENABLED` | `true` | Master switch de OCR de cotas (ADR-011). En `false`, `_detect_scale` retorna `source=none` sin llamar a tesseract. |
| `CV_SCALE_OCR_CONSISTENCY_TOLERANCE` | `0.10` | Desviación relativa máxima aceptada entre lecturas de cota para el consensus check. `0.10` = 10%. |
| `CV_SCALE_OCR_TESSERACT_CMD` | `` | Path opcional al binario tesseract. Vacío = autodetección en PATH. Ejemplo: `/opt/homebrew/bin/tesseract`. |
