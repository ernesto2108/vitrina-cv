# Patrones de Diseño — vitrina-cv

last_updated: 2026-07-02

<!-- Este archivo se construye por inferencia estructural, no por nombres.
     Un patrón puede llamarse de cualquier manera o no tener nombre explícito.
     La firma del código es lo que importa. -->

## Creacionales

### Factory de motor CV via variable de entorno — engines/
- **Archivo:** `engines/` (por crear en scaffold)
- **Qué construye:** instancia de `GeometryEngine` según `CV_ENGINE`
- **Firma prevista:**
  ```python
  def get_engine(cv_engine: str) -> GeometryEngine:
      match cv_engine:
          case "opencv": return OpenCVClassicEngine()
          case "rasterscan": return RasterScanEngine()
          case _: raise ValueError(f"Unknown CV_ENGINE: {cv_engine}")
  ```
- **Cuándo usar:** al inicializar el servicio o en el startup de FastAPI
- **Anti-pattern:** NO instanciar `OpenCVClassicEngine` o cualquier motor concreto directamente en los routers — siempre pasar por la factory

## Estructurales

### Settings por Pydantic BaseSettings — config/settings.py
- **Archivo:** `config/settings.py` (por crear)
- **Qué encapsula:** lectura y validación de variables de entorno (`CV_ENGINE`, `CV_MODEL_PATH`, `CV_PREFLIGHT_*`)
- **Firma prevista:** clase `Settings(BaseSettings)` con valores por defecto y validación de tipos
- **Cuándo usar:** al agregar cualquier umbral o configuración nueva — nunca hardcodear valores

## De comportamiento

### Strategy — GeometryEngine
- **Archivo:** `engines/base.py` (por crear)
- **Qué varía:** el algoritmo de extracción de geometría (OpenCV clásico vs. motor ML futuro)
- **Implementaciones previstas:** `OpenCVClassicEngine`, `RasterScanEngine` (futuro)
- **Cuándo agregar nueva implementación:**
  1. Crear clase en `engines/<nombre>.py` que implemente `GeometryEngine`
  2. Registrar en la factory de `engines/`
  3. Agregar valor a `CV_ENGINE` en variables de entorno
  4. Actualizar `contracts.md` y `risks.md` en `.project-context/`

## Python-específicos

### DTOs Pydantic v2 centralizados — models.py
- **Archivo:** `src/vitrina_cv/models.py`
- **Tipo:** módulo de tipos único que espeja el OpenAPI canónico
- **Cuándo usar:** al definir cualquier tipo de request/response del contrato
- **Patrón detectado en:** `engines/base.py` (retorna `Geometry`), `engines/opencv_classic.py`, `preflight/checks.py` (retorna `PreflightReport`)
- **Anti-pattern:** definir Pydantic models inline en routers o duplicarlos en engines

### Dependency Injection via FastAPI lifespan — app startup
- **Archivo:** `main.py` o `app.py` (por crear)
- **Tipo:** el motor CV se instancia una vez en el startup de FastAPI (lifespan) y se inyecta en los routers via `Depends`
- **Cuándo usar:** para el motor CV y cualquier recurso inicializado una sola vez (evitar reinicialización por request)

## Patrones a evitar en este proyecto

- **Motor hardcodeado en router:** viola ADR-008 y hace imposible el cambio de motor sin tocar código de producción
- **Retención de estado entre requests:** viola ADR-002; el servicio es stateless por diseño
- **Llamadas a LLM externo en cualquier handler:** viola ADR-005 (preflight) y el objetivo del producto
- **Importar directamente clientes S3/DB en cualquier módulo:** viola ADR-002
