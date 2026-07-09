# SPEC: Fidelidad visual 2D — cierre de junctions (extend-to-intersection) — capa cv-service (vitrina-cv)

> Milestone: 08-fidelidad-visual-2d

## Fuentes consumidas

| Fuente | Tipo | Origen |
|---|---|---|
| ADR-013 — extend-to-intersection para cierre de junctions de muros | ADR | `.project-context/decisions/ADR-013-extend-to-intersection-junctions.md` |
| Fase F4 del motor (`_snap_walls_orthogonal`, `_fuse_junctions`) | código | `src/vitrina_cv/engines/opencv_classic.py:1339-1500` |
| Bloque de settings F4 / thresholds del motor | código | `src/vitrina_cv/config/settings.py` |
| Estructura del DTO `Wall` | código | `src/vitrina_cv/models.py` |
| Estilo de tests de junctions (helpers `_h_wall`/`_v_wall`, patrón de asserts) | código | `tests/test_centerline_junctions.py` |
| Spec del run 08 (formato de referencia) | spec | `/Users/ernestodiaz/projects/vitrina/docs/specs/08-fidelidad-visual-2d/spec-cv-service.md` |

## Contexto y objetivo

Insertar una fase pura `_extend_to_intersection()` entre `_snap_walls_orthogonal` y `_fuse_junctions` en la fase F4 de `opencv_classic.py`, para cerrar junctions de muros en planos ruidosos donde la limpieza de máscara removió los píxeles de esquina. Controlada por la nueva env var `CV_JUNCTION_EXTEND_PX` (default `40`, calibrada para imágenes normalizadas a ~2000px). La fase extiende el extremo de un muro H y el de un muro V hasta su intersección geométrica **solo si** el gap es `<= CV_JUNCTION_EXTEND_PX` y la intersección cae en la prolongación (no en el interior) del segmento, dejando extremos coincidentes que `_fuse_junctions` colapsa en un junction real.

## Scope

- Solo el repo **vitrina-cv**, fase F4 de `src/vitrina_cv/engines/opencv_classic.py`.
- Nueva función pura `list[Wall] -> list[Wall]`, nuevo campo de settings, y una env var documentada en `docker-compose-local.yml` (repo `vitrina`).

## No-objetivos

- No se toca `mask_cleanup` ni la limpieza de máscara (`retain_rectilinear`, thickness filter, etc.).
- No se cambia `_fuse_junctions` — la nueva fase solo la alimenta con extremos ya coincidentes.
- No se cambia `_snap_walls_orthogonal` ni ninguna otra fase del pipeline.
- No se modifica el contrato `Geometry` v0.2.0 — el DTO `Wall` no cambia (`start`, `end`, `thickness`).
- No se tocan otros repos aguas abajo (vitrina-web, backend Go de vitrina) más allá de la env var documentada en compose.
- Cualquier comportamiento no declarado en los criterios de aceptación está fuera de scope.

## Nota de modelo (contrato interno del DTO)

El pseudocódigo de abajo usa `x1/x2/y1/y2` y `orientation` por claridad geométrica. El DTO real (`models.py`) **no tiene esos atributos**: `Wall` expone `start: Point`, `end: Point` y `thickness: float | None`. El developer deriva:

- `orientation`: tras el snapping, un muro es **H** si `start.y == end.y` (misma y), **V** si `start.x == end.x` (misma x). Los diagonales (ni H ni V) se ignoran.
- Para un muro H, los extremos en x son `start.x` y `end.x` (con y común); para un muro V, los extremos en y son `start.y` y `end.y` (con x común).
- Al extender, se reconstruye un nuevo `Wall(...)` preservando `thickness` (mismo patrón que `_snap_walls_orthogonal`).

## Decisiones tomadas (ADR)

Todas las decisiones vienen de **ADR-013** (`Accepted`, en `.project-context/decisions/ADR-013-extend-to-intersection-junctions.md`):

- Se elige extend-to-intersection sobre (a) aumentar `HOUGH_MAX_GAP` y (b) relajar el diagonal filter, por ser la única alternativa que ataca la causa raíz (geometría faltante en la esquina) sin degradar el filtrado de ruido ni regresionar la detección de openings.
- La fase corre **después** del snapping (muros ya H/V exactos → cálculo de intersección trivial) y **antes** de la fusión (para poblar `_junctions`).
- El umbral vive en `settings.py` (convención del módulo: todo threshold desde env, nunca hardcode). `gt=0`; `0` no permitido.
- El developer decide la geometría exacta de "extremo más cercano" y el test de prolongación; el ADR fija solo los invariantes.

## Algoritmo / pseudocódigo

```
para cada par (wall_h, wall_v) donde wall_h es H y wall_v es V:
    ix = x del muro vertical      # wall_v.start.x (== wall_v.end.x)
    iy = y del muro horizontal    # wall_h.start.y (== wall_h.end.y)
    # intersección geométrica = (ix, iy)

    # --- extremos del muro H (varían en x, y = iy fija) ---
    # extremo derecho de wall_h hacia ix (prolongación a la derecha)
    si abs(x2_h - ix) <= CV_JUNCTION_EXTEND_PX y x2_h <= ix:
        x2_h = ix
    # extremo izquierdo de wall_h hacia ix (prolongación a la izquierda)
    si abs(x1_h - ix) <= CV_JUNCTION_EXTEND_PX y x1_h >= ix:
        x1_h = ix

    # --- extremos del muro V (varían en y, x = ix fija) ---
    # extremo inferior de wall_v hacia iy (prolongación hacia abajo)
    si abs(y2_v - iy) <= CV_JUNCTION_EXTEND_PX y y2_v <= iy:
        y2_v = iy
    # extremo superior de wall_v hacia iy (prolongación hacia arriba)
    si abs(y1_v - iy) <= CV_JUNCTION_EXTEND_PX y y1_v >= iy:
        y1_v = iy

# INVARIANTES:
#   - solo pares estrictamente ortogonales (H x V); H-H y V-V se ignoran.
#   - un extremo se mueve SOLO si la intersección cae en su PROLONGACIÓN
#     (fuera del segmento actual), nunca en el interior ya cubierto.
#   - no extender si el gap > CV_JUNCTION_EXTEND_PX.
#   - función pura, misma cardinalidad de entrada/salida: no crea ni borra muros.
#   - idempotencia: si todos los gaps son ~0 (plan-004), devuelve los muros sin cambios.
#   - preservar wall.thickness al reconstruir el Wall.
```

> Nota geométrica: las comparaciones de dirección (`<=`/`>=`) asumen el sistema de coordenadas de imagen (x crece a la derecha, y crece hacia abajo). El developer define el criterio exacto de "prolongación" por extremo — el invariante duro es que la intersección esté fuera del intervalo `[min, max]` cubierto por el segmento en el eje relevante.

## Criterios de aceptación

1. GIVEN una lista de `Wall` post-snapping WHEN se corre `_extend_to_intersection(walls)` THEN devuelve una `list[Wall]` de la misma longitud que la entrada, sin crear ni borrar muros, preservando `thickness`.
   → Ejemplo: entrada `[H(0,100,y=50), V(x=102,y1=50,y2=200)]` (2 muros) → salida con exactamente 2 muros, cada uno con su `thickness` original.
   _Implementa: brief-1_

2. GIVEN el módulo de settings WHEN se instancia `Settings()` THEN existe el campo `cv_junction_extend_px: int` con `default=40` y restricción `gt=0`, leído de la env var `CV_JUNCTION_EXTEND_PX`.
   → Ejemplo: `Settings().cv_junction_extend_px == 40`; `Settings(cv_junction_extend_px=0)` levanta `ValidationError`.
   _Implementa: brief-2_

3. GIVEN el pipeline F4 WHEN corre la extracción de geometría THEN `_extend_to_intersection` se invoca entre `_snap_walls_orthogonal` y `_fuse_junctions`, sobre la salida del snapping y antes de la fusión.
   → Ejemplo: orden de llamadas efectivo `_snap_walls_orthogonal(...) → _extend_to_intersection(...) → _fuse_junctions(...)`; los extremos extendidos coinciden y `_fuse_junctions` emite el junction correspondiente.
   _Implementa: brief-3_

4. GIVEN un plano sintético cuyos muros ya llegan a sus intersecciones (gap ≈ 0, caso plan-004) WHEN corre `_extend_to_intersection` THEN devuelve los muros sin cambios (no-op / idempotente).
   → Ejemplo: `H(0,100,y=50)` + `V(x=100,y1=50,y2=200)` (intersección ya en (100,50)) → coordenadas de salida idénticas a las de entrada.
   _Implementa: brief-4_

5. GIVEN un par H×V con gap `> CV_JUNCTION_EXTEND_PX` WHEN corre `_extend_to_intersection` THEN ningún extremo se mueve.
   → Ejemplo: con `CV_JUNCTION_EXTEND_PX=40`, `H(0,100,y=50)` + `V(x=150,y1=50,y2=200)` (gap 50px) → salida idéntica a la entrada; sin extensión.
   _Implementa: brief-5_

6. GIVEN una mezcla de muros con pares H-H, V-V y H×V WHEN corre `_extend_to_intersection` THEN solo los pares ortogonales H×V se consideran para extensión; los pares paralelos (H-H, V-V) se ignoran y no se tocan.
   → Ejemplo: dos muros H colineales `H(0,100,y=50)` y `H(120,200,y=50)` → ninguno se extiende hacia el otro (no forman esquina).
   _Implementa: brief-6_

7. GIVEN un par H×V donde la intersección cae en el **interior** de un segmento (muros que ya se cruzan) WHEN corre `_extend_to_intersection` THEN ese extremo no se mueve — solo se extiende cuando la intersección cae en la prolongación del extremo.
   → Ejemplo: `H(0,200,y=50)` (cruza x=100) + `V(x=100,y1=0,y2=200)` → el extremo del H no se acorta ni se altera, porque la intersección (100,50) está dentro de `[0,200]`.
   _Implementa: brief-7_

## Señales de alerta

- La decisión se **renumeró a ADR-013** porque la ADR-012 ya existía en el repo; su ubicación canónica es `.project-context/decisions/` (no `docs/adr/`). Verificar que el `architect` haya relocalizado y renumerado el archivo (de `docs/adr/ADR-012-...` a `.project-context/decisions/ADR-013-...`) antes de mergear, y que no exista otra ADR ≥ 013 en conflicto.
- Un par de muros **paralelos** cercanos a ≤ 40px de una intersección proyectada se extiende y crea una esquina que no existe → indica que la restricción "solo H×V ortogonales" no se aplicó.
- Un muro que ya cruzaba a otro se **acorta o modifica** → indica que la restricción de "prolongación, no interior" falló (ver AC-07).
- La cardinalidad de la lista de salida difiere de la de entrada → la fase dejó de ser pura.
- `plan-004` deja de ser no-op (salida ≠ entrada con gaps ≈ 0) → regresión de idempotencia.
- `_fuse_junctions` no produce el junction esperado tras la extensión → la fase no dejó los extremos exactamente coincidentes.
- Hardcodear el umbral `40` en el dominio en lugar de leerlo de `settings.py`.

## Tests requeridos

Ubicación: `tests/test_centerline_junctions.py` (reutilizar helpers `_h_wall`, `_v_wall` y el estilo `pytest.approx`). Sin código aquí — descripción funcional:

| Test | Escenario | Verifica |
|---|---|---|
| Esquina L | `wall_h` corto + `wall_v` corto, gap 30px < 40px | ambos extremos se extienden a la intersección; quedan coincidentes |
| Esquina T | un muro largo + muro perpendicular corto que no llega, gap < umbral | solo el extremo del muro corto se extiende; el largo no se toca |
| Gap > umbral | par H×V con gap 50px > 40px | ningún extremo se mueve (salida = entrada) |
| Pares paralelos | dos muros H (o dos V) | no se tocan entre sí (no forman esquina) |
| Idempotencia plan-004 | par H×V con gaps ≈ 0 | la función devuelve la lista sin cambios de coordenadas |
| Prolongación vs interior (AC-07) | H que ya cruza al V (intersección interior) | el extremo del H no se altera |
| Cardinalidad / thickness | lista con N muros y thickness explícito | salida de N muros con thickness preservado |
| Integración F4 (opcional) | snap → extend → fuse sobre par con gap 30px | `_fuse_junctions` emite ≥ 1 junction en la esquina cerrada |

## Variables de entorno nuevas

| Variable | Default | Tipo | Restricción | Notas |
|---|---|---|---|---|
| `CV_JUNCTION_EXTEND_PX` | `40` | `int` | `gt=0` | Gap máximo (px) para extender un extremo H o V a su intersección ortogonal en F4. Calibrada para imágenes normalizadas a ~2000px (`CV_UPSCALE_TARGET_PX`); a resolución nativa mayor requiere ajuste proporcional. |

## Mapa de implementación

| Acción | Archivo | Qué | Ubicación: por qué aquí |
|---|---|---|---|
| MODIFY | `src/vitrina_cv/engines/opencv_classic.py` | Nueva función pura `_extend_to_intersection(walls: list[Wall]) -> list[Wall]` + integración en el pipeline entre `_snap_walls_orthogonal` y `_fuse_junctions` | Es una fase F4 más, cohesiva con las dos que la rodean (sección "F4 — Orthogonal snapping and junction fusion", ~líneas 1339–1495, la nueva función va entre `_snap_walls_orthogonal` (1344) y `_fuse_junctions` (1390)). Replica el patrón de las fases vecinas: función pura `list[Wall] -> list[Wall]`, docstring con invariantes, `_engine_logger.debug` para trazabilidad. No amerita archivo nuevo. |
| MODIFY | `src/vitrina_cv/config/settings.py` | Nuevo campo `cv_junction_extend_px: int = Field(default=40, gt=0, description=...)` + entrada en el docstring de defaults del módulo | Junto al bloque F4 / junction, siguiendo la convención "todo threshold desde env, nunca hardcode" (mismo patrón que `cv_wall_diagonal_filter_*`, `cv_opening_min_wall_span_px`). |
| MODIFY | `/Users/ernestodiaz/projects/vitrina/docker-compose-local.yml` | Agregar `CV_JUNCTION_EXTEND_PX=40` en el bloque de env del servicio cv-service | En línea con las demás `CV_*` ya presentes en el sidecar CV; documenta el valor operativo por defecto. |

## Impacto cross-service

Ninguno de contrato. El cambio es interno a vitrina-cv: los `Wall[]` emitidos cierran mejor sus esquinas (extremos extendidos ≤ umbral), pero el schema `Geometry` v0.2.0 no varía. La única superficie externa es la env var `CV_JUNCTION_EXTEND_PX` documentada en `docker-compose-local.yml` del repo `vitrina` (aditiva, con default seguro → cambio siempre seguro). No se inspecciona código de repos consumidores.

## Design references

_No aplica — capa CV sin UI._
