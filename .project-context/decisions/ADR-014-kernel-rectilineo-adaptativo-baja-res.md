# ADR-014 — Kernel rectilíneo adaptativo a la resolución en imágenes por debajo del target de upscale

> Milestone: run 09 (fidelidad CV — segmentación) | Motivado por: defecto P0 de fidelidad geométrica (plan-001-denso-achurado)

## Status
Accepted

## Context

El paso 2 de `mask_cleanup.clean_mask` (`retain_rectilinear`,
`src/vitrina_cv/mask_cleanup.py:120`) aplica dos aperturas morfológicas
direccionales con un kernel de longitud `L = CV_CLEANUP_RECTILINEAR_LEN_PX`
(default `150`, `settings.py:200`) para suprimir el achurado diagonal y retener
solo corridas H/V. La lógica de branching en `clean_mask` (líneas 449-510)
selecciona el kernel según la escala de resolución:

```
resolution_scale_raw = long_side_raw / CV_UPSCALE_TARGET_PX   # target = 2000
```

- **Branch fijo** (`resolution_scale_raw <= CV_CLEANUP_RECTILINEAR_MAX_RES_SCALE`,
  default cap `1.0`) → usa el kernel FIJO de 150px (línea 468).
- **Branch adaptativo** (`resolution_scale_raw > 1.0` y flag adaptativa on) →
  usa `max(50, round(150 * min(h,w) / 2000))` (línea 483, introducido en 08-cv-03).

**Causa raíz del defecto P0 (verificada).** El fixture `plan-001-denso-achurado`
tiene resolución **1346×788 px** (`long_side_raw = 1346`). Como
`1346 / 2000 = 0.67 <= 1.0`, la imagen entra por el **branch fijo** y recibe el
kernel de 150px sin adaptación. A esta resolución nativa (menor al target), 150px
es un fragmento enorme del plano: para el patrón de achurado denso de este plano
específico, el open direccional de 150px erosiona las esquinas de los muros y
fragmentos de corridas H/V cortas legítimas. El resultado: huecos de muro y
esquinas abiertas aguas abajo. El ADR-013 ya documentó el mismo efecto colateral
de `retain_rectilinear@150px` sobre las esquinas ("removes valid junction corner
pieces that are critical for room-boundary closure") — este ADR ataca la misma
raíz por el lado del dimensionamiento del kernel.

El precedente de fórmula adaptativa **existe pero solo cubre el branch alta-res**
(>2000px). El branch de baja-res nunca fue adaptado. Nótese también que en este
repo hubo un precedente de fórmula adaptativa desactivada por regresión
(`cv_room_close_scale_with_upscale`, `settings.py:351`, default `False` "porque la
multiplicación completa por upscale_factor es demasiado agresiva") — evidencia de
que una fórmula proporcional mal calibrada regresiona fixtures que comparten
resolución de origen. Cualquier fórmula nueva debe ser un no-op sobre los planos
que hoy pasan.

**Evidencia de riesgo de regresión (resoluciones de los fixtures):**

| Fixture | Long side | Branch actual | Riesgo con fórmula proporcional |
|---|---|---|---|
| plan-001-denso-achurado | 1346 px (verificado) | fijo 150 | el defecto — se busca corregir |
| plan-002-simple-limpio | por verificar (≤2000 asumido) | fijo 150 | bajo: sin achurado denso |
| plan-003-reticula-cotas | por verificar (≤2000 asumido) | fijo 150 | medio: retícula H/V dominante |
| plan-004-sintetico-alta-res | >2000 px (alta-res, dir name) | adaptativo | nulo: no toca el branch fijo |
| plan-005-amueblado-limpio | por verificar (≤2000 asumido) | fijo 150 | bajo: limpio |

> Invariante de verificación: el developer/QA debe confirmar las resoluciones
> marcadas "por verificar" antes de fijar la constante de la fórmula, ejecutando
> el harness de eval sobre los 5 planos con la fórmula propuesta y comparando
> `expected_rooms` / áreas contra `ground_truth.json`.

## Decision

Adoptar la **opción (b): hacer el kernel del paso 2 proporcional a la resolución
de la imagen normalizada también en el branch de baja-res** (`resolution_scale_raw
<= max_res_scale`), reutilizando la misma forma de fórmula que ya existe en el
branch alta-res, con un piso configurable. La longitud efectiva pasa de:

```
L_fijo = CV_CLEANUP_RECTILINEAR_LEN_PX                              # hoy
```

a la forma adaptativa, invariante (no implementación):

- Invariante: `L_efectivo = max(L_min, round(L_base * min(h,w) / CV_UPSCALE_TARGET_PX))`
  donde `L_base = CV_CLEANUP_RECTILINEAR_LEN_PX` y `L_min = CV_CLEANUP_RECTILINEAR_MIN_LEN_PX` (NEW).
- Invariante de no-op sobre el target: cuando `min(h,w) ≈ CV_UPSCALE_TARGET_PX`,
  `L_efectivo ≈ L_base` (150) — los planos ya calibrados a ~2000px no cambian.
- Invariante de monotonía: a menor resolución, menor kernel — 150px sobre 1346px
  se reduce proporcionalmente, dejando de erosionar esquinas.
- El comportamiento del branch alta-res (línea 478-500) queda **idéntico**: la
  fórmula ya vive ahí; este ADR la unifica hacia el branch fijo bajo un solo
  parámetro, eliminando la discontinuidad en `resolution_scale = 1.0`.

**Alternativas consideradas:**

- **(a) Bajar el default de `CV_CLEANUP_RECTILINEAR_LEN_PX` globalmente.** Pro:
  cambio de un solo número, sin lógica nueva. Contra: un valor menor global
  degrada la supresión de achurado en planos que hoy están bien calibrados a
  ~2000px (plan-004 y cualquier imagen alta-res que dependa del kernel completo);
  cambia el comportamiento en todo el espectro de resolución en vez de solo donde
  falla. Descartada: regresiona por resolución en lugar de corregir por resolución.
- **(b) Kernel proporcional a la resolución normalizada (elegida).** Ataca la
  causa raíz (kernel sobredimensionado *relativo a la imagen*), reutiliza fórmula
  ya probada en el branch alta-res, elimina la discontinuidad en el cap, y es
  no-op sobre los planos calibrados a 2000px. Riesgo controlado por el piso
  `L_min` y por el gate de regresión sobre los 5 fixtures.
- **(c) Detectar densidad de achurado y ajustar el kernel dinámicamente.** Pro:
  el más preciso conceptualmente (adapta al contenido, no solo a la resolución).
  Contra: introduce un estimador de densidad nuevo (heurística no trivial,
  superficie de fallo grande, más difícil de calibrar y testear determinísticamente),
  y el achurado ya se suprime por corridas H/V cortas — el problema no es *cuánto*
  achurado hay sino que el kernel es grande *relativo a la imagen*. Sobre-ingeniería
  para la causa raíz observada. Descartada por costo/riesgo frente al beneficio.

**Config (contrato de env var):**

- `CV_CLEANUP_RECTILINEAR_MIN_LEN_PX: int` (NEW), default `50`, `gt=0`. Piso del
  kernel adaptativo — mismo valor que el piso ya hardcodeado en el branch alta-res
  (`max(50, ...)`, línea 483/599), ahora extraído a env var para no volver a
  hardcodear un umbral (convención dura del módulo, `settings.py:3`). Reemplaza el
  literal `50` en ambos branches.
- `CV_CLEANUP_RECTILINEAR_LEN_PX` conserva default `150` (semántica: longitud base
  calibrada al target de 2000px, no longitud absoluta aplicada).

## Consequences

**Positivas:**

- Corrige plan-001 sin tocar la lógica de detección de muros ni la limpieza de
  otras fases — el kernel se dimensiona a la escala real de la imagen.
- Elimina la discontinuidad de comportamiento en `resolution_scale = 1.0`: hoy un
  plano a 1999px usa 150px y uno a 2001px usa la fórmula adaptativa; tras el cambio
  ambos lados de la frontera se comportan de forma continua.
- Unifica la fórmula en un solo lugar conceptual; el literal `50` deja de estar
  hardcodeado (se extrae a `CV_CLEANUP_RECTILINEAR_MIN_LEN_PX`).
- No-op verificable sobre planos calibrados a ~2000px (invariante de no-op).

**Negativas / Trade-offs aceptados:**

- **Riesgo de bajo poder de supresión de achurado en imágenes muy pequeñas:** si
  `min(h,w)` es muy chico, `L_efectivo` cae al piso `L_min=50`; si 50px sigue
  siendo grande relativo a esa imagen, el problema persiste, pero el piso evita el
  extremo opuesto (kernel de pocos px que no suprime nada). Mitigación: el piso es
  env var, ajustable por despliegue.
- **Requiere gate de regresión sobre los 5 fixtures** antes de fijar la fórmula
  (ver tabla de evidencia). El cambio es de bajo riesgo pero toca una fase
  compartida por wall/room/opening detection.
- `clean_mask` y `clean_mask_steps_1_to_3` deben aplicar la fórmula de forma
  idéntica (hoy duplican la lógica del branch alta-res, líneas 478-500 y 596-607);
  la unificación debe cubrir ambas para no divergir.

## Implementation notes

- **`CV_CLEANUP_RECTILINEAR_MIN_LEN_PX`** — NEW campo en `Settings`
  (`src/vitrina_cv/config/settings.py`), junto al bloque de mask cleanup (tras
  `cv_cleanup_rectilinear_max_res_scale`, ~línea 229). Agregar su entrada al
  docstring de defaults del módulo (bloque líneas 17-49). Justificación de
  ubicación: pertenece al mismo bounded context de cleanup que las demás
  `cv_cleanup_rectilinear_*`; no amerita archivo nuevo.
- El developer decide la geometría exacta del cálculo de `L_efectivo` y si usa
  `min(h,w)` o `long_side`; este ADR fija el invariante (proporcional a la
  resolución normalizada, no-op en el target, piso configurable). Preferir
  `min(h,w)` por consistencia con el branch alta-res ya existente (línea 485).
- Extraer el literal `50` de las líneas 483 y 599 hacia el nuevo campo.
- Actualizar `docker-compose-local.yml` (repo `vitrina`) con
  `CV_CLEANUP_RECTILINEAR_MIN_LEN_PX=50` como env var documentada del sidecar CV.
- Referencia cruzada: ADR-013 documenta el mismo efecto de erosión de esquinas de
  `retain_rectilinear` — este ADR reduce ese efecto en baja-res; ADR-013 lo
  compensa aguas abajo con `_extend_to_intersection`. Son complementarios.
- Gate de regresión: correr `eval/` sobre plan-001..005 y comparar contra cada
  `ground_truth.json` (`expected_rooms`, `room_areas_m2`) antes de aceptar.
