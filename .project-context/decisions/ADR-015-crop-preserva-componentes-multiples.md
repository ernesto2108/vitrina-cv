# ADR-015 — Preservación de componentes múltiples en el crop de la máscara

> Milestone: run 09 (fidelidad CV — segmentación) | Motivado por: defecto P1 de fidelidad geométrica (footprint no rectangular / L-shape)

## Status
Accepted

## Context

El paso 3 de `mask_cleanup.clean_mask` (`crop_to_main_component`,
`src/vitrina_cv/mask_cleanup.py:249`) elimina cotas perimetrales y marcos de
escaneo recortando la máscara al bounding box del **único componente conexo de
mayor área** (+ margen). Su contrato actual (líneas 282-305):

1. Corre CCA sobre la máscara ya limpia (pasos 1-2).
2. Selecciona el label con `CC_STAT_AREA` máxima (un solo componente).
3. Pone a cero todo lo que cae fuera del bbox de ese componente + margen.

**Causa raíz del defecto P1.** Cuando el achurado denso o el paso 2 fragmentan el
muro perimetral, o cuando el plano tiene un **footprint no contiguo** (planta en L,
alas separadas por una abertura ancha, cuerpo principal + porche), la envolvente
del plano deja de ser un solo componente conexo. El fixture
`plan-001-denso-achurado` (1346×788, verificado visualmente) tiene precisamente
esta forma: un cuerpo rectangular superior conectado y un ala inferior derecha
(Salón-Comedor) que puede quedar como componente separado si el achurado abre la
esquina de unión. En ese caso `crop_to_main_component` conserva solo el bbox del
componente mayor y **descarta el resto del plano**, perdiendo habitaciones enteras
aguas abajo.

La consecuencia downstream es delicada: el consumidor del paso 3 es la detección
de muros (HoughLinesP) y la CCA de habitaciones. Ambas corren sobre la máscara
recortada. Si se preserva una **región no contigua** (dos alas separadas), la
detección de muros no se rompe — HoughLinesP opera sobre segmentos locales, no
requiere conectividad global — pero la CCA de habitaciones y la fase de cierre
direccional (`cv_room_close_h_gap_px`/`v_gap_px`) sí asumen que el espacio entre
componentes preservados es "fondo", lo que es correcto (no debe cerrarse a través
del vacío entre alas). El riesgo real no es romper wall detection sino **re-admitir
ruido** (cotas o marcos que sobrevivieron como componentes grandes) al ampliar el
criterio de "qué se conserva".

**Estrategias de preservación consideradas (el humano delegó la elección):**

- **Ratio de área mínimo:** conservar todo componente cuya área sea `>= ratio *
  area_del_mayor`. Pro: simple, un parámetro. Contra: sensible a la escala relativa
  — un ala legítima pequeña (baño, pasillo) puede caer bajo el ratio; una cota
  larga puede superarlo.
- **Unión de bboxes (crop a la envolvente de los N mayores):** tomar los K
  componentes mayores y recortar a la unión de sus bboxes. Pro: preserva alas
  separadas dentro de una sola caja contigua, manteniendo el pipeline downstream
  sobre una máscara rectangular contigua (sin cambiar sus supuestos). Contra: si
  una cota perimetral quedó como componente grande, su bbox infla la envolvente y
  re-admite el marco que el paso 3 debía eliminar.

## Decision

Adoptar la **unión de bboxes de los componentes que superen un ratio de área
mínimo**, combinando ambas ideas y mitigando sus contras mutuos. El contrato pasa
de "bbox del mayor" a "envolvente de los componentes significativos". Invariantes
(no implementación):

- Sea `A_max` el área del componente mayor. Un componente es **significativo** si
  `area >= CV_CLEANUP_CROP_MIN_AREA_RATIO * A_max` (NEW, default `0.05`).
- El crop resultante es el bbox que envuelve **la unión de los bboxes** de todos
  los componentes significativos, expandido por `CV_CLEANUP_CROP_MARGIN_PX`,
  clampeado a los límites de la imagen.
- Invariante de compatibilidad: cuando solo hay un componente significativo
  (planta rectangular contigua, el caso hoy), la envolvente == bbox del mayor →
  **comportamiento idéntico al actual**. plan-002/003/005 (limpios/contiguos) no
  cambian.
- Invariante anti-ruido: el ratio mínimo excluye cotas/marcos que sobreviven como
  componentes pequeños; solo alas de tamaño comparable al cuerpo entran a la
  envolvente. Una cota que sí sea grande sigue siendo un riesgo residual (ver
  Consequences → guarda opcional).
- La máscara resultante **sigue siendo un rectángulo contiguo** (una sola caja),
  por lo que el pipeline downstream (Hough + CCA + cierre direccional) NO cambia
  sus supuestos: no recibe una región no contigua, recibe una envolvente mayor que
  contiene ambas alas y el vacío entre ellas como fondo. Esto evita la necesidad de
  procesar cada componente por separado y fusionar resultados.

**Guarda explícita para el footprint no contiguo (condicional).** Si el gate de
regresión demuestra que la envolvente única re-admite ruido perimetral de forma
que degrada la detección (una cota grande infla la caja y reintroduce el marco),
el ADR habilita como fallback la **variante por-componente**: en lugar de recortar
a una sola envolvente, poner a cero todo lo que caiga fuera de la **unión de los
bboxes individuales** (no de su envolvente) — es decir, conservar cada bbox
significativo pero cero en las regiones intermedias que no pertenecen a ninguno.
Esto preserva la geometría de cada ala sin re-admitir el rectángulo completo entre
ellas, a costa de una máscara no rectangular. Esta variante NO cambia los supuestos
de Hough (opera local) pero debe validarse contra la CCA de habitaciones; se activa
solo si la envolvente simple falla el gate.

**Alternativa descartada — ratio de área puro (sin unión de bboxes):** conservar
los componentes significativos *in situ* sin recortar a ninguna caja. Descartada
porque no elimina cotas/marcos que caen fuera de las alas pero superan el ratio, y
deja la máscara con la misma dispersión que el paso 3 busca acotar.

## Consequences

**Positivas:**

- Preserva plantas en L / footprints no contiguos: el ala descartada hoy entra en
  la envolvente y sus habitaciones se detectan.
- No-op sobre plantas rectangulares contiguas (invariante de compatibilidad) →
  plan-002/003/005 sin regresión.
- La máscara sigue siendo un rectángulo contiguo en el caso por defecto: cero
  cambios en los supuestos de Hough y CCA downstream.

**Negativas / Trade-offs aceptados:**

- **Re-admisión de ruido si una cota/marco sobrevive como componente grande:** su
  bbox infla la envolvente. Mitigación primaria: el paso 2 (retain_rectilinear, ver
  ADR-014) y el paso 1 ya eliminan la mayoría de cotas; el ratio mínimo filtra las
  pequeñas. Mitigación secundaria: la guarda por-componente (fallback) si el gate lo
  exige.
- **Parámetro nuevo a calibrar** (`CV_CLEANUP_CROP_MIN_AREA_RATIO`): un ratio muy
  bajo re-admite ruido; muy alto descarta alas legítimas. Default conservador 0.05,
  ajustable por env var.
- La variante por-componente (si se activa) produce una máscara no rectangular; hay
  que verificar que la CCA de habitaciones y el cierre direccional la toleran (no
  cierran a través de las regiones puestas a cero entre alas). Es el único punto que
  requiere validación downstream explícita.

## Implementation notes

- **`CV_CLEANUP_CROP_MIN_AREA_RATIO`** — NEW campo `float` en `Settings`
  (`src/vitrina_cv/config/settings.py`), default `0.05`, `ge=0.0`, `le=1.0`, junto
  a `cv_cleanup_crop_margin_px` (~línea 251). Agregar entrada al docstring de
  defaults. Justificación de ubicación: mismo bounded context de cleanup/crop.
- `crop_to_main_component` (`mask_cleanup.py:249`) cambia su firma/semántica para
  aceptar el ratio y devolver la envolvente de los componentes significativos. El
  developer decide si añade un parámetro o una función hermana; este ADR fija el
  invariante de compatibilidad (un componente → mismo bbox de hoy).
- El return type actual `tuple[mask, bbox|None]` se conserva; el `bbox` reportado
  pasa a ser la envolvente (o `None` si no hay foreground).
- La guarda por-componente es un camino condicional; documentarla como flag futura
  solo si el gate de regresión la exige — no implementarla especulativamente.
- Actualizar `docker-compose-local.yml` (repo `vitrina`) con
  `CV_CLEANUP_CROP_MIN_AREA_RATIO=0.05`.
- Referencia cruzada: depende de ADR-014 — con el kernel adaptativo, la esquina de
  unión del ala en plan-001 tiene más probabilidad de quedar conectada, reduciendo
  la frecuencia con que se activa la preservación multi-componente. Los dos ADRs se
  refuerzan.
- Gate de regresión: correr `eval/` sobre plan-001..005 y comparar `expected_rooms`
  y `room_areas_m2` contra cada `ground_truth.json`.
