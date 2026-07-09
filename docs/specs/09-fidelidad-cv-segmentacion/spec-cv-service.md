# SPEC: Fidelidad CV / segmentación — pipeline de limpieza de máscara y filtro de diagonal residual — run-09-cv

> Milestone: run 09 (fidelidad CV — segmentación)

## Fuentes consumidas

| Fuente | Tipo | Origen |
|---|---|---|
| Architecture View — backend (pipeline mask_cleanup) | arch-view | `/Users/ernestodiaz/projects/vitrina-cv/docs/specs/09-fidelidad-cv-segmentacion/arch-backend.md` |
| ADR-014 — Kernel rectilíneo adaptativo baja-res | adr | `.project-context/decisions/ADR-014-kernel-rectilineo-adaptativo-baja-res.md` |
| ADR-015 — Crop preserva componentes múltiples | adr | `.project-context/decisions/ADR-015-crop-preserva-componentes-multiples.md` |
| ADR-016 — Trazabilidad diagnóstica pipeline cleanup | adr | `.project-context/decisions/ADR-016-trazabilidad-diagnostica-pipeline-cleanup.md` |
| ADR-017 — Filtro de diagonal residual robusto | adr | `.project-context/decisions/ADR-017-filtro-diagonal-residual-robusto.md` |

## Contexto y objetivo

El pipeline clásico de `vitrina-cv` pierde fidelidad geométrica en la fase de limpieza de máscara y en la consolidación de muros: en planos de baja resolución con achurado denso el kernel rectilíneo fijo de 150px erosiona esquinas y corridas H/V cortas (P0, plan-001-denso-achurado 1346×788); el crop al único componente mayor descarta alas de footprints no contiguos / plantas en L (P1); y sobrevive una diagonal residual (stub de escalera/puerta) que ni el snap ortogonal ni el filtro de banda `[20°,70°]` capturan (P3, plan-002/004/005). Este spec cubre los cuatro cambios (ADR-014/015/016/017) sobre `mask_cleanup.py` y `opencv_classic.py`, todos calibrados como no-op sobre los planos rectilíneos contiguos ya validados, con un gate de no-regresión sobre los 5 fixtures de eval.

## No-objetivos

- No modifica la detección de muros (HoughLinesP), la CCA de habitaciones ni el cierre direccional — solo las fases previas (limpieza, crop) y un pase posterior al fuse.
- No cambia la tolerancia de snap ortogonal (`_SNAP_ANGLE_TOL_DEG`) ni el orden de F4 (ADR-013 intacto).
- No implementa la variante por-componente del crop (ADR-015): es un fallback condicional que solo se documenta como flag futura si el gate de regresión la exige; no se codifica especulativamente.
- No estima densidad de achurado dinámicamente (opción (c) de ADR-014, descartada).
- No vuelca máscaras intermedias a disco (opción (b) de ADR-016, descartada); la inspección offline sigue en `eval/tools/diag_mask.py`.
- No baja el default global de `CV_CLEANUP_RECTILINEAR_LEN_PX` (opción (a) de ADR-014, descartada).

## Pre-condiciones

- [ ] Dataset de eval disponible con los 5 fixtures y sus `ground_truth.json`: `plan-001-denso-achurado`, `plan-002-simple-limpio`, `plan-003-reticula-cotas`, `plan-004-sintetico-alta-res`, `plan-005-amueblado-limpio`.
- [ ] Harness E2E `fidelity-loop.mjs` operativo sobre los 5 fixtures.
- [ ] Resoluciones "por verificar" de plan-002/003/005 confirmadas antes de fijar constantes (invariante de verificación de ADR-014).

## Decisiones tomadas (ADR)

### ADR-014: Kernel rectilíneo adaptativo en baja-res

- **Opciones consideradas:** (a) bajar el default global de `CV_CLEANUP_RECTILINEAR_LEN_PX` (regresiona por resolución en todo el espectro) · (b) kernel proporcional a la resolución normalizada, con piso configurable (elegida) · (c) estimador de densidad de achurado (sobre-ingeniería, superficie de fallo grande).
- **Decisión:** (b). El branch de baja-res (`resolution_scale_raw <= max_res_scale`) usa la misma forma adaptativa que ya vive en el branch alta-res: `L_efectivo = max(L_min, round(L_base * min(h,w) / CV_UPSCALE_TARGET_PX))`, con `L_base = CV_CLEANUP_RECTILINEAR_LEN_PX` (150) y `L_min = CV_CLEANUP_RECTILINEAR_MIN_LEN_PX` (NEW, 50). Elimina la discontinuidad en `resolution_scale = 1.0` y unifica el literal `50` hoy hardcodeado en ambos branches.
- **Tradeoff aceptado:** en imágenes muy pequeñas `L_efectivo` cae al piso `L_min` y puede seguir siendo grande relativo a la imagen; el piso evita el extremo opuesto (kernel que no suprime nada) y es ajustable por env var.

### ADR-015: Crop preserva componentes significativos (envolvente)

- **Opciones consideradas:** ratio de área puro in situ (no elimina cotas/marcos fuera de las alas) · unión de bboxes de los N mayores (infla la caja si una cota es grande) · **unión de bboxes de los componentes que superan un ratio de área mínimo** (elegida, combina ambas mitigando sus contras).
- **Decisión:** un componente es significativo si `area >= CV_CLEANUP_CROP_MIN_AREA_RATIO * A_max` (NEW, default 0.05). El crop recorta a la envolvente (bbox unión) de los significativos + margen, clampeado a la imagen. Cuando hay un solo componente significativo → envolvente == bbox del mayor → comportamiento idéntico al actual. La máscara resultante sigue siendo un rectángulo contiguo; el pipeline downstream (Hough/CCA/cierre) no cambia sus supuestos.
- **Tradeoff aceptado:** si una cota/marco sobrevive como componente grande, su bbox infla la envolvente y re-admite ruido. Mitigado por los pasos 1-2 y el ratio mínimo; fallback por-componente disponible solo si el gate lo exige.

### ADR-016: Trazabilidad diagnóstica simétrica y branch canónico

- **Opciones consideradas:** (a) no hacer nada / diagnosticar caso a caso · (b) volcar máscaras intermedias a disco (I/O en hot path, viola pureza) · (c) logging estructurado simétrico + branch canónico (elegida).
- **Decisión:** (c). `clean_mask_steps_1_to_3` emite los mismos logs por paso que `clean_mask` para los pasos 1-3 (deja de ser caja negra). El log de step2 incluye `branch` ∈ {`fixed`|`adaptive`|`skip`} más los insumos de la fórmula (`long_side`, `min_hw`, `upscale_target_px`, `rectilinear_len_px_used`, `min_len_px`). El log de step3 incluye `significant_components_count` y `crop_bbox_xywh`. INFO estructurado vía `extra={...}`; campos como valores/enums, nunca prosa. **Cero cambio de comportamiento** — solo observabilidad.
- **Tradeoff aceptado:** ligero aumento de volumen de logs (dos rutas) y duplicación de la lógica de logging; ~6-8 líneas INFO por imagen, aceptable.

### ADR-017: Filtro de diagonal residual de doble mecanismo

- **Opciones consideradas:** (a) solo ampliar la banda `[low,high]` a `[5,85]` (roza la tolerancia de snap, no cubre C2) · (b) solo segundo pase por ángulo (deja abierta la banda muerta `[5,20)∪(70,85]`) · (c) bajar la tolerancia de snap (cambia F4/ADR-013) · **(d) doble mecanismo** (elegida).
- **Decisión:** (d). Tras `_snap_walls_orthogonal → _extend_to_intersection → _fuse_junctions` se añade un pase 2 con: **Mec.1** — re-evaluar el ángulo de cada `Wall` y descartar la misma banda `[low_deg,high_deg]` del pase 1 (idempotente: los muros snapeados son 0°/90° exactos → nunca caen); **Mec.2** — descartar todo `Wall` que (a) no sea H/V exacto tras el snap **y** (b) tenga longitud euclidiana `< CV_WALL_MIN_DIAGONAL_LEN_PX` (NEW). Ambos gated por el master switch existente `cv_wall_diagonal_filter_enabled`. Robusto a que la causa real sea C1 (stub casi-ortogonal corto en banda muerta) o C2 (residual materializado post-fuse) sin medir el ángulo exacto del stub.
- **Tradeoff aceptado:** un muro diagonal genuino y corto podría descartarse por Mec.2 (raro en un motor rectilíneo-orientado que ya descarta diagonales por diseño). El default de `CV_WALL_MIN_DIAGONAL_LEN_PX` se fija sin el ángulo/longitud medidos del stub — calibración final contra plan-002/004/005 queda como validación del developer/tester.

## Criterios de aceptación

### Kernel rectilíneo adaptativo (P0 — ADR-014)

1. GIVEN un plano de baja resolución con achurado denso (`resolution_scale_raw <= max_res_scale`) WHEN corre el paso 2 de `clean_mask` THEN el kernel efectivo se dimensiona proporcional a la resolución normalizada, con piso `CV_CLEANUP_RECTILINEAR_MIN_LEN_PX`, y ya no erosiona las esquinas ni las corridas H/V cortas legítimas.
   → Ejemplo: `plan-001-denso-achurado` (1346×788, `min_hw=788`) → `L_efectivo = max(50, round(150 * 788 / 2000)) = max(50, 59) = 59` px (antes: 150 px fijo). El plano recupera sus 12 habitaciones contra `ground_truth.json`.
   _Implementa: brief-1_

2. GIVEN un plano calibrado a ~2000px (`min(h,w) ≈ 2000`) WHEN corre el paso 2 THEN `L_efectivo ≈ L_base` (150) — no-op respecto del comportamiento previo.
   → Ejemplo: `min_hw=2000` → `L_efectivo = max(50, round(150 * 2000/2000)) = 150`. Idéntico al kernel fijo previo; sin cambio en la máscara de salida.
   _Implementa: brief-1_

3. GIVEN un plano alta-res (`resolution_scale_raw > 1.0`, plan-004) WHEN corre el paso 2 THEN el comportamiento del branch adaptativo queda idéntico al actual (la fórmula ya vivía ahí; el literal `50` ahora proviene de `CV_CLEANUP_RECTILINEAR_MIN_LEN_PX`).
   → Ejemplo: `plan-004-sintetico-alta-res` produce la misma máscara y los mismos `expected_rooms` que antes del cambio.
   _Implementa: brief-1_

### Crop multi-componente (P1 — ADR-015)

4. GIVEN un footprint no contiguo (planta en L / alas separadas) donde el achurado abrió la esquina de unión WHEN corre el paso 3 (`crop_to_main_component`) THEN el crop recorta a la envolvente de los componentes con `area >= CV_CLEANUP_CROP_MIN_AREA_RATIO * A_max`, preservando ambas alas dentro de un rectángulo contiguo.
   → Ejemplo: `plan-001` con cuerpo superior (`A_max`) + ala inferior derecha de área `0.4·A_max` (≥ 0.05·A_max) → ambos entran en la envolvente; el Salón-Comedor del ala ya no se pierde.
   _Implementa: brief-2_

5. GIVEN una planta rectangular contigua con un único componente significativo WHEN corre el paso 3 THEN la envolvente == bbox del componente mayor — comportamiento idéntico al actual.
   → Ejemplo: `plan-002-simple-limpio` → `significant_components_count=1` → mismo `crop_bbox_xywh` que antes; sin regresión de `room_areas_m2`.
   _Implementa: brief-2_

6. GIVEN una cota o marco perimetral que sobrevive como componente pequeño (`area < 0.05 * A_max`) WHEN corre el paso 3 THEN ese componente NO entra en la envolvente y el marco se elimina como hoy.
   → Ejemplo: una cota residual de área `0.02·A_max` queda fuera del conjunto significativo → no infla la caja → el crop la descarta.
   _Implementa: brief-2_

### Trazabilidad diagnóstica (P3-observabilidad — ADR-016)

7. GIVEN cualquier extracción que recorre `clean_mask_steps_1_to_3` (ruta de ventanas/escaleras) WHEN completa los pasos 1-3 THEN emite los mismos logs INFO estructurados por paso que `clean_mask` (deja de ser caja negra).
   → Ejemplo: una detección de escaleras genera las líneas `cv_cleanup_step1_small_components`, `cv_cleanup_step2_rectilinear`, `cv_cleanup_step3_crop` con `extra={...}`, tal como la ruta principal.
   _Implementa: brief-3_

8. GIVEN el log de step2 WHEN se emite THEN incluye `branch` con valor de {`fixed`|`adaptive`|`skip`} más `long_side`, `min_hw`, `upscale_target_px`, `rectilinear_len_px_used`, `min_len_px`; y el log de step3 incluye `significant_components_count` y `crop_bbox_xywh` — todos como valores, nunca prosa.
   → Ejemplo: `plan-001` → `cv_cleanup_step2_rectilinear branch=fixed long_side=1346 min_hw=788 upscale_target_px=2000 rectilinear_len_px_used=59 min_len_px=50` y `cv_cleanup_step3_crop significant_components_count=2 crop_bbox_xywh=[...]`.
   _Implementa: brief-3_

### Filtro de diagonal residual (P3 — ADR-017)

9. GIVEN un stub oblicuo que sobrevive al pase 1 y cae en la banda `[low_deg,high_deg]` tras `_fuse_junctions` (causa C2) WHEN corre el pase 2 (Mec.1) THEN se descarta re-aplicando la misma banda del pase 1, sin banda nueva.
   → Ejemplo: un tramo reconstituido a 45° post-fuse cae en `[20,70]` → Mec.1 lo descarta; el `count` de descartes por ángulo del pase 2 se incrementa en 1.
   _Implementa: brief-4_

10. GIVEN un `Wall` que no es H/V exacto tras el snap y tiene longitud euclidiana `< CV_WALL_MIN_DIAGONAL_LEN_PX` (banda muerta `[5,20)∪(70,85]`, causa C1) WHEN corre el pase 2 (Mec.2) THEN se descarta por longitud; el filtro NO aplica a muros H/V exactos.
   → Ejemplo: un stub de escalera a 12° de 30 px (< default) → descartado por Mec.2; un muro H/V exacto de 30 px → conservado (no es candidato). El `count` de descartes por longitud se incrementa en 1.
   _Implementa: brief-4_

11. GIVEN el master switch `cv_wall_diagonal_filter_enabled=False` WHEN corre la consolidación de muros THEN ni el pase 1 ni el pase 2 (Mec.1 ni Mec.2) se ejecutan — compatibilidad pre-08 intacta.
   → Ejemplo: con el flag en False, la salida de muros es byte-idéntica a la del comportamiento pre-run-08.
   _Implementa: brief-4_

12. GIVEN una planta rectilínea contigua sin diagonal residual (plan-003) WHEN corre el pase 2 THEN es no-op: los muros ortogonales (0°/90°) nunca caen en la banda y nunca son candidatos del filtro de longitud.
   → Ejemplo: `plan-003-reticula-cotas` → `count` por ángulo = 0 y `count` por longitud = 0 en el pase 2; misma cantidad de muros que antes.
   _Implementa: brief-4_

### No-regresión (gate global)

13. GIVEN los 5 fixtures de eval WHEN corre el harness E2E `fidelity-loop.mjs` con todos los cambios aplicados THEN plan-001 mejora (recupera 12 habitaciones y sus áreas) y plan-002/003/004/005 no regresionan (`expected_rooms` y `room_areas_m2` dentro de tolerancia contra cada `ground_truth.json`).
   → Ejemplo: corrida completa → plan-001 pasa de N<12 a 12 habitaciones; plan-002..005 mantienen su conteo y áreas previos; el harness cierra en verde.
   _Implementa: brief-5_

## Señales de alerta

- El kernel adaptativo cambia la máscara de plan-002/003/004/005 (debe ser no-op sobre planos calibrados a ~2000px).
- La envolvente del crop re-admite un marco/cota perimetral (la caja se infla por un componente que debía descartarse).
- El pase 2 del filtro de diagonal descarta muros H/V exactos (Mec.1 no es idempotente, o Mec.2 se aplica a ortogonales).
- Con `cv_wall_diagonal_filter_enabled=False` cualquier pase del filtro se ejecuta.
- `clean_mask_steps_1_to_3` sigue sin emitir logs, o `branch` aparece como string libre en vez de un valor del conjunto cerrado.
- Se hardcodea el literal `50` o cualquier umbral en px en lugar de leerlo de env var (viola la convención dura del módulo).
- Se implementa la variante por-componente del crop (ADR-015) sin que el gate de regresión la haya exigido.

## Tests por criterio de aceptación

| AC | Tipo | Qué verifica |
|---|---|---|
| AC-1: kernel adaptativo baja-res corrige plan-001 | unit + e2e | `L_efectivo` proporcional con piso; plan-001 recupera sus habitaciones |
| AC-2: no-op en el target ~2000px | unit | `L_efectivo ≈ 150` cuando `min_hw ≈ 2000` |
| AC-3: branch alta-res idéntico | unit | plan-004 sin cambio; literal `50` viene de env var |
| AC-4: envolvente preserva alas del footprint no contiguo | unit + e2e | crop a la unión de bboxes significativos; ala del plan-001 preservada |
| AC-5: crop no-op con un solo componente significativo | unit | envolvente == bbox del mayor; plan-002 sin regresión |
| AC-6: ratio mínimo excluye cotas pequeñas | unit | componente `< 0.05·A_max` no entra en la envolvente |
| AC-7: logging simétrico en `clean_mask_steps_1_to_3` | unit | los 3 eventos por paso presentes en la ruta ventanas/escaleras |
| AC-8: campos canónicos en step2/step3 | unit | `branch` enum + insumos de fórmula; `significant_components_count`, `crop_bbox_xywh` |
| AC-9: Mec.1 descarta residual post-fuse (C2) | unit | pase 2 por ángulo descarta banda; `count` ángulo incrementa |
| AC-10: Mec.2 descarta oblicuo corto (C1) | unit | filtro de longitud solo a oblicuos; H/V exacto conservado |
| AC-11: master switch off desactiva ambos pases | unit | con flag False, salida byte-idéntica pre-08 |
| AC-12: pase 2 no-op sobre planta rectilínea | unit + e2e | plan-003 con `count` ángulo=0 y longitud=0 |
| AC-13: gate de no-regresión sobre los 5 fixtures | e2e | `fidelity-loop.mjs` verde; plan-001 mejora, resto no regresiona |

## Requerimientos de observabilidad

- **Logs (pipeline cleanup — ADR-016):** eventos INFO estructurados `cv_cleanup_step1_small_components`, `cv_cleanup_step2_rectilinear`, `cv_cleanup_step3_crop` emitidos simétricamente por `clean_mask` y `clean_mask_steps_1_to_3`. step2: `branch` ∈ {`fixed`|`adaptive`|`skip`}, `long_side`, `min_hw`, `upscale_target_px`, `rectilinear_len_px_used`, `min_len_px`. step3: `significant_components_count`, `crop_bbox_xywh`. Todos como valores/enums, vía `extra={...}`.
- **Logs (filtro diagonal — ADR-017):** el pase 2 emite un evento estructurado con `count` de descartes **por ángulo** y **por longitud** como campos separados (valores, no prosa), para que el gate de eval distinga qué mecanismo actuó por fixture.
- **Métricas:** N/A — pipeline batch de extracción; la observabilidad relevante es de logs estructurados consumidos por el harness de eval, no counters/gauges de runtime.
- **Spans / traces:** N/A — no hay operaciones distribuidas nuevas.

## Variables de entorno nuevas

| Variable | Ejemplo | Secreto | Notas |
|---|---|---|---|
| `CV_CLEANUP_RECTILINEAR_MIN_LEN_PX` | `50` | No | Piso del kernel adaptativo del paso 2 (`gt=0`). Reemplaza el literal `50` hardcodeado en ambos branches. Documentar en `docker-compose-local.yml` (repo `vitrina`). |
| `CV_CLEANUP_CROP_MIN_AREA_RATIO` | `0.05` | No | Ratio de área mínimo para que un componente sea significativo en el crop (`ge=0.0`, `le=1.0`). Documentar en `docker-compose-local.yml`. |
| `CV_WALL_MIN_DIAGONAL_LEN_PX` | `40` | No | Longitud euclidiana mínima (px, calibrada a ~2000px) por debajo de la cual un `Wall` oblicuo sobreviviente se descarta (Mec.2, ADR-017). Gated por `cv_wall_diagonal_filter_enabled`. Default a calibrar contra plan-002/004/005; documentar en `docker-compose-local.yml`. |

<!-- ADR-016 no introduce env var: la observabilidad es incondicional (controlada por el nivel de logging global de la app). -->
