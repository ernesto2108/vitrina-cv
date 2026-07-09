---
name: "CV09-02-kernel-rectilineo-adaptativo-baja-res"
type: "implementation"
priority: "HIGH"
agent: "developer-backend"
points: 3
milestone: "run-09-cv"
feature_id: "CV09"
dependencies: ["CV09-01-nuevas-env-vars-settings"]
inputs:
  - "settings.cv_cleanup_rectilinear_len_px: int (L_base, ya existe)"
  - "settings.cv_cleanup_rectilinear_min_len_px: int (L_min, nuevo — CV09-01)"
  - "settings.cv_upscale_target_px: int (ya existe)"
outputs:
  - "clean_mask() paso 2: kernel efectivo adaptativo también en el branch de baja resolución"
validation_rules:
  - "L_efectivo = max(L_min, round(L_base * min(h,w) / upscale_target_px))"
---

# CV09-02-kernel-rectilineo-adaptativo-baja-res

## Objetivo
Que el paso 2 (`retain_rectilinear`) de `clean_mask()` en `mask_cleanup.py` deje de erosionar esquinas y corridas H/V cortas legítimas en planos de baja resolución con achurado denso, aplicando la misma fórmula de kernel adaptativo que ya existe en el branch alta-res, ahora también en el branch fijo (`resolution_scale_raw <= max_res_scale`).

## Contexto Técnico
Referencia: ADR-014 (`.project-context/decisions/ADR-014-kernel-rectilineo-adaptativo-baja-res.md`), spec AC-1, AC-2, AC-3.

Fórmula: `L_efectivo = max(cv_cleanup_rectilinear_min_len_px, round(cv_cleanup_rectilinear_len_px * min(h,w) / cv_upscale_target_px))`.

Archivo: `src/vitrina_cv/mask_cleanup.py`, función `retain_rectilinear()` (~líneas 120-153) y el branch condicional dentro de `clean_mask()` (~líneas 460-510) que hoy decide entre kernel fijo (150px) y kernel adaptativo según si la imagen es alta-res.

**No-objetivo explícito (del spec):** no cambiar el comportamiento del branch alta-res ya existente, solo unificar el literal `50` hardcodeado para que provenga de `cv_cleanup_rectilinear_min_len_px`, y aplicar la misma fórmula al branch de baja resolución.

## Interfaces
- Llamado por: `clean_mask()` (paso 2 del pipeline de limpieza de máscara)
- Llama a: `settings` (Pydantic Settings, CV09-01)

## Criterios de Aceptación
- [ ] AC-1: en `plan-001-denso-achurado` (1346×788, min_hw=788), `L_efectivo = max(50, round(150*788/2000)) = 59` en vez del kernel fijo de 150px.
- [ ] AC-2: en un plano calibrado a ~2000px (min_hw≈2000), `L_efectivo ≈ 150` — no-op respecto al comportamiento previo.
- [ ] AC-3: el branch alta-res (plan-004, resolution_scale_raw > 1.0) produce exactamente la misma máscara que antes del cambio; el literal `50` ahora proviene de `cv_cleanup_rectilinear_min_len_px`.
- [ ] No se hardcodea ningún umbral en px fuera de `settings.py` (convención dura del módulo).
