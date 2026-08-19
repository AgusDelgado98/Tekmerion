# ADR 0016 — ML evaluation foundation (V0.8.0 Block A)

## Estado
Aceptada (Bloque A). El contrato de evaluación sigue vigente en V0.8 completo.

## Contexto
V0.7 deja un pipeline determinista de role family (regex) usado por Evidence.
Antes de entrenar modelos hace falta un contrato de evaluación reproducible,
con etiquetas humanas independientes de esas reglas.

## Decisión
Agregar `analysis/ml/` **fuera** de Evidence y del pipeline productivo:

1. Gold Dataset `tekmerion.ml.gold_dataset.v1` con `gold_role_family` y `label_source=human`.
2. Split agrupado por fingerprint de título+descripción, seed fija, anti-leakage.
3. Features (título, descripción, skills extraídas) con vocabulario de skills fit en train.
4. Evaluador común (accuracy, precision/recall/F1 por clase, macro F1, confusion matrix).
5. Baseline = `classify_role_family` como **predictor**, nunca como ground truth.
6. Manifest `tekmerion.ml.evaluation_manifest.v1` (dataset hash, split ids, seed, config).

Flask/Evidence no cambian.

## Consecuencias
- La suite productiva sigue midiendo el mismo comportamiento.
- El gold humano se amplió en Bloques B.0–B.1 hasta n=159 (≥10 por clase).
- Bloque B usa este contrato para entrenar sklearn y decidir `promote_ml`.

## Alternativas consideradas
- Usar la regex como gold → filtra el “progreso” de ML hacia el baseline; rechazado.
- sklearn como dependencia de métricas del Bloque A → innecesario; métricas locales.
- Stratified split con clases singleton → indefinido; split agrupado documentado.
