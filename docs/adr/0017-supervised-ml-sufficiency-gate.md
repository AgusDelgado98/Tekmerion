# ADR 0017 — Supervised ML behind a data-sufficiency gate (V0.8 Block B)

## Estado
Aceptada (Bloque B cerrado con `promote_ml=false`)

## Contexto
El Bloque A dejó un contrato de Gold Dataset humano, split anti-leakage y métricas
comparables. Bloque B pide, *solo si* el gate se cumple, entrenar sklearn contra
el baseline de reglas.

## Decisión
1. Harvest unlabeled desde ingestión local; dedupe por fingerprint **antes** de etiquetar.
2. Ampliar gold solo con etiquetas humanas. **No** inventar vacantes ni copiar la regex.
3. Gate: n≥100 y ≥10 por clase presente.
4. Si el gate falla → `DATA INSUFFICIENT`, no se llama a `fit`.
5. Si el gate pasa → Logistic Regression, Linear SVM, Random Forest; TF-IDF+skills
   fit en train/CV fold; GridSearch acotado; mismas métricas del Bloque A.
6. Criterio de promoción: test **macro F1** (no accuracy). `promote_ml=true` solo
   con lift ≥0.02 y sin más clases con support y F1=0 que el baseline.
7. sklearn es extra (`pip install -e ".[ml]"` / `.[dev]`). Flask/Evidence no
   importan `analysis.ml.train`.

## Resultado en este repo
Gold local n=159 (no redistribuido), gate SUFFICIENT. Test 112/47. Rules macro F1 **0.866** vs Linear SVM
**0.816** (accuracy empatada 0.872). **`promote_ml=false`**. Bloque C no se abre
por este ADR. Públicos: schema, fixtures, evaluation card, reports sanitizados y el chart.

## Alternativas consideradas
- Relajar el gate → rechazado.
- Etiquetar fixtures de test como “reales” → rechazado.
- Promover ML aunque no gane a las reglas → rechazado.
