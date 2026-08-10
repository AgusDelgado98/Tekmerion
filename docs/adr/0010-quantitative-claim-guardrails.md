# ADR 0010 — Quantitative claim guardrails (V0.5.1)

## Estado

Aceptada

## Contexto

V0.5.0 validaba evidence_refs y porcentajes globales de forma liviana.
Un modelo podía citar refs correctas e inventar counts/denominadores.

## Decisión

1. **NumericEvidenceIndex** derivado del GroundingPayload (counts, percents, valores embebidos en rankings, dataset size).
2. **Extractor conservador** de claims: `%` (`.`/`,`), `X de/of Y`, `N + sustantivo` (vacantes, registros, …).
3. **Ignorar** años, timestamps ISO, `gpt-*`, `market_summary.v*`, versiones `vX.Y`, ids largos.
4. **Por finding**: números deben ser soportados por los valores de sus `evidence_refs` (o por `dataset.n_analysis_records` como denominador).
5. **Summary / limitations**: índice global.
6. **Errores** `UnsupportedNumericClaim` con value/unit/location/reason.
7. Prompt activo **market_summary.v2** (disciplina de refs por número).
8. Sin segundo LLM, sin embeddings, sin verificación semántica general.

## Qué NO cubre

Afirmaciones cualitativas, causalidad, rankings en prosa sin patrón estructurado, equivalencias semánticas, conocimiento externo sin números.

## Consecuencias

Mayor rechazo de alucinaciones cuantitativas explícitas; riesgo residual de claims no numéricos engañosos.
