# ADR 0011 — Ranking claim guardrails (V0.5.2)

## Estado

Aceptada

## Contexto

V0.5.1 auditaba percentages/counts/denominators. Faltaban afirmaciones de posición
(`#1`, `puesto 2`) y superlativos de frecuencia.

## Decisión

1. **RankingEvidenceIndex** desde `skills.ranking`, `roles.ranking`, `seniority.ranking`
   (orden canónico del grounding; no se recalcula).
2. **Posición técnica** (1-based): acepta `#N` / `rank N` / `puesto N` / ordinales 1–3
   si el item está en esa posición de la lista citada.
3. **Liderazgo estadístico** (`más frecuente` / `most frequent` / `top skill`):
   solo si `is_unique_leader` (count estrictamente mayor). Empates → rechazo.
4. Findings validan solo contra ranking refs citadas.
5. Normalización de items: casefold + espacios/guiones → `_` (p.ej. `BI Analyst` → `bi_analyst`).
6. El `1` de `#1` no se interpreta como count numérico.
7. Prompt **market_summary.v3**.

## Política de empates

El orden del ranking aplica tie-break determinista (count desc, nombre asc).
Eso define posición técnica. No implica superioridad estadística única.

## Fuera de alcance

Frases vagas (“top”, “una de las principales”), ordinales > 3, co-occurrence como ranking de entidades.
