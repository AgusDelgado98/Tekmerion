# Case study — Tekmérion

## Context

Roles de Data, BI y AI se publican en múltiples plataformas, con títulos inconsistentes,
descripciones incompletas y poca estructura. Comparar “qué piden” entre familias de rol
es difícil sin un pipeline que separe señal de ruido.

## Problem

¿Cómo transformar vacantes dispersas en **evidencia comparable**, y solo después
explicarla, sin que un modelo generativo invente el mercado?

## Approach

1. **Ingestión desacoplada** (adapters, p.ej. Adzuna) con procedencia (`source`, `source_url`, `retrieved_at`).
2. **Pipeline determinista**: normalización, role family, seniority, skills, deduplicación.
3. **Market batch**: varias queries → un artifact `tekmerion.market_batch.v1`.
4. **EvidenceReport**: conteos, rankings, co-ocurrencias — calculados en código, no por el LLM.
5. **Grounded AI**: el modelo recibe un `GroundingPayload` y produce narrativa validada
   (refs, números, rankings, scope de roles).
6. **Flask demo**: synthetic / showroom / market local; la UI no llama APIs externas.

## Architecture decisions (resumen)

- Identity namespaced por fuente → evita colisiones entre adapters.
- Evidence antes que generación → el LLM no es fuente de verdad.
- Guardrails locales → rechazan claims cuantitativos y rankings no soportados.
- Showroom versionado → demo clonable sin secretos.

## Results (verificables en el repo)

- Demo offline con **showroom** (~14 vacantes, varias role families).
- Dos tareas grounded: `market_summary`, `role_comparison`.
- Un adapter de API real (Adzuna) + fixtures offline.
- Suite automatizada **253+** tests (pipeline, ingestión, Flask, guardrails).

## Reliability

- Determinismo en IDs, merge, evidence y fingerprints de grounding.
- Provenance en registros reales.
- Provider `fake` explícito para demos (no se confunde con live).
- Runtime web sin dependencia de red para datasets locales.

## Limitations

- Showroom ≠ mercado laboral completo.
- Descriptions de APIs pueden ser snippets.
- Skill no detectada ≠ skill no requerida.
- Guardrails no cubren toda la prosa cualitativa ni causalidad.
- Sin auth, deploy ni scheduler de producción.

## What I learned

1. Separar métricas de narrativa reduce alucinaciones de forma medible.
2. Contratos de artifact (`market_batch.v1`) simplifican la UI y los tests.
3. Un “demo provider” determinista es más honesto que fingir llamadas live.
4. Registry + sesión bastan para multi-dataset sin base de datos.
5. Documentar limitaciones aumenta credibilidad frente a hiring managers técnicos.
