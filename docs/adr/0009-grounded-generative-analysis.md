# ADR 0009 — Grounded generative analysis (V0.5.0)

## Estado

Aceptada

## Contexto

Tekmérion ya produce EvidenceReport determinista. Se necesita una capa narrativa
sin que el modelo invente hechos de mercado ni reemplace el pipeline.

## Decisión

1. **GroundingPayload** derivado solo de EvidenceReport + DatasetMeta, con
   `evidence_ref` ids estables (`skills.sql.pct`, `roles.data_analyst.count`, …).
2. **GenerativeProvider** abstracto:
   - `disabled` (default)
   - `fake` (determinista, tests/demo)
   - `openai_compatible` (HTTP `/v1/chat/completions`, un solo proveedor real)
3. Única tarea: `market_summary`. Sin chatbot ni prompts libres del usuario.
4. Prompt versionado (`market_summary.v1`) con instrucciones anti-hallucination.
5. **GeneratedAnalysis** estructurado + validación determinista:
   - refs deben existir en grounding
   - summary/findings/limitations no vacíos
   - porcentajes literales en texto deben coincidir con values de grounding
6. Flask `/analysis`: GET no llama al modelo; POST genera explícitamente.
7. Credenciales solo por env; ausencia de key no rompe el app.

### Por qué openai_compatible

Un endpoint HTTP estándar cubre OpenAI y clones sin SDK pesado ni multi-provider.
La lógica de negocio no importa el vendor.

## Consecuencias

- Portfolio demuestra “IA sobre evidencia”, no “IA como fuente de verdad”.
- Alucinaciones de refs se bloquean; prosa no cuantitativa no se audita palabra a palabra.
- Costos controlados (acción explícita + cache en memoria por fingerprint).

## Alternativas

- Llamar al LLM con vacantes raw → viola evidence-first.
- Multi-provider en V0.5.0 → complejidad innecesaria.
- Auto-generar en cada page load → costo e irreproducibilidad.
