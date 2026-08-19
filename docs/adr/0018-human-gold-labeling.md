# ADR 0018 — Human gold labeling workflow (V0.8)

## Estado
Aceptada

## Decisión
Ampliar gold solo con etiquetas humanas sobre candidatos unlabeled:

1. Fetch Adzuna explícito (`scripts/fetch_gold_candidates.py`) escribe snapshots **sin** `gold_role_family`.
2. Harvest dedupe por fingerprint de título+descripción.
3. CLI `scripts/label_gold.py`: título, descripción, skills, query/origen (contexto, no label) y distribución del gold.
4. Decisiones: familias evaluables o `skip/ambiguous`. Sesión incremental local (`label_session.json`, no versionada).
5. Merge a gold solo de `decision=label`. Prohibido copiar `classify_role_family`, regex o ML.
6. Cola prioriza familias con pocos ejemplos usando *hints de revisión*; el hint no se guarda como label.
7. El gold etiquetado con texto de vacantes **no se versiona**. Snapshots raw y el JSON de 159 ejemplos quedan gitignored. GitHub publica schema, fixtures, distribución, hash, métricas y manifests sanitizados.

Si no hay credenciales Adzuna ni snapshots live, no se inventan vacantes.

Vacancy data used in the ML evaluation was sourced from [The Adzuna API](https://developer.adzuna.com/).

## Resultado
Gold local n=159 (16 sintéticos, 4 curados, 139 snippets Adzuna únicos). No redistribuido.
`data_analyst` se cerró con un fetch dirigido; no se forzó ninguna etiqueta dudosa.
