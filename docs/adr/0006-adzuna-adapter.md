# ADR 0006 — Adaptador Adzuna (primera fuente externa)

## Estado

Aceptada

## Contexto

ADR 0001 anticipaba Adzuna como fuente de vacantes reales actuales. V0.4.2 dejó lista la capa de ingestión (adapters, IDs namespaced, `IngestionContext`). Faltaba el primer adaptador HTTP real.

Documentación oficial revisada (2026-08-10):

- Base: `https://api.adzuna.com/v1/api`
- Búsqueda: `GET /jobs/{country}/search/{page}`
- Auth: query params `app_id` + `app_key` (no header Bearer)
- Descripción: **solo un snippet**, no el texto completo del aviso
- País en el path (`ar`, `gb`, …), no como filtro genérico único
- `results_per_page` configurable; sin SDK oficial obligatorio

Diferencias respecto a lo asumido en ADR 0001:

- No hay un endpoint “Argentina-only” especial: se usa el country code estándar.
- La descripción truncada limita la extracción de skills respecto a avisos completos.
- No se usa la API de Intelligence (producto distinto, auth distinta).

## Decisión

1. **Adaptador** `analysis/ingestion/adzuna.py` implementando `SourceAdapter`.
2. **HTTP** con `urllib` (stdlib) + función inyectable `http_get` para tests sin red.
3. **Credenciales** solo por env: `ADZUNA_APP_ID`, `ADZUNA_API_KEY`.
4. **Mapping** Adzuna → dict compatible con `normalize_to_internal`; metadata extra en `source_metadata`.
5. **Snapshots** opcionales en `data/raw/real/adzuna/` (gitignored salvo `.gitkeep`).
6. **Tests** 100 % offline con fixture `tests/fixtures/adzuna/search_response.json`.
7. **CLI** `scripts/fetch_adzuna.py` para corridas live manuales (fuera de pytest).
8. **Límites conservadores**: 1 página, `results_per_page` default 10, techo 50.

Fuente canónica: `source = "adzuna"`.

## Consecuencias

- Tekmérion puede ingerir vacantes reales con trazabilidad (snapshot + `retrieved_at` + ids namespaced).
- pytest no depende de internet ni de secretos.
- Flask sigue en muestra sintética; no hay selector de fuentes todavía.
- Skills/evidence sobre Adzuna serán más pobres que sobre descripciones completas (limitación de la API pública de search).

## Alternativas consideradas

- SDK de terceros → dependencia innecesaria para un GET JSON.
- `requests` → útil, pero stdlib alcanza y mantiene el árbol mínimo.
- Paginar todo el mercado → fuera de alcance; riesgo de rate limits y ruido.
