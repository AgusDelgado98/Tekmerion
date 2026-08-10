# ADR 0007 — Market Batch multi-query (V0.4.4)

## Estado

Aceptada

## Contexto

V0.4.3 permite una query Adzuna a la vez. Para evidencia de mercado hace falta consultar varias familias de roles en una sola corrida, consolidar por identidad y producir un artefacto reutilizable (futuro input de Flask) sin volver a llamar a la API.

Comportamiento previo observado:

- El mismo `adzuna:<id>` se genera de forma estable desde el external id.
- Si el mismo id se alimentaba dos veces a `process_records`, el pipeline marcaba el segundo como *content duplicate* pero **conservaba ambas filas**.
- Eso no basta para un dataset de mercado: se necesita merge por identidad **antes** del pipeline.

## Decisión

1. Módulo `analysis/ingestion/market.py` con `run_market_batch`, `MarketQuery`, `MarketBatchResult`.
2. Preset configurable de queries (default: 6 familias del dominio).
3. Un solo `IngestionContext` por corrida.
4. Merge por internal id:
   - Una fila por id.
   - `matched_queries_by_id` registra todas las queries que lo vieron.
5. Conflictos (mismo id, contenido distinto):
   - Mayor completeness (campos no vacíos).
   - Empate → descripción más larga.
   - Empate → query lexicográficamente menor.
   - Independiente del orden de ejecución.
6. Fail-fast por defecto si una query falla.
7. Artefacto consolidado `tekmerion.market_batch.v1` en `data/processed/market/` (no es raw).
8. Snapshots raw por query siguen siendo opcionales y separados.
9. CLI `scripts/fetch_market.py`.

## Consecuencias

- Dataset de mercado determinista y auditable.
- Contrato claro para que Flask cargue un artifact sin red (bloque futuro).
- No hay paginación multi-página ni scheduler.

## Alternativas

- Dejar dedup solo al pipeline → deja filas duplicadas por id.
- Preferir “primera query ejecutada” → no determinista si cambia el orden.
- Partial batch ante errores → oculta fallos; se deja como opción no default.
