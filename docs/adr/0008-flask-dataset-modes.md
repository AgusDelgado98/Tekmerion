# ADR 0008 — Flask dataset modes (synthetic | market)

## Estado

Aceptada

## Contexto

Hasta V0.4.4 Flask llamaba siempre a `process_file(data/raw/sample_jobs.json)`.
El market artifact (`tekmerion.market_batch.v1`) ya materializa records procesados.
Hacía falta consumirlos en la UI sin red y sin re-pipeline.

## Decisión

1. Capa `app/dataset.py` con `load_app_dataset`.
2. Modos vía env (no query params HTTP):
   - `TEKMERION_DATA_MODE=synthetic|market`
   - `TEKMERION_MARKET_FILE` opcional
3. Market + path explícito inválido → error (sin fallback silencioso).
4. Market sin path → discovery por `retrieved_at` interno entre JSON válidos.
5. Hydrate `ProcessedJob` desde `records`; **no** `process_records` de nuevo.
6. Evidence UI = `build_evidence(hydrated_records)` (canónico y completo para vistas).
7. Metadata de dataset en templates; badge discreto en header.
8. Rutas mínimas `/jobs` y `/jobs/<path:id>` para listado/detalle y `source_url`.

## Consecuencias

- Misma UI sirve evidencia sintética o de mercado.
- Runtime web desacoplado de Adzuna.
- No hay selector web todavía (configuración de servidor únicamente).
