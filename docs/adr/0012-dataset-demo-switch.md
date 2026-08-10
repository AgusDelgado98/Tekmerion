# ADR 0012 — Dataset demo switch (V0.5.3)

## Estado

Aceptada

## Contexto

Hasta V0.5.2 el dataset se fijaba al arrancar Flask (`app.config`). Para demos de
portfolio hacía falta cambiar synthetic ↔ market sin reiniciar ni tocar red.

## Decisión

1. **DatasetRegistry**: synthetic + artifacts válidos de `data/processed/market/`.
2. **IDs internos** `synthetic` / `market:{country}:{ts}:{stem}` — nunca paths.
3. **Sesión Flask**: solo `active_dataset_id`.
4. **before_request**: resuelve `AppDataset` → `g.dataset`.
5. **POST /dataset**: cambia id; redirige a `/` (evita detalle huérfano).
6. **Análisis**: `ANALYSIS_BY_DATASET[dataset_id]` — no mezclar datasets.
7. Sin CSRF framework (demo local sin auth); documentado, no production-ready.
8. Rechazo de path/URL/`..` en el formulario.

## Consecuencias

- Demo interactiva multi-dataset.
- Registry global; selección por sesión (clientes independientes).
- Cache de datasets cargados en memoria del proceso.
