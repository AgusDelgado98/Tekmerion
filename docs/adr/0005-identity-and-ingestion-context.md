# ADR 0005 — Identidad global y contexto de ingestión (V0.4.2)

## Estado

Aceptada

## Contexto

En V0.4 la capa de ingestión pasaba el `id` del registro tal cual venía de la fuente. La muestra real usaba prefijos manuales (`real_001`). Eso no escala: dos fuentes pueden reutilizar el mismo identificador externo, y un registro sin id quedaba vacío o arbitrario.

Además, `normalize_to_internal` rellenaba `retrieved_at` faltante con `datetime.now()`, rompiendo el determinismo de una corrida.

La deduplicación del pipeline era solo por contenido (title + company + description), sin considerar la fuente: dos vacantes idénticas publicadas en portales distintos se marcaban como duplicado.

## Decisión

### Identidad interna

```
internal_id = "{source}:{external_id}"
```

- `external_id` = id original aportado por la fuente (`id` / `source_record_id` / `external_id`).
- Si no hay external id → fallback determinista:

  ```
  auto:{sha256(source|company|title|location|source_url)[:12]}
  ```

  Nunca UUID aleatorio.

- El external id original se conserva en `source_record_id` (None si fue inventado).
- Los registros sintéticos cargados vía `process_file` **no** pasan por esta capa y mantienen sus ids legacy (`job_001`, …).

### Contexto de ingestión

```python
IngestionContext(retrieved_at: str)
```

- Una corrida completa recibe un único contexto.
- `retrieved_at` del registro gana; si falta, se usa el del contexto.
- Si no hay ni registro ni contexto → `retrieved_at = None` (no se inventa reloj).

### Deduplicación

El fingerprint de contenido incluye `source`:

| Caso | Comportamiento |
|------|----------------|
| A. Mismo contenido, misma fuente | Duplicado (como antes) |
| B. Mismo external id, fuentes distintas | IDs internos distintos (`src_a:X` vs `src_b:X`); no colisionan |
| C. Mismo contenido, fuentes distintas | Registros independientes (deuda: cross-source dedup semántica) |

## Consecuencias

- Mezclar fuentes es seguro a nivel de identidad.
- Tests y pipelines son reproducibles con el mismo contexto.
- Cross-source “es la misma vacante?” queda como deuda explícita (no fuzzy matching en este bloque).
- Adaptadores futuros (Adzuna, etc.) solo necesitan emitir un external id; la capa de ingestión namespaced.

## Alternativas consideradas

- Prefijos manuales por archivo (`real_`, `adz_`) → frágil, no composable.
- UUID aleatorio para fallback → no determinista, imposible de re-ejecutar.
- Deduplicación semántica cross-source ya → fuera de alcance de V0.4.2.
