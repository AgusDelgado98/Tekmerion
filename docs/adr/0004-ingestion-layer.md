# ADR 0004 — Capa de ingestión desacoplada (V0.4)

## Estado

Aceptada

## Contexto

Hasta V0.3 el pipeline recibía directamente una lista de diccionarios que ya cumplían (aproximadamente) el schema interno. La única fuente era la muestra sintética en `data/raw/sample_jobs.json`.

Para incorporar vacantes reales sin acoplar el pipeline a un proveedor concreto, ni romper la reproducibilidad, se necesita una capa intermedia que:

- Cargue datos desde orígenes distintos (archivos locales primero; APIs después).
- Preserve procedencia (`source`, `source_url`, `retrieved_at`).
- Normalice hacia el schema que ya consume `process_records`.
- Maneje registros incompletos o estructuralmente inválidos de forma explícita.
- Permita agregar nuevas fuentes mediante adaptadores sin modificar el núcleo analítico.

## Decisión

Se introduce el paquete `analysis/ingestion/` con:

1. **Contrato `SourceAdapter`** — interfaz mínima (`source_name()`, `load()`).
2. **`LocalJsonSource`** — primer adaptador concreto (archivos JSON offline).
3. **`normalize_to_internal`** — mapeo determinista al schema raw esperado por el pipeline.
4. **`ingest` / `ingest_local_file`** — orquestación que produce un `IngestionResult`.

Los datos reales controlados viven en `data/raw/real/`, separados de la muestra sintética.

La normalización **no** clasifica roles ni extrae skills; eso sigue siendo responsabilidad exclusiva del pipeline existente.

## Consecuencias

- El pipeline y la capa de evidencia no cambian su contrato público.
- Los registros sintéticos legacy siguen funcionando (campos de procedencia opcionales = `None`).
- Agregar una fuente futura (p. ej. Adzuna) implica implementar un nuevo `SourceAdapter` y, si hace falta, reglas de mapeo en `normalize_to_internal` o en el propio adaptador.
- No se introduce scraping ni dependencia de una API externa en este bloque.
- La UI Flask continúa cargando por defecto la muestra sintética; la ingestión de reales se usa desde scripts/tests/código de análisis.

## Alternativas consideradas

- Meter la carga de archivos reales dentro de `pipeline.process_file` → acopla orígenes al pipeline y dificulta multi-fuente.
- Un único formato de “mega-JSON” mezclado → pierde trazabilidad de procedencia y complica tests.
- Scraping directo → fuera de alcance, frágil y con riesgos legales/ToS.
