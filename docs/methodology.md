# Metodología — Tekmérion V0.8

## Filosofía

Tekmérion no es un dashboard de empleos ni un generador de texto genérico.

El flujo fundamental es:

```
Datos crudos
  → validación y normalización
  → clasificación determinista
  → extracción de skills estructuradas
  → detección de duplicados
  → evidencia calculada
  → interpretación grounded (opcional) con guardrails
  → evaluación ML offline (opcional; no está en Evidence)
```

La fuente primaria de verdad son los datos transformados de forma reproducible, no el lenguaje natural de un modelo.

## Pipeline actual

Ubicación: `analysis/pipeline.py`

### Entrada

Lista de diccionarios (JSON) con al menos:

- `id`
- `title`
- `company`
- `description`

Campos opcionales: `location`, `salary_min`, `salary_max`, `currency`, `posted_date`, `source`.

### Etapas

1. **Validación**  
   Un registro es válido si tiene `id`, `title`, `company` y `description` no vacíos.

2. **Normalización de título**  
   Espacios colapsados + Title Case. Determinista.

3. **Clasificación de Role Family**  
   Reglas explícitas basadas en patrones sobre título + descripción.  
   Familias soportadas:
   - `data_analyst`
   - `bi_analyst`
   - `data_scientist`
   - `ml_engineer`
   - `ai_analyst`
   - `data_engineer`
   - `business_analyst`
   - `unknown`

4. **Clasificación de Seniority**  
   `junior` | `mid` | `senior` | `lead` | `unknown`  
   También basada en reglas explícitas y auditables.

5. **Extracción de skills**  
   Matching de aliases → nombre canónico.  
   Resultado: lista ordenada (tupla) de skills normalizadas.  
   Esto permite análisis de frecuencia, co-ocurrencia y gaps.

6. **Detección de duplicados**  
   Fingerprint basado en título normalizado + company + fragmento de descripción.  
   Determinista. El primer registro visto se considera original.

### Salida

Cada registro de entrada produce exactamente un `ProcessedJob` (dataclass inmutable).

El resultado agregado es un `PipelineResult` con:

- lista de registros procesados
- conteos de válidos / inválidos / duplicados
- distribución por role family y seniority (sobre válidos)

### Garantías

- **No mutación** del input.
- **Determinismo**: mismo input → mismo output.
- **Trazabilidad**: cada decisión de clasificación está en código explícito.
- **Testabilidad**: las reglas se pueden verificar unitariamente.

## Datos de desarrollo

Se utilizan dos conjuntos controlados:

1. **Muestra sintética** (`data/raw/sample_jobs.json`) — 17 registros.  
   Diseñada para ejercitar validación, clasificación, skills, duplicados e invalidos.
2. **Muestra real controlada** (`data/raw/real/sample_real_jobs.json`) — 4 registros.  
   Representa el camino de ingestión de datos reales con procedencia explícita.  
   No es un scrape ni un feed en vivo; es material de arquitectura y tests end-to-end.

Ninguna de las dos pretente ser estadísticamente representativa del mercado.

## Capa de evidencia (V0.2)

Ubicación: `analysis/evidence.py`

La evidencia se calcula **exclusivamente** sobre registros ya procesados por el pipeline.
Esta capa no reclasifica ni re-extrae skills; solo agrega.

### Qué consideramos “evidencia”

Métricas estructuradas, deterministas y auditables que responden preguntas del tipo:

- ¿Qué skills aparecen con mayor frecuencia?
- ¿Cómo se distribuyen los roles y los niveles de seniority?
- ¿Qué skills caracterizan a cada role family?
- ¿Qué pares de skills co-ocurren?
- ¿Qué diferencia a Data Analyst de BI Analyst (u otras familias)?

### Regla de inclusión (única y consistente)

Solo entran en el análisis los registros que cumplen:

```text
is_valid is True  AND  is_duplicate is False
```

- Los inválidos se excluyen porque no pasaron validación básica.
- Los duplicados se excluyen para no inflar frecuencias contando la misma vacante varias veces.
- `RoleFamily.UNKNOWN` y `Seniority.UNKNOWN` **sí se incluyen** y aparecen explícitamente en las métricas. No se reasignan.

### Métricas disponibles

| Función                  | Descripción                                      |
|--------------------------|--------------------------------------------------|
| `skill_frequency`        | Frecuencia global de skills (+ proporción)       |
| `skills_by_role`         | Skills más frecuentes por role family            |
| `skills_by_seniority`    | Skills más frecuentes por seniority              |
| `role_distribution`      | Cantidad y proporción por role family            |
| `seniority_distribution` | Cantidad y proporción por seniority              |
| `skill_cooccurrence`     | Pares de skills que aparecen juntos (no dirigidos)|
| `compare_roles`          | Comparación entre dos familias (comunes / exclusivas) |
| `build_evidence`         | Empaqueta todo en un `EvidenceReport`            |

### Cómo se calculan las frecuencias

- Dentro de una vacante, cada skill se cuenta como máximo una vez (set).
- La proporción de una skill es:  
  `count(skill) / n_analysis_records`  
  (no sobre el total de menciones).
- Ordenamiento determinista: frecuencia descendente → nombre alfabético ascendente.

### Cómo se calcula la co-ocurrencia

- Se generan pares no ordenados `(skill_a, skill_b)` con `skill_a < skill_b` lexicográficamente.
- Un par se incrementa una vez por cada vacante que contiene ambas skills.
- No existen pares invertidos (`python+sql` y `sql+python` son el mismo).
- Ordenamiento: count desc → skill_a asc → skill_b asc.

### Limitación actual de los datos

Todas las métricas se calculan por ahora sobre la **muestra sintética de 17 registros**.

Esta muestra sirve para validar el motor analítico.  
No se usa como Gold de ML (el gold vive en `data/ml/gold/`).  
**No** representa el mercado laboral real ni permite conclusiones estadísticas externas.

### Interfaz web (V0.3)

Ubicación: `app/`

Flask solo presenta la evidencia ya calculada.  
No contiene lógica analítica propia.

Al iniciar la aplicación se ejecuta una sola vez:

```text
sample_jobs.json → pipeline → EvidenceReport → memoria de la app
```

Las rutas leen de `app.config["EVIDENCE"]` y `app.config["PIPELINE_RESULT"]`.

Páginas:

- `/` — overview (conteos, distribuciones, top skills)
- `/skills` — frecuencia global o filtrada por role / seniority
- `/roles` y `/roles/<family>` — detalle por role family
- `/compare` — comparación de dos familias vía `compare_roles`
- `/cooccurrence` — tabla de pares

### Ingestión de datos reales (V0.4)

Ubicación: `analysis/ingestion/`

Se separan claramente tres estados de los datos:

| Estado        | Ubicación / producto                         | Qué contiene                                      |
|---------------|----------------------------------------------|---------------------------------------------------|
| **raw**       | `data/raw/` (synthetic o `real/`)            | Datos tal como se recibieron / curaron             |
| **ingested**  | Salida de `ingest()` / `normalize_to_internal` | Dicts con schema interno + procedencia            |
| **processed** | `ProcessedJob` / `data/processed/`           | Registros enriquecidos (rol, seniority, skills…)  |

**Política de procedencia**

Todo registro que entra por la capa de ingestión debe poder responder:

- `source` — identificador estable de la fuente (`curated_real_sample`, futuro `adzuna`, etc.)
- `source_url` — URL original si existe
- `retrieved_at` — timestamp ISO de cuándo se obtuvo el registro
- `source_record_id` — id original de la fuente (antes del namespacing)

Los registros sintéticos legacy llevan `source="synthetic"` y los campos de URL/retrieved/`source_record_id` en `None`.

**Estrategia de identidad (V0.4.2)**

```
internal_id = "{source}:{external_id}"
```

- Si la fuente aporta id → se usa como `external_id` y se preserva en `source_record_id`.
- Si no aporta id → fallback determinista `auto:{hash12}` a partir de source + company + title + location + source_url.
- Nunca UUID aleatorio.
- Los sintéticos cargados con `process_file` no pasan por esta capa y conservan `job_00x`.

**Timestamps**

- `IngestionContext(retrieved_at=...)` aporta un timestamp estable por corrida.
- El `retrieved_at` del registro tiene prioridad; si falta, se usa el del contexto.
- Sin registro ni contexto → `None`. No se llama a `datetime.now()`.

**Deduplicación (límites actuales)**

- Fingerprint = source + title + company + fragmento de description.
- Mismo contenido + misma fuente → duplicado.
- Mismo contenido + fuentes distintas → registros independientes (deuda: cross-source dedup).
- Mismo external id + fuentes distintas → ids internos distintos, sin colisión.

**Limitaciones actuales**

- La primera fuente real es una **muestra controlada** de 4 vacantes (`data/raw/real/sample_real_jobs.json`), no un feed en vivo.
- No hay llamadas a APIs externas ni scraping.
- La UI Flask sigue cargando por defecto la muestra sintética.
- No hay deduplicación semántica entre fuentes.

**Cómo agregar una nueva fuente en el futuro**

1. Implementar un `SourceAdapter` (método `load()` + `source_name()`).
2. Asegurar que los dicts resultantes pasen por `normalize_to_internal` (o que el adaptador ya emita el schema interno).
3. Crear un `IngestionContext` con el `retrieved_at` de la corrida.
4. Registrar el adaptador: `ingest([adapter], context=ctx)`.
5. Agregar tests de carga, identidad, procedencia y compatibilidad con el pipeline.
6. Documentar la fuente en un ADR si introduce dependencias o restricciones nuevas.

### Fuentes de datos: tres categorías (no mezclar)

| Categoría | Ejemplo | Cómo entra | Uso |
|-----------|---------|------------|-----|
| **Synthetic** | `data/raw/sample_jobs.json` | `process_file` directo | Validar pipeline, tests, UI por defecto |
| **Curated real sample** | `data/raw/real/sample_real_jobs.json` | `LocalJsonSource` + ingest | Desarrollo de ingestión sin red |
| **Live external data** | Adzuna API → snapshot opcional en `data/raw/real/adzuna/` | `AdzunaSource` + ingest | Vacantes actuales con trazabilidad |

### Adaptador Adzuna (V0.4.3)

- Módulo: `analysis/ingestion/adzuna.py`
- Auth: `ADZUNA_APP_ID` + `ADZUNA_API_KEY` (ver `.env.example`)
- Endpoint: `GET https://api.adzuna.com/v1/api/jobs/{country}/search/{page}`
- La API pública devuelve **snippet** de descripción, no el texto completo
- Tests: fixture offline en `tests/fixtures/adzuna/` (pytest no hace red)
- CLI live: `python scripts/fetch_adzuna.py --what "data analyst" --country ar --limit 10 --save-snapshot`
- Snapshots: JSON con `retrieved_at` + payload; no se versionan en git (solo `.gitkeep`)


### Market batch (V0.4.4)

Flujo:

```text
Adzuna query set
  → raw snapshots (opcional, por query)
  → map + normalize (un IngestionContext)
  → identity merge (un registro por adzuna:id)
  → process_records (una sola pasada)
  → evidence
  → market artifact (data/processed/market/)
```

- Preset de queries configurable (`DEFAULT_MARKET_QUERIES`).
- Merge determinista; conflictos por completeness + descripción + nombre de query.
- El **market artifact** (`schema: tekmerion.market_batch.v1`) es el contrato futuro de entrada para Flask.
- **Raw** = respuesta por query. **Consolidated** = dataset único post-merge + pipeline. No intercambiar los nombres.

CLI:

```bash
python scripts/fetch_market.py --country ar --limit-per-query 5 --save-raw --save-market
```


### Role comparison (V0.5.4)

```text
EvidenceReport
  → RoleComparisonGrounding(A,B)  # shared/exclusive deterministas
  → role_comparison.v1
  → provider
  → structured lists + differences
  → numeric/ranking/scope validation
  → /analysis/roles
```

La IA redacta; no calcula shared/exclusive ni recomienda carreras.

### Applied ML (V0.8)

Capa offline `analysis/ml/`. **No** modifica Evidence ni el pipeline productivo.

Bloque A: contrato de evaluación (gold humano, split anti-leakage, métricas, baseline).
Bloque B: harvest + gate; sklearn **solo** si n≥100 y ≥10 por clase; comparación Rules vs LogReg / Linear SVM / RF.

```text
Gold Dataset (human gold_role_family)
  → grouped train/test split (seed fija, fingerprint de título+descripción)
  → TF-IDF + skills fit en train / fold de CV
  → predictor.predict(test)    # rules o sklearn
  → métricas comunes + evaluation manifest
```

- Contrato público: `tekmerion.ml.gold_dataset.v1` (`analysis/ml/models.py`, fixtures en `tests/fixtures/ml/`).
- El corpus etiquetado real (n=159) **no se redistribuye**. Queda local/gitignored (`data/ml/gold/role_family_v1.json` y `data/ml/gold/local/`).
- Tarjeta pública: `data/ml/gold/evaluation_card.json` (n=159, ≥10/clase, split 112/47, seed 42, dataset hash).
- Reportes sanitizados: `data/ml/reports/block_b.json` y `data/ml/artifacts/evaluation_manifest*.json` (ids `ex_*`, sin títulos/empresas/snippets/URLs).
- `gold_role_family` con `label_source=human`. Prohibido copiar `ProcessedJob.role_family` o la regex.
- Métrica de promoción: **macro F1** en test. Accuracy se reporta pero no decide.
- Resultado: Rules 0.866 vs Linear SVM 0.816 → **`promote_ml=false`**.
- sklearn extra `ml`/`dev`. Flask/Evidence no cargan modelos.

Vacancy data used in the ML evaluation was sourced from [The Adzuna API](https://developer.adzuna.com/).

Código, metodología, fixtures y artifacts de evaluación son públicos. El corpus real completo no.

CLI (evaluación publicada; no requiere gold local ni credenciales):

```bash
pytest -q
```

Reentrenamiento sobre el gold local (solo si el archivo gitignored está presente):

```bash
python scripts/eval_ml_baseline.py
python scripts/run_ml_block_b.py --no-harvest-file
python scripts/label_gold.py --show-next
```

### Portfolio packaging (V0.7.0)

Demo scripts (`scripts/run_demo.ps1` / `.sh`), LICENSE MIT, case study, assets folder for screenshots.

### Showroom (V0.6.0)

`data/showroom/showroom_market_ar.json` es un artifact de demo offline (`dataset_kind=showroom`), compuesto desde fixtures Adzuna del repo. No es un snapshot live.

### Dataset demo switch (V0.5.3)

```text
DatasetRegistry (synthetic + market artifacts locales)
      ↓
session.active_dataset_id
      ↓
AppDataset (pipeline / evidence / meta)
      ↓
views · grounding · AI
```

El selector UI solo elige ids internos del registry. No acepta paths, URLs ni uploads. Flask nunca llama a Adzuna.

### Flask dataset modes (V0.4.5)

```text
                     ┌─ synthetic sample  (process once at startup)
Flask → dataset loader
                     └─ processed market artifact  (hydrate ProcessedJob, no re-pipeline)
```

- `TEKMERION_DATA_MODE=synthetic|market` (default synthetic)
- `TEKMERION_MARKET_FILE` opcional; si falta en modo market → discovery por `retrieved_at` en `data/processed/market/`
- Modo market con archivo inválido/ausente → **error claro** (no fallback silencioso a synthetic)
- Flask **nunca** llama a Adzuna
- Evidence en UI se reconstruye desde los records hidratados (`build_evidence`); el `evidence_summary` del artifact es informativo


### Grounded generative analysis (V0.5.0)

```text
deterministic analytics
        ↓
EvidenceReport
        ↓
GroundingPayload (+ evidence_ref ids)
        ↓
GenerativeProvider (disabled | fake | openai_compatible)
        ↓
GeneratedAnalysis
        ↓
deterministic validation
        ↓
UI /analysis (POST explícito)
```

**Qué hace la IA:** redacta un market summary a partir de métricas ya calculadas.  
**Qué no hace:** clasificar vacantes, recalcular metrics, consultar web, inventar tendencias o skills.

Providers: `TEKMERION_LLM_PROVIDER=disabled|fake|openai_compatible`.

Guardrails cuantitativos (V0.5.1): `NumericEvidenceIndex` + extractor conservador de `%`, `X de Y` y counts con sustantivo. Findings validan números contra sus evidence_refs; summary/limitations contra el índice global. Prompt activo: `market_summary.v3`.

Ranking guardrails (V0.5.2): posiciones técnicas `#N`/`puesto N`/`rank N`/ordinales 1–3 contra `skills|roles|seniority.ranking`. Superlativos (“más frecuente”) solo si hay líder único por count (empates rechazados).
  
Sin API key el resto de Tekmérion sigue intacto.

### Próximos pasos metodológicos

- Bloque C solo si un modelo gana a rules en test macro F1 con el mismo contrato
- Más adaptadores (Kaggle offline)

## Principio rector

Primero evidencia. Después lenguaje.
