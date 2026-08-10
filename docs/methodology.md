# Metodología — Tekmérion V1

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
  → (futuro) interpretación con IA fundamentada
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

Se utiliza una muestra sintética de 17 registros (`data/raw/sample_jobs.json`).

Esta muestra **no** pretende ser estadísticamente representativa.  
Sirve para:

- validar schema
- ejercitar todos los caminos del pipeline
- desarrollar y mantener tests
- detectar edge cases
- asegurar reproducibilidad

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
**No** representa el mercado laboral real ni permite conclusiones estadísticas externas.

### Próximos pasos metodológicos

- Capa Flask para explorar la evidencia de forma interactiva
- Integración de fuentes reales (Kaggle / Adzuna) manteniendo el mismo contrato de evidencia
- Capa de IA generativa **solo** sobre evidencia ya calculada (grounded)

## Principio rector

Primero evidencia. Después lenguaje.
