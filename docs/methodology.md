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

## Próximos pasos metodológicos (no implementados aún)

- Análisis exploratorio reproducible sobre los datos procesados
- Métricas de evidencia (frecuencias, co-ocurrencias, diferencias entre roles)
- Capa Flask para exploración
- Capa de IA generativa **solo** sobre evidencia ya calculada (grounded)

## Principio rector

Primero evidencia. Después lenguaje.
