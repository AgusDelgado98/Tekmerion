# ADR 0002 — Skills como dato estructurado

## Estado

Aceptada

## Contexto

Las vacantes contienen skills de forma libre en el texto (título + descripción).  
Si se dejan como texto libre, se dificulta enormemente:

- conteo de frecuencias
- co-ocurrencia
- análisis por role family / seniority
- detección de gaps
- clustering posterior
- grounding de explicaciones de IA

## Decisión

Las skills se extraen y normalizan a un conjunto canónico de identificadores (`skills_extracted`).

- Matching basado en aliases explícitos y auditables.
- Salida ordenada (determinista).
- Representadas como `tuple[str, ...]` en el modelo para inmutabilidad.

Ejemplos de canónicos: `python`, `sql`, `power_bi`, `airflow`, `scikit_learn`, `llm`, etc.

## Consecuencias

- El diccionario de aliases se convierte en un artefacto importante del sistema (debe mantenerse y documentarse).
- Falsos negativos son preferibles a falsos positivos ruidosos en esta etapa.
- Se habilita análisis cuantitativo real sobre skills.
- La capa de IA futura podrá hablar de “las 5 skills más frecuentes en data_engineer senior” con evidencia verificable.

## Alternativas consideradas

- Dejar skills como texto libre + embeddings posteriores → más flexible pero menos interpretable y menos “evidence-first”.
- NER / modelos de extracción → overhead y opacidad innecesarios para V1.
