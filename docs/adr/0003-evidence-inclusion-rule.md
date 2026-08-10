# ADR 0003 — Regla de inclusión para la capa de evidencia

## Estado

Aceptada

## Contexto

El pipeline produce registros con flags `is_valid` e `is_duplicate`.  
La capa de evidencia necesita una regla clara y única sobre qué registros participan en todas las métricas (frecuencias, co-ocurrencias, distribuciones, comparaciones).

Contar duplicados inflaría artificialmente las frecuencias.  
Incluir inválidos introduciría ruido de registros que no pasaron validación básica.

## Decisión

Solo se incluyen en el análisis de evidencia los registros que cumplen:

```text
is_valid is True  AND  is_duplicate is False
```

Esta regla es:

- única para todas las métricas de la capa;
- explícita en código (`analysis_records`);
- documentada en `docs/methodology.md`.

`RoleFamily.UNKNOWN` y `Seniority.UNKNOWN` **no** se excluyen ni se reasignan.

## Consecuencias

- Las métricas son comparables entre sí (mismo denominador).
- El número de “analysis records” puede ser menor que `valid_count` del pipeline cuando existen duplicados.
- Cualquier cambio futuro de esta regla requiere actualizar código, tests y esta ADR.

## Alternativas consideradas

- Contar todos los válidos (incluyendo duplicados) → sesgo por postings repetidos.
- Excluir también UNKNOWN → pierde visibilidad de la calidad de clasificación.
- Reglas distintas por métrica → inconsistencia y deuda cognitiva.
