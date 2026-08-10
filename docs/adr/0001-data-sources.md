# ADR 0001 — Fuentes de datos iniciales

## Estado

Aceptada

## Contexto

Tekmérion necesita datos de vacantes laborales del dominio Data / BI / AI para:

- validar el pipeline de procesamiento
- producir evidencia sobre demanda de skills y roles
- servir como base para una futura capa de interpretación

Se requieren tanto datos históricos/reproducibles como la posibilidad de acercarse al mercado actual.

## Decisión

Se adoptan dos fuentes principales:

1. **Kaggle**  
   Datasets públicos de vacantes (históricos o relativamente grandes).  
   Ventaja: reproducibilidad, tamaño, sin costos de API, sin rate limits.

2. **Adzuna API**  
   Vacantes reales y actuales.  
   Ventaja: cercanía al mercado laboral real en el momento de la consulta.

## Consecuencias

- El pipeline debe poder ingerir ambos orígenes (normalizando a un schema común).
- En V1 se trabaja primero con una muestra sintética controlada para validar el diseño.
- No se introduce scraping genérico ni otras APIs sin una nueva ADR que lo justifique.
- Las credenciales de Adzuna nunca deben commitearse; se usará variables de entorno.

## Alternativas consideradas

- Solo scraping de portales → más frágil, posibles problemas legales/ToS, menos reproducible.
- Solo una API de pagos → dependencia temprana y costos innecesarios para la etapa de validación.
