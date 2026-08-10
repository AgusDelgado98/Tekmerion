# ADR 0013 — Grounded role comparison (V0.5.4)

## Estado

Aceptada

## Contexto

Tekmérion ya genera `market_summary`. Hacía falta comparar dos role families
sin abrir un chatbot ni inventar métricas.

## Decisión

1. Tarea `role_comparison` con prompt `role_comparison.v1`.
2. RoleComparisonGrounding derivado de EvidenceReport:
   counts/pct por rol, skills por rol, shared/exclusive deterministas,
   shared_detail (more_frequent_in_a|b|equal), seniority opcional, sample threshold=5.
3. Validación de roles antes del provider (ids canónicos, distintos, presentes).
4. Output: summary + structured skill lists + differences + limitations.
5. Guardrails: numeric/ranking existentes + scope (sin refs de tercer rol) +
   igualdad de listas shared/only vs grounding.
6. Cache: `role_comparison|canonical_pair|fingerprint|provider|model`.
7. UI: `/analysis/roles` (selects, POST only).

La IA redacta; no calcula shared/exclusive ni recomienda carreras.
