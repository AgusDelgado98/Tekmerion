# ADR 0014 — Showroom portfolio demo (V0.6.0)

## Estado

Aceptada

## Contexto

Hacía falta una demo clonable sin Adzuna key ni LLM key, distinguible de synthetic y de snapshots live.

## Decisión

1. Artifact `data/showroom/showroom_market_ar.json` (`tekmerion.market_batch.v1` + `dataset_kind=showroom`).
2. Origen: fixtures Adzuna offline del repo (no etiquetado como live).
3. Registry: id fijo `showroom`, label “Showroom · Market demo”.
4. Default de tests/desarrollo sigue siendo synthetic; demo documentada con `TEKMERION_DATA_MODE=showroom`.
5. UI: charts CSS desde EvidenceReport; disclosure de AI; recorrido en home.

## Consecuencias

Cualquiera puede clonar y recorrer jobs → evidence → AI → role comparison sin secretos.
