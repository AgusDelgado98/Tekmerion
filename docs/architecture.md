# Architecture — Tekmérion

## Layers

```text
Acquisition          Adzuna API | curated fixtures | synthetic sample | showroom
        ↓
Ingestion            Source adapters · IngestionContext · namespaced IDs
        ↓
Processing           Normalize · classify role/seniority · extract skills · dedupe
        ↓
Market batch         Multi-query merge · tekmerion.market_batch.v1 artifact
        ↓
Evidence             EvidenceReport (deterministic metrics)
        ↓
Generative           GroundingPayload → Provider → Guardrails
        ↓
Presentation         Flask · DatasetRegistry · session dataset switch

ML eval (offline)    analysis/ml · Gold Dataset · split · features · metrics
                     (parallel to Evidence; not in the production path)
                     Rules vs LogReg / Linear SVM / RF · promote_ml=false
```

## Internet boundaries

| Layer | May use network? |
|-------|------------------|
| Acquisition (CLI live) | Yes, optional |
| Ingestion offline / fixtures | No |
| Pipeline / evidence | No |
| ML evaluation (`analysis/ml`) | No |
| Generative live provider | Yes, optional |
| Flask UI | **No** external calls |

## Components

- `analysis/ingestion/` — adapters (Adzuna, local), market batch
- `analysis/pipeline.py` · `classifiers.py` · `skills.py`
- `analysis/evidence.py` — EvidenceReport
- `analysis/ml/` — Gold Dataset, harvest unlabeled, split, features, evaluator, sklearn training behind a sufficiency gate (not in Evidence)
- `data/ml/` — versioned gold + evaluation reports (unlabeled live harvest gitignored)
- `analysis/grounding.py` · `analysis/generative/` — grounded AI
- `app/dataset.py` · `app/registry.py` — datasets for UI
- `data/showroom/` — offline portfolio demo artifact
- `data/processed/market/` — processed market snapshots

## Dataset kinds

- **synthetic** — development sample
- **showroom** — fixed offline market demo (`dataset_kind=showroom`)
- **market** — local processed artifact (may originate from a past live fetch)

## Design rules

1. Metrics are computed deterministically before any LLM.
2. ML evaluation uses human `gold_role_family`; regex output is never training gold.
3. The model may only interpret grounding derived from evidence.
4. Guardrails reject unsupported numeric/ranking claims and out-of-scope refs.
5. Flask never fetches Adzuna; it only loads local datasets.
6. sklearn models are not promoted into Evidence unless test macro F1 beats rules by a documented margin.
