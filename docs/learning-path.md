# Learning path — Tekmérion

Tekmérion is an **applied** Data/AI project: each version adds a capability the
previous contract made possible, instead of a pile of unrelated notebooks.

The protagonist is the **system** (evidence, guardrails, evaluation) — not a
course certificate.

## How the project grew

| Version | What landed | Why it matters |
|---------|-------------|----------------|
| ≤0.4 | Ingestion, identity, market batch, Flask | Reproducible data path |
| 0.5 | Grounded generation + numeric/ranking guardrails | The model cannot invent the market |
| 0.6–0.7 | Showroom demo, MIT packaging | Clonable portfolio without secrets |
| **0.8** | Human gold, anti-leakage split, Rules vs sklearn | ML is evaluated, not assumed |

## V0.8 — ML concepts in production-shaped code

| Concept | Where it shows up |
|---------|-------------------|
| Supervised classification | `gold_role_family` on job title + description |
| Human labels vs rules | Regex/`classify_role_family` is a **predictor**, never gold |
| Leakage control | Grouped split by content fingerprint; TF-IDF fit on train/CV fold only |
| Class imbalance | Macro F1 over accuracy; `class_weight=balanced` |
| Cross-validation | 3-fold GridSearch on **train**, scoring `f1_macro` |
| Hyperparameter search | Bounded C / tree depth grids in `analysis/ml/train.py` |
| Baseline comparison | Same test set for rules, LogReg, Linear SVM, Random Forest |
| Sufficiency gate | No `fit` until n≥100 and ≥10 examples per present class |
| Promotion rule | `promote_ml=true` only if test macro F1 lifts by ≥0.02 without collapsing classes |

On this gold set, **rules won**. That is a valid ML outcome: the experiment is the
comparison, not a requirement to replace the baseline.

## What to run

```powershell
pip install -e ".[dev]"
pytest -q
```

Published comparison: `data/ml/reports/block_b.json` and `docs/assets/05-rules-vs-ml.png`.
The real Gold Dataset stays local (gitignored). Retrain scripts need that file.

Evidence, the job pipeline, and Flask still do not import sklearn models.
