# Gold Dataset (public contract)

The labeled vacancy corpus used for the V0.8 benchmark (`n=159`) is **not redistributed**.
It remains on the author’s machine under gitignored paths so local work is not lost.

What this repository publishes:

| Artifact | Role |
|----------|------|
| `tekmerion.ml.gold_dataset.v1` in `analysis/ml/models.py` | Schema / loader contract |
| `tests/fixtures/ml/gold_*.json` | Synthetic fixtures for tests |
| `evaluation_card.json` | n=159, class counts, 112/47 split, dataset hash, rounded metrics |
| `../reports/block_b.json` | Full sanitized comparison (CV, hyperparameters, confusion matrices) |
| `../artifacts/evaluation_manifest*.json` | Sanitized manifests (anonymous `ex_*` ids) |
| `docs/assets/05-rules-vs-ml.png` | Rules vs ML chart |

Vacancy data used in the ML evaluation was sourced from [The Adzuna API](https://developer.adzuna.com/).

Local (gitignored) files, if present:

- `role_family_v1.json` — working gold used to produce the published hash
- `local/role_family_v1.json` — backup copy of the same file
