# ML data — Tekmérion V0.8

Offline evaluation artifacts. **Not** wired into Evidence or Flask.

The real Gold Dataset (titles, employers, Adzuna snippets/URLs) is **not** in git.
Public clones get the contract, fixtures, sanitized reports, and the published metrics.

| Path | Public? | Role |
|------|---------|------|
| `gold/evaluation_card.json` | Yes | n, class mix, hash, split, rounded metrics |
| `gold/README.md` | Yes | What is and is not redistributed |
| `reports/block_b.json` | Yes | Sanitized Rules vs ML comparison |
| `reports/gold_expansion.json` | Yes | Gate / harvest counts (no vacancy text) |
| `artifacts/` | Yes | Sanitized evaluation manifests |
| `gold/role_family_v1.json` | No (gitignored) | Local labeled corpus |
| `gold/local/` | No (gitignored) | Local backup of that corpus |
| `gold/candidates_unlabeled_v1.json` | No (gitignored) | Live unlabeled harvest |
| `gold/label_session.json` | No (gitignored) | Local labeling session |

## Reproduce the public experiment (no gold file, no Adzuna key)

```powershell
pip install -e ".[dev]"
pytest -q
```

Inspect `data/ml/gold/evaluation_card.json`, `data/ml/reports/block_b.json`, and `docs/assets/05-rules-vs-ml.png`.

Retraining sklearn on the original 159 examples requires the local gitignored gold and is not part of Quick Start.

`gold_role_family` is human-only. Do not copy `classify_role_family`.

## Sufficiency gate

`n >= 100` and `>= 10` per present class. Promotion uses test **macro F1**, not accuracy.
