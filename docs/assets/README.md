# Documentation assets

Real application screenshots (Chrome, viewport ≈ 1440×900) plus a static ML chart.

Showroom captures use `TEKMERION_LLM_PROVIDER=fake`.

| File | Page / source | Notes |
|------|------|--------|
| `01-home-showroom.png` | `/` | Showroom badge, metrics, role/seniority bars |
| `02-evidence.png` | `/skills` | Top skills table (deterministic evidence) |
| `03-ai-analysis.png` | `/analysis` after POST | **Demo provider** badge + grounded findings |
| `04-role-comparison.png` | `/analysis/roles` after POST | DA vs BI, shared/exclusive skills |
| `05-rules-vs-ml.png` | `data/ml/reports/block_b.json` | Test **macro F1**: Rules vs sklearn |
| `05-job-detail.png` | `/jobs/<id>` | Provenance: source, retrieved_at, source URL |

`05-rules-vs-ml.png` is generated from the Block B report (not from Cursor canvases). The AI screenshots intentionally show the fake/demo provider label.
