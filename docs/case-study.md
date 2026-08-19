# Case study — Tekmérion

## Context

Data, BI and AI roles are posted across platforms with inconsistent titles and
thin descriptions. Comparing “what employers ask for” needs a pipeline that
separates **measured signal** from **model opinion**.

## Problem

How do you turn messy vacancies into **comparable evidence**, and only then
explain them, without letting a generative model invent the market?

## Approach

1. **Decoupled ingestion** (e.g. Adzuna) with provenance (`source`, `source_url`, `retrieved_at`).
2. **Deterministic pipeline**: normalize, role family, seniority, skills, dedupe.
3. **Market batch**: several queries → one `tekmerion.market_batch.v1` artifact.
4. **EvidenceReport**: counts and rankings in code, not in the LLM.
5. **Grounded AI**: the model sees a `GroundingPayload`; guardrails reject unsupported numbers and rankings.
6. **Offline ML eval (V0.8)**: human Gold Dataset, anti-leakage split, Rules vs LogReg / Linear SVM / Random Forest. sklearn is **not** in Evidence or Flask.
7. **Flask demo**: synthetic / showroom / local market; the UI does not call external APIs.

## Results (checkable in the repo)

- Offline **showroom** demo (~14 jobs, several role families).
- Two grounded tasks: `market_summary`, `role_comparison`.
- One real API adapter (Adzuna) plus offline fixtures.
- Human gold **n=159** (≥10 per class). Test: 112 train / 47 test.
- Rules **macro F1 0.866** vs best sklearn (Linear SVM) **0.816** → `promote_ml=false`.
- **312** automated tests (pipeline, ingestion, Flask, guardrails, ML).

Not promoting ML is the intended outcome of the evaluation contract: the baseline stayed better on the metric chosen for imbalance (macro F1).

## Reliability

- Deterministic IDs, merge, evidence, and grounding fingerprints.
- Provenance on real records; live raw Adzuna files stay gitignored.
- Explicit `fake` provider for demos (not confused with a live LLM).
- Web runtime without network for local datasets.

## Limitations

- Showroom ≠ the full labour market.
- The labeled Adzuna corpus is not redistributed; evaluation artifacts and fixtures are.
- Evaluation text was Adzuna search snippets (typically ≤500 characters), not full ads.
- A missing extracted skill is not proof the employer did not require it.
- Guardrails do not cover all qualitative prose or causality.
- No auth, production deploy, or crawl scheduler.

## What I learned

1. Separating metrics from narrative reduces hallucinations in a testable way.
2. Artifact contracts (`market_batch.v1`, gold schema) simplify UI and tests.
3. A deterministic demo provider is more honest than faking live calls.
4. Registry + session is enough for multi-dataset without a database.
5. Publishing a negative ML promotion result is stronger than shipping a weaker model.
