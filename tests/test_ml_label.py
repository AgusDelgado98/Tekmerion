"""Human labeling queue and apply — no classifier involvement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.ml.gold import dump_gold_dataset, load_gold_dataset
from analysis.ml.harvest import harvest_unlabeled_candidates
from analysis.ml.label import (
    LabelError,
    apply_human_labels,
    format_label_card,
    unlabeled_queue,
)
from analysis.ml.models import GOLD_SCHEMA, LABEL_SOURCE_HUMAN, RoleFamily

FIXTURES = Path("tests/fixtures/ml")
SMALL = FIXTURES / "gold_small.json"


def test_queue_skips_already_labeled_and_prefers_sparse_hints():
    ds = load_gold_dataset(SMALL)
    harvest = harvest_unlabeled_candidates()
    real_q = unlabeled_queue(harvest["records"], ds, real_only=True)
    labeled_ids = {r.id for r in ds.records}
    assert all(row["id"] not in labeled_ids for row in real_q)
    assert all("gold_role_family" not in row for row in real_q)
    if any(r.get("source_kind") == "adzuna_snapshot" for r in harvest["records"]):
        assert real_q
        assert all(row["source_kind"] in {"curated_real_sample", "adzuna_snapshot"} for row in real_q)
    full_q = unlabeled_queue(harvest["records"], ds, real_only=False)
    assert full_q
    assert all(row["id"] not in labeled_ids for row in full_q)


def test_card_shows_title_skills_and_distribution():
    ds = load_gold_dataset(SMALL)
    cand = {
        "id": "demo-1",
        "title": "Analista funcional",
        "description": "Relevamiento de procesos, documentación de requisitos y SQL básico.",
        "company": "Test",
        "location": "BA",
        "source_url": "https://example.com/x",
        "retrieved_at": "2026-08-19T00:00:00Z",
        "source_kind": "curated_real_sample",
    }
    dist = {"business_analyst": 1, "ai_analyst": 1}
    card = format_label_card(cand, distribution=dist, index=1, total=1)
    assert "Analista funcional" in card
    assert "Relevamiento" in card
    assert "sql" in card.lower() or "skills:" in card
    assert "business_analyst=1" in card
    assert "query_ctx:" in card
    assert "not a label" in card
    assert "skip/ambiguous" in card
    assert "classify" not in card.lower()


def test_apply_human_labels_rejects_classifier_fields(tmp_path):
    ds = load_gold_dataset(SMALL)
    cand = {
        "id": "new-ba-1",
        "title": "Business Analyst",
        "description": "Requirements and process mapping with workshops.",
        "company": "Co",
        "source_kind": "curated_real_sample",
        "source_ref": "test",
        "source_url": "https://example.com/ba",
        "retrieved_at": "2026-08-19T00:00:00Z",
        "source_record_id": "ba-1",
    }
    with pytest.raises(LabelError, match="classifier"):
        apply_human_labels(
            ds,
            [cand],
            [{"id": "new-ba-1", "gold_role_family": "business_analyst", "role_family": "business_analyst"}],
        )


def test_apply_human_labels_merges(tmp_path):
    src = json.loads(Path(FIXTURES / "gold_small.json").read_text(encoding="utf-8"))
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps(src), encoding="utf-8")
    ds = load_gold_dataset(gold_path)
    cand = {
        "id": "new-ai-1",
        "title": "AI Analyst",
        "description": "Evaluate LLM quality, prompts and generative product metrics.",
        "company": "Co",
        "source_kind": "curated_real_sample",
        "source_ref": "test",
        "source_url": "https://example.com/ai",
        "retrieved_at": "2026-08-19T00:00:00Z",
        "source_record_id": "ai-1",
    }
    labels = [
        {
            "id": "new-ai-1",
            "gold_role_family": RoleFamily.AI_ANALYST.value,
            "annotator_id": "test.human",
            "labeled_at": "2026-08-19",
            "label_source": LABEL_SOURCE_HUMAN,
        }
    ]
    merged, stats = apply_human_labels(ds, [cand], labels)
    assert stats["n_added"] == 1
    assert merged.n == ds.n + 1
    assert any(r.id == "new-ai-1" for r in merged.records)
    out = tmp_path / "out.json"
    dump_gold_dataset(merged, out)
    reloaded = load_gold_dataset(out)
    assert reloaded.schema == GOLD_SCHEMA
    ai = next(r for r in reloaded.records if r.id == "new-ai-1")
    assert ai.label_source == LABEL_SOURCE_HUMAN
    assert ai.gold_role_family == RoleFamily.AI_ANALYST
    assert "role_family" not in json.loads(out.read_text(encoding="utf-8"))["records"][-1]


def test_session_skip_keeps_gold_unchanged_and_drops_from_queue(tmp_path):
    from analysis.ml.label import (
        record_session_decision,
        session_reviewed_ids,
        unlabeled_queue,
    )

    src = json.loads(Path(FIXTURES / "gold_small.json").read_text(encoding="utf-8"))
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps(src), encoding="utf-8")
    ds = load_gold_dataset(gold_path)
    cand = {
        "id": "skip-1",
        "title": "Head of Data and AI",
        "description": "Lead a mixed org spanning product, engineering and analytics.",
        "company": "Co",
        "source_kind": "adzuna_snapshot",
        "source_ref": "adzuna_gb_p1_ai_analyst_20260819T180000.json",
        "source_url": "https://example.com/x",
        "retrieved_at": "2026-08-19T00:00:00Z",
        "source_record_id": "s1",
    }
    session = {"schema": "tekmerion.ml.label_session.v1", "annotator_id": "t", "decisions": []}
    record_session_decision(
        session,
        candidate_id="skip-1",
        decision="skip",
        labeled_at="2026-08-19T00:00:00Z",
    )
    q = unlabeled_queue([cand], ds, real_only=True, extra_skip_ids=session_reviewed_ids(session))
    assert q == []
    merged, stats = apply_human_labels(ds, [cand], [])
    assert stats["n_added"] == 0
    assert merged.n == ds.n


def test_session_skip_and_rejects_unknown_family():
    from analysis.ml.label import record_session_decision

    session = {"decisions": []}
    record_session_decision(
        session,
        candidate_id="ok",
        decision="skip/ambiguous",
        labeled_at="2026-08-19",
    )
    assert session["decisions"][0]["decision"] == "skip"
    with pytest.raises(LabelError, match="unsupported|must be"):
        record_session_decision(
            session,
            candidate_id="bad",
            decision="software_engineer",
            labeled_at="2026-08-19",
        )


def test_label_module_source_has_no_classifier_call():
    text = Path("analysis/ml/label.py").read_text(encoding="utf-8")
    assert "classify_role_family" not in text
    assert "analysis.classifiers" not in text
    fetch = Path("analysis/ml/fetch_candidates.py").read_text(encoding="utf-8")
    assert "classify_role_family" not in fetch
    assert "process_records" not in fetch
