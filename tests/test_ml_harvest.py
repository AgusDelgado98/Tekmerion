"""Harvest unlabeled candidates; never treat classifier output as gold."""

from __future__ import annotations

from analysis.ml.harvest import harvest_unlabeled_candidates
from analysis.ml.models import FORBIDDEN_GOLD_KEYS


def test_harvest_dedupes_and_strips_pipeline_labels():
    payload = harvest_unlabeled_candidates()
    assert payload["schema"] == "tekmerion.ml.gold_candidates.v1"
    assert payload["label_status"] == "unlabeled"
    assert payload["n_unique"] >= 1
    assert payload["n_unique"] <= payload["n_loaded"]
    ids = [r["id"] for r in payload["records"]]
    assert len(ids) == len(set(ids))
    fps = [r["content_fingerprint"] for r in payload["records"]]
    assert len(fps) == len(set(fps))
    for row in payload["records"]:
        assert row["label_status"] == "unlabeled"
        assert "gold_role_family" not in row
        assert FORBIDDEN_GOLD_KEYS.isdisjoint(row.keys())
        assert row["title"]
        assert row["description"]
        assert row["source_kind"]
        assert row["source_ref"]


def test_harvest_includes_curated_real_provenance():
    payload = harvest_unlabeled_candidates()
    real = [r for r in payload["records"] if r["source_kind"] == "curated_real_sample"]
    assert len(real) == 4
    assert all(r.get("source_url") for r in real)
    assert all(r.get("retrieved_at") for r in real)
