"""Live fetch writes unlabeled snapshots only (mocked HTTP)."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.ingestion.adzuna import AdzunaClient, AdzunaCredentials
from analysis.ml.fetch_candidates import DATA_ANALYST_GAP_QUERIES, fetch_adzuna_snapshots
from analysis.ml.harvest import harvest_unlabeled_candidates
from analysis.ml.models import FORBIDDEN_GOLD_KEYS

FIXTURE = Path("tests/fixtures/adzuna/search_response.json")


def test_data_analyst_gap_queries_are_search_strings_not_labels():
    assert "data analyst" in DATA_ANALYST_GAP_QUERIES
    assert "insights analyst" in DATA_ANALYST_GAP_QUERIES
    assert "gold_role_family" not in DATA_ANALYST_GAP_QUERIES


def test_fetch_snapshots_are_unlabeled(tmp_path):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    body = json.dumps(payload).encode("utf-8")

    def fake_get(url: str, timeout: float) -> bytes:
        return body

    client = AdzunaClient(
        AdzunaCredentials(app_id="id", api_key="key"),
        http_get=fake_get,
    )
    snap_dir = tmp_path / "adzuna"
    result = fetch_adzuna_snapshots(
        queries=("data analyst",),
        countries=("ar",),
        pages=1,
        results_per_page=10,
        client=client,
        snapshot_dir=snap_dir,
    )
    assert result["n_unique"] >= 1
    assert result["fetched"] is True
    assert result["label_status"] == "unlabeled"
    assert result["n_requests"] == 1
    assert result["by_query"][0]["query"] == "data analyst"
    assert result["by_query"][0]["ok"] is True
    for row in result["records"]:
        assert "gold_role_family" not in row
        assert FORBIDDEN_GOLD_KEYS.isdisjoint(row.keys())
        assert row["source_url"]
    assert list(snap_dir.glob("*.json"))


def test_missing_credentials_does_not_invent(monkeypatch):
    monkeypatch.delenv("ADZUNA_APP_ID", raising=False)
    monkeypatch.delenv("ADZUNA_API_KEY", raising=False)
    result = fetch_adzuna_snapshots(queries=("x",), countries=("ar",), pages=1)
    assert result["fetched"] is False
    assert result["reason"] == "missing_credentials"
    assert result["records"] == []
    harvest = harvest_unlabeled_candidates()
    assert harvest["n_unique"] >= 1
