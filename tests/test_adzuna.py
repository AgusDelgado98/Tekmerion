"""
Tests for the Adzuna adapter (V0.4.3).

All tests are offline: fixtures + injected HTTP mocks.
No network, no real credentials.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from analysis.ingestion.adzuna import (
    SOURCE_NAME,
    AdzunaAPIError,
    AdzunaClient,
    AdzunaConfigError,
    AdzunaCredentials,
    AdzunaSource,
    build_search_url,
    load_credentials_from_env,
    map_adzuna_job,
    map_adzuna_results,
    save_raw_snapshot,
    ENV_APP_ID,
    ENV_API_KEY,
)
from analysis.ingestion import ingest, IngestionContext
from analysis.pipeline import process_records
from analysis.evidence import build_evidence


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "adzuna" / "search_response.json"


@pytest.fixture
def fixture_payload() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def test_credentials_from_env(monkeypatch):
    monkeypatch.setenv(ENV_APP_ID, "app-123")
    monkeypatch.setenv(ENV_API_KEY, "key-secret")
    creds = load_credentials_from_env()
    assert creds.app_id == "app-123"
    assert creds.api_key == "key-secret"
    # Secrets must not appear in repr
    assert "key-secret" not in repr(creds)
    assert "app-123" not in repr(creds)
    assert "***" in repr(creds)


def test_credentials_missing_both(monkeypatch):
    monkeypatch.delenv(ENV_APP_ID, raising=False)
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    with pytest.raises(AdzunaConfigError) as exc:
        load_credentials_from_env()
    msg = str(exc.value)
    assert ENV_APP_ID in msg
    assert ENV_API_KEY in msg
    # No fabricated secret values
    assert "key-secret" not in msg


def test_credentials_missing_key_only(monkeypatch):
    monkeypatch.setenv(ENV_APP_ID, "app-123")
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    with pytest.raises(AdzunaConfigError) as exc:
        load_credentials_from_env()
    assert ENV_API_KEY in str(exc.value)
    assert "app-123" not in str(exc.value)  # don't echo whatever was set oddly


def test_credentials_empty_string(monkeypatch):
    monkeypatch.setenv(ENV_APP_ID, "  ")
    monkeypatch.setenv(ENV_API_KEY, "key")
    with pytest.raises(AdzunaConfigError):
        load_credentials_from_env()


# ---------------------------------------------------------------------------
# URL / request params
# ---------------------------------------------------------------------------

def test_build_search_url_params():
    url = build_search_url(
        country="ar",
        page=1,
        app_id="AID",
        api_key="AKEY",
        what="data analyst",
        results_per_page=10,
        where="Buenos Aires",
    )
    assert "api.adzuna.com/v1/api/jobs/ar/search/1" in url
    assert "app_id=AID" in url
    assert "app_key=AKEY" in url
    assert "results_per_page=10" in url
    assert "what=data+analyst" in url or "what=data%20analyst" in url
    assert "where=Buenos" in url


def test_client_respects_results_cap(monkeypatch):
    """Hard ceiling: results_per_page cannot exceed MAX."""
    captured: dict[str, str] = {}

    def fake_get(url: str, timeout: float) -> bytes:
        captured["url"] = url
        return b'{"results": []}'

    client = AdzunaClient(
        AdzunaCredentials("id", "key"),
        http_get=fake_get,
    )
    client.search(what="ml engineer", results_per_page=999, country="gb")
    assert "results_per_page=50" in captured["url"]


def test_client_configurable_query_and_market():
    seen: list[str] = []

    def fake_get(url: str, timeout: float) -> bytes:
        seen.append(url)
        return b'{"results": []}'

    client = AdzunaClient(AdzunaCredentials("id", "key"), http_get=fake_get)
    client.search(what="data scientist", country="gb", page=2, results_per_page=5)
    assert "jobs/gb/search/2" in seen[0]
    assert "what=data+scientist" in seen[0] or "what=data%20scientist" in seen[0]
    assert "results_per_page=5" in seen[0]


# ---------------------------------------------------------------------------
# Response handling
# ---------------------------------------------------------------------------

def test_map_valid_fixture(fixture_payload):
    records = map_adzuna_results(fixture_payload)
    assert len(records) == 4
    first = records[0]
    assert first["id"] == "4829103751"
    assert first["title"] == "Data Analyst"
    assert first["company"] == "Tech Solutions SA"
    assert first["source"] == SOURCE_NAME
    assert first["source_url"] and "adzuna" in first["source_url"]
    assert "SQL" in first["description"]
    assert first["location"]
    assert first["salary_min"] == 1200000


def test_map_empty_results():
    assert map_adzuna_results({"results": []}) == []
    assert map_adzuna_results({}) == []


def test_map_missing_optional_fields():
    minimal = {
        "id": "1",
        "title": "Analyst",
        "description": "SQL",
        # no company, location, salary, redirect
    }
    mapped = map_adzuna_job(minimal)
    assert mapped["id"] == "1"
    assert mapped["company"] == ""
    assert mapped["location"] == ""
    assert mapped["source_url"] is None
    assert mapped["salary_min"] is None


def test_map_unexpected_structure():
    with pytest.raises(AdzunaAPIError):
        map_adzuna_results("not a dict")  # type: ignore
    with pytest.raises(AdzunaAPIError):
        map_adzuna_results({"results": "oops"})  # type: ignore


def test_client_http_error():
    def boom(url: str, timeout: float) -> bytes:
        raise AdzunaAPIError("Adzuna HTTP error: status=401", status_code=401)

    client = AdzunaClient(AdzunaCredentials("id", "key"), http_get=boom)
    with pytest.raises(AdzunaAPIError) as exc:
        client.search(what="x")
    assert exc.value.status_code == 401
    assert "key" not in str(exc.value).lower() or "api key" not in str(exc.value).lower()


def test_client_invalid_json():
    def bad_json(url: str, timeout: float) -> bytes:
        return b"not-json{{{"

    client = AdzunaClient(AdzunaCredentials("id", "key"), http_get=bad_json)
    with pytest.raises(AdzunaAPIError, match="not valid JSON"):
        client.search(what="x")


# ---------------------------------------------------------------------------
# Mapping → ingestion → pipeline → evidence
# ---------------------------------------------------------------------------

def test_adapter_from_payload_load(fixture_payload):
    source = AdzunaSource.from_payload(fixture_payload)
    rows = source.load()
    assert len(rows) == 4
    assert source.source_name() == "adzuna"
    assert all(r["source"] == "adzuna" for r in rows)


def test_ingest_namespaced_ids(fixture_payload):
    source = AdzunaSource.from_payload(fixture_payload)
    ctx = IngestionContext(retrieved_at="2026-08-10T18:00:00Z")
    result = ingest([source], context=ctx)

    assert result.accepted_count == 4
    for rec in result.records:
        assert rec["id"].startswith("adzuna:")
        assert rec["source_record_id"]  # original Adzuna id
        assert rec["source"] == "adzuna"
        assert rec["retrieved_at"] == "2026-08-10T18:00:00Z"


def test_end_to_end_fixture_pipeline_and_evidence(fixture_payload):
    source = AdzunaSource.from_payload(fixture_payload)
    ctx = IngestionContext(retrieved_at="2026-08-10T18:00:00Z")
    ing = ingest([source], context=ctx)
    pipe = process_records(ing.records)

    assert pipe.total_input == 4
    assert pipe.valid_count == 4
    assert pipe.duplicate_count == 0

    families = {r.role_family.value for r in pipe.records}
    assert "data_analyst" in families
    assert "bi_analyst" in families or "data_engineer" in families or "ml_engineer" in families

    # Skills extracted from descriptions
    all_skills = set()
    for r in pipe.records:
        all_skills.update(r.skills_extracted)
    assert "sql" in all_skills or "python" in all_skills

    ev = build_evidence(pipe.records)
    assert ev.n_analysis_records == 4
    assert len(ev.skill_frequency) > 0
    assert len(ev.role_distribution) > 0


def test_determinism_same_fixture_same_context(fixture_payload):
    ctx = IngestionContext(retrieved_at="2026-08-10T18:00:00Z")

    def run():
        src = AdzunaSource.from_payload(fixture_payload)
        return ingest([src], context=ctx)

    r1 = run()
    r2 = run()
    assert r1.records == r2.records

    p1 = process_records(r1.records)
    p2 = process_records(r2.records)
    assert p1.summary() == p2.summary()
    for a, b in zip(p1.records, p2.records):
        assert a.to_dict() == b.to_dict()


def test_record_with_missing_optionals_still_valid():
    """Company/location empty is ok for validity if title+company+description required —
    wait: company empty makes it invalid. Map a complete minimal ad."""
    ad = {
        "id": "99",
        "title": "Data Scientist",
        "description": "Python scikit-learn pandas statistics",
        "company": {"display_name": "Lab"},
        "redirect_url": "https://example.com/99",
    }
    mapped = map_adzuna_job(ad)
    ctx = IngestionContext(retrieved_at="2026-01-01T00:00:00Z")
    from analysis.ingestion.normalize import normalize_to_internal
    norm = normalize_to_internal(mapped, default_source="adzuna", context=ctx)
    pipe = process_records([norm])
    assert pipe.valid_count == 1
    assert pipe.records[0].role_family.value == "data_scientist"


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

def test_save_raw_snapshot_no_overwrite(tmp_path, fixture_payload):
    p1 = save_raw_snapshot(
        fixture_payload,
        directory=tmp_path,
        retrieved_at="2026-08-10T18:00:00Z",
        country="ar",
        query="data analyst",
        page=1,
    )
    assert p1.exists()
    data = json.loads(p1.read_text(encoding="utf-8"))
    assert data["source"] == "adzuna"
    assert data["retrieved_at"] == "2026-08-10T18:00:00Z"
    assert "payload" in data
    # No secrets in snapshot
    blob = p1.read_text(encoding="utf-8")
    assert "app_key" not in blob
    assert "ADZUNA_API_KEY" not in blob

    # Second save with same metadata must not overwrite
    p2 = save_raw_snapshot(
        fixture_payload,
        directory=tmp_path,
        retrieved_at="2026-08-10T18:00:00Z",
        country="ar",
        query="data analyst",
        page=1,
    )
    assert p2 != p1
    assert p1.exists() and p2.exists()


# ---------------------------------------------------------------------------
# Live path refuses without credentials (adapter construction still ok offline)
# ---------------------------------------------------------------------------

def test_source_without_client_raises_on_load():
    src = AdzunaSource(what="x")  # no client, no preload
    with pytest.raises(AdzunaConfigError):
        src.load()
