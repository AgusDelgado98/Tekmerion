"""
Offline tests for Market Batch (V0.4.4).

No network. Uses fixtures under tests/fixtures/adzuna/batch/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.ingestion.market import (
    DEFAULT_MARKET_QUERIES,
    MarketQuery,
    merge_by_identity,
    run_market_batch,
    market_artifact_dict,
    save_market_artifact,
    _pick_canonical,
    _completeness_score,
)
from analysis.ingestion.normalize import normalize_to_internal
from analysis.ingestion.base import IngestionContext
from analysis.ingestion.adzuna import map_adzuna_results, SOURCE_NAME


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "adzuna" / "batch"

QUERY_FILES = {
    "data analyst": "query_data_analyst.json",
    "business intelligence analyst": "query_bi_analyst.json",
    "data scientist": "query_data_scientist.json",
    "machine learning engineer": "query_ml_engineer.json",
    "data engineer": "query_data_engineer.json",
    "business analyst": "query_business_analyst.json",
}


def _load_payloads(*query_names: str) -> dict[str, dict]:
    out = {}
    for name in query_names:
        path = FIXTURE_DIR / QUERY_FILES[name]
        out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _all_payloads() -> dict[str, dict]:
    return _load_payloads(*QUERY_FILES.keys())


RETRIEVED = "2026-08-10T19:00:00Z"


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def test_completeness_prefers_richer_record():
    thin = {"title": "A", "company": "C", "description": "x", "location": "", "source_url": None}
    rich = {
        "title": "A",
        "company": "C",
        "description": "longer description here",
        "location": "BA",
        "source_url": "https://x",
        "salary_min": 1,
    }
    assert _completeness_score(rich) > _completeness_score(thin)


def test_pick_canonical_tie_breaks_on_query_name():
    a = ("zebra query", {"title": "T", "company": "C", "description": "same", "location": "L", "source_url": "u"})
    b = ("alpha query", {"title": "T", "company": "C", "description": "same", "location": "L", "source_url": "u"})
    winner_q, _ = _pick_canonical([a, b])
    assert winner_q == "alpha query"


def test_merge_by_identity_single():
    rec = normalize_to_internal(
        {"id": "1", "title": "DA", "company": "Co", "description": "SQL", "source": "adzuna"},
        default_source="adzuna",
        context=IngestionContext(retrieved_at=RETRIEVED),
    )
    cons, matched, removed = merge_by_identity([("data analyst", rec)])
    assert len(cons) == 1
    assert removed == 0
    assert matched[rec["id"]] == ["data analyst"]


def test_merge_overlap_keeps_one():
    ctx = IngestionContext(retrieved_at=RETRIEVED)
    base = {
        "id": "999",
        "title": "Data Analyst",
        "company": "Shared",
        "description": "SQL Python",
        "source": "adzuna",
    }
    a = normalize_to_internal(base, default_source="adzuna", context=ctx)
    b = normalize_to_internal(base, default_source="adzuna", context=ctx)
    cons, matched, removed = merge_by_identity(
        [("data analyst", a), ("business intelligence analyst", b)]
    )
    assert len(cons) == 1
    assert removed == 1
    assert matched[cons[0]["id"]] == [
        "business intelligence analyst",
        "data analyst",
    ]


# ---------------------------------------------------------------------------
# Full batch with fixtures
# ---------------------------------------------------------------------------

def test_batch_normal_run():
    payloads = _all_payloads()
    result = run_market_batch(
        country="ar",
        queries=list(QUERY_FILES.keys()),
        limit_per_query=10,
        retrieved_at=RETRIEVED,
        payload_by_query=payloads,
    )
    assert result.total_received == sum(
        len(p["results"]) for p in payloads.values()
    )
    assert result.unique_count < result.total_received
    assert result.duplicates_removed == result.total_received - result.unique_count
    assert result.pipeline_result is not None
    assert result.pipeline_result.valid_count == result.unique_count
    assert result.evidence is not None
    assert result.evidence.n_analysis_records == result.unique_count


def test_batch_overlap_vacancy_once():
    """7001999 appears in data analyst and BI fixtures → one consolidated record."""
    payloads = _load_payloads("data analyst", "business intelligence analyst")
    result = run_market_batch(
        country="ar",
        queries=["data analyst", "business intelligence analyst"],
        limit_per_query=10,
        retrieved_at=RETRIEVED,
        payload_by_query=payloads,
    )
    ids = [r["id"] for r in result.consolidated_records]
    assert ids.count("adzuna:7001999") == 1
    assert "data analyst" in result.matched_queries_by_id["adzuna:7001999"]
    assert "business intelligence analyst" in result.matched_queries_by_id["adzuna:7001999"]


def test_batch_order_independent():
    payloads = _all_payloads()
    queries_a = list(QUERY_FILES.keys())
    queries_b = list(reversed(queries_a))

    r1 = run_market_batch(
        country="ar",
        queries=queries_a,
        limit_per_query=10,
        retrieved_at=RETRIEVED,
        payload_by_query=payloads,
    )
    r2 = run_market_batch(
        country="ar",
        queries=queries_b,
        limit_per_query=10,
        retrieved_at=RETRIEVED,
        payload_by_query=payloads,
    )

    assert r1.unique_count == r2.unique_count
    assert r1.duplicates_removed == r2.duplicates_removed
    # Same set of ids
    assert {r["id"] for r in r1.consolidated_records} == {
        r["id"] for r in r2.consolidated_records
    }
    # Canonical content per id identical
    by_id_1 = {r["id"]: r for r in r1.consolidated_records}
    by_id_2 = {r["id"]: r for r in r2.consolidated_records}
    for rid in by_id_1:
        assert by_id_1[rid] == by_id_2[rid]
    # Pipeline determinism
    assert r1.pipeline_result.summary() == r2.pipeline_result.summary()
    for a, b in zip(
        sorted(r1.pipeline_result.records, key=lambda x: x.id),
        sorted(r2.pipeline_result.records, key=lambda x: x.id),
    ):
        assert a.to_dict() == b.to_dict()


def test_conflict_prefers_more_complete():
    """
    7001999 in DA fixture lacks salary; BI fixture has salary + longer description.
    Winner must be the richer BI representation regardless of query order.
    """
    payloads = _load_payloads("data analyst", "business intelligence analyst")

    for order in (
        ["data analyst", "business intelligence analyst"],
        ["business intelligence analyst", "data analyst"],
    ):
        result = run_market_batch(
            country="ar",
            queries=order,
            limit_per_query=10,
            retrieved_at=RETRIEVED,
            payload_by_query=payloads,
        )
        winner = next(
            r for r in result.consolidated_records if r["id"] == "adzuna:7001999"
        )
        assert winner.get("salary_min") == 1600000
        assert "Extra note from BI feed" in winner["description"]


def test_empty_query_still_works():
    payloads = _load_payloads("data analyst")
    payloads["empty role"] = json.loads(
        (FIXTURE_DIR / "query_empty.json").read_text(encoding="utf-8")
    )
    result = run_market_batch(
        country="ar",
        queries=["data analyst", "empty role"],
        limit_per_query=10,
        retrieved_at=RETRIEVED,
        payload_by_query=payloads,
    )
    assert result.query_outcomes[1].received_count == 0
    assert result.unique_count == len(payloads["data analyst"]["results"])


def test_fail_fast_missing_fixture():
    with pytest.raises(KeyError):
        run_market_batch(
            country="ar",
            queries=["data analyst", "missing query"],
            retrieved_at=RETRIEVED,
            payload_by_query=_load_payloads("data analyst"),
        )


def test_default_preset_keys():
    assert "data analyst" in DEFAULT_MARKET_QUERIES
    assert "machine learning engineer" in DEFAULT_MARKET_QUERIES
    assert len(DEFAULT_MARKET_QUERIES) == 6


def test_artifact_roundtrip(tmp_path):
    result = run_market_batch(
        country="ar",
        queries=["data analyst", "data engineer"],
        limit_per_query=10,
        retrieved_at=RETRIEVED,
        payload_by_query=_load_payloads("data analyst", "data engineer"),
    )
    path = save_market_artifact(result, directory=tmp_path)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema"] == "tekmerion.market_batch.v1"
    assert data["source"] == SOURCE_NAME
    assert data["unique_records"] == result.unique_count
    assert len(data["records"]) == result.unique_count
    assert "app_key" not in path.read_text()
    # no silent overwrite
    path2 = save_market_artifact(result, directory=tmp_path)
    assert path2 != path


def test_pipeline_roles_and_skills_from_batch():
    result = run_market_batch(
        country="ar",
        queries=list(QUERY_FILES.keys()),
        limit_per_query=10,
        retrieved_at=RETRIEVED,
        payload_by_query=_all_payloads(),
    )
    assert result.pipeline_result.valid_count >= 10
    families = result.pipeline_result.role_family_counts
    assert any(k in families for k in ("data_analyst", "data_engineer", "ml_engineer"))
    top = [s["item"] for s in result.evidence.skill_frequency[:5]]
    assert "sql" in top or "python" in top


def test_shared_retrieved_at():
    result = run_market_batch(
        country="ar",
        queries=["data analyst"],
        retrieved_at=RETRIEVED,
        payload_by_query=_load_payloads("data analyst"),
    )
    assert all(r["retrieved_at"] == RETRIEVED for r in result.consolidated_records)


def test_market_query_object_accepted():
    payloads = _load_payloads("data analyst")
    result = run_market_batch(
        country="ar",
        queries=[MarketQuery(what="data analyst", results_per_page=5)],
        retrieved_at=RETRIEVED,
        payload_by_query=payloads,
    )
    assert result.unique_count == 4
