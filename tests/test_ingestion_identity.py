"""
V0.4.2 — Ingestion identity, context, and cross-source dedup tests.

Covers:
- same external id from different sources → no id collision
- same external id within one source → content dedup still works
- record without external id → deterministic fallback
- same input + same context → same result
- retrieved_at from explicit context
- different context timestamps reflected correctly
- synthetic legacy unchanged
- same-source content duplicate
- cross-source identical content stays independent
- pipeline accepts namespaced results
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.ingestion import (
    IngestionContext,
    ingest,
    ingest_local_file,
    normalize_to_internal,
    build_internal_id,
    deterministic_fallback_external_id,
    LocalJsonSource,
    DEFAULT_REAL_SAMPLE,
    DEFAULT_SYNTHETIC_SAMPLE,
)
from analysis.pipeline import process_records, process_file
from analysis.models import ProcessedJob


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------

def test_build_internal_id_with_external():
    internal, external = build_internal_id("adzuna", "12345")
    assert internal == "adzuna:12345"
    assert external == "12345"


def test_build_internal_id_sanitizes_source():
    internal, external = build_internal_id("my source!", "abc")
    assert internal == "my_source:abc"
    assert external == "abc"


def test_build_internal_id_fallback_is_deterministic():
    a, ext_a = build_internal_id(
        "src",
        None,
        company="Acme",
        title="Data Analyst",
        location="BA",
        source_url="https://example.com/1",
    )
    b, ext_b = build_internal_id(
        "src",
        None,
        company="Acme",
        title="Data Analyst",
        location="BA",
        source_url="https://example.com/1",
    )
    assert a == b
    assert a.startswith("src:auto:")
    assert ext_a is None and ext_b is None


def test_build_internal_id_fallback_changes_with_fields():
    a, _ = build_internal_id("src", None, company="Acme", title="DA", location="", source_url="")
    b, _ = build_internal_id("src", None, company="Other", title="DA", location="", source_url="")
    assert a != b


def test_deterministic_fallback_external_id_stable():
    x = deterministic_fallback_external_id(
        source="s", company="c", title="t", location="l", source_url="u"
    )
    y = deterministic_fallback_external_id(
        source="s", company="c", title="t", location="l", source_url="u"
    )
    assert x == y
    assert x.startswith("auto:")


# ---------------------------------------------------------------------------
# Same external id, different sources → no collision
# ---------------------------------------------------------------------------

def test_same_external_id_different_sources_no_collision():
    rec_a = {
        "id": "shared-999",
        "title": "Data Analyst",
        "company": "CoA",
        "description": "SQL Python",
        "source": "source_alpha",
    }
    rec_b = {
        "id": "shared-999",
        "title": "BI Analyst",
        "company": "CoB",
        "description": "Power BI Tableau",
        "source": "source_beta",
    }
    na = normalize_to_internal(rec_a, default_source="source_alpha")
    nb = normalize_to_internal(rec_b, default_source="source_beta")

    assert na["id"] == "source_alpha:shared-999"
    assert nb["id"] == "source_beta:shared-999"
    assert na["id"] != nb["id"]
    assert na["source_record_id"] == "shared-999"
    assert nb["source_record_id"] == "shared-999"

    pipe = process_records([na, nb])
    assert pipe.total_input == 2
    assert pipe.valid_count == 2
    ids = {r.id for r in pipe.records}
    assert ids == {"source_alpha:shared-999", "source_beta:shared-999"}
    assert pipe.duplicate_count == 0


# ---------------------------------------------------------------------------
# Same external id, same source → content dedup when content matches
# ---------------------------------------------------------------------------

def test_same_source_same_content_is_duplicate():
    base = {
        "id": "dup-1",
        "title": "Data Analyst",
        "company": "SameCo",
        "description": "Identical description for same-source dedup test.",
        "source": "portal_x",
    }
    a = normalize_to_internal(base, default_source="portal_x")
    b = normalize_to_internal({**base, "id": "dup-2"}, default_source="portal_x")

    pipe = process_records([a, b])
    assert pipe.duplicate_count == 1
    assert pipe.records[0].is_duplicate is False
    assert pipe.records[1].is_duplicate is True
    assert pipe.records[1].duplicate_of == a["id"]


def test_same_external_id_same_source_different_content_not_content_dup():
    """Two rows with different content are not content-duplicates even if ids collide at source."""
    a = normalize_to_internal(
        {
            "id": "x1",
            "title": "Data Analyst",
            "company": "Co",
            "description": "SQL only",
            "source": "portal",
        },
        default_source="portal",
    )
    b = normalize_to_internal(
        {
            "id": "x1",  # same external id — namespaced id will also collide
            "title": "ML Engineer",
            "company": "Co",
            "description": "Python Docker Kubernetes",
            "source": "portal",
        },
        default_source="portal",
    )
    # Internal ids are identical (source:external) — that is a data-quality issue
    # at the source, not something we invent uniqueness for.
    assert a["id"] == b["id"] == "portal:x1"
    pipe = process_records([a, b])
    # Content differs → not a content-fingerprint duplicate
    assert pipe.duplicate_count == 0


# ---------------------------------------------------------------------------
# No external id → deterministic fallback
# ---------------------------------------------------------------------------

def test_record_without_external_id_gets_deterministic_fallback():
    raw = {
        "title": "Senior Data Engineer",
        "company": "Pipeline Corp",
        "location": "Remote",
        "description": "Airflow and Spark",
        "source": "manual",
        "source_url": "https://example.com/jobs/42",
    }
    n1 = normalize_to_internal(raw, default_source="manual")
    n2 = normalize_to_internal(raw, default_source="manual")

    assert n1["id"] == n2["id"]
    assert n1["id"].startswith("manual:auto:")
    assert n1["source_record_id"] is None

    pipe = process_records([n1])
    assert pipe.valid_count == 1
    assert pipe.records[0].id == n1["id"]
    assert pipe.records[0].source_record_id is None


# ---------------------------------------------------------------------------
# IngestionContext / retrieved_at determinism
# ---------------------------------------------------------------------------

def test_context_supplies_retrieved_at():
    ctx = IngestionContext(retrieved_at="2026-08-10T12:00:00Z")
    raw = {
        "id": "t1",
        "title": "Analyst",
        "company": "Co",
        "description": "SQL",
        # no retrieved_at
    }
    norm = normalize_to_internal(raw, default_source="test", context=ctx)
    assert norm["retrieved_at"] == "2026-08-10T12:00:00Z"


def test_record_retrieved_at_wins_over_context():
    ctx = IngestionContext(retrieved_at="2026-01-01T00:00:00Z")
    raw = {
        "id": "t1",
        "title": "Analyst",
        "company": "Co",
        "description": "SQL",
        "retrieved_at": "2026-07-15T10:00:00Z",
    }
    norm = normalize_to_internal(raw, default_source="test", context=ctx)
    assert norm["retrieved_at"] == "2026-07-15T10:00:00Z"


def test_same_input_same_context_same_result(tmp_path: Path):
    path = tmp_path / "jobs.json"
    data = [
        {
            "id": "a1",
            "title": "Data Analyst",
            "company": "Acme",
            "description": "SQL Python",
            # no retrieved_at
        },
        {
            "title": "BI Analyst",  # no id
            "company": "Beta",
            "location": "CABA",
            "description": "Power BI",
            "source_url": "https://example.com/b",
        },
    ]
    path.write_text(json.dumps(data), encoding="utf-8")
    ctx = IngestionContext(retrieved_at="2026-08-10T15:00:00Z")

    r1 = ingest_local_file(path, source_name="portal", context=ctx)
    r2 = ingest_local_file(path, source_name="portal", context=ctx)

    assert r1.summary() == r2.summary()
    assert r1.records == r2.records
    assert all(rec["retrieved_at"] == "2026-08-10T15:00:00Z" for rec in r1.records)


def test_different_context_timestamps_reflected(tmp_path: Path):
    path = tmp_path / "jobs.json"
    data = [
        {
            "id": "a1",
            "title": "Data Analyst",
            "company": "Acme",
            "description": "SQL",
        }
    ]
    path.write_text(json.dumps(data), encoding="utf-8")

    c1 = IngestionContext(retrieved_at="2026-08-01T00:00:00Z")
    c2 = IngestionContext(retrieved_at="2026-08-02T00:00:00Z")

    r1 = ingest_local_file(path, source_name="portal", context=c1)
    r2 = ingest_local_file(path, source_name="portal", context=c2)

    assert r1.records[0]["retrieved_at"] == "2026-08-01T00:00:00Z"
    assert r2.records[0]["retrieved_at"] == "2026-08-02T00:00:00Z"
    # ids stay the same (content/identity unchanged)
    assert r1.records[0]["id"] == r2.records[0]["id"]


def test_context_rejects_empty_timestamp():
    with pytest.raises(ValueError):
        IngestionContext(retrieved_at="")
    with pytest.raises(ValueError):
        IngestionContext(retrieved_at="   ")


def test_no_context_no_timestamp_stays_none():
    raw = {"id": "x", "title": "T", "company": "C", "description": "D"}
    norm = normalize_to_internal(raw, default_source="s")
    assert norm["retrieved_at"] is None


# ---------------------------------------------------------------------------
# Cross-source content: independent (no false dedup)
# ---------------------------------------------------------------------------

def test_cross_source_identical_content_not_duplicate():
    content = {
        "title": "Data Analyst",
        "company": "SharedCo",
        "description": "Exact same posting text for cross-source test.",
    }
    a = normalize_to_internal({**content, "id": "1", "source": "source_a"}, default_source="source_a")
    b = normalize_to_internal({**content, "id": "1", "source": "source_b"}, default_source="source_b")

    assert a["id"] != b["id"]
    pipe = process_records([a, b])
    assert pipe.duplicate_count == 0
    assert all(not r.is_duplicate for r in pipe.records)


# ---------------------------------------------------------------------------
# Synthetic legacy still works
# ---------------------------------------------------------------------------

def test_synthetic_legacy_ids_unchanged():
    """process_file on synthetic sample does not go through ingestion namespacing."""
    result = process_file(DEFAULT_SYNTHETIC_SAMPLE)
    assert result.total_input == 17
    assert result.valid_count == 16
    assert result.records[0].id == "job_001"
    assert result.records[0].source == "synthetic"
    assert result.records[0].source_record_id is None
    assert result.records[0].source_url is None
    assert result.records[0].retrieved_at is None


def test_synthetic_duplicate_still_detected():
    """job_013 is a content duplicate of job_001 within synthetic source."""
    result = process_file(DEFAULT_SYNTHETIC_SAMPLE)
    dups = [r for r in result.records if r.is_duplicate]
    assert len(dups) == 1
    assert dups[0].duplicate_of == "job_001"


# ---------------------------------------------------------------------------
# Real sample end-to-end with namespaced ids
# ---------------------------------------------------------------------------

def test_real_sample_namespaced_ids():
    ing = ingest_local_file(DEFAULT_REAL_SAMPLE, source_name="curated_real_sample")
    assert ing.accepted_count == 4
    for rec in ing.records:
        assert rec["id"].startswith("curated_real_sample:")
        assert ":" in rec["id"]
        assert not rec["id"].startswith("real_")  # old manual prefix gone
        assert rec["source_record_id"]  # original external id kept

    pipe = process_records(ing.records)
    assert pipe.valid_count == 4
    assert pipe.duplicate_count == 0
    ids = [r.id for r in pipe.records]
    assert len(ids) == len(set(ids))


def test_pipeline_accepts_namespaced_results():
    ing = ingest_local_file(DEFAULT_REAL_SAMPLE)
    pipe = process_records(ing.records)
    for job in pipe.records:
        assert isinstance(job, ProcessedJob)
        assert job.is_valid
        assert job.source_record_id is not None
        d = job.to_dict()
        assert "source_record_id" in d
        assert d["id"].startswith("curated_real_sample:")
