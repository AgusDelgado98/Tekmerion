"""
Tests for the V0.4 ingestion layer.

Coverage:
- correct load of real sample records
- provenance fields (source, source_url, retrieved_at)
- incomplete / invalid handling
- compatibility with existing pipeline
- determinism
- mix of synthetic + real without unexpected collisions
- no mutation of inputs
"""

from __future__ import annotations

import json
import copy
from pathlib import Path

import pytest

from analysis.ingestion import (
    ingest,
    ingest_local_file,
    LocalJsonSource,
    normalize_to_internal,
    IngestionResult,
    DEFAULT_REAL_SAMPLE,
    DEFAULT_SYNTHETIC_SAMPLE,
)
from analysis.ingestion.normalize import is_minimally_usable, reject_reason
from analysis.pipeline import process_records, process_file
from analysis.models import ProcessedJob


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REAL_SAMPLE_PATH = DEFAULT_REAL_SAMPLE
SYNTHETIC_PATH = DEFAULT_SYNTHETIC_SAMPLE


@pytest.fixture
def real_records_raw() -> list[dict]:
    with REAL_SAMPLE_PATH.open(encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Local loader
# ---------------------------------------------------------------------------

def test_local_json_source_loads_real_sample():
    adapter = LocalJsonSource(REAL_SAMPLE_PATH, source_name="curated_real_sample")
    rows = adapter.load()
    assert len(rows) == 4
    assert adapter.source_name() == "curated_real_sample"
    assert all(isinstance(r, dict) for r in rows)


def test_local_json_source_missing_file():
    adapter = LocalJsonSource("/tmp/tekmerion_does_not_exist_xyz.json")
    with pytest.raises(FileNotFoundError):
        adapter.load()


def test_local_json_source_rejects_non_list(tmp_path: Path):
    bad = tmp_path / "not_list.json"
    bad.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    adapter = LocalJsonSource(bad)
    with pytest.raises(ValueError, match="Expected a JSON list"):
        adapter.load()


# ---------------------------------------------------------------------------
# Normalization & provenance
# ---------------------------------------------------------------------------

def test_normalize_preserves_provenance():
    raw = {
        "id": "r1",
        "title": "Data Analyst",
        "company": "Acme",
        "description": "SQL and Python",
        "source": "curated_real_sample",
        "source_url": "https://example.com/job/1",
        "retrieved_at": "2026-08-01T12:00:00Z",
    }
    norm = normalize_to_internal(raw)
    assert norm["source"] == "curated_real_sample"
    assert norm["source_url"] == "https://example.com/job/1"
    assert norm["retrieved_at"] == "2026-08-01T12:00:00Z"
    assert norm["title"] == "Data Analyst"
    assert norm["id"] == "curated_real_sample:r1"
    assert norm["source_record_id"] == "r1"


def test_normalize_fills_missing_provenance_without_clock():
    """Without context, missing retrieved_at stays None (no datetime.now)."""
    raw = {
        "id": "r2",
        "title": "BI Analyst",
        "company": "Co",
        "description": "Power BI",
    }
    norm = normalize_to_internal(raw, default_source="test_source")
    assert norm["source"] == "test_source"
    assert norm["source_url"] is None
    assert norm["retrieved_at"] is None
    # Internal id is namespaced
    assert norm["id"] == "test_source:r2"
    assert norm["source_record_id"] == "r2"


def test_normalize_does_not_mutate_input():
    raw = {
        "id": "r3",
        "title": "ML Engineer",
        "company": "X",
        "description": "Python",
        "source": "orig",
    }
    snapshot = copy.deepcopy(raw)
    normalize_to_internal(raw)
    assert raw == snapshot


def test_normalize_incomplete_still_returns_dict():
    raw = {"id": "incomplete", "title": ""}  # missing company/description
    norm = normalize_to_internal(raw, default_source="test")
    assert isinstance(norm, dict)
    assert norm["id"] == "test:incomplete"
    assert norm["source_record_id"] == "incomplete"
    assert norm["title"] == ""
    assert norm["company"] == ""
    assert norm["description"] == ""
    assert norm["source"] == "test"


def test_normalize_non_dict():
    norm = normalize_to_internal("garbage", default_source="test")  # type: ignore
    assert norm["id"].startswith("test:")
    assert "invalid_non_dict" in norm["id"]
    assert norm["source"] == "test"


# ---------------------------------------------------------------------------
# Ingest high-level
# ---------------------------------------------------------------------------

def test_ingest_real_sample():
    result = ingest_local_file(REAL_SAMPLE_PATH, source_name="curated_real_sample")
    assert isinstance(result, IngestionResult)
    assert result.accepted_count == 4
    assert result.rejected_count == 0
    assert result.total_loaded == 4
    assert "curated_real_sample" in result.source_names

    for rec in result.records:
        assert rec["source"] == "curated_real_sample"
        assert rec.get("source_url")
        assert rec.get("retrieved_at")
        # Namespaced internal id: source:external_id
        assert rec["id"].startswith("curated_real_sample:")
        assert rec.get("source_record_id")  # original external id preserved


def test_ingest_preserves_required_fields():
    result = ingest_local_file(REAL_SAMPLE_PATH)
    for rec in result.records:
        for key in ("id", "title", "company", "description", "source"):
            assert key in rec


def test_ingest_incomplete_records_are_kept_by_default(tmp_path: Path):
    """Incomplete records are normalized and included; pipeline will mark invalid."""
    path = tmp_path / "mixed.json"
    data = [
        {
            "id": "ok1",
            "title": "Data Analyst",
            "company": "GoodCo",
            "description": "SQL Python",
            "source": "test",
        },
        {
            "id": "bad1",
            "title": "",
            "company": "",
            "description": "",
            "source": "test",
        },
    ]
    path.write_text(json.dumps(data), encoding="utf-8")

    result = ingest_local_file(path, source_name="test")
    assert result.accepted_count == 2
    assert result.rejected_count == 0

    # Pipeline should mark the incomplete one invalid
    pipe = process_records(result.records)
    assert pipe.valid_count == 1
    assert pipe.invalid_count == 1
    invalid = next(r for r in pipe.records if not r.is_valid)
    assert "missing_or_empty_title" in invalid.validation_errors or \
           "missing_or_empty_company" in invalid.validation_errors


def test_ingest_reject_unusable(tmp_path: Path):
    path = tmp_path / "emptyish.json"
    data = [
        {"id": "ok", "title": "Analyst", "company": "C", "description": "SQL"},
        {"id": "", "title": "", "company": "", "description": ""},
    ]
    path.write_text(json.dumps(data), encoding="utf-8")

    result = ingest_local_file(path, reject_unusable=True)
    assert result.accepted_count == 1
    assert result.rejected_count == 1
    assert result.rejections[0]["reason"] == "not_minimally_usable"


def test_ingest_rejects_non_dict_items(tmp_path: Path):
    path = tmp_path / "with_garbage.json"
    data = [
        {"id": "ok", "title": "A", "company": "B", "description": "C"},
        "not a dict",
        42,
    ]
    path.write_text(json.dumps(data), encoding="utf-8")

    result = ingest_local_file(path)
    assert result.accepted_count == 1
    assert result.rejected_count == 2
    assert all(r["reason"] == "not_a_dict" for r in result.rejections)


# ---------------------------------------------------------------------------
# Pipeline compatibility
# ---------------------------------------------------------------------------

def test_real_records_pass_through_pipeline():
    ing = ingest_local_file(REAL_SAMPLE_PATH, source_name="curated_real_sample")
    pipe = process_records(ing.records)

    assert pipe.total_input == 4
    assert pipe.valid_count == 4
    assert pipe.invalid_count == 0
    assert pipe.duplicate_count == 0

    for job in pipe.records:
        assert isinstance(job, ProcessedJob)
        assert job.is_valid is True
        assert job.source == "curated_real_sample"
        assert job.source_url is not None
        assert job.retrieved_at is not None
        assert job.id.startswith("curated_real_sample:")
        assert job.source_record_id is not None


def test_provenance_survives_pipeline():
    ing = ingest_local_file(REAL_SAMPLE_PATH)
    pipe = process_records(ing.records)
    job = pipe.records[0]
    assert job.source_url.startswith("https://example.com/")
    assert "2026-08-01" in (job.retrieved_at or "")


def test_mix_synthetic_and_real_no_unexpected_collisions():
    """Mixing both sources must not create false duplicates or id clashes."""
    synthetic = json.loads(SYNTHETIC_PATH.read_text(encoding="utf-8"))
    real_ing = ingest_local_file(REAL_SAMPLE_PATH)
    combined = synthetic + real_ing.records

    pipe = process_records(combined)
    assert pipe.total_input == len(synthetic) + 4
    # All real ones should be valid and non-duplicate
    real_jobs = [r for r in pipe.records if r.id.startswith("curated_real_sample:")]
    assert len(real_jobs) == 4
    assert all(r.is_valid for r in real_jobs)
    assert all(not r.is_duplicate for r in real_jobs)

    # IDs must be unique across the batch
    ids = [r.id for r in pipe.records]
    assert len(ids) == len(set(ids))


def test_synthetic_still_works_unchanged():
    """Existing synthetic path must keep producing the same counts."""
    result = process_file(SYNTHETIC_PATH)
    assert result.total_input == 17
    assert result.valid_count == 16  # one intentional invalid in the sample
    assert result.invalid_count == 1
    assert result.duplicate_count == 1
    # Provenance fields exist and are None for legacy synthetic
    for job in result.records:
        assert job.source == "synthetic"
        assert job.source_url is None
        assert job.retrieved_at is None


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_ingest_determinism():
    r1 = ingest_local_file(REAL_SAMPLE_PATH, source_name="curated_real_sample")
    r2 = ingest_local_file(REAL_SAMPLE_PATH, source_name="curated_real_sample")
    assert r1.summary() == r2.summary()
    assert r1.records == r2.records


def test_pipeline_on_real_is_deterministic():
    ing = ingest_local_file(REAL_SAMPLE_PATH)
    p1 = process_records(ing.records)
    p2 = process_records(ing.records)
    assert p1.summary() == p2.summary()
    for a, b in zip(p1.records, p2.records):
        assert a.to_dict() == b.to_dict()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_is_minimally_usable():
    assert is_minimally_usable({"title": "X", "company": "Y", "description": "Z"}) is True
    assert is_minimally_usable({"title": "X"}) is False  # missing company/description
    assert is_minimally_usable({"id": "1", "title": "", "company": "", "description": ""}) is False
    assert is_minimally_usable("nope") is False  # type: ignore


def test_reject_reason():
    assert reject_reason("x") == "not_a_dict"
    assert reject_reason({"id": "1"}) is None
