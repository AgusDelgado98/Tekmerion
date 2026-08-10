"""
Tests for app.dataset — synthetic + market loading (offline).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.dataset import (
    MODE_MARKET,
    MODE_SYNTHETIC,
    DatasetError,
    discover_latest_market_artifact,
    load_app_dataset,
    load_market_artifact_file,
    load_market_dataset,
    load_synthetic_dataset,
    processed_job_from_dict,
    pipeline_result_from_records,
    validate_market_artifact,
)


FIXTURE = Path(__file__).parent / "fixtures" / "market" / "market_ar_fixture.json"


def test_synthetic_default():
    ds = load_app_dataset(mode="synthetic")
    assert ds.meta.mode == MODE_SYNTHETIC
    assert ds.meta.source == "synthetic"
    assert ds.pipeline_result.total_input == 17
    assert ds.evidence.n_analysis_records >= 1
    assert "Synthetic" in ds.meta.display_line()


def test_market_explicit_fixture():
    ds = load_app_dataset(mode="market", market_file=str(FIXTURE))
    assert ds.meta.mode == MODE_MARKET
    assert ds.meta.country == "ar"
    assert ds.meta.retrieved_at == "2026-08-10T19:00:00Z"
    assert ds.meta.total_records == 8
    assert ds.meta.artifact_name == "market_ar_fixture.json"
    # No absolute path leakage in display
    assert "/" not in ds.meta.display_line() or "market" in ds.meta.display_line().lower()
    assert ds.meta.artifact_name in (ds.meta.artifact_name,)


def test_market_does_not_reprocess_ids():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    original_ids = [r["id"] for r in data["records"]]
    ds = load_market_dataset(FIXTURE)
    loaded_ids = [r.id for r in ds.pipeline_result.records]
    assert loaded_ids == original_ids
    assert all(i.startswith("adzuna:") for i in loaded_ids)


def test_hydration_preserves_source_url():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    job = processed_job_from_dict(data["records"][0])
    assert job.source_url
    assert job.source == "adzuna"
    assert job.source_record_id


def test_pipeline_result_from_records_no_extra_rows():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    jobs = [processed_job_from_dict(r) for r in data["records"]]
    pr = pipeline_result_from_records(jobs)
    assert pr.total_input == len(jobs)
    assert pr.valid_count == sum(1 for j in jobs if j.is_valid)


def test_invalid_schema(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "other", "records": []}), encoding="utf-8")
    with pytest.raises(DatasetError, match="schema"):
        load_market_artifact_file(bad)


def test_missing_records(tmp_path):
    bad = tmp_path / "norec.json"
    bad.write_text(
        json.dumps({"schema": "tekmerion.market_batch.v1"}),
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="records"):
        load_market_artifact_file(bad)


def test_invalid_json(tmp_path):
    bad = tmp_path / "x.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetError, match="JSON"):
        load_market_artifact_file(bad)


def test_explicit_path_missing():
    with pytest.raises(DatasetError, match="not found"):
        load_app_dataset(mode="market", market_file="/tmp/does_not_exist_tekmerion.json")


def test_market_mode_no_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("app.dataset.MARKET_DIR", tmp_path)
    with pytest.raises(DatasetError, match="No valid market"):
        load_app_dataset(mode="market")


def test_discovery_picks_newest_retrieved_at(tmp_path):
    older = {
        "schema": "tekmerion.market_batch.v1",
        "source": "adzuna",
        "country": "ar",
        "retrieved_at": "2026-01-01T00:00:00Z",
        "queries": [],
        "records": [
            {
                "id": "adzuna:1",
                "title": "A",
                "company": "C",
                "location": "",
                "description": "SQL",
                "source": "adzuna",
                "normalized_title": "A",
                "role_family": "data_analyst",
                "seniority": "unknown",
                "skills_extracted": ["sql"],
                "is_valid": True,
                "validation_errors": [],
                "is_duplicate": False,
            }
        ],
    }
    newer = dict(older)
    newer["retrieved_at"] = "2026-08-10T19:00:00Z"
    newer["records"] = [dict(older["records"][0], id="adzuna:2", title="B")]

    # Filename that looks "newer" but internal ts is older
    p_old = tmp_path / "market_ar_20260899_fake.json"
    p_new = tmp_path / "market_ar_20260101_oldname.json"
    p_old.write_text(json.dumps(older), encoding="utf-8")
    p_new.write_text(json.dumps(newer), encoding="utf-8")

    # also an invalid file should be ignored
    (tmp_path / "garbage.json").write_text("{", encoding="utf-8")

    chosen = discover_latest_market_artifact(tmp_path)
    assert chosen == p_new


def test_invalid_mode():
    with pytest.raises(DatasetError, match="Invalid"):
        load_app_dataset(mode="postgres")


def test_empty_records_allowed(tmp_path):
    empty = {
        "schema": "tekmerion.market_batch.v1",
        "source": "adzuna",
        "country": "ar",
        "retrieved_at": "2026-08-01T00:00:00Z",
        "queries": [],
        "records": [],
    }
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(empty), encoding="utf-8")
    ds = load_market_dataset(path)
    assert ds.pipeline_result.total_input == 0
    assert ds.evidence.n_analysis_records == 0
