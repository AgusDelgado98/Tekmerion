"""Gold Dataset contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.ml.gold import GoldDatasetError, load_gold_dataset, repo_relative_path
from analysis.ml.models import FORBIDDEN_GOLD_KEYS, GOLD_SCHEMA, LABEL_SOURCE_HUMAN
from analysis.models import RoleFamily

FIXTURES = Path("tests/fixtures/ml")


def test_small_fixture_loads():
    ds = load_gold_dataset(FIXTURES / "gold_small.json")
    assert ds.schema == GOLD_SCHEMA
    assert ds.label_source == LABEL_SOURCE_HUMAN
    assert ds.label_field == "gold_role_family"
    assert ds.n == 5
    assert ds.limitations
    ids = {r.id for r in ds.records}
    assert ids == {"fx_a1", "fx_a2", "fx_b1", "fx_c1", "fx_d1"}
    for rec in ds.records:
        assert rec.label_source == LABEL_SOURCE_HUMAN
        assert isinstance(rec.gold_role_family, RoleFamily)
        assert rec.content_fingerprint


def test_fixture_file_has_no_classifier_fields():
    payload = json.loads((FIXTURES / "gold_small.json").read_text(encoding="utf-8"))
    for rec in payload["records"]:
        assert FORBIDDEN_GOLD_KEYS.isdisjoint(rec.keys())
        assert "gold_role_family" in rec
        assert rec.get("label_source") == "human"


def test_repo_relative_path_strips_drive_prefix():
    from analysis.ml.gold import DEFAULT_GOLD_PATH

    rel = repo_relative_path(DEFAULT_GOLD_PATH)
    assert rel == "data/ml/gold/role_family_v1.json"


def test_rejects_classifier_label_source():
    with pytest.raises(GoldDatasetError, match="label_source"):
        load_gold_dataset(FIXTURES / "gold_invalid_label_source.json")


def test_rejects_pipeline_role_family_on_record():
    with pytest.raises(GoldDatasetError, match="classifier/prediction"):
        load_gold_dataset(FIXTURES / "gold_forbidden_field.json")
