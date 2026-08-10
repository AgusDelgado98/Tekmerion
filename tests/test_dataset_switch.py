"""
V0.5.3 — Dataset demo switch (offline).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.registry import (
    SYNTHETIC_ID,
    build_registry,
    discover_market_entries,
    make_market_dataset_id,
)
from app.dataset import DatasetError, MODE_SYNTHETIC
from analysis.generative.providers import FakeProvider
from analysis.generative.service import clear_analysis_cache
from app import create_app


FIXTURE = Path(__file__).parent / "fixtures" / "market" / "market_ar_fixture.json"


@pytest.fixture
def market_dir(tmp_path):
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    valid = tmp_path / "market_ar_2026-08-10.json"
    valid.write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "garbage.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "wrong_schema.json").write_text(
        json.dumps({"schema": "other", "records": []}), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def app_syn(market_dir):
    return create_app(data_mode="synthetic", market_dir=market_dir)


def test_synthetic_always_present(market_dir):
    reg = build_registry(market_dir=market_dir)
    assert SYNTHETIC_ID in [e.id for e in reg.list_entries()]


def test_market_valid_listed(market_dir, tmp_path):
    # isolate showroom so only the tmp market_dir artifact appears as market
    missing_showroom = tmp_path / "no_showroom" / "x.json"
    reg = build_registry(market_dir=market_dir, showroom_file=missing_showroom)
    markets = [e for e in reg.list_entries() if e.mode == "market"]
    assert len(markets) == 1
    assert markets[0].country == "ar"
    assert markets[0].total_records == 8
    assert "/" not in markets[0].option_label()


def test_invalid_artifacts_ignored(market_dir):
    entries, _ = discover_market_entries(market_dir)
    assert len(entries) == 1


def test_order_by_retrieved_at(tmp_path):
    older = json.loads(FIXTURE.read_text(encoding="utf-8"))
    newer = dict(older)
    older["retrieved_at"] = "2026-01-01T00:00:00Z"
    newer["retrieved_at"] = "2026-08-10T19:00:00Z"
    (tmp_path / "old.json").write_text(json.dumps(older), encoding="utf-8")
    (tmp_path / "new.json").write_text(json.dumps(newer), encoding="utf-8")
    entries, _ = discover_market_entries(tmp_path)
    assert entries[0].retrieved_at.startswith("2026-08-10")


def test_ids_deterministic(market_dir):
    a = build_registry(market_dir=market_dir)
    b = build_registry(market_dir=market_dir)
    assert [e.id for e in a.list_entries()] == [e.id for e in b.list_entries()]


def test_resolve_synthetic(market_dir):
    reg = build_registry(market_dir=market_dir)
    ds = reg.resolve(SYNTHETIC_ID)
    assert ds.meta.mode == MODE_SYNTHETIC
    assert ds.pipeline_result.total_input == 17


def test_resolve_market(market_dir):
    reg = build_registry(market_dir=market_dir)
    mid = next(e.id for e in reg.list_entries() if e.mode == "market")
    ds = reg.resolve(mid)
    assert ds.meta.mode == "market"
    assert any(r.id.startswith("adzuna:") for r in ds.pipeline_result.records)


def test_resolve_unknown(market_dir):
    reg = build_registry(market_dir=market_dir)
    with pytest.raises(DatasetError):
        reg.resolve("market:does-not-exist")


def test_make_id_not_path():
    mid = make_market_dataset_id(
        country="ar",
        retrieved_at="2026-08-10T19:00:00Z",
        artifact_name="market_ar.json",
    )
    assert mid.startswith("market:")
    assert "/" not in mid


def test_default_synthetic_session(app_syn):
    c = app_syn.test_client()
    assert c.get("/").status_code == 200
    assert b"Synthetic" in c.get("/").data or b"sint" in c.get("/").data.lower()


def test_switch_synthetic_to_market(app_syn):
    reg = app_syn.config["DATASET_REGISTRY"]
    mid = next(e.id for e in reg.list_entries() if e.mode == "market")
    c = app_syn.test_client()
    assert b"job_001" in c.get("/jobs").data
    c.post("/dataset", data={"dataset_id": mid}, follow_redirects=True)
    assert b"adzuna:" in c.get("/jobs").data
    assert b"job_001" not in c.get("/jobs").data


def test_two_clients_independent(app_syn):
    reg = app_syn.config["DATASET_REGISTRY"]
    mid = next(e.id for e in reg.list_entries() if e.mode == "market")
    a = app_syn.test_client()
    b = app_syn.test_client()
    a.post("/dataset", data={"dataset_id": mid})
    assert b"adzuna:" in a.get("/jobs").data
    assert b"job_001" in b.get("/jobs").data


def test_reject_path_as_dataset_id(app_syn):
    c = app_syn.test_client()
    c.get("/")
    c.post("/dataset", data={"dataset_id": "../etc/passwd"}, follow_redirects=True)
    assert b"job_001" in c.get("/jobs").data


def test_reject_url_as_dataset_id(app_syn):
    c = app_syn.test_client()
    c.post("/dataset", data={"dataset_id": "https://evil.example/x"}, follow_redirects=True)
    assert b"job_001" in c.get("/jobs").data


def test_unknown_id_keeps_current(app_syn):
    c = app_syn.test_client()
    c.get("/")
    c.post("/dataset", data={"dataset_id": "market:xx:nope"}, follow_redirects=True)
    assert b"job_001" in c.get("/jobs").data


def test_analysis_follows_dataset(app_syn):
    clear_analysis_cache()
    app_syn.config["GENERATIVE_PROVIDER"] = FakeProvider()
    app_syn.config["GENERATIVE_AVAILABLE"] = True
    reg = app_syn.config["DATASET_REGISTRY"]
    mid = next(e.id for e in reg.list_entries() if e.mode == "market")
    c = app_syn.test_client()
    c.post("/analysis")
    c.post("/dataset", data={"dataset_id": mid}, follow_redirects=True)
    body = c.get("/analysis").data.decode("utf-8").lower()
    assert "todav" in body or "no hay" in body or "disponible" in body
    calls = {"n": 0}
    prov = app_syn.config["GENERATIVE_PROVIDER"]
    orig = prov.generate

    def counted(req):
        calls["n"] += 1
        return orig(req)

    prov.generate = counted  # type: ignore
    c.get("/analysis")
    assert calls["n"] == 0
    c.post("/analysis")
    assert calls["n"] == 1
    c.post("/dataset", data={"dataset_id": SYNTHETIC_ID}, follow_redirects=True)
    assert b"job_001" in c.get("/jobs").data


def test_e2e_demo_flow(app_syn):
    clear_analysis_cache()
    app_syn.config["GENERATIVE_PROVIDER"] = FakeProvider()
    app_syn.config["GENERATIVE_AVAILABLE"] = True
    reg = app_syn.config["DATASET_REGISTRY"]
    mid = next(e.id for e in reg.list_entries() if e.mode == "market")
    c = app_syn.test_client()
    assert b"job_001" in c.get("/jobs").data
    c.post("/dataset", data={"dataset_id": mid}, follow_redirects=True)
    assert b"adzuna:" in c.get("/jobs").data
    calls = {"n": 0}
    p = app_syn.config["GENERATIVE_PROVIDER"]
    orig = p.generate

    def counted(req):
        calls["n"] += 1
        return orig(req)

    p.generate = counted  # type: ignore
    c.get("/analysis")
    assert calls["n"] == 0
    assert c.post("/analysis").status_code == 200
    assert calls["n"] == 1
    c.post("/dataset", data={"dataset_id": SYNTHETIC_ID}, follow_redirects=True)
    assert b"job_001" in c.get("/jobs").data
