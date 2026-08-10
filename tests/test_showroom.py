"""V0.6.0 — Showroom registry and demo smoke (offline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.registry import (
    SHOWROOM_ID,
    SHOWROOM_FILE,
    SYNTHETIC_ID,
    build_registry,
    load_showroom_entry,
)
from analysis.generative.providers import FakeProvider
from analysis.generative.service import clear_analysis_cache
from app import create_app


def test_showroom_file_exists():
    assert SHOWROOM_FILE.exists()


def test_showroom_entry_metadata():
    entry, path = load_showroom_entry()
    assert entry is not None and path is not None
    assert entry.id == SHOWROOM_ID
    assert entry.is_showroom is True
    assert entry.dataset_kind == "showroom"
    assert entry.total_records and entry.total_records >= 10
    assert "Showroom" in entry.option_label()
    assert "/" not in entry.option_label()


def test_registry_includes_showroom():
    reg = build_registry()
    ids = [e.id for e in reg.list_entries()]
    assert SYNTHETIC_ID in ids
    assert SHOWROOM_ID in ids
    # order: synthetic first, showroom before other markets ideally
    assert ids.index(SYNTHETIC_ID) == 0
    assert ids.index(SHOWROOM_ID) == 1


def test_resolve_showroom():
    reg = build_registry()
    ds = reg.resolve(SHOWROOM_ID)
    assert ds.meta.mode == "market"
    assert "Showroom" in ds.meta.label
    assert ds.evidence.n_analysis_records >= 1
    roles = set(ds.evidence.skills_by_role.keys())
    assert "data_analyst" in roles
    assert len(roles) >= 4


def test_showroom_default_mode():
    app = create_app(data_mode="showroom")
    assert app.config["DEFAULT_DATASET_ID"] == SHOWROOM_ID
    c = app.test_client()
    body = c.get("/").data.decode("utf-8")
    assert "Showroom" in body or "showroom" in body.lower()


def test_demo_smoke_pages():
    clear_analysis_cache()
    app = create_app(data_mode="showroom")
    app.config["GENERATIVE_PROVIDER"] = FakeProvider()
    app.config["GENERATIVE_AVAILABLE"] = True
    c = app.test_client()
    for path in ("/", "/jobs", "/skills", "/roles", "/analysis", "/analysis/roles"):
        rv = c.get(path)
        assert rv.status_code == 200, path
    # charts present on home
    assert b"bar-chart" in c.get("/").data or b"Distribuci" in c.get("/").data
    # AI generate
    rv = c.post("/analysis")
    assert rv.status_code == 200
    assert b"Hallazgos" in rv.data or b"hallazgos" in rv.data.lower() or b"summary" in rv.data.lower()
    # role comparison
    roles_page = c.get("/analysis/roles").data.decode("utf-8")
    # pick two roles that exist in showroom
    rv2 = c.post(
        "/analysis/roles",
        data={"role_a": "data_analyst", "role_b": "bi_analyst"},
    )
    assert rv2.status_code == 200
    body = rv2.data.decode("utf-8").lower()
    assert "data_analyst" in body and "bi_analyst" in body
