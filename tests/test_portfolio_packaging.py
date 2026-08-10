"""V0.7.0 — portfolio packaging smoke (no network)."""

from pathlib import Path

from app import create_app
from app.registry import SHOWROOM_ID, SHOWROOM_FILE
from analysis.generative.providers import FakeProvider, get_provider_from_env
from analysis.generative.service import clear_analysis_cache


def test_license_exists():
    assert Path("LICENSE").exists()
    assert "MIT" in Path("LICENSE").read_text(encoding="utf-8")


def test_showroom_asset_versioned():
    assert SHOWROOM_FILE.exists()


def test_demo_scripts_exist():
    assert Path("scripts/run_demo.ps1").exists()
    assert Path("scripts/run_demo.sh").exists()


def test_case_study_exists():
    assert Path("docs/case-study.md").exists()


def test_assets_readme():
    assert Path("docs/assets/README.md").exists()


def test_fake_provider_explicit():
    p = get_provider_from_env(provider_name="fake")
    assert p.name == "fake"
    assert p.is_available()


def test_disabled_default_safe(monkeypatch):
    monkeypatch.delenv("TEKMERION_LLM_PROVIDER", raising=False)
    p = get_provider_from_env()
    assert p.name == "disabled" or not p.is_available()


def test_showroom_demo_smoke():
    clear_analysis_cache()
    app = create_app(data_mode="showroom")
    assert app.config["DEFAULT_DATASET_ID"] == SHOWROOM_ID
    app.config["GENERATIVE_PROVIDER"] = FakeProvider()
    app.config["GENERATIVE_AVAILABLE"] = True
    c = app.test_client()
    for path in ("/", "/jobs", "/skills", "/roles", "/analysis", "/analysis/roles"):
        assert c.get(path).status_code == 200
    body = c.get("/analysis").data.decode("utf-8")
    assert "fake" in body
    # badge language
    assert "Demo" in body or "demo" in body.lower() or "determinista" in body.lower()
    assert c.post("/analysis").status_code == 200
    assert c.post(
        "/analysis/roles",
        data={"role_a": "data_analyst", "role_b": "bi_analyst"},
    ).status_code == 200
