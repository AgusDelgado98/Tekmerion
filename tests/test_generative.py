"""
Offline tests for grounded generative analysis (V0.5.0).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.evidence import build_evidence
from analysis.pipeline import process_file
from analysis.grounding import build_grounding_payload, GroundingPayload
from analysis.generative.models import AnalysisRequest, GeneratedAnalysis, Finding, GenerativeError
from analysis.generative.prompts import PROMPT_VERSION, build_market_summary_messages, SYSTEM_INSTRUCTIONS
from analysis.generative.providers import (
    DisabledProvider,
    FakeProvider,
    get_provider_from_env,
    ENV_PROVIDER,
    ENV_API_KEY,
)
from analysis.generative.validate import validate_generated_analysis
from analysis.generative.service import run_market_summary, clear_analysis_cache
from app import create_app
from app.dataset import load_app_dataset


MARKET_FIXTURE = Path(__file__).parent / "fixtures" / "market" / "market_ar_fixture.json"


@pytest.fixture
def synthetic_grounding():
    pipe = process_file("data/raw/sample_jobs.json")
    ev = build_evidence(pipe.records)
    return build_grounding_payload(
        ev,
        dataset_mode="synthetic",
        dataset_source="synthetic",
        dataset_label="Synthetic sample",
    )


@pytest.fixture
def market_grounding():
    ds = load_app_dataset(mode="market", market_file=str(MARKET_FIXTURE))
    return build_grounding_payload(
        ds.evidence,
        dataset_mode=ds.meta.mode,
        dataset_source=ds.meta.source,
        dataset_label=ds.meta.label,
        retrieved_at=ds.meta.retrieved_at,
        country=ds.meta.country,
        query_count=ds.meta.query_count,
    )


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------

def test_grounding_from_synthetic(synthetic_grounding):
    g = synthetic_grounding
    assert g.n_analysis_records >= 1
    assert g.dataset_mode == "synthetic"
    assert "dataset.n_analysis_records" in g.item_ids()
    assert "skills.ranking" in g.item_ids()
    assert "roles.ranking" in g.item_ids()


def test_grounding_market_metadata(market_grounding):
    assert market_grounding.dataset_mode == "market"
    assert market_grounding.country == "ar"
    assert market_grounding.retrieved_at
    assert "dataset.country" in market_grounding.item_ids()


def test_grounding_serialization_deterministic(synthetic_grounding):
    a = synthetic_grounding.to_json()
    b = synthetic_grounding.to_json()
    assert a == b
    assert synthetic_grounding.fingerprint() == synthetic_grounding.fingerprint()


def test_grounding_unique_ids(synthetic_grounding):
    ids = [i.id for i in synthetic_grounding.items]
    assert len(ids) == len(set(ids))


def test_grounding_skill_counts_match_evidence():
    pipe = process_file("data/raw/sample_jobs.json")
    ev = build_evidence(pipe.records)
    g = build_grounding_payload(
        ev,
        dataset_mode="synthetic",
        dataset_source="synthetic",
        dataset_label="Synthetic",
    )
    items = {i.id: i for i in g.items}
    top = ev.skill_frequency[0]
    safe = top["item"].replace(" ", "_")
    assert items[f"skills.{safe}.count"].value == top["count"]


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

def test_disabled_provider():
    p = DisabledProvider()
    assert not p.is_available()
    g = build_grounding_payload(
        build_evidence(process_file("data/raw/sample_jobs.json").records),
        dataset_mode="synthetic",
        dataset_source="synthetic",
        dataset_label="S",
    )
    with pytest.raises(GenerativeError):
        p.generate(AnalysisRequest(grounding=g))


def test_fake_provider_valid(market_grounding):
    clear_analysis_cache()
    analysis = FakeProvider().generate(AnalysisRequest(grounding=market_grounding))
    validated = validate_generated_analysis(analysis, market_grounding)
    assert validated.summary
    assert len(validated.key_findings) >= 1
    assert validated.limitations
    assert validated.prompt_version == PROMPT_VERSION
    assert all(ref in market_grounding.item_ids() for ref in validated.evidence_refs)


def test_fake_corrupt_bad_ref(market_grounding):
    raw = FakeProvider(corrupt="bad_ref").generate(AnalysisRequest(grounding=market_grounding))
    with pytest.raises(GenerativeError, match="Unknown evidence_ref"):
        validate_generated_analysis(raw, market_grounding)


def test_fake_corrupt_empty(market_grounding):
    raw = FakeProvider(corrupt="empty").generate(AnalysisRequest(grounding=market_grounding))
    with pytest.raises(GenerativeError):
        validate_generated_analysis(raw, market_grounding)


def test_secrets_not_in_provider_repr(monkeypatch):
    monkeypatch.setenv(ENV_API_KEY, "sk-secret-value-should-not-leak")
    monkeypatch.setenv(ENV_PROVIDER, "openai_compatible")
    from analysis.generative.providers import OpenAICompatibleProvider
    p = OpenAICompatibleProvider(api_key="sk-secret-value-should-not-leak")
    assert "sk-secret" not in repr(p)
    assert "***" in repr(p)


def test_get_provider_missing_key(monkeypatch):
    monkeypatch.setenv(ENV_PROVIDER, "openai_compatible")
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    with pytest.raises(GenerativeError, match=ENV_API_KEY):
        get_provider_from_env()


def test_get_provider_disabled(monkeypatch):
    monkeypatch.setenv(ENV_PROVIDER, "disabled")
    p = get_provider_from_env()
    assert isinstance(p, DisabledProvider)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def test_prompt_version_and_anti_hallucination(market_grounding):
    msgs = build_market_summary_messages(market_grounding)
    assert PROMPT_VERSION
    assert msgs[0]["role"] == "system"
    assert "ONLY use facts" in msgs[0]["content"] or "ONLY" in msgs[0]["content"]
    assert "invent" in msgs[0]["content"].lower() or "NOT" in msgs[0]["content"]
    assert market_grounding.to_dict()["dataset"]["mode"] in msgs[1]["content"]


# ---------------------------------------------------------------------------
# Service + validation numbers
# ---------------------------------------------------------------------------

def test_run_market_summary_fake(market_grounding):
    clear_analysis_cache()
    ds = load_app_dataset(mode="market", market_file=str(MARKET_FIXTURE))
    result = run_market_summary(
        evidence=ds.evidence,
        dataset_mode=ds.meta.mode,
        dataset_source=ds.meta.source,
        dataset_label=ds.meta.label,
        provider=FakeProvider(),
        retrieved_at=ds.meta.retrieved_at,
        country=ds.meta.country,
        query_count=ds.meta.query_count,
    )
    assert result.summary
    assert "14" in result.limitations[0] or str(ds.evidence.n_analysis_records) in result.limitations[0] or "vacantes" in result.limitations[0].lower()


def test_reject_unknown_percentage_in_text(market_grounding):
    bad = GeneratedAnalysis(
        summary="Growth is 87.5% year over year.",
        key_findings=[
            Finding(text="SQL is important.", evidence_refs=["skills.ranking"])
        ],
        limitations=["small sample"],
        evidence_refs=["skills.ranking"],
    )
    # skills.ranking exists; 87.5% does not
    with pytest.raises(GenerativeError, match="Unsupported numeric claim"):
        validate_generated_analysis(bad, market_grounding)


# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------

def test_flask_analysis_disabled():
    app = create_app(data_mode="synthetic")
    # force disabled
    from analysis.generative.providers import DisabledProvider
    app.config["GENERATIVE_PROVIDER"] = DisabledProvider()
    app.config["GENERATIVE_AVAILABLE"] = False
    c = app.test_client()
    rv = c.get("/analysis")
    assert rv.status_code == 200
    assert b"desactivado" in rv.data or b"disabled" in rv.data.lower()
    # POST should not crash
    rv2 = c.post("/analysis")
    assert rv2.status_code == 200
    assert b"no configurado" in rv2.data.lower() or b"desactivado" in rv2.data.lower()


def test_flask_analysis_fake_post():
    clear_analysis_cache()
    app = create_app(data_mode="market", market_file=str(MARKET_FIXTURE))
    app.config["GENERATIVE_PROVIDER"] = FakeProvider()
    app.config["GENERATIVE_AVAILABLE"] = True
    c = app.test_client()
    # GET does not generate
    rv = c.get("/analysis")
    assert rv.status_code == 200
    assert b"Todav" in rv.data or b"no hay un an" in rv.data.lower() or b"disponible" in rv.data.lower()
    # POST generates
    rv2 = c.post("/analysis")
    assert rv2.status_code == 200
    body = rv2.data.decode("utf-8")
    assert "Hallazgos" in body or "hallazgos" in body.lower()
    assert "Limitaciones" in body
    assert "skills." in body or "roles." in body or "dataset." in body


def test_flask_get_does_not_call_provider():
    clear_analysis_cache()
    calls = {"n": 0}

    class CountingFake(FakeProvider):
        def generate(self, request):
            calls["n"] += 1
            return super().generate(request)

    app = create_app(data_mode="synthetic")
    app.config["GENERATIVE_PROVIDER"] = CountingFake()
    app.config["GENERATIVE_AVAILABLE"] = True
    c = app.test_client()
    c.get("/analysis")
    c.get("/analysis")
    assert calls["n"] == 0
    c.post("/analysis")
    assert calls["n"] == 1


def test_flask_synthetic_analysis_mentions_synthetic():
    clear_analysis_cache()
    app = create_app(data_mode="synthetic")
    app.config["GENERATIVE_PROVIDER"] = FakeProvider()
    app.config["GENERATIVE_AVAILABLE"] = True
    c = app.test_client()
    body = c.post("/analysis").data.decode("utf-8").lower()
    assert "sint" in body
