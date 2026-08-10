"""
V0.5.1 — Quantitative claim guardrails (offline).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.grounding import build_grounding_payload
from analysis.generative.models import GeneratedAnalysis, GenerativeError, Finding
from analysis.generative.numeric import (
    build_numeric_index,
    extract_numeric_claims,
    validate_claims_against_index,
    UnsupportedNumericClaim,
)
from analysis.generative.validate import validate_generated_analysis
from analysis.generative.prompts import PROMPT_VERSION
from analysis.generative.service import run_market_summary, clear_analysis_cache
from analysis.generative.providers import FakeProvider
from app.dataset import load_app_dataset
from app import create_app


MARKET_FIXTURE = Path(__file__).parent / "fixtures" / "market" / "market_ar_fixture.json"
CORPUS = Path(__file__).parent / "fixtures" / "generative" / "corrupt_corpus.json"


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


@pytest.fixture
def index(market_grounding):
    return build_numeric_index(market_grounding)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def test_index_counts_and_percents(index, market_grounding):
    assert index.dataset_size == 8.0
    assert 62.5 in index.percent_values
    assert 5.0 in index.count_values
    assert "skills.python.count" in index.count_values[5.0]
    assert index.values_for_refs(["skills.python.count"], unit="count") == {5.0}


def test_index_deterministic(market_grounding):
    a = build_numeric_index(market_grounding)
    b = build_numeric_index(market_grounding)
    assert a.percent_values == b.percent_values
    assert a.count_values == b.count_values


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def test_parse_percent_dot_and_comma():
    claims = extract_numeric_claims("Python en 62.5% y SQL en 62,5 % de los casos")
    pcts = [c for c in claims if c.unit == "percent"]
    assert {c.value for c in pcts} == {62.5}


def test_parse_ratio():
    claims = extract_numeric_claims("aparece en 5 de 8 vacantes")
    assert any(c.kind == "ratio_part" and c.value == 5 for c in claims)
    assert any(c.kind == "ratio_whole" and c.value == 8 for c in claims)


def test_parse_count_noun():
    claims = extract_numeric_claims("hay 8 vacantes y 3 roles en total")
    assert any(c.value == 8 and c.kind == "count" for c in claims)
    assert any(c.value == 3 and c.kind == "count" for c in claims)


def test_ignore_year_and_timestamp():
    text = "Snapshot 2026-08-10T19:00:00Z del año 2026 con 8 vacantes"
    claims = extract_numeric_claims(text)
    values = {c.value for c in claims}
    assert 2026 not in values
    assert 8.0 in values


def test_ignore_model_and_prompt_version():
    text = "Generado con gpt-4o-mini y prompt market_summary.v2 sobre 8 registros"
    claims = extract_numeric_claims(text)
    assert all(c.value == 8 for c in claims)


# ---------------------------------------------------------------------------
# Finding validation
# ---------------------------------------------------------------------------

def test_finding_correct_count_and_pct(index):
    text = "Python aparece en 5 de 8 vacantes (62.5%)."
    refs = ["skills.python.count", "skills.python.pct", "dataset.n_analysis_records"]
    stats = validate_claims_against_index(
        text, index, location="key_findings[0]", mode="finding", evidence_refs=refs
    )
    assert stats.numeric_claims_found >= 3
    assert stats.numeric_claims_supported == stats.numeric_claims_found
    assert stats.numeric_claims_rejected == 0


def test_finding_wrong_count(index):
    with pytest.raises(UnsupportedNumericClaim) as exc:
        validate_claims_against_index(
            "Python aparece en 6 vacantes.",
            index,
            location="key_findings[0]",
            mode="finding",
            evidence_refs=["skills.python.count"],
        )
    assert exc.value.value == 6
    assert "count" in exc.value.reason


def test_finding_wrong_percent(index):
    with pytest.raises(UnsupportedNumericClaim):
        validate_claims_against_index(
            "Python en 99% de las vacantes.",
            index,
            location="key_findings[0]",
            mode="finding",
            evidence_refs=["skills.python.pct"],
        )


def test_finding_wrong_denominator(index):
    with pytest.raises(UnsupportedNumericClaim) as exc:
        validate_claims_against_index(
            "Python aparece en 5 de 10 vacantes.",
            index,
            location="key_findings[0]",
            mode="finding",
            evidence_refs=["skills.python.count", "dataset.n_analysis_records"],
        )
    assert exc.value.value == 10


def test_summary_global_invented(index):
    with pytest.raises(UnsupportedNumericClaim):
        validate_claims_against_index(
            "El mercado creció un 41.2% este año.",
            index,
            location="summary",
            mode="global",
        )


def test_summary_global_valid(index):
    stats = validate_claims_against_index(
        "Análisis sobre 8 vacantes del snapshot.",
        index,
        location="summary",
        mode="global",
    )
    assert stats.numeric_claims_rejected == 0


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def test_corrupt_corpus(market_grounding):
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    results = []
    for case in corpus:
        analysis = GeneratedAnalysis.from_dict(case["analysis"])
        analysis.prompt_version = PROMPT_VERSION
        analysis.provider = "corpus"
        analysis.model = "fixture"
        if case["expect"] == "pass":
            out = validate_generated_analysis(analysis, market_grounding)
            stats = getattr(out, "_claim_stats", None)
            results.append((case["id"], "pass", stats.to_dict() if stats else None))
        else:
            with pytest.raises((GenerativeError, UnsupportedNumericClaim)):
                validate_generated_analysis(analysis, market_grounding)
            results.append((case["id"], "fail", None))
    # all cases executed
    assert len(results) == len(corpus)
    assert any(r[0] == "valid_full" and r[1] == "pass" for r in results)


# ---------------------------------------------------------------------------
# Fake + prompt version + Flask
# ---------------------------------------------------------------------------

def test_fake_still_valid_under_v2(market_grounding):
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
    assert result.prompt_version == PROMPT_VERSION
    assert PROMPT_VERSION == "market_summary.v3"
    stats = getattr(result, "_claim_stats", None)
    if stats:
        assert stats.numeric_claims_supported == stats.numeric_claims_found


def test_flask_rejects_invalid_numeric_analysis():
    clear_analysis_cache()
    app = create_app(data_mode="market", market_file=str(MARKET_FIXTURE))

    class BadFake(FakeProvider):
        def generate(self, request):
            return GeneratedAnalysis(
                summary="Mercado en auge.",
                key_findings=[
                    Finding(
                        text="Python aparece en 6 de 8 vacantes (62.5%).",
                        evidence_refs=[
                            "skills.python.count",
                            "skills.python.pct",
                            "dataset.n_analysis_records",
                        ],
                    )
                ],
                limitations=["Muestra de 8 vacantes."],
                evidence_refs=["skills.python.count"],
                prompt_version=PROMPT_VERSION,
                provider="fake",
                model="bad",
            )

    app.config["GENERATIVE_PROVIDER"] = BadFake()
    app.config["GENERATIVE_AVAILABLE"] = True
    c = app.test_client()
    calls = {"n": 0}
    original = app.config["GENERATIVE_PROVIDER"].generate

    def counting(req):
        calls["n"] += 1
        return original(req)

    app.config["GENERATIVE_PROVIDER"].generate = counting  # type: ignore
    rv = c.post("/analysis")
    assert rv.status_code == 200
    body = rv.data.decode("utf-8").lower()
    assert "unsupported" in body or "error" in body or "claim" in body
    assert "hallazgos" not in body or "python aparece en 6" not in body
    assert calls["n"] == 1  # no auto-retry
    # second GET must not call again
    c.get("/analysis")
    assert calls["n"] == 1
