"""
V0.5.2 — Ranking claim guardrails (offline).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.grounding import build_grounding_payload
from analysis.generative.models import GeneratedAnalysis, GenerativeError, Finding
from analysis.generative.numeric import extract_numeric_claims
from analysis.generative.ranking import (
    build_ranking_index,
    extract_ranking_claims,
    validate_ranking_claims,
    UnsupportedRankingClaim,
    normalize_item,
)
from analysis.generative.validate import validate_generated_analysis
from analysis.generative.prompts import PROMPT_VERSION
from analysis.generative.service import run_market_summary, clear_analysis_cache
from analysis.generative.providers import FakeProvider
from app.dataset import load_app_dataset
from app import create_app


MARKET_FIXTURE = Path(__file__).parent / "fixtures" / "market" / "market_ar_fixture.json"
CORPUS = Path(__file__).parent / "fixtures" / "generative" / "ranking_corpus.json"


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
def ranking_index(market_grounding):
    return build_ranking_index(market_grounding)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def test_ranking_tables_present(ranking_index):
    assert "skills.ranking" in ranking_index.tables
    assert "roles.ranking" in ranking_index.tables
    assert "seniority.ranking" in ranking_index.tables


def test_ranking_positions(ranking_index):
    sk = ranking_index.tables["skills.ranking"]
    assert sk.get("python").position == 1
    assert sk.get("sql").position == 2
    roles = ranking_index.tables["roles.ranking"]
    assert roles.get("bi_analyst").position == 1


def test_tie_flags(ranking_index):
    sk = ranking_index.tables["skills.ranking"]
    assert sk.get("python").tied_for_position is True
    assert sk.get("python").is_unique_leader is False
    assert sk.get("sql").is_unique_leader is False


def test_normalize_item():
    assert normalize_item("BI Analyst") == "bi_analyst"
    assert normalize_item("python") == "python"


def test_ranking_index_deterministic(market_grounding):
    a = build_ranking_index(market_grounding)
    b = build_ranking_index(market_grounding)
    assert a.tables["skills.ranking"].by_item.keys() == b.tables["skills.ranking"].by_item.keys()


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def test_parse_hash_and_rank():
    assert extract_ranking_claims("Python es #1")[0].rank == 1
    assert extract_ranking_claims("SQL rank 2")[0].item.lower() == "sql"
    assert extract_ranking_claims("bi_analyst ocupa el puesto 1")[0].rank == 1
    assert extract_ranking_claims("SQL ocupa la posición 2")[0].rank == 2


def test_parse_ordinals():
    c = extract_ranking_claims("Python es el primero")
    assert c and c[0].rank == 1
    c2 = extract_ranking_claims("SQL es el segundo")
    assert c2 and c2[0].rank == 2


def test_parse_superlative():
    c = extract_ranking_claims("Python es la skill más frecuente")
    assert c and c[0].claim_type == "superlative"


def test_hash1_not_numeric_count():
    text = "Python es #1 y aparece en 5 vacantes"
    ranks = extract_ranking_claims(text)
    nums = extract_numeric_claims(text)
    assert any(r.rank == 1 for r in ranks)
    assert all(n.value != 1 for n in nums)  # #1 must not become count 1
    assert any(n.value == 5 for n in nums)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_position_correct(ranking_index):
    stats = validate_ranking_claims(
        "Python es #1",
        ranking_index,
        location="key_findings[0]",
        mode="finding",
        evidence_refs=["skills.ranking"],
    )
    assert stats.ranking_claims_supported == 1


def test_position_incorrect(ranking_index):
    with pytest.raises(UnsupportedRankingClaim) as exc:
        validate_ranking_claims(
            "Python es #2",
            ranking_index,
            location="key_findings[0]",
            mode="finding",
            evidence_refs=["skills.ranking"],
        )
    assert exc.value.reason == "rank_mismatch"


def test_wrong_ref(ranking_index):
    with pytest.raises(UnsupportedRankingClaim) as exc:
        validate_ranking_claims(
            "Python es #1",
            ranking_index,
            location="key_findings[0]",
            mode="finding",
            evidence_refs=["roles.ranking"],
        )
    assert "item_not_in" in exc.value.reason


def test_superlative_tie_rejected(ranking_index):
    with pytest.raises(UnsupportedRankingClaim) as exc:
        validate_ranking_claims(
            "Python es la skill más frecuente",
            ranking_index,
            location="key_findings[0]",
            mode="finding",
            evidence_refs=["skills.ranking"],
        )
    assert exc.value.reason == "tied_or_not_leader"


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def test_ranking_corpus(market_grounding):
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    for case in corpus:
        analysis = GeneratedAnalysis.from_dict(case["analysis"])
        analysis.prompt_version = PROMPT_VERSION
        if case["expect"] == "pass":
            out = validate_generated_analysis(analysis, market_grounding)
            assert out is not None
        else:
            with pytest.raises(GenerativeError):
                validate_generated_analysis(analysis, market_grounding)


# ---------------------------------------------------------------------------
# Fake + coexistence + Flask
# ---------------------------------------------------------------------------

def test_fake_valid_under_v3(market_grounding):
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
    assert result.prompt_version == "market_summary.v3"
    rs = getattr(result, "_ranking_stats", None)
    assert rs is not None
    assert rs.ranking_claims_supported == rs.ranking_claims_found


def test_rank_plus_percent_ok(market_grounding):
    analysis = GeneratedAnalysis(
        summary="Resumen de 8 vacantes.",
        key_findings=[
            Finding(
                text="Python es #1 y aparece en 62.5% de las vacantes.",
                evidence_refs=["skills.ranking", "skills.python.pct"],
            )
        ],
        limitations=["Muestra de 8 vacantes."],
        evidence_refs=["skills.ranking", "skills.python.pct"],
    )
    out = validate_generated_analysis(analysis, market_grounding)
    assert out._ranking_stats.ranking_claims_supported >= 1


def test_rank_ok_number_bad(market_grounding):
    analysis = GeneratedAnalysis(
        summary="Resumen de 8 vacantes.",
        key_findings=[
            Finding(
                text="Python es #1 y aparece en 6 vacantes.",
                evidence_refs=["skills.ranking", "skills.python.count"],
            )
        ],
        limitations=["Muestra de 8 vacantes."],
        evidence_refs=["skills.ranking"],
    )
    with pytest.raises(GenerativeError, match="Unsupported numeric claim"):
        validate_generated_analysis(analysis, market_grounding)


def test_flask_rejects_bad_rank():
    clear_analysis_cache()
    app = create_app(data_mode="market", market_file=str(MARKET_FIXTURE))

    class BadRankFake(FakeProvider):
        def generate(self, request):
            return GeneratedAnalysis(
                summary="Resumen de 8 vacantes.",
                key_findings=[
                    Finding(
                        text="Python es #2.",
                        evidence_refs=["skills.ranking"],
                    )
                ],
                limitations=["Muestra de 8 vacantes."],
                evidence_refs=["skills.ranking"],
                prompt_version=PROMPT_VERSION,
                provider="fake",
                model="bad-rank",
            )

    provider = BadRankFake()
    calls = {"n": 0}
    orig = provider.generate

    def counted(req):
        calls["n"] += 1
        return orig(req)

    provider.generate = counted  # type: ignore
    app.config["GENERATIVE_PROVIDER"] = provider
    app.config["GENERATIVE_AVAILABLE"] = True
    c = app.test_client()
    rv = c.post("/analysis")
    assert rv.status_code == 200
    body = rv.data.decode("utf-8").lower()
    assert "unsupported" in body or "ranking" in body or "claim" in body
    assert calls["n"] == 1
    c.get("/analysis")
    assert calls["n"] == 1
