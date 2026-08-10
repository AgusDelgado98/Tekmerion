"""V0.5.4 — Grounded role comparison (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.pipeline import process_file
from analysis.evidence import build_evidence
from analysis.generative.comparison import (
    build_role_comparison_grounding,
    validate_role_pair,
    available_roles,
    TASK_ROLE_COMPARISON,
)
from analysis.generative.models import AnalysisRequest, GenerativeError
from analysis.generative.providers import FakeProvider
from analysis.generative.validate import validate_role_comparison
from analysis.generative.service import run_role_comparison, clear_analysis_cache
from analysis.generative.prompts import ROLE_COMPARISON_PROMPT_VERSION
from app import create_app


FIXTURE = Path(__file__).parent / "fixtures" / "market" / "market_ar_fixture.json"


@pytest.fixture
def synthetic_ev():
    pipe = process_file(str(Path(__file__).resolve().parents[1] / "data" / "raw" / "sample_jobs.json"))
    return build_evidence(pipe.records), pipe.records


def test_available_roles(synthetic_ev):
    ev, _ = synthetic_ev
    roles = available_roles(ev)
    assert "data_analyst" in roles and "bi_analyst" in roles


def test_validate_pair_ok(synthetic_ev):
    ev, _ = synthetic_ev
    pair = validate_role_pair(ev, "data_analyst", "bi_analyst")
    assert pair.canonical() == ("bi_analyst", "data_analyst")


def test_validate_pair_same(synthetic_ev):
    ev, _ = synthetic_ev
    with pytest.raises(GenerativeError, match="distinct"):
        validate_role_pair(ev, "data_analyst", "data_analyst")


def test_validate_pair_unknown(synthetic_ev):
    ev, _ = synthetic_ev
    with pytest.raises(GenerativeError, match="Unknown"):
        validate_role_pair(ev, "data_analyst", "wizard")


def test_shared_and_exclusive(synthetic_ev):
    ev, records = synthetic_ev
    g = build_role_comparison_grounding(
        ev, "data_analyst", "bi_analyst",
        dataset_mode="synthetic", dataset_source="synthetic", dataset_label="S",
        records=records,
    )
    items = {i.id: i for i in g.items}
    shared = items["comparison.shared_skills"].value
    only_a = items["comparison.only_data_analyst"].value
    only_b = items["comparison.only_bi_analyst"].value
    assert "sql" in shared
    assert "python" in only_a
    assert "looker" in only_b


def test_grounding_deterministic(synthetic_ev):
    ev, records = synthetic_ev
    a = build_role_comparison_grounding(
        ev, "data_analyst", "bi_analyst",
        dataset_mode="synthetic", dataset_source="s", dataset_label="S", records=records,
    )
    b = build_role_comparison_grounding(
        ev, "data_analyst", "bi_analyst",
        dataset_mode="synthetic", dataset_source="s", dataset_label="S", records=records,
    )
    assert a.fingerprint() == b.fingerprint()


def test_fake_valid(synthetic_ev):
    ev, records = synthetic_ev
    clear_analysis_cache()
    result = run_role_comparison(
        evidence=ev, role_a="data_analyst", role_b="bi_analyst",
        dataset_mode="synthetic", dataset_source="s", dataset_label="S",
        provider=FakeProvider(), records=records,
    )
    assert result.task == TASK_ROLE_COMPARISON
    assert result.prompt_version == ROLE_COMPARISON_PROMPT_VERSION
    assert "sql" in result.shared_skills


def test_invented_shared_rejected(synthetic_ev):
    ev, _ = synthetic_ev
    g = build_role_comparison_grounding(
        ev, "data_analyst", "bi_analyst",
        dataset_mode="synthetic", dataset_source="s", dataset_label="S",
    )
    raw = FakeProvider(corrupt="invented_shared").generate(
        AnalysisRequest(grounding=g, task="role_comparison",
                        parameters={"role_a": "data_analyst", "role_b": "bi_analyst"})
    )
    with pytest.raises(GenerativeError, match="shared_skills"):
        validate_role_comparison(raw, g, "data_analyst", "bi_analyst")


def test_third_role_ref_rejected(synthetic_ev):
    ev, _ = synthetic_ev
    g = build_role_comparison_grounding(
        ev, "data_analyst", "bi_analyst",
        dataset_mode="synthetic", dataset_source="s", dataset_label="S",
    )
    raw = FakeProvider(corrupt="third_role_ref").generate(
        AnalysisRequest(grounding=g, task="role_comparison",
                        parameters={"role_a": "data_analyst", "role_b": "bi_analyst"})
    )
    with pytest.raises(GenerativeError):
        validate_role_comparison(raw, g, "data_analyst", "bi_analyst")


def test_bad_count_rejected(synthetic_ev):
    ev, _ = synthetic_ev
    g = build_role_comparison_grounding(
        ev, "data_analyst", "bi_analyst",
        dataset_mode="synthetic", dataset_source="s", dataset_label="S",
    )
    raw = FakeProvider(corrupt="bad_count").generate(
        AnalysisRequest(grounding=g, task="role_comparison",
                        parameters={"role_a": "data_analyst", "role_b": "bi_analyst"})
    )
    with pytest.raises(GenerativeError):
        validate_role_comparison(raw, g, "data_analyst", "bi_analyst")


def test_no_provider_call_on_invalid_roles(synthetic_ev):
    ev, _ = synthetic_ev
    calls = {"n": 0}

    class Counting(FakeProvider):
        def generate(self, request):
            calls["n"] += 1
            return super().generate(request)

    with pytest.raises(GenerativeError):
        run_role_comparison(
            evidence=ev, role_a="data_analyst", role_b="nope",
            dataset_mode="synthetic", dataset_source="s", dataset_label="S",
            provider=Counting(),
        )
    assert calls["n"] == 0


def test_cache_preserves_presentation_order(synthetic_ev):
    ev, records = synthetic_ev
    clear_analysis_cache()
    p = FakeProvider()
    a = run_role_comparison(
        evidence=ev, role_a="data_analyst", role_b="bi_analyst",
        dataset_mode="synthetic", dataset_source="s", dataset_label="S",
        provider=p, records=records,
    )
    b = run_role_comparison(
        evidence=ev, role_a="bi_analyst", role_b="data_analyst",
        dataset_mode="synthetic", dataset_source="s", dataset_label="S",
        provider=p, records=records,
    )
    assert a.role_a == "data_analyst"
    assert b.role_a == "bi_analyst"


def test_flask_get_no_provider():
    clear_analysis_cache()
    app = create_app(data_mode="synthetic")
    app.config["GENERATIVE_PROVIDER"] = FakeProvider()
    app.config["GENERATIVE_AVAILABLE"] = True
    calls = {"n": 0}
    orig = app.config["GENERATIVE_PROVIDER"].generate

    def counted(req):
        calls["n"] += 1
        return orig(req)

    app.config["GENERATIVE_PROVIDER"].generate = counted  # type: ignore
    c = app.test_client()
    assert c.get("/analysis/roles").status_code == 200
    assert calls["n"] == 0
    assert b"data_analyst" in c.get("/analysis/roles").data


def test_flask_same_roles_rejected():
    app = create_app(data_mode="synthetic")
    app.config["GENERATIVE_PROVIDER"] = FakeProvider()
    app.config["GENERATIVE_AVAILABLE"] = True
    c = app.test_client()
    rv = c.post("/analysis/roles", data={"role_a": "data_analyst", "role_b": "data_analyst"})
    body = rv.data.decode("utf-8").lower()
    assert "distint" in body or "error" in body


def test_flask_valid_comparison():
    clear_analysis_cache()
    app = create_app(data_mode="synthetic")
    app.config["GENERATIVE_PROVIDER"] = FakeProvider()
    app.config["GENERATIVE_AVAILABLE"] = True
    c = app.test_client()
    rv = c.post("/analysis/roles", data={"role_a": "data_analyst", "role_b": "bi_analyst"})
    assert rv.status_code == 200
    body = rv.data.decode("utf-8")
    assert "data_analyst" in body and "bi_analyst" in body
    assert "compartidas" in body.lower() or "sql" in body.lower()


def test_flask_dataset_switch_clears_scope(tmp_path):
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    (tmp_path / "m.json").write_text(json.dumps(data), encoding="utf-8")
    clear_analysis_cache()
    app = create_app(data_mode="synthetic", market_dir=tmp_path)
    app.config["GENERATIVE_PROVIDER"] = FakeProvider()
    app.config["GENERATIVE_AVAILABLE"] = True
    reg = app.config["DATASET_REGISTRY"]
    mid = next(e.id for e in reg.list_entries() if e.mode == "market")
    c = app.test_client()
    c.post("/analysis/roles", data={"role_a": "data_analyst", "role_b": "bi_analyst"})
    c.post("/dataset", data={"dataset_id": mid}, follow_redirects=True)
    body = c.get("/analysis/roles").data.decode("utf-8").lower()
    assert "todavía no hay" in body or "no hay una comparación" in body or "role a" in body
