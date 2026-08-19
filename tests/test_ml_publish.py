"""Public ML artifacts must not redistribute vacancy text or Adzuna ids."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.ml.publish import public_example_id, sanitize_public_payload

PUBLIC_JSON = [
    Path("data/ml/gold/evaluation_card.json"),
    Path("data/ml/reports/block_b.json"),
    Path("data/ml/reports/gold_expansion.json"),
    Path("data/ml/artifacts/evaluation_manifest.json"),
    Path("data/ml/artifacts/evaluation_manifest_logreg.json"),
    Path("data/ml/artifacts/evaluation_manifest_linearsvc.json"),
    Path("data/ml/artifacts/evaluation_manifest_random_forest.json"),
]
FORBIDDEN_SNIPPETS = (
    "adzuna.co.uk",
    "adzuna.com/jobs",
    "adzuna_snapshot:",
    "ADZUNA_APP",
    "ADZUNA_API_KEY",
    "utm_source=",
)


def test_public_example_id_is_stable_and_anonymous():
    assert public_example_id("adzuna_snapshot:123") == public_example_id("adzuna_snapshot:123")
    assert public_example_id("adzuna_snapshot:123").startswith("ex_")
    hashed = public_example_id("adzuna_snapshot:123")
    assert public_example_id(hashed) == hashed


def test_sanitize_drops_vacancy_fields():
    raw = {
        "id": "adzuna_snapshot:99",
        "title": "Secret Title",
        "company": "Acme",
        "description": "SQL dashboards…",
        "source_url": "https://www.adzuna.co.uk/jobs/details/99",
        "metrics": {"macro_f1": 0.866},
        "train_ids": ["adzuna_snapshot:99"],
        "credentials": {"available": True},
    }
    out = sanitize_public_payload(raw)
    assert "title" not in out
    assert "company" not in out
    assert "description" not in out
    assert "source_url" not in out
    assert "credentials" not in out
    assert out["metrics"]["macro_f1"] == 0.866
    assert out["id"].startswith("ex_")
    assert out["train_ids"][0].startswith("ex_")


def test_published_ml_json_has_no_vacancy_text_or_adzuna_ids():
    for path in PUBLIC_JSON:
        text = path.read_text(encoding="utf-8")
        for snippet in FORBIDDEN_SNIPPETS:
            assert snippet not in text, f"{path} contains {snippet!r}"
        payload = json.loads(text)
        blob = json.dumps(payload)
        assert '"source_url"' not in blob
        assert '"source_record_id"' not in blob


def test_evaluation_card_matches_frozen_benchmark():
    card = json.loads(Path("data/ml/gold/evaluation_card.json").read_text(encoding="utf-8"))
    assert card["n"] == 159
    assert card["n_train"] == 112
    assert card["n_test"] == 47
    assert card["redistributed_gold"] is False
    assert card["promote_ml"] is False
    assert card["dataset_sha256"].startswith("84ec0885")
    assert sum(card["class_distribution"].values()) == 159
    assert min(card["class_distribution"].values()) >= 10
    m = card["metrics"]
    assert m["deterministic_role_family"]["test_macro_f1"] == 0.866
    assert m["linearsvc"]["test_macro_f1"] == 0.816
    assert m["random_forest"]["test_macro_f1"] == 0.749
    assert m["logreg"]["test_macro_f1"] == 0.746
    assert "Adzuna API" in card["attribution"]
    assert card["attribution_url"].startswith("https://developer.adzuna.com")


def test_block_b_metrics_unchanged_after_sanitize():
    report = json.loads(Path("data/ml/reports/block_b.json").read_text(encoding="utf-8"))
    by_name = {row["name"]: row for row in report["models"]}
    assert round(by_name["deterministic_role_family"]["test_macro_f1"], 3) == 0.866
    assert round(by_name["linearsvc"]["test_macro_f1"], 3) == 0.816
    assert round(by_name["random_forest"]["test_macro_f1"], 3) == 0.749
    assert round(by_name["logreg"]["test_macro_f1"], 3) == 0.746
    assert report["winner"]["promote_ml"] is False
    assert report["dataset_n"] == 159
    assert report["split"]["n_train"] == 112
    assert report["split"]["n_test"] == 47
    assert report["dataset_sha256"].startswith("84ec0885")
    assert Path("docs/assets/05-rules-vs-ml.png").is_file()
