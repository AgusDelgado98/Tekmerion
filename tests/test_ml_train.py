"""Supervised training path (sklearn) — only on a sufficient *test* gold file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

sklearn = pytest.importorskip("sklearn")

from analysis.ml.compare import run_block_b
from analysis.ml.gold import load_gold_dataset
from analysis.ml.models import GOLD_SCHEMA, LABEL_SOURCE_HUMAN, RoleFamily
from analysis.ml.split import grouped_train_test_split
from analysis.ml.vectorize import JobTextSkillVectorizer, examples_to_x

_FAMILY_TEXT = {
    RoleFamily.DATA_ANALYST: (
        "Data Analyst",
        "SQL dashboards KPI reporting excel python product metrics token_data_analyst",
    ),
    RoleFamily.BI_ANALYST: (
        "Business Intelligence Analyst",
        "Power BI DAX dimensional model tableau reporting token_bi_analyst",
    ),
    RoleFamily.DATA_SCIENTIST: (
        "Data Scientist",
        "statistics scikit-learn experiments pandas nlp research token_data_scientist",
    ),
    RoleFamily.ML_ENGINEER: (
        "Machine Learning Engineer",
        "mlflow docker kubernetes fastapi production models token_ml_engineer",
    ),
    RoleFamily.AI_ANALYST: (
        "AI Analyst",
        "llm prompt evaluation generative systems token_ai_analyst",
    ),
    RoleFamily.DATA_ENGINEER: (
        "Data Engineer",
        "airflow dbt spark snowflake etl pipelines token_data_engineer",
    ),
    RoleFamily.BUSINESS_ANALYST: (
        "Business Analyst",
        "requirements process mapping functional documentation agile token_business_analyst",
    ),
}


def _write_sufficient_gold(path: Path, per_class: int = 15) -> Path:
    records = []
    for family, (title, desc) in _FAMILY_TEXT.items():
        for i in range(per_class):
            records.append(
                {
                    "id": f"{family.value}_{i:02d}",
                    "title": f"{title} {i}",
                    "description": f"{desc} variant {i}",
                    "company": "TestCo",
                    "gold_role_family": family.value,
                    "label_source": LABEL_SOURCE_HUMAN,
                    "annotator_id": "test.block_b",
                    "labeled_at": "2026-08-19",
                    "source_kind": "test_fixture",
                    "notes": "Synthetic fixture for training-path tests; not market data.",
                }
            )
    payload = {
        "schema": GOLD_SCHEMA,
        "task": "role_family_classification",
        "label_field": "gold_role_family",
        "label_source": LABEL_SOURCE_HUMAN,
        "label_policy": "test fixture",
        "limitations": ["test only"],
        "records": records,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_vectorizer_fit_on_train_only(tmp_path):
    gold = load_gold_dataset(_write_sufficient_gold(tmp_path / "g.json"))
    split = grouped_train_test_split(gold.records, seed=42, test_ratio=0.3)
    vec = JobTextSkillVectorizer(max_features=200)
    vec.fit(examples_to_x(split.train))
    X_test = vec.transform(examples_to_x(split.test))
    assert X_test.shape[0] == len(split.test)
    # Fitted vocabulary comes from train rows; transform still accepts test rows.
    assert vec.tfidf_ is not None


def test_train_and_compare_on_sufficient_fixture(tmp_path):
    gold_path = _write_sufficient_gold(tmp_path / "gold.json")
    report = run_block_b(
        gold_path,
        seed=42,
        test_ratio=0.3,
        report_path=tmp_path / "cmp.json",
        artifacts_dir=tmp_path / "art",
        harvest=False,
        allow_train=True,
    )
    assert report["status"] == "SUFFICIENT"
    assert report["training_ran"] is True
    names = {row["name"] for row in report["models"]}
    assert names == {"deterministic_role_family", "logreg", "linearsvc", "random_forest"}
    for row in report["models"]:
        assert "test_macro_f1" in row
        assert "test_accuracy" in row
    assert report["winner"]["criterion"].startswith("macro_f1")
    assert report["productive_integration"] == {
        "evidence": False,
        "pipeline": False,
        "flask": False,
    }
    kinds = {d["kind"] for d in report["models_detail"]}
    assert "baseline_rules" in kinds
    assert "ml_sklearn" in kinds
    # CV recorded only for sklearn models
    ml_detail = [d for d in report["models_detail"] if d["kind"] == "ml_sklearn"]
    assert all(d["cv_macro_f1"] is not None for d in ml_detail)
    assert all("confusion_matrix" in d["test"]["metrics"] for d in report["models_detail"])
    assert (tmp_path / "art" / "evaluation_manifest.json").is_file()
    assert (tmp_path / "art" / "evaluation_manifest_logreg.json").is_file()


def test_block_b_sufficient_gold_does_not_train_when_disallowed(tmp_path):
    gold_path = _write_sufficient_gold(tmp_path / "gold.json")
    report = run_block_b(
        gold_path,
        seed=42,
        test_ratio=0.3,
        report_path=tmp_path / "block_b.json",
        artifacts_dir=tmp_path / "art",
        harvest=True,
        allow_train=False,
    )
    assert report["status"] == "SUFFICIENT"
    assert report["training_ran"] is False
    assert report["models"][0]["name"] == "deterministic_role_family"
    assert len(report["models"]) == 1
    assert report["harvest"]["n_unique"] >= 1
    assert report["winner"]["promote_ml"] is False
    dumped = json.loads((tmp_path / "block_b.json").read_text(encoding="utf-8"))
    assert "source_url" not in json.dumps(dumped)


def _metrics(macro_f1: float, zero_f1_supported: bool = False) -> dict:
    per_class = {
        "data_analyst": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 3},
        "bi_analyst": {
            "precision": 0.0 if zero_f1_supported else 0.8,
            "recall": 0.0 if zero_f1_supported else 0.8,
            "f1": 0.0 if zero_f1_supported else 0.8,
            "support": 3,
        },
    }
    return {"macro_f1": macro_f1, "accuracy": 0.7, "per_class": per_class}


def test_promote_ml_requires_defensible_macro_f1_delta():
    from analysis.ml.compare import MIN_MACRO_F1_DELTA_TO_PROMOTE, _pick_winner

    rules = {
        "name": "deterministic_role_family",
        "kind": "baseline_rules",
        "test": {"metrics": _metrics(0.70)},
    }
    tiny = {
        "name": "logreg",
        "kind": "ml_sklearn",
        "test": {"metrics": _metrics(0.70 + MIN_MACRO_F1_DELTA_TO_PROMOTE / 2)},
    }
    assert _pick_winner([rules, tiny])["promote_ml"] is False
    strong = {
        "name": "logreg",
        "kind": "ml_sklearn",
        "test": {"metrics": _metrics(0.70 + MIN_MACRO_F1_DELTA_TO_PROMOTE + 0.01)},
    }
    assert _pick_winner([rules, strong])["promote_ml"] is True
    collapsed = {
        "name": "logreg",
        "kind": "ml_sklearn",
        "test": {"metrics": _metrics(0.90, zero_f1_supported=True)},
    }
    assert _pick_winner([rules, collapsed])["promote_ml"] is False

