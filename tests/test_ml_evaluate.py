"""Shared evaluator, metrics, baseline, and reproducible manifests."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.ml.baseline import DeterministicRoleFamilyBaseline
from analysis.ml.evaluate import evaluate_gold_dataset, evaluate_predictor
from analysis.ml.gold import load_gold_dataset
from analysis.ml.metrics import compute_classification_metrics
from analysis.ml.models import MANIFEST_SCHEMA, Predictor, RoleFamilyExample, role_family_label_order
from analysis.models import RoleFamily

SMALL = Path("tests/fixtures/ml/gold_small.json")


class ConstantPredictor(Predictor):
    name = "constant_data_analyst"
    kind = "ml_stub"

    def predict(self, examples):
        return ["data_analyst"] * len(examples)


def test_metrics_perfect_scores():
    y = ["data_analyst", "bi_analyst", "data_analyst"]
    labels = ("data_analyst", "bi_analyst")
    m = compute_classification_metrics(y, y, labels=labels)
    assert m.accuracy == 1.0
    assert m.macro_f1 == 1.0
    assert m.per_class["data_analyst"].f1 == 1.0
    assert m.per_class["data_analyst"].precision == 1.0
    assert m.per_class["data_analyst"].recall == 1.0
    assert m.confusion_matrix == ((2, 0), (0, 1))


def test_metrics_zero_division_is_zero():
    labels = ("data_analyst", "ml_engineer")
    m = compute_classification_metrics(
        ["data_analyst", "data_analyst"],
        ["data_analyst", "data_analyst"],
        labels=labels,
    )
    assert m.per_class["ml_engineer"].precision == 0.0
    assert m.per_class["ml_engineer"].recall == 0.0
    assert m.per_class["ml_engineer"].f1 == 0.0
    assert m.per_class["ml_engineer"].support == 0
    assert m.macro_f1 == 1.0
    assert m.macro_f1_all_labels == 0.5


def test_baseline_and_stub_share_metric_contract():
    ds = load_gold_dataset(SMALL)
    baseline = evaluate_predictor(DeterministicRoleFamilyBaseline(), ds.records)
    stub = evaluate_predictor(ConstantPredictor(), ds.records)
    for result in (baseline, stub):
        d = result.metrics.to_dict()
        assert "accuracy" in d
        assert "macro_f1" in d
        assert "macro_f1_all_labels" in d
        assert "per_class" in d
        assert "confusion_matrix" in d
        for label in role_family_label_order():
            scores = d["per_class"][label]
            assert set(scores) == {"precision", "recall", "f1", "support"}
        assert result.predictor_kind in {"baseline_rules", "ml_stub"}


def test_baseline_is_not_used_as_gold():
    ds = load_gold_dataset(SMALL)
    baseline = DeterministicRoleFamilyBaseline()
    preds = baseline.predict(ds.records)
    golds = [e.gold_role_family.value for e in ds.records]
    assert golds == [e.gold_role_family.value for e in ds.records]
    # Predictions are a separate channel even if they happen to match.
    assert preds is not golds
    assert all(isinstance(e.gold_role_family, RoleFamily) for e in ds.records)


def test_evaluation_manifest_is_reproducible(tmp_path):
    path_a = tmp_path / "m1.json"
    path_b = tmp_path / "m2.json"
    _, man_a = evaluate_gold_dataset(SMALL, seed=42, test_ratio=0.4, manifest_path=path_a)
    _, man_b = evaluate_gold_dataset(SMALL, seed=42, test_ratio=0.4, manifest_path=path_b)
    assert man_a.to_dict() == man_b.to_dict()
    assert path_a.read_text(encoding="utf-8") == path_b.read_text(encoding="utf-8")
    payload = json.loads(path_a.read_text(encoding="utf-8"))
    assert payload["schema"] == MANIFEST_SCHEMA
    assert payload["seed"] == 42
    assert payload["split"]["train_ids"]
    assert payload["split"]["test_ids"]
    assert payload["dataset_sha256"]
    assert payload["features"]["version"]
    assert payload["predictor"]["name"] == "deterministic_role_family"
    assert payload["predictor"]["kind"] == "baseline_rules"
    assert payload["dataset_sufficient_for_training"] is False
    assert payload["config"]["gold_independent_of_classifiers"] is True
    metrics = payload["metrics"]
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "confusion_matrix" in metrics


def test_fixture_gold_baseline_evaluation_offline(tmp_path):
    result, manifest = evaluate_gold_dataset(
        SMALL, seed=42, test_ratio=0.4, manifest_path=tmp_path / "m.json"
    )
    assert result.metrics.n == len(manifest.split["test_ids"])
    assert manifest.dataset_sufficient_for_training is False
    assert "accuracy" in result.metrics.to_dict()
    assert set(manifest.split["train_ids"]).isdisjoint(manifest.split["test_ids"])
    dumped = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert all(str(i).startswith("ex_") for i in dumped["split"]["train_ids"] + dumped["split"]["test_ids"])


def test_example_type_keeps_gold_field_name():
    ds = load_gold_dataset(SMALL)
    ex: RoleFamilyExample = ds.records[0]
    dumped = ex.to_dict()
    assert "gold_role_family" in dumped
    assert "role_family" not in dumped
