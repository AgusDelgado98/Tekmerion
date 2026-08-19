"""Training sufficiency gate."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.ml.gate import (
    STATUS_DATA_INSUFFICIENT,
    STATUS_SUFFICIENT,
    require_sufficient,
    sufficiency_report,
)
from analysis.ml.gold import load_gold_dataset
from analysis.ml.models import GOLD_SCHEMA, LABEL_SOURCE_HUMAN, MIN_EXAMPLES_PER_CLASS_FOR_TRAINING, MIN_N_FOR_TRAINING
from analysis.models import RoleFamily

SMALL = Path("tests/fixtures/ml/gold_small.json")
CARD = Path("data/ml/gold/evaluation_card.json")


def test_small_fixture_gate_is_insufficient():
    ds = load_gold_dataset(SMALL)
    report = sufficiency_report(ds)
    assert report["status"] == STATUS_DATA_INSUFFICIENT
    assert report["sufficient_for_training"] is False
    assert report["dataset_n"] < MIN_N_FOR_TRAINING


def test_synthetic_sufficient_gold_passes_gate(tmp_path):
    records = []
    for family in RoleFamily:
        if family is RoleFamily.UNKNOWN:
            continue
        for i in range(15):
            records.append(
                {
                    "id": f"{family.value}_{i}",
                    "title": f"{family.value} {i}",
                    "description": f"synthetic {family.value} example {i} sql python",
                    "gold_role_family": family.value,
                    "label_source": LABEL_SOURCE_HUMAN,
                    "annotator_id": "test",
                    "labeled_at": "2026-08-19",
                }
            )
    path = tmp_path / "gold.json"
    path.write_text(
        json.dumps(
            {
                "schema": GOLD_SCHEMA,
                "task": "role_family_classification",
                "label_field": "gold_role_family",
                "label_source": LABEL_SOURCE_HUMAN,
                "label_policy": "test",
                "limitations": ["test"],
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    ds = load_gold_dataset(path)
    report = sufficiency_report(ds)
    assert report["status"] == STATUS_SUFFICIENT
    assert report["dataset_n"] >= MIN_N_FOR_TRAINING
    assert min(report["class_distribution"].values()) >= MIN_EXAMPLES_PER_CLASS_FOR_TRAINING
    require_sufficient(ds)


def test_published_evaluation_card_documents_the_gate():
    card = json.loads(CARD.read_text(encoding="utf-8"))
    assert card["n"] >= MIN_N_FOR_TRAINING
    assert min(card["class_distribution"].values()) >= MIN_EXAMPLES_PER_CLASS_FOR_TRAINING
    assert card["n_train"] + card["n_test"] == card["n"]
