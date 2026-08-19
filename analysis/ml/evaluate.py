"""Common evaluation entry point and reproducible manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from analysis.ml.baseline import DeterministicRoleFamilyBaseline
from analysis.ml.features import FeatureBuilder, build_split_features
from analysis.ml.gold import dataset_sha256, load_gold_dataset, repo_relative_path
from analysis.ml.publish import write_sanitized_json
from analysis.ml.metrics import compute_classification_metrics
from analysis.ml.models import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_RATIO,
    FEATURE_VERSION,
    MANIFEST_SCHEMA,
    EvaluationManifest,
    EvaluationResult,
    GoldDataset,
    Predictor,
    RoleFamilyExample,
    role_family_label_order,
)
from analysis.ml.split import grouped_train_test_split


def evaluate_predictor(
    predictor: Predictor,
    examples: Sequence[RoleFamilyExample],
    *,
    labels: Sequence[str] | None = None,
) -> EvaluationResult:
    y_true = tuple(ex.gold_role_family.value for ex in examples)
    y_pred = tuple(predictor.predict(examples))
    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        labels=labels or role_family_label_order(),
    )
    return EvaluationResult(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        predictor_name=predictor.name,
        predictor_kind=predictor.kind,
    )


def dump_canonical_json(payload: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    out.write_text(text, encoding="utf-8")


def build_evaluation_manifest(
    dataset: GoldDataset,
    *,
    split,
    feature_builder: FeatureBuilder,
    result: EvaluationResult,
    seed: int,
    test_ratio: float,
    extra_config: Optional[dict[str, Any]] = None,
) -> EvaluationManifest:
    limitations = list(dataset.limitations)
    if not dataset.sufficient_for_training():
        limitations.append(
            "Gold N is below the training sufficiency gate "
            "(see MIN_N_FOR_TRAINING / MIN_EXAMPLES_PER_CLASS_FOR_TRAINING). "
            "Do not train production models on this file."
        )
    ds_hash = dataset_sha256(dataset.path) if dataset.path else ""
    return EvaluationManifest(
        schema=MANIFEST_SCHEMA,
        dataset_path=repo_relative_path(dataset.path) if dataset.path else "",
        dataset_sha256=ds_hash,
        dataset_n=dataset.n,
        dataset_sufficient_for_training=dataset.sufficient_for_training(),
        limitations=limitations,
        seed=seed,
        split={
            "strategy": split.strategy,
            "seed": split.seed,
            "test_ratio": test_ratio,
            "grouped_by": split.grouped_by,
            "train_ids": list(split.train_ids),
            "test_ids": list(split.test_ids),
            "n_train": len(split.train),
            "n_test": len(split.test),
            "warnings": list(split.warnings),
        },
        features=feature_builder.config(),
        predictor={
            "name": result.predictor_name,
            "kind": result.predictor_kind,
        },
        metrics=result.metrics.to_dict(),
        config={
            "feature_version": FEATURE_VERSION,
            "evaluated_on": "test_split",
            "gold_label_field": "gold_role_family",
            "gold_independent_of_classifiers": True,
            **(extra_config or {}),
        },
    )


def evaluate_gold_dataset(
    path: str | Path | None = None,
    *,
    predictor: Predictor | None = None,
    seed: int = DEFAULT_SPLIT_SEED,
    test_ratio: float = DEFAULT_TEST_RATIO,
    manifest_path: str | Path | None = None,
) -> tuple[EvaluationResult, EvaluationManifest]:
    """
    Load gold, split, fit features on train, score predictor on test.

    Default predictor is the current deterministic role-family rules.
    """
    dataset = load_gold_dataset(path)
    split = grouped_train_test_split(dataset.records, seed=seed, test_ratio=test_ratio)
    _, _, feature_builder = build_split_features(split.train, split.test)
    pred = predictor or DeterministicRoleFamilyBaseline()
    result = evaluate_predictor(pred, split.test)
    manifest = build_evaluation_manifest(
        dataset,
        split=split,
        feature_builder=feature_builder,
        result=result,
        seed=seed,
        test_ratio=test_ratio,
    )
    if manifest_path is not None:
        write_sanitized_json(manifest.to_dict(), manifest_path)
    return result, manifest
