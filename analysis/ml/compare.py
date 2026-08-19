"""Rules vs ML comparison under the Block A evaluation contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from analysis.ml.baseline import DeterministicRoleFamilyBaseline
from analysis.ml.evaluate import build_evaluation_manifest, evaluate_predictor
from analysis.ml.features import build_split_features
from analysis.ml.gate import STATUS_DATA_INSUFFICIENT, sufficiency_report
from analysis.ml.gold import dataset_sha256, load_gold_dataset, repo_relative_path
from analysis.ml.harvest import harvest_unlabeled_candidates
from analysis.ml.models import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_RATIO,
    EvaluationResult,
    GoldDataset,
)
from analysis.ml.publish import write_sanitized_json
from analysis.ml.split import grouped_train_test_split
from analysis.ml.train import train_supervised_models

COMPARISON_SCHEMA = "tekmerion.ml.rules_vs_ml.v1"
DEFAULT_REPORT_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "ml" / "reports" / "block_b.json"
)
DEFAULT_ARTIFACTS_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "ml" / "artifacts"
)
# Tiny test-set F1 gaps are not a defensible promotion case (n_test is small).
MIN_MACRO_F1_DELTA_TO_PROMOTE = 0.02


def _result_blob(result: EvaluationResult) -> dict[str, Any]:
    return {
        "predictor_name": result.predictor_name,
        "predictor_kind": result.predictor_kind,
        "metrics": result.metrics.to_dict(),
        "y_true": list(result.y_true),
        "y_pred": list(result.y_pred),
    }


def _errors(
    examples,
    result: EvaluationResult,
    *,
    limit: int = 25,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ex, true, pred in zip(examples, result.y_true, result.y_pred):
        if true == pred:
            continue
        rows.append(
            {
                "id": ex.id,
                "gold_role_family": true,
                "predicted": pred,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _supported_zero_f1_count(metrics: dict[str, Any]) -> int:
    per_class = metrics.get("per_class") or {}
    n = 0
    for scores in per_class.values():
        if not isinstance(scores, dict):
            continue
        if int(scores.get("support") or 0) > 0 and float(scores.get("f1") or 0.0) <= 0.0:
            n += 1
    return n


def _pick_winner(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = []
    for row in rows:
        metrics = row["test"]["metrics"]
        scored.append(
            (
                float(metrics["macro_f1"]),
                float(metrics["accuracy"]),
                row["name"],
                row["kind"],
            )
        )
    scored.sort(reverse=True)
    best = scored[0]
    baseline = next(r for r in rows if r["kind"] == "baseline_rules")
    base_metrics = baseline["test"]["metrics"]
    base_f1 = float(base_metrics["macro_f1"])
    ml_rows = [r for r in rows if r["kind"] != "baseline_rules"]
    best_ml = max(ml_rows, key=lambda r: float(r["test"]["metrics"]["macro_f1"])) if ml_rows else None
    best_ml_f1 = (
        float(best_ml["test"]["metrics"]["macro_f1"]) if best_ml is not None else None
    )
    delta = (best_ml_f1 - base_f1) if best_ml_f1 is not None else None
    ml_beats = best_ml_f1 is not None and best_ml_f1 > base_f1 + 1e-12
    collapsed = False
    if best_ml is not None:
        collapsed = _supported_zero_f1_count(best_ml["test"]["metrics"]) > _supported_zero_f1_count(
            base_metrics
        )
    defensible = (
        ml_beats
        and delta is not None
        and delta >= MIN_MACRO_F1_DELTA_TO_PROMOTE
        and not collapsed
    )
    if defensible:
        note = (
            f"ML beats classify_role_family on test macro F1 by {delta:.3f} "
            f"(threshold {MIN_MACRO_F1_DELTA_TO_PROMOTE}). Still offline — not wired to Evidence."
        )
    elif ml_beats and collapsed:
        note = (
            "Do not promote ML: higher macro F1 but more supported classes with F1=0 "
            "than the rules baseline."
        )
    elif ml_beats:
        note = (
            "Do not promote ML: test macro F1 gain is below "
            f"{MIN_MACRO_F1_DELTA_TO_PROMOTE} and n_test is small — not a defensible lift."
        )
    else:
        note = "Do not promote ML: it does not beat classify_role_family on test macro F1."
    return {
        "name": best[2],
        "kind": best[3],
        "criterion": "macro_f1 then accuracy on held-out test",
        "macro_f1": best[0],
        "accuracy": best[1],
        "ml_beats_baseline": ml_beats,
        "promote_ml": defensible,
        "macro_f1_delta_vs_rules": delta,
        "min_macro_f1_delta_to_promote": MIN_MACRO_F1_DELTA_TO_PROMOTE,
        "note": note,
    }


def run_block_b(
    gold_path: str | Path | None = None,
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    test_ratio: float = DEFAULT_TEST_RATIO,
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
    artifacts_dir: str | Path | None = DEFAULT_ARTIFACTS_DIR,
    harvest: bool = True,
    allow_train: bool = True,
) -> dict[str, Any]:
    dataset: GoldDataset = load_gold_dataset(gold_path)
    split = grouped_train_test_split(dataset.records, seed=seed, test_ratio=test_ratio)
    _, _, feature_builder = build_split_features(split.train, split.test)

    harvest_info: Optional[dict[str, Any]] = None
    if harvest:
        raw = harvest_unlabeled_candidates()
        harvest_info = {
            "n_loaded": raw["n_loaded"],
            "n_unique": raw["n_unique"],
            "n_dropped_duplicates": raw["n_dropped_duplicates"],
            "unique_by_source_kind": raw["unique_by_source_kind"],
            "sources_used": raw["sources_used"],
            "limitations": raw["limitations"],
        }

    gate = sufficiency_report(
        dataset,
        extra={"unlabeled_unique_candidates": (harvest_info or {}).get("n_unique")},
    )

    baseline = DeterministicRoleFamilyBaseline()
    base_result = evaluate_predictor(baseline, split.test)
    base_manifest = build_evaluation_manifest(
        dataset,
        split=split,
        feature_builder=feature_builder,
        result=base_result,
        seed=seed,
        test_ratio=test_ratio,
        extra_config={
            "block": "B",
            "role": "deterministic_baseline",
            "primary_metric": "macro_f1",
        },
    )

    written_manifests: list[str] = []
    if artifacts_dir is not None:
        art = Path(artifacts_dir)
        write_sanitized_json(base_manifest.to_dict(), art / "evaluation_manifest.json")
        written_manifests.append(repo_relative_path(art / "evaluation_manifest.json"))

    model_rows: list[dict[str, Any]] = [
        {
            "name": baseline.name,
            "kind": baseline.kind,
            "params": {},
            "cv_macro_f1": None,
            "test": _result_blob(base_result),
            "errors": _errors(split.test, base_result),
        }
    ]

    training_ran = False
    training_error: Optional[str] = None
    if allow_train and dataset.sufficient_for_training():
        predictors, metas = train_supervised_models(dataset, split.train)
        training_ran = True
        for pred, meta in zip(predictors, metas):
            result = evaluate_predictor(pred, split.test)
            ml_manifest = build_evaluation_manifest(
                dataset,
                split=split,
                feature_builder=feature_builder,
                result=result,
                seed=seed,
                test_ratio=test_ratio,
                extra_config={
                    "block": "B",
                    "role": "sklearn_supervised",
                    "best_params": meta["best_params"],
                    "cv_macro_f1": meta["cv_macro_f1"],
                    "cv_splits": meta["cv_splits"],
                    "cv_scoring": "f1_macro",
                    "vectorizer": meta["vectorizer"],
                    "class_weight": "balanced",
                },
            )
            if artifacts_dir is not None:
                man_path = Path(artifacts_dir) / f"evaluation_manifest_{pred.name}.json"
                write_sanitized_json(ml_manifest.to_dict(), man_path)
                written_manifests.append(repo_relative_path(man_path))
            model_rows.append(
                {
                    "name": pred.name,
                    "kind": pred.kind,
                    "params": meta["best_params"],
                    "cv_macro_f1": meta["cv_macro_f1"],
                    "cv": {
                        "splits": meta["cv_splits"],
                        "scoring": "f1_macro",
                        "seed": meta["seed"],
                    },
                    "vectorizer": meta["vectorizer"],
                    "test": _result_blob(result),
                    "errors": _errors(split.test, result),
                }
            )
    elif allow_train:
        training_error = gate["message"]

    comparison = {
        "schema": COMPARISON_SCHEMA,
        "status": gate["status"],
        "training_ran": training_ran,
        "training_skipped_reason": training_error,
        "dataset_path": repo_relative_path(dataset.path) if dataset.path else "",
        "dataset_sha256": dataset_sha256(dataset.path) if dataset.path else "",
        "dataset_n": dataset.n,
        "seed": seed,
        "split": {
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
        "gate": gate,
        "harvest": harvest_info,
        "baseline_manifest_excerpt": {
            "schema": base_manifest.schema,
            "predictor": base_manifest.predictor,
            "metrics": base_manifest.metrics,
        },
        "models": [
            {
                "name": r["name"],
                "kind": r["kind"],
                "params": r["params"],
                "cv_macro_f1": r["cv_macro_f1"],
                "test_macro_f1": r["test"]["metrics"]["macro_f1"],
                "test_accuracy": r["test"]["metrics"]["accuracy"],
                "test_macro_f1_all_labels": r["test"]["metrics"].get("macro_f1_all_labels"),
            }
            for r in model_rows
        ],
        "artifacts": written_manifests,
        "models_detail": model_rows,
        "winner": (
            {
                "name": baseline.name,
                "kind": baseline.kind,
                "criterion": "macro_f1 then accuracy on held-out test",
                "macro_f1": base_result.metrics.macro_f1,
                "accuracy": base_result.metrics.accuracy,
                "ml_beats_baseline": False,
                "promote_ml": False,
                "note": "Training was not run. Baseline rules remain the only scored model.",
            }
            if not training_ran
            else _pick_winner(model_rows)
        ),
        "limitations": list(dataset.limitations),
        "productive_integration": {
            "evidence": False,
            "pipeline": False,
            "flask": False,
        },
    }

    if report_path is not None:
        write_sanitized_json(comparison, report_path)
    return comparison
