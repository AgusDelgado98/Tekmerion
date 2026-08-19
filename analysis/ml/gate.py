"""Sufficiency gate for supervised ML (Block B). Training must not run below this."""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from analysis.ml.models import (
    MIN_EXAMPLES_PER_CLASS_FOR_TRAINING,
    MIN_N_FOR_TRAINING,
    GoldDataset,
)

STATUS_DATA_INSUFFICIENT = "DATA_INSUFFICIENT"
STATUS_SUFFICIENT = "SUFFICIENT"


class DataInsufficientError(RuntimeError):
    """Gold does not meet the Block A training gate. Do not fit models."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__(report.get("message") or STATUS_DATA_INSUFFICIENT)


def class_distribution(dataset: GoldDataset) -> dict[str, int]:
    counts = Counter(r.gold_role_family.value for r in dataset.records)
    return dict(sorted(counts.items()))


def sufficiency_report(
    dataset: GoldDataset,
    *,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    dist = class_distribution(dataset)
    min_class = min(dist.values()) if dist else 0
    ok = dataset.sufficient_for_training()
    gaps = {
        label: max(0, MIN_EXAMPLES_PER_CLASS_FOR_TRAINING - n)
        for label, n in dist.items()
        if n < MIN_EXAMPLES_PER_CLASS_FOR_TRAINING
    }
    report: dict[str, Any] = {
        "status": STATUS_SUFFICIENT if ok else STATUS_DATA_INSUFFICIENT,
        "dataset_n": dataset.n,
        "min_n_for_training": MIN_N_FOR_TRAINING,
        "min_examples_per_class": MIN_EXAMPLES_PER_CLASS_FOR_TRAINING,
        "n_gap": max(0, MIN_N_FOR_TRAINING - dataset.n),
        "class_distribution": dist,
        "class_gaps": gaps,
        "singleton_or_sparse_classes": sorted(
            k for k, n in dist.items() if n < MIN_EXAMPLES_PER_CLASS_FOR_TRAINING
        ),
        "sufficient_for_training": ok,
        "message": (
            "Gold meets the training gate."
            if ok
            else (
                f"{STATUS_DATA_INSUFFICIENT}: n={dataset.n} "
                f"(need {MIN_N_FOR_TRAINING}); "
                f"min per class={min_class} "
                f"(need {MIN_EXAMPLES_PER_CLASS_FOR_TRAINING}). "
                "Do not invent vacancies. Collect more human labels from ingested sources."
            )
        ),
    }
    if extra:
        report.update(extra)
    return report


def require_sufficient(dataset: GoldDataset) -> None:
    report = sufficiency_report(dataset)
    if not dataset.sufficient_for_training():
        raise DataInsufficientError(report)
