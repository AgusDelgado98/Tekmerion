"""Classification metrics shared by the deterministic baseline and future models."""

from __future__ import annotations

from typing import Sequence

from analysis.ml.models import ClassScores, ClassificationMetrics, role_family_label_order


def _safe_div(num: float, den: float) -> float:
    if den == 0.0:
        return 0.0
    return num / den


def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
) -> ClassificationMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("cannot score empty predictions")

    axis = tuple(labels) if labels is not None else role_family_label_order()
    index = {label: i for i, label in enumerate(axis)}
    n_labels = len(axis)
    matrix = [[0 for _ in range(n_labels)] for _ in range(n_labels)]

    n = len(y_true)
    correct = 0
    support = [0] * n_labels
    pred_count = [0] * n_labels
    tp = [0] * n_labels

    for t, p in zip(y_true, y_pred):
        if t not in index or p not in index:
            raise ValueError(f"label out of axis: true={t!r} pred={p!r}")
        ti, pi = index[t], index[p]
        matrix[ti][pi] += 1
        support[ti] += 1
        pred_count[pi] += 1
        if t == p:
            correct += 1
            tp[ti] += 1

    per_class: dict[str, ClassScores] = {}
    f1_all: list[float] = []
    f1_supported: list[float] = []
    for i, label in enumerate(axis):
        prec = _safe_div(float(tp[i]), float(pred_count[i]))
        rec = _safe_div(float(tp[i]), float(support[i]))
        f1 = _safe_div(2.0 * prec * rec, prec + rec)
        per_class[label] = ClassScores(
            precision=prec,
            recall=rec,
            f1=f1,
            support=support[i],
        )
        f1_all.append(f1)
        if support[i] > 0:
            f1_supported.append(f1)

    macro_f1 = (
        sum(f1_supported) / float(len(f1_supported)) if f1_supported else 0.0
    )
    macro_all = sum(f1_all) / float(len(f1_all)) if f1_all else 0.0
    return ClassificationMetrics(
        accuracy=_safe_div(float(correct), float(n)),
        macro_f1=macro_f1,
        per_class=per_class,
        confusion_matrix=tuple(tuple(row) for row in matrix),
        labels=axis,
        n=n,
        macro_f1_all_labels=macro_all,
    )
