"""Supervised role-family training. Refuses to fit when the gold gate fails."""

from __future__ import annotations

from typing import Any, Sequence

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from analysis.ml.gate import require_sufficient
from analysis.ml.models import (
    DEFAULT_SPLIT_SEED,
    Predictor,
    RoleFamilyExample,
)
from analysis.ml.vectorize import JobTextSkillVectorizer, examples_to_x

TRAIN_SEED = DEFAULT_SPLIT_SEED
CV_SPLITS = 3

# Bounded grids — Block B, not a wide search.
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "logreg": {"clf__C": [0.25, 1.0, 4.0]},
    "linearsvc": {"clf__C": [0.25, 1.0, 4.0]},
    "random_forest": {
        "clf__n_estimators": [50, 100],
        "clf__max_depth": [8, None],
    },
}


def _classifier(name: str):
    if name == "logreg":
        return LogisticRegression(
            max_iter=400,
            class_weight="balanced",
            solver="lbfgs",
            random_state=TRAIN_SEED,
        )
    if name == "linearsvc":
        return LinearSVC(
            class_weight="balanced",
            random_state=TRAIN_SEED,
            max_iter=2000,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced",
            random_state=TRAIN_SEED,
            n_jobs=1,
        )
    raise ValueError(f"unknown model {name}")


def _cv_splitter(y: Sequence[str]):
    counts: dict[str, int] = {}
    for label in y:
        counts[label] = counts.get(label, 0) + 1
    min_count = min(counts.values()) if counts else 0
    if min_count >= CV_SPLITS:
        return StratifiedKFold(
            n_splits=CV_SPLITS, shuffle=True, random_state=TRAIN_SEED
        )
    n_splits = min(CV_SPLITS, max(2, len(y) // 2))
    return KFold(n_splits=n_splits, shuffle=True, random_state=TRAIN_SEED)


class SklearnRoleFamilyPredictor(Predictor):
    def __init__(self, pipeline: Pipeline, name: str, kind: str = "ml_sklearn"):
        self.pipeline = pipeline
        self.name = name
        self.kind = kind

    def predict(self, examples: Sequence[RoleFamilyExample]) -> list[str]:
        X = examples_to_x(examples)
        pred = self.pipeline.predict(X)
        return [str(p) for p in pred]


def fit_model(
    name: str,
    train: Sequence[RoleFamilyExample],
    *,
    seed: int = TRAIN_SEED,
) -> tuple[SklearnRoleFamilyPredictor, dict[str, Any]]:
    y = [ex.gold_role_family.value for ex in train]
    X = examples_to_x(train)
    pipe = Pipeline(
        [
            ("vec", JobTextSkillVectorizer()),
            ("clf", _classifier(name)),
        ]
    )
    search = GridSearchCV(
        pipe,
        PARAM_GRIDS[name],
        cv=_cv_splitter(y),
        scoring="f1_macro",
        n_jobs=1,
        refit=True,
    )
    search.fit(X, y)
    best: Pipeline = search.best_estimator_
    meta = {
        "name": name,
        "kind": "ml_sklearn",
        "best_params": search.best_params_,
        "cv_macro_f1": float(search.best_score_),
        "cv_splits": int(getattr(search.cv, "n_splits", CV_SPLITS)),
        "n_train": len(train),
        "seed": seed,
        "vectorizer": best.named_steps["vec"].config(),
    }
    return SklearnRoleFamilyPredictor(best, name=name, kind="ml_sklearn"), meta


def train_supervised_models(
    dataset,
    train: Sequence[RoleFamilyExample],
    *,
    model_names: Sequence[str] = ("logreg", "linearsvc", "random_forest"),
) -> tuple[list[SklearnRoleFamilyPredictor], list[dict[str, Any]]]:
    require_sufficient(dataset)
    predictors: list[SklearnRoleFamilyPredictor] = []
    metas: list[dict[str, Any]] = []
    for name in model_names:
        pred, meta = fit_model(name, train)
        predictors.append(pred)
        metas.append(meta)
    return predictors, metas
