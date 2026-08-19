"""Immutable contracts for the ML evaluation layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

from analysis.models import RoleFamily

GOLD_SCHEMA = "tekmerion.ml.gold_dataset.v1"
FEATURE_VERSION = "tekmerion.ml.features.v1"
MANIFEST_SCHEMA = "tekmerion.ml.evaluation_manifest.v1"
DEFAULT_SPLIT_SEED = 42
DEFAULT_TEST_RATIO = 0.30
# Explicitly conservative: this repo's labeled sample is not training-scale.
MIN_N_FOR_TRAINING = 100
MIN_EXAMPLES_PER_CLASS_FOR_TRAINING = 10

LABEL_SOURCE_HUMAN = "human"

FORBIDDEN_GOLD_KEYS = frozenset(
    {
        "role_family",
        "predicted_role_family",
        "classifier_role_family",
        "pipeline_role_family",
        "predicted_label",
        "y_pred",
    }
)


def role_family_label_order() -> tuple[str, ...]:
    """Stable label axis for metrics (includes unknown)."""
    return tuple(m.value for m in RoleFamily)


@dataclass(frozen=True)
class RoleFamilyExample:
    """One gold-labeled vacancy. The label is human, never a classifier copy."""

    id: str
    title: str
    description: str
    gold_role_family: RoleFamily
    label_source: str
    annotator_id: str
    labeled_at: str
    content_fingerprint: str
    company: str = ""
    source_kind: str = "synthetic"
    source_ref: str = ""
    notes: str = ""
    location: str = ""
    source_url: str = ""
    retrieved_at: str = ""
    source_record_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gold_role_family"] = self.gold_role_family.value
        return d


@dataclass(frozen=True)
class GoldDataset:
    schema: str
    task: str
    label_field: str
    label_source: str
    label_policy: str
    limitations: tuple[str, ...]
    records: tuple[RoleFamilyExample, ...]
    path: Optional[str] = None
    extra_meta: tuple[tuple[str, str], ...] = ()

    @property
    def n(self) -> int:
        return len(self.records)

    def sufficient_for_training(self) -> bool:
        if self.n < MIN_N_FOR_TRAINING:
            return False
        counts: dict[str, int] = {}
        for r in self.records:
            key = r.gold_role_family.value
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            return False
        return min(counts.values()) >= MIN_EXAMPLES_PER_CLASS_FOR_TRAINING


@dataclass(frozen=True)
class FeatureVector:
    """Per-example features. Must not contain gold or rule-based role labels."""

    example_id: str
    title: str
    description: str
    skills_extracted: tuple[str, ...]
    skill_indicators: tuple[tuple[str, int], ...]
    feature_version: str = FEATURE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "title": self.title,
            "description": self.description,
            "skills_extracted": list(self.skills_extracted),
            "skill_indicators": [[k, v] for k, v in self.skill_indicators],
            "feature_version": self.feature_version,
        }


@dataclass(frozen=True)
class SplitResult:
    seed: int
    test_ratio: float
    strategy: str
    train: tuple[RoleFamilyExample, ...]
    test: tuple[RoleFamilyExample, ...]
    grouped_by: str
    warnings: tuple[str, ...] = ()

    @property
    def train_ids(self) -> tuple[str, ...]:
        return tuple(e.id for e in self.train)

    @property
    def test_ids(self) -> tuple[str, ...]:
        return tuple(e.id for e in self.test)


@dataclass(frozen=True)
class ClassScores:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    per_class: dict[str, ClassScores]
    confusion_matrix: tuple[tuple[int, ...], ...]
    labels: tuple[str, ...]
    n: int
    macro_f1_all_labels: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "macro_f1_all_labels": self.macro_f1_all_labels,
            "per_class": {
                k: {
                    "precision": v.precision,
                    "recall": v.recall,
                    "f1": v.f1,
                    "support": v.support,
                }
                for k, v in self.per_class.items()
            },
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "labels": list(self.labels),
            "n": self.n,
        }


@dataclass(frozen=True)
class EvaluationResult:
    metrics: ClassificationMetrics
    y_true: tuple[str, ...]
    y_pred: tuple[str, ...]
    predictor_name: str
    predictor_kind: str


@dataclass
class EvaluationManifest:
    schema: str = MANIFEST_SCHEMA
    dataset_path: str = ""
    dataset_sha256: str = ""
    dataset_n: int = 0
    dataset_sufficient_for_training: bool = False
    limitations: list[str] = field(default_factory=list)
    split: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    predictor: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    seed: int = DEFAULT_SPLIT_SEED
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "dataset_path": self.dataset_path,
            "dataset_sha256": self.dataset_sha256,
            "dataset_n": self.dataset_n,
            "dataset_sufficient_for_training": self.dataset_sufficient_for_training,
            "limitations": list(self.limitations),
            "seed": self.seed,
            "split": self.split,
            "features": self.features,
            "predictor": self.predictor,
            "metrics": self.metrics,
            "config": self.config,
        }


class Predictor:
    """Minimal predictor interface (baseline or future ML)."""

    name: str
    kind: str

    def predict(self, examples: Sequence[RoleFamilyExample]) -> list[str]:
        raise NotImplementedError
