"""
ML evaluation layer (V0.8 Block A).

Offline, separate from Evidence and the production pipeline.
Does not train models and does not write predictions into Evidence.
"""

from analysis.ml.models import (
    ClassificationMetrics,
    EvaluationManifest,
    EvaluationResult,
    FeatureVector,
    GoldDataset,
    RoleFamilyExample,
    SplitResult,
)

__all__ = [
    "ClassificationMetrics",
    "EvaluationManifest",
    "EvaluationResult",
    "FeatureVector",
    "GoldDataset",
    "RoleFamilyExample",
    "SplitResult",
]
