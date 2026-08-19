"""Deterministic role-family baseline used as an ML-comparable predictor.

Wraps the existing rule classifier. This is a *predictor*, never gold.
"""

from __future__ import annotations

from typing import Sequence

from analysis.classifiers import classify_role_family
from analysis.ml.models import Predictor, RoleFamilyExample


class DeterministicRoleFamilyBaseline(Predictor):
    name = "deterministic_role_family"
    kind = "baseline_rules"

    def predict(self, examples: Sequence[RoleFamilyExample]) -> list[str]:
        return [
            classify_role_family(ex.title, ex.description).value for ex in examples
        ]
