"""Leakage-safe feature builder (title, description, extracted skills)."""

from __future__ import annotations

from typing import Optional, Sequence

from analysis.ml.models import FEATURE_VERSION, FeatureVector, RoleFamilyExample
from analysis.skills import extract_skills

FEATURE_FIELDS = ("title", "description", "skills_extracted")


class FeatureBuilderError(ValueError):
    """Invalid feature fit/transform usage."""


class FeatureBuilder:
    """
    Fit skill indicator vocabulary on *train only*.

    Transform never reads gold_role_family and never calls role-family rules.
    """

    def __init__(self) -> None:
        self._skill_vocab: tuple[str, ...] | None = None
        self._fitted_on_ids: tuple[str, ...] | None = None

    @property
    def skill_vocab(self) -> tuple[str, ...]:
        if self._skill_vocab is None:
            raise FeatureBuilderError("FeatureBuilder.fit() must be called first")
        return self._skill_vocab

    @property
    def is_fitted(self) -> bool:
        return self._skill_vocab is not None

    def fit(self, examples: Sequence[RoleFamilyExample]) -> "FeatureBuilder":
        vocab: set[str] = set()
        ids: list[str] = []
        for ex in examples:
            text = f"{ex.title} {ex.description}"
            vocab.update(extract_skills(text))
            ids.append(ex.id)
        self._skill_vocab = tuple(sorted(vocab))
        self._fitted_on_ids = tuple(ids)
        return self

    def transform(self, examples: Sequence[RoleFamilyExample]) -> list[FeatureVector]:
        if self._skill_vocab is None:
            raise FeatureBuilderError("FeatureBuilder.fit() must be called first")
        vectors: list[FeatureVector] = []
        for ex in examples:
            skills = tuple(extract_skills(f"{ex.title} {ex.description}"))
            skill_set = set(skills)
            indicators = tuple((name, 1 if name in skill_set else 0) for name in self._skill_vocab)
            vectors.append(
                FeatureVector(
                    example_id=ex.id,
                    title=ex.title,
                    description=ex.description,
                    skills_extracted=skills,
                    skill_indicators=indicators,
                    feature_version=FEATURE_VERSION,
                )
            )
        return vectors

    def fit_transform(self, examples: Sequence[RoleFamilyExample]) -> list[FeatureVector]:
        return self.fit(examples).transform(examples)

    def assert_no_label_leakage(self, vectors: Sequence[FeatureVector]) -> None:
        forbidden = (
            "gold_role_family",
            "role_family",
            "label",
            "y",
            "y_true",
        )
        for vec in vectors:
            payload = vec.to_dict()
            for key in forbidden:
                if key in payload:
                    raise FeatureBuilderError(f"feature payload contains forbidden key {key}")
            blob = json_safe_lower(payload)
            if "gold_role_family" in blob:
                raise FeatureBuilderError("gold label leaked into features")

    def config(self) -> dict:
        return {
            "version": FEATURE_VERSION,
            "fields": list(FEATURE_FIELDS),
            "skill_vocab": list(self.skill_vocab) if self.is_fitted else [],
            "fitted_on_ids": list(self._fitted_on_ids or ()),
            "corpus_statistics": "train_skill_vocabulary_only",
        }


def json_safe_lower(obj: object) -> str:
    return str(obj).lower()


def build_split_features(
    train: Sequence[RoleFamilyExample],
    test: Sequence[RoleFamilyExample],
    builder: Optional[FeatureBuilder] = None,
) -> tuple[list[FeatureVector], list[FeatureVector], FeatureBuilder]:
    """Fit on train, transform train and test. Never fit on the union."""
    fb = builder or FeatureBuilder()
    fb.fit(train)
    train_vec = fb.transform(train)
    test_vec = fb.transform(test)
    fb.assert_no_label_leakage(train_vec + test_vec)
    return train_vec, test_vec, fb
