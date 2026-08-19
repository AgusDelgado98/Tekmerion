"""Train-only text + skill vectorizer (sklearn). Gold labels are never inputs."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from analysis.ml.models import FEATURE_VERSION, RoleFamilyExample
from analysis.skills import extract_skills

FEATURE_VERSION_TFIDF = "tekmerion.ml.features.v2_tfidf_skills"


def examples_to_x(examples: Sequence[RoleFamilyExample]) -> list[dict[str, str]]:
    return [{"title": ex.title, "description": ex.description} for ex in examples]


def _text(row: Any) -> str:
    if isinstance(row, dict):
        return f"{row.get('title') or ''} {row.get('description') or ''}".strip()
    return str(row or "")


class JobTextSkillVectorizer(BaseEstimator, TransformerMixin):
    """TF-IDF on title+description plus binary skill indicators.

    ``fit`` must see only the current train (or CV fold) rows.
    """

    def __init__(self, ngram_range: tuple[int, int] = (1, 2), max_features: int = 4000):
        self.ngram_range = ngram_range
        self.max_features = max_features

    def fit(self, X: Iterable[Any], y: Any = None):
        rows = list(X)
        texts = [_text(r) for r in rows]
        self.tfidf_ = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=1,
            lowercase=True,
        )
        self.tfidf_.fit(texts)
        vocab: set[str] = set()
        for text in texts:
            vocab.update(extract_skills(text))
        self.skill_vocab_ = tuple(sorted(vocab))
        return self

    def transform(self, X: Iterable[Any]):
        if not hasattr(self, "tfidf_"):
            raise RuntimeError("JobTextSkillVectorizer.fit() must be called first")
        rows = list(X)
        texts = [_text(r) for r in rows]
        text_mat = self.tfidf_.transform(texts)
        vocab = self.skill_vocab_
        if not vocab:
            return text_mat
        skill_rows = []
        for text in texts:
            found = set(extract_skills(text))
            skill_rows.append([1 if name in found else 0 for name in vocab])
        skill_mat = csr_matrix(skill_rows, dtype="float64")
        return hstack([text_mat, skill_mat], format="csr")

    def config(self) -> dict[str, Any]:
        vocab_size = len(getattr(self, "skill_vocab_", ()))
        return {
            "version": FEATURE_VERSION_TFIDF,
            "fields": ["title", "description", "skills_extracted"],
            "tfidf_ngram_range": list(self.ngram_range),
            "tfidf_max_features": self.max_features,
            "skill_vocab_size": vocab_size,
            "corpus_statistics": "fit_on_current_X_only",
            "base_feature_version": FEATURE_VERSION,
        }
