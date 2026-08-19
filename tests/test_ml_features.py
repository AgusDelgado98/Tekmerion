"""Feature builder: train-only vocabulary, no gold in features."""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis.ml.features import FeatureBuilder, FeatureBuilderError, build_split_features
from analysis.ml.gold import load_gold_dataset
from analysis.ml.split import grouped_train_test_split

SMALL = Path("tests/fixtures/ml/gold_small.json")


def test_transform_requires_fit():
    ds = load_gold_dataset(SMALL)
    fb = FeatureBuilder()
    with pytest.raises(FeatureBuilderError, match="fit"):
        fb.transform(ds.records)


def test_features_exclude_gold_and_role_family():
    ds = load_gold_dataset(SMALL)
    split = grouped_train_test_split(ds.records, seed=42, test_ratio=0.4)
    train_vec, test_vec, fb = build_split_features(split.train, split.test)
    fb.assert_no_label_leakage(train_vec + test_vec)
    for vec in train_vec + test_vec:
        payload = vec.to_dict()
        assert "gold_role_family" not in payload
        assert "role_family" not in payload
        assert "title" in payload
        assert "description" in payload
        assert "skills_extracted" in payload
        assert vec.feature_version.startswith("tekmerion.ml.features")


def test_skill_vocabulary_comes_from_train_only():
    ds = load_gold_dataset(SMALL)
    split = grouped_train_test_split(ds.records, seed=42, test_ratio=0.4)
    _, test_vec, fb = build_split_features(split.train, split.test)
    vocab = set(fb.skill_vocab)
    for vec in test_vec:
        indicator_names = {name for name, _ in vec.skill_indicators}
        assert indicator_names == vocab


def test_features_are_deterministic():
    ds = load_gold_dataset(SMALL)
    split = grouped_train_test_split(ds.records, seed=42, test_ratio=0.4)
    a_train, a_test, _ = build_split_features(split.train, split.test)
    b_train, b_test, _ = build_split_features(split.train, split.test)
    assert [v.to_dict() for v in a_train] == [v.to_dict() for v in b_train]
    assert [v.to_dict() for v in a_test] == [v.to_dict() for v in b_test]
