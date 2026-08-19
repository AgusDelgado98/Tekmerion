"""Reproducible grouped split and leakage protection."""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis.ml.gold import load_gold_dataset
from analysis.ml.models import DEFAULT_SPLIT_SEED
from analysis.ml.split import (
    SplitLeakageError,
    assert_no_split_leakage,
    grouped_train_test_split,
)

SMALL = Path("tests/fixtures/ml/gold_small.json")


def test_split_is_reproducible_with_fixed_seed():
    ds = load_gold_dataset(SMALL)
    a = grouped_train_test_split(ds.records, seed=DEFAULT_SPLIT_SEED, test_ratio=0.4)
    b = grouped_train_test_split(ds.records, seed=DEFAULT_SPLIT_SEED, test_ratio=0.4)
    assert a.train_ids == b.train_ids
    assert a.test_ids == b.test_ids
    assert a.strategy == "grouped_shuffle_by_content_fingerprint"


def test_split_ignores_input_order():
    ds = load_gold_dataset(SMALL)
    reversed_records = tuple(reversed(ds.records))
    a = grouped_train_test_split(ds.records, seed=7, test_ratio=0.4)
    b = grouped_train_test_split(reversed_records, seed=7, test_ratio=0.4)
    assert a.train_ids == b.train_ids
    assert a.test_ids == b.test_ids


def test_near_duplicates_stay_on_same_side():
    ds = load_gold_dataset(SMALL)
    split = grouped_train_test_split(ds.records, seed=DEFAULT_SPLIT_SEED, test_ratio=0.4)
    sides = {}
    for ex in split.train:
        sides[ex.id] = "train"
    for ex in split.test:
        sides[ex.id] = "test"
    assert sides["fx_a1"] == sides["fx_a2"]


def test_no_id_or_fingerprint_overlap():
    ds = load_gold_dataset(SMALL)
    split = grouped_train_test_split(ds.records, seed=0, test_ratio=0.4)
    assert_no_split_leakage(split.train, split.test)
    assert set(split.train_ids).isdisjoint(split.test_ids)
    assert {e.content_fingerprint for e in split.train}.isdisjoint(
        {e.content_fingerprint for e in split.test}
    )


def test_leakage_guard_raises_on_shared_id():
    ds = load_gold_dataset(SMALL)
    ex = ds.records[0]
    with pytest.raises(SplitLeakageError, match="id leakage"):
        assert_no_split_leakage([ex], [ex])


def test_empty_split_rejected():
    with pytest.raises(ValueError, match="empty"):
        grouped_train_test_split([], seed=1, test_ratio=0.3)
