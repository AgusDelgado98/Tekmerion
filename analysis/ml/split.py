"""Reproducible grouped train/test split with leakage guards."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Sequence

from analysis.ml.models import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_RATIO,
    RoleFamilyExample,
    SplitResult,
)


class SplitLeakageError(ValueError):
    """Train and test share identity or near-duplicate content."""


def example_fingerprint(title: str, description: str) -> str:
    """
    Stable near-duplicate key over classification inputs.

    Title + a description prefix. Company is excluded so the same posting
    text cannot leak across the split via a different employer field.
    """
    norm_title = " ".join((title or "").strip().lower().split())
    desc_part = " ".join((description or "")[:300].strip().lower().split())
    payload = f"{norm_title}|{desc_part}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def assert_no_split_leakage(
    train: Sequence[RoleFamilyExample],
    test: Sequence[RoleFamilyExample],
) -> None:
    train_ids = {e.id for e in train}
    test_ids = {e.id for e in test}
    id_overlap = train_ids & test_ids
    if id_overlap:
        raise SplitLeakageError(f"id leakage: {sorted(id_overlap)}")

    train_fp = {e.content_fingerprint for e in train}
    test_fp = {e.content_fingerprint for e in test}
    fp_overlap = train_fp & test_fp
    if fp_overlap:
        raise SplitLeakageError(
            f"content fingerprint leakage (near-duplicates across split): "
            f"{sorted(fp_overlap)}"
        )


def grouped_train_test_split(
    examples: Sequence[RoleFamilyExample],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> SplitResult:
    """
    Shuffle *groups* (content fingerprints), then assign whole groups.

    Two records with the same title+description prefix always stay together.
    Stratification is not applied: several gold classes currently have a
    single example, which would make a stratified split undefined or leaky.
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1 exclusive")
    if not examples:
        raise ValueError("cannot split an empty example list")

    groups: dict[str, list[RoleFamilyExample]] = defaultdict(list)
    for ex in examples:
        groups[ex.content_fingerprint].append(ex)

    # Stable group order before shuffle so the same seed is reproducible
    # even if input record order changes.
    group_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    warnings: list[str] = []
    n_groups = len(group_keys)
    if n_groups == 1:
        warnings.append(
            "only one content group; all examples assigned to train to avoid leakage"
        )
        n_test_groups = 0
    else:
        n_test_groups = int(round(n_groups * test_ratio))
        n_test_groups = min(max(n_test_groups, 1), n_groups - 1)

    test_keys = set(group_keys[:n_test_groups])
    train_list: list[RoleFamilyExample] = []
    test_list: list[RoleFamilyExample] = []
    for key in sorted(groups.keys()):
        bucket = sorted(groups[key], key=lambda e: e.id)
        if key in test_keys:
            test_list.extend(bucket)
        else:
            train_list.extend(bucket)

    train_list.sort(key=lambda e: e.id)
    test_list.sort(key=lambda e: e.id)

    singleton_labels = _singleton_labels(examples)
    if singleton_labels:
        warnings.append(
            "classes with a single labeled example (no reliable stratification): "
            + ", ".join(singleton_labels)
        )

    result = SplitResult(
        seed=seed,
        test_ratio=test_ratio,
        strategy="grouped_shuffle_by_content_fingerprint",
        train=tuple(train_list),
        test=tuple(test_list),
        grouped_by="content_fingerprint",
        warnings=tuple(warnings),
    )
    assert_no_split_leakage(result.train, result.test)
    return result


def _singleton_labels(examples: Sequence[RoleFamilyExample]) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for ex in examples:
        counts[ex.gold_role_family.value] += 1
    return sorted(k for k, n in counts.items() if n == 1)
