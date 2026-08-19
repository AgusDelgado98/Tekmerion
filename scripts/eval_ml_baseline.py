#!/usr/bin/env python3
"""Score the deterministic role-family baseline against the Gold Dataset (offline)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.ml.evaluate import evaluate_gold_dataset
from analysis.ml.gold import DEFAULT_GOLD_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help="Path to tekmerion.ml.gold_dataset.v1 JSON",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "ml" / "artifacts" / "evaluation_manifest.json",
    )
    args = parser.parse_args()

    result, manifest = evaluate_gold_dataset(
        args.gold,
        seed=args.seed,
        test_ratio=args.test_ratio,
        manifest_path=args.manifest,
    )
    print("=== ML evaluation (baseline, offline) ===")
    print(f"  gold:        {manifest.dataset_path}")
    print(f"  sha256:      {manifest.dataset_sha256}")
    print(f"  n:           {manifest.dataset_n}")
    print(f"  train/test:  {manifest.split['n_train']}/{manifest.split['n_test']}")
    print(f"  seed:        {manifest.seed}")
    print(f"  predictor:   {manifest.predictor}")
    print(f"  accuracy:    {result.metrics.accuracy:.4f}")
    print(f"  macro_f1:    {result.metrics.macro_f1:.4f}")
    print(f"  sufficient:  {manifest.dataset_sufficient_for_training}")
    print(f"  manifest:    {args.manifest}")
    if not manifest.dataset_sufficient_for_training:
        print("  NOTE: gold sample is not sufficient for training (documented).")


if __name__ == "__main__":
    main()
