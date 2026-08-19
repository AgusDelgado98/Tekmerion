#!/usr/bin/env python3
"""Harvest unlabeled gold candidates from local ingestion sources (offline)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.ml.harvest import DEFAULT_CANDIDATES_PATH, harvest_unlabeled_candidates, write_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_CANDIDATES_PATH)
    args = parser.parse_args()
    payload = harvest_unlabeled_candidates()
    path = write_candidates(payload, args.out)
    print("=== Unlabeled candidate harvest (offline) ===")
    print(f"  loaded:     {payload['n_loaded']}")
    print(f"  unique:     {payload['n_unique']}")
    print(f"  dropped:    {payload['n_dropped_duplicates']}")
    print(f"  by kind:    {payload['unique_by_source_kind']}")
    print(f"  wrote:      {path}")
    print("  NOTE: records are unlabeled; do not copy classifier output as gold.")


if __name__ == "__main__":
    main()
