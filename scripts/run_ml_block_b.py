#!/usr/bin/env python3
"""Block B: harvest + sufficiency gate + optional sklearn training (offline)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.ml.compare import DEFAULT_REPORT_PATH, run_block_b
from analysis.ml.gold import DEFAULT_GOLD_PATH
from analysis.ml.harvest import DEFAULT_CANDIDATES_PATH, harvest_unlabeled_candidates, write_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--no-harvest-file", action="store_true")
    args = parser.parse_args()

    if not args.no_harvest_file:
        write_candidates(harvest_unlabeled_candidates(), args.candidates)

    report = run_block_b(
        args.gold,
        seed=args.seed,
        test_ratio=args.test_ratio,
        report_path=args.report,
        harvest=True,
        allow_train=True,
    )
    print("=== Block B (Rules vs ML) ===")
    print(f"  status:     {report['status']}")
    print(f"  gold n:     {report['dataset_n']}")
    print(f"  sha256:     {report['dataset_sha256']}")
    print(f"  trained:    {report['training_ran']}")
    if report.get("harvest"):
        print(f"  unlabeled:  {report['harvest']['n_unique']} unique candidates")
    print(f"  models:     {report['models']}")
    print(f"  winner:     {report.get('winner')}")
    print(f"  report:     {args.report}")
    if report.get("artifacts"):
        print(f"  manifests:  {len(report['artifacts'])} files")
    if not report.get("winner", {}).get("promote_ml"):
        print("  promote_ml=false — Evidence / pipeline unchanged.")
    if not report.get("training_ran"):
        print("  sklearn training was not run.")


if __name__ == "__main__":
    main()
