#!/usr/bin/env python3
"""
Run the full Tekmérion evidence pipeline on the synthetic sample.

Flow:
  sample_jobs.json → pipeline → processed records → evidence layer → evidence.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.pipeline import process_file
from analysis.evidence import build_evidence, compare_roles


def main() -> None:
    input_path = ROOT / "data" / "raw" / "sample_jobs.json"
    processed_path = ROOT / "data" / "processed" / "sample_processed.json"
    evidence_path = ROOT / "data" / "processed" / "evidence.json"

    print("=== Tekmérion V0.2 — Evidence ===\n")
    print(f"1. Pipeline: {input_path.name}")
    result = process_file(input_path, processed_path)
    print(f"   → {result.valid_count} valid, {result.duplicate_count} duplicates, "
          f"{result.invalid_count} invalid")

    print("\n2. Evidence layer (valid & non-duplicate only)")
    report = build_evidence(result.records)
    print(f"   → analysis records: {report.n_analysis_records}")

    # Persist full report
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with evidence_path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"   → written: {evidence_path}")

    # Console summary
    print("\n--- Role distribution ---")
    for d in report.role_distribution:
        print(f"  {d['item']:20} {d['count']:3}  ({d['proportion']:.1%})")

    print("\n--- Seniority distribution ---")
    for d in report.seniority_distribution:
        print(f"  {d['item']:20} {d['count']:3}  ({d['proportion']:.1%})")

    print("\n--- Top skills (global) ---")
    for d in report.skill_frequency[:10]:
        print(f"  {d['item']:20} {d['count']:3}  ({d['proportion']:.1%})")

    print("\n--- Top co-occurrences ---")
    for p in report.skill_cooccurrence[:8]:
        print(f"  {p['skill_a']} + {p['skill_b']}: {p['count']}")

    # Example comparison
    print("\n--- Example: data_analyst vs bi_analyst ---")
    cmp = compare_roles(result.records, "data_analyst", "bi_analyst")
    print(f"  counts: {cmp['count_a']} vs {cmp['count_b']}")
    print(f"  common: {', '.join(cmp['common_skills']) or '(none)'}")
    print(f"  only data_analyst: {', '.join(cmp['only_in_a']) or '(none)'}")
    print(f"  only bi_analyst:   {', '.join(cmp['only_in_b']) or '(none)'}")

    print("\n(Note: metrics are computed on the synthetic sample of 17 records.")
    print(" They demonstrate the engine; they do not represent the real market.)")


if __name__ == "__main__":
    main()
