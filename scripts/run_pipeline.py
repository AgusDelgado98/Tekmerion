#!/usr/bin/env python3
"""Simple CLI to run the Tekmérion pipeline on the sample data."""

from pathlib import Path
import sys

# Ensure project root is on path when run as script
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.pipeline import process_file


def main() -> None:
    input_path = ROOT / "data" / "raw" / "sample_jobs.json"
    output_path = ROOT / "data" / "processed" / "sample_processed.json"

    print(f"Processing: {input_path}")
    result = process_file(input_path, output_path)

    print("\n=== Pipeline summary ===")
    for k, v in result.summary().items():
        print(f"  {k}: {v}")

    print(f"\nProcessed file written to: {output_path}")


if __name__ == "__main__":
    main()
