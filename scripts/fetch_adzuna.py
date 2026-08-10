#!/usr/bin/env python3
"""
Fetch a small Adzuna job sample and run it through Tekmérion ingestion + pipeline.

Usage examples
--------------
  # Requires ADZUNA_APP_ID and ADZUNA_API_KEY in the environment
  python scripts/fetch_adzuna.py
  python scripts/fetch_adzuna.py --what "data scientist" --country ar --limit 10
  python scripts/fetch_adzuna.py --what "machine learning" --country gb --save-snapshot

Does NOT start Flask. Does not page beyond a single page by default.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.ingestion import (
    IngestionContext,
    ingest,
)
from analysis.ingestion.adzuna import (
    AdzunaClient,
    AdzunaConfigError,
    AdzunaSource,
    load_credentials_from_env,
    save_raw_snapshot,
    DEFAULT_COUNTRY,
    DEFAULT_RESULTS_PER_PAGE,
)
from analysis.pipeline import process_records
from analysis.evidence import build_evidence


SNAPSHOT_DIR = ROOT / "data" / "raw" / "real" / "adzuna"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Adzuna jobs into Tekmérion pipeline")
    p.add_argument("--what", default="data analyst", help="Search keywords (default: data analyst)")
    p.add_argument("--country", default=DEFAULT_COUNTRY, help="Adzuna country code (default: ar)")
    p.add_argument("--where", default=None, help="Optional location filter")
    p.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_RESULTS_PER_PAGE,
        help=f"Results per page, max 50 (default: {DEFAULT_RESULTS_PER_PAGE})",
    )
    p.add_argument("--page", type=int, default=1, help="Page number (default: 1)")
    p.add_argument(
        "--save-snapshot",
        action="store_true",
        help=f"Write raw response under {SNAPSHOT_DIR}",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        creds = load_credentials_from_env()
    except AdzunaConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "Set ADZUNA_APP_ID and ADZUNA_API_KEY (see .env.example).",
            file=sys.stderr,
        )
        return 1

    client = AdzunaClient(creds)
    source = AdzunaSource(
        client=client,
        what=args.what,
        country=args.country,
        page=args.page,
        results_per_page=args.limit,
        where=args.where,
    )

    print(f"Fetching Adzuna: country={args.country} what={args.what!r} "
          f"page={args.page} limit={args.limit}")

    try:
        raw_rows = source.load()
    except Exception as exc:  # noqa: BLE001 — surface cleanly to CLI users
        print(f"ERROR: fetch failed: {exc}", file=sys.stderr)
        return 2

    snapshot_path = None
    if args.save_snapshot and source.last_raw_payload is not None:
        snapshot_path = save_raw_snapshot(
            source.last_raw_payload,
            directory=SNAPSHOT_DIR,
            retrieved_at=retrieved_at,
            country=args.country,
            query=args.what,
            page=args.page,
        )

    ctx = IngestionContext(retrieved_at=retrieved_at)
    # Re-ingest via the adapter with preloaded rows so mapping stays consistent
    offline = AdzunaSource(
        preloaded_records=raw_rows,
        what=args.what,
        country=args.country,
        page=args.page,
        results_per_page=args.limit,
    )
    ing = ingest([offline], context=ctx)
    pipe = process_records(ing.records)
    evidence = build_evidence(pipe.records)

    print("\n=== Summary ===")
    print(f"  received (API mapped): {len(raw_rows)}")
    print(f"  ingested:              {ing.accepted_count}")
    print(f"  valid:                 {pipe.valid_count}")
    print(f"  invalid:               {pipe.invalid_count}")
    print(f"  duplicates:            {pipe.duplicate_count}")
    print(f"  role families:         {pipe.role_family_counts}")
    print(f"  seniority:             {pipe.seniority_counts}")

    top_skills = evidence.skill_frequency[:8]
    if top_skills:
        print("  top skills:")
        for item in top_skills:
            print(f"    - {item['item']}: {item['count']}")

    if snapshot_path:
        print(f"  snapshot:              {snapshot_path}")

    print(f"\n  retrieved_at:          {retrieved_at}")
    print(f"  source:                adzuna")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
