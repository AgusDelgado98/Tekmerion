#!/usr/bin/env python3
"""
Run a multi-query Adzuna market batch and optionally persist artifacts.

Examples
--------
  # Offline is not supported here — this script is for live runs.
  # Requires ADZUNA_APP_ID and ADZUNA_API_KEY.

  python scripts/fetch_market.py
  python scripts/fetch_market.py --country ar --limit-per-query 5
  python scripts/fetch_market.py --query "data analyst" --query "data engineer"
  python scripts/fetch_market.py --save-raw --save-market
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.ingestion.adzuna import (  # noqa: E402
    AdzunaClient,
    AdzunaConfigError,
    load_credentials_from_env,
    DEFAULT_COUNTRY,
)
from analysis.ingestion.market import (  # noqa: E402
    DEFAULT_MARKET_QUERIES,
    run_market_batch,
    save_market_artifact,
    save_batch_raw_snapshots,
)

RAW_DIR = ROOT / "data" / "raw" / "real" / "adzuna"
MARKET_DIR = ROOT / "data" / "processed" / "market"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tekmérion multi-query Adzuna market batch")
    p.add_argument("--country", default=DEFAULT_COUNTRY, help="Adzuna country code")
    p.add_argument(
        "--limit-per-query",
        type=int,
        default=5,
        help="Max results per query (default 5, hard cap 50)",
    )
    p.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Custom query (repeatable). Default: role-family preset",
    )
    p.add_argument("--save-raw", action="store_true", help="Save per-query raw snapshots")
    p.add_argument(
        "--save-market",
        action="store_true",
        help=f"Save consolidated market artifact under {MARKET_DIR}",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    queries = args.queries if args.queries else list(DEFAULT_MARKET_QUERIES)

    try:
        creds = load_credentials_from_env()
    except AdzunaConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Set ADZUNA_APP_ID and ADZUNA_API_KEY (see .env.example).", file=sys.stderr)
        return 1

    client = AdzunaClient(creds)

    print(
        f"Market batch: country={args.country} "
        f"queries={len(queries)} limit_per_query={args.limit_per_query}"
    )
    for q in queries:
        print(f"  - {q}")

    try:
        result = run_market_batch(
            country=args.country,
            queries=queries,
            limit_per_query=args.limit_per_query,
            retrieved_at=retrieved_at,
            client=client,
            run_pipeline=True,
            fail_fast=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: batch failed: {exc}", file=sys.stderr)
        return 2

    raw_paths = []
    if args.save_raw:
        raw_paths = save_batch_raw_snapshots(result, directory=RAW_DIR)

    market_path = None
    if args.save_market:
        market_path = save_market_artifact(result, directory=MARKET_DIR)

    print("\n=== Market batch summary ===")
    print(f"  retrieved_at:       {result.retrieved_at}")
    print(f"  queries:            {len(result.queries)}")
    for q, n in ((o.query.what, o.received_count) for o in result.query_outcomes):
        print(f"    {q!r}: {n}")
    print(f"  total received:     {result.total_received}")
    print(f"  unique:             {result.unique_count}")
    print(f"  duplicates removed: {result.duplicates_removed}")

    if result.pipeline_result:
        ps = result.pipeline_result.summary()
        print(f"  valid:              {ps['valid_count']}")
        print(f"  invalid:            {ps['invalid_count']}")
        print(f"  role families:      {ps['role_family_counts']}")
        print(f"  seniority:          {ps['seniority_counts']}")

    if result.evidence:
        print("  top skills:")
        for item in result.evidence.skill_frequency[:8]:
            print(f"    - {item['item']}: {item['count']}")

    if raw_paths:
        print(f"  raw snapshots:      {len(raw_paths)} files under {RAW_DIR}")
    if market_path:
        print(f"  market artifact:    {market_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
