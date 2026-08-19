#!/usr/bin/env python3
"""Fetch live Adzuna jobs into unlabeled snapshots (no gold_role_family)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.ml.evaluate import dump_canonical_json
from analysis.ml.fetch_candidates import (
    DATA_ANALYST_GAP_QUERIES,
    DEFAULT_GOLD_FETCH_COUNTRIES,
    DEFAULT_GOLD_FETCH_QUERIES,
    adzuna_credentials_status,
    fetch_adzuna_snapshots,
)
from analysis.ml.gold import load_gold_dataset
from analysis.ml.harvest import DEFAULT_CANDIDATES_PATH, harvest_unlabeled_candidates, write_candidates
from analysis.ml.label import expansion_report, unlabeled_queue

_ADZUNA_ENV_KEYS = ("ADZUNA_APP_ID", "ADZUNA_API_KEY")


def _load_dotenv_adzuna(root: Path) -> str:
    """Load Adzuna keys from .env into os.environ if missing. Never logs values."""
    path = root / ".env"
    if not path.is_file():
        return "no_dotenv"
    loaded = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key not in _ADZUNA_ENV_KEYS:
            continue
        if os.environ.get(key, "").strip():
            continue
        val = val.strip().strip('"').strip("'")
        if val:
            os.environ[key] = val
            loaded += 1
    return f"dotenv_keys_applied={loaded}"


def _public_fetch(fetch: dict) -> dict:
    """Drop records so logs never include full postings dump or secrets."""
    return {k: v for k, v in fetch.items() if k != "records"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", action="append", dest="countries")
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Search string (repeatable). Context only — never used as gold_role_family.",
    )
    parser.add_argument(
        "--data-analyst-gap",
        action="store_true",
        help="Use DATA_ANALYST_GAP_QUERIES instead of the default family list.",
    )
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--candidates-out", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data" / "ml" / "reports" / "gold_expansion.json",
    )
    args = parser.parse_args()

    _load_dotenv_adzuna(ROOT)
    creds = adzuna_credentials_status()
    print("=== Credentials ===")
    print(f"  ADZUNA_APP_ID set: {bool(creds.get('available'))}")
    print(f"  source: env ({creds.get('reason')})")

    if args.queries:
        queries = tuple(args.queries)
    elif args.data_analyst_gap:
        queries = DATA_ANALYST_GAP_QUERIES
    else:
        queries = DEFAULT_GOLD_FETCH_QUERIES
    fetch = fetch_adzuna_snapshots(
        queries=queries,
        countries=tuple(args.countries) if args.countries else DEFAULT_GOLD_FETCH_COUNTRIES,
        pages=args.pages,
        results_per_page=args.limit,
    )
    print("=== Live Adzuna fetch (unlabeled) ===")
    print(json.dumps(_public_fetch(fetch), ensure_ascii=False, indent=2))

    harvest = harvest_unlabeled_candidates()
    write_candidates(harvest, args.candidates_out)
    gold = load_gold_dataset()
    queue = unlabeled_queue(harvest.get("records") or [], gold, real_only=True)
    report = expansion_report(
        gold,
        harvest=harvest,
        fetch={
            "live_attempted": True,
            "credentials": {"available": creds.get("available"), "reason": creds.get("reason")},
            **_public_fetch(fetch),
        },
        queue_n=len(queue),
    )
    dump_canonical_json(report, args.report)

    n_snap = harvest.get("unique_by_source_kind", {}).get("adzuna_snapshot", 0)
    print("=== Harvest after fetch ===")
    print(f"  unique_all:     {harvest['n_unique']}")
    print(f"  dropped_dupes:  {harvest['n_dropped_duplicates']}")
    print(f"  by kind:        {harvest['unique_by_source_kind']}")
    print(f"  adzuna_snapshot unique in harvest: {n_snap}")
    print(f"  unlabeled real queue: {len(queue)}")
    print(f"  report: {args.report}")
    if not fetch.get("fetched"):
        print("  LIVE FETCH SKIPPED — set ADZUNA_APP_ID and ADZUNA_API_KEY in the process env.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
