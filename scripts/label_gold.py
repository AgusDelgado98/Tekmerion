#!/usr/bin/env python3
"""Human gold_role_family labeling. Does not call classify_role_family or ML."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.ml.evaluate import dump_canonical_json
from analysis.ml.fetch_candidates import adzuna_credentials_status
from analysis.ml.gate import class_distribution
from analysis.ml.gold import DEFAULT_GOLD_PATH, dump_gold_dataset, load_gold_dataset
from analysis.ml.harvest import DEFAULT_CANDIDATES_PATH
from analysis.ml.label import (
    DEFAULT_SESSION_PATH,
    apply_human_labels,
    expansion_report,
    format_label_card,
    labels_from_session,
    load_candidates_file,
    load_human_labels,
    load_label_session,
    record_session_decision,
    save_label_session,
    session_reviewed_ids,
    session_stats,
    unlabeled_queue,
)
from analysis.models import RoleFamily


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _queue(gold, harvest, session, *, real_only: bool):
    return unlabeled_queue(
        harvest.get("records") or [],
        gold,
        real_only=real_only,
        extra_skip_ids=session_reviewed_ids(session),
    )


def _sync_gold(gold, harvest, session, gold_path: Path):
    labels = labels_from_session(session)
    merged, stats = apply_human_labels(gold, harvest.get("records") or [], labels)
    dump_gold_dataset(merged, gold_path)
    return merged, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD_PATH)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION_PATH)
    parser.add_argument("--print-queue", action="store_true", help="Show next unlabeled real cards")
    parser.add_argument("--show-next", action="store_true", help="Show a single next card")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--include-non-real", action="store_true")
    parser.add_argument("--id", dest="decide_id", help="Candidate id to decide")
    parser.add_argument("--label", dest="decide_label", help="Role family or skip/ambiguous")
    parser.add_argument("--notes", default="")
    parser.add_argument("--apply", type=Path, help="JSON human labels to merge into gold")
    parser.add_argument("--import-decisions", type=Path, help="Session-style JSON list of decisions")
    parser.add_argument("--sync-gold", action="store_true", help="Apply session labels into gold")
    parser.add_argument("--report", type=Path, default=ROOT / "data" / "ml" / "reports" / "gold_expansion.json")
    args = parser.parse_args()

    gold = load_gold_dataset(args.gold)
    harvest = load_candidates_file(args.candidates)
    session = load_label_session(args.session)
    queue = _queue(gold, harvest, session, real_only=not args.include_non_real)
    dist = class_distribution(gold)

    if args.show_next or args.print_queue:
        n = 1 if args.show_next else args.limit
        qname = "unlabeled queue" if args.include_non_real else "unlabeled real queue"
        print(f"gold n={gold.n}  dist={dict(dist)}")
        print(f"{qname}={len(queue)}  session={session_stats(session)}")
        if not queue:
            print("Queue empty (all labeled, skipped, or no real candidates).")
        for i, cand in enumerate(queue[:n], start=1):
            print(format_label_card(cand, distribution=dict(dist), index=i, total=len(queue)))
            print()

    if args.import_decisions:
        payload = json.loads(args.import_decisions.read_text(encoding="utf-8"))
        rows = payload.get("decisions") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            print("import-decisions must be a list or {decisions: []}", file=sys.stderr)
            return 2
        for item in rows:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("id") or "")
            if cid in session_reviewed_ids(session):
                continue
            record_session_decision(
                session,
                candidate_id=cid,
                decision=str(item.get("decision") or item.get("gold_role_family") or "skip"),
                gold_role_family=str(item.get("gold_role_family") or ""),
                notes=str(item.get("notes") or ""),
                labeled_at=str(item.get("labeled_at") or _now()),
            )
        save_label_session(session, args.session)
        gold, stats = _sync_gold(gold, harvest, session, args.gold)
        print("=== Imported decisions (session saved, gold synced) ===")
        print(json.dumps({**session_stats(session), **stats}, indent=2))
        queue = _queue(gold, harvest, session, real_only=not args.include_non_real)

    if args.decide_id:
        if not args.decide_label:
            print("--label is required with --id (family or skip)", file=sys.stderr)
            return 2
        record_session_decision(
            session,
            candidate_id=args.decide_id,
            decision=args.decide_label,
            notes=args.notes,
            labeled_at=_now(),
        )
        save_label_session(session, args.session)
        gold, stats = _sync_gold(gold, harvest, session, args.gold)
        print("=== Recorded (session + gold) ===")
        print(json.dumps({**session_stats(session), **stats}, indent=2))
        queue = _queue(gold, harvest, session, real_only=not args.include_non_real)

    if args.apply:
        labels = load_human_labels(args.apply)
        gold, stats = apply_human_labels(gold, harvest.get("records") or [], labels)
        dump_gold_dataset(gold, args.gold)
        print("=== Applied human labels ===")
        print(json.dumps(stats, indent=2))
        queue = _queue(gold, harvest, session, real_only=not args.include_non_real)

    if args.sync_gold and not args.decide_id and not args.import_decisions:
        gold, stats = _sync_gold(gold, harvest, session, args.gold)
        print("=== Synced session → gold ===")
        print(json.dumps(stats, indent=2))
        queue = _queue(gold, harvest, session, real_only=not args.include_non_real)

    report = expansion_report(
        gold,
        harvest=harvest,
        fetch={"live_attempted": False, "credentials": adzuna_credentials_status()},
        queue_n=len(queue),
    )
    report["session"] = session_stats(session)
    dump_canonical_json(report, args.report)
    print("=== Gate ===")
    print(report["gate"]["message"])
    print(f"  session: {session_stats(session)}")
    print(f"  pending: {len(queue)}")
    print(f"  report: {args.report}")
    valid = ", ".join(m.value for m in RoleFamily if m != RoleFamily.UNKNOWN)
    print(f"  choices: {valid} | skip/ambiguous")
    # Labeling CLI succeeds even when the training gate is closed.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
