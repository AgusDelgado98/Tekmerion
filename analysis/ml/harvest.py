"""
Harvest unlabeled vacancy candidates from existing ingestion sources.

Deduplicates by content fingerprint *before* any human labeling.
Never copies pipeline ``role_family`` or classifier output onto candidates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

from analysis.ingestion import (
    DEFAULT_REAL_SAMPLE,
    DEFAULT_SYNTHETIC_SAMPLE,
    ingest_local_file,
    map_adzuna_results,
)
from analysis.ingestion.base import IngestionContext
from analysis.ml.models import FORBIDDEN_GOLD_KEYS
from analysis.ml.split import example_fingerprint

CANDIDATE_SCHEMA = "tekmerion.ml.gold_candidates.v1"
STRIP_FIELDS = FORBIDDEN_GOLD_KEYS | {
    "gold_role_family",
    "seniority",
    "skills_extracted",
    "normalized_title",
    "is_valid",
    "validation_errors",
    "is_duplicate",
    "duplicate_of",
}

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHOWROOM = _ROOT / "data" / "showroom" / "showroom_market_ar.json"
DEFAULT_ADZUNA_FIXTURES = _ROOT / "tests" / "fixtures" / "adzuna"
DEFAULT_ADZUNA_SNAPSHOTS = _ROOT / "data" / "raw" / "real" / "adzuna"
DEFAULT_CANDIDATES_PATH = _ROOT / "data" / "ml" / "gold" / "candidates_unlabeled_v1.json"

# Frozen so harvest of repo sources is reproducible in tests.
HARVEST_RETRIEVED_AT = "2026-08-19T00:00:00Z"


def _namespaced_id(source_kind: str, record_id: str) -> str:
    record_id = (record_id or "").strip() or "unknown"
    prefix = f"{source_kind}:"
    if record_id.startswith(prefix):
        return record_id
    return f"{prefix}{record_id}"


class HarvestError(ValueError):
    """Invalid unlabeled candidate payload."""


def _strip_labels(raw: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in raw.items() if k not in STRIP_FIELDS}


def _candidate(
    *,
    record_id: str,
    title: str,
    description: str,
    company: str = "",
    location: str = "",
    source: str = "",
    source_kind: str = "",
    source_ref: str = "",
    source_url: str = "",
    retrieved_at: str = "",
    source_record_id: str = "",
) -> Optional[dict[str, Any]]:
    title = (title or "").strip()
    description = (description or "").strip()
    if not title or not description:
        return None
    row = {
        "id": record_id,
        "title": title,
        "description": description,
        "company": (company or "").strip(),
        "location": (location or "").strip(),
        "source": (source or "").strip(),
        "source_kind": source_kind,
        "source_ref": source_ref,
        "source_url": (source_url or "").strip(),
        "retrieved_at": (retrieved_at or "").strip(),
        "source_record_id": (source_record_id or "").strip(),
        "content_fingerprint": example_fingerprint(title, description),
        "label_status": "unlabeled",
    }
    leak = STRIP_FIELDS.intersection(row.keys())
    if leak:
        raise HarvestError(f"candidate would carry forbidden fields: {sorted(leak)}")
    return row


def _from_ingested(records: Iterable[dict[str, Any]], *, source_kind: str, source_ref: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in records:
        clean = _strip_labels(raw if isinstance(raw, dict) else {})
        orig_id = str(clean.get("id") or "").strip()
        cand = _candidate(
            record_id=_namespaced_id(source_kind, orig_id),
            title=str(clean.get("title") or ""),
            description=str(clean.get("description") or ""),
            company=str(clean.get("company") or ""),
            location=str(clean.get("location") or ""),
            source=str(clean.get("source") or source_kind),
            source_kind=source_kind,
            source_ref=source_ref,
            source_url=str(clean.get("source_url") or ""),
            retrieved_at=str(clean.get("retrieved_at") or ""),
            source_record_id=str(clean.get("source_record_id") or orig_id),
        )
        if cand:
            out.append(cand)
    return out


def _load_showroom(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return []
    return _from_ingested(records, source_kind="showroom_demo", source_ref=str(path).replace("\\", "/"))


def _iter_adzuna_json_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    files = [p for p in directory.rglob("*.json") if p.name != "query_empty.json"]
    return sorted(files)


def _payload_from_adzuna_file(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("payload"), dict):
        return data["payload"]
    if isinstance(data, dict):
        return data
    return {}


def _load_adzuna_files(directory: Path, *, source_kind: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in _iter_adzuna_json_files(directory):
        rel = str(path).replace("\\", "/")
        try:
            mapped = map_adzuna_results(_payload_from_adzuna_file(path))
        except Exception:
            continue
        out.extend(
            _from_ingested(
                mapped,
                source_kind=source_kind,
                source_ref=rel,
            )
        )
    return out


def dedupe_candidates(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep first record per content fingerprint (stable: source_kind, id)."""
    ordered = sorted(
        candidates,
        key=lambda r: (str(r.get("source_kind") or ""), str(r.get("id") or "")),
    )
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    dropped = 0
    for row in ordered:
        fp = str(row.get("content_fingerprint") or "")
        if not fp or fp in seen:
            dropped += 1
            continue
        seen.add(fp)
        unique.append(row)
    unique.sort(key=lambda r: (str(r.get("source_kind") or ""), str(r.get("id") or "")))
    seen_ids: set[str] = set()
    with_unique_ids: list[dict[str, Any]] = []
    for row in unique:
        cid = str(row.get("id") or "")
        if cid in seen_ids:
            fp = str(row.get("content_fingerprint") or "")[:8]
            row = dict(row)
            row["id"] = f"{cid}~{fp}"
            cid = row["id"]
        seen_ids.add(cid)
        with_unique_ids.append(row)
    return with_unique_ids, dropped


def harvest_unlabeled_candidates(
    *,
    include_synthetic: bool = True,
    include_curated_real: bool = True,
    include_showroom: bool = True,
    include_adzuna_fixtures: bool = True,
    include_adzuna_snapshots: bool = True,
) -> dict[str, Any]:
    """
    Load available local sources, strip labels, dedupe.

    Does not call live APIs. Live Adzuna snapshots are included only if
    already present under data/raw/real/adzuna/.
    """
    ctx = IngestionContext(retrieved_at=HARVEST_RETRIEVED_AT)
    collected: list[dict[str, Any]] = []
    sources_used: list[str] = []

    if include_curated_real and DEFAULT_REAL_SAMPLE.exists():
        ingested = ingest_local_file(
            DEFAULT_REAL_SAMPLE,
            source_name="curated_real_sample",
            context=ctx,
        )
        collected.extend(
            _from_ingested(
                ingested.records,
                source_kind="curated_real_sample",
                source_ref="data/raw/real/sample_real_jobs.json",
            )
        )
        sources_used.append("curated_real_sample")

    if include_synthetic and DEFAULT_SYNTHETIC_SAMPLE.exists():
        ingested = ingest_local_file(
            DEFAULT_SYNTHETIC_SAMPLE,
            source_name="synthetic",
            context=ctx,
        )
        collected.extend(
            _from_ingested(
                ingested.records,
                source_kind="synthetic",
                source_ref="data/raw/sample_jobs.json",
            )
        )
        sources_used.append("synthetic")

    if include_showroom:
        collected.extend(_load_showroom(DEFAULT_SHOWROOM))
        if DEFAULT_SHOWROOM.exists():
            sources_used.append("showroom_demo")

    if include_adzuna_fixtures:
        collected.extend(
            _load_adzuna_files(DEFAULT_ADZUNA_FIXTURES, source_kind="adzuna_test_fixture")
        )
        sources_used.append("adzuna_test_fixture")

    if include_adzuna_snapshots:
        collected.extend(
            _load_adzuna_files(DEFAULT_ADZUNA_SNAPSHOTS, source_kind="adzuna_snapshot")
        )
        sources_used.append("adzuna_snapshot")

    unique, dropped = dedupe_candidates(collected)
    by_kind: dict[str, int] = {}
    for row in unique:
        kind = str(row.get("source_kind") or "unknown")
        by_kind[kind] = by_kind.get(kind, 0) + 1

    return {
        "schema": CANDIDATE_SCHEMA,
        "label_status": "unlabeled",
        "label_policy": (
            "Candidates are unlabeled. Human gold_role_family must be assigned "
            "later by reading title/description. Classifier output is not stored."
        ),
        "n_loaded": len(collected),
        "n_unique": len(unique),
        "n_dropped_duplicates": dropped,
        "sources_used": sources_used,
        "unique_by_source_kind": dict(sorted(by_kind.items())),
        "limitations": [
            "Offline harvest only; no live Adzuna call in this path.",
            "Showroom and test fixtures are not live market snapshots.",
            "Unique N after dedupe is the upper bound for new human labels without a live fetch.",
        ],
        "records": unique,
    }


def write_candidates(payload: dict[str, Any], path: str | Path | None = None) -> Path:
    out = Path(path) if path is not None else DEFAULT_CANDIDATES_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    out.write_text(text, encoding="utf-8")
    return out
