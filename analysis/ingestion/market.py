"""
Market batch: multi-query Adzuna run with deterministic identity merge (V0.4.4).

Flow
----
  queries → fetch/map each → normalize (shared IngestionContext)
         → merge by internal id → process_records once → evidence

Does not call Flask. Does not page beyond one page per query.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from analysis.evidence import EvidenceReport, build_evidence
from analysis.ingestion.adzuna import (
    SOURCE_NAME,
    AdzunaClient,
    AdzunaSource,
    map_adzuna_results,
    save_raw_snapshot,
    DEFAULT_COUNTRY,
    DEFAULT_RESULTS_PER_PAGE,
)
from analysis.ingestion.base import IngestionContext
from analysis.ingestion.normalize import normalize_to_internal
from analysis.models import PipelineResult, ProcessedJob
from analysis.pipeline import process_records


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

DEFAULT_MARKET_QUERIES: tuple[str, ...] = (
    "data analyst",
    "business intelligence analyst",
    "data scientist",
    "machine learning engineer",
    "data engineer",
    "business analyst",
)

MARKET_ARTIFACT_SCHEMA = "tekmerion.market_batch.v1"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketQuery:
    """One search within a market batch."""

    what: str
    where: Optional[str] = None
    results_per_page: Optional[int] = None  # None → batch default


@dataclass
class MarketQueryOutcome:
    """Result of executing one MarketQuery (live or offline)."""

    query: MarketQuery
    raw_payload: dict[str, Any]
    mapped_records: list[dict[str, Any]]
    received_count: int


@dataclass
class MarketBatchResult:
    """
    Full outcome of a market batch run.

    ``consolidated_records`` are pipeline-ready raw dicts (post-normalize,
    post-identity-merge). ``matched_queries_by_id`` maps internal id →
    sorted list of query strings that returned that vacancy.
    """

    country: str
    queries: list[str]
    retrieved_at: str
    query_outcomes: list[MarketQueryOutcome]
    consolidated_records: list[dict[str, Any]]
    matched_queries_by_id: dict[str, list[str]]
    total_received: int
    unique_count: int
    duplicates_removed: int
    pipeline_result: Optional[PipelineResult] = None
    evidence: Optional[EvidenceReport] = None

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "country": self.country,
            "queries": list(self.queries),
            "retrieved_at": self.retrieved_at,
            "received_per_query": {
                o.query.what: o.received_count for o in self.query_outcomes
            },
            "total_received": self.total_received,
            "unique_records": self.unique_count,
            "duplicates_removed": self.duplicates_removed,
        }
        if self.pipeline_result is not None:
            out["pipeline"] = self.pipeline_result.summary()
        if self.evidence is not None:
            out["evidence"] = {
                "n_analysis_records": self.evidence.n_analysis_records,
                "role_distribution": self.evidence.role_distribution,
                "seniority_distribution": self.evidence.seniority_distribution,
                "top_skills": self.evidence.skill_frequency[:10],
            }
        return out


# ---------------------------------------------------------------------------
# Completeness / conflict resolution
# ---------------------------------------------------------------------------

_COMPLETENESS_FIELDS = (
    "title",
    "company",
    "description",
    "location",
    "source_url",
    "salary_min",
    "salary_max",
)


def _completeness_score(record: dict[str, Any]) -> tuple[int, int]:
    """
    Deterministic richness score.

    Returns (non_empty_field_count, description_length) for tie-breaking.
    """
    score = 0
    for key in _COMPLETENESS_FIELDS:
        val = record.get(key)
        if val is None:
            continue
        if isinstance(val, str) and not val.strip():
            continue
        score += 1
    desc_len = len(str(record.get("description") or ""))
    return score, desc_len


def _pick_canonical(
    candidates: list[tuple[str, dict[str, Any]]],
) -> tuple[str, dict[str, Any]]:
    """
    Choose one (query, record) from candidates sharing the same internal id.

    Rule (documented, order-independent):
      1. Highest completeness score
      2. Longer description
      3. Lexicographically smallest query string
    """
    def sort_key(item: tuple[str, dict[str, Any]]) -> tuple:
        query, rec = item
        score, desc_len = _completeness_score(rec)
        # Negate score/desc so higher is better with ascending sort of query
        return (-score, -desc_len, query)

    return sorted(candidates, key=sort_key)[0]


def merge_by_identity(
    items: Sequence[tuple[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]], int]:
    """
    Merge normalized records by internal ``id``.

    Parameters
    ----------
    items :
        Sequence of (query_string, normalized_record).

    Returns
    -------
    consolidated, matched_queries_by_id, duplicates_removed
    """
    groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for query, rec in items:
        rid = str(rec.get("id") or "")
        groups.setdefault(rid, []).append((query, rec))

    consolidated: list[dict[str, Any]] = []
    matched: dict[str, list[str]] = {}
    duplicates_removed = 0

    # Stable iteration order by id for deterministic output list order
    for rid in sorted(groups.keys()):
        candidates = groups[rid]
        queries = sorted({q for q, _ in candidates})
        matched[rid] = queries
        if len(candidates) > 1:
            duplicates_removed += len(candidates) - 1
        _, winner = _pick_canonical(candidates)
        # Shallow copy so callers cannot mutate the stored group
        consolidated.append(dict(winner))

    return consolidated, matched, duplicates_removed


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def _normalize_query_list(
    queries: Optional[Sequence[str | MarketQuery]],
    *,
    limit_per_query: int,
) -> list[MarketQuery]:
    if not queries:
        queries = list(DEFAULT_MARKET_QUERIES)
    out: list[MarketQuery] = []
    for q in queries:
        if isinstance(q, MarketQuery):
            out.append(
                MarketQuery(
                    what=q.what,
                    where=q.where,
                    results_per_page=q.results_per_page or limit_per_query,
                )
            )
        else:
            out.append(MarketQuery(what=str(q), results_per_page=limit_per_query))
    return out


def run_market_batch(
    *,
    country: str = DEFAULT_COUNTRY,
    queries: Optional[Sequence[str | MarketQuery]] = None,
    limit_per_query: int = 5,
    retrieved_at: str,
    client: Optional[AdzunaClient] = None,
    payload_by_query: Optional[dict[str, dict[str, Any]]] = None,
    run_pipeline: bool = True,
    fail_fast: bool = True,
) -> MarketBatchResult:
    """
    Execute a multi-query market batch.

    Provide either ``client`` (live) or ``payload_by_query`` (offline fixtures).
    All records share one ``IngestionContext(retrieved_at=...)``.

    Fail-fast (default): any query error aborts the whole batch.
    """
    if client is None and payload_by_query is None:
        raise ValueError("Provide either client (live) or payload_by_query (offline)")

    mq_list = _normalize_query_list(queries, limit_per_query=limit_per_query)
    context = IngestionContext(retrieved_at=retrieved_at)

    outcomes: list[MarketQueryOutcome] = []
    tagged: list[tuple[str, dict[str, Any]]] = []

    for mq in mq_list:
        try:
            if payload_by_query is not None:
                if mq.what not in payload_by_query:
                    raise KeyError(
                        f"No fixture payload for query {mq.what!r}. "
                        f"Available: {sorted(payload_by_query)}"
                    )
                payload = payload_by_query[mq.what]
                mapped = map_adzuna_results(payload)
            else:
                assert client is not None
                search = client.search(
                    what=mq.what,
                    country=country,
                    page=1,
                    results_per_page=mq.results_per_page or limit_per_query,
                    where=mq.where,
                )
                payload = search.raw_payload
                mapped = search.records
        except Exception:
            if fail_fast:
                raise
            # Partial mode: skip this query
            outcomes.append(
                MarketQueryOutcome(
                    query=mq,
                    raw_payload={},
                    mapped_records=[],
                    received_count=0,
                )
            )
            continue

        outcomes.append(
            MarketQueryOutcome(
                query=mq,
                raw_payload=payload,
                mapped_records=mapped,
                received_count=len(mapped),
            )
        )

        for rec in mapped:
            normalized = normalize_to_internal(
                rec,
                default_source=SOURCE_NAME,
                context=context,
            )
            tagged.append((mq.what, normalized))

    consolidated, matched, dups_removed = merge_by_identity(tagged)
    total_received = sum(o.received_count for o in outcomes)

    pipeline_result: Optional[PipelineResult] = None
    evidence: Optional[EvidenceReport] = None
    if run_pipeline:
        pipeline_result = process_records(consolidated)
        evidence = build_evidence(pipeline_result.records)

    return MarketBatchResult(
        country=country,
        queries=[mq.what for mq in mq_list],
        retrieved_at=retrieved_at,
        query_outcomes=outcomes,
        consolidated_records=consolidated,
        matched_queries_by_id=matched,
        total_received=total_received,
        unique_count=len(consolidated),
        duplicates_removed=dups_removed,
        pipeline_result=pipeline_result,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def market_artifact_dict(result: MarketBatchResult) -> dict[str, Any]:
    """Serialize a MarketBatchResult to a JSON-friendly audit artifact."""
    processed: list[dict[str, Any]] = []
    if result.pipeline_result is not None:
        processed = [r.to_dict() for r in result.pipeline_result.records]

    return {
        "schema": MARKET_ARTIFACT_SCHEMA,
        "source": SOURCE_NAME,
        "country": result.country,
        "retrieved_at": result.retrieved_at,
        "queries": list(result.queries),
        "received_per_query": {
            o.query.what: o.received_count for o in result.query_outcomes
        },
        "total_received": result.total_received,
        "unique_records": result.unique_count,
        "duplicates_removed": result.duplicates_removed,
        "matched_queries_by_id": {
            k: list(v) for k, v in sorted(result.matched_queries_by_id.items())
        },
        "pipeline_summary": (
            result.pipeline_result.summary() if result.pipeline_result else None
        ),
        "evidence_summary": (
            {
                "n_analysis_records": result.evidence.n_analysis_records,
                "role_distribution": result.evidence.role_distribution,
                "seniority_distribution": result.evidence.seniority_distribution,
                "top_skills": result.evidence.skill_frequency[:15],
            }
            if result.evidence
            else None
        ),
        "records": processed,
    }


def save_market_artifact(
    result: MarketBatchResult,
    *,
    directory: str | Path,
) -> Path:
    """
    Write a consolidated market artifact. Never overwrites an existing file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    safe_ts = result.retrieved_at.replace(":", "").replace("-", "")[:15]
    filename = f"market_{result.country}_{safe_ts}.json"
    path = directory / filename
    n = 1
    while path.exists():
        path = directory / f"market_{result.country}_{safe_ts}_{n}.json"
        n += 1

    payload = market_artifact_dict(result)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def save_batch_raw_snapshots(
    result: MarketBatchResult,
    *,
    directory: str | Path,
) -> list[Path]:
    """Persist one raw snapshot per query that returned a payload."""
    paths: list[Path] = []
    for outcome in result.query_outcomes:
        if not outcome.raw_payload:
            continue
        paths.append(
            save_raw_snapshot(
                outcome.raw_payload,
                directory=directory,
                retrieved_at=result.retrieved_at,
                country=result.country,
                query=outcome.query.what,
                page=1,
            )
        )
    return paths
