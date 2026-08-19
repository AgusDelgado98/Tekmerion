"""Explicit live Adzuna fetch into unlabeled snapshots (no role labels)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from analysis.ingestion.adzuna import (
    AdzunaAPIError,
    AdzunaClient,
    AdzunaConfigError,
    load_credentials_from_env,
    map_adzuna_results,
    save_raw_snapshot,
    SOURCE_NAME,
)
from analysis.ml.harvest import DEFAULT_ADZUNA_SNAPSHOTS, _from_ingested, dedupe_candidates

# Queries biased toward current gold gaps (BA / AI) plus the usual families.
# These are search strings, not labels.
DEFAULT_GOLD_FETCH_QUERIES: tuple[str, ...] = (
    "data analyst",
    "business intelligence analyst",
    "data scientist",
    "machine learning engineer",
    "data engineer",
    "business analyst",
    "analista funcional",
    "ai analyst",
    "ai engineer",
    "prompt engineer",
    "analista de datos",
)

DEFAULT_GOLD_FETCH_COUNTRIES: tuple[str, ...] = ("ar", "gb")

# Targeted follow-up search strings for the data_analyst gold gap. Not labels.
DATA_ANALYST_GAP_QUERIES: tuple[str, ...] = (
    "data analyst",
    "junior data analyst",
    "reporting analyst",
    "data reporting analyst",
    "analytics analyst",
    "insights analyst",
)


def adzuna_credentials_status() -> dict[str, Any]:
    try:
        load_credentials_from_env()
    except AdzunaConfigError as exc:
        return {"available": False, "reason": "missing_credentials", "message": str(exc)}
    return {"available": True, "reason": "env_set"}


def fetch_adzuna_snapshots(
    *,
    queries: Sequence[str] = DEFAULT_GOLD_FETCH_QUERIES,
    countries: Sequence[str] = DEFAULT_GOLD_FETCH_COUNTRIES,
    pages: int = 1,
    results_per_page: int = 50,
    client: AdzunaClient | None = None,
    snapshot_dir=DEFAULT_ADZUNA_SNAPSHOTS,
) -> dict[str, Any]:
    """
    Live search → raw snapshots on disk → unlabeled mapped records.

    Does not run the analysis pipeline and does not assign gold_role_family.
    """
    try:
        used_client = client or AdzunaClient(load_credentials_from_env())
    except AdzunaConfigError as exc:
        return {
            "fetched": False,
            "reason": "missing_credentials",
            "message": str(exc),
            "n_queries_ok": 0,
            "n_requests": 0,
            "n_raw_results": 0,
            "n_unique": 0,
            "n_dropped_duplicates": 0,
            "n_rate_limited": 0,
            "n_snapshots": 0,
            "by_query": [],
            "errors": [],
            "records": [],
        }

    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    collected: list[dict[str, Any]] = []
    snapshot_paths: list[str] = []
    by_query: list[dict[str, Any]] = []
    n_raw = 0
    n_ok = 0
    n_requests = 0
    n_rate_limited = 0
    errors: list[dict[str, Any]] = []

    for country in countries:
        for query in queries:
            for page in range(1, pages + 1):
                n_requests += 1
                try:
                    result = used_client.search(
                        what=query,
                        country=country,
                        page=page,
                        results_per_page=results_per_page,
                    )
                except AdzunaAPIError as exc:
                    status = exc.status_code
                    rate = status == 429
                    if rate:
                        n_rate_limited += 1
                    errors.append(
                        {
                            "country": country,
                            "query": query,
                            "page": page,
                            "error": "AdzunaAPIError",
                            "status_code": status,
                            "rate_limit": rate,
                        }
                    )
                    by_query.append(
                        {
                            "country": country,
                            "query": query,
                            "page": page,
                            "n_raw": 0,
                            "ok": False,
                            "status_code": status,
                        }
                    )
                    continue
                except OSError:
                    errors.append(
                        {
                            "country": country,
                            "query": query,
                            "page": page,
                            "error": "OSError",
                            "status_code": None,
                            "rate_limit": False,
                        }
                    )
                    by_query.append(
                        {
                            "country": country,
                            "query": query,
                            "page": page,
                            "n_raw": 0,
                            "ok": False,
                            "status_code": None,
                        }
                    )
                    continue
                n_ok += 1
                n_batch = len(result.records)
                n_raw += n_batch
                path = save_raw_snapshot(
                    result.raw_payload,
                    directory=snapshot_dir,
                    retrieved_at=retrieved_at,
                    country=country,
                    query=query,
                    page=page,
                )
                snapshot_paths.append(str(path).replace("\\", "/"))
                collected.extend(
                    _from_ingested(
                        map_adzuna_results(result.raw_payload),
                        source_kind="adzuna_snapshot",
                        source_ref=str(path).replace("\\", "/"),
                    )
                )
                by_query.append(
                    {
                        "country": country,
                        "query": query,
                        "page": page,
                        "n_raw": n_batch,
                        "ok": True,
                        "status_code": 200,
                    }
                )

    unique, dropped = dedupe_candidates(collected)
    return {
        "fetched": True,
        "reason": "ok" if n_ok else "all_queries_failed",
        "source": SOURCE_NAME,
        "retrieved_at": retrieved_at,
        "n_requests": n_requests,
        "n_queries_ok": n_ok,
        "n_raw_results": n_raw,
        "n_unique": len(unique),
        "n_dropped_duplicates": dropped,
        "n_rate_limited": n_rate_limited,
        "n_snapshots": len(snapshot_paths),
        "by_query": by_query,
        "errors": errors,
        "records": unique,
        "label_status": "unlabeled",
    }
