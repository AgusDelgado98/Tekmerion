"""
Tekmérion ingestion layer (V0.4+).

Responsibilities:
- Load records from one or more sources via adapters
- Assign globally safe, source-namespaced internal IDs
- Normalize to the internal raw schema expected by analysis.pipeline
- Keep provenance (source, source_url, retrieved_at, source_record_id) explicit
- Use an explicit IngestionContext so timestamps are deterministic
- Surface incomplete / structurally invalid items without silent drops

Does NOT:
- Scrape the web
- Call external APIs (yet)
- Run classification or skill extraction (pipeline owns that)
- Mutate the synthetic sample
- Depend on datetime.now()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

from analysis.ingestion.base import IngestionContext, IngestionResult, SourceAdapter
from analysis.ingestion.local import LocalJsonSource
from analysis.ingestion.normalize import (
    normalize_to_internal,
    is_minimally_usable,
    reject_reason,
    build_internal_id,
    deterministic_fallback_external_id,
)
from analysis.ingestion.adzuna import (
    AdzunaSource,
    AdzunaClient,
    AdzunaCredentials,
    AdzunaConfigError,
    AdzunaAPIError,
    load_credentials_from_env,
    map_adzuna_job,
    map_adzuna_results,
    save_raw_snapshot,
    SOURCE_NAME as ADZUNA_SOURCE_NAME,
)
from analysis.ingestion.market import (
    MarketQuery,
    MarketBatchResult,
    DEFAULT_MARKET_QUERIES,
    run_market_batch,
    merge_by_identity,
    market_artifact_dict,
    save_market_artifact,
    save_batch_raw_snapshots,
)


__all__ = [
    "IngestionContext",
    "IngestionResult",
    "SourceAdapter",
    "LocalJsonSource",
    "normalize_to_internal",
    "build_internal_id",
    "deterministic_fallback_external_id",
    "ingest",
    "ingest_local_file",
    "DEFAULT_REAL_SAMPLE",
    "DEFAULT_SYNTHETIC_SAMPLE",
    # Adzuna
    "AdzunaSource",
    "AdzunaClient",
    "AdzunaCredentials",
    "AdzunaConfigError",
    "AdzunaAPIError",
    "load_credentials_from_env",
    "map_adzuna_job",
    "map_adzuna_results",
    "save_raw_snapshot",
        # Market batch
    "MarketQuery",
    "MarketBatchResult",
    "DEFAULT_MARKET_QUERIES",
    "run_market_batch",
    "merge_by_identity",
    "market_artifact_dict",
    "save_market_artifact",
    "save_batch_raw_snapshots",
]


_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYNTHETIC_SAMPLE = _ROOT / "data" / "raw" / "sample_jobs.json"
DEFAULT_REAL_SAMPLE = _ROOT / "data" / "raw" / "real" / "sample_real_jobs.json"


def ingest(
    adapters: Sequence[SourceAdapter],
    *,
    context: Optional[IngestionContext] = None,
    reject_unusable: bool = False,
) -> IngestionResult:
    """
    Load from multiple adapters, normalize each record, and return an
    IngestionResult ready for process_records.

    Parameters
    ----------
    adapters :
        Ordered sequence of SourceAdapter instances.
    context :
        Optional IngestionContext. When provided, its ``retrieved_at`` is
        applied to every record that does not already carry one. A single
        context per run keeps the whole batch deterministic.
    reject_unusable :
        If True, records that fail ``is_minimally_usable`` are moved to
        ``rejections`` and omitted from ``records``. If False (default),
        they are still normalized and included so the pipeline can mark
        them invalid explicitly.
    """
    all_records: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    source_names: list[str] = []
    total_loaded = 0

    for adapter in adapters:
        name = adapter.source_name()
        source_names.append(name)
        raw_list = adapter.load()
        total_loaded += len(raw_list)

        for idx, item in enumerate(raw_list):
            reason = reject_reason(item)
            if reason is not None:
                rejections.append(
                    {
                        "source": name,
                        "index": idx,
                        "reason": reason,
                        "raw_preview": str(item)[:120],
                    }
                )
                continue

            normalized = normalize_to_internal(
                item,
                default_source=name,
                context=context,
            )

            if reject_unusable and not is_minimally_usable(normalized):
                rejections.append(
                    {
                        "source": name,
                        "index": idx,
                        "reason": "not_minimally_usable",
                        "id": normalized.get("id"),
                    }
                )
                continue

            all_records.append(normalized)

    return IngestionResult(
        records=all_records,
        source_names=source_names,
        total_loaded=total_loaded,
        accepted_count=len(all_records),
        rejected_count=len(rejections),
        rejections=rejections,
        context_retrieved_at=context.retrieved_at if context else None,
    )


def ingest_local_file(
    path: Union[str, Path],
    *,
    source_name: Optional[str] = None,
    context: Optional[IngestionContext] = None,
    reject_unusable: bool = False,
) -> IngestionResult:
    """Convenience: ingest a single local JSON file."""
    adapter = LocalJsonSource(path, source_name=source_name)
    return ingest([adapter], context=context, reject_unusable=reject_unusable)
