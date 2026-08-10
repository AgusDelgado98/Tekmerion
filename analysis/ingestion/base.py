"""
Ingestion interfaces and common types for Tekmérion.

Adapters implement a minimal contract so new sources can be added
without changing the core pipeline. Normalization to the internal
raw schema happens after load, before process_records.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class IngestionContext:
    """
    Execution context for one ingestion run.

    Provides a stable retrieved_at (and optional metadata) so that
    normalization never depends on the wall clock.

    Typical use for a future API adapter:
        ctx = IngestionContext(retrieved_at="2026-08-10T15:00:00Z")
        raw = adapter.fetch()
        result = ingest([adapter], context=ctx)
    """

    retrieved_at: str
    """ISO-8601 UTC timestamp applied to records that lack their own."""

    def __post_init__(self) -> None:
        if not self.retrieved_at or not str(self.retrieved_at).strip():
            raise ValueError("IngestionContext.retrieved_at must be a non-empty string")


@dataclass
class IngestionResult:
    """
    Outcome of loading + normalizing one or more sources.

    Records in `records` are already shaped for analysis.pipeline.process_records.
    Incomplete / rejected items are kept explicit for auditability.
    """

    records: list[dict[str, Any]]
    source_names: list[str]
    total_loaded: int
    accepted_count: int
    rejected_count: int
    rejections: list[dict[str, Any]] = field(default_factory=list)
    context_retrieved_at: Optional[str] = None

    def summary(self) -> dict[str, Any]:
        return {
            "source_names": self.source_names,
            "total_loaded": self.total_loaded,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "rejection_reasons": [r.get("reason") for r in self.rejections],
            "context_retrieved_at": self.context_retrieved_at,
        }


class SourceAdapter(ABC):
    """
    Base contract for a data source.

    Implementations load source-native or semi-structured records.
    They do NOT run the full analysis pipeline; they only produce
    dicts that normalize_to_internal can turn into pipeline-ready rows.
    """

    @abstractmethod
    def source_name(self) -> str:
        """Stable identifier for this source (used in provenance and IDs)."""
        ...

    @abstractmethod
    def load(self) -> list[dict[str, Any]]:
        """
        Return raw records from the source.

        Each dict should preferably already contain the fields that
        the internal schema expects, but adapters may return partial
        or source-specific shapes; normalization will handle mapping.
        """
        ...

    def describe(self) -> str:
        """Human-readable description for docs / logging."""
        return self.source_name()
