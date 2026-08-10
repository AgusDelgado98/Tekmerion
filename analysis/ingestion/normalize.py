"""
Normalization from source-oriented records to the internal raw schema
expected by analysis.pipeline.

This layer is intentionally thin:
- builds globally safe internal IDs
- ensures required keys exist (even if empty)
- attaches / preserves provenance fields
- does not classify roles or extract skills (that is the pipeline's job)
- never depends on the wall clock
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from analysis.ingestion.base import IngestionContext


# Fields the pipeline treats as required for validity
_PIPELINE_REQUIRED = ("id", "title", "company", "description")

# Characters that are awkward inside an internal id segment
_ID_UNSAFE = re.compile(r"[^\w.\-]+", re.UNICODE)


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _sanitize_id_segment(value: str) -> str:
    """Make a string safe to embed in an internal id (no colons, no spaces)."""
    cleaned = _ID_UNSAFE.sub("_", value.strip())
    return cleaned.strip("_") or "empty"


def deterministic_fallback_external_id(
    *,
    source: str,
    company: str,
    title: str,
    location: str,
    source_url: str,
) -> str:
    """
    Build a deterministic external-id substitute when the source provides none.

    Uses a stable hash of source + company + title + location + source_url.
    Prefix ``auto:`` marks it as generated, not supplier-provided.
    """
    payload = "|".join(
        [
            source.strip().lower(),
            company.strip().lower(),
            title.strip().lower(),
            location.strip().lower(),
            (source_url or "").strip().lower(),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"auto:{digest}"


def build_internal_id(
    source: str,
    external_id: Optional[str],
    *,
    company: str = "",
    title: str = "",
    location: str = "",
    source_url: str = "",
) -> tuple[str, Optional[str]]:
    """
    Produce a globally safe internal id and the preserved external id.

    Strategy
    --------
    internal_id  = ``{source}:{external_id}``
    source_record_id = original external id (or None if we had to invent one)

    When ``external_id`` is missing/empty, a deterministic fallback is
    generated from stable record fields (never a random UUID).

    Returns
    -------
    (internal_id, source_record_id)
    """
    src = _sanitize_id_segment(source) if source else "unknown"
    ext = _safe_str(external_id)

    if ext:
        # Keep external readable; only sanitize characters that break the scheme
        safe_ext = ext.replace(":", "_")
        return f"{src}:{safe_ext}", ext

    fallback = deterministic_fallback_external_id(
        source=source or "unknown",
        company=company,
        title=title,
        location=location,
        source_url=source_url or "",
    )
    # fallback already starts with "auto:"
    return f"{src}:{fallback}", None


def normalize_to_internal(
    record: dict[str, Any],
    *,
    default_source: str = "unknown",
    default_retrieved_at: Optional[str] = None,
    context: Optional["IngestionContext"] = None,
) -> dict[str, Any]:
    """
    Map a source record to the internal raw shape used by process_records.

    Guarantees
    ----------
    - Returns a new dict (never mutates input)
    - Always includes keys expected by the pipeline
    - Internal ``id`` is source-namespaced (``source:external_id``)
    - Original external id preserved as ``source_record_id`` when present
    - ``retrieved_at`` comes from the record, else from context /
      default_retrieved_at; never from ``datetime.now()``
    - Incomplete records are still returned so the pipeline can mark them invalid
    """
    # Resolve retrieved_at default once (context wins over explicit default)
    ctx_ts: Optional[str] = None
    if context is not None:
        ctx_ts = context.retrieved_at
    elif default_retrieved_at:
        ctx_ts = default_retrieved_at

    if not isinstance(record, dict):
        internal_id, _ = build_internal_id(default_source, "invalid_non_dict")
        return {
            "id": internal_id,
            "source_record_id": None,
            "title": "",
            "company": "",
            "location": "",
            "description": "",
            "source": default_source,
            "source_url": None,
            "retrieved_at": ctx_ts,
        }

    out: dict[str, Any] = {}

    # Content fields first (needed for fallback id)
    title = _safe_str(record.get("title"))
    company = _safe_str(record.get("company"))
    location = _safe_str(record.get("location"))
    description = _safe_str(record.get("description"))
    source = _safe_str(record.get("source"), default=default_source) or default_source
    source_url = _safe_str(record.get("source_url")) or None

    # Identity
    external_raw = record.get("id")
    # Also accept an explicit source_record_id / external_id from adapters
    if external_raw is None or (isinstance(external_raw, str) and not external_raw.strip()):
        external_raw = record.get("source_record_id") or record.get("external_id")

    internal_id, source_record_id = build_internal_id(
        source,
        _safe_str(external_raw) or None,
        company=company,
        title=title,
        location=location,
        source_url=source_url or "",
    )
    out["id"] = internal_id
    out["source_record_id"] = source_record_id

    out["title"] = title
    out["company"] = company
    out["location"] = location
    out["description"] = description

    # Optional salary / date fields
    for key in ("salary_min", "salary_max"):
        out[key] = record.get(key)

    out["currency"] = _safe_str(record.get("currency")) or None
    out["posted_date"] = _safe_str(record.get("posted_date")) or None

    # Provenance
    out["source"] = source
    out["source_url"] = source_url
    record_ts = _safe_str(record.get("retrieved_at")) or None
    out["retrieved_at"] = record_ts or ctx_ts  # may still be None — explicit, not invented

    return out


def is_minimally_usable(record: dict[str, Any]) -> bool:
    """
    Cheap pre-check before sending to the pipeline.

    A record is minimally usable if it is a dict and has non-empty
    title, company and description. ``id`` alone is not enough.
    Full validity is still decided by the pipeline (REQUIRED_FIELDS).
    """
    if not isinstance(record, dict):
        return False
    for key in ("title", "company", "description"):
        val = record.get(key)
        if not (isinstance(val, str) and val.strip()):
            return False
    return True


def reject_reason(record: Any) -> Optional[str]:
    """Return a short reason if the record should be rejected at ingestion, else None."""
    if not isinstance(record, dict):
        return "not_a_dict"
    return None
