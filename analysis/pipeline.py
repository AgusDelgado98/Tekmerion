"""
Tekmérion core pipeline.

Transforms raw job records into normalized, enriched, analyzable records.

Principles enforced:
- No mutation of input records
- Deterministic output for the same input
- Explicit validity and duplicate handling
- Separation of concerns (validation, classification, skills, dedup)
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Optional
from collections import Counter

from analysis.models import (
    ProcessedJob,
    PipelineResult,
    RoleFamily,
    Seniority,
)
from analysis.classifiers import classify_role_family, classify_seniority
from analysis.skills import extract_skills


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = ("id", "title", "company", "description")


def _validate_record(raw: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return (is_valid, list_of_errors)."""
    errors: list[str] = []

    if not isinstance(raw, dict):
        return False, ["record_is_not_a_dict"]

    for field in REQUIRED_FIELDS:
        value = raw.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"missing_or_empty_{field}")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_title(title: str) -> str:
    if not title:
        return ""
    # Simple deterministic normalization
    t = " ".join(title.strip().split())
    return t.title()  # Title Case for consistency


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Duplicate detection (content-based, deterministic)
# ---------------------------------------------------------------------------

def _content_fingerprint(
    title: str,
    company: str,
    description: str,
    source: str = "",
) -> str:
    """
    Create a stable fingerprint for near-duplicate detection.

    Uses source + normalized title + company + a hash of the description.

    Including ``source`` means:
    - Same content from the *same* source → treated as duplicate (expected).
    - Same content from *different* sources → independent records for now.
      Cross-source semantic deduplication is deliberate technical debt.
    """
    norm_title = _normalize_title(title).lower()
    norm_company = company.strip().lower()
    norm_source = (source or "").strip().lower()
    # Use first 300 chars of description to catch near-identical postings
    desc_part = (description or "")[:300].strip().lower()
    payload = f"{norm_source}|{norm_title}|{norm_company}|{desc_part}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Single record processing
# ---------------------------------------------------------------------------

def _process_single(
    raw: dict[str, Any],
    seen_fingerprints: dict[str, str],
) -> ProcessedJob:
    """
    Process one raw record into a ProcessedJob.
    Does not mutate `raw`.
    """
    is_valid, errors = _validate_record(raw)

    # Extract raw fields safely (never assume keys exist)
    job_id = _safe_str(raw.get("id"), default="unknown_id")
    title = _safe_str(raw.get("title"))
    company = _safe_str(raw.get("company"))
    location = _safe_str(raw.get("location"))
    description = _safe_str(raw.get("description"))
    source = _safe_str(raw.get("source"), default="unknown")
    source_url = _safe_str(raw.get("source_url")) or None
    retrieved_at = _safe_str(raw.get("retrieved_at")) or None
    source_record_id = _safe_str(raw.get("source_record_id")) or None

    normalized_title = _normalize_title(title)

    # Classification (even for invalid records we still attempt it for debugging)
    role_family = classify_role_family(title, description)
    seniority = classify_seniority(title, description)

    # Skills from title + description
    skills = extract_skills(f"{title} {description}")

    # Duplicate detection (source-aware: cross-source matches stay independent)
    fingerprint = _content_fingerprint(title, company, description, source=source)
    is_duplicate = False
    duplicate_of: Optional[str] = None

    if fingerprint in seen_fingerprints:
        is_duplicate = True
        duplicate_of = seen_fingerprints[fingerprint]
    else:
        # Only register non-empty valid-looking records as seen
        if is_valid and title:
            seen_fingerprints[fingerprint] = job_id

    return ProcessedJob(
        id=job_id,
        title=title,
        company=company,
        location=location,
        description=description,
        salary_min=_safe_int(raw.get("salary_min")),
        salary_max=_safe_int(raw.get("salary_max")),
        currency=_safe_str(raw.get("currency")) or None,
        posted_date=_safe_str(raw.get("posted_date")) or None,
        source=source,
        normalized_title=normalized_title,
        role_family=role_family,
        seniority=seniority,
        skills_extracted=tuple(skills),
        is_valid=is_valid,
        validation_errors=tuple(errors),
        is_duplicate=is_duplicate,
        duplicate_of=duplicate_of,
        source_url=source_url,
        retrieved_at=retrieved_at,
        source_record_id=source_record_id,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_records(records: list[dict[str, Any]]) -> PipelineResult:
    """
    Main entry point: process a list of raw records.

    Guarantees:
    - Input list is never mutated
    - Same input → same output (deterministic)
    - Every input record produces exactly one ProcessedJob
    """
    if not isinstance(records, list):
        raise TypeError("records must be a list of dicts")

    # Work on a shallow copy of the list to avoid any accidental mutation
    # of the caller's list object (we never touch the dicts either).
    input_records = list(records)

    seen_fingerprints: dict[str, str] = {}
    processed: list[ProcessedJob] = []

    for raw in input_records:
        # Defensive: if someone passes non-dict, still produce a record
        if not isinstance(raw, dict):
            processed.append(
                ProcessedJob(
                    id="invalid",
                    title="",
                    company="",
                    location="",
                    description="",
                    salary_min=None,
                    salary_max=None,
                    currency=None,
                    posted_date=None,
                    source="unknown",
                    normalized_title="",
                    role_family=RoleFamily.UNKNOWN,
                    seniority=Seniority.UNKNOWN,
                    skills_extracted=(),
                    is_valid=False,
                    validation_errors=("record_is_not_a_dict",),
                    is_duplicate=False,
                    duplicate_of=None,
                    source_url=None,
                    retrieved_at=None,
                    source_record_id=None,
                )
            )
            continue

        processed.append(_process_single(raw, seen_fingerprints))

    # Aggregate stats (only over valid non-duplicate for some counts? 
    # We count everything for transparency)
    valid_count = sum(1 for r in processed if r.is_valid)
    invalid_count = len(processed) - valid_count
    duplicate_count = sum(1 for r in processed if r.is_duplicate)

    role_counts = Counter(r.role_family.value for r in processed if r.is_valid)
    seniority_counts = Counter(r.seniority.value for r in processed if r.is_valid)

    return PipelineResult(
        records=processed,
        total_input=len(processed),
        valid_count=valid_count,
        invalid_count=invalid_count,
        duplicate_count=duplicate_count,
        role_family_counts=dict(role_counts),
        seniority_counts=dict(seniority_counts),
    )


def process_file(
    input_path: str | Path,
    output_path: Optional[str | Path] = None,
) -> PipelineResult:
    """
    Load JSON from input_path, run pipeline, optionally write results.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records")

    result = process_records(data)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        serializable = [r.to_dict() for r in result.records]
        with out.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    return result
