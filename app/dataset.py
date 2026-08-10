"""
Dataset loading for the Flask layer (V0.4.5).

Flask never calls Adzuna. It only loads:
  - synthetic: run pipeline once on the sample (unchanged behaviour)
  - market: hydrate ProcessedJob objects from a market artifact (no re-pipeline)

Config (environment / Flask config):
  TEKMERION_DATA_MODE   = synthetic | market   (default: synthetic)
  TEKMERION_MARKET_FILE = optional path to a market_*.json artifact
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from analysis.evidence import EvidenceReport, build_evidence
from analysis.models import PipelineResult, ProcessedJob, RoleFamily, Seniority
from analysis.pipeline import process_file


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "raw" / "sample_jobs.json"
MARKET_DIR = ROOT / "data" / "processed" / "market"

MODE_SYNTHETIC = "synthetic"
MODE_MARKET = "market"
MARKET_SCHEMA = "tekmerion.market_batch.v1"

ENV_DATA_MODE = "TEKMERION_DATA_MODE"
ENV_MARKET_FILE = "TEKMERION_MARKET_FILE"


class DatasetError(RuntimeError):
    """Invalid configuration or unreadable market artifact."""


@dataclass(frozen=True)
class DatasetMeta:
    """Runtime metadata about the active dataset (safe for templates)."""

    mode: str
    source: str
    label: str
    retrieved_at: Optional[str] = None
    country: Optional[str] = None
    total_records: int = 0
    query_count: int = 0
    artifact_name: Optional[str] = None  # basename only — never absolute paths in UI

    def display_line(self) -> str:
        if self.mode == MODE_SYNTHETIC:
            return f"Dataset: Synthetic sample · {self.total_records} records"
        if self.label and "Showroom" in self.label:
            parts = ["Showroom · Market demo"]
            if self.total_records:
                parts.append(f"{self.total_records} vacantes")
            return " · ".join(parts)
        parts = ["Market snapshot"]
        if self.country:
            parts.append(self.country.upper())
        if self.retrieved_at:
            # date portion if ISO
            parts.append(self.retrieved_at[:10])
        parts.append(f"{self.total_records} vacantes")
        if self.query_count:
            parts.append(f"{self.query_count} queries")
        return " · ".join(parts)


@dataclass
class AppDataset:
    pipeline_result: PipelineResult
    evidence: EvidenceReport
    meta: DatasetMeta


# ---------------------------------------------------------------------------
# Hydration (no pipeline)
# ---------------------------------------------------------------------------

def processed_job_from_dict(d: dict[str, Any]) -> ProcessedJob:
    """Rebuild a ProcessedJob from ProcessedJob.to_dict() output. No re-classification."""
    if not isinstance(d, dict):
        raise DatasetError("Market record is not an object")

    def _enum(cls, value, default):
        if value is None:
            return default
        try:
            return cls(value)
        except ValueError:
            return default

    skills = d.get("skills_extracted") or ()
    if isinstance(skills, list):
        skills = tuple(skills)
    errors = d.get("validation_errors") or ()
    if isinstance(errors, list):
        errors = tuple(errors)

    return ProcessedJob(
        id=str(d.get("id") or ""),
        title=str(d.get("title") or ""),
        company=str(d.get("company") or ""),
        location=str(d.get("location") or ""),
        description=str(d.get("description") or ""),
        salary_min=d.get("salary_min"),
        salary_max=d.get("salary_max"),
        currency=d.get("currency"),
        posted_date=d.get("posted_date"),
        source=str(d.get("source") or "unknown"),
        normalized_title=str(d.get("normalized_title") or ""),
        role_family=_enum(RoleFamily, d.get("role_family"), RoleFamily.UNKNOWN),
        seniority=_enum(Seniority, d.get("seniority"), Seniority.UNKNOWN),
        skills_extracted=tuple(skills),
        is_valid=bool(d.get("is_valid", False)),
        validation_errors=tuple(errors),
        is_duplicate=bool(d.get("is_duplicate", False)),
        duplicate_of=d.get("duplicate_of"),
        source_url=d.get("source_url"),
        retrieved_at=d.get("retrieved_at"),
        source_record_id=d.get("source_record_id"),
    )


def pipeline_result_from_records(records: list[ProcessedJob]) -> PipelineResult:
    """Aggregate PipelineResult stats from already-processed jobs (no re-pipeline)."""
    valid = [r for r in records if r.is_valid]
    return PipelineResult(
        records=records,
        total_input=len(records),
        valid_count=len(valid),
        invalid_count=len(records) - len(valid),
        duplicate_count=sum(1 for r in records if r.is_duplicate),
        role_family_counts=dict(Counter(r.role_family.value for r in valid)),
        seniority_counts=dict(Counter(r.seniority.value for r in valid)),
    )


# ---------------------------------------------------------------------------
# Market artifact validation & discovery
# ---------------------------------------------------------------------------

def validate_market_artifact(data: Any, *, path_hint: str = "") -> dict[str, Any]:
    """Raise DatasetError if payload is not a usable market artifact."""
    where = f" ({path_hint})" if path_hint else ""
    if not isinstance(data, dict):
        raise DatasetError(f"Market artifact is not a JSON object{where}")
    schema = data.get("schema")
    if schema != MARKET_SCHEMA:
        raise DatasetError(
            f"Unsupported market artifact schema {schema!r}{where}; "
            f"expected {MARKET_SCHEMA!r}"
        )
    records = data.get("records")
    if records is None:
        raise DatasetError(f"Market artifact missing 'records'{where}")
    if not isinstance(records, list):
        raise DatasetError(f"Market artifact 'records' must be a list{where}")
    # Empty list is allowed but unusual — not an error
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise DatasetError(f"Market artifact records[{i}] is not an object{where}")
        if "id" not in rec:
            raise DatasetError(f"Market artifact records[{i}] missing 'id'{where}")
    return data


def load_market_artifact_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DatasetError(f"Market artifact not found: {path.name}")
    if not path.is_file():
        raise DatasetError(f"Market path is not a file: {path.name}")
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Market artifact is not valid JSON ({path.name}): {exc}") from None
    except OSError as exc:
        raise DatasetError(f"Cannot read market artifact ({path.name}): {exc}") from None
    return validate_market_artifact(data, path_hint=path.name)


def discover_latest_market_artifact(directory: Path = MARKET_DIR) -> Path:
    """
    Pick the newest *valid* artifact by internal retrieved_at (ISO string).
    Falls back to lexicographic basename if timestamps tie or are missing.
    Ignores invalid JSON / wrong schema during discovery.
    """
    if not directory.exists() or not directory.is_dir():
        raise DatasetError(
            f"No market artifacts available under data/processed/market/ "
            f"(mode=market requires a valid artifact or TEKMERION_MARKET_FILE)"
        )

    candidates: list[tuple[str, str, Path]] = []
    for path in directory.glob("*.json"):
        try:
            data = load_market_artifact_file(path)
        except DatasetError:
            continue
        ts = str(data.get("retrieved_at") or "")
        candidates.append((ts, path.name, path))

    if not candidates:
        raise DatasetError(
            "No valid market artifacts found under data/processed/market/ "
            f"(expected schema {MARKET_SCHEMA})"
        )

    # retrieved_at desc, then name desc — deterministic
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_synthetic_dataset() -> AppDataset:
    pipeline_result = process_file(SAMPLE_PATH)
    evidence = build_evidence(pipeline_result.records)
    meta = DatasetMeta(
        mode=MODE_SYNTHETIC,
        source="synthetic",
        label="Synthetic sample",
        retrieved_at=None,
        country=None,
        total_records=pipeline_result.total_input,
        query_count=0,
        artifact_name=None,
    )
    return AppDataset(pipeline_result=pipeline_result, evidence=evidence, meta=meta)


def load_market_dataset(path: Path) -> AppDataset:
    data = load_market_artifact_file(path)
    jobs = [processed_job_from_dict(r) for r in data["records"]]
    # Reconstruct aggregates from hydrated records — do NOT call process_records
    pipeline_result = pipeline_result_from_records(jobs)
    # Evidence from the same records (deterministic; matches a fresh build_evidence
    # on those ProcessedJobs). Artifact evidence_summary is a subset snapshot only.
    evidence = build_evidence(jobs)

    queries = data.get("queries") or []
    meta = DatasetMeta(
        mode=MODE_MARKET,
        source=str(data.get("source") or "adzuna"),
        label="Market snapshot",
        retrieved_at=data.get("retrieved_at"),
        country=data.get("country"),
        total_records=len(jobs),
        query_count=len(queries) if isinstance(queries, list) else 0,
        artifact_name=path.name,
    )
    return AppDataset(pipeline_result=pipeline_result, evidence=evidence, meta=meta)


def resolve_data_mode(
    *,
    mode: Optional[str] = None,
    market_file: Optional[str] = None,
) -> tuple[str, Optional[Path]]:
    """
    Resolve mode + optional explicit market path from args or environment.

    Returns (mode, path_or_none).
    """
    raw_mode = (mode if mode is not None else os.environ.get(ENV_DATA_MODE, MODE_SYNTHETIC))
    raw_mode = str(raw_mode or MODE_SYNTHETIC).strip().lower()
    if raw_mode not in (MODE_SYNTHETIC, MODE_MARKET):
        raise DatasetError(
            f"Invalid {ENV_DATA_MODE}={raw_mode!r}; expected 'synthetic' or 'market'"
        )

    explicit = market_file if market_file is not None else os.environ.get(ENV_MARKET_FILE)
    path: Optional[Path] = None
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        else:
            path = path.resolve()

    return raw_mode, path


def load_app_dataset(
    *,
    mode: Optional[str] = None,
    market_file: Optional[str] = None,
) -> AppDataset:
    """
    Main entry used by create_app.

    synthetic → sample pipeline (always works)
    market + explicit file → that file (fail if bad)
    market without file → discover latest valid under data/processed/market/
    """
    resolved_mode, path = resolve_data_mode(mode=mode, market_file=market_file)

    if resolved_mode == MODE_SYNTHETIC:
        return load_synthetic_dataset()

    # market mode — no silent fallback to synthetic
    if path is not None:
        return load_market_dataset(path)

    discovered = discover_latest_market_artifact(MARKET_DIR)
    return load_market_dataset(discovered)
