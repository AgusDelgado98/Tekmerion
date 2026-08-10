"""
Data models for Tekmérion pipeline.

Design principles:
- Immutable processed records (frozen dataclasses where sensible)
- Clear separation between raw input and enriched output
- Explicit validity and duplicate flags
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from enum import Enum


class RoleFamily(str, Enum):
    DATA_ANALYST = "data_analyst"
    BI_ANALYST = "bi_analyst"
    DATA_SCIENTIST = "data_scientist"
    ML_ENGINEER = "ml_engineer"
    AI_ANALYST = "ai_analyst"
    DATA_ENGINEER = "data_engineer"
    BUSINESS_ANALYST = "business_analyst"
    UNKNOWN = "unknown"


class Seniority(str, Enum):
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProcessedJob:
    """Normalized and enriched job record. Immutable by design."""

    id: str
    title: str
    company: str
    location: str
    description: str
    salary_min: Optional[int]
    salary_max: Optional[int]
    currency: Optional[str]
    posted_date: Optional[str]
    source: str

    # Enrichment
    normalized_title: str
    role_family: RoleFamily
    seniority: Seniority
    skills_extracted: tuple[str, ...]  # tuple for hashability / immutability

    # Quality flags
    is_valid: bool
    validation_errors: tuple[str, ...]
    is_duplicate: bool
    duplicate_of: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["role_family"] = self.role_family.value
        d["seniority"] = self.seniority.value
        d["skills_extracted"] = list(self.skills_extracted)
        d["validation_errors"] = list(self.validation_errors)
        return d


@dataclass
class PipelineResult:
    """Result of running the full pipeline."""

    records: list[ProcessedJob]
    total_input: int
    valid_count: int
    invalid_count: int
    duplicate_count: int
    role_family_counts: dict[str, int] = field(default_factory=dict)
    seniority_counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "total_input": self.total_input,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "duplicate_count": self.duplicate_count,
            "role_family_counts": self.role_family_counts,
            "seniority_counts": self.seniority_counts,
        }
