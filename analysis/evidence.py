"""
Tekmérion evidence layer.

Consumes already-processed records and produces structured, deterministic
metrics. This layer never re-classifies or re-extracts; it only aggregates.

Inclusion rule (documented and consistent across all metrics):
  Only records where `is_valid is True` AND `is_duplicate is False`.

UNKNOWN role families and seniorities are kept visible.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Optional, Sequence

from analysis.models import ProcessedJob, RoleFamily, Seniority


# ---------------------------------------------------------------------------
# Inclusion filter
# ---------------------------------------------------------------------------

def analysis_records(records: Sequence[ProcessedJob]) -> list[ProcessedJob]:
    """
    Return the subset of records that enter every evidence metric.

    Rule: valid and not marked as duplicate.
    """
    return [r for r in records if r.is_valid and not r.is_duplicate]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sorted_freq(counter: Counter, *, limit: Optional[int] = None) -> list[dict[str, Any]]:
    """
    Convert a Counter into a deterministic list of {item, count, proportion?}.
    Sort: count descending, then item ascending.
    """
    items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    if limit is not None:
        items = items[:limit]
    return [{"item": k, "count": v} for k, v in items]


def _with_proportion(freq_list: list[dict[str, Any]], total: int) -> list[dict[str, Any]]:
    if total <= 0:
        return [{**d, "proportion": 0.0} for d in freq_list]
    return [
        {**d, "proportion": round(d["count"] / total, 4)}
        for d in freq_list
    ]


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def skill_frequency(
    records: Sequence[ProcessedJob],
    *,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Global skill frequency over analysis records.

    Returns list of {"item": skill, "count": n, "proportion": p}
    sorted by count desc, then skill asc.
    Proportion is relative to the number of analysis records (not total skill mentions).
    """
    subset = analysis_records(records)
    n = len(subset)
    counter: Counter = Counter()
    for r in subset:
        # Use set to avoid counting the same skill twice inside one job
        counter.update(set(r.skills_extracted))
    freq = _sorted_freq(counter, limit=limit)
    return _with_proportion(freq, n)


def skills_by_role(
    records: Sequence[ProcessedJob],
    *,
    limit_per_role: Optional[int] = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Skill frequency broken down by role_family.

    Keys are role_family values (including "unknown").
    Each value is a sorted frequency list (count + proportion within that role).
    """
    subset = analysis_records(records)
    by_role: dict[str, list[ProcessedJob]] = defaultdict(list)
    for r in subset:
        by_role[r.role_family.value].append(r)

    result: dict[str, list[dict[str, Any]]] = {}
    # Deterministic key order
    for role in sorted(by_role.keys()):
        jobs = by_role[role]
        n = len(jobs)
        counter: Counter = Counter()
        for j in jobs:
            counter.update(set(j.skills_extracted))
        freq = _sorted_freq(counter, limit=limit_per_role)
        result[role] = _with_proportion(freq, n)
    return result


def skills_by_seniority(
    records: Sequence[ProcessedJob],
    *,
    limit_per_level: Optional[int] = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Skill frequency broken down by seniority (including "unknown").
    """
    subset = analysis_records(records)
    by_sen: dict[str, list[ProcessedJob]] = defaultdict(list)
    for r in subset:
        by_sen[r.seniority.value].append(r)

    result: dict[str, list[dict[str, Any]]] = {}
    for level in sorted(by_sen.keys()):
        jobs = by_sen[level]
        n = len(jobs)
        counter: Counter = Counter()
        for j in jobs:
            counter.update(set(j.skills_extracted))
        freq = _sorted_freq(counter, limit=limit_per_level)
        result[level] = _with_proportion(freq, n)
    return result


def role_distribution(records: Sequence[ProcessedJob]) -> list[dict[str, Any]]:
    """
    Count and proportion of analysis records per role_family.
    Includes UNKNOWN. Sorted by count desc, then name asc.
    """
    subset = analysis_records(records)
    n = len(subset)
    counter = Counter(r.role_family.value for r in subset)
    freq = _sorted_freq(counter)
    return _with_proportion(freq, n)


def seniority_distribution(records: Sequence[ProcessedJob]) -> list[dict[str, Any]]:
    """
    Count and proportion of analysis records per seniority.
    Includes UNKNOWN.
    """
    subset = analysis_records(records)
    n = len(subset)
    counter = Counter(r.seniority.value for r in subset)
    freq = _sorted_freq(counter)
    return _with_proportion(freq, n)


def skill_cooccurrence(
    records: Sequence[ProcessedJob],
    *,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Undirected skill co-occurrence pairs.

    A pair (a, b) with a < b alphabetically is counted once per job that contains both.
    Sorted by count desc, then pair name asc.
    Returns [{"skill_a": ..., "skill_b": ..., "count": n}, ...]
    """
    subset = analysis_records(records)
    pair_counter: Counter = Counter()

    for r in subset:
        skills = sorted(set(r.skills_extracted))  # unique + sorted
        for i in range(len(skills)):
            for j in range(i + 1, len(skills)):
                pair = (skills[i], skills[j])  # already ordered
                pair_counter[pair] += 1

    items = sorted(pair_counter.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
    if limit is not None:
        items = items[:limit]

    return [
        {"skill_a": a, "skill_b": b, "count": c}
        for (a, b), c in items
    ]


def compare_roles(
    records: Sequence[ProcessedJob],
    role_a: str,
    role_b: str,
    *,
    top_n: int = 10,
) -> dict[str, Any]:
    """
    Compare two role families.

    Returns structured evidence:
    - counts of jobs in each role
    - top skills in each
    - common skills (present in both)
    - exclusive to A / exclusive to B
    """
    subset = analysis_records(records)

    jobs_a = [r for r in subset if r.role_family.value == role_a]
    jobs_b = [r for r in subset if r.role_family.value == role_b]

    def _skill_set(jobs: list[ProcessedJob]) -> set[str]:
        s: set[str] = set()
        for j in jobs:
            s.update(j.skills_extracted)
        return s

    def _skill_counter(jobs: list[ProcessedJob]) -> Counter:
        c: Counter = Counter()
        for j in jobs:
            c.update(set(j.skills_extracted))
        return c

    set_a = _skill_set(jobs_a)
    set_b = _skill_set(jobs_b)
    counter_a = _skill_counter(jobs_a)
    counter_b = _skill_counter(jobs_b)

    common = sorted(set_a & set_b)
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)

    top_a = _sorted_freq(counter_a, limit=top_n)
    top_b = _sorted_freq(counter_b, limit=top_n)

    return {
        "role_a": role_a,
        "role_b": role_b,
        "count_a": len(jobs_a),
        "count_b": len(jobs_b),
        "top_skills_a": top_a,
        "top_skills_b": top_b,
        "common_skills": common,
        "only_in_a": only_a,
        "only_in_b": only_b,
    }


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

@dataclass
class EvidenceReport:
    """Full structured evidence package."""

    n_analysis_records: int
    skill_frequency: list[dict[str, Any]] = field(default_factory=list)
    skills_by_role: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    skills_by_seniority: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    role_distribution: list[dict[str, Any]] = field(default_factory=list)
    seniority_distribution: list[dict[str, Any]] = field(default_factory=list)
    skill_cooccurrence: list[dict[str, Any]] = field(default_factory=list)
    # Optional pairwise comparisons can be added by the caller

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_evidence(
    records: Sequence[ProcessedJob],
    *,
    skill_limit: Optional[int] = None,
    cooccurrence_limit: Optional[int] = 30,
) -> EvidenceReport:
    """
    Build a complete EvidenceReport from processed records.
    Pure function: does not mutate input.
    """
    subset = analysis_records(records)
    return EvidenceReport(
        n_analysis_records=len(subset),
        skill_frequency=skill_frequency(records, limit=skill_limit),
        skills_by_role=skills_by_role(records),
        skills_by_seniority=skills_by_seniority(records),
        role_distribution=role_distribution(records),
        seniority_distribution=seniority_distribution(records),
        skill_cooccurrence=skill_cooccurrence(records, limit=cooccurrence_limit),
    )
