"""
Tests for the Tekmérion evidence layer.
"""

from __future__ import annotations

import copy
from collections import Counter

import pytest

from analysis.models import ProcessedJob, RoleFamily, Seniority
from analysis.evidence import (
    analysis_records,
    skill_frequency,
    skills_by_role,
    skills_by_seniority,
    role_distribution,
    seniority_distribution,
    skill_cooccurrence,
    compare_roles,
    build_evidence,
)


# ---------------------------------------------------------------------------
# Helpers to build minimal ProcessedJob instances
# ---------------------------------------------------------------------------

def _job(
    id: str = "x",
    *,
    role: RoleFamily = RoleFamily.DATA_ANALYST,
    seniority: Seniority = Seniority.UNKNOWN,
    skills: tuple[str, ...] = (),
    is_valid: bool = True,
    is_duplicate: bool = False,
) -> ProcessedJob:
    return ProcessedJob(
        id=id,
        title="T",
        company="C",
        location="",
        description="",
        salary_min=None,
        salary_max=None,
        currency=None,
        posted_date=None,
        source="test",
        normalized_title="T",
        role_family=role,
        seniority=seniority,
        skills_extracted=skills,
        is_valid=is_valid,
        validation_errors=() if is_valid else ("err",),
        is_duplicate=is_duplicate,
        duplicate_of=None,
    )


# ---------------------------------------------------------------------------
# Inclusion filter
# ---------------------------------------------------------------------------

def test_analysis_records_excludes_invalid_and_duplicates():
    records = [
        _job("1", is_valid=True, is_duplicate=False),
        _job("2", is_valid=False, is_duplicate=False),
        _job("3", is_valid=True, is_duplicate=True),
        _job("4", is_valid=True, is_duplicate=False),
    ]
    subset = analysis_records(records)
    assert len(subset) == 2
    assert {r.id for r in subset} == {"1", "4"}


def test_analysis_records_empty():
    assert analysis_records([]) == []


# ---------------------------------------------------------------------------
# Skill frequency
# ---------------------------------------------------------------------------

def test_skill_frequency_basic():
    records = [
        _job("1", skills=("python", "sql")),
        _job("2", skills=("python", "excel")),
        _job("3", skills=("sql",)),
    ]
    freq = skill_frequency(records)
    # python:2, sql:2, excel:1
    assert freq[0]["item"] in ("python", "sql")
    assert freq[0]["count"] == 2
    assert freq[0]["proportion"] == pytest.approx(2 / 3, abs=0.001)
    items = [d["item"] for d in freq]
    assert items == sorted(items, key=lambda x: (-next(d["count"] for d in freq if d["item"] == x), x))


def test_skill_frequency_does_not_count_same_skill_twice_in_one_job():
    records = [_job("1", skills=("python", "python", "sql"))]  # tuple can have dups theoretically
    # Our pipeline produces unique, but evidence should still be robust
    freq = skill_frequency(records)
    assert any(d["item"] == "python" and d["count"] == 1 for d in freq)


def test_skill_frequency_ignores_duplicates_and_invalid():
    records = [
        _job("1", skills=("python",), is_valid=True, is_duplicate=False),
        _job("2", skills=("python",), is_valid=True, is_duplicate=True),
        _job("3", skills=("python",), is_valid=False, is_duplicate=False),
    ]
    freq = skill_frequency(records)
    assert len(freq) == 1
    assert freq[0]["count"] == 1


def test_skill_frequency_empty():
    assert skill_frequency([]) == []


def test_skill_frequency_determinism():
    records = [
        _job("1", skills=("b", "a")),
        _job("2", skills=("a", "c")),
        _job("3", skills=("b",)),
    ]
    f1 = skill_frequency(records)
    f2 = skill_frequency(records)
    assert f1 == f2


# ---------------------------------------------------------------------------
# Skills by role / seniority
# ---------------------------------------------------------------------------

def test_skills_by_role():
    records = [
        _job("1", role=RoleFamily.DATA_ANALYST, skills=("sql", "excel")),
        _job("2", role=RoleFamily.DATA_ANALYST, skills=("sql", "python")),
        _job("3", role=RoleFamily.BI_ANALYST, skills=("tableau",)),
        _job("4", role=RoleFamily.UNKNOWN, skills=("git",)),
    ]
    by_role = skills_by_role(records)
    assert set(by_role.keys()) == {"data_analyst", "bi_analyst", "unknown"}
    da_skills = {d["item"]: d["count"] for d in by_role["data_analyst"]}
    assert da_skills["sql"] == 2
    assert da_skills["excel"] == 1
    assert da_skills["python"] == 1


def test_skills_by_seniority():
    records = [
        _job("1", seniority=Seniority.JUNIOR, skills=("python",)),
        _job("2", seniority=Seniority.SENIOR, skills=("python", "aws")),
        _job("3", seniority=Seniority.UNKNOWN, skills=("sql",)),
    ]
    by_sen = skills_by_seniority(records)
    assert "junior" in by_sen
    assert "senior" in by_sen
    assert "unknown" in by_sen
    assert by_sen["senior"][0]["item"] in ("python", "aws")


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def test_role_distribution():
    records = [
        _job("1", role=RoleFamily.DATA_ANALYST),
        _job("2", role=RoleFamily.DATA_ANALYST),
        _job("3", role=RoleFamily.ML_ENGINEER),
        _job("4", role=RoleFamily.UNKNOWN),
    ]
    dist = role_distribution(records)
    counts = {d["item"]: d["count"] for d in dist}
    assert counts["data_analyst"] == 2
    assert counts["ml_engineer"] == 1
    assert counts["unknown"] == 1
    # proportions sum ~ 1
    assert abs(sum(d["proportion"] for d in dist) - 1.0) < 0.01


def test_seniority_distribution():
    records = [
        _job("1", seniority=Seniority.JUNIOR),
        _job("2", seniority=Seniority.SENIOR),
        _job("3", seniority=Seniority.SENIOR),
    ]
    dist = seniority_distribution(records)
    counts = {d["item"]: d["count"] for d in dist}
    assert counts["senior"] == 2
    assert counts["junior"] == 1


# ---------------------------------------------------------------------------
# Co-occurrence
# ---------------------------------------------------------------------------

def test_skill_cooccurrence_basic():
    records = [
        _job("1", skills=("python", "sql", "excel")),
        _job("2", skills=("python", "sql")),
        _job("3", skills=("tableau",)),
    ]
    pairs = skill_cooccurrence(records)
    # Expected pairs from job1: excel-python, excel-sql, python-sql
    # From job2: python-sql
    # So python-sql count=2, others=1
    pair_map = {(p["skill_a"], p["skill_b"]): p["count"] for p in pairs}
    assert pair_map[("python", "sql")] == 2
    assert pair_map[("excel", "python")] == 1
    assert pair_map[("excel", "sql")] == 1
    # No inverted pairs
    assert ("sql", "python") not in pair_map


def test_skill_cooccurrence_no_inverted_duplicates():
    records = [_job("1", skills=("b", "a", "c"))]
    pairs = skill_cooccurrence(records)
    keys = [(p["skill_a"], p["skill_b"]) for p in pairs]
    # All pairs must be lexicographically ordered
    for a, b in keys:
        assert a < b


def test_skill_cooccurrence_empty_skills():
    records = [_job("1", skills=()), _job("2", skills=("python",))]
    pairs = skill_cooccurrence(records)
    assert pairs == []


def test_skill_cooccurrence_determinism():
    records = [
        _job("1", skills=("z", "a", "m")),
        _job("2", skills=("a", "z")),
    ]
    p1 = skill_cooccurrence(records)
    p2 = skill_cooccurrence(records)
    assert p1 == p2


# ---------------------------------------------------------------------------
# Compare roles
# ---------------------------------------------------------------------------

def test_compare_roles():
    records = [
        _job("1", role=RoleFamily.DATA_ANALYST, skills=("sql", "excel", "python")),
        _job("2", role=RoleFamily.DATA_ANALYST, skills=("sql", "power_bi")),
        _job("3", role=RoleFamily.BI_ANALYST, skills=("sql", "tableau", "power_bi")),
        _job("4", role=RoleFamily.BI_ANALYST, skills=("tableau",)),
    ]
    cmp = compare_roles(records, "data_analyst", "bi_analyst")
    assert cmp["count_a"] == 2
    assert cmp["count_b"] == 2
    assert "sql" in cmp["common_skills"]
    assert "power_bi" in cmp["common_skills"]
    assert "excel" in cmp["only_in_a"]
    assert "python" in cmp["only_in_a"]
    assert "tableau" in cmp["only_in_b"]
    # common and exclusives are sorted
    assert cmp["common_skills"] == sorted(cmp["common_skills"])
    assert cmp["only_in_a"] == sorted(cmp["only_in_a"])


def test_compare_roles_with_unknown():
    records = [
        _job("1", role=RoleFamily.UNKNOWN, skills=("git",)),
        _job("2", role=RoleFamily.DATA_ANALYST, skills=("sql",)),
    ]
    cmp = compare_roles(records, "unknown", "data_analyst")
    assert cmp["count_a"] == 1
    assert "git" in cmp["only_in_a"]


# ---------------------------------------------------------------------------
# Build evidence + invariants
# ---------------------------------------------------------------------------

def test_build_evidence_smoke():
    records = [
        _job("1", role=RoleFamily.DATA_ANALYST, seniority=Seniority.JUNIOR, skills=("python", "sql")),
        _job("2", role=RoleFamily.ML_ENGINEER, seniority=Seniority.SENIOR, skills=("python", "docker")),
        _job("3", is_valid=False, skills=("should_ignore",)),
        _job("4", is_duplicate=True, skills=("should_ignore",)),
    ]
    report = build_evidence(records)
    assert report.n_analysis_records == 2
    assert len(report.skill_frequency) >= 1
    assert "data_analyst" in report.skills_by_role
    assert "ml_engineer" in report.skills_by_role
    assert report.to_dict()["n_analysis_records"] == 2


def test_no_mutation_of_input():
    records = [
        _job("1", skills=("python", "sql")),
        _job("2", skills=("excel",)),
    ]
    snapshot = copy.deepcopy(records)
    skill_frequency(records)
    skills_by_role(records)
    skill_cooccurrence(records)
    build_evidence(records)
    # ProcessedJob is frozen, but the list itself must not be mutated
    assert len(records) == len(snapshot)
    assert records[0].id == snapshot[0].id
    assert records[0].skills_extracted == snapshot[0].skills_extracted


def test_determinism_full_report():
    records = [
        _job("1", role=RoleFamily.DATA_ANALYST, skills=("b", "a")),
        _job("2", role=RoleFamily.BI_ANALYST, skills=("a", "c")),
        _job("3", role=RoleFamily.DATA_ANALYST, skills=("b",)),
    ]
    r1 = build_evidence(records).to_dict()
    r2 = build_evidence(records).to_dict()
    assert r1 == r2


def test_empty_dataset():
    report = build_evidence([])
    assert report.n_analysis_records == 0
    assert report.skill_frequency == []
    assert report.role_distribution == []
    assert report.skill_cooccurrence == []
