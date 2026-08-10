"""
Tests for the Tekmérion pipeline.

Coverage goals:
- valid / invalid records
- multiple records
- mixture
- normalization
- seniority & role family
- skills extraction
- duplicates
- no mutation of input
- determinism
- process_file (with and without output_path)
"""

from __future__ import annotations

import json
import copy
from pathlib import Path

import pytest

from analysis.pipeline import process_records, process_file
from analysis.models import RoleFamily, Seniority, ProcessedJob


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_VALID = {
    "id": "t001",
    "title": "Data Analyst",
    "company": "TestCo",
    "location": "BA",
    "description": "SQL, Python, Power BI and Excel required.",
    "salary_min": 100,
    "salary_max": 200,
    "currency": "ARS",
    "posted_date": "2025-01-01",
    "source": "test",
}

SAMPLE_INVALID = {
    "id": "t002",
    "title": "",  # empty title → invalid
    "company": "TestCo",
    "location": "BA",
    "description": "Something",
    "source": "test",
}


def _make_job(**overrides):
    base = copy.deepcopy(SAMPLE_VALID)
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------

def test_process_single_valid_record():
    result = process_records([SAMPLE_VALID])
    assert result.total_input == 1
    assert result.valid_count == 1
    assert result.invalid_count == 0
    assert result.duplicate_count == 0

    job = result.records[0]
    assert isinstance(job, ProcessedJob)
    assert job.is_valid is True
    assert job.id == "t001"
    assert job.role_family == RoleFamily.DATA_ANALYST
    assert "sql" in job.skills_extracted
    assert "python" in job.skills_extracted
    assert "power_bi" in job.skills_extracted
    assert "excel" in job.skills_extracted


def test_process_invalid_record():
    result = process_records([SAMPLE_INVALID])
    assert result.valid_count == 0
    assert result.invalid_count == 1
    job = result.records[0]
    assert job.is_valid is False
    assert "missing_or_empty_title" in job.validation_errors


def test_process_multiple_records():
    records = [SAMPLE_VALID, _make_job(id="t003", title="Senior Data Engineer")]
    result = process_records(records)
    assert result.total_input == 2
    assert result.valid_count == 2


def test_mixture_valid_and_invalid():
    records = [SAMPLE_VALID, SAMPLE_INVALID, _make_job(id="t004", title="BI Analyst")]
    result = process_records(records)
    assert result.total_input == 3
    assert result.valid_count == 2
    assert result.invalid_count == 1


# ---------------------------------------------------------------------------
# No mutation
# ---------------------------------------------------------------------------

def test_input_is_not_mutated():
    original = [copy.deepcopy(SAMPLE_VALID)]
    snapshot = copy.deepcopy(original)

    process_records(original)

    assert original == snapshot
    # Also check the dict itself wasn't touched
    assert original[0]["title"] == "Data Analyst"


def test_input_list_identity_preserved_conceptually():
    """We make a shallow copy of the list, so caller's list object stays intact."""
    records = [SAMPLE_VALID]
    process_records(records)
    assert len(records) == 1  # still the same list


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism():
    records = [
        SAMPLE_VALID,
        _make_job(id="t010", title="ML Engineer", description="Python, Docker, MLflow"),
        SAMPLE_INVALID,
    ]
    r1 = process_records(records)
    r2 = process_records(records)

    assert r1.summary() == r2.summary()
    for a, b in zip(r1.records, r2.records):
        assert a.to_dict() == b.to_dict()


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_role_family_data_analyst():
    result = process_records([_make_job(title="Analista de Datos")])
    assert result.records[0].role_family == RoleFamily.DATA_ANALYST


def test_role_family_bi_analyst():
    result = process_records([_make_job(title="Senior Business Intelligence Analyst")])
    assert result.records[0].role_family == RoleFamily.BI_ANALYST


def test_role_family_ml_engineer():
    result = process_records([_make_job(title="Machine Learning Engineer")])
    assert result.records[0].role_family == RoleFamily.ML_ENGINEER


def test_role_family_data_scientist():
    result = process_records([_make_job(title="Junior Data Scientist")])
    assert result.records[0].role_family == RoleFamily.DATA_SCIENTIST


def test_role_family_data_engineer():
    result = process_records([_make_job(title="Senior Data Engineer")])
    assert result.records[0].role_family == RoleFamily.DATA_ENGINEER


def test_role_family_ai_analyst():
    result = process_records([_make_job(title="AI Analyst", description="prompt engineering and LLMs")])
    assert result.records[0].role_family == RoleFamily.AI_ANALYST


def test_role_family_business_analyst():
    result = process_records([_make_job(title="Business Analyst")])
    assert result.records[0].role_family == RoleFamily.BUSINESS_ANALYST


def test_seniority_junior():
    result = process_records([_make_job(title="Junior Data Analyst")])
    assert result.records[0].seniority == Seniority.JUNIOR


def test_seniority_senior():
    result = process_records([_make_job(title="Senior BI Analyst")])
    assert result.records[0].seniority == Seniority.SENIOR


def test_seniority_lead():
    result = process_records([_make_job(title="Lead Data Scientist")])
    assert result.records[0].seniority == Seniority.LEAD


def test_seniority_unknown_when_no_signal():
    result = process_records([_make_job(title="Data Analyst")])
    # No junior/senior/lead keyword → UNKNOWN (explicit)
    assert result.records[0].seniority == Seniority.UNKNOWN


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

def test_skills_extraction_basic():
    desc = "We need Python, SQL, Tableau and Docker experience."
    result = process_records([_make_job(description=desc)])
    skills = result.records[0].skills_extracted
    assert "python" in skills
    assert "sql" in skills
    assert "tableau" in skills
    assert "docker" in skills
    # Must be sorted (deterministic)
    assert list(skills) == sorted(skills)


def test_skills_are_tuple():
    result = process_records([SAMPLE_VALID])
    assert isinstance(result.records[0].skills_extracted, tuple)


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

def test_duplicate_detection():
    job_a = _make_job(id="dup1", title="Data Analyst", company="SameCo",
                      description="Identical description for testing duplicates.")
    job_b = _make_job(id="dup2", title="Data Analyst", company="SameCo",
                      description="Identical description for testing duplicates.")

    result = process_records([job_a, job_b])
    assert result.duplicate_count == 1
    assert result.records[0].is_duplicate is False
    assert result.records[1].is_duplicate is True
    assert result.records[1].duplicate_of == "dup1"


def test_non_duplicates_different_company():
    job_a = _make_job(id="a", title="Data Analyst", company="CoA", description="Same text")
    job_b = _make_job(id="b", title="Data Analyst", company="CoB", description="Same text")
    result = process_records([job_a, job_b])
    assert result.duplicate_count == 0


# ---------------------------------------------------------------------------
# process_file
# ---------------------------------------------------------------------------

def test_process_file_without_output(tmp_path: Path):
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps([SAMPLE_VALID]), encoding="utf-8")

    result = process_file(input_file)
    assert result.valid_count == 1
    assert result.records[0].id == "t001"


def test_process_file_with_output(tmp_path: Path):
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "out" / "processed.json"
    input_file.write_text(json.dumps([SAMPLE_VALID, SAMPLE_INVALID]), encoding="utf-8")

    result = process_file(input_file, output_path=output_file)

    assert output_file.exists()
    written = json.loads(output_file.read_text(encoding="utf-8"))
    assert len(written) == 2
    assert written[0]["id"] == "t001"
    assert written[0]["is_valid"] is True
    assert written[1]["is_valid"] is False


def test_process_file_missing_raises():
    with pytest.raises(FileNotFoundError):
        process_file("/tmp/non_existent_tekmerion_file_12345.json")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_list():
    result = process_records([])
    assert result.total_input == 0
    assert result.valid_count == 0
    assert result.records == []


def test_non_dict_record():
    result = process_records(["not a dict", 42])
    assert result.total_input == 2
    assert result.invalid_count == 2
    assert all(not r.is_valid for r in result.records)


def test_normalized_title():
    result = process_records([_make_job(title="  data   analyst  ")])
    assert result.records[0].normalized_title == "Data Analyst"
