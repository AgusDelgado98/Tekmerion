"""
Tests for the Tekmérion Flask evidence explorer.
"""

from __future__ import annotations

import pytest

from app import create_app


from pathlib import Path

MARKET_FIXTURE = Path(__file__).parent / "fixtures" / "market" / "market_ar_fixture.json"


@pytest.fixture
def client():
    """Default app uses synthetic sample (explicit for isolation)."""
    app = create_app(data_mode="synthetic")
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def market_client():
    app = create_app(data_mode="market", market_file=str(MARKET_FIXTURE))
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_app_starts(client):
    assert client is not None


def test_index_ok(client):
    rv = client.get("/")
    assert rv.status_code == 200
    assert b"Overview" in rv.data
    assert b"muestra sint" in rv.data.lower() or b"sint" in rv.data.lower()
    # Key metrics should appear
    assert b"Analizados" in rv.data or b"analizados" in rv.data.lower()


def test_skills_ok(client):
    rv = client.get("/skills")
    assert rv.status_code == 200
    assert b"Skills" in rv.data
    assert b"sql" in rv.data or b"python" in rv.data


def test_skills_filter_by_role(client):
    rv = client.get("/skills?role=data_analyst")
    assert rv.status_code == 200
    assert b"data_analyst" in rv.data


def test_skills_filter_by_seniority(client):
    rv = client.get("/skills?seniority=senior")
    assert rv.status_code == 200
    assert b"senior" in rv.data


def test_roles_list_ok(client):
    rv = client.get("/roles")
    assert rv.status_code == 200
    assert b"Roles" in rv.data


def test_roles_detail_ok(client):
    rv = client.get("/roles/data_analyst")
    assert rv.status_code == 200
    assert b"data_analyst" in rv.data


def test_roles_unknown_redirects(client):
    rv = client.get("/roles/this_role_does_not_exist", follow_redirects=False)
    assert rv.status_code in (302, 303)


def test_compare_get_ok(client):
    rv = client.get("/compare")
    assert rv.status_code == 200
    assert b"Comparar" in rv.data


def test_compare_valid(client):
    rv = client.get("/compare?role_a=data_analyst&role_b=bi_analyst")
    assert rv.status_code == 200
    assert b"data_analyst" in rv.data
    assert b"bi_analyst" in rv.data
    # Should show common / only sections
    assert b"Comunes" in rv.data or b"comunes" in rv.data.lower()


def test_compare_same_role_error(client):
    rv = client.get("/compare?role_a=data_analyst&role_b=data_analyst")
    assert rv.status_code == 200
    assert b"distintas" in rv.data.lower() or b"error" in rv.data.lower()


def test_compare_missing_role_error(client):
    rv = client.get("/compare?role_a=data_analyst")
    assert rv.status_code == 200
    # Should show validation message
    body = rv.data.decode("utf-8").lower()
    assert "seleccion" in body or "eleg" in body or "dos" in body


def test_cooccurrence_ok(client):
    rv = client.get("/cooccurrence")
    assert rv.status_code == 200
    assert b"Co-ocurrencia" in rv.data or b"co-ocurrencia" in rv.data.lower()
    # At least one known pair from sample
    assert b"python" in rv.data or b"sql" in rv.data


def test_evidence_is_used(client):
    """Smoke: index should reflect the known analysis-record count from sample."""
    rv = client.get("/")
    # 15 analysis records in the synthetic sample
    assert b"15" in rv.data


# ---------------------------------------------------------------------------
# Market mode
# ---------------------------------------------------------------------------

def test_market_index(market_client):
    rv = market_client.get("/")
    assert rv.status_code == 200
    body = rv.data.decode("utf-8")
    assert "Market" in body or "market" in body.lower()
    assert "AR" in body or "ar" in body
    assert "Flask no realiza requests" in body or "artifact" in body.lower()


def test_market_skills(market_client):
    rv = market_client.get("/skills")
    assert rv.status_code == 200
    assert b"sql" in rv.data or b"python" in rv.data


def test_market_jobs_list(market_client):
    rv = market_client.get("/jobs")
    assert rv.status_code == 200
    assert b"Vacantes" in rv.data
    assert b"adzuna" in rv.data


def test_market_job_detail_with_source_url(market_client):
    # Known id from fixture
    rv = market_client.get("/jobs/adzuna:7001001")
    assert rv.status_code == 200
    body = rv.data.decode("utf-8")
    assert "Data Analyst" in body
    assert "Ver fuente original" in body
    assert "source_url" not in body.lower() or "fuente" in body.lower()
    assert "Retrieved" in body or "retrieved" in body.lower()
    assert "7001001" in body


def test_market_job_detail_404(market_client):
    rv = market_client.get("/jobs/adzuna:does-not-exist")
    assert rv.status_code == 404


def test_synthetic_jobs_list(client):
    rv = client.get("/jobs")
    assert rv.status_code == 200
    assert b"job_001" in rv.data or b"Data Analyst" in rv.data


def test_synthetic_job_detail_no_source_link(client):
    rv = client.get("/jobs/job_001")
    assert rv.status_code == 200
    body = rv.data.decode("utf-8")
    assert "Data Analyst" in body
    # Synthetic records typically have no source_url
    assert "Ver fuente original" not in body


def test_dataset_badge_synthetic(client):
    rv = client.get("/")
    assert b"Synthetic" in rv.data or b"sint" in rv.data.lower()


def test_market_mode_missing_file_raises():
    with pytest.raises(Exception):
        create_app(data_mode="market", market_file="/tmp/no_such_market_artifact.json")
