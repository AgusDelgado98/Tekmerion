"""
Tekmérion Flask application factory.

The app never recomputes analytics inside views.
It loads pipeline + evidence once at startup
and serves the already-calculated EvidenceReport.
"""

from __future__ import annotations

from pathlib import Path
from flask import Flask

from analysis.pipeline import process_file
from analysis.evidence import build_evidence, EvidenceReport
from analysis.models import PipelineResult


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "raw" / "sample_jobs.json"


def _load_data() -> tuple[PipelineResult, EvidenceReport]:
    """
    Single source of truth for the web layer.
    Runs pipeline + evidence once. Deterministic for the same sample.
    """
    pipeline_result = process_file(SAMPLE_PATH)
    evidence = build_evidence(pipeline_result.records)
    return pipeline_result, evidence


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["SECRET_KEY"] = "tekmerion-dev-only-change-in-prod"

    # Load once at startup
    pipeline_result, evidence = _load_data()
    app.config["PIPELINE_RESULT"] = pipeline_result
    app.config["EVIDENCE"] = evidence

    from app.routes import bp
    app.register_blueprint(bp)

    return app
