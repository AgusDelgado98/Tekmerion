"""
Tekmérion Flask application factory.

V0.5.3: dataset is selected per session via DatasetRegistry.
The app never calls Adzuna or other external APIs.
"""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, session, g

from app.dataset import (
    ENV_DATA_MODE,
    ENV_MARKET_FILE,
    MODE_SYNTHETIC,
    DatasetError,
    resolve_data_mode,
)
from app.registry import (
    SESSION_KEY,
    DatasetRegistry,
    build_registry,
)


def create_app(*, data_mode: str | None = None, market_file: str | None = None, market_dir=None) -> Flask:
    """
    Application factory.

    Optional ``data_mode`` / ``market_file`` set the *default* dataset
    (server config). Users can switch among registry entries per session.
    """
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["SECRET_KEY"] = os.environ.get(
        "TEKMERION_SECRET_KEY", "tekmerion-dev-only-change-in-prod"
    )

    mode = data_mode if data_mode is not None else os.environ.get(ENV_DATA_MODE)
    mfile = market_file if market_file is not None else os.environ.get(ENV_MARKET_FILE)

    # showroom is a demo alias for registry default; not a pipeline mode
    if (mode or "").strip().lower() == "showroom":
        resolved_mode, path = "showroom", None
    else:
        resolved_mode, path = resolve_data_mode(mode=mode, market_file=mfile)
    # Explicit market file must be valid (no silent fallback) — preserves V0.4.5 contract
    if resolved_mode == "market" and path is not None and resolved_mode != "showroom":
        from app.dataset import load_market_dataset

        load_market_dataset(path)  # raises DatasetError if missing/invalid
    registry_kwargs = {
        "default_mode": resolved_mode,
        "default_market_file": path,
    }
    if market_dir is not None:
        registry_kwargs["market_dir"] = market_dir
    registry = build_registry(**registry_kwargs)
    app.config["DATASET_REGISTRY"] = registry
    app.config["DEFAULT_DATASET_ID"] = registry.default_id
    # Analysis results keyed by dataset_id (content is dataset-scoped, not user-scoped)
    app.config["ANALYSIS_BY_DATASET"] = {}
    app.config["ROLE_COMPARISON_BY_DATASET"] = {}

    # Generative provider (never required for core app)
    from analysis.generative.providers import get_provider_from_env, GenerativeError as _GE

    try:
        provider = get_provider_from_env()
    except _GE:
        from analysis.generative.providers import DisabledProvider

        provider = DisabledProvider()
    app.config["GENERATIVE_PROVIDER"] = provider
    app.config["GENERATIVE_AVAILABLE"] = provider.is_available()

    from app.routes import bp

    app.register_blueprint(bp)

    @app.before_request
    def _bind_active_dataset():
        reg: DatasetRegistry = app.config["DATASET_REGISTRY"]
        ds_id = session.get(SESSION_KEY)
        if not ds_id or not reg.is_known(ds_id):
            ds_id = app.config["DEFAULT_DATASET_ID"]
            session[SESSION_KEY] = ds_id
        try:
            dataset = reg.resolve(ds_id)
        except DatasetError:
            ds_id = reg.default_id
            session[SESSION_KEY] = ds_id
            dataset = reg.resolve(ds_id)
        g.dataset = dataset
        g.dataset_id = ds_id
        g.dataset_meta = dataset.meta

    @app.context_processor
    def _inject_dataset_meta():
        reg: DatasetRegistry = app.config["DATASET_REGISTRY"]
        return {
            "dataset_meta": getattr(g, "dataset_meta", None),
            "dataset_id": getattr(g, "dataset_id", None),
            "dataset_options": reg.list_entries(),
            "generative_available": app.config.get("GENERATIVE_AVAILABLE", False),
            "generative_provider": getattr(
                app.config.get("GENERATIVE_PROVIDER"), "name", "disabled"
            ),
        }

    return app
