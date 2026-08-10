"""
Flask routes for Tekmérion evidence explorer.

All metrics come from the pre-loaded EvidenceReport / PipelineResult.
No analytical logic and no external HTTP live here.
"""

from __future__ import annotations

from collections import Counter
from urllib.parse import unquote

from flask import (
    Blueprint,
    current_app,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    abort,
)

from analysis.evidence import compare_roles, analysis_records
from analysis.models import RoleFamily


bp = Blueprint("main", __name__)

ROLE_CHOICES = [r.value for r in RoleFamily]


def _dataset():
    from flask import g
    return g.dataset


def _evidence():
    return _dataset().evidence


def _pipeline():
    return _dataset().pipeline_result


def _meta():
    return _dataset().meta


def _dataset_id():
    from flask import g
    return g.dataset_id


@bp.route("/")
def index():
    ev = _evidence()
    pipe = _pipeline()
    return render_template(
        "index.html",
        pipeline=pipe,
        evidence=ev,
        top_skills=ev.skill_frequency[:12],
        role_dist=ev.role_distribution,
        sen_dist=ev.seniority_distribution,
    )


@bp.route("/skills")
def skills():
    ev = _evidence()
    role = request.args.get("role", "").strip() or None
    seniority = request.args.get("seniority", "").strip() or None

    if role and role in ev.skills_by_role:
        skills_list = ev.skills_by_role[role]
        context_label = f"Role family: {role}"
    elif seniority and seniority in ev.skills_by_seniority:
        skills_list = ev.skills_by_seniority[seniority]
        context_label = f"Seniority: {seniority}"
    else:
        skills_list = ev.skill_frequency
        context_label = "Global (all analysis records)"
        role = None
        seniority = None

    return render_template(
        "skills.html",
        skills=skills_list,
        context_label=context_label,
        selected_role=role,
        selected_seniority=seniority,
        role_choices=sorted(ev.skills_by_role.keys()),
        seniority_choices=sorted(ev.skills_by_seniority.keys()),
        n_records=ev.n_analysis_records,
    )


@bp.route("/roles")
@bp.route("/roles/<role_family>")
def roles(role_family: str | None = None):
    ev = _evidence()
    available = sorted(ev.skills_by_role.keys())

    if role_family is None:
        return render_template(
            "roles.html",
            role_family=None,
            available_roles=available,
            role_dist=ev.role_distribution,
        )

    if role_family not in available:
        flash(f"Role family «{role_family}» no tiene registros en el dataset activo.", "error")
        return redirect(url_for("main.roles"))

    skills = ev.skills_by_role.get(role_family, [])
    pipe = _pipeline()
    subset = [
        r for r in analysis_records(pipe.records)
        if r.role_family.value == role_family
    ]
    sen_counter = Counter(r.seniority.value for r in subset)
    seniority_in_role = sorted(
        [{"item": k, "count": v} for k, v in sen_counter.items()],
        key=lambda x: (-x["count"], x["item"]),
    )

    return render_template(
        "roles.html",
        role_family=role_family,
        available_roles=available,
        skills=skills,
        count=len(subset),
        seniority_in_role=seniority_in_role,
        role_dist=ev.role_distribution,
    )


@bp.route("/compare", methods=["GET", "POST"])
def compare():
    ev = _evidence()
    available = sorted(ev.skills_by_role.keys())

    role_a = request.values.get("role_a", "").strip()
    role_b = request.values.get("role_b", "").strip()

    comparison = None
    error = None

    if role_a or role_b:
        if not role_a or not role_b:
            error = "Seleccioná las dos role families."
        elif role_a == role_b:
            error = "Elegí dos role families distintas."
        elif role_a not in available or role_b not in available:
            error = "Una o ambas role families no existen en el dataset activo."
        else:
            pipe = _pipeline()
            comparison = compare_roles(pipe.records, role_a, role_b)

    return render_template(
        "compare.html",
        available_roles=available,
        role_a=role_a or "",
        role_b=role_b or "",
        comparison=comparison,
        error=error,
    )


@bp.route("/cooccurrence")
def cooccurrence():
    ev = _evidence()
    pairs = ev.skill_cooccurrence
    return render_template(
        "cooccurrence.html",
        pairs=pairs,
        n_records=ev.n_analysis_records,
    )


@bp.route("/jobs")
def jobs():
    """Minimal list of processed jobs in the active dataset."""
    pipe = _pipeline()
    records = sorted(pipe.records, key=lambda r: (r.title.lower(), r.id))
    return render_template(
        "jobs.html",
        jobs=records,
        total=len(records),
    )


@bp.route("/jobs/<path:job_id>")
def job_detail(job_id: str):
    """
    Job detail. ``job_id`` may contain colons (e.g. adzuna:123).
    Flask path converter preserves them; we also accept URL-encoded forms.
    """
    job_id = unquote(job_id)
    pipe = _pipeline()
    job = next((r for r in pipe.records if r.id == job_id), None)
    if job is None:
        abort(404)
    return render_template("job_detail.html", job=job)


@bp.route("/analysis", methods=["GET", "POST"])
def analysis():
    """
    Grounded market summary.

    GET  — show status + last analysis if any (never calls the provider).
    POST — explicit generate action only.
    """
    from analysis.generative.service import run_market_summary
    from analysis.generative.models import GenerativeError

    provider = current_app.config.get("GENERATIVE_PROVIDER")
    available = bool(current_app.config.get("GENERATIVE_AVAILABLE"))
    meta = _meta()
    ds_id = _dataset_id()
    store = current_app.config.setdefault("ANALYSIS_BY_DATASET", {})
    analysis_result = store.get(ds_id)
    error = None

    if request.method == "POST":
        if not available or provider is None:
            error = "Análisis generativo no configurado (provider disabled)."
        else:
            try:
                analysis_result = run_market_summary(
                    evidence=_evidence(),
                    dataset_mode=meta.mode,
                    dataset_source=meta.source,
                    dataset_label=meta.label,
                    provider=provider,
                    retrieved_at=meta.retrieved_at,
                    country=meta.country,
                    query_count=meta.query_count,
                )
                store[ds_id] = analysis_result
            except GenerativeError as exc:
                error = str(exc)
                analysis_result = None

    return render_template(
        "analysis.html",
        available=available,
        provider_name=getattr(provider, "name", "disabled"),
        analysis=analysis_result,
        error=error,
    )


@bp.route("/dataset", methods=["POST"])
def set_dataset():
    """
    Switch active dataset for this session.

    Accepts only internal registry ids. Never paths or URLs.
    Redirects to a stable page (home) to avoid stale job detail IDs.
    """
    from flask import session
    from app.registry import SESSION_KEY

    reg = current_app.config["DATASET_REGISTRY"]
    requested = (request.form.get("dataset_id") or "").strip()

    # Reject path-like / URL-like values explicitly
    if (
        not requested
        or ".." in requested
        or "/" in requested
        or "\\" in requested
        or requested.startswith("http:")
        or requested.startswith("https:")
        or requested.startswith("file:")
    ):
        flash("Dataset no válido.", "error")
        return redirect(url_for("main.index"))

    if not reg.is_known(requested):
        flash("Dataset desconocido; se mantiene la selección actual.", "error")
        return redirect(url_for("main.index"))

    session[SESSION_KEY] = requested
    flash("Dataset actualizado.", "ok")
    return redirect(url_for("main.index"))


@bp.route("/analysis/roles", methods=["GET", "POST"])
def analysis_roles():
    """
    Grounded role comparison (V0.5.4).

    GET  — form + last comparison for active dataset (no provider call).
    POST — validate roles, generate comparison.
    """
    from analysis.generative.service import run_role_comparison
    from analysis.generative.models import GenerativeError
    from analysis.generative.comparison import available_roles

    provider = current_app.config.get("GENERATIVE_PROVIDER")
    available = bool(current_app.config.get("GENERATIVE_AVAILABLE"))
    meta = _meta()
    ds_id = _dataset_id()
    roles = available_roles(_evidence())

    store = current_app.config.setdefault("ROLE_COMPARISON_BY_DATASET", {})
    # store last comparison per dataset_id only (clears scope on dataset switch)
    comparison = store.get(ds_id)
    error = None
    role_a = request.values.get("role_a", "").strip()
    role_b = request.values.get("role_b", "").strip()

    if request.method == "POST":
        if not available or provider is None:
            error = "Análisis generativo no configurado (provider disabled)."
        elif not role_a or not role_b:
            error = "Seleccioná las dos role families."
        elif role_a == role_b:
            error = "Elegí dos role families distintas."
        elif role_a not in roles or role_b not in roles:
            error = "Una o ambas role families no existen en el dataset activo."
        else:
            try:
                comparison = run_role_comparison(
                    evidence=_evidence(),
                    role_a=role_a,
                    role_b=role_b,
                    dataset_mode=meta.mode,
                    dataset_source=meta.source,
                    dataset_label=meta.label,
                    provider=provider,
                    records=_pipeline().records,
                    retrieved_at=meta.retrieved_at,
                    country=meta.country,
                )
                store[ds_id] = comparison
            except GenerativeError as exc:
                error = str(exc)
                comparison = None

    return render_template(
        "analysis_roles.html",
        available=available,
        provider_name=getattr(provider, "name", "disabled"),
        roles=roles,
        role_a=role_a,
        role_b=role_b,
        comparison=comparison,
        error=error,
    )
