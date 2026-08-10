"""
Flask routes for Tekmérion evidence explorer.

All metrics come from the pre-loaded EvidenceReport / PipelineResult.
No analytical logic lives here.
"""

from __future__ import annotations

from flask import (
    Blueprint,
    current_app,
    render_template,
    request,
    redirect,
    url_for,
    flash,
)

from analysis.evidence import compare_roles
from analysis.models import RoleFamily


bp = Blueprint("main", __name__)

ROLE_CHOICES = [r.value for r in RoleFamily]


def _evidence():
    return current_app.config["EVIDENCE"]


def _pipeline():
    return current_app.config["PIPELINE_RESULT"]


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
        flash(f"Role family «{role_family}» no tiene registros en la muestra.", "error")
        return redirect(url_for("main.roles"))

    skills = ev.skills_by_role.get(role_family, [])
    # Seniority breakdown for this role (from raw processed records)
    pipe = _pipeline()
    from analysis.evidence import analysis_records
    subset = [
        r for r in analysis_records(pipe.records)
        if r.role_family.value == role_family
    ]
    from collections import Counter
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
            error = "Una o ambas role families no existen en la muestra."
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
