"""
Role comparison grounding (V0.5.4).

Builds a focused GroundingPayload for two role families from EvidenceReport
(+ optional ProcessedJob list for seniority breakdown and compare_roles).
Shared / exclusive skills are computed deterministically — never by the LLM.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from analysis.evidence import EvidenceReport, analysis_records, compare_roles
from analysis.grounding import EvidenceItem, GroundingPayload, PROMPT_TASK_MARKET_SUMMARY
from analysis.models import ProcessedJob
from analysis.generative.models import GenerativeError

TASK_ROLE_COMPARISON = "role_comparison"
SMALL_SAMPLE_THRESHOLD = 5
PROMPT_VERSION_ROLE_COMPARISON = "role_comparison.v1"


@dataclass(frozen=True)
class RolePair:
    role_a: str
    role_b: str

    def canonical(self) -> tuple[str, str]:
        return tuple(sorted((self.role_a, self.role_b)))  # type: ignore[return-value]


def available_roles(evidence: EvidenceReport) -> list[str]:
    """Canonical role ids present in the active evidence (sorted)."""
    return sorted(evidence.skills_by_role.keys())


def validate_role_pair(evidence: EvidenceReport, role_a: str, role_b: str) -> RolePair:
    roles = set(available_roles(evidence))
    if role_a not in roles:
        raise GenerativeError(f"Unknown role family: {role_a!r}")
    if role_b not in roles:
        raise GenerativeError(f"Unknown role family: {role_b!r}")
    if role_a == role_b:
        raise GenerativeError("role_a and role_b must be distinct")
    return RolePair(role_a=role_a, role_b=role_b)


def _pct(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * count / total, 1)


def _role_count(evidence: EvidenceReport, role: str) -> int:
    for entry in evidence.role_distribution:
        if entry["item"] == role:
            return int(entry["count"])
    return 0


def _role_pct(evidence: EvidenceReport, role: str) -> float:
    for entry in evidence.role_distribution:
        if entry["item"] == role:
            return round(float(entry.get("proportion", 0)) * 100, 1)
    return 0.0


def _skill_map(evidence: EvidenceReport, role: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for entry in evidence.skills_by_role.get(role, []):
        skill = entry["item"]
        out[skill] = {
            "count": int(entry["count"]),
            "pct": round(float(entry.get("proportion", 0)) * 100, 1),
        }
    return out


def _seniority_for_role(records: Sequence[ProcessedJob], role: str) -> list[dict[str, Any]]:
    jobs = [r for r in analysis_records(records) if r.role_family.value == role]
    n = len(jobs)
    counter = Counter(r.seniority.value for r in jobs)
    items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    return [
        {"seniority": k, "count": v, "pct": _pct(v, n)}
        for k, v in items
    ]


def build_role_comparison_grounding(
    evidence: EvidenceReport,
    role_a: str,
    role_b: str,
    *,
    dataset_mode: str,
    dataset_source: str,
    dataset_label: str,
    records: Optional[Sequence[ProcessedJob]] = None,
    retrieved_at: Optional[str] = None,
    country: Optional[str] = None,
) -> GroundingPayload:
    """
    Focused grounding for role_comparison.

    Shared/exclusive skills come from set algebra on skills_by_role
    (or compare_roles when records are provided — same result).
    """
    pair = validate_role_pair(evidence, role_a, role_b)
    n = evidence.n_analysis_records
    map_a = _skill_map(evidence, pair.role_a)
    map_b = _skill_map(evidence, pair.role_b)
    set_a = set(map_a)
    set_b = set(map_b)
    shared = sorted(set_a & set_b)
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)

    count_a = _role_count(evidence, pair.role_a)
    count_b = _role_count(evidence, pair.role_b)
    pct_a = _role_pct(evidence, pair.role_a)
    pct_b = _role_pct(evidence, pair.role_b)

    # Frequency diffs for shared skills
    shared_detail = []
    for skill in shared:
        ca = map_a[skill]["count"]
        cb = map_b[skill]["count"]
        pa = map_a[skill]["pct"]
        pb = map_b[skill]["pct"]
        if pa > pb:
            relation = "more_frequent_in_a"
        elif pb > pa:
            relation = "more_frequent_in_b"
        else:
            relation = "equal"
        shared_detail.append(
            {
                "skill": skill,
                "count_a": ca,
                "count_b": cb,
                "pct_a": pa,
                "pct_b": pb,
                "relation": relation,
            }
        )

    items: list[EvidenceItem] = [
        EvidenceItem(id="dataset.n_analysis_records", label="Analysis records", value=n, unit="count"),
        EvidenceItem(id="dataset.mode", label="Dataset mode", value=dataset_mode),
        EvidenceItem(id="dataset.source", label="Dataset source", value=dataset_source),
        EvidenceItem(id="comparison.role_a", label="Role A", value=pair.role_a),
        EvidenceItem(id="comparison.role_b", label="Role B", value=pair.role_b),
        EvidenceItem(
            id=f"roles.{pair.role_a}.count",
            label=f"Count of {pair.role_a}",
            value=count_a,
            unit="count",
        ),
        EvidenceItem(
            id=f"roles.{pair.role_b}.count",
            label=f"Count of {pair.role_b}",
            value=count_b,
            unit="count",
        ),
        EvidenceItem(
            id=f"roles.{pair.role_a}.pct",
            label=f"Percentage of {pair.role_a}",
            value=pct_a,
            unit="percent",
        ),
        EvidenceItem(
            id=f"roles.{pair.role_b}.pct",
            label=f"Percentage of {pair.role_b}",
            value=pct_b,
            unit="percent",
        ),
        EvidenceItem(
            id="comparison.shared_skills",
            label="Skills present in both roles",
            value=shared,
            unit="list",
        ),
        EvidenceItem(
            id=f"comparison.only_{pair.role_a}",
            label=f"Skills only in {pair.role_a}",
            value=only_a,
            unit="list",
        ),
        EvidenceItem(
            id=f"comparison.only_{pair.role_b}",
            label=f"Skills only in {pair.role_b}",
            value=only_b,
            unit="list",
        ),
        EvidenceItem(
            id="comparison.shared_detail",
            label="Shared skill frequency comparison",
            value=shared_detail,
            unit="list",
        ),
        EvidenceItem(
            id="comparison.small_sample_threshold",
            label="Product threshold for small-sample disclaimer (not statistical significance)",
            value=SMALL_SAMPLE_THRESHOLD,
            unit="count",
        ),
    ]

    for role, smap in ((pair.role_a, map_a), (pair.role_b, map_b)):
        for skill, vals in sorted(smap.items()):
            safe = skill.replace(" ", "_")
            items.append(
                EvidenceItem(
                    id=f"roles.{role}.skills.{safe}.count",
                    label=f"{skill} count within {role}",
                    value=vals["count"],
                    unit="count",
                )
            )
            items.append(
                EvidenceItem(
                    id=f"roles.{role}.skills.{safe}.pct",
                    label=f"{skill} pct within {role}",
                    value=vals["pct"],
                    unit="percent",
                )
            )

    if records is not None:
        for role in (pair.role_a, pair.role_b):
            sen = _seniority_for_role(records, role)
            items.append(
                EvidenceItem(
                    id=f"roles.{role}.seniority",
                    label=f"Seniority breakdown for {role}",
                    value=sen,
                    unit="list",
                )
            )

    if country:
        items.append(EvidenceItem(id="dataset.country", label="Country", value=country))
    if retrieved_at:
        items.append(
            EvidenceItem(id="dataset.retrieved_at", label="Retrieved at", value=retrieved_at)
        )

    return GroundingPayload(
        task=TASK_ROLE_COMPARISON,
        dataset_mode=dataset_mode,
        dataset_source=dataset_source,
        dataset_label=dataset_label,
        n_analysis_records=n,
        items=items,
        retrieved_at=retrieved_at,
        country=country,
        query_count=0,
    )


def allowed_role_refs(role_a: str, role_b: str) -> set[str]:
    """Prefix/id set used to reject third-role refs."""
    return {
        f"roles.{role_a}.",
        f"roles.{role_b}.",
        "comparison.",
        "dataset.",
    }


def ref_in_scope(ref: str, role_a: str, role_b: str) -> bool:
    if ref.startswith("dataset.") or ref.startswith("comparison."):
        return True
    if ref.startswith(f"roles.{role_a}.") or ref.startswith(f"roles.{role_b}."):
        return True
    # exact role count/pct ids already covered by prefix
    return False
