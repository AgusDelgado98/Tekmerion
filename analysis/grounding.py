"""
Grounding payload for grounded generative analysis (V0.5.0).

Built exclusively from EvidenceReport + DatasetMeta.
Never includes raw vacancies, secrets, or filesystem paths.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from analysis.evidence import EvidenceReport


PROMPT_TASK_MARKET_SUMMARY = "market_summary"


@dataclass(frozen=True)
class EvidenceItem:
    """One traceable fact exposed to the model."""

    id: str
    label: str
    value: Any
    unit: Optional[str] = None  # e.g. "count", "proportion", "list"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "label": self.label, "value": self.value}
        if self.unit is not None:
            d["unit"] = self.unit
        return d


@dataclass
class GroundingPayload:
    """
    Serializable grounding for one analysis request.

    All quantitative claims in GeneratedAnalysis must cite evidence_refs
    whose ids appear in ``items``.
    """

    task: str
    dataset_mode: str
    dataset_source: str
    dataset_label: str
    n_analysis_records: int
    items: list[EvidenceItem] = field(default_factory=list)
    retrieved_at: Optional[str] = None
    country: Optional[str] = None
    query_count: int = 0

    def item_ids(self) -> set[str]:
        return {i.id for i in self.items}

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "dataset": {
                "mode": self.dataset_mode,
                "source": self.dataset_source,
                "label": self.dataset_label,
                "retrieved_at": self.retrieved_at,
                "country": self.country,
                "query_count": self.query_count,
                "n_analysis_records": self.n_analysis_records,
            },
            "evidence": [i.to_dict() for i in self.items],
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Deterministic JSON (sorted keys)."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
        )

    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:16]


def _pct(proportion: float) -> float:
    return round(float(proportion) * 100, 1)


def build_grounding_payload(
    evidence: EvidenceReport,
    *,
    dataset_mode: str,
    dataset_source: str,
    dataset_label: str,
    retrieved_at: Optional[str] = None,
    country: Optional[str] = None,
    query_count: int = 0,
    task: str = PROMPT_TASK_MARKET_SUMMARY,
    top_skills: int = 10,
    top_cooccurrence: int = 5,
) -> GroundingPayload:
    """
    Derive a GroundingPayload from an EvidenceReport + dataset metadata.

    Only aggregates already computed by the evidence layer are included.
    """
    n = evidence.n_analysis_records
    items: list[EvidenceItem] = []

    items.append(
        EvidenceItem(
            id="dataset.n_analysis_records",
            label="Number of analysis records (valid, non-duplicate)",
            value=n,
            unit="count",
        )
    )
    items.append(
        EvidenceItem(
            id="dataset.mode",
            label="Dataset mode",
            value=dataset_mode,
        )
    )
    items.append(
        EvidenceItem(
            id="dataset.source",
            label="Dataset source identifier",
            value=dataset_source,
        )
    )
    if country:
        items.append(
            EvidenceItem(id="dataset.country", label="Market country code", value=country)
        )
    if retrieved_at:
        items.append(
            EvidenceItem(
                id="dataset.retrieved_at",
                label="Dataset retrieval timestamp",
                value=retrieved_at,
            )
        )
    if query_count:
        items.append(
            EvidenceItem(
                id="dataset.query_count",
                label="Number of search queries in market batch",
                value=query_count,
                unit="count",
            )
        )

    # Role distribution
    role_ranking: list[dict[str, Any]] = []
    for entry in evidence.role_distribution:
        role = entry["item"]
        count = entry["count"]
        prop = entry.get("proportion", 0)
        items.append(
            EvidenceItem(
                id=f"roles.{role}.count",
                label=f"Count of role family {role}",
                value=count,
                unit="count",
            )
        )
        items.append(
            EvidenceItem(
                id=f"roles.{role}.pct",
                label=f"Percentage of role family {role}",
                value=_pct(prop),
                unit="percent",
            )
        )
        role_ranking.append({"role": role, "count": count, "pct": _pct(prop)})
    items.append(
        EvidenceItem(
            id="roles.ranking",
            label="Role families ranked by frequency",
            value=role_ranking,
            unit="list",
        )
    )

    # Seniority
    sen_ranking: list[dict[str, Any]] = []
    for entry in evidence.seniority_distribution:
        level = entry["item"]
        count = entry["count"]
        prop = entry.get("proportion", 0)
        items.append(
            EvidenceItem(
                id=f"seniority.{level}.count",
                label=f"Count of seniority {level}",
                value=count,
                unit="count",
            )
        )
        items.append(
            EvidenceItem(
                id=f"seniority.{level}.pct",
                label=f"Percentage of seniority {level}",
                value=_pct(prop),
                unit="percent",
            )
        )
        sen_ranking.append({"seniority": level, "count": count, "pct": _pct(prop)})
    items.append(
        EvidenceItem(
            id="seniority.ranking",
            label="Seniority levels ranked by frequency",
            value=sen_ranking,
            unit="list",
        )
    )

    # Skills global
    skill_ranking: list[dict[str, Any]] = []
    for entry in evidence.skill_frequency[:top_skills]:
        skill = entry["item"]
        count = entry["count"]
        prop = entry.get("proportion", 0)
        # sanitize id segment
        safe = skill.replace(" ", "_")
        items.append(
            EvidenceItem(
                id=f"skills.{safe}.count",
                label=f"Vacancies mentioning skill {skill}",
                value=count,
                unit="count",
            )
        )
        items.append(
            EvidenceItem(
                id=f"skills.{safe}.pct",
                label=f"Percentage of vacancies mentioning {skill}",
                value=_pct(prop),
                unit="percent",
            )
        )
        skill_ranking.append({"skill": skill, "count": count, "pct": _pct(prop)})
    items.append(
        EvidenceItem(
            id="skills.ranking",
            label=f"Top {top_skills} skills by frequency",
            value=skill_ranking,
            unit="list",
        )
    )

    # Top skill per major role (compact)
    for role in sorted(evidence.skills_by_role.keys()):
        role_skills = evidence.skills_by_role[role][:3]
        if not role_skills:
            continue
        compact = [
            {"skill": s["item"], "count": s["count"], "pct": _pct(s.get("proportion", 0))}
            for s in role_skills
        ]
        items.append(
            EvidenceItem(
                id=f"skills_by_role.{role}.top",
                label=f"Top skills within role {role}",
                value=compact,
                unit="list",
            )
        )

    # Co-occurrence sample
    pairs = []
    for entry in evidence.skill_cooccurrence[:top_cooccurrence]:
        pairs.append(
            {
                "skill_a": entry["skill_a"],
                "skill_b": entry["skill_b"],
                "count": entry["count"],
            }
        )
    if pairs:
        items.append(
            EvidenceItem(
                id="skills.cooccurrence.top",
                label="Top skill co-occurrence pairs",
                value=pairs,
                unit="list",
            )
        )

    return GroundingPayload(
        task=task,
        dataset_mode=dataset_mode,
        dataset_source=dataset_source,
        dataset_label=dataset_label,
        n_analysis_records=n,
        items=items,
        retrieved_at=retrieved_at,
        country=country,
        query_count=query_count,
    )
