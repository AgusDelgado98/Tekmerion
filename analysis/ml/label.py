"""Human labeling helpers. Never calls role-family classifiers or ML models."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from analysis.ml.gate import sufficiency_report
from analysis.ml.gold import dump_gold_dataset, load_gold_dataset
from analysis.ml.harvest import CANDIDATE_SCHEMA
from analysis.ml.models import (
    FORBIDDEN_GOLD_KEYS,
    GOLD_SCHEMA,
    LABEL_SOURCE_HUMAN,
    MIN_EXAMPLES_PER_CLASS_FOR_TRAINING,
    MIN_N_FOR_TRAINING,
    GoldDataset,
    RoleFamily,
    RoleFamilyExample,
)
from analysis.ml.split import example_fingerprint
from analysis.skills import extract_skills

LABELS_SCHEMA = "tekmerion.ml.human_labels.v1"
SESSION_SCHEMA = "tekmerion.ml.label_session.v1"
REAL_SOURCE_KINDS = frozenset({"curated_real_sample", "adzuna_snapshot"})
DEFAULT_SESSION_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "ml" / "gold" / "label_session.json"
)
ALLOWED_LABELS = frozenset(
    m.value for m in RoleFamily if m not in {RoleFamily.UNKNOWN}
)
SKIP_DECISIONS = frozenset({"skip", "ambiguous", "skip/ambiguous"})

# Queue hints only — never written as gold_role_family.
_SPARSE_TITLE_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ai_analyst", ("ai analyst", "ai engineer", "prompt engineer", "llm", "generative ai", "analista de ia")),
    (
        "business_analyst",
        ("business analyst", "analista de negocios", "analista funcional"),
    ),
    ("bi_analyst", ("bi analyst", "business intelligence", "bi developer", "power bi")),
    ("data_scientist", ("data scientist", "research scientist")),
    ("ml_engineer", ("ml engineer", "machine learning engineer", "mlops")),
    ("data_engineer", ("data engineer", "etl engineer")),
    ("data_analyst", ("data analyst", "analista de datos", "product analyst")),
)


class LabelError(ValueError):
    """Invalid human label payload."""


def adzuna_query_context(source_ref: str) -> str:
    """Parse search query from snapshot filename. Context only — never a label."""
    name = Path(str(source_ref or "")).name
    if not name.startswith("adzuna_"):
        return ""
    stem = name.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) < 5:
        return ""
    return " ".join(parts[3:-1]).replace("-", " ")


def load_label_session(path: str | Path | None = None) -> dict[str, Any]:
    file_path = Path(path) if path is not None else DEFAULT_SESSION_PATH
    if not file_path.exists():
        return {
            "schema": SESSION_SCHEMA,
            "annotator_id": "human.v0.8.b1",
            "decisions": [],
        }
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LabelError("session file must be an object")
    if payload.get("schema") not in (None, SESSION_SCHEMA):
        raise LabelError(f"unexpected session schema {payload.get('schema')!r}")
    decisions = payload.get("decisions") or []
    if not isinstance(decisions, list):
        raise LabelError("session.decisions must be a list")
    return {
        "schema": SESSION_SCHEMA,
        "annotator_id": str(payload.get("annotator_id") or "human.v0.8.b1"),
        "decisions": decisions,
    }


def save_label_session(session: dict[str, Any], path: str | Path | None = None) -> Path:
    file_path = Path(path) if path is not None else DEFAULT_SESSION_PATH
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SESSION_SCHEMA,
        "annotator_id": session.get("annotator_id") or "human.v0.8.b1",
        "decisions": session.get("decisions") or [],
    }
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return file_path


def session_reviewed_ids(session: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for item in session.get("decisions") or []:
        if isinstance(item, dict) and item.get("id"):
            ids.add(str(item["id"]))
    return ids


def session_stats(session: dict[str, Any]) -> dict[str, int]:
    n_label = 0
    n_skip = 0
    for item in session.get("decisions") or []:
        if not isinstance(item, dict):
            continue
        decision = str(item.get("decision") or "").strip().lower()
        if decision in SKIP_DECISIONS:
            n_skip += 1
        elif decision == "label" or item.get("gold_role_family"):
            n_label += 1
    return {
        "n_reviewed": n_label + n_skip,
        "n_labeled": n_label,
        "n_skipped": n_skip,
    }


def record_session_decision(
    session: dict[str, Any],
    *,
    candidate_id: str,
    decision: str,
    gold_role_family: str = "",
    notes: str = "",
    labeled_at: str,
) -> dict[str, Any]:
    cid = candidate_id.strip()
    if not cid:
        raise LabelError("candidate id is required")
    choice = decision.strip().lower()
    reviewed = session_reviewed_ids(session)
    if cid in reviewed:
        raise LabelError(f"already reviewed in session: {cid}")
    row: dict[str, Any] = {
        "id": cid,
        "decision": choice,
        "labeled_at": labeled_at,
        "notes": notes,
    }
    if choice in SKIP_DECISIONS:
        row["decision"] = "skip"
        row["reason"] = "ambiguous"
    elif choice in ALLOWED_LABELS or choice == "label":
        family = (gold_role_family or choice).strip()
        if family == "label":
            raise LabelError("gold_role_family is required when decision=label")
        if family not in ALLOWED_LABELS:
            raise LabelError(f"unsupported label {family!r}; use a role family or skip")
        row["decision"] = "label"
        row["gold_role_family"] = family
        row["label_source"] = LABEL_SOURCE_HUMAN
    else:
        raise LabelError("decision must be a role family or skip/ambiguous")
    leak = FORBIDDEN_GOLD_KEYS.intersection(row.keys())
    if leak:
        raise LabelError(f"session decision carries classifier fields: {sorted(leak)}")
    session.setdefault("decisions", []).append(row)
    return row


def labels_from_session(session: dict[str, Any]) -> list[dict[str, Any]]:
    annotator = str(session.get("annotator_id") or "human.v0.8.b1")
    out: list[dict[str, Any]] = []
    for item in session.get("decisions") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("decision") or "") != "label":
            continue
        out.append(
            {
                "id": item["id"],
                "gold_role_family": item["gold_role_family"],
                "annotator_id": annotator,
                "labeled_at": item.get("labeled_at") or "",
                "label_source": LABEL_SOURCE_HUMAN,
                "notes": item.get("notes") or "Human label from title+description; query is context only.",
            }
        )
    return out


def summarize_description(text: str, *, max_chars: int = 420) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


def gold_fingerprints(dataset: GoldDataset) -> set[str]:
    return {ex.content_fingerprint for ex in dataset.records}


def gold_ids(dataset: GoldDataset) -> set[str]:
    return {ex.id for ex in dataset.records}


def _hint_family(title: str, description: str) -> str:
    blob = f"{title} {description}".lower()
    for family, needles in _SPARSE_TITLE_HINTS:
        for needle in needles:
            if needle in blob:
                return family
    return ""


def unlabeled_queue(
    candidates: Sequence[dict[str, Any]],
    dataset: GoldDataset,
    *,
    real_only: bool = True,
    extra_skip_ids: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """Candidates not already in gold, ordered to help fill sparse classes.

    Review-order hints are not labels.
    """
    dist = Counter(ex.gold_role_family.value for ex in dataset.records)
    labeled_fp = gold_fingerprints(dataset)
    labeled_ids = gold_ids(dataset)
    skip_extra = extra_skip_ids or set()
    pending: list[dict[str, Any]] = []
    for raw in candidates:
        if raw.get("gold_role_family") is not None:
            raise LabelError("candidate payload must not include gold_role_family")
        leak = FORBIDDEN_GOLD_KEYS.intersection(raw.keys())
        if leak:
            raise LabelError(f"candidate carries forbidden fields: {sorted(leak)}")
        fp = str(raw.get("content_fingerprint") or "") or example_fingerprint(
            str(raw.get("title") or ""),
            str(raw.get("description") or ""),
        )
        cid = str(raw.get("id") or "")
        if fp in labeled_fp or cid in labeled_ids or cid in skip_extra:
            continue
        kind = str(raw.get("source_kind") or "")
        if real_only and kind not in REAL_SOURCE_KINDS:
            continue
        pending.append(dict(raw, content_fingerprint=fp))

    def sort_key(row: dict[str, Any]) -> tuple:
        hint = _hint_family(str(row.get("title") or ""), str(row.get("description") or ""))
        have = dist.get(hint, 0) if hint else 10**6
        # Fewer labeled examples of the hinted family → earlier in the queue.
        return (have, hint or "zzzz", str(row.get("id") or ""))

    pending.sort(key=sort_key)
    return pending


def format_label_card(
    candidate: dict[str, Any],
    *,
    distribution: dict[str, int],
    index: int,
    total: int,
) -> str:
    title = str(candidate.get("title") or "")
    desc = summarize_description(str(candidate.get("description") or ""))
    skills = extract_skills(f"{title} {candidate.get('description') or ''}")
    query_ctx = adzuna_query_context(str(candidate.get("source_ref") or ""))
    families = ", ".join(sorted(ALLOWED_LABELS))
    dist_txt = ", ".join(f"{k}={v}" for k, v in sorted(distribution.items())) or "(vacío)"
    lines = [
        f"--- gold label {index}/{total} ---",
        f"id:           {candidate.get('id')}",
        f"source_kind:  {candidate.get('source_kind')}",
        f"query_ctx:    {query_ctx or '(n/a)'}  [context only — not a label]",
        f"company:      {candidate.get('company')}",
        f"location:     {candidate.get('location')}",
        f"source_url:   {candidate.get('source_url')}",
        f"retrieved_at: {candidate.get('retrieved_at')}",
        f"title:        {title}",
        f"skills:       {', '.join(skills) if skills else '(ninguna detectada)'}",
        f"gold dist:    {dist_txt}",
        f"choices:      {families} | skip/ambiguous",
        "description:",
        f"  {desc}",
        "Enter a family or skip. Do not paste classifier output. Query is not the label.",
    ]
    return "\n".join(lines)


def load_human_labels(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
        annotator = ""
        labeled_at = ""
    elif isinstance(payload, dict):
        if payload.get("schema") not in (None, LABELS_SCHEMA):
            raise LabelError(f"unexpected labels schema {payload.get('schema')!r}")
        records = payload.get("records") or []
        annotator = str(payload.get("annotator_id") or "")
        labeled_at = str(payload.get("labeled_at") or "")
    else:
        raise LabelError("labels file must be a list or object")
    if not isinstance(records, list):
        raise LabelError("labels records must be a list")
    out: list[dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            raise LabelError("each label must be an object")
        leak = FORBIDDEN_GOLD_KEYS.intersection(item.keys())
        if leak:
            raise LabelError(f"label row carries classifier fields: {sorted(leak)}")
        if item.get("label_source") not in (None, "", LABEL_SOURCE_HUMAN):
            raise LabelError("label_source must be human")
        row = dict(item)
        if annotator and not row.get("annotator_id"):
            row["annotator_id"] = annotator
        if labeled_at and not row.get("labeled_at"):
            row["labeled_at"] = labeled_at
        out.append(row)
    return out


def _example_from_candidate(cand: dict[str, Any], label: dict[str, Any]) -> RoleFamilyExample:
    family_raw = str(label.get("gold_role_family") or "").strip()
    try:
        family = RoleFamily(family_raw)
    except ValueError as exc:
        raise LabelError(f"unknown gold_role_family {family_raw!r}") from exc
    annotator = str(label.get("annotator_id") or "").strip()
    labeled_at = str(label.get("labeled_at") or "").strip()
    if not annotator or not labeled_at:
        raise LabelError("annotator_id and labeled_at are required")
    title = str(cand.get("title") or "")
    description = str(cand.get("description") or "")
    return RoleFamilyExample(
        id=str(cand.get("id") or "").strip(),
        title=title,
        description=description,
        gold_role_family=family,
        label_source=LABEL_SOURCE_HUMAN,
        annotator_id=annotator,
        labeled_at=labeled_at,
        content_fingerprint=example_fingerprint(title, description),
        company=str(cand.get("company") or ""),
        source_kind=str(cand.get("source_kind") or ""),
        source_ref=str(cand.get("source_ref") or ""),
        notes=str(label.get("notes") or "Human label from title+description; not from regex."),
        location=str(cand.get("location") or ""),
        source_url=str(cand.get("source_url") or ""),
        retrieved_at=str(cand.get("retrieved_at") or ""),
        source_record_id=str(cand.get("source_record_id") or ""),
    )


def apply_human_labels(
    dataset: GoldDataset,
    candidates: Sequence[dict[str, Any]],
    labels: Sequence[dict[str, Any]],
) -> tuple[GoldDataset, dict[str, Any]]:
    by_id = {str(c.get("id") or ""): c for c in candidates}
    existing_fp = gold_fingerprints(dataset)
    existing_ids = gold_ids(dataset)
    added: list[RoleFamilyExample] = []
    skipped_dup = 0
    skipped_unknown = 0
    for lab in labels:
        leak = FORBIDDEN_GOLD_KEYS.intersection(lab.keys())
        if leak:
            raise LabelError(f"label row carries classifier fields: {sorted(leak)}")
        cid = str(lab.get("id") or "").strip()
        cand = by_id.get(cid)
        if cand is None:
            skipped_unknown += 1
            continue
        example = _example_from_candidate(cand, lab)
        if example.id in existing_ids or example.content_fingerprint in existing_fp:
            skipped_dup += 1
            continue
        added.append(example)
        existing_ids.add(example.id)
        existing_fp.add(example.content_fingerprint)
    merged = GoldDataset(
        schema=dataset.schema or GOLD_SCHEMA,
        task=dataset.task,
        label_field=dataset.label_field,
        label_source=LABEL_SOURCE_HUMAN,
        label_policy=dataset.label_policy,
        limitations=dataset.limitations,
        records=tuple(list(dataset.records) + added),
        path=dataset.path,
        extra_meta=dataset.extra_meta,
    )
    stats = {
        "n_labels_in": len(labels),
        "n_added": len(added),
        "n_skipped_duplicate": skipped_dup,
        "n_skipped_unknown_id": skipped_unknown,
        "n_gold_after": merged.n,
    }
    return merged, stats


def expansion_report(
    dataset: GoldDataset,
    *,
    harvest: Optional[dict[str, Any]] = None,
    fetch: Optional[dict[str, Any]] = None,
    queue_n: int = 0,
) -> dict[str, Any]:
    gate = sufficiency_report(dataset)
    unique_real = 0
    if harvest:
        unique_real = sum(
            1
            for r in harvest.get("records") or []
            if r.get("source_kind") in REAL_SOURCE_KINDS
        )
    return {
        "schema": "tekmerion.ml.gold_expansion.v1",
        "gate": gate,
        "gold_n": dataset.n,
        "class_distribution": gate["class_distribution"],
        "class_gaps": gate["class_gaps"],
        "n_gap_to_min_n": gate["n_gap"],
        "min_n_for_training": MIN_N_FOR_TRAINING,
        "min_examples_per_class": MIN_EXAMPLES_PER_CLASS_FOR_TRAINING,
        "unlabeled_queue_real": queue_n,
        "harvest": None
        if harvest is None
        else {
            "n_loaded": harvest.get("n_loaded"),
            "n_unique": harvest.get("n_unique"),
            "n_dropped_duplicates": harvest.get("n_dropped_duplicates"),
            "unique_by_source_kind": harvest.get("unique_by_source_kind"),
            "sources_used": harvest.get("sources_used"),
            "n_unique_real_kinds": unique_real,
        },
        "live_fetch": fetch,
        "training_allowed": bool(gate["sufficient_for_training"]),
        "invented_records": False,
    }


def load_candidates_file(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LabelError("candidates file must be an object")
    if payload.get("schema") not in (None, CANDIDATE_SCHEMA):
        raise LabelError("unexpected candidates schema")
    records = payload.get("records") or []
    if not isinstance(records, list):
        raise LabelError("candidates.records must be a list")
    return payload
