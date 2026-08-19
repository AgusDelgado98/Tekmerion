"""Load and validate a human-labeled Gold Dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from analysis.ml.models import (
    FORBIDDEN_GOLD_KEYS,
    GOLD_SCHEMA,
    LABEL_SOURCE_HUMAN,
    GoldDataset,
    RoleFamilyExample,
)
from analysis.ml.split import example_fingerprint
from analysis.models import RoleFamily

DEFAULT_GOLD_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "ml" / "gold" / "role_family_v1.json"
)
_REPO_ROOT = Path(__file__).resolve().parents[2]


class GoldDatasetError(ValueError):
    """Invalid gold dataset contract."""


def repo_relative_path(path: str | Path | None) -> str:
    """Store paths relative to the repo root (no local drive prefixes)."""
    if path is None:
        return ""
    raw = str(path).replace("\\", "/")
    if raw.startswith(("data/", "tests/", "docs/", "analysis/", "app/", "scripts/")):
        return raw
    try:
        rel = Path(path).resolve().relative_to(_REPO_ROOT.resolve())
        return str(rel).replace("\\", "/")
    except (OSError, ValueError):
        for marker in ("/data/", "/tests/", "/docs/"):
            idx = raw.find(marker)
            if idx != -1:
                return raw[idx + 1 :]
        return Path(raw).name


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise GoldDatasetError(message)


def dataset_sha256(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def _parse_role_family(value: Any, record_id: str) -> RoleFamily:
    _require(isinstance(value, str) and value.strip(), f"{record_id}: gold_role_family required")
    try:
        return RoleFamily(value.strip())
    except ValueError as exc:
        raise GoldDatasetError(
            f"{record_id}: unknown gold_role_family {value!r}"
        ) from exc


def _example_from_record(raw: dict[str, Any]) -> RoleFamilyExample:
    overlap = FORBIDDEN_GOLD_KEYS.intersection(raw.keys())
    _require(
        not overlap,
        "gold records must not carry classifier/prediction fields: "
        + ", ".join(sorted(overlap)),
    )
    record_id = str(raw.get("id") or "").strip()
    _require(bool(record_id), "record id is required")
    title = str(raw.get("title") or "").strip()
    description = str(raw.get("description") or "").strip()
    _require(bool(title), f"{record_id}: title is required")
    _require(bool(description), f"{record_id}: description is required")

    label_source = str(raw.get("label_source") or LABEL_SOURCE_HUMAN).strip()
    _require(
        label_source == LABEL_SOURCE_HUMAN,
        f"{record_id}: label_source must be '{LABEL_SOURCE_HUMAN}', got {label_source!r}",
    )
    annotator = str(raw.get("annotator_id") or "").strip()
    _require(bool(annotator), f"{record_id}: annotator_id is required")
    labeled_at = str(raw.get("labeled_at") or "").strip()
    _require(bool(labeled_at), f"{record_id}: labeled_at is required")

    gold = _parse_role_family(raw.get("gold_role_family"), record_id)
    return RoleFamilyExample(
        id=record_id,
        title=title,
        description=description,
        gold_role_family=gold,
        label_source=label_source,
        annotator_id=annotator,
        labeled_at=labeled_at,
        content_fingerprint=example_fingerprint(title, description),
        company=str(raw.get("company") or "").strip(),
        source_kind=str(raw.get("source_kind") or "synthetic").strip(),
        source_ref=str(raw.get("source_ref") or "").strip(),
        notes=str(raw.get("notes") or "").strip(),
        location=str(raw.get("location") or "").strip(),
        source_url=str(raw.get("source_url") or "").strip(),
        retrieved_at=str(raw.get("retrieved_at") or "").strip(),
        source_record_id=str(raw.get("source_record_id") or "").strip(),
    )


def load_gold_dataset(path: str | Path | None = None) -> GoldDataset:
    file_path = Path(path) if path is not None else DEFAULT_GOLD_PATH
    if not file_path.exists():
        raise FileNotFoundError(f"Gold dataset not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    _require(isinstance(payload, dict), "gold dataset root must be an object")
    schema = str(payload.get("schema") or "")
    _require(schema == GOLD_SCHEMA, f"expected schema {GOLD_SCHEMA}, got {schema!r}")
    label_source = str(payload.get("label_source") or "")
    _require(
        label_source == LABEL_SOURCE_HUMAN,
        f"dataset label_source must be '{LABEL_SOURCE_HUMAN}'",
    )
    label_field = str(payload.get("label_field") or "")
    _require(
        label_field == "gold_role_family",
        "label_field must be 'gold_role_family' (not pipeline role_family)",
    )
    raw_records = payload.get("records")
    _require(isinstance(raw_records, list), "records must be a list")

    examples: list[RoleFamilyExample] = []
    seen_ids: set[str] = set()
    for item in raw_records:
        _require(isinstance(item, dict), "each record must be an object")
        example = _example_from_record(item)
        _require(example.id not in seen_ids, f"duplicate gold id: {example.id}")
        seen_ids.add(example.id)
        examples.append(example)

    limitations = payload.get("limitations") or []
    _require(isinstance(limitations, list), "limitations must be a list")
    _require(
        all(isinstance(x, str) for x in limitations),
        "limitations must be strings",
    )

    extra: list[tuple[str, str]] = []
    for key in ("created_for", "corpus_note"):
        if key in payload and payload[key] is not None:
            extra.append((key, str(payload[key])))

    return GoldDataset(
        schema=schema,
        task=str(payload.get("task") or "role_family_classification"),
        label_field=label_field,
        label_source=label_source,
        label_policy=str(payload.get("label_policy") or ""),
        limitations=tuple(limitations),
        records=tuple(examples),
        path=str(file_path).replace("\\", "/"),
        extra_meta=tuple(extra),
    )


def dump_gold_dataset(dataset: GoldDataset, path: str | Path | None = None) -> Path:
    """Write a gold JSON. Does not add classifier fields."""
    file_path = Path(path) if path is not None else (
        Path(dataset.path) if dataset.path else DEFAULT_GOLD_PATH
    )
    extra = dict(dataset.extra_meta)
    payload: dict[str, Any] = {
        "schema": dataset.schema,
        "task": dataset.task,
        "label_field": dataset.label_field,
        "label_source": dataset.label_source,
        "label_policy": dataset.label_policy,
        "created_for": extra.get("created_for", ""),
        "corpus_note": extra.get("corpus_note", ""),
        "limitations": list(dataset.limitations),
        "records": [
            {**r.to_dict(), "source_ref": repo_relative_path(r.source_ref)}
            for r in sorted(dataset.records, key=lambda e: e.id)
        ],
    }
    file_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    file_path.write_text(text, encoding="utf-8")
    return file_path
