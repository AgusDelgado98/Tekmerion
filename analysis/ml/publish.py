"""Sanitize ML evaluation artifacts for public redistribution.

Drops vacancy text and hashes example ids. Metrics, matrices, CV and
hyperparameters are kept unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

TEXT_KEYS = frozenset(
    {
        "title",
        "description",
        "company",
        "location",
        "source_url",
        "source_ref",
        "source_record_id",
        "notes",
        "retrieved_at",
        "credentials",
        "app_id",
        "api_key",
    }
)
ID_LIST_KEYS = frozenset({"train_ids", "test_ids", "fitted_on_ids"})
ID_SCALAR_KEYS = frozenset({"id", "example_id"})
_PUBLIC_ID = re.compile(r"^ex_[0-9a-f]{12}$")


def public_example_id(raw: str) -> str:
    """Stable anonymous id; not reversible to Adzuna or employer keys."""
    text = str(raw)
    if _PUBLIC_ID.fullmatch(text):
        return text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"ex_{digest}"


def sanitize_public_payload(value: Any) -> Any:
    if isinstance(value, list):
        return [sanitize_public_payload(item) for item in value]
    if not isinstance(value, dict):
        return value
    out: dict[str, Any] = {}
    for key, item in value.items():
        if key in TEXT_KEYS:
            continue
        if key in ID_LIST_KEYS and isinstance(item, list):
            out[key] = [public_example_id(str(x)) for x in item]
            continue
        if key in ID_SCALAR_KEYS and isinstance(item, str):
            out[key] = public_example_id(item)
            continue
        out[key] = sanitize_public_payload(item)
    return out


def write_sanitized_json(payload: dict[str, Any], path: str | Path) -> None:
    from analysis.ml.evaluate import dump_canonical_json

    dump_canonical_json(sanitize_public_payload(payload), path)


def sanitize_json_file(path: str | Path) -> None:
    file_path = Path(path)
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{file_path} must be a JSON object")
    write_sanitized_json(payload, file_path)
