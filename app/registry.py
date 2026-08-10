"""
Dataset registry for Flask (V0.5.3+ / V0.6.0 showroom).

Known local datasets only:
  - synthetic (always)
  - showroom demo artifact under data/showroom/
  - valid market artifacts under data/processed/market/

Never accepts paths, URLs, or uploads from the client.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.dataset import (
    MODE_MARKET,
    MODE_SYNTHETIC,
    MARKET_DIR,
    ROOT,
    AppDataset,
    DatasetError,
    DatasetMeta,
    load_market_artifact_file,
    load_market_dataset,
    load_synthetic_dataset,
)


SESSION_KEY = "active_dataset_id"
SYNTHETIC_ID = "synthetic"
SHOWROOM_ID = "showroom"
SHOWROOM_DIR = ROOT / "data" / "showroom"
SHOWROOM_FILE = SHOWROOM_DIR / "showroom_market_ar.json"

KIND_SYNTHETIC = "synthetic"
KIND_SHOWROOM = "showroom"
KIND_MARKET = "market"


@dataclass(frozen=True)
class DatasetEntry:
    """Public metadata for a selectable dataset (no filesystem paths)."""

    id: str
    mode: str
    label: str
    source: str
    country: Optional[str] = None
    retrieved_at: Optional[str] = None
    total_records: Optional[int] = None
    artifact_name: Optional[str] = None
    dataset_kind: str = KIND_MARKET
    is_showroom: bool = False

    def option_label(self) -> str:
        if self.dataset_kind == KIND_SYNTHETIC or self.mode == MODE_SYNTHETIC:
            n = self.total_records
            return f"Synthetic sample" + (f" · {n} records" if n is not None else "")
        if self.is_showroom or self.dataset_kind == KIND_SHOWROOM:
            n = self.total_records
            parts = ["Showroom · Market demo"]
            if n is not None:
                parts.append(f"{n} jobs")
            return " · ".join(parts)
        parts = ["Market snapshot"]
        if self.country:
            parts.append(self.country.upper())
        if self.total_records is not None:
            parts.append(f"{self.total_records} jobs")
        if self.retrieved_at:
            parts.append(self.retrieved_at[:10])
        return " · ".join(parts)


def _safe_ts(retrieved_at: Optional[str]) -> str:
    if not retrieved_at:
        return "unknown"
    return re.sub(r"[^0-9A-Za-z.\-]", "-", str(retrieved_at))


def make_market_dataset_id(
    *,
    country: Optional[str],
    retrieved_at: Optional[str],
    artifact_name: str,
) -> str:
    c = (country or "xx").lower()
    ts = _safe_ts(retrieved_at)
    stem = Path(artifact_name).stem
    return f"market:{c}:{ts}:{stem}"


class DatasetRegistry:
    def __init__(
        self,
        entries: list[DatasetEntry],
        *,
        market_paths: Optional[dict[str, Path]] = None,
        default_id: str = SYNTHETIC_ID,
    ) -> None:
        self._entries = list(entries)
        self._by_id = {e.id: e for e in self._entries}
        self._market_paths = dict(market_paths or {})
        self._cache: dict[str, AppDataset] = {}
        if default_id not in self._by_id:
            default_id = SYNTHETIC_ID
        self.default_id = default_id

    def list_entries(self) -> list[DatasetEntry]:
        return list(self._entries)

    def get_entry(self, dataset_id: str) -> Optional[DatasetEntry]:
        return self._by_id.get(dataset_id)

    def is_known(self, dataset_id: str) -> bool:
        return dataset_id in self._by_id

    def resolve(self, dataset_id: str) -> AppDataset:
        if dataset_id not in self._by_id:
            raise DatasetError(f"Unknown dataset id: {dataset_id!r}")
        if dataset_id in self._cache:
            return self._cache[dataset_id]

        entry = self._by_id[dataset_id]
        if entry.mode == MODE_SYNTHETIC:
            ds = load_synthetic_dataset()
        elif entry.mode == MODE_MARKET:
            path = self._market_paths.get(dataset_id)
            if path is None:
                raise DatasetError(f"No path registered for dataset {dataset_id!r}")
            ds = load_market_dataset(path)
            # Enrich meta label for showroom without mutating frozen entry
            if entry.is_showroom:
                from dataclasses import replace
                meta = replace(
                    ds.meta,
                    label="Showroom · Market demo",
                    source=ds.meta.source or "showroom",
                )
                ds = AppDataset(
                    pipeline_result=ds.pipeline_result,
                    evidence=ds.evidence,
                    meta=meta,
                )
        else:
            raise DatasetError(f"Unsupported dataset mode: {entry.mode}")

        self._cache[dataset_id] = ds
        return ds


def discover_market_entries(
    directory: Path = MARKET_DIR,
) -> tuple[list[DatasetEntry], dict[str, Path]]:
    entries: list[DatasetEntry] = []
    paths: dict[str, Path] = {}
    if not directory.exists() or not directory.is_dir():
        return entries, paths

    candidates: list[tuple[str, str, Path, dict]] = []
    for path in directory.glob("*.json"):
        try:
            data = load_market_artifact_file(path)
        except DatasetError:
            continue
        ts = str(data.get("retrieved_at") or "")
        candidates.append((ts, path.name, path, data))

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

    for ts, name, path, data in candidates:
        country = data.get("country")
        retrieved_at = data.get("retrieved_at")
        records = data.get("records") or []
        ds_id = make_market_dataset_id(
            country=country if isinstance(country, str) else None,
            retrieved_at=retrieved_at if isinstance(retrieved_at, str) else None,
            artifact_name=name,
        )
        if ds_id in paths:
            ds_id = f"{ds_id}-{abs(hash(name)) % 10000}"
        entry = DatasetEntry(
            id=ds_id,
            mode=MODE_MARKET,
            label="Market snapshot",
            source=str(data.get("source") or "adzuna"),
            country=country if isinstance(country, str) else None,
            retrieved_at=retrieved_at if isinstance(retrieved_at, str) else None,
            total_records=len(records) if isinstance(records, list) else 0,
            artifact_name=name,
            dataset_kind=KIND_MARKET,
            is_showroom=False,
        )
        entries.append(entry)
        paths[ds_id] = path

    return entries, paths


def load_showroom_entry(
    showroom_file: Path = SHOWROOM_FILE,
) -> tuple[Optional[DatasetEntry], Optional[Path]]:
    if not showroom_file.exists():
        return None, None
    try:
        data = load_market_artifact_file(showroom_file)
    except DatasetError:
        return None, None
    records = data.get("records") or []
    entry = DatasetEntry(
        id=SHOWROOM_ID,
        mode=MODE_MARKET,
        label="Showroom · Market demo",
        source=str(data.get("source") or "showroom"),
        country=data.get("country") if isinstance(data.get("country"), str) else None,
        retrieved_at=data.get("retrieved_at") if isinstance(data.get("retrieved_at"), str) else None,
        total_records=len(records) if isinstance(records, list) else 0,
        artifact_name=showroom_file.name,
        dataset_kind=KIND_SHOWROOM,
        is_showroom=True,
    )
    return entry, showroom_file


def build_registry(
    *,
    market_dir: Path = MARKET_DIR,
    default_mode: str = MODE_SYNTHETIC,
    default_market_file: Optional[Path] = None,
    showroom_file: Optional[Path] = None,
) -> DatasetRegistry:
    """
    Order: synthetic → showroom → other market artifacts (newest first).
    """
    market_entries, market_paths = discover_market_entries(market_dir)

    try:
        syn = load_synthetic_dataset()
        syn_count = syn.meta.total_records
    except Exception:
        syn_count = None

    entries: list[DatasetEntry] = [
        DatasetEntry(
            id=SYNTHETIC_ID,
            mode=MODE_SYNTHETIC,
            label="Synthetic sample",
            source="synthetic",
            total_records=syn_count,
            dataset_kind=KIND_SYNTHETIC,
            is_showroom=False,
        )
    ]

    show_path = showroom_file if showroom_file is not None else SHOWROOM_FILE
    show_entry, show_file = load_showroom_entry(show_path)
    if show_entry and show_file is not None:
        entries.append(show_entry)
        market_paths[SHOWROOM_ID] = show_file

    entries.extend(market_entries)

    default_id = SYNTHETIC_ID
    if default_mode == "showroom" and show_entry:
        default_id = SHOWROOM_ID
    elif default_mode == MODE_MARKET:
        if default_market_file is not None:
            target = Path(default_market_file).name
            for e in market_entries:
                if e.artifact_name == target:
                    default_id = e.id
                    break
            else:
                try:
                    data = load_market_artifact_file(Path(default_market_file))
                    name = Path(default_market_file).name
                    mid = make_market_dataset_id(
                        country=data.get("country") if isinstance(data.get("country"), str) else None,
                        retrieved_at=data.get("retrieved_at")
                        if isinstance(data.get("retrieved_at"), str)
                        else None,
                        artifact_name=name,
                    )
                    entry = DatasetEntry(
                        id=mid,
                        mode=MODE_MARKET,
                        label="Market snapshot",
                        source=str(data.get("source") or "adzuna"),
                        country=data.get("country") if isinstance(data.get("country"), str) else None,
                        retrieved_at=data.get("retrieved_at")
                        if isinstance(data.get("retrieved_at"), str)
                        else None,
                        total_records=len(data.get("records") or []),
                        artifact_name=name,
                        dataset_kind=KIND_MARKET,
                    )
                    entries.append(entry)
                    market_paths[mid] = Path(default_market_file)
                    default_id = mid
                except DatasetError:
                    default_id = market_entries[0].id if market_entries else (
                        SHOWROOM_ID if show_entry else SYNTHETIC_ID
                    )
        elif market_entries:
            default_id = market_entries[0].id
        elif show_entry:
            default_id = SHOWROOM_ID

    return DatasetRegistry(entries, market_paths=market_paths, default_id=default_id)
