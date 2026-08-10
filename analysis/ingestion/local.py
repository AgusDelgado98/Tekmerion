"""
Local file source adapter.

Loads JSON lists from disk. Suitable for:
- the controlled real sample under data/raw/real/
- any future offline dumps produced by other adapters
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from analysis.ingestion.base import SourceAdapter


class LocalJsonSource(SourceAdapter):
    """
    Adapter that reads a JSON file containing a list of job records.

    The file is expected to be UTF-8 JSON array of objects.
    No network calls; purely offline.
    """

    def __init__(
        self,
        path: str | Path,
        source_name: Optional[str] = None,
        description: str = "",
    ) -> None:
        self.path = Path(path)
        self._source_name = source_name or self.path.stem
        self._description = description or f"Local JSON source: {self.path.name}"

    def source_name(self) -> str:
        return self._source_name

    def describe(self) -> str:
        return self._description

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            raise FileNotFoundError(f"Local source file not found: {self.path}")

        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON list in {self.path}, got {type(data).__name__}"
            )

        # Return shallow copies so downstream cannot mutate the loaded objects
        # through shared references (defensive).
        return [dict(item) if isinstance(item, dict) else item for item in data]
