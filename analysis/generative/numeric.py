"""
Numeric claim extraction and NumericEvidenceIndex (V0.5.1).

Conservative, deterministic. No NLP, no embeddings, no second LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from analysis.grounding import GroundingPayload


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NumericValue:
    value: float
    unit: str  # "count" | "percent"
    refs: tuple[str, ...]


@dataclass
class NumericEvidenceIndex:
    """Lookup table derived from GroundingPayload."""

    by_ref: dict[str, list[NumericValue]] = field(default_factory=dict)
    percent_values: dict[float, set[str]] = field(default_factory=dict)
    count_values: dict[float, set[str]] = field(default_factory=dict)
    dataset_size: Optional[float] = None
    dataset_size_refs: tuple[str, ...] = ()

    def values_for_refs(self, refs: Iterable[str], *, unit: str) -> set[float]:
        out: set[float] = set()
        for ref in refs:
            for nv in self.by_ref.get(ref, []):
                if nv.unit == unit:
                    out.add(nv.value)
        return out

    def global_has(self, value: float, unit: str) -> bool:
        target = _normalize(value, unit)
        table = self.percent_values if unit == "percent" else self.count_values
        return target in table

    def supports_with_refs(self, value: float, unit: str, refs: list[str]) -> bool:
        """
        Finding-level check:
        1. value present among numeric values of the cited refs (same unit)
        2. OR value equals dataset size (denominators / sample size)
        """
        target = _normalize(value, unit)
        local = self.values_for_refs(refs, unit=unit)
        if target in local:
            return True
        if unit == "count" and self.dataset_size is not None and target == self.dataset_size:
            return True
        return False

    def supports_global(self, value: float, unit: str) -> bool:
        return self.global_has(value, unit)


def build_numeric_index(grounding: GroundingPayload) -> NumericEvidenceIndex:
    idx = NumericEvidenceIndex()
    by_ref: dict[str, list[NumericValue]] = {}
    pct: dict[float, set[str]] = {}
    cnt: dict[float, set[str]] = {}

    def _add(ref: str, value: Any, unit: str) -> None:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return
        num = _normalize(num, unit)
        by_ref.setdefault(ref, []).append(NumericValue(value=num, unit=unit, refs=(ref,)))
        table = pct if unit == "percent" else cnt
        table.setdefault(num, set()).add(ref)

    for item in grounding.items:
        if item.unit == "percent":
            _add(item.id, item.value, "percent")
        elif item.unit == "count":
            _add(item.id, item.value, "count")
        elif item.unit == "list" and isinstance(item.value, list):
            for entry in item.value:
                if not isinstance(entry, dict):
                    continue
                if "pct" in entry:
                    _add(item.id, entry["pct"], "percent")
                if "count" in entry:
                    _add(item.id, entry["count"], "count")

        if item.id == "dataset.n_analysis_records":
            try:
                idx.dataset_size = _normalize(float(item.value), "count")
                idx.dataset_size_refs = (item.id,)
            except (TypeError, ValueError):
                pass

    idx.by_ref = by_ref
    idx.percent_values = pct
    idx.count_values = cnt
    return idx


def _normalize(value: float, unit: str) -> float:
    if unit == "percent":
        return round(float(value), 1)
    v = float(value)
    return float(int(v)) if v.is_integer() else v


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NumericClaim:
    value: float
    unit: str
    raw: str
    kind: str  # percent | ratio_part | ratio_whole | count
    span: tuple[int, int] = (0, 0)


_IGNORE_PATTERNS = [
    re.compile(r"\b20\d{2}\b"),
    re.compile(
        r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)?"
    ),
    re.compile(r"\bgpt-[\w.\-]+", re.I),
    re.compile(r"\bmarket_summary\.v\d+", re.I),
    re.compile(r"\bv\d+\.\d+(?:\.\d+)?\b"),
    re.compile(r"\b\d{5,}\b"),
]


def _ignored_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for pat in _IGNORE_PATTERNS for m in pat.finditer(text)]


def _in_ignored(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in spans)


_PCT_RE = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*%")
_RATIO_RE = re.compile(r"(?<![\w.])(\d+)\s*(?:de|of|/)\s*(\d+)(?!\s*%)", re.I)
_COUNT_NOUN_RE = re.compile(
    r"(?<![\w.])(\d+)\s+"
    r"(?:vacantes?|registros?|records?|roles?|skills?|queries?|familias?|"
    r"menciones?|avisos?|jobs?|posiciones?|resultados?)",
    re.I,
)


def extract_numeric_claims(text: str) -> list[NumericClaim]:
    if not text:
        return []
    ignored = _ignored_spans(text)
    claims: list[NumericClaim] = []
    occupied: list[tuple[int, int]] = []

    def _take(span: tuple[int, int]) -> bool:
        if _in_ignored(span[0], ignored):
            return False
        if any(not (span[1] <= a or span[0] >= b) for a, b in occupied):
            return False
        occupied.append(span)
        return True

    for m in _RATIO_RE.finditer(text):
        if not _take(m.span()):
            continue
        a, b = int(m.group(1)), int(m.group(2))
        claims.append(
            NumericClaim(
                value=float(a), unit="count", raw=m.group(0), kind="ratio_part", span=m.span()
            )
        )
        claims.append(
            NumericClaim(
                value=float(b), unit="count", raw=m.group(0), kind="ratio_whole", span=m.span()
            )
        )

    for m in _PCT_RE.finditer(text):
        if not _take(m.span()):
            continue
        raw_num = m.group(1).replace(",", ".")
        try:
            val = float(raw_num)
        except ValueError:
            continue
        claims.append(
            NumericClaim(
                value=_normalize(val, "percent"),
                unit="percent",
                raw=m.group(0),
                kind="percent",
                span=m.span(),
            )
        )

    for m in _COUNT_NOUN_RE.finditer(text):
        if not _take(m.span()):
            continue
        claims.append(
            NumericClaim(
                value=float(m.group(1)),
                unit="count",
                raw=m.group(0),
                kind="count",
                span=m.span(),
            )
        )

    return claims


# ---------------------------------------------------------------------------
# Errors + stats
# ---------------------------------------------------------------------------

class UnsupportedNumericClaim(Exception):
    def __init__(
        self,
        *,
        value: float,
        unit: str,
        location: str,
        raw: str = "",
        evidence_refs: Optional[list[str]] = None,
        reason: str = "unsupported",
    ) -> None:
        self.value = value
        self.unit = unit
        self.location = location
        self.raw = raw
        self.evidence_refs = list(evidence_refs or [])
        self.reason = reason
        msg = f"Unsupported numeric claim: {value} ({unit}) at {location} [reason={reason}]"
        if raw:
            msg += f" raw={raw!r}"
        super().__init__(msg)


@dataclass
class ClaimValidationStats:
    numeric_claims_found: int = 0
    numeric_claims_supported: int = 0
    numeric_claims_rejected: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "numeric_claims_found": self.numeric_claims_found,
            "numeric_claims_supported": self.numeric_claims_supported,
            "numeric_claims_rejected": self.numeric_claims_rejected,
        }


def validate_claims_against_index(
    text: str,
    index: NumericEvidenceIndex,
    *,
    location: str,
    mode: str,
    evidence_refs: Optional[list[str]] = None,
    stats: Optional[ClaimValidationStats] = None,
) -> ClaimValidationStats:
    """
    mode:
      - "finding": prefer cited refs; allow dataset size as denominator
      - "global": any value present in the global numeric index
    """
    stats = stats or ClaimValidationStats()
    for claim in extract_numeric_claims(text):
        stats.numeric_claims_found += 1
        if mode == "finding":
            ok = index.supports_with_refs(
                claim.value, claim.unit, list(evidence_refs or [])
            )
        else:
            ok = index.supports_global(claim.value, claim.unit)

        if not ok:
            stats.numeric_claims_rejected += 1
            raise UnsupportedNumericClaim(
                value=claim.value,
                unit=claim.unit,
                location=location,
                raw=claim.raw,
                evidence_refs=list(evidence_refs or []),
                reason=f"unsupported_{claim.unit}",
            )
        stats.numeric_claims_supported += 1
    return stats
