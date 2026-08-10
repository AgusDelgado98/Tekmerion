"""
Ranking claim extraction and RankingEvidenceIndex (V0.5.2).

Deterministic. No NLP beyond conservative patterns.
Technical position follows the ordered list already in GroundingPayload.
Statistical leadership ("más frecuente") requires a unique max count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from analysis.grounding import GroundingPayload


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RankEntry:
    item: str
    position: int  # 1-based technical position in ordered ranking
    count: Optional[float]
    pct: Optional[float]
    is_unique_leader: bool  # count strictly greater than every other entry
    tied_for_position: bool  # same count as previous or next neighbor group


@dataclass
class RankingTable:
    ref: str
    kind: str  # skills | roles | seniority
    entries: list[RankEntry] = field(default_factory=list)
    by_item: dict[str, RankEntry] = field(default_factory=dict)

    def get(self, item: str) -> Optional[RankEntry]:
        return self.by_item.get(normalize_item(item))


@dataclass
class RankingEvidenceIndex:
    tables: dict[str, RankingTable] = field(default_factory=dict)  # ref -> table

    def tables_for_refs(self, refs: list[str]) -> list[RankingTable]:
        return [self.tables[r] for r in refs if r in self.tables]

    def all_tables(self) -> list[RankingTable]:
        return list(self.tables.values())


def normalize_item(name: str) -> str:
    """Deterministic item key: casefold, spaces/hyphens → underscore."""
    s = str(name).casefold().strip()
    s = s.replace("-", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" ", "_")
    return s


def build_ranking_index(grounding: GroundingPayload) -> RankingEvidenceIndex:
    idx = RankingEvidenceIndex()
    for item in grounding.items:
        if item.unit != "list" or not isinstance(item.value, list):
            continue
        if item.id == "skills.ranking":
            kind, key_field = "skills", "skill"
        elif item.id == "roles.ranking":
            kind, key_field = "roles", "role"
        elif item.id == "seniority.ranking":
            kind, key_field = "seniority", "seniority"
        else:
            continue  # cooccurrence etc. not treated as entity rankings

        raw_entries = [e for e in item.value if isinstance(e, dict) and key_field in e]
        if not raw_entries:
            continue

        counts = []
        for e in raw_entries:
            try:
                counts.append(float(e.get("count")) if e.get("count") is not None else None)
            except (TypeError, ValueError):
                counts.append(None)

        max_count = None
        present_counts = [c for c in counts if c is not None]
        if present_counts:
            max_count = max(present_counts)
        leaders = sum(1 for c in present_counts if c == max_count) if max_count is not None else 0
        unique_leader = leaders == 1

        entries: list[RankEntry] = []
        by_item: dict[str, RankEntry] = {}
        for pos, e in enumerate(raw_entries, start=1):
            name = str(e[key_field])
            cnt = counts[pos - 1]
            try:
                pct = float(e["pct"]) if e.get("pct") is not None else None
            except (TypeError, ValueError):
                pct = None
            # tied if same count as any other entry
            tied = False
            if cnt is not None:
                tied = sum(1 for c in counts if c == cnt) > 1
            is_leader = bool(
                unique_leader and cnt is not None and max_count is not None and cnt == max_count
            )
            entry = RankEntry(
                item=name,
                position=pos,
                count=cnt,
                pct=pct,
                is_unique_leader=is_leader,
                tied_for_position=tied,
            )
            entries.append(entry)
            by_item[normalize_item(name)] = entry

        table = RankingTable(ref=item.id, kind=kind, entries=entries, by_item=by_item)
        idx.tables[item.id] = table
    return idx


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RankingClaim:
    item: str
    rank: Optional[int]  # explicit position; None for pure superlative
    ranking_kind: Optional[str]  # skills|roles|seniority if inferred from phrasing
    claim_type: str  # position | superlative
    raw: str
    span: tuple[int, int] = (0, 0)


class UnsupportedRankingClaim(Exception):
    def __init__(
        self,
        *,
        item: str = "",
        rank: Optional[int] = None,
        location: str = "",
        raw: str = "",
        evidence_refs: Optional[list[str]] = None,
        reason: str = "unsupported",
    ) -> None:
        self.item = item
        self.rank = rank
        self.location = location
        self.raw = raw
        self.evidence_refs = list(evidence_refs or [])
        self.reason = reason
        msg = (
            f"Unsupported ranking claim: item={item!r} rank={rank} "
            f"at {location} [reason={reason}]"
        )
        if raw:
            msg += f" raw={raw!r}"
        super().__init__(msg)


@dataclass
class RankingValidationStats:
    ranking_claims_found: int = 0
    ranking_claims_supported: int = 0
    ranking_claims_rejected: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "ranking_claims_found": self.ranking_claims_found,
            "ranking_claims_supported": self.ranking_claims_supported,
            "ranking_claims_rejected": self.ranking_claims_rejected,
        }


# Structured position markers (do not overlap with numeric count parser)
# Forms: "Python es #1", "SQL rank 2", "bi_analyst ocupa el puesto 1"
_ITEM = r"(?P<item>[A-Za-zÁÉÍÓÚáéíóúÑñ][A-Za-zÁÉÍÓÚáéíóúÑñ0-9_]*(?:\s+[A-Za-zÁÉÍÓÚáéíóúÑñ][A-Za-zÁÉÍÓÚáéíóúÑñ0-9_]*){0,3})"
_RANK_MARK = r"(?:#|rank\s+|puesto\s+|posici[oó]n\s+)(?P<rank>[1-9]\d?)\b"

_HASH_RANK = re.compile(
    _ITEM
    + r"\s+(?:es|está|ocupa|is|ranks?)\s+(?:el\s+|la\s+|as\s+)?"
    + _RANK_MARK,
    re.I,
)
_HASH_RANK_PREFIX = re.compile(
    r"(?:#|rank\s+|puesto\s+|posici[oó]n\s+)(?P<rank>[1-9]\d?)\s*[:\-]?\s*"
    + r"(?P<item>[A-Za-zÁÉÍÓÚáéíóúÑñ][A-Za-zÁÉÍÓÚáéíóúÑñ0-9_]*(?:\s+[A-Za-zÁÉÍÓÚáéíóúÑñ][A-Za-zÁÉÍÓÚáéíóúÑñ0-9_]*){0,3})",
    re.I,
)

_HASH_RANK_COMPACT = re.compile(
    _ITEM + r"\s+" + _RANK_MARK,
    re.I,
)

_ORDINAL_MAP = {
    "primero": 1,
    "primera": 1,
    "1ro": 1,
    "1ra": 1,
    "segundo": 2,
    "segunda": 2,
    "2do": 2,
    "2da": 2,
    "tercero": 3,
    "tercera": 3,
    "3ro": 3,
    "3ra": 3,
    "first": 1,
    "second": 2,
    "third": 3,
}

_ORDINAL_RE = re.compile(
    _ITEM
    + r"\s+(?:es\s+(?:el|la)\s+|está\s+(?:en\s+)?(?:el|la)\s+|ocupa\s+(?:el|la)\s+|is\s+(?:the\s+)?)"
    + r"(?P<ord>primero|primera|segundo|segunda|tercero|tercera|first|second|third)\b",
    re.I,
)

# Superlatives — only with explicit ranking domain words
_SUPERLATIVE_RE = re.compile(
    _ITEM
    + r"\s+(?:es\s+(?:la|el)\s+|is\s+(?:the\s+)?)"
    + r"(?:"
    r"(?P<kind_skill>skill|habilidad)\s+m[aá]s\s+frecuente"
    r"|(?P<kind_role>rol|role\s+family|familia)\s+m[aá]s\s+frecuente"
    r"|m[aá]s\s+frecuente(?:\s+(?P<kind_skill2>skill|habilidad))?"
    r"|most\s+frequent\s+(?P<kind_en>skill|role)"
    r"|top\s+(?P<kind_top>skill|role)"
    r")",
    re.I,
)


def _clean_item_token(raw: str) -> str:
    s = raw.strip().strip(".,;:()[]\"'")
    # drop trailing domain words accidentally captured
    s = re.sub(
        r"\b(es|está|ocupa|is|ranks?|the|el|la|skill|rol|role|familia)\b$",
        "",
        s,
        flags=re.I,
    ).strip()
    return s


def extract_ranking_claims(text: str) -> list[RankingClaim]:
    if not text:
        return []
    claims: list[RankingClaim] = []
    occupied: list[tuple[int, int]] = []

    def _take(span: tuple[int, int]) -> bool:
        if any(not (span[1] <= a or span[0] >= b) for a, b in occupied):
            return False
        occupied.append(span)
        return True

    for pat in (_HASH_RANK, _HASH_RANK_PREFIX, _HASH_RANK_COMPACT):
        for m in pat.finditer(text):
            if not _take(m.span()):
                continue
            item = _clean_item_token(m.group("item"))
            rank = int(m.group("rank"))
            if not item or rank < 1:
                continue
            claims.append(
                RankingClaim(
                    item=item,
                    rank=rank,
                    ranking_kind=None,
                    claim_type="position",
                    raw=m.group(0),
                    span=m.span(),
                )
            )

    for m in _ORDINAL_RE.finditer(text):
        if not _take(m.span()):
            continue
        item = _clean_item_token(m.group("item"))
        ord_word = m.group("ord").casefold()
        rank = _ORDINAL_MAP.get(ord_word)
        if not item or rank is None:
            continue
        claims.append(
            RankingClaim(
                item=item,
                rank=rank,
                ranking_kind=None,
                claim_type="position",
                raw=m.group(0),
                span=m.span(),
            )
        )

    for m in _SUPERLATIVE_RE.finditer(text):
        if not _take(m.span()):
            continue
        item = _clean_item_token(m.group("item"))
        kind = None
        if m.groupdict().get("kind_skill") or m.groupdict().get("kind_skill2"):
            kind = "skills"
        elif m.groupdict().get("kind_role"):
            kind = "roles"
        elif m.groupdict().get("kind_en"):
            kind = "skills" if "skill" in m.group("kind_en").lower() else "roles"
        elif m.groupdict().get("kind_top"):
            kind = "skills" if "skill" in m.group("kind_top").lower() else "roles"
        else:
            kind = "skills"  # bare "más frecuente" after item — soft default skills
        if not item:
            continue
        claims.append(
            RankingClaim(
                item=item,
                rank=None,
                ranking_kind=kind,
                claim_type="superlative",
                raw=m.group(0),
                span=m.span(),
            )
        )

    return claims


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_ranking_claims(
    text: str,
    index: RankingEvidenceIndex,
    *,
    location: str,
    mode: str,
    evidence_refs: Optional[list[str]] = None,
    stats: Optional[RankingValidationStats] = None,
) -> RankingValidationStats:
    """
    mode:
      - finding: only ranking tables cited in evidence_refs
      - global: any ranking table
    """
    stats = stats or RankingValidationStats()
    claims = extract_ranking_claims(text)

    if mode == "finding":
        tables = index.tables_for_refs(list(evidence_refs or []))
    else:
        tables = index.all_tables()

    for claim in claims:
        stats.ranking_claims_found += 1
        matched_entry: Optional[RankEntry] = None
        matched_table: Optional[RankingTable] = None

        # Prefer kind hint for superlatives
        candidates = tables
        if claim.ranking_kind:
            kind_filtered = [t for t in tables if t.kind == claim.ranking_kind]
            if kind_filtered:
                candidates = kind_filtered

        for table in candidates:
            entry = table.get(claim.item)
            if entry is not None:
                matched_entry = entry
                matched_table = table
                break

        if matched_entry is None or matched_table is None:
            stats.ranking_claims_rejected += 1
            raise UnsupportedRankingClaim(
                item=claim.item,
                rank=claim.rank,
                location=location,
                raw=claim.raw,
                evidence_refs=list(evidence_refs or []),
                reason="item_not_in_cited_rankings",
            )

        if claim.claim_type == "position":
            if claim.rank is None or matched_entry.position != claim.rank:
                stats.ranking_claims_rejected += 1
                raise UnsupportedRankingClaim(
                    item=claim.item,
                    rank=claim.rank,
                    location=location,
                    raw=claim.raw,
                    evidence_refs=list(evidence_refs or []),
                    reason="rank_mismatch",
                )
        elif claim.claim_type == "superlative":
            # Statistical leadership: unique max count required
            if not matched_entry.is_unique_leader:
                stats.ranking_claims_rejected += 1
                raise UnsupportedRankingClaim(
                    item=claim.item,
                    rank=1,
                    location=location,
                    raw=claim.raw,
                    evidence_refs=list(evidence_refs or []),
                    reason="tied_or_not_leader",
                )

        stats.ranking_claims_supported += 1

    return stats
