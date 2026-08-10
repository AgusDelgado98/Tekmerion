"""
Rule-based classifiers for role family and seniority.

These are deterministic, explicit and easy to audit.
They can be improved later with more sophisticated methods,
but the evidence must remain traceable to rules.
"""

from __future__ import annotations

import re
from analysis.models import RoleFamily, Seniority


# ---------------------------------------------------------------------------
# Role Family
# ---------------------------------------------------------------------------

# Ordered by specificity. First match wins.
_ROLE_PATTERNS: list[tuple[RoleFamily, list[str]]] = [
    (
        RoleFamily.ML_ENGINEER,
        [
            r"\bml[\s\-]?engineer\b",
            r"\bmachine[\s\-]?learning[\s\-]?engineer\b",
            r"\bmlops\b",
            r"\bml[\s\-]?ops\b",
        ],
    ),
    (
        RoleFamily.DATA_ENGINEER,
        [
            r"\bdata[\s\-]?engineer\b",
            r"\betl[\s\-]?engineer\b",
            r"\bdata[\s\-]?platform\b",
        ],
    ),
    (
        RoleFamily.DATA_SCIENTIST,
        [
            r"\bdata[\s\-]?scientist\b",
            r"\bresearch[\s\-]?scientist\b",
            r"\bai[\s\-]?research\b",
        ],
    ),
    (
        RoleFamily.AI_ANALYST,
        [
            r"\bai[\s\-]?analyst\b",
            r"\binteligencia[\s\-]?artificial\b",
            r"\bgenerative[\s\-]?ai\b",
            r"\bllm[\s\-]?analyst\b",
            r"\bprompt[\s\-]?engineer\b",
        ],
    ),
    (
        RoleFamily.BI_ANALYST,
        [
            r"\bbi[\s\-]?analyst\b",
            r"\bbi[\s\-]?developer\b",
            r"\bbusiness[\s\-]?intelligence\b",
            r"\bpower[\s\-]?bi\b.*\banalyst\b",
            r"\btableau\b.*\banalyst\b",
        ],
    ),
    (
        RoleFamily.DATA_ANALYST,
        [
            r"\bdata[\s\-]?analyst\b",
            r"\banalista[\s\-]?de[\s\-]?datos\b",
            r"\banalista[\s\-]?de[\s\-]?data\b",
            r"\bproduct[\s\-]?analyst\b",
        ],
    ),
    (
        RoleFamily.BUSINESS_ANALYST,
        [
            r"\bbusiness[\s\-]?analyst\b",
            r"\banalista[\s\-]?de[\s\-]?negocios\b",
            r"\banalista[\s\-]?funcional\b",
        ],
    ),
]


def classify_role_family(title: str, description: str = "") -> RoleFamily:
    """Classify role family from title (primary) and description (secondary)."""
    text = f"{title} {description}".lower()

    for family, patterns in _ROLE_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return family

    return RoleFamily.UNKNOWN


# ---------------------------------------------------------------------------
# Seniority
# ---------------------------------------------------------------------------

_SENIORITY_PATTERNS: list[tuple[Seniority, list[str]]] = [
    (
        Seniority.LEAD,
        [
            r"\blead\b",
            r"\bhead\b",
            r"\bprincipal\b",
            r"\bstaff\b",
            r"\bmanager\b",
            r"\bdirector\b",
            r"\+7\s*años\b",
            r"\+8\s*años\b",
            r"\+10\s*años\b",
        ],
    ),
    (
        Seniority.SENIOR,
        [
            r"\bsenior\b",
            r"\bsr\b",
            r"\bsr\.\b",
            r"\bexperto\b",
            r"\bavanzad[oa]\b",
            r"\+5\s*años\b",
            r"\+6\s*años\b",
        ],
    ),
    (
        Seniority.JUNIOR,
        [
            r"\bjunior\b",
            r"\bjr\b",
            r"\bjr\.\b",
            r"\btrainee\b",
            r"\bentry[\s\-]?level\b",
            r"\begresad[oa]\b",
            r"\bjúnior\b",
        ],
    ),
    (
        Seniority.MID,
        [
            r"\bmid[\s\-]?level\b",
            r"\bmid\b",
            r"\bsemi[\s\-]?senior\b",
            r"\bssr\b",
            r"\bintermedio\b",
        ],
    ),
]


def classify_seniority(title: str, description: str = "") -> Seniority:
    """Classify seniority. Title has higher weight conceptually."""
    text = f"{title} {description}".lower()

    for seniority, patterns in _SENIORITY_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return seniority

    # Default heuristic: if nothing matches, treat as mid (common in market)
    # but we keep it explicit as UNKNOWN for transparency.
    return Seniority.UNKNOWN
