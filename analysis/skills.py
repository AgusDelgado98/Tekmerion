"""
Skill extraction and normalization.

Skills are treated as structured data, not free text.
This enables frequency analysis, co-occurrence, gaps, etc.
"""

from __future__ import annotations

import re
from typing import Iterable


# Canonical skill → list of aliases / surface forms
# Order matters only for documentation; matching is independent.
_SKILL_ALIASES: dict[str, list[str]] = {
    # Languages & core
    "python": [r"\bpython\b", r"\bpy\b"],
    "sql": [r"\bsql\b", r"\bt[\-\s]?sql\b", r"\bpl[\-\s]?sql\b"],
    "r": [r"\br\b(?!\w)", r"\brstudio\b"],  # careful with single letter
    "excel": [r"\bexcel\b", r"\bmicrosoft[\s\-]?excel\b"],
    # BI & Visualization
    "power_bi": [r"\bpower[\s\-]?bi\b", r"\bpbi\b", r"\bdax\b"],
    "tableau": [r"\btableau\b"],
    "looker": [r"\blooker\b", r"\blookml\b"],
    "qlik": [r"\bqlik\b", r"\bqlikview\b", r"\bqliksense\b"],
    # Data engineering
    "airflow": [r"\bairflow\b", r"\bapache[\s\-]?airflow\b"],
    "dbt": [r"\bdbt\b", r"\bdata[\s\-]?build[\s\-]?tool\b"],
    "spark": [r"\bspark\b", r"\bpyspark\b", r"\bapache[\s\-]?spark\b"],
    "kafka": [r"\bkafka\b", r"\bapache[\s\-]?kafka\b"],
    "snowflake": [r"\bsnowflake\b"],
    "bigquery": [r"\bbigquery\b", r"\bbq\b"],
    "redshift": [r"\bredshift\b"],
    "databricks": [r"\bdatabricks\b"],
    # ML / AI
    "scikit_learn": [r"\bscikit[\-\s]?learn\b", r"\bsklearn\b"],
    "tensorflow": [r"\btensorflow\b", r"\btf\b"],
    "pytorch": [r"\bpytorch\b", r"\btorch\b"],
    "mlflow": [r"\bmlflow\b"],
    "llm": [r"\bllm\b", r"\bllms\b", r"\blarge[\s\-]?language[\s\-]?model"],
    "prompt_engineering": [r"\bprompt[\s\-]?engineering\b", r"\bprompt[\s\-]?engineer\b"],
    "nlp": [r"\bnlp\b", r"\bnatural[\s\-]?language[\s\-]?processing\b"],
    "deep_learning": [r"\bdeep[\s\-]?learning\b", r"\bdl\b"],
    # Infra / MLOps
    "docker": [r"\bdocker\b"],
    "kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "fastapi": [r"\bfastapi\b"],
    "aws": [r"\baws\b", r"\bamazon[\s\-]?web[\s\-]?services\b"],
    "terraform": [r"\bterraform\b"],
    "git": [r"\bgit\b", r"\bgithub\b", r"\bgitlab\b"],
    # Analytics / Product
    "amplitude": [r"\bamplitude\b"],
    "mixpanel": [r"\bmixpanel\b"],
    "a_b_testing": [r"\ba[\s\-/]?b[\s\-]?test", r"\bexperimentation\b"],
    "statistics": [r"\bestadística\b", r"\bstatistics\b", r"\bestadisticas\b"],
    "pandas": [r"\bpandas\b"],
    "numpy": [r"\bnumpy\b"],
}


# Pre-compile for performance and determinism
_COMPILED_SKILLS: list[tuple[str, list[re.Pattern]]] = [
    (canonical, [re.compile(p, re.IGNORECASE) for p in patterns])
    for canonical, patterns in _SKILL_ALIASES.items()
]


def extract_skills(text: str) -> list[str]:
    """
    Extract and normalize skills from free text.
    Returns a sorted list of canonical skill names (deterministic).
    """
    if not text:
        return []

    found: set[str] = set()
    for canonical, patterns in _COMPILED_SKILLS:
        for pattern in patterns:
            if pattern.search(text):
                found.add(canonical)
                break  # one match is enough per skill

    return sorted(found)


def normalize_skill_list(skills: Iterable[str]) -> list[str]:
    """Utility for external lists (future use)."""
    return sorted({s.strip().lower().replace(" ", "_") for s in skills if s and s.strip()})
