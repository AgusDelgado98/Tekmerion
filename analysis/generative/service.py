"""
Orchestration: grounding → provider → validation.
"""

from __future__ import annotations

from typing import Optional, Sequence

from analysis.evidence import EvidenceReport
from analysis.generative.models import AnalysisRequest, GeneratedAnalysis, GenerativeError
from analysis.generative.providers import GenerativeProvider
from analysis.generative.validate import validate_generated_analysis, validate_role_comparison
from analysis.generative.comparison import (
    TASK_ROLE_COMPARISON,
    build_role_comparison_grounding,
    validate_role_pair,
)
from analysis.grounding import build_grounding_payload
from analysis.models import ProcessedJob


_CACHE: dict[str, GeneratedAnalysis] = {}


def clear_analysis_cache() -> None:
    _CACHE.clear()


def run_market_summary(
    *,
    evidence: EvidenceReport,
    dataset_mode: str,
    dataset_source: str,
    dataset_label: str,
    provider: GenerativeProvider,
    retrieved_at: Optional[str] = None,
    country: Optional[str] = None,
    query_count: int = 0,
    use_cache: bool = True,
) -> GeneratedAnalysis:
    if not provider.is_available():
        raise GenerativeError("Generative provider is not available")

    grounding = build_grounding_payload(
        evidence,
        dataset_mode=dataset_mode,
        dataset_source=dataset_source,
        dataset_label=dataset_label,
        retrieved_at=retrieved_at,
        country=country,
        query_count=query_count,
    )
    request = AnalysisRequest(grounding=grounding, task="market_summary")

    cache_key = (
        f"market_summary|{grounding.fingerprint()}|{provider.name}|"
        f"{getattr(provider, 'model', '')}"
    )
    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    raw = provider.generate(request)
    validated = validate_generated_analysis(raw, grounding)
    if use_cache:
        _CACHE[cache_key] = validated
    return validated


def run_role_comparison(
    *,
    evidence: EvidenceReport,
    role_a: str,
    role_b: str,
    dataset_mode: str,
    dataset_source: str,
    dataset_label: str,
    provider: GenerativeProvider,
    records: Optional[Sequence[ProcessedJob]] = None,
    retrieved_at: Optional[str] = None,
    country: Optional[str] = None,
    use_cache: bool = True,
) -> GeneratedAnalysis:
    """
    Compare two role families with grounded generation.
    Validates roles locally before any provider call.
    """
    if not provider.is_available():
        raise GenerativeError("Generative provider is not available")

    pair = validate_role_pair(evidence, role_a, role_b)
    grounding = build_role_comparison_grounding(
        evidence,
        pair.role_a,
        pair.role_b,
        dataset_mode=dataset_mode,
        dataset_source=dataset_source,
        dataset_label=dataset_label,
        records=records,
        retrieved_at=retrieved_at,
        country=country,
    )
    request = AnalysisRequest(
        grounding=grounding,
        task=TASK_ROLE_COMPARISON,
        parameters={"role_a": pair.role_a, "role_b": pair.role_b},
    )

    # Cache key uses canonical pair so A/B order does not duplicate work
    canon = "|".join(pair.canonical())
    cache_key = (
        f"role_comparison|{canon}|{grounding.fingerprint()}|{provider.name}|"
        f"{getattr(provider, 'model', '')}"
    )
    if use_cache and cache_key in _CACHE:
        cached = _CACHE[cache_key]
        # Preserve requested presentation order
        cached.role_a = pair.role_a
        cached.role_b = pair.role_b
        return cached

    raw = provider.generate(request)
    validated = validate_role_comparison(raw, grounding, pair.role_a, pair.role_b)
    if use_cache:
        _CACHE[cache_key] = validated
    return validated
