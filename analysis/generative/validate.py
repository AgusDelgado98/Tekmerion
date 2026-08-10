"""
Deterministic validation of GeneratedAnalysis against a GroundingPayload.

V0.5.2: structure + evidence refs + quantitative + ranking claim guardrails.
Does not use an LLM.
"""

from __future__ import annotations

from analysis.generative.models import GeneratedAnalysis, GenerativeError, Finding
from analysis.generative.numeric import (
    ClaimValidationStats,
    UnsupportedNumericClaim,
    build_numeric_index,
    validate_claims_against_index,
)
from analysis.generative.ranking import (
    RankingValidationStats,
    UnsupportedRankingClaim,
    build_ranking_index,
    validate_ranking_claims,
)
from analysis.grounding import GroundingPayload


def validate_generated_analysis(
    analysis: GeneratedAnalysis,
    grounding: GroundingPayload,
) -> GeneratedAnalysis:
    """Validate and normalize a GeneratedAnalysis. Raises GenerativeError on violations."""
    if not isinstance(analysis, GeneratedAnalysis):
        raise GenerativeError("Analysis is not a GeneratedAnalysis instance")

    if not analysis.summary or not str(analysis.summary).strip():
        raise GenerativeError("Analysis summary is empty")

    if not analysis.key_findings:
        raise GenerativeError("Analysis has no key_findings")

    allowed = grounding.item_ids()
    all_refs: list[str] = []

    for i, finding in enumerate(analysis.key_findings):
        if not isinstance(finding, Finding):
            raise GenerativeError(f"key_findings[{i}] is not a Finding")
        if not finding.text or not finding.text.strip():
            raise GenerativeError(f"key_findings[{i}] has empty text")
        if not finding.evidence_refs:
            raise GenerativeError(f"key_findings[{i}] has no evidence_refs")
        for ref in finding.evidence_refs:
            if ref not in allowed:
                raise GenerativeError(f"Unknown evidence_ref: {ref}")
            all_refs.append(ref)

    for ref in analysis.evidence_refs:
        if ref not in allowed:
            raise GenerativeError(f"Unknown evidence_ref: {ref}")
        all_refs.append(ref)

    if not analysis.limitations:
        raise GenerativeError("Analysis must include at least one limitation")

    analysis.evidence_refs = sorted(set(all_refs))

    numeric_index = build_numeric_index(grounding)
    ranking_index = build_ranking_index(grounding)
    numeric_stats = ClaimValidationStats()
    ranking_stats = RankingValidationStats()

    try:
        validate_claims_against_index(
            analysis.summary, numeric_index, location="summary", mode="global", stats=numeric_stats
        )
        validate_ranking_claims(
            analysis.summary, ranking_index, location="summary", mode="global", stats=ranking_stats
        )

        for i, finding in enumerate(analysis.key_findings):
            loc = f"key_findings[{i}]"
            refs = list(finding.evidence_refs)
            validate_claims_against_index(
                finding.text, numeric_index, location=loc, mode="finding",
                evidence_refs=refs, stats=numeric_stats,
            )
            validate_ranking_claims(
                finding.text, ranking_index, location=loc, mode="finding",
                evidence_refs=refs, stats=ranking_stats,
            )

        for i, lim in enumerate(analysis.limitations):
            loc = f"limitations[{i}]"
            validate_claims_against_index(
                lim, numeric_index, location=loc, mode="global", stats=numeric_stats
            )
            validate_ranking_claims(
                lim, ranking_index, location=loc, mode="global", stats=ranking_stats
            )
    except (UnsupportedNumericClaim, UnsupportedRankingClaim) as exc:
        raise GenerativeError(str(exc)) from exc

    analysis._claim_stats = numeric_stats  # type: ignore[attr-defined]
    analysis._ranking_stats = ranking_stats  # type: ignore[attr-defined]
    return analysis



def validate_role_comparison(
    analysis: GeneratedAnalysis,
    grounding: GroundingPayload,
    role_a: str,
    role_b: str,
) -> GeneratedAnalysis:
    """
    Validate role_comparison output: base guards + structured skill lists + role scope.
    """
    from analysis.generative.comparison import ref_in_scope

    # Normalize: differences → key_findings if needed
    if not analysis.key_findings and analysis.differences:
        analysis.key_findings = list(analysis.differences)
    if analysis.differences and not analysis.key_findings:
        analysis.key_findings = list(analysis.differences)

    analysis.task = "role_comparison"
    analysis.role_a = role_a
    analysis.role_b = role_b

    # Base structure + numeric + ranking
    analysis = validate_generated_analysis(analysis, grounding)

    # Third-role / out-of-scope refs
    for ref in analysis.evidence_refs:
        if not ref_in_scope(ref, role_a, role_b):
            raise GenerativeError(f"Out-of-scope evidence_ref for role comparison: {ref}")
    for i, finding in enumerate(analysis.key_findings):
        for ref in finding.evidence_refs:
            if not ref_in_scope(ref, role_a, role_b):
                raise GenerativeError(
                    f"Out-of-scope evidence_ref in key_findings[{i}]: {ref}"
                )

    # Structured skill lists must match grounding
    items = {it.id: it for it in grounding.items}
    expected_shared = list(items["comparison.shared_skills"].value) if "comparison.shared_skills" in items else []
    exp_only_a = list(items[f"comparison.only_{role_a}"].value) if f"comparison.only_{role_a}" in items else []
    exp_only_b = list(items[f"comparison.only_{role_b}"].value) if f"comparison.only_{role_b}" in items else []

    def _norm_list(xs):
        return sorted(str(x) for x in (xs or []))

    if analysis.shared_skills is not None and analysis.shared_skills != []:
        if _norm_list(analysis.shared_skills) != _norm_list(expected_shared):
            raise GenerativeError("shared_skills does not match grounding comparison.shared_skills")
    else:
        analysis.shared_skills = list(expected_shared)

    if analysis.role_a_only_skills:
        if _norm_list(analysis.role_a_only_skills) != _norm_list(exp_only_a):
            raise GenerativeError(f"role_a_only_skills does not match grounding for {role_a}")
    else:
        analysis.role_a_only_skills = list(exp_only_a)

    if analysis.role_b_only_skills:
        if _norm_list(analysis.role_b_only_skills) != _norm_list(exp_only_b):
            raise GenerativeError(f"role_b_only_skills does not match grounding for {role_b}")
    else:
        analysis.role_b_only_skills = list(exp_only_b)

    analysis.differences = list(analysis.key_findings)
    return analysis
