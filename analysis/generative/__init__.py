"""
Grounded generative analysis (V0.5.0).

The model interprets EvidenceReport-derived grounding only.
It does not classify jobs, compute metrics, or invent market facts.
"""

from analysis.generative.models import (
    AnalysisRequest,
    GeneratedAnalysis,
    Finding,
    GenerativeError,
)
from analysis.generative.prompts import PROMPT_VERSION, build_market_summary_messages
from analysis.generative.providers import (
    GenerativeProvider,
    DisabledProvider,
    FakeProvider,
    OpenAICompatibleProvider,
    get_provider_from_env,
)
from analysis.generative.validate import validate_generated_analysis
from analysis.generative.numeric import (
    NumericEvidenceIndex,
    build_numeric_index,
    extract_numeric_claims,
    UnsupportedNumericClaim,
    ClaimValidationStats,
)
from analysis.generative.service import run_market_summary

__all__ = [
    "AnalysisRequest",
    "GeneratedAnalysis",
    "Finding",
    "GenerativeError",
    "PROMPT_VERSION",
    "build_market_summary_messages",
    "GenerativeProvider",
    "DisabledProvider",
    "FakeProvider",
    "OpenAICompatibleProvider",
    "get_provider_from_env",
    "validate_generated_analysis",
    "run_market_summary",
    "NumericEvidenceIndex",
    "build_numeric_index",
    "extract_numeric_claims",
    "UnsupportedNumericClaim",
    "ClaimValidationStats",
]
