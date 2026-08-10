"""
Generative providers for Tekmérion.

Modes:
  disabled  — no generation (default)
  fake      — deterministic offline responses for tests / demos
  openai_compatible — HTTP chat completions (OpenAI, or compatible base URL)

Never log or raise messages that include API keys.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Optional

from analysis.generative.models import (
    AnalysisRequest,
    Finding,
    GeneratedAnalysis,
    GenerativeError,
)
from analysis.generative.prompts import PROMPT_VERSION, build_market_summary_messages, build_role_comparison_messages, ROLE_COMPARISON_PROMPT_VERSION
from analysis.grounding import GroundingPayload


ENV_PROVIDER = "TEKMERION_LLM_PROVIDER"  # disabled | fake | openai_compatible
ENV_API_KEY = "TEKMERION_LLM_API_KEY"
ENV_BASE_URL = "TEKMERION_LLM_BASE_URL"
ENV_MODEL = "TEKMERION_LLM_MODEL"

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


class GenerativeProvider(ABC):
    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def generate(self, request: AnalysisRequest) -> GeneratedAnalysis:
        ...


class DisabledProvider(GenerativeProvider):
    name = "disabled"

    def is_available(self) -> bool:
        return False

    def generate(self, request: AnalysisRequest) -> GeneratedAnalysis:
        raise GenerativeError(
            "Generative analysis is disabled. "
            f"Set {ENV_PROVIDER}=fake or {ENV_PROVIDER}=openai_compatible with credentials."
        )


class FakeProvider(GenerativeProvider):
    """
    Deterministic provider for tests and offline demos.

    Builds a valid analysis strictly from grounding items (no external calls).
    Optional ``corrupt`` modes inject invalid payloads for validation tests.
    """

    name = "fake"

    def __init__(self, *, corrupt: Optional[str] = None) -> None:
        # corrupt: None | "empty" | "bad_ref" | "bad_structure"
        self.corrupt = corrupt

    def is_available(self) -> bool:
        return True

    def generate(self, request: AnalysisRequest) -> GeneratedAnalysis:
        if request.task == "role_comparison":
            return _build_fake_role_comparison(request, corrupt=self.corrupt)
        if self.corrupt == "empty":
            return GeneratedAnalysis(
                summary="",
                key_findings=[],
                limitations=[],
                evidence_refs=[],
                prompt_version=PROMPT_VERSION,
                provider=self.name,
                model="fake-deterministic",
                grounding_fingerprint=request.grounding.fingerprint(),
            )
        if self.corrupt == "bad_ref":
            return GeneratedAnalysis(
                summary="Invented claim about the market.",
                key_findings=[
                    Finding(
                        text="Quantum computing is required in 99% of roles.",
                        evidence_refs=["skills.quantum.pct"],
                    )
                ],
                limitations=["None"],
                evidence_refs=["skills.quantum.pct"],
                prompt_version=PROMPT_VERSION,
                provider=self.name,
                model="fake-deterministic",
                grounding_fingerprint=request.grounding.fingerprint(),
            )
        if self.corrupt == "bad_structure":
            # Will fail when parsed elsewhere; here we return empty findings
            return GeneratedAnalysis(
                summary="Something",
                key_findings=[],
                limitations=[],
                evidence_refs=[],
                prompt_version=PROMPT_VERSION,
                provider=self.name,
                model="fake-deterministic",
                grounding_fingerprint=request.grounding.fingerprint(),
            )

        return _build_fake_from_grounding(request.grounding)


def _item_map(g: GroundingPayload) -> dict[str, Any]:
    return {i.id: i for i in g.items}


def _build_fake_from_grounding(g: GroundingPayload) -> GeneratedAnalysis:
    items = _item_map(g)
    n = g.n_analysis_records
    mode = g.dataset_mode

    skill_rank = items.get("skills.ranking")
    role_rank = items.get("roles.ranking")
    top_skill = None
    top_skill_pct = None
    top_skill_count = None
    if skill_rank and isinstance(skill_rank.value, list) and skill_rank.value:
        top_skill = skill_rank.value[0].get("skill")
        top_skill_pct = skill_rank.value[0].get("pct")
        top_skill_count = skill_rank.value[0].get("count")

    top_role = None
    top_role_pct = None
    if role_rank and isinstance(role_rank.value, list) and role_rank.value:
        top_role = role_rank.value[0].get("role")
        top_role_pct = role_rank.value[0].get("pct")

    findings: list[Finding] = []
    refs: list[str] = ["dataset.n_analysis_records"]

    if top_skill is not None:
        safe = str(top_skill).replace(" ", "_")
        ref_pct = f"skills.{safe}.pct"
        ref_count = f"skills.{safe}.count"
        # Use technical position (#1) — not "más frecuente" — so ties remain valid.
        text = (
            f"{top_skill} ocupa el puesto 1 en skills.ranking, presente en "
            f"{top_skill_pct}% de las vacantes analizadas ({top_skill_count} de {n})."
        )
        findings.append(Finding(text=text, evidence_refs=[ref_pct, ref_count, "skills.ranking"]))
        refs.extend([ref_pct, ref_count, "skills.ranking"])

    if top_role is not None:
        ref_r = f"roles.{top_role}.pct"
        text = (
            f"{top_role} ocupa el puesto 1 en roles.ranking "
            f"({top_role_pct}% de la muestra)."
        )
        findings.append(Finding(text=text, evidence_refs=[ref_r, "roles.ranking"]))
        refs.extend([ref_r, "roles.ranking"])

    if len(findings) < 2 and skill_rank and isinstance(skill_rank.value, list) and len(skill_rank.value) > 1:
        second = skill_rank.value[1]
        s2 = second.get("skill")
        safe2 = str(s2).replace(" ", "_")
        findings.append(
            Finding(
                text=f"{s2} ocupa el puesto 2 en skills.ranking ({second.get('pct')}%).",
                evidence_refs=[f"skills.{safe2}.pct", "skills.ranking"],
            )
        )
        refs.extend([f"skills.{safe2}.pct", "skills.ranking"])

    if mode == "synthetic":
        summary = (
            f"Análisis de una muestra sintética de desarrollo con {n} vacantes válidas "
            f"no duplicadas. Los resultados demuestran el motor analítico y no representan "
            f"el mercado laboral real."
        )
        limitations = [
            f"Dataset sintético de {n} registros de análisis; no representa el mercado real.",
            "Las descripciones y empresas son artificiales.",
        ]
    else:
        country = g.country or "n/d"
        summary = (
            f"Análisis fundamentado de un market snapshot ({country}) con {n} vacantes "
            f"válidas no duplicadas. Las afirmaciones se limitan a la evidencia calculada "
            f"por Tekmérion para esta corrida."
        )
        limitations = [
            f"Muestra limitada a {n} vacantes de análisis en esta corrida.",
            "Las descripciones de Adzuna son snippets; la extracción de skills puede ser incompleta.",
        ]
        if g.retrieved_at:
            limitations.append(f"Snapshot retrieved_at={g.retrieved_at}.")

    if not findings:
        findings.append(
            Finding(
                text=f"La muestra de análisis contiene {n} registros válidos no duplicados.",
                evidence_refs=["dataset.n_analysis_records"],
            )
        )

    return GeneratedAnalysis(
        summary=summary,
        key_findings=findings,
        limitations=limitations,
        evidence_refs=sorted(set(refs)),
        task=g.task,
        prompt_version=PROMPT_VERSION,
        provider="fake",
        model="fake-deterministic",
        grounding_fingerprint=g.fingerprint(),
    )


class OpenAICompatibleProvider(GenerativeProvider):
    """
    Minimal chat-completions client via stdlib urllib.

    Compatible with OpenAI and other /v1/chat/completions endpoints.
    """

    name = "openai_compatible"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
    ) -> None:
        if not api_key or not api_key.strip():
            raise GenerativeError(f"{ENV_API_KEY} is empty")
        self._api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def __repr__(self) -> str:
        return f"OpenAICompatibleProvider(model={self.model!r}, base_url={self.base_url!r}, api_key=***)"

    def is_available(self) -> bool:
        return bool(self._api_key)

    def generate(self, request: AnalysisRequest) -> GeneratedAnalysis:
        if request.task == "role_comparison":
            role_a = str(request.parameters.get("role_a", ""))
            role_b = str(request.parameters.get("role_b", ""))
            messages = build_role_comparison_messages(request.grounding, role_a, role_b)
        else:
            messages = build_market_summary_messages(request.grounding)
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise GenerativeError(f"LLM HTTP error: status={exc.code}") from None
        except urllib.error.URLError as exc:
            raise GenerativeError(f"LLM network error: {exc.reason}") from None

        try:
            payload = json.loads(raw)
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise GenerativeError("LLM response is not valid chat-completions JSON") from exc

        content = _strip_fences(content)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise GenerativeError("LLM content is not valid JSON") from exc

        if not isinstance(parsed, dict):
            raise GenerativeError("LLM JSON root must be an object")

        analysis = GeneratedAnalysis.from_dict(parsed)
        analysis.prompt_version = (
            ROLE_COMPARISON_PROMPT_VERSION if request.task == "role_comparison" else PROMPT_VERSION
        )
        analysis.provider = self.name
        analysis.model = self.model
        analysis.grounding_fingerprint = request.grounding.fingerprint()
        analysis.task = request.task
        return analysis


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()



def _build_fake_role_comparison(request: AnalysisRequest, *, corrupt: str | None = None) -> GeneratedAnalysis:
    from analysis.generative.prompts import ROLE_COMPARISON_PROMPT_VERSION
    g = request.grounding
    items = {i.id: i for i in g.items}
    role_a = str(items.get("comparison.role_a").value if items.get("comparison.role_a") else request.parameters.get("role_a", ""))
    role_b = str(items.get("comparison.role_b").value if items.get("comparison.role_b") else request.parameters.get("role_b", ""))
    shared = list(items["comparison.shared_skills"].value) if "comparison.shared_skills" in items else []
    only_a = list(items[f"comparison.only_{role_a}"].value) if f"comparison.only_{role_a}" in items else []
    only_b = list(items[f"comparison.only_{role_b}"].value) if f"comparison.only_{role_b}" in items else []
    count_a = items[f"roles.{role_a}.count"].value if f"roles.{role_a}.count" in items else 0
    count_b = items[f"roles.{role_b}.count"].value if f"roles.{role_b}.count" in items else 0
    pct_a = items[f"roles.{role_a}.pct"].value if f"roles.{role_a}.pct" in items else 0
    pct_b = items[f"roles.{role_b}.pct"].value if f"roles.{role_b}.pct" in items else 0
    n = g.n_analysis_records
    threshold = items["comparison.small_sample_threshold"].value if "comparison.small_sample_threshold" in items else 5

    if corrupt == "empty":
        return GeneratedAnalysis(
            summary="", key_findings=[], limitations=[], evidence_refs=[],
            task="role_comparison", prompt_version=ROLE_COMPARISON_PROMPT_VERSION,
            provider="fake", model="fake-deterministic",
            grounding_fingerprint=g.fingerprint(), role_a=role_a, role_b=role_b,
        )
    if corrupt == "invented_shared":
        return GeneratedAnalysis(
            summary=f"Comparación {role_a} vs {role_b}.",
            key_findings=[Finding(text="Ambos usan quantum.", evidence_refs=["comparison.shared_skills"])],
            limitations=["Muestra limitada."],
            evidence_refs=["comparison.shared_skills"],
            task="role_comparison", prompt_version=ROLE_COMPARISON_PROMPT_VERSION,
            provider="fake", model="fake-deterministic",
            grounding_fingerprint=g.fingerprint(),
            role_a=role_a, role_b=role_b,
            shared_skills=list(shared) + ["quantum_computing"],
            role_a_only_skills=list(only_a),
            role_b_only_skills=list(only_b),
            differences=[Finding(text="Ambos usan quantum.", evidence_refs=["comparison.shared_skills"])],
        )
    if corrupt == "third_role_ref":
        return GeneratedAnalysis(
            summary=f"Comparación {role_a} vs {role_b}.",
            key_findings=[Finding(
                text=f"{role_a} tiene {count_a} vacantes.",
                evidence_refs=[f"roles.{role_a}.count", "roles.data_engineer.count"],
            )],
            limitations=[f"Muestra de {n} registros."],
            evidence_refs=[f"roles.{role_a}.count", "roles.data_engineer.count"],
            task="role_comparison", prompt_version=ROLE_COMPARISON_PROMPT_VERSION,
            provider="fake", model="fake-deterministic",
            grounding_fingerprint=g.fingerprint(),
            role_a=role_a, role_b=role_b,
            shared_skills=list(shared),
            role_a_only_skills=list(only_a),
            role_b_only_skills=list(only_b),
        )
    if corrupt == "bad_count":
        return GeneratedAnalysis(
            summary=f"Comparación {role_a} vs {role_b}.",
            key_findings=[Finding(
                text=f"{role_a} concentra 99 vacantes.",
                evidence_refs=[f"roles.{role_a}.count"],
            )],
            limitations=[f"Muestra de {n} registros."],
            evidence_refs=[f"roles.{role_a}.count"],
            task="role_comparison", prompt_version=ROLE_COMPARISON_PROMPT_VERSION,
            provider="fake", model="fake-deterministic",
            grounding_fingerprint=g.fingerprint(),
            role_a=role_a, role_b=role_b,
            shared_skills=list(shared),
            role_a_only_skills=list(only_a),
            role_b_only_skills=list(only_b),
        )

    findings = []
    refs = [f"roles.{role_a}.count", f"roles.{role_b}.count", "dataset.n_analysis_records"]
    findings.append(Finding(
        text=(
            f"{role_a} aporta {count_a} vacantes ({pct_a}%) y {role_b} aporta "
            f"{count_b} ({pct_b}%) sobre {n} registros de análisis."
        ),
        evidence_refs=[f"roles.{role_a}.count", f"roles.{role_a}.pct",
                       f"roles.{role_b}.count", f"roles.{role_b}.pct",
                       "dataset.n_analysis_records"],
    ))
    refs.extend([f"roles.{role_a}.pct", f"roles.{role_b}.pct"])
    if shared:
        findings.append(Finding(
            text=f"Skills compartidas en este dataset: {', '.join(shared[:5])}.",
            evidence_refs=["comparison.shared_skills"],
        ))
        refs.append("comparison.shared_skills")
    if only_a:
        findings.append(Finding(
            text=f"Skills solo en {role_a}: {', '.join(only_a[:5])}.",
            evidence_refs=[f"comparison.only_{role_a}"],
        ))
        refs.append(f"comparison.only_{role_a}")
    if only_b:
        findings.append(Finding(
            text=f"Skills solo en {role_b}: {', '.join(only_b[:5])}.",
            evidence_refs=[f"comparison.only_{role_b}"],
        ))
        refs.append(f"comparison.only_{role_b}")

    limitations = [
        f"Comparación basada en {count_a} vacantes de {role_a} y {count_b} de {role_b} "
        f"(dataset n={n}).",
    ]
    if count_a < threshold or count_b < threshold:
        limitations.append(
            f"Al menos un rol tiene menos de {threshold} registros; las diferencias son ilustrativas, "
            f"no inferencia estadística."
        )
    if g.dataset_mode == "synthetic":
        limitations.append("Dataset sintético de desarrollo; no representa el mercado real.")

    summary = (
        f"Comparación fundamentada entre {role_a} y {role_b} en el dataset activo "
        f"({g.dataset_mode}). Se limitan afirmaciones a skills y conteos observados."
    )
    return GeneratedAnalysis(
        summary=summary,
        key_findings=findings,
        limitations=limitations,
        evidence_refs=sorted(set(refs)),
        task="role_comparison",
        prompt_version=ROLE_COMPARISON_PROMPT_VERSION,
        provider="fake",
        model="fake-deterministic",
        grounding_fingerprint=g.fingerprint(),
        role_a=role_a,
        role_b=role_b,
        shared_skills=list(shared),
        role_a_only_skills=list(only_a),
        role_b_only_skills=list(only_b),
        differences=list(findings),
    )


def get_provider_from_env(
    *,
    provider_name: Optional[str] = None,
) -> GenerativeProvider:
    """
    Resolve provider from environment.

    TEKMERION_LLM_PROVIDER=disabled|fake|openai_compatible
    """
    name = (provider_name or os.environ.get(ENV_PROVIDER) or "disabled").strip().lower()
    if name in ("", "disabled", "off", "none"):
        return DisabledProvider()
    if name == "fake":
        return FakeProvider()
    if name in ("openai_compatible", "openai"):
        key = os.environ.get(ENV_API_KEY, "")
        if not key or not key.strip():
            raise GenerativeError(
                f"Provider {name} requires {ENV_API_KEY}. "
                "Deterministic Tekmérion features remain available without it."
            )
        base = os.environ.get(ENV_BASE_URL, DEFAULT_BASE_URL) or DEFAULT_BASE_URL
        model = os.environ.get(ENV_MODEL, DEFAULT_MODEL) or DEFAULT_MODEL
        return OpenAICompatibleProvider(api_key=key, base_url=base, model=model)
    raise GenerativeError(
        f"Unknown {ENV_PROVIDER}={name!r}. Use disabled, fake, or openai_compatible."
    )
