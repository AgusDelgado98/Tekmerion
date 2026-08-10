"""Contracts for grounded analysis requests and responses."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from analysis.grounding import GroundingPayload


class GenerativeError(RuntimeError):
    """Provider or validation failure (safe messages — never include secrets)."""


@dataclass
class Finding:
    text: str
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "evidence_refs": list(self.evidence_refs)}


@dataclass
class GeneratedAnalysis:
    summary: str
    key_findings: list[Finding]
    limitations: list[str]
    evidence_refs: list[str]  # union of refs used across summary/findings
    task: str = "market_summary"
    prompt_version: str = ""
    provider: str = ""
    model: str = ""
    grounding_fingerprint: str = ""
    # role_comparison structured fields (optional)
    role_a: str = ""
    role_b: str = ""
    shared_skills: list[str] = field(default_factory=list)
    role_a_only_skills: list[str] = field(default_factory=list)
    role_b_only_skills: list[str] = field(default_factory=list)
    differences: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "task": self.task,
            "summary": self.summary,
            "key_findings": [f.to_dict() for f in self.key_findings],
            "limitations": list(self.limitations),
            "evidence_refs": list(self.evidence_refs),
            "prompt_version": self.prompt_version,
            "provider": self.provider,
            "model": self.model,
            "grounding_fingerprint": self.grounding_fingerprint,
        }
        if self.task == "role_comparison" or self.role_a:
            d.update({
                "role_a": self.role_a,
                "role_b": self.role_b,
                "shared_skills": list(self.shared_skills),
                "role_a_only_skills": list(self.role_a_only_skills),
                "role_b_only_skills": list(self.role_b_only_skills),
                "differences": [f.to_dict() for f in (self.differences or self.key_findings)],
            })
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeneratedAnalysis":
        findings_raw = data.get("key_findings") or []
        findings: list[Finding] = []
        for item in findings_raw:
            if not isinstance(item, dict):
                raise GenerativeError("key_findings entries must be objects")
            findings.append(
                Finding(
                    text=str(item.get("text") or ""),
                    evidence_refs=[str(x) for x in (item.get("evidence_refs") or [])],
                )
            )
        diffs_raw = data.get("differences") or []
        differences: list[Finding] = []
        for item in diffs_raw:
            if isinstance(item, dict):
                differences.append(
                    Finding(
                        text=str(item.get("text") or ""),
                        evidence_refs=[str(x) for x in (item.get("evidence_refs") or [])],
                    )
                )
        # role_comparison may use differences instead of key_findings
        if not findings and differences:
            findings = list(differences)
        return cls(
            summary=str(data.get("summary") or ""),
            key_findings=findings,
            limitations=[str(x) for x in (data.get("limitations") or [])],
            evidence_refs=[str(x) for x in (data.get("evidence_refs") or [])],
            task=str(data.get("task") or "market_summary"),
            prompt_version=str(data.get("prompt_version") or ""),
            provider=str(data.get("provider") or ""),
            model=str(data.get("model") or ""),
            grounding_fingerprint=str(data.get("grounding_fingerprint") or ""),
            role_a=str(data.get("role_a") or ""),
            role_b=str(data.get("role_b") or ""),
            shared_skills=[str(x) for x in (data.get("shared_skills") or [])],
            role_a_only_skills=[str(x) for x in (data.get("role_a_only_skills") or [])],
            role_b_only_skills=[str(x) for x in (data.get("role_b_only_skills") or [])],
            differences=differences or findings,
        )


@dataclass(frozen=True)
class AnalysisRequest:
    grounding: GroundingPayload
    task: str = "market_summary"
    parameters: dict = field(default_factory=dict)
