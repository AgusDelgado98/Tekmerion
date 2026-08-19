"""
Adzuna API source adapter (V0.4.3).

Official API (reviewed 2026-08-10):
  Base:  https://api.adzuna.com/v1/api
  Search: GET /jobs/{country}/search/{page}
  Auth:  app_id + app_key as query parameters
  Docs:  https://developer.adzuna.com/docs/search

This module only:
  - loads credentials from the environment
  - performs (or accepts) the HTTP call
  - maps Adzuna job objects to Tekmérion-oriented raw dicts
  - optionally persists a raw snapshot

Normalization, namespaced IDs and pipeline enrichment stay outside.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from analysis.ingestion.base import SourceAdapter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_NAME = "adzuna"
API_BASE = "https://api.adzuna.com/v1/api"

# Conservative defaults — never page through the whole market by accident
DEFAULT_COUNTRY = "ar"
DEFAULT_RESULTS_PER_PAGE = 10
DEFAULT_PAGE = 1
MAX_RESULTS_PER_PAGE = 50  # hard ceiling inside this client

ENV_APP_ID = "ADZUNA_APP_ID"
ENV_API_KEY = "ADZUNA_API_KEY"

# Optional injectable HTTP getter: (url: str, timeout: float) -> bytes
HttpGetter = Callable[[str, float], bytes]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class AdzunaConfigError(RuntimeError):
    """Missing or invalid Adzuna credentials / configuration."""


class AdzunaAPIError(RuntimeError):
    """HTTP or payload error from the Adzuna API."""

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdzunaCredentials:
    app_id: str
    api_key: str

    def __post_init__(self) -> None:
        if not self.app_id or not self.app_id.strip():
            raise AdzunaConfigError(
                f"Adzuna app id is empty. Set the {ENV_APP_ID} environment variable."
            )
        if not self.api_key or not self.api_key.strip():
            raise AdzunaConfigError(
                f"Adzuna API key is empty. Set the {ENV_API_KEY} environment variable."
            )

    def __repr__(self) -> str:
        # Never leak secrets in repr / logs
        return "AdzunaCredentials(app_id=***, api_key=***)"


def load_credentials_from_env(
    *,
    app_id_var: str = ENV_APP_ID,
    api_key_var: str = ENV_API_KEY,
) -> AdzunaCredentials:
    """
    Load credentials from environment variables.

    Raises AdzunaConfigError with a clear message if either is missing.
    Never includes secret values in the error message.
    """
    app_id = os.environ.get(app_id_var)
    api_key = os.environ.get(api_key_var)

    missing = []
    if not app_id or not str(app_id).strip():
        missing.append(app_id_var)
    if not api_key or not str(api_key).strip():
        missing.append(api_key_var)

    if missing:
        raise AdzunaConfigError(
            "Missing Adzuna credentials. Set the following environment variable(s): "
            + ", ".join(missing)
        )

    return AdzunaCredentials(app_id=str(app_id).strip(), api_key=str(api_key).strip())


# ---------------------------------------------------------------------------
# HTTP (stdlib, injectable)
# ---------------------------------------------------------------------------

def _default_http_get(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        # Do not include request URL (may contain app_key) in the message
        raise AdzunaAPIError(
            f"Adzuna HTTP error: status={exc.code}",
            status_code=exc.code,
        ) from None
    except urllib.error.URLError as exc:
        raise AdzunaAPIError(f"Adzuna network error: {exc.reason}") from None


def build_search_url(
    *,
    country: str,
    page: int,
    app_id: str,
    api_key: str,
    what: str,
    results_per_page: int,
    where: Optional[str] = None,
) -> str:
    """Build the full search URL. Exposed for tests; includes credentials."""
    path = f"{API_BASE}/jobs/{country}/search/{page}"
    params: dict[str, str] = {
        "app_id": app_id,
        "app_key": api_key,
        "results_per_page": str(results_per_page),
        "what": what,
        "content-type": "application/json",
    }
    if where:
        params["where"] = where
    return f"{path}?{urllib.parse.urlencode(params)}"


# ---------------------------------------------------------------------------
# Mapping Adzuna job → Tekmérion-oriented raw dict
# ---------------------------------------------------------------------------

def map_adzuna_job(ad: dict[str, Any]) -> dict[str, Any]:
    """
    Map a single Adzuna job object to a dict ready for normalize_to_internal.

    Essential fields are promoted to the Tekmérion raw schema.
    Source-specific extras live under ``source_metadata`` (minimal, explicit).
    """
    if not isinstance(ad, dict):
        raise AdzunaAPIError("Adzuna job entry is not an object")

    company_obj = ad.get("company") or {}
    location_obj = ad.get("location") or {}
    category_obj = ad.get("category") or {}

    company_name = ""
    if isinstance(company_obj, dict):
        company_name = str(company_obj.get("display_name") or "").strip()
    elif isinstance(company_obj, str):
        company_name = company_obj.strip()

    location_name = ""
    if isinstance(location_obj, dict):
        location_name = str(location_obj.get("display_name") or "").strip()
    elif isinstance(location_obj, str):
        location_name = location_obj.strip()

    external_id = ad.get("id")
    if external_id is not None:
        external_id = str(external_id).strip()

    mapped: dict[str, Any] = {
        "id": external_id or None,
        "title": str(ad.get("title") or "").strip(),
        "company": company_name,
        "location": location_name,
        "description": str(ad.get("description") or "").strip(),
        "source": SOURCE_NAME,
        "source_url": str(ad.get("redirect_url") or "").strip() or None,
        "posted_date": str(ad.get("created") or "").strip() or None,
        "salary_min": ad.get("salary_min"),
        "salary_max": ad.get("salary_max"),
        "currency": None,  # Adzuna search response does not always include currency
        # retrieved_at is filled by IngestionContext / normalize
    }

    # Minimal source-specific metadata (not required by the pipeline)
    meta: dict[str, Any] = {}
    if isinstance(category_obj, dict):
        if category_obj.get("tag"):
            meta["category_tag"] = category_obj["tag"]
        if category_obj.get("label"):
            meta["category_label"] = category_obj["label"]
    if ad.get("contract_type"):
        meta["contract_type"] = ad["contract_type"]
    if ad.get("contract_time"):
        meta["contract_time"] = ad["contract_time"]
    if ad.get("salary_is_predicted") is not None:
        meta["salary_is_predicted"] = ad["salary_is_predicted"]
    if meta:
        mapped["source_metadata"] = meta

    return mapped


def map_adzuna_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract and map the ``results`` array from an Adzuna search response.

    Raises AdzunaAPIError on unexpected top-level structure.
    """
    if not isinstance(payload, dict):
        raise AdzunaAPIError("Adzuna response is not a JSON object")

    results = payload.get("results")
    if results is None:
        # Empty but well-formed responses may omit results; treat as empty list
        return []
    if not isinstance(results, list):
        raise AdzunaAPIError("Adzuna response 'results' is not a list")

    mapped: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            # Skip garbage entries rather than failing the whole batch
            continue
        mapped.append(map_adzuna_job(item))
    return mapped


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

@dataclass
class AdzunaSearchResult:
    """Outcome of one search call (live or from a preloaded payload)."""

    records: list[dict[str, Any]]
    raw_payload: dict[str, Any]
    country: str
    query: str
    page: int
    results_per_page: int


class AdzunaClient:
    """
    Thin HTTP client for Adzuna job search.

    ``http_get`` is injectable so unit tests never touch the network.
    """

    def __init__(
        self,
        credentials: AdzunaCredentials,
        *,
        http_get: Optional[HttpGetter] = None,
        timeout: float = 30.0,
    ) -> None:
        self.credentials = credentials
        self._http_get = http_get or _default_http_get
        self.timeout = timeout

    def search(
        self,
        *,
        what: str = "data analyst",
        country: str = DEFAULT_COUNTRY,
        page: int = DEFAULT_PAGE,
        results_per_page: int = DEFAULT_RESULTS_PER_PAGE,
        where: Optional[str] = None,
    ) -> AdzunaSearchResult:
        if page < 1:
            raise ValueError("page must be >= 1")
        if results_per_page < 1:
            raise ValueError("results_per_page must be >= 1")
        results_per_page = min(results_per_page, MAX_RESULTS_PER_PAGE)

        url = build_search_url(
            country=country,
            page=page,
            app_id=self.credentials.app_id,
            api_key=self.credentials.api_key,
            what=what,
            results_per_page=results_per_page,
            where=where,
        )

        body = self._http_get(url, self.timeout)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AdzunaAPIError("Adzuna response is not valid JSON") from exc

        if not isinstance(payload, dict):
            raise AdzunaAPIError("Adzuna response is not a JSON object")

        records = map_adzuna_results(payload)
        return AdzunaSearchResult(
            records=records,
            raw_payload=payload,
            country=country,
            query=what,
            page=page,
            results_per_page=results_per_page,
        )


# ---------------------------------------------------------------------------
# Snapshot helper
# ---------------------------------------------------------------------------

def save_raw_snapshot(
    payload: dict[str, Any],
    *,
    directory: str | Path,
    retrieved_at: str,
    country: str,
    query: str,
    page: int,
) -> Path:
    """
    Persist a raw Adzuna response for auditability.

    Filename includes timestamp + country + page so consecutive runs never
    silently overwrite each other. Secrets are never written (payload is the
    API response body only).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # Sanitize for filename
    safe_ts = retrieved_at.replace(":", "").replace("-", "")[:15]
    safe_query = "".join(c if c.isalnum() else "_" for c in query)[:40].strip("_") or "query"
    filename = f"adzuna_{country}_p{page}_{safe_query}_{safe_ts}.json"

    envelope = {
        "retrieved_at": retrieved_at,
        "source": SOURCE_NAME,
        "country": country,
        "query": query,
        "page": page,
        "payload": payload,
    }
    path = directory / filename
    # Never overwrite
    if path.exists():
        stem = path.stem
        suffix = path.suffix
        n = 1
        while path.exists():
            path = directory / f"{stem}_{n}{suffix}"
            n += 1

    with path.open("w", encoding="utf-8") as f:
        json.dump(envelope, f, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------------------
# SourceAdapter
# ---------------------------------------------------------------------------

class AdzunaSource(SourceAdapter):
    """
    SourceAdapter backed by the Adzuna job search API.

    Prefer constructing with an explicit ``AdzunaClient`` (or a preloaded
    list of already-mapped records for offline / fixture use).
    """

    def __init__(
        self,
        *,
        client: Optional[AdzunaClient] = None,
        what: str = "data analyst",
        country: str = DEFAULT_COUNTRY,
        page: int = DEFAULT_PAGE,
        results_per_page: int = DEFAULT_RESULTS_PER_PAGE,
        where: Optional[str] = None,
        preloaded_records: Optional[list[dict[str, Any]]] = None,
        last_raw_payload: Optional[dict[str, Any]] = None,
    ) -> None:
        self._client = client
        self.what = what
        self.country = country
        self.page = page
        self.results_per_page = results_per_page
        self.where = where
        self._preloaded = preloaded_records
        self.last_raw_payload: Optional[dict[str, Any]] = last_raw_payload
        self.last_search: Optional[AdzunaSearchResult] = None

    def source_name(self) -> str:
        return SOURCE_NAME

    def describe(self) -> str:
        return f"Adzuna job search ({self.country}, what={self.what!r})"

    def load(self) -> list[dict[str, Any]]:
        if self._preloaded is not None:
            return [dict(r) for r in self._preloaded]

        if self._client is None:
            raise AdzunaConfigError(
                "AdzunaSource has no client and no preloaded records. "
                "Provide a client or use from_payload()."
            )

        result = self._client.search(
            what=self.what,
            country=self.country,
            page=self.page,
            results_per_page=self.results_per_page,
            where=self.where,
        )
        self.last_search = result
        self.last_raw_payload = result.raw_payload
        return result.records

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        what: str = "fixture",
        country: str = DEFAULT_COUNTRY,
        page: int = 1,
        results_per_page: int = DEFAULT_RESULTS_PER_PAGE,
    ) -> "AdzunaSource":
        """Build an offline adapter from a fixture / snapshot payload."""
        records = map_adzuna_results(payload)
        return cls(
            preloaded_records=records,
            last_raw_payload=payload,
            what=what,
            country=country,
            page=page,
            results_per_page=results_per_page,
        )
