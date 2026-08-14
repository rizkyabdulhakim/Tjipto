"""Optional, untrusted wording adapters for server-rendered answers."""

from __future__ import annotations

import os
from typing import Protocol
from urllib.parse import urlparse


class WordingProvider(Protocol):
    def propose(self, deterministic_answer: str) -> dict[str, object] | None: ...


def wording_enabled_from_environment() -> bool:
    return os.environ.get("TJIPTO_EXTERNAL_WORDING", "").strip().casefold() == "enabled"


def wording_provider_from_environment() -> WordingProvider | None:
    """Build an opted-in adapter; invalid optional configuration is harmless."""
    if not wording_enabled_from_environment():
        return None
    provider = os.environ.get("TJIPTO_WORDING_PROVIDER", "").strip().casefold()
    api_key = os.environ.get("TJIPTO_WORDING_API_KEY", "").strip()
    model = os.environ.get("TJIPTO_WORDING_MODEL", "").strip()
    base_url = os.environ.get("TJIPTO_WORDING_BASE_URL", "").strip()
    if not provider or not api_key or not model:
        return None
    try:
        timeout = max(1.0, float(os.environ.get("TJIPTO_WORDING_TIMEOUT_SECONDS", "12")))
    except ValueError:
        return None
    if provider == "gemini":
        from tjipto.runtime.gemini import GeminiAnswerProvider

        endpoint = base_url or "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        if not _https(endpoint):
            return None
        return GeminiAnswerProvider(api_key, model=model, endpoint=endpoint, timeout=timeout)
    if provider == "openai_compatible":
        from tjipto.runtime.openai_compatible import OpenAICompatibleWordingProvider

        endpoint = base_url.rstrip("/") + "/chat/completions" if base_url else ""
        if not _https(endpoint):
            return None
        return OpenAICompatibleWordingProvider(api_key, model=model, endpoint=endpoint, timeout=timeout)
    return None


def valid_proposal(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    presentation = value.get("presentation")
    references = value.get("referenced_fact_ids")
    if presentation in {"direct", "grounded"} and isinstance(references, list) and all(isinstance(item, str) for item in references):
        return {"presentation": presentation, "referenced_fact_ids": tuple(references)}
    sentences = value.get("sentences")
    if set(value) != {"sentences"} or not isinstance(sentences, list) or not sentences:
        return None
    normalized = []
    for sentence in sentences:
        if not isinstance(sentence, dict) or set(sentence) != {"style", "referenced_fact_ids"}:
            return None
        style = sentence.get("style")
        refs = sentence.get("referenced_fact_ids")
        if style not in {"direct", "grounded"} or not isinstance(refs, list) or not refs or not all(isinstance(item, str) for item in refs):
            return None
        normalized.append({"style": style, "referenced_fact_ids": tuple(refs)})
    return {"sentences": tuple(normalized)}


def _https(value: str) -> bool:
    parsed = urlparse(value.format(model="model"))
    return parsed.scheme == "https" and bool(parsed.netloc)
