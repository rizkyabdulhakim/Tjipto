"""Shared, provider-neutral configuration for optional external LLM calls."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Generic, Protocol, TypeVar
from urllib.parse import urlparse


RequestT = TypeVar("RequestT", contravariant=True)
ResponseT = TypeVar("ResponseT", covariant=True)
OPENAI_COMPATIBLE_USER_AGENT = "Tjipto"


class ProposalProvider(Protocol[RequestT, ResponseT]):
    def propose(self, request: RequestT) -> ResponseT: ...


@dataclass(frozen=True)
class ExternalLLMConfig:
    provider: str
    api_key: str
    model: str
    base_url: str
    timeout: float


def external_llm_config(scope: str) -> ExternalLLMConfig | None:
    """Resolve capability overrides first, then the shared LLM owner."""
    prefix = f"TJIPTO_{scope}_"

    def value(name: str, default: str = "") -> str:
        return os.environ.get(prefix + name, "").strip() or os.environ.get(f"TJIPTO_LLM_{name}", default).strip()

    provider = value("PROVIDER").casefold()
    api_key = value("API_KEY")
    model = value("MODEL")
    if not provider or not api_key or not model:
        return None
    try:
        timeout = max(1.0, float(value("TIMEOUT_SECONDS", "12")))
    except ValueError:
        return None
    return ExternalLLMConfig(provider, api_key, model, value("BASE_URL").rstrip("/"), timeout)


def fallback_external_llm_config() -> ExternalLLMConfig | None:
    """Resolve one shared fallback for both planner and answer wording."""
    prefix = "TJIPTO_FALLBACK_LLM_"

    def value(name: str, default: str = "") -> str:
        return os.environ.get(prefix + name, default).strip()

    provider, api_key, model = value("PROVIDER").casefold(), value("API_KEY"), value("MODEL")
    if not provider or not api_key or not model:
        return None
    try:
        timeout = max(1.0, float(value("TIMEOUT_SECONDS", "12")))
    except ValueError:
        return None
    return ExternalLLMConfig(provider, api_key, model, value("BASE_URL").rstrip("/"), timeout)


@dataclass(frozen=True)
class FallbackProposalProvider(Generic[RequestT, ResponseT]):
    primary: ProposalProvider[RequestT, ResponseT]
    fallback: ProposalProvider[RequestT, ResponseT]

    def propose(self, request: RequestT) -> ResponseT:
        try:
            result = self.primary.propose(request)
        except Exception:
            result = None
        return result if result is not None else self.fallback.propose(request)


def openai_compatible_latency_options(model: str, endpoint: str) -> dict[str, str]:
    """Use Gemini's smallest supported thinking level for bounded JSON tasks."""
    host = urlparse(endpoint).hostname
    if host == "generativelanguage.googleapis.com" and model.casefold().startswith("gemini-"):
        return {"reasoning_effort": "minimal"}
    return {}


def is_allowed_llm_endpoint(endpoint: str) -> bool:
    """Require HTTPS, except for explicitly local loopback development endpoints."""
    parsed = urlparse(endpoint.format(model="model"))
    if not parsed.netloc:
        return False
    if parsed.scheme == "https":
        return True
    return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}


__all__ = [
    "ExternalLLMConfig",
    "FallbackProposalProvider",
    "OPENAI_COMPATIBLE_USER_AGENT",
    "ProposalProvider",
    "external_llm_config",
    "fallback_external_llm_config",
    "is_allowed_llm_endpoint",
    "openai_compatible_latency_options",
]
