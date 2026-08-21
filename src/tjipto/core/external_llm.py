"""Shared, provider-neutral configuration for optional external LLM calls."""

from __future__ import annotations

from dataclasses import dataclass
import os


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


__all__ = ["ExternalLLMConfig", "external_llm_config"]
