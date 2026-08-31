from __future__ import annotations

import json
from urllib.request import Request, urlopen

from tjipto.core.external_llm import OPENAI_COMPATIBLE_USER_AGENT, openai_compatible_latency_options
from tjipto.runtime.wording import answer_prompt, valid_proposal


class OpenAICompatibleWordingProvider:
    """Small adapter for a schema-capable OpenAI-compatible endpoint."""

    def __init__(self, api_key: str, *, model: str, endpoint: str, timeout: float = 12.0):
        self._api_key, self._model, self._endpoint, self._timeout = api_key, model, endpoint, timeout

    def propose(self, verified_context: str) -> dict[str, object] | None:
        payload = {
            "model": self._model,
            "temperature": 0,
            "stream": False,
            **openai_compatible_latency_options(self._model, self._endpoint),
            "messages": [{"role": "user", "content": answer_prompt(verified_context)}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "tjipto_verified_wording",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "sentences": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 4,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "text": {
                                            "type": "string",
                                            "minLength": 1,
                                            "description": "Concise Indonesian prose preserving at least two distinctive terms from every listed claim.",
                                        },
                                        "claim_ids": {
                                            "type": "array",
                                            "minItems": 1,
                                            "items": {"type": "string"},
                                            "description": "Exact verified claim IDs supporting every fact in this sentence.",
                                        },
                                    },
                                    "required": ["text", "claim_ids"],
                                },
                            },
                        },
                        "required": ["sentences"],
                    },
                },
            },
        }
        request = Request(
            self._endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": OPENAI_COMPATIBLE_USER_AGENT,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:  # nosec B310 - factory permits HTTPS only.
                body = json.load(response)
            return valid_proposal(json.loads(str(body["choices"][0]["message"]["content"])))
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            return None
