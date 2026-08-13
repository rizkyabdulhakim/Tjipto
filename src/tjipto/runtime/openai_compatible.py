from __future__ import annotations

import json
from urllib.request import Request, urlopen

from tjipto.runtime.wording import valid_proposal


class OpenAICompatibleWordingProvider:
    """Small adapter for a schema-capable OpenAI-compatible endpoint."""

    def __init__(self, api_key: str, *, model: str, endpoint: str, timeout: float = 12.0):
        self._api_key, self._model, self._endpoint, self._timeout = api_key, model, endpoint, timeout

    def propose(self, deterministic_answer: str) -> dict[str, object] | None:
        payload = {
            "model": self._model,
            "temperature": 0,
            "messages": [{"role": "user", "content": _prompt(deterministic_answer)}],
            "response_format": {"type": "json_object"},
        }
        request = Request(
            self._endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:  # nosec B310 - factory permits HTTPS only.
                body = json.load(response)
            return valid_proposal(json.loads(str(body["choices"][0]["message"]["content"])))
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            return None


def _prompt(answer: str) -> str:
    return (
        "Return JSON only. Rewrite only the provided verified facts into natural sentences. "
        "Do not add or change facts, numbers, references, modality, or negation. "
        "Return {sentences:[{text:string,referenced_fact_ids:string[]}]}; every sentence must reference provided facts.\n"
        + json.dumps({"deterministic_answer": answer}, ensure_ascii=False)
    )
