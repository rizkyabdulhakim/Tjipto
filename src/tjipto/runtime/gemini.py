from __future__ import annotations

import json
from urllib.request import Request, urlopen

from tjipto.runtime.wording import answer_prompt, valid_proposal


class GeminiAnswerProvider:
    """Optional answer wording layer over already verified local evidence."""

    def __init__(self, api_key: str, *, model: str, endpoint: str, timeout: float = 12.0):
        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint.format(model=model)
        self._timeout = timeout

    def propose(self, verified_context: str) -> dict[str, object] | None:
        if not verified_context:
            return None
        payload = {
            "contents": [{"role": "user", "parts": [{"text": answer_prompt(verified_context)}]}],
            "generationConfig": {
                "candidateCount": 1,
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "sentences": {
                            "type": "ARRAY",
                            "maxItems": 4,
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "text": {
                                        "type": "STRING",
                                        "description": "Concise Indonesian prose preserving at least two distinctive terms from every listed claim.",
                                    },
                                    "claim_ids": {
                                        "type": "ARRAY",
                                        "items": {"type": "STRING"},
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
        }
        request = Request(
            self._endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self._api_key},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:  # nosec B310 - endpoint is restricted to HTTPS above.
                result = json.load(response)
            parts = result["candidates"][0]["content"]["parts"]
            proposal = json.loads("".join(str(part.get("text") or "") for part in parts))
            return valid_proposal(proposal)
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            return None
