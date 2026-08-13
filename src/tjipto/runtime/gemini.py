from __future__ import annotations

import json
from urllib.request import Request, urlopen

from tjipto.runtime.wording import valid_proposal


class GeminiAnswerProvider:
    """Optional answer wording layer over already verified local evidence."""

    def __init__(self, api_key: str, *, model: str, endpoint: str, timeout: float = 12.0):
        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint.format(model=model)
        self._timeout = timeout

    def propose(self, deterministic_answer: str) -> dict[str, object] | None:
        if not deterministic_answer:
            return None
        prompt = (
            "Kembalikan JSON saja. Susun ulang hanya fakta yang diberikan menjadi kalimat alami. "
            "Jangan menambah atau mengubah fakta, angka, rujukan, modalitas, atau negasi. "
            "Setiap kalimat wajib memiliki text dan referenced_fact_ids.\n\n"
            + json.dumps({"deterministic_answer": deterministic_answer}, ensure_ascii=False)
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "candidateCount": 1,
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "sentences": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "text": {"type": "STRING"},
                                    "referenced_fact_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
                                },
                                "required": ["text", "referenced_fact_ids"],
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
