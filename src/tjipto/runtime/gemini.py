from __future__ import annotations

import json
import os
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class GeminiAnswerProvider:
    """Optional answer wording layer over already verified local evidence."""

    def __init__(self, api_key: str, *, model: str, endpoint: str, timeout: float = 12.0):
        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint.format(model=model)
        self._timeout = timeout

    @classmethod
    def from_environment(cls) -> GeminiAnswerProvider | None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None
        model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash").strip()
        endpoint = os.environ.get(
            "GEMINI_API_ENDPOINT",
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        ).strip()
        parsed_endpoint = urlparse(endpoint)
        if parsed_endpoint.scheme != "https" or not parsed_endpoint.netloc:
            return None
        try:
            timeout = float(os.environ.get("GEMINI_TIMEOUT_SECONDS", "12"))
        except ValueError:
            timeout = 12.0
        return cls(api_key, model=model, endpoint=endpoint, timeout=max(1.0, timeout))

    def answer(self, deterministic_answer: str, facts: tuple[dict, ...]) -> dict | None:
        if not deterministic_answer or not facts:
            return None
        prompt = (
            "Susun jawaban dengan mempertahankan deterministic_answer secara utuh dan hanya "
            "menambahkan frasa pengantar netral. Jangan menambah atau mengubah fakta. Kembalikan "
            "JSON sesuai schema.\n\n"
            + json.dumps({"deterministic_answer": deterministic_answer, "allowed_facts": facts}, ensure_ascii=False)
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
                        "answer": {"type": "STRING"},
                        "referenced_fact_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
                    },
                    "required": ["answer", "referenced_fact_ids"],
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
            if not isinstance(proposal, dict) or not isinstance(proposal.get("answer"), str):
                return None
            identifiers = proposal.get("referenced_fact_ids")
            if not isinstance(identifiers, list) or not all(isinstance(item, str) for item in identifiers):
                return None
            return {"answer": proposal["answer"].strip(), "referenced_fact_ids": tuple(identifiers)}
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            return None
