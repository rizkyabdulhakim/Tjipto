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

    def answer(self, query: str, evidence: tuple[dict, ...]) -> str | None:
        context = "\n\n".join(
            f"Evidence {index}: {row.get('quoted_text') or row.get('citation') or ''}"
            for index, row in enumerate(evidence, start=1)
        ).strip()
        if not context:
            return None
        prompt = (
            "Jawab pertanyaan pengguna hanya berdasarkan konteks evidence berikut. "
            "Jika konteks tidak cukup, katakan bukti tidak cukup. Jangan membuat fakta, "
            "tanggal, pasal, citation, page, BBox, atau source baru. Jangan menyebut ID "
            "internal.\n\nPertanyaan: "
            f"{query}\n\nKonteks evidence:\n{context}"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"candidateCount": 1, "temperature": 0},
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
            text = "".join(str(part.get("text") or "") for part in parts).strip()
            return text or None
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            return None
