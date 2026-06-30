from __future__ import annotations

import hashlib
from pathlib import Path


def pdf_content_fingerprint(path: Path, markers: tuple[str, ...]) -> dict:
    try:
        import fitz
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required to inspect PDF source documents") from error

    doc = fitz.open(path)
    first_page_text = doc[0].get_text("text")
    last_page_text = doc[doc.page_count - 1].get_text("text")
    all_text = "\n".join(doc[index].get_text("text") for index in range(doc.page_count))
    return {
        "content_fingerprint_status": "recorded",
        "first_page_text_sha256": _sha256_text(first_page_text),
        "key_marker_presence": {marker: marker in all_text for marker in markers},
        "last_page_text_sha256": _sha256_text(last_page_text),
        "page_count": doc.page_count,
    }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
