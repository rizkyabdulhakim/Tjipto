from __future__ import annotations

import platform

from tjipto.ingestion.pdf.text_spans import NORMALIZATION_CONTRACT


def extractor_fingerprint() -> dict[str, object]:
    """Stable description of the PDF extraction contract used by artifacts."""
    try:
        import fitz
    except ImportError as error:  # pragma: no cover - build environments install PyMuPDF
        raise RuntimeError("pymupdf_unavailable") from error
    return {
        "python": platform.python_version(),
        "pymupdf": getattr(fitz, "VersionBind", "unknown"),
        "mupdf": getattr(fitz, "VersionFitz", "unknown"),
        "rawdict_sort": True,
        "words_sort": True,
        "page_box_basis": "media_box",
        "unicode_normalization": NORMALIZATION_CONTRACT,
    }
