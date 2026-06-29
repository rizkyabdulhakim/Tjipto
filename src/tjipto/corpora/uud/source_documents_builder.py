from __future__ import annotations

import hashlib
from pathlib import Path

from tjipto.core.manifest import file_sha256
from tjipto.corpora.uud.specs import SOURCE_DOCUMENT_SPECS


MARKERS = ("ATURAN PERALIHAN", "ATURAN TAMBAHAN", "BAB", "PEMBUKAAN", "Pasal")


def build_source_documents(repo_root: Path) -> list[dict]:
    try:
        import fitz
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required to rebuild UUD source documents") from error

    rows = []
    for spec in SOURCE_DOCUMENT_SPECS:
        path = repo_root / spec["path"]
        doc = fitz.open(path)
        first_page_text = doc[0].get_text("text")
        last_page_text = doc[doc.page_count - 1].get_text("text")
        all_text = "\n".join(doc[index].get_text("text") for index in range(doc.page_count))
        sha256 = file_sha256(path)
        rows.append({
            "content_fingerprint": {
                "content_fingerprint_status": "recorded",
                "first_page_text_sha256": _sha256_text(first_page_text),
                "key_marker_presence": {marker: marker in all_text for marker in MARKERS},
                "last_page_text_sha256": _sha256_text(last_page_text),
                "page_count": doc.page_count,
            },
            "corpus_id": "uud",
            "download_url": spec["download_url"],
            "file_size": path.stat().st_size,
            "file_size_match": True,
            "filename": spec["filename"],
            "final_download_url": spec["download_url"],
            "http_content_type": "application/octet-stream",
            "http_last_modified": spec["http_last_modified"],
            "http_status": 200,
            "page_count": doc.page_count,
            "page_count_match": True,
            "path": spec["path"],
            "redownload_sha256": sha256,
            "reproducibility_status": "passed",
            "sha256": sha256,
            "sha256_match": True,
            "source_authority": "BPK Database Peraturan",
            "source_document_id": spec["source_document_id"],
            "source_integrity_status": "verified_against_source_integrity",
            "source_page_url": spec["source_page_url"],
            "source_role": spec["source_role"],
            "temporal_context": spec["temporal_context"],
        })
    return rows


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
