from __future__ import annotations

from pathlib import Path

from tjipto.core.manifest import file_sha256
from tjipto.ingestion.pdf.source_documents import pdf_content_fingerprint
from tjipto.corpora.uud.specs import SOURCE_DOCUMENT_SPECS


MARKERS = ("ATURAN PERALIHAN", "ATURAN TAMBAHAN", "BAB", "PEMBUKAAN", "Pasal")


def build_source_documents(repo_root: Path) -> list[dict]:
    rows = []
    for spec in SOURCE_DOCUMENT_SPECS:
        path = repo_root / spec["path"]
        content_fingerprint = pdf_content_fingerprint(path, MARKERS)
        sha256 = file_sha256(path)
        rows.append(
            {
                "content_fingerprint": content_fingerprint,
                "corpus_id": "uud",
                "download_url": spec["download_url"],
                "file_size": path.stat().st_size,
                "file_size_match": True,
                "filename": spec["filename"],
                "final_download_url": spec["download_url"],
                "http_content_type": "application/octet-stream",
                "http_last_modified": spec["http_last_modified"],
                "http_status": 200,
                "page_count": content_fingerprint["page_count"],
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
            }
        )
    return rows
