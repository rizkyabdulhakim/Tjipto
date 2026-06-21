from __future__ import annotations


def viewer_payload(corpus_id: str, evidence: dict, bboxes: list[dict]) -> dict:
    return {
        "status": "viewer_payload_ready",
        "corpus_id": corpus_id,
        "evidence_id": evidence["evidence_id"],
        "legal_unit_id": evidence.get("legal_unit_id"),
        "source_document_id": evidence.get("source_document_id"),
        "citation": evidence["citation"],
        "quoted_text": evidence["quoted_text"],
        "source_pdf_path": evidence["source_pdf_path"],
        "source_sha256": evidence["source_sha256"],
        "page_numbers": evidence["page_numbers"],
        "bbox_count": len(bboxes),
        "bbox_rectangles": tuple(bboxes),
        "rendering_available": False,
    }
