from __future__ import annotations


def viewer_payload(evidence: dict, bboxes: list[dict]) -> dict:
    return {
        "status": "viewer_payload_ready",
        "evidence_id": evidence["evidence_id"],
        "citation": evidence["citation"],
        "quoted_text": evidence["quoted_text"],
        "source_pdf_path": evidence["source_pdf_path"],
        "source_sha256": evidence["source_sha256"],
        "page_numbers": evidence["page_numbers"],
        "bbox_count": len(bboxes),
        "bbox_rectangles": tuple(bboxes),
    }
