from __future__ import annotations

import re

from .config import CorpusConfig
from .manifest import read_json


def validate_counts(final_dir) -> dict[str, int]:
    manifest_path = final_dir / "manifest.json"
    manifest = read_json(manifest_path)
    config = CorpusConfig(manifest["corpus_id"], manifest_path, manifest)
    return {
        "source_documents": len(config.jsonl("source_documents")),
        "evidence_records": len(config.jsonl("evidence")),
        "bbox_records": len(config.jsonl("bbox")),
        "graph_nodes": len(config.jsonl("graph_nodes")),
        "graph_edges": len(config.jsonl("graph_edges")),
    }


def validate_text_provenance(config: CorpusConfig, header_stripper=None) -> dict:
    page_text = {(page["source_document_id"], page["page_number"]): page["text"] for page in config.jsonl("pages")}
    legal_units = config.jsonl("legal_units")
    source_by_legal_unit = {row["legal_unit_id"]: row["source_document_id"] for row in legal_units}
    evidence_ids = {row["legal_unit_id"] for row in config.jsonl("evidence")}
    results = {
        "legal_units": _validate_rows(legal_units, page_text, evidence_ids, header_stripper, source_by_legal_unit),
        "chunks": _validate_rows(config.jsonl("chunks"), page_text, evidence_ids, header_stripper, source_by_legal_unit),
    }
    results["status"] = "pass" if all(part["needs_review"] == 0 for part in results.values()) else "needs_review"
    return results


def _validate_rows(
    rows: list[dict],
    page_text: dict,
    evidence_ids: set[str],
    header_stripper,
    source_by_legal_unit: dict,
) -> dict:
    counts = {
        "total": len(rows),
        "raw_pdf_match": 0,
        "normalized_pdf_match": 0,
        "header_stripped_pdf_match": 0,
        "evidence_grounded_match": 0,
        "needs_review": 0,
    }
    for row in rows:
        pdf_text = _row_pages(row, page_text, source_by_legal_unit)
        raw_match = _normalized_text(row["text"]) in _normalized_text(pdf_text)
        stripped_text = header_stripper(pdf_text) if header_stripper else pdf_text
        stripped_match = _normalized_text(row["text"]) in _normalized_text(stripped_text)
        counts["raw_pdf_match"] += int(raw_match)
        counts["normalized_pdf_match"] += int(raw_match)
        counts["header_stripped_pdf_match"] += int(stripped_match)
        counts["evidence_grounded_match"] += int(row.get("legal_unit_id") in evidence_ids)
        counts["needs_review"] += int(not stripped_match)
    counts["status"] = "pass" if counts["needs_review"] == 0 else "needs_review"
    return counts


def _row_pages(row: dict, page_text: dict, source_by_legal_unit: dict) -> str:
    source_document_id = row.get("source_document_id") or source_by_legal_unit.get(row.get("legal_unit_id"))
    if "page_start" in row:
        start, end = row["page_start"], row["page_end"]
    else:
        page_range = row["page_range"]
        start, end = page_range["start_page_number"], page_range["end_page_number"]
    return "\n".join(page_text.get((source_document_id, page), "") for page in range(start, end + 1))


def _normalized_text(text: str) -> str:
    text = (text or "").replace("\u00ad", "").replace("Â­", "").replace("Ã‚Â­", "")
    return re.sub(r"\s+", " ", text).strip().casefold()
