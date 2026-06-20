from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

from tjipto.corpora.registry import CorpusRegistry
from tjipto.core.manifest import read_json, read_jsonl, validate_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").replace("\u00ad", "")
    return "".join(re.findall(r"\w+", text.casefold()))


def validate_uud_ingestion_artifacts(repo_root: Path) -> dict:
    config = CorpusRegistry(repo_root).resolve("uud")
    errors: list[str] = []
    if config is None:
        return {"status": "invalid", "errors": ["missing_uud_corpus"]}

    final_dir = config.manifest_path.parent
    errors.extend(validate_manifest(final_dir))
    manifest = read_json(config.manifest_path)

    try:
        import fitz
    except ImportError:
        return {"status": "invalid", "errors": ["missing_test_dependency:PyMuPDF"]}

    source_docs = {row["source_document_id"]: row for row in config.jsonl("source_documents")}
    pdfs = {}
    page_text = {}
    for source_id, row in source_docs.items():
        path = repo_root / row["path"]
        if not path.exists():
            errors.append(f"missing_pdf:{source_id}")
            continue
        if _sha256(path) != row["sha256"]:
            errors.append(f"source_sha256_mismatch:{source_id}")
        if manifest.get("source_files", {}).get(row["path"]) != row["sha256"]:
            errors.append(f"manifest_source_sha256_mismatch:{source_id}")
        pdf = fitz.open(path)
        pdfs[source_id] = pdf
        if row.get("page_count") and pdf.page_count != row["page_count"]:
            errors.append(f"page_count_mismatch:{source_id}")
        page_text[source_id] = {i: _compact_text(pdf[i - 1].get_text()) for i in range(1, pdf.page_count + 1)}

    pages = config.jsonl("pages")
    page_keys = {(row["source_document_id"], row["page_number"]) for row in pages}
    pages_by_key = {(row["source_document_id"], row["page_number"]): _compact_text(row["text"]) for row in pages}
    for source_id, page_number in page_keys:
        if source_id not in source_docs:
            errors.append(f"page_unknown_source:{source_id}:{page_number}")
        elif page_number not in page_text.get(source_id, {}):
            errors.append(f"page_out_of_range:{source_id}:{page_number}")
        elif pages_by_key[(source_id, page_number)] not in page_text[source_id][page_number]:
            errors.append(f"page_text_mismatch:{source_id}:{page_number}")

    legal_units = {row["legal_unit_id"]: row for row in config.jsonl("legal_units")}
    for row in legal_units.values():
        if row["source_document_id"] not in source_docs:
            errors.append(f"legal_unit_unknown_source:{row['legal_unit_id']}")
        for parent_id in row.get("parent_legal_unit_ids") or []:
            if parent_id not in legal_units:
                errors.append(f"legal_unit_unknown_parent:{row['legal_unit_id']}:{parent_id}")

    chunks = {row["chunk_id"]: row for row in config.jsonl("chunks")}
    for row in chunks.values():
        if row["legal_unit_id"] not in legal_units:
            errors.append(f"chunk_unknown_legal_unit:{row['chunk_id']}")

    evidence = {row["evidence_id"]: row for row in config.jsonl("evidence")}
    bboxes = {row["bbox_id"]: row for row in config.jsonl("bbox")}
    for row in evidence.values():
        if row["legal_unit_id"] not in legal_units:
            errors.append(f"evidence_unknown_legal_unit:{row['evidence_id']}")
        if row["source_document_id"] not in source_docs:
            errors.append(f"evidence_unknown_source:{row['evidence_id']}")
        for page_number in row.get("page_numbers") or []:
            if (row["source_document_id"], page_number) not in page_keys:
                errors.append(f"evidence_unknown_page:{row['evidence_id']}:{page_number}")
        for bbox_id in row.get("bbox_refs") or []:
            if bbox_id not in bboxes:
                errors.append(f"evidence_unknown_bbox:{row['evidence_id']}:{bbox_id}")
    for row in bboxes.values():
        source_id = row.get("source_document_id")
        page_number = row.get("page_number")
        if row.get("evidence_id") not in evidence:
            errors.append(f"bbox_unknown_evidence:{row.get('bbox_id')}")
        if (source_id, page_number) not in page_keys:
            errors.append(f"bbox_unknown_page:{row.get('bbox_id')}")
        elif source_id in pdfs:
            rect = pdfs[source_id][page_number - 1].rect
            if not (0 <= row["x0"] <= row["x1"] <= rect.width and 0 <= row["y0"] <= row["y1"] <= rect.height):
                errors.append(f"bbox_out_of_bounds:{row.get('bbox_id')}")

    metadata_bbox_ids = {row["bbox_id"] for row in config.jsonl("metadata_grounding_registry")}
    for row in config.jsonl("metadata_grounding"):
        if row["source_document_id"] not in source_docs:
            errors.append(f"metadata_grounding_unknown_source:{row['metadata_grounding_id']}")
        for bbox_id in row.get("bbox_refs") or []:
            if bbox_id not in metadata_bbox_ids:
                errors.append(f"metadata_grounding_unknown_bbox:{row['metadata_grounding_id']}:{bbox_id}")

    graph_nodes = {row["node_id"]: row for row in config.jsonl("graph_nodes")}
    for node_id, row in graph_nodes.items():
        if row.get("source_pdf_path"):
            path = repo_root / row["source_pdf_path"]
            if not path.exists():
                errors.append(f"graph_node_missing_pdf:{node_id}")
            elif row.get("source_sha256") and _sha256(path) != row["source_sha256"]:
                errors.append(f"graph_node_source_sha256_mismatch:{node_id}")
    for edge in config.jsonl("graph_edges"):
        if edge["source_id"] not in graph_nodes or edge["target_id"] not in graph_nodes:
            errors.append(f"graph_edge_unknown_endpoint:{edge['edge_id']}")

    for row in config.jsonl("retrieval_units"):
        if row["evidence_id"] not in evidence:
            errors.append(f"retrieval_unit_unknown_evidence:{row['retrieval_unit_id']}")

    return {
        "status": "valid" if not errors else "invalid",
        "errors": tuple(errors),
        "counts": {
            "source_documents": len(source_docs),
            "pages": len(pages),
            "legal_units": len(legal_units),
            "chunks": len(chunks),
            "evidence": len(evidence),
            "bbox": len(bboxes),
            "graph_nodes": len(graph_nodes),
            "graph_edges": len(config.jsonl("graph_edges")),
            "retrieval_units": len(config.jsonl("retrieval_units")),
        },
    }
