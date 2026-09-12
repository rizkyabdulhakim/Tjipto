from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tjipto.corpora.registry import CorpusRegistry
from tjipto.core.config import CorpusConfig
from tjipto.core.manifest import file_sha256, read_json, validate_manifest


JsonRow = Mapping[str, Any]
PageKey = tuple[str, int]


def _compact_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").replace("\u00ad", "")
    return "".join(re.findall(r"\w+", text.casefold()))


def _manifest_errors(final_dir: Path, manifest: Mapping[str, object]) -> list[str]:
    errors = list(validate_manifest(final_dir))
    files = manifest.get("files") or {}
    if not isinstance(files, Mapping):
        return [*errors, "invalid_manifest_files"]
    for rel, expected in files.items():
        path = final_dir / str(rel)
        if not path.exists():
            errors.append(f"missing_file:{rel}")
            continue
        if not isinstance(expected, Mapping):
            errors.append(f"invalid_file_manifest:{rel}")
            continue
        if path.stat().st_size != expected.get("bytes"):
            errors.append(f"bytes_mismatch:{rel}")
        if file_sha256(path) != expected.get("sha256"):
            errors.append(f"sha256_mismatch:{rel}")
    return errors


def _jsonl_keys(manifest: Mapping[str, object]) -> set[str]:
    files = manifest.get("files") or {}
    if not isinstance(files, Mapping):
        return set()
    return {
        str(record["logical_key"])
        for record in files.values()
        if isinstance(record, Mapping) and record.get("format") == "jsonl" and record.get("logical_key")
    }


def _artifact_key(count_name: str, jsonl_keys: set[str]) -> str | None:
    if count_name in jsonl_keys:
        return count_name
    if count_name.endswith("_records"):
        base = count_name.removesuffix("_records")
        for candidate in (f"{base}_registry", base):
            if candidate in jsonl_keys:
                return candidate
    return None


def _artifact_counts(config: CorpusConfig, manifest: Mapping[str, object]) -> tuple[dict[str, int], list[str]]:
    expected = manifest.get("counts") or {}
    if not isinstance(expected, Mapping):
        return {}, ["invalid_manifest_counts"]
    keys = _jsonl_keys(manifest)
    counts: dict[str, int] = {}
    errors: list[str] = []
    for count_name, expected_count in expected.items():
        artifact_key = _artifact_key(str(count_name), keys)
        if artifact_key is None:
            continue
        try:
            actual = len(config.jsonl(artifact_key))
        except (KeyError, FileNotFoundError, ValueError):
            actual = -1
        counts[str(count_name)] = actual
        if actual != expected_count:
            errors.append(f"count_mismatch:{count_name}:{actual}!={expected_count}")
    return counts, errors


def _load_pdf_sources(
    config: CorpusConfig,
    manifest: Mapping[str, object],
    errors: list[str],
) -> tuple[dict[str, JsonRow], dict[PageKey, str], dict[PageKey, tuple[float, float]]]:
    import fitz

    source_docs = {str(row["source_document_id"]): row for row in config.jsonl("source_documents")}
    source_files = manifest.get("source_files") or {}
    page_text: dict[PageKey, str] = {}
    page_sizes: dict[PageKey, tuple[float, float]] = {}
    for source_id, row in source_docs.items():
        path = config.source_path(str(row["path"]))
        if not path.exists():
            errors.append(f"missing_pdf:{source_id}")
            continue
        if file_sha256(path) != row["sha256"]:
            errors.append(f"source_sha256_mismatch:{source_id}")
        if not isinstance(source_files, Mapping) or source_files.get(row["path"]) != row["sha256"]:
            errors.append(f"manifest_source_sha256_mismatch:{source_id}")
        with fitz.open(path) as pdf:
            if row.get("page_count") and pdf.page_count != row["page_count"]:
                errors.append(f"page_count_mismatch:{source_id}")
            for page_number, page in enumerate(pdf, start=1):
                key = source_id, page_number
                page_text[key] = _compact_text(page.get_text())
                page_sizes[key] = page.rect.width, page.rect.height
    return source_docs, page_text, page_sizes


def _validate_pages(
    config: CorpusConfig, source_docs: Mapping[str, JsonRow], pdf_text: Mapping[PageKey, str], errors: list[str]
) -> set[PageKey]:
    pages = config.jsonl("pages")
    page_keys = {(str(row["source_document_id"]), int(row["page_number"])) for row in pages}
    for row in pages:
        key = str(row["source_document_id"]), int(row["page_number"])
        if key[0] not in source_docs:
            errors.append(f"page_unknown_source:{key[0]}:{key[1]}")
        elif key not in pdf_text:
            errors.append(f"page_out_of_range:{key[0]}:{key[1]}")
        elif _compact_text(str(row["text"])) not in pdf_text[key]:
            errors.append(f"page_text_mismatch:{key[0]}:{key[1]}")
    return page_keys


def _validate_structure(config: CorpusConfig, source_docs: Mapping[str, JsonRow], errors: list[str]) -> dict[str, JsonRow]:
    legal_units = {str(row["legal_unit_id"]): row for row in config.jsonl("legal_units")}
    for unit_id, row in legal_units.items():
        if row["source_document_id"] not in source_docs:
            errors.append(f"legal_unit_unknown_source:{unit_id}")
        for parent_id in row.get("parent_legal_unit_ids") or []:
            if parent_id not in legal_units:
                errors.append(f"legal_unit_unknown_parent:{unit_id}:{parent_id}")
    for row in config.jsonl("chunks"):
        if row["legal_unit_id"] not in legal_units:
            errors.append(f"chunk_unknown_legal_unit:{row['chunk_id']}")
    return legal_units


def _validate_evidence(
    config: CorpusConfig,
    source_docs: Mapping[str, JsonRow],
    legal_units: Mapping[str, JsonRow],
    page_keys: set[PageKey],
    page_sizes: Mapping[PageKey, tuple[float, float]],
    errors: list[str],
) -> dict[str, JsonRow]:
    evidence = {str(row["evidence_id"]): row for row in config.jsonl("evidence")}
    bboxes = {str(row["bbox_id"]): row for row in config.jsonl("bbox")}
    for evidence_id, row in evidence.items():
        if row["legal_unit_id"] not in legal_units:
            errors.append(f"evidence_unknown_legal_unit:{evidence_id}")
        if row["source_document_id"] not in source_docs:
            errors.append(f"evidence_unknown_source:{evidence_id}")
        for page_number in row.get("page_numbers") or []:
            if (str(row["source_document_id"]), int(page_number)) not in page_keys:
                errors.append(f"evidence_unknown_page:{evidence_id}:{page_number}")
        for bbox_id in row.get("bbox_refs") or []:
            if bbox_id not in bboxes:
                errors.append(f"evidence_unknown_bbox:{evidence_id}:{bbox_id}")
    _validate_bboxes(bboxes, evidence, page_keys, page_sizes, errors)
    return evidence


def _validate_bboxes(
    bboxes: Mapping[str, JsonRow],
    evidence: Mapping[str, JsonRow],
    page_keys: set[PageKey],
    page_sizes: Mapping[PageKey, tuple[float, float]],
    errors: list[str],
) -> None:
    referenced = {str(ref) for row in evidence.values() for ref in row.get("bbox_refs") or ()}
    for bbox_id, row in bboxes.items():
        key = str(row["source_document_id"]), int(row["page_number"])
        if bbox_id not in referenced:
            errors.append(f"bbox_orphan_geometry:{bbox_id}")
        if "evidence_id" in row and row.get("evidence_id") not in evidence:
            errors.append(f"bbox_unknown_evidence:{bbox_id}")
        if key not in page_keys:
            errors.append(f"bbox_unknown_page:{bbox_id}")
        elif key in page_sizes:
            width, height = page_sizes[key]
            if not (0 <= row["x0"] <= row["x1"] <= width and 0 <= row["y0"] <= row["y1"] <= height):
                errors.append(f"bbox_out_of_bounds:{bbox_id}")


def _validate_metadata(config: CorpusConfig, source_docs: Mapping[str, JsonRow], errors: list[str]) -> None:
    bbox_ids = {row["bbox_id"] for row in config.jsonl("metadata_grounding_registry")}
    for row in config.jsonl("metadata_grounding"):
        grounding_id = row["metadata_grounding_id"]
        if row["source_document_id"] not in source_docs:
            errors.append(f"metadata_grounding_unknown_source:{grounding_id}")
        for bbox_id in row.get("bbox_refs") or []:
            if bbox_id not in bbox_ids:
                errors.append(f"metadata_grounding_unknown_bbox:{grounding_id}:{bbox_id}")


def _validate_graph(config: CorpusConfig, evidence: Mapping[str, JsonRow], errors: list[str]) -> None:
    nodes = {str(row["node_id"]): row for row in config.jsonl("graph_nodes")}
    for node_id, row in nodes.items():
        if not row.get("source_pdf_path"):
            continue
        path = config.source_path(str(row["source_pdf_path"]))
        if not path.exists():
            errors.append(f"graph_node_missing_pdf:{node_id}")
        elif row.get("source_sha256") and file_sha256(path) != row["source_sha256"]:
            errors.append(f"graph_node_source_sha256_mismatch:{node_id}")

    for edge in config.jsonl("graph_edges"):
        _validate_graph_edge(edge, nodes, evidence, errors)


def _validate_graph_edge(
    edge: JsonRow,
    nodes: Mapping[str, JsonRow],
    evidence: Mapping[str, JsonRow],
    errors: list[str],
) -> None:
    edge_id = edge["edge_id"]
    if edge["source_id"] not in nodes or edge["target_id"] not in nodes:
        errors.append(f"graph_edge_unknown_endpoint:{edge_id}")
    if edge.get("relation_id") or edge.get("runtime_loadable") is None:
        return
    required = {
        "source_document_id": "graph_legal_edge_missing_source_document",
        "validation_status": "graph_runtime_edge_missing_validation",
        "confidence_policy": "graph_runtime_edge_missing_confidence",
    }
    for field, error_code in required.items():
        if not edge.get(field):
            errors.append(f"{error_code}:{edge_id}")
    if "evidence_ref" in edge:
        errors.append(f"graph_legacy_evidence_contract:{edge_id}")
    support_ids = edge.get("support_evidence_ids") or ()
    for evidence_id in support_ids:
        if evidence_id not in evidence:
            errors.append(f"graph_legal_edge_unknown_evidence_ref:{edge_id}:{evidence_id}")
    if edge.get("support_kind") == "exact_source_relation" and not support_ids:
        errors.append(f"graph_exact_relation_missing_evidence:{edge_id}")


def _validate_retrieval_units(config: CorpusConfig, evidence: Mapping[str, JsonRow], errors: list[str]) -> None:
    for row in config.jsonl("retrieval_units"):
        if row["evidence_id"] not in evidence:
            errors.append(f"retrieval_unit_unknown_evidence:{row['retrieval_unit_id']}")


def validate_corpus_ingestion_artifacts(corpus_id: str, repo_root: Path) -> dict:
    config = CorpusRegistry(repo_root).resolve(corpus_id)
    if config is None:
        return {"status": "invalid", "errors": ("missing_corpus",), "counts": {}}

    manifest = read_json(config.manifest_path)
    errors = _manifest_errors(config.manifest_path.parent, manifest)
    counts, count_errors = _artifact_counts(config, manifest)
    errors.extend(count_errors)
    try:
        source_docs, pdf_text, page_sizes = _load_pdf_sources(config, manifest, errors)
    except ImportError:
        return {"status": "invalid", "errors": ("missing_test_dependency:PyMuPDF",), "counts": counts}

    page_keys = _validate_pages(config, source_docs, pdf_text, errors)
    legal_units = _validate_structure(config, source_docs, errors)
    evidence = _validate_evidence(config, source_docs, legal_units, page_keys, page_sizes, errors)
    _validate_metadata(config, source_docs, errors)
    _validate_graph(config, evidence, errors)
    _validate_retrieval_units(config, evidence, errors)
    return {"status": "valid" if not errors else "invalid", "errors": tuple(errors), "counts": counts}
