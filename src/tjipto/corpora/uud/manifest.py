from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from tjipto.core.manifest import file_sha256, read_jsonl


ARTIFACT_FILES = (
    ("article_versions", "article_versions.jsonl"),
    ("bbox_registry", "bbox_registry.jsonl"),
    ("chunks", "chunks.jsonl"),
    ("document_metadata", "document_metadata.jsonl"),
    ("evidence_registry", "evidence_registry.jsonl"),
    ("excluded_records", "excluded_records.jsonl"),
    ("graph_edges", "graph_edges.jsonl"),
    ("graph_nodes", "graph_nodes.jsonl"),
    ("legal_units", "legal_units.jsonl"),
    ("metadata", "metadata.jsonl"),
    ("metadata_graph_edges", "metadata_graph_edges.jsonl"),
    ("metadata_grounding", "metadata_grounding.jsonl"),
    ("metadata_grounding_registry", "metadata_grounding_registry.jsonl"),
    ("pages", "pages.jsonl"),
    ("retrieval_units", "retrieval_units.jsonl"),
    ("source_conflicts", "source_conflicts.jsonl"),
    ("source_documents", "source_documents.jsonl"),
    ("source_integrity", "source_integrity.json"),
    ("validation_alignment_results", "validation_alignment_results.jsonl"),
    ("validation_exception_review_labels", "validation_exception_review_labels.jsonl"),
    ("validation_exceptions", "validation_exceptions.jsonl"),
    ("validation_report", "validation_report.json"),
    ("page_text_spans", "page_text_spans.jsonl"),
)

COUNT_FILES = (
    ("article_versions", "article_versions.jsonl"),
    ("bbox_records", "bbox_registry.jsonl"),
    ("chunks", "chunks.jsonl"),
    ("document_metadata", "document_metadata.jsonl"),
    ("evidence_records", "evidence_registry.jsonl"),
    ("excluded_records", "excluded_records.jsonl"),
    ("graph_edges", "graph_edges.jsonl"),
    ("graph_nodes", "graph_nodes.jsonl"),
    ("legal_units", "legal_units.jsonl"),
    ("metadata_assertions", "metadata.jsonl"),
    ("metadata_graph_edges", "metadata_graph_edges.jsonl"),
    ("metadata_grounding", "metadata_grounding.jsonl"),
    ("metadata_grounding_records", "metadata_grounding_registry.jsonl"),
    ("not_promoted_amends_edges", "not_promoted_amends_edges.jsonl"),
    ("pages", "pages.jsonl"),
    ("retrieval_units", "retrieval_units.jsonl"),
    ("source_conflicts", "source_conflicts.jsonl"),
    ("source_documents", "source_documents.jsonl"),
    ("validation_alignment_results", "validation_alignment_results.jsonl"),
    ("validation_exception_review_labels", "validation_exception_review_labels.jsonl"),
    ("validation_exceptions", "validation_exceptions.jsonl"),
    ("page_text_spans", "page_text_spans.jsonl"),
)

LEGACY_COUNTS = {
    "not_promoted_amends_edges": 8,
}

FIXTURES = {
    "eval_fixtures": "tests/fixtures/uud/eval_fixtures.jsonl",
    "graph_retrieval_eval_cases": "tests/fixtures/uud/graph_retrieval_eval_cases.jsonl",
    "graph_retrieval_eval_results": "tests/fixtures/uud/graph_retrieval_eval_results.jsonl",
    "graph_retrieval_traces": "tests/fixtures/uud/graph_retrieval_traces.jsonl",
    "orchestrator_eval_results": "tests/fixtures/uud/orchestrator_eval_results.jsonl",
    "orchestrator_eval_summary": "tests/fixtures/uud/orchestrator_eval_summary.json",
    "retrieval_metrics_baseline": "tests/fixtures/uud/retrieval_metrics_baseline.json",
    "retrieval_summary_baseline": "tests/fixtures/uud/retrieval_summary_baseline.json",
}


def build_manifest(source_documents: dict[str, dict]) -> dict:
    manifest = {
        "article_versions": "article_versions.jsonl",
        "bbox_registry": "bbox_registry.jsonl",
        "chunks": "chunks.jsonl",
        "corpus_id": "uud",
        "counts": {},
        "document_metadata": "document_metadata.jsonl",
        "evidence_registry": "evidence_registry.jsonl",
        "excluded_records": "excluded_records.jsonl",
        "files": {filename: {} for _, filename in ARTIFACT_FILES},
        "fixtures": FIXTURES,
        "graph_edges": "graph_edges.jsonl",
        "graph_nodes": "graph_nodes.jsonl",
        "legal_units": "legal_units.jsonl",
        "metadata": "metadata.jsonl",
        "metadata_graph_edges": "metadata_graph_edges.jsonl",
        "metadata_grounding": "metadata_grounding.jsonl",
        "metadata_grounding_registry": "metadata_grounding_registry.jsonl",
        "pages": "pages.jsonl",
        "retrieval_units": "retrieval_units.jsonl",
        "schema_version": 1,
        "source_conflicts": "source_conflicts.jsonl",
        "source_documents": "source_documents.jsonl",
        "source_files": {
            row["path"]: row["sha256"]
            for row in sorted(source_documents.values(), key=lambda item: item["path"])
        },
        "source_integrity": "source_integrity.json",
        "status": "final",
        "validation_alignment_results": "validation_alignment_results.jsonl",
        "validation_exception_review_labels": "validation_exception_review_labels.jsonl",
        "validation_exceptions": "validation_exceptions.jsonl",
        "validation_report": "validation_report.json",
        "page_text_spans": "page_text_spans.jsonl",
    }
    return manifest


def write_json(path: Path, data: dict) -> None:
    path.write_bytes((json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8"))


def refresh_manifest(final_dir: Path, manifest: dict) -> None:
    counts = manifest.setdefault("counts", {})
    for key, filename in COUNT_FILES:
        path = final_dir / filename
        if path.exists():
            counts[key] = len(read_jsonl(path))
        elif key in LEGACY_COUNTS:
            counts[key] = LEGACY_COUNTS[key]
    for rel in manifest["files"]:
        path = final_dir / rel
        if path.exists():
            manifest["files"][rel]["bytes"] = path.stat().st_size
            manifest["files"][rel]["sha256"] = file_sha256(path)
    write_json(final_dir / "manifest.json", manifest)


def atomic_promote_artifacts(
    *,
    final_dir: Path,
    build: Callable[[Path], None],
    validate: Callable[[Path], tuple[str, ...]],
) -> None:
    final_dir = final_dir.resolve()
    with tempfile.TemporaryDirectory(prefix=".uud-stage-", dir=final_dir.parent) as tmp:
        tmp_dir = Path(tmp)
        stage_dir = tmp_dir / "stage"
        snapshot_dir = tmp_dir / "snapshot"
        shutil.copytree(final_dir, stage_dir)
        shutil.copytree(final_dir, snapshot_dir)
        build(stage_dir)
        errors = validate(stage_dir)
        if errors:
            raise ValueError(";".join(errors))
        promoted: list[str] = []
        try:
            for path in sorted(stage_dir.iterdir()):
                if path.is_file():
                    target = final_dir / path.name
                    path.replace(target)
                    promoted.append(path.name)
        except Exception:
            for name in promoted:
                (snapshot_dir / name).replace(final_dir / name)
            raise
