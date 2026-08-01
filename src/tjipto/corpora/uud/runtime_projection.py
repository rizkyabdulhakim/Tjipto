from __future__ import annotations

from tjipto.contracts.relations import is_relevance_relation, is_query_relation
from tjipto.corpora.uud.policy.source_text import project_source_text_rows

RUNTIME_ARTIFACTS = (
    "evidence_registry",
    "bbox_registry",
    "legal_units",
    "chunks",
    "retrieval_units",
    "graph_edges",
    "source_documents",
    "page_text_spans",
    "document_metadata",
    "metadata_grounding",
    "metadata_grounding_registry",
    "source_conflicts",
)

_GRAPH_EDGE_FIELDS = (
    "edge_id", "source_id", "target_id", "edge_type", "relation_type", "relation_id",
    "runtime_loadable", "support_kind", "support_relation_ids", "support_evidence_ids", "derived_from_edge_id",
    "text_span_ids", "bbox_refs", "source_role", "temporal_context", "relation_projection",
)

_SOURCE_SPAN_FIELDS = (
    "raw_source_span_id",
    "source_support_id",
    "source_document_id",
    "source_sha256",
    "source_pdf_path",
    "source_role",
    "page_number",
    "extraction_order",
    "raw_stream_id",
    "raw_text_start",
    "raw_text_end",
    "raw_text",
    "semantic_text",
    "semantic_exact_quote",
    "text_span_id",
    "x0",
    "y0",
    "x1",
    "y1",
    "legal_text",
    "citation_eligible",
    "relevant_quote_eligible",
    "viewer_eligible",
    "default_highlight_eligible",
    "viewer_highlightable",
    "classification",
    "disposition_reason",
    "semantic_text_span_id",
    "semantic_classification",
    "semantic_join_status",
    "temporal_context",
    "disposition",
    "legal_force",
    "capabilities",
    "legal_answer_eligible",
    "source_answer_eligible",
    "legal_citation_eligible",
    "source_citation_eligible",
    "abstention_reason",
    "target_legal_unit_id",
    "annotation_target_basis",
)

_RUNTIME_PAGE_SPAN_FIELDS = (
    "text_span_id", "source_document_id", "source_pdf_path", "source_sha256", "source_role", "temporal_context",
    "page_number", "text", "exact_quote", "stream_id", "text_start", "text_end", "text_prefix", "text_suffix",
    # Span coordinates retain source line layout without retaining every
    # word-level geometry row in the runtime snapshot.
    "x0", "y0", "x1", "y1", "bbox_precision", "viewer_highlightable", "object_role",
)


def build_runtime_projection(**artifacts: list[dict]) -> dict:
    """The sole runtime payload; release artifacts remain the audit authority."""
    rows = {name: list(artifacts.get(name, ())) for name in RUNTIME_ARTIFACTS}
    rows["page_text_spans"] = [
        {key: row[key] for key in _RUNTIME_PAGE_SPAN_FIELDS if key in row}
        for row in artifacts.get("page_text_spans", ())
    ]
    rows["graph_edges"] = [
        {key: row[key] for key in _GRAPH_EDGE_FIELDS if key in row}
        for row in artifacts.get("graph_edges", ())
        if row.get("edge_type") in {"HAS_FINAL_EVIDENCE", "PAGE_GROUNDED_AT"}
        or (
            row.get("runtime_loadable") is True
            and (is_relevance_relation(row.get("edge_type")) or is_query_relation(row.get("edge_type")))
        )
    ]
    refs = _bbox_refs(rows.values())
    words = []
    for word in artifacts.get("word_bboxes", ()):
        word_id = word.get("word_bbox_id")
        characters = [char for char in word.get("characters") or () if char.get("character_bbox_id") in refs]
        if word_id not in refs and not characters:
            continue
        compact = {key: value for key, value in word.items() if key != "characters"}
        if characters:
            compact["characters"] = characters
        words.append(compact)
    rows["word_bboxes"] = words
    source_rows = project_source_text_rows(
        artifacts.get("raw_source_spans", ()),
        artifacts.get("page_text_spans", ()),
    )
    rows["raw_source_spans"] = [
        {key: row.get(key) for key in _SOURCE_SPAN_FIELDS if key in row}
        for row in source_rows
    ]
    return {"schema": 2, "artifacts": rows}


def _bbox_refs(artifact_sets) -> set[str]:
    refs: set[str] = set()
    for rows in artifact_sets:
        for row in rows:
            _add_bbox_refs(row, refs)
    return refs


def _add_bbox_refs(value, refs: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if "bbox" in key and isinstance(item, (list, tuple)):
                refs.update(ref for ref in item if isinstance(ref, str))
            elif isinstance(item, (dict, list, tuple)):
                _add_bbox_refs(item, refs)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _add_bbox_refs(item, refs)
