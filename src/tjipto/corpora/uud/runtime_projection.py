from __future__ import annotations


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
    "document_relations",
    "article_amendment_relations",
    "source_conflicts",
)

_GRAPH_EDGE_FIELDS = ("source_id", "target_id", "edge_type")

_SOURCE_SPAN_FIELDS = (
    "source_support_id",
    "source_document_id",
    "source_sha256",
    "source_pdf_path",
    "source_role",
    "page_number",
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
)


def build_runtime_projection(**artifacts: list[dict]) -> dict:
    """The sole runtime payload; release artifacts remain the audit authority."""
    rows = {name: list(artifacts.get(name, ())) for name in RUNTIME_ARTIFACTS}
    rows["graph_edges"] = [
        {key: row[key] for key in _GRAPH_EDGE_FIELDS}
        for row in artifacts.get("graph_edges", ())
        if row.get("edge_type") in {"HAS_FINAL_EVIDENCE", "PAGE_GROUNDED_AT"}
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
    rows["raw_source_spans"] = [
        {key: row.get(key) for key in _SOURCE_SPAN_FIELDS if key in row}
        for row in artifacts.get("raw_source_spans", ())
        if row.get("semantic_text")
        and row.get("citation_eligible") is True
        and row.get("default_highlight_eligible") is True
    ]
    return {"schema": 1, "artifacts": rows}


def _bbox_refs(artifact_sets) -> set[str]:
    refs: set[str] = set()
    for rows in artifact_sets:
        for row in rows:
            for key, value in row.items():
                if "bbox" in key and isinstance(value, (list, tuple)):
                    refs.update(item for item in value if isinstance(item, str))
    return refs
