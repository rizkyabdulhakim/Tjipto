from __future__ import annotations

from collections import OrderedDict
from threading import RLock

from tjipto.contracts.relations import is_relevance_relation
from tjipto.contracts.evidence import source_lineage_reason


class EvidenceStore:
    _shared_stores: OrderedDict[tuple[str, str], "EvidenceStore"] = OrderedDict()
    _shared_lock = RLock()
    _shared_limit = 1

    @classmethod
    def shared(cls, config) -> "EvidenceStore":
        key = (str(config.manifest_path.resolve()), str(config.manifest_digest or config.artifact_set_digest or ""))
        with cls._shared_lock:
            store = cls._shared_stores.get(key)
            if store is None:
                store = cls(config)
                cls._shared_stores[key] = store
                while len(cls._shared_stores) > cls._shared_limit:
                    cls._shared_stores.popitem(last=False)
            else:
                cls._shared_stores.move_to_end(key)
            return store

    @classmethod
    def clear_shared_cache(cls) -> None:
        with cls._shared_lock:
            cls._shared_stores.clear()

    def __init__(self, config):
        self.config = config
        self._evidence: list[dict] | None = None
        self._legal_units: list[dict] | None = None
        self._chunks: list[dict] | None = None
        self._retrieval_units: list[dict] | None = None
        self._source_documents: list[dict] | None = None
        self._document_metadata: list[dict] | None = None
        self._metadata_grounding: list[dict] | None = None
        self._metadata_bbox_by_grounding: dict[str, list[dict]] | None = None
        self._source_conflicts: list[dict] | None = None
        self._graph_edges: list[dict] | None = None
        self._semantic_graph_edges: list[dict] | None = None
        self._bbox_by_evidence: dict[str, list[dict]] | None = None
        self._bbox_rows: list[dict] | None = None
        self._bbox_by_id: dict[str, dict] | None = None
        self._word_bboxes: list[dict] | None = None
        self._word_bbox_by_id: dict[str, dict] | None = None
        self._page_text_spans: list[dict] | None = None
        self._page_text_span_by_id: dict[str, dict] | None = None
        self._raw_source_spans: list[dict] | None = None
        self._meaningful_support_units: list[dict] | None = None
        self._propositions: list[dict] | None = None
        self._raw_source_span_by_support_id: dict[str, dict] | None = None
        self._raw_source_span_by_id: dict[str, dict] | None = None
        self._meaningful_support_unit_by_id: dict[str, dict] | None = None

    @property
    def evidence(self) -> list[dict]:
        if self._evidence is None:
            self._evidence = _rows(self.config, "evidence_registry")
        return self._evidence

    @property
    def legal_units(self) -> list[dict]:
        if self._legal_units is None:
            self._legal_units = _rows(self.config, "legal_units")
        return self._legal_units

    @property
    def chunks(self) -> list[dict]:
        if self._chunks is None:
            self._chunks = _rows(self.config, "chunks")
        return self._chunks

    @property
    def retrieval_units(self) -> list[dict]:
        if self._retrieval_units is None:
            self._retrieval_units = _rows(self.config, "retrieval_units")
        return self._retrieval_units

    @property
    def graph_edges(self) -> list[dict]:
        if self._graph_edges is None:
            self._graph_edges = _rows(self.config, "graph_edges")
        return self._graph_edges

    @property
    def semantic_graph_edges(self) -> list[dict]:
        """Validated semantic projection; provenance rows are intentionally absent."""
        if self._semantic_graph_edges is None:
            self._semantic_graph_edges = sorted(
                (
                dict(row)
                for row in self.graph_edges
                if row.get("runtime_loadable") is True and is_relevance_relation(row.get("edge_type"))
                ),
                key=lambda row: (row["edge_id"], row["source_id"], row["target_id"]),
            )
        return self._semantic_graph_edges

    @property
    def source_documents(self) -> list[dict]:
        if self._source_documents is None:
            self._source_documents = _rows(self.config, "source_documents")
        return self._source_documents

    @property
    def document_metadata(self) -> list[dict]:
        if self._document_metadata is None:
            self._document_metadata = _optional_jsonl(self.config, "document_metadata")
        return self._document_metadata

    @property
    def metadata_grounding(self) -> list[dict]:
        if self._metadata_grounding is None:
            self._metadata_grounding = _optional_jsonl(self.config, "metadata_grounding")
        return self._metadata_grounding

    @property
    def source_conflicts(self) -> list[dict]:
        if self._source_conflicts is None:
            self._source_conflicts = _optional_jsonl(self.config, "source_conflicts")
        return self._source_conflicts

    @property
    def page_text_spans(self) -> list[dict]:
        if self._page_text_spans is None:
            self._page_text_spans = _optional_jsonl(self.config, "page_text_spans")
        return self._page_text_spans

    @property
    def raw_source_spans(self) -> list[dict]:
        if self._raw_source_spans is None:
            self._raw_source_spans = _optional_jsonl(self.config, "raw_source_spans")
        return self._raw_source_spans

    @property
    def meaningful_support_units(self) -> list[dict]:
        if self._meaningful_support_units is None:
            self._meaningful_support_units = _optional_jsonl(self.config, "meaningful_support_units")
        return self._meaningful_support_units

    @property
    def propositions(self) -> list[dict]:
        if self._propositions is None:
            self._propositions = _optional_jsonl(self.config, "propositions")
        return self._propositions

    def get(self, evidence_id: str) -> dict | None:
        return next((row for row in self.evidence if row["evidence_id"] == evidence_id), None)

    def source_span_for_support(self, support_id: str) -> dict | None:
        if self._raw_source_span_by_support_id is None:
            self._raw_source_span_by_support_id = {
                str(row["source_support_id"]): row
                for row in self.raw_source_spans
                if isinstance(row.get("source_support_id"), str) and row["source_support_id"]
            }
        return self._raw_source_span_by_support_id.get(support_id)

    def source_span(self, raw_source_span_id: str) -> dict | None:
        if self._raw_source_span_by_id is None:
            self._raw_source_span_by_id = {
                str(row["raw_source_span_id"]): row
                for row in self.raw_source_spans
                if row.get("raw_source_span_id")
            }
        return self._raw_source_span_by_id.get(raw_source_span_id)

    def meaningful_support_unit(self, support_unit_id: str) -> dict | None:
        if self._meaningful_support_unit_by_id is None:
            self._meaningful_support_unit_by_id = {
                str(row["support_unit_id"]): row
                for row in self.meaningful_support_units
                if row.get("support_unit_id")
            }
        return self._meaningful_support_unit_by_id.get(support_unit_id)

    def page_text_span(self, text_span_id: str) -> dict | None:
        return self._page_text_span(text_span_id)

    def source_span_bboxes(self, support_id: str) -> list[dict]:
        row = self.source_span_for_support(support_id)
        if not row or not row.get("semantic_text") or row.get("citation_eligible") is not True:
            return []
        page_reference = next(
            (
                bbox
                for bbox in self._bbox_rows_all()
                if bbox.get("source_document_id") == row.get("source_document_id")
                and bbox.get("page_number") == row.get("page_number")
                and bbox.get("page_width") is not None
            ),
            {},
        )
        return [{
            "bbox_id": support_id,
            "bbox_precision": "exact",
            "viewer_highlightable": row.get("default_highlight_eligible") is True,
            "source_document_id": row.get("source_document_id"),
            "source_pdf_path": row.get("source_pdf_path"),
            "source_sha256": row.get("source_sha256"),
            "page_number": row.get("page_number"),
            "page_width": page_reference.get("page_width"),
            "page_height": page_reference.get("page_height"),
            "coordinate_space": page_reference.get("coordinate_space"),
            "coordinate_origin": page_reference.get("coordinate_origin"),
            "page_rotation": page_reference.get("page_rotation"),
            "page_box_basis": page_reference.get("page_box_basis"),
            "transform_version": page_reference.get("transform_version"),
            "x0": row.get("x0"),
            "y0": row.get("y0"),
            "x1": row.get("x1"),
            "y1": row.get("y1"),
        }]

    def lineage_error(self, evidence: dict) -> str | None:
        return source_lineage_reason(
            evidence=evidence,
            source_documents_by_id={row["source_document_id"]: row for row in self.source_documents},
            spans_by_id={row["text_span_id"]: row for row in self.page_text_spans},
            bboxes_by_id=self._bbox_rows_by_id(),
        )

    def bboxes_for(self, evidence_id: str) -> list[dict]:
        if self._bbox_by_evidence is None:
            by_id = self._bbox_rows_by_id()
            grouped = {}
            for evidence in self.evidence:
                refs = tuple(evidence.get("bbox_refs") or ())
                if refs:
                    grouped[evidence["evidence_id"]] = [by_id[bbox_id] for bbox_id in refs if bbox_id in by_id]
                else:
                    grouped[evidence["evidence_id"]] = [
                        row for row in self._bbox_rows_all() if row.get("evidence_id") == evidence["evidence_id"]
                    ]
            self._bbox_by_evidence = grouped
        return self._bbox_by_evidence.get(evidence_id, [])

    def bboxes_for_refs(self, bbox_refs: tuple[str, ...] | list[str]) -> list[dict]:
        by_id = self._bbox_rows_by_id()
        return [by_id[bbox_id] for bbox_id in bbox_refs if bbox_id in by_id]

    def metadata_bboxes_for(self, metadata_grounding_id: str) -> list[dict]:
        if self._metadata_bbox_by_grounding is None:
            bbox_by_id = self._bbox_rows_by_id()
            grouped: dict[str, list[dict]] = {}
            for row in _optional_jsonl(self.config, "metadata_grounding_registry"):
                bbox = bbox_by_id.get(row["bbox_id"])
                grouped.setdefault(row["metadata_grounding_id"], []).append((bbox or {}) | row)
            self._metadata_bbox_by_grounding = grouped
        return self._metadata_bbox_by_grounding.get(metadata_grounding_id, [])

    def exact_bboxes_for_text_spans(self, text_span_ids: tuple[str, ...] | list[str]) -> list[dict]:
        spans = [self._page_text_span(text_span_id) for text_span_id in text_span_ids]
        bbox_by_id = self._bbox_rows_by_id()
        rows: list[dict] = []
        seen: set[str] = set()
        for span in spans:
            if not span:
                continue
            span_matches = exact_bboxes_for_text_spans([span], self._bbox_rows_all())
            if not span_matches:
                span_matches = [bbox_by_id[bbox_id] for bbox_id in span.get("span_bbox_ids") or () if bbox_id in bbox_by_id]
            for bbox in span_matches:
                bbox_id = str(bbox.get("bbox_id") or "")
                resolved_bbox = bbox_by_id.get(bbox_id)
                if not resolved_bbox or bbox_id in seen:
                    continue
                seen.add(bbox_id)
                rows.append(resolved_bbox)
        return rows

    def _bbox_rows_all(self) -> list[dict]:
        if self._bbox_rows is None:
            self._bbox_rows = _rows(self.config, "bbox_registry")
        return self._bbox_rows

    def _word_bboxes_all(self) -> list[dict]:
        if self._word_bboxes is None:
            self._word_bboxes = _optional_jsonl(self.config, "word_bboxes")
        return self._word_bboxes

    def _bbox_rows_by_id(self) -> dict[str, dict]:
        if self._bbox_by_id is not None:
            return self._bbox_by_id
        rows = {row["bbox_id"]: row for row in self._bbox_rows_all()}
        referenced_bbox_ids = {
            value
            for artifact_rows in (
                self.evidence,
                self.metadata_grounding,
                self.page_text_spans,
                self.graph_edges,
                self.source_conflicts,
            )
            for artifact_row in artifact_rows
            for value in _bbox_refs(artifact_row)
        }
        for row in self._word_bboxes_all():
            word_id = row["word_bbox_id"]
            characters = row.get("characters") or ()
            if word_id not in referenced_bbox_ids and not any(
                character.get("character_bbox_id") in referenced_bbox_ids for character in characters
            ):
                continue
            if word_id in referenced_bbox_ids:
                rows[word_id] = {
                    "bbox_id": word_id,
                    "bbox_precision": "exact",
                    "viewer_highlightable": True,
                    **row,
                }
            for character in characters:
                if character.get("character_bbox_id") not in referenced_bbox_ids:
                    continue
                rows[character["character_bbox_id"]] = {
                    "bbox_id": character["character_bbox_id"],
                    "bbox_precision": "exact",
                    "viewer_highlightable": True,
                    "source_document_id": row["source_document_id"],
                    "source_pdf": row.get("source_pdf"),
                    "source_pdf_path": row.get("source_pdf_path"),
                    "source_sha256": row.get("source_sha256"),
                    "page_number": row["page_number"],
                    "page_width": row.get("page_width"),
                    "page_height": row.get("page_height"),
                    "coordinate_space": row.get("coordinate_space"),
                    "coordinate_origin": row.get("coordinate_origin"),
                    "page_rotation": row.get("page_rotation"),
                    "page_box_basis": row.get("page_box_basis"),
                    "transform_version": row.get("transform_version"),
                    **character,
                }
        self._bbox_by_id = rows
        return rows

    def _page_text_span(self, text_span_id: str) -> dict | None:
        if self._page_text_span_by_id is None:
            self._page_text_span_by_id = {row["text_span_id"]: row for row in self.page_text_spans if row.get("text_span_id")}
        return self._page_text_span_by_id.get(text_span_id)


def _optional_jsonl(config, logical_key: str) -> list[dict]:
    try:
        return _rows(config, logical_key)
    except (KeyError, OSError, ValueError):
        return []


def _rows(config, logical_key: str) -> list[dict]:
    try:
        projection = config.json("runtime_projection")
        rows = projection.get("artifacts", {}).get(logical_key)
        if isinstance(rows, (list, tuple)):
            return list(rows)
    except (KeyError, OSError, ValueError):
        pass
    return list(config.jsonl(logical_key))


def _bbox_refs(value: object) -> tuple[str, ...]:
    refs: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if "bbox" in key and isinstance(item, (list, tuple)):
                refs.extend(ref for ref in item if isinstance(ref, str))
            elif isinstance(item, (dict, list, tuple)):
                refs.extend(_bbox_refs(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            refs.extend(_bbox_refs(item))
    return tuple(refs)


def exact_bboxes_for_text_spans(text_spans: list[dict | None], bbox_rows: list[dict]) -> list[dict]:
    matches: list[dict] = []
    seen: set[str] = set()
    for span in text_spans:
        if not span or span.get("bbox_precision") != "exact":
            continue
        for bbox in bbox_rows:
            if not _same_span_bbox(span, bbox):
                continue
            bbox_id = str(bbox.get("bbox_id") or "")
            if bbox_id in seen or bbox.get("bbox_precision") != "exact" or bbox.get("viewer_highlightable") is not True:
                continue
            seen.add(bbox_id)
            matches.append(bbox)
    return matches


def _same_span_bbox(span: dict, bbox: dict) -> bool:
    return all(
        span.get(field) == bbox.get(field)
        for field in ("source_document_id", "source_sha256", "page_number", "text", "x0", "y0", "x1", "y1")
    )
