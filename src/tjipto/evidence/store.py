from __future__ import annotations


class EvidenceStore:
    def __init__(self, config):
        self.config = config
        self._evidence: list[dict] | None = None
        self._legal_units: list[dict] | None = None
        self._chunks: list[dict] | None = None
        self._retrieval_units: list[dict] | None = None
        self._source_documents: list[dict] | None = None
        self._document_metadata: list[dict] | None = None
        self._metadata_grounding: list[dict] | None = None
        self._document_relations: list[dict] | None = None
        self._article_amendment_relations: list[dict] | None = None
        self._metadata_bbox_by_grounding: dict[str, list[dict]] | None = None
        self._source_conflicts: list[dict] | None = None
        self._graph_edges: list[dict] | None = None
        self._bbox_by_evidence: dict[str, list[dict]] | None = None

    @property
    def evidence(self) -> list[dict]:
        if self._evidence is None:
            self._evidence = self.config.jsonl("evidence")
        return self._evidence

    @property
    def legal_units(self) -> list[dict]:
        if self._legal_units is None:
            self._legal_units = self.config.jsonl("legal_units")
        return self._legal_units

    @property
    def chunks(self) -> list[dict]:
        if self._chunks is None:
            self._chunks = self.config.jsonl("chunks")
        return self._chunks

    @property
    def retrieval_units(self) -> list[dict]:
        if self._retrieval_units is None:
            self._retrieval_units = self.config.jsonl("retrieval_units")
        return self._retrieval_units

    @property
    def graph_edges(self) -> list[dict]:
        if self._graph_edges is None:
            self._graph_edges = self.config.jsonl("graph_edges")
        return self._graph_edges

    @property
    def source_documents(self) -> list[dict]:
        if self._source_documents is None:
            self._source_documents = self.config.jsonl("source_documents")
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
    def document_relations(self) -> list[dict]:
        if self._document_relations is None:
            self._document_relations = _optional_jsonl(self.config, "document_relations")
        return self._document_relations

    @property
    def article_amendment_relations(self) -> list[dict]:
        if self._article_amendment_relations is None:
            self._article_amendment_relations = _optional_jsonl(self.config, "article_amendment_relations")
        return self._article_amendment_relations

    @property
    def source_conflicts(self) -> list[dict]:
        if self._source_conflicts is None:
            self._source_conflicts = _optional_jsonl(self.config, "source_conflicts")
        return self._source_conflicts

    def get(self, evidence_id: str) -> dict | None:
        return next((row for row in self.evidence if row["evidence_id"] == evidence_id), None)

    def bboxes_for(self, evidence_id: str) -> list[dict]:
        if self._bbox_by_evidence is None:
            grouped: dict[str, list[dict]] = {}
            for row in self.config.jsonl("bbox"):
                grouped.setdefault(row["evidence_id"], []).append(row)
            self._bbox_by_evidence = grouped
        return self._bbox_by_evidence.get(evidence_id, [])

    def metadata_bboxes_for(self, metadata_grounding_id: str) -> list[dict]:
        if self._metadata_bbox_by_grounding is None:
            grouped: dict[str, list[dict]] = {}
            for row in _optional_jsonl(self.config, "metadata_grounding_registry"):
                grouped.setdefault(row["metadata_grounding_id"], []).append(row)
            self._metadata_bbox_by_grounding = grouped
        return self._metadata_bbox_by_grounding.get(metadata_grounding_id, [])


def _optional_jsonl(config, logical_key: str) -> list[dict]:
    try:
        return config.jsonl(logical_key)
    except (KeyError, OSError, ValueError):
        return []
