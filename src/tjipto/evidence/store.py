from __future__ import annotations

class EvidenceStore:
    def __init__(self, config):
        self.config = config
        self._evidence: list[dict] | None = None
        self._legal_units: list[dict] | None = None
        self._chunks: list[dict] | None = None
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
    def graph_edges(self) -> list[dict]:
        if self._graph_edges is None:
            self._graph_edges = self.config.jsonl("graph_edges")
        return self._graph_edges

    def get(self, evidence_id: str) -> dict | None:
        return next((row for row in self.evidence if row["evidence_id"] == evidence_id), None)

    def bboxes_for(self, evidence_id: str) -> list[dict]:
        if self._bbox_by_evidence is None:
            grouped: dict[str, list[dict]] = {}
            for row in self.config.jsonl("bbox"):
                grouped.setdefault(row["evidence_id"], []).append(row)
            self._bbox_by_evidence = grouped
        return self._bbox_by_evidence.get(evidence_id, [])
