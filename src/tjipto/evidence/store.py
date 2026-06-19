from __future__ import annotations

from tjipto.core.manifest import read_jsonl


class EvidenceStore:
    def __init__(self, config):
        self.config = config
        self._evidence: list[dict] | None = None
        self._bbox_by_evidence: dict[str, list[dict]] | None = None

    @property
    def evidence(self) -> list[dict]:
        if self._evidence is None:
            self._evidence = read_jsonl(self.config.evidence_registry_path)
        return self._evidence

    def get(self, evidence_id: str) -> dict | None:
        return next((row for row in self.evidence if row["evidence_id"] == evidence_id), None)

    def bboxes_for(self, evidence_id: str) -> list[dict]:
        if self._bbox_by_evidence is None:
            grouped: dict[str, list[dict]] = {}
            for row in read_jsonl(self.config.bbox_registry_path):
                grouped.setdefault(row["evidence_id"], []).append(row)
            self._bbox_by_evidence = grouped
        return self._bbox_by_evidence.get(evidence_id, [])
