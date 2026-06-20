from __future__ import annotations

class EvidenceStore:
    def __init__(self, config):
        self.config = config
        self._evidence: list[dict] | None = None
        self._bbox_by_evidence: dict[str, list[dict]] | None = None

    @property
    def evidence(self) -> list[dict]:
        if self._evidence is None:
            self._evidence = self.config.jsonl("evidence")
        return self._evidence

    def get(self, evidence_id: str) -> dict | None:
        return next((row for row in self.evidence if row["evidence_id"] == evidence_id), None)

    def bboxes_for(self, evidence_id: str) -> list[dict]:
        if self._bbox_by_evidence is None:
            grouped: dict[str, list[dict]] = {}
            for row in self.config.jsonl("bbox"):
                grouped.setdefault(row["evidence_id"], []).append(row)
            self._bbox_by_evidence = grouped
        return self._bbox_by_evidence.get(evidence_id, [])
