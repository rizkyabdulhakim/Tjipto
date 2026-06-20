from __future__ import annotations

from pathlib import Path

from tjipto.corpora.registry import CorpusRegistry
from tjipto.evidence.store import EvidenceStore
from tjipto.retrieval.service import RetrievalService
from tjipto.runtime.viewer import viewer_payload


class LegalRuntimeService:
    def __init__(self, repo_root: Path | None = None):
        self.registry = CorpusRegistry(repo_root)

    def _store(self, corpus_id: str):
        config = self.registry.resolve(corpus_id)
        if config is None:
            return None
        return EvidenceStore(config)

    def search(self, corpus_id: str, query: str, limit: int = 10) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return {"status": "unsupported_corpus", "matches": ()}
        matches = RetrievalService(store).search(query, limit)
        return {"status": "found" if matches else "no_results", "matches": tuple(matches)}

    def citation(self, corpus_id: str, query: str, source_role: str | None = None) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return {"status": "unsupported_corpus", "matches": ()}
        matches = RetrievalService(store).citation(query, source_role)
        return {"status": "found" if matches else "citation_not_found", "matches": tuple(matches)}

    def viewer(self, corpus_id: str, evidence_id: str) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return {"status": "unsupported_corpus"}
        evidence = store.get(evidence_id)
        if evidence is None:
            return {"status": "not_found"}
        return viewer_payload(evidence, store.bboxes_for(evidence_id))

    def ask(self, corpus_id: str, query: str, limit: int = 3) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return {"status": "unsupported_corpus", "evidence": ()}
        lowered = query.casefold()
        unavailable = ("kuhp", "kuhap", "uu ite", "uu pers", "ketenagakerjaan", "uu pdp", "perseroan", "pemilu")
        if any(term in lowered for term in unavailable):
            return {"status": "insufficient_corpus", "evidence": (), "required_corpus": "non_uud"}
        matches = RetrievalService(store).search(query, limit)
        if not matches:
            return {"status": "no_results", "evidence": ()}
        evidence = tuple(self._answer_evidence(store, row) for row in matches if store.bboxes_for(row["evidence_id"]))
        if not evidence:
            return {"status": "insufficient_evidence", "evidence": ()}
        status = "answer_ready" if self.citation(corpus_id, query)["matches"] else "limited_answer"
        return {"status": status, "answer": "Evidence-grounded UUD support is available; no legal conclusion is generated.", "evidence": evidence}

    def _answer_evidence(self, store: EvidenceStore, row: dict) -> dict:
        bboxes = store.bboxes_for(row["evidence_id"])
        return {
            "evidence_id": row["evidence_id"],
            "citation": row.get("citation"),
            "source_role": row.get("source_role"),
            "source_pdf_path": row.get("source_pdf_path"),
            "source_sha256": row.get("source_sha256"),
            "page_numbers": tuple(row.get("page_numbers") or ()),
            "bbox_count": len(bboxes),
            "quoted_text": row.get("quoted_text"),
            "viewer_ref": {"action": "viewer", "evidence_id": row["evidence_id"]},
        }
