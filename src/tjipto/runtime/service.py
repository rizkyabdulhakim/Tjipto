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
