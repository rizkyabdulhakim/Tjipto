from __future__ import annotations

from tjipto.evidence.store import EvidenceStore
from tjipto.retrieval.bm25 import SparseIndex, sparse_index_for_store
from tjipto.retrieval.exact import exact_citation


class RetrievalService:
    def __init__(self, store: EvidenceStore):
        self.store = store

    @property
    def sparse_index(self) -> SparseIndex:
        return sparse_index_for_store(self.store)

    def citation(self, query: str, source_role: str | None = None) -> list[dict]:
        return exact_citation(
            self.store.evidence,
            query,
            source_role,
            corpus_id=self.store.config.corpus_id,
            preferred_source_role=getattr(self.store.config, "preferred_source_role", None),
        )

    def search(self, query: str, limit: int = 10) -> list[dict]:
        exact = self.citation(query)
        return exact[:limit] if exact else self.sparse_index.search(query, limit)
