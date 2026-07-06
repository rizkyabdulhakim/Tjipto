from __future__ import annotations

from tjipto.evidence.store import EvidenceStore
from tjipto.retrieval.bm25 import lexical_search
from tjipto.retrieval.exact import exact_citation


class RetrievalService:
    def __init__(self, store: EvidenceStore):
        self.store = store

    def citation(self, query: str, source_role: str | None = None) -> list[dict]:
        return exact_citation(
            self.store.evidence,
            query,
            source_role,
            preferred_source_role=getattr(self.store.config, "preferred_source_role", None),
        )

    def search(self, query: str, limit: int = 10) -> list[dict]:
        exact = self.citation(query)
        return exact[:limit] if exact else lexical_search(self.store.evidence, query, limit, config=self.store.config)
