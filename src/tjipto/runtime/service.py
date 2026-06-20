from __future__ import annotations

from pathlib import Path

from tjipto.corpora.registry import CorpusRegistry
from tjipto.evidence.store import EvidenceStore
from tjipto.retrieval.answer import assemble_context_pack, empty_context_pack
from tjipto.retrieval.router import route_retrieval
from tjipto.runtime.viewer import viewer_payload


class LegalRuntimeService:
    def __init__(self, repo_root: Path | None = None):
        self.registry = CorpusRegistry(repo_root)

    def _store(self, corpus_id: str):
        config = self.registry.resolve(corpus_id)
        if config is None:
            return None
        return EvidenceStore(config)

    def search(self, corpus_id: str, query: str, limit: int = 10, filters: dict | None = None) -> dict:
        store = self._store(corpus_id)
        routed = route_retrieval(
            corpus_id,
            query,
            store,
            limit=limit,
            allow_bm25_after_citation_miss=True,
            metadata_filters=filters,
        )
        return routed | {"status": "found" if routed["matches"] else routed["status"]}

    def citation(
        self,
        corpus_id: str,
        query: str,
        source_role: str | None = None,
        filters: dict | None = None,
    ) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return route_retrieval(corpus_id, query, None)
        metadata_filters = dict(filters or {})
        if source_role is not None:
            metadata_filters["source_role"] = source_role
        routed = route_retrieval(corpus_id, query, store, metadata_filters=metadata_filters)
        if routed["intent"] != "exact_citation":
            return routed | {
                "status": "citation_not_found",
                "route": "citation_not_found",
                "matches": (),
                "reason": "not_a_citation",
            }
        if not routed["matches"]:
            return routed | {"status": routed["status"]}
        context_pack = assemble_context_pack(store, routed["matches"])
        return routed | {
            "status": "found",
            "context_pack": context_pack,
            "citation_payloads": context_pack["citation_payloads"],
            "viewer_refs": context_pack["viewer_refs"],
            "validation_reasons": context_pack["validation_reasons"],
        }

    def viewer(self, corpus_id: str, evidence_id: str) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return {"status": "unsupported_corpus"}
        evidence = store.get(evidence_id)
        if evidence is None:
            return {"status": "not_found"}
        return viewer_payload(evidence, store.bboxes_for(evidence_id))

    def ask(self, corpus_id: str, query: str, limit: int = 3, filters: dict | None = None) -> dict:
        store = self._store(corpus_id)
        routed = route_retrieval(corpus_id, query, store, limit=limit, metadata_filters=filters)
        if routed["status"] != "found":
            context_pack = empty_context_pack(routed.get("reason") or routed["status"])
            return routed | {
                "answer_type": "none",
                "answer": "Bukti tidak cukup atau database belum tersedia dalam korpus terverifikasi saat ini.",
                "context_pack": context_pack,
                "evidence": (),
                "citations": (),
                "viewer_refs": (),
            }
        context_pack = assemble_context_pack(store, routed["matches"])
        evidence = context_pack["answer_evidence"]
        if not evidence:
            return routed | {
                "status": "insufficient_evidence",
                "answer_type": "none",
                "answer": "Bukti tidak cukup atau database belum tersedia dalam korpus terverifikasi saat ini.",
                "context_pack": context_pack,
                "evidence": (),
                "citations": (),
                "viewer_refs": (),
            }
        status = "answer_ready" if routed["route"] == "exact" else "limited_answer"
        return routed | {
            "status": status,
            "answer_type": "quoted_evidence" if status == "answer_ready" else "limited_evidence_summary",
            "answer": self._answer_text(status, evidence),
            "context_pack": context_pack,
            "evidence": evidence,
            "citations": context_pack["citation_payloads"],
            "viewer_refs": context_pack["viewer_refs"],
        }

    def _answer_text(self, status: str, evidence: tuple[dict, ...]) -> str:
        citation = evidence[0].get("citation") or "evidence"
        if status == "answer_ready":
            return f"Evidence-grounded citation support is available for {citation}; no legal conclusion is generated."
        return "Limited evidence-grounded support is available; no legal conclusion is generated."
