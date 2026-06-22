from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from tjipto.corpora.registry import CorpusRegistry
from tjipto.evidence.store import EvidenceStore
from tjipto.retrieval.answer import assemble_context_pack, empty_context_pack
from tjipto.retrieval.router import route_retrieval
from tjipto.runtime.viewer import resolve_pdf_access, viewer_payload


_BOOKMARKS: dict[str, dict] = {}


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
        context_pack = assemble_context_pack(store, routed["matches"]) if store and routed["matches"] else empty_context_pack(routed.get("reason"))
        public_status = "found" if context_pack["answer_evidence"] else ("no_results" if routed["status"] == "found" else routed["status"])
        return routed | {
            "status": "found" if routed["matches"] else routed["status"],
            "public_status": public_status,
            "results": tuple(_search_result(row, routed, context_pack) for row in context_pack["answer_evidence"]),
            "context_pack": context_pack,
        }

    def citation(
        self,
        corpus_id: str,
        query: str,
        source_role: str | None = None,
        filters: dict | None = None,
    ) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return route_retrieval(corpus_id, query, None) | _empty_citation_fields()
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
                **_empty_citation_fields(),
            }
        if not routed["matches"]:
            return routed | {"status": routed["status"], **_empty_citation_fields()}
        context_pack = assemble_context_pack(store, routed["matches"])
        return routed | {
            "status": "found",
            "context_pack": context_pack,
            "citation_payloads": context_pack["citation_payloads"],
            "viewer_refs": context_pack["viewer_refs"],
            "validation_reasons": context_pack["validation_reasons"],
        }

    def viewer(
        self,
        corpus_id: str,
        evidence_id: str,
        *,
        source_document_id: str | None = None,
        page_number: int | None = None,
        bbox_id: str | None = None,
        source_pdf_path: str | None = None,
    ) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return {"status": "unsupported_corpus", "corpus_id": corpus_id}
        evidence = store.get(evidence_id)
        if evidence is None:
            return {"status": "not_found", "reason": "invalid_evidence", "corpus_id": corpus_id}
        return viewer_payload(
            store,
            corpus_id,
            evidence,
            store.bboxes_for(evidence_id),
            source_document_id=source_document_id,
            page_number=page_number,
            bbox_id=bbox_id,
            source_pdf_path=source_pdf_path,
        )

    def pdf_access(
        self,
        corpus_id: str,
        evidence_id: str,
        *,
        source_document_id: str,
        page_number: int,
        source_sha256: str,
        bbox_id: str | None = None,
        source_pdf_path: str | None = None,
    ) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return {"status": "unsupported_corpus", "corpus_id": corpus_id}
        evidence = store.get(evidence_id)
        if evidence is None:
            return {"status": "not_found", "reason": "invalid_evidence", "corpus_id": corpus_id}
        return resolve_pdf_access(
            store,
            corpus_id,
            evidence,
            store.bboxes_for(evidence_id),
            source_document_id=source_document_id,
            page_number=page_number,
            bbox_id=bbox_id,
            source_sha256=source_sha256,
            source_pdf_path=source_pdf_path,
        )

    def capabilities(self, corpus_id: str) -> dict:
        if self._store(corpus_id) is None:
            return {"status": "unsupported_corpus", "corpus_id": corpus_id, "capabilities": ()}
        return {
            "status": "ok",
            "corpus_id": corpus_id,
            "capabilities": ("search", "ask", "citation", "viewer", "bookmarks"),
        }

    def bookmarks(self, corpus_id: str) -> dict:
        if self._store(corpus_id) is None:
            return {"status": "unsupported_corpus", "corpus_id": corpus_id, "bookmarks": ()}
        bookmarks = tuple(
            self._bookmark_status(row)
            for row in _BOOKMARKS.values()
            if row["corpus_id"] == corpus_id
        )
        return {
            "status": "ok",
            "corpus_id": corpus_id,
            "persistence": "memory",
            "persistence_label": "temporary_process_memory",
            "bookmarks": bookmarks,
        }

    def bookmark(
        self,
        corpus_id: str,
        evidence_id: str,
        note: str | None = None,
        citation_id: str | None = None,
        viewer_ref_id: str | None = None,
    ) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return {"status": "unsupported_corpus", "corpus_id": corpus_id}
        evidence = store.get(evidence_id)
        if evidence is None or evidence.get("status") != "final":
            return {"status": "unavailable", "reason": "evidence_unavailable", "corpus_id": corpus_id}
        bookmark = {
            "bookmark_id": f"bm_{uuid4().hex}",
            "corpus_id": corpus_id,
            "legal_unit_id": evidence.get("legal_unit_id"),
            "evidence_id": evidence_id,
            "citation_id": citation_id or evidence_id,
            "viewer_ref_id": viewer_ref_id or evidence_id,
            "note": note,
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "status": "active",
        }
        _BOOKMARKS[bookmark["bookmark_id"]] = bookmark
        return {"status": "saved", "bookmark": bookmark}

    def _bookmark_status(self, bookmark: dict) -> dict:
        store = self._store(bookmark["corpus_id"])
        evidence = store.get(bookmark["evidence_id"]) if store else None
        status = "active" if evidence and evidence.get("status") == "final" else "unavailable"
        return bookmark | {"status": status}

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
        citation = evidence[0].get("label") or evidence[0].get("citation") or "evidence"
        if status == "answer_ready":
            return f"Dukungan sitasi berbasis bukti tersedia untuk {citation}; sistem tidak menghasilkan kesimpulan hukum."
        return "Dukungan bukti terbatas tersedia; sistem tidak menghasilkan kesimpulan hukum."


def _empty_citation_fields() -> dict:
    return {"citation_payloads": (), "viewer_refs": (), "validation_reasons": {}}


def _search_result(row: dict, routed: dict, context_pack: dict) -> dict:
    viewer_ref = row.get("viewer_ref") or {}
    return {
        "corpus_id": routed["corpus_id"],
        "legal_unit_id": row.get("legal_unit_id"),
        "evidence_id": row["evidence_id"],
        "citation_id": row["evidence_id"],
        "viewer_ref_id": viewer_ref.get("evidence_id"),
        "title": row.get("label") or row.get("citation") or row.get("legal_unit_id") or routed["corpus_id"].upper(),
        "snippet": row.get("quoted_text"),
        "retrieval_method": routed["route"],
        "reasons": context_pack["validation_reasons"].get(row["evidence_id"]),
        "status": "evidence",
    }
