from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import threading
from uuid import uuid4

from tjipto.corpora.intent_config import contains_intent_phrase, intent_config_for, normalize_intent_text, resolve_instrument_intent
from tjipto.corpora.parser_dispatch import parse_legal_reference, parse_legal_references
from tjipto.corpora.registry import CorpusRegistry
from tjipto.corpora.verified import CorpusIntegrityError, VerifiedCorpusRepository
from tjipto.evidence.store import EvidenceStore
from tjipto.retrieval.answer import assemble_context_pack, empty_context_pack, validate_answer_candidate
from tjipto.retrieval.metadata import (
    has_metadata_target,
    metadata_lookup,
    normalize_filters,
    public_filters,
    resolve_source_scope,
    source_roles_for_query,
)
from tjipto.retrieval.relations import has_relation_target
from tjipto.retrieval.router import route_retrieval
from tjipto.runtime.intent import classify_relation_intent
from tjipto.runtime.scope_guard import scope_guard_context
from tjipto.runtime.viewer import document_viewer_payload, resolve_document_pdf_access, resolve_pdf_access, viewer_payload


_BOOKMARKS: dict[str, dict] = {}
_BOOKMARK_LOCK = threading.RLock()

_ANSWER_TEMPLATES = {
    "insufficient": "Bukti tidak cukup atau database belum tersedia dalam korpus terverifikasi saat ini.",
    "clarification": "Naskah sumber mana yang dimaksud? Pilih salah satu konteks berikut: {options}.",
    "metadata": "{answer} (from grounded document metadata).",
    "legal_relation": "Dukungan relasi hukum berbasis bukti tersedia; sistem tidak menghasilkan kesimpulan hukum.",
    "citation": "Dukungan sitasi berbasis bukti tersedia untuk {citation}; sistem tidak menghasilkan kesimpulan hukum.",
    "limited": "Dukungan bukti terbatas tersedia; sistem tidak menghasilkan kesimpulan hukum.",
}


def _integrity_failure(corpus_id: str, query: str, error_code: str | None) -> dict:
    unknown = error_code in {"unknown_corpus", "registry_unavailable"}
    route = "unsupported_corpus" if unknown else "corpus_integrity"
    return {
        "status": "unsupported_corpus" if unknown else "corpus_not_ready",
        "route": route,
        "intent": route,
        "corpus_id": corpus_id,
        "original_query": query,
        "normalized_query": query.strip(),
        "reason": error_code or "corpus_load_failure",
        "reason_code": error_code or "corpus_load_failure",
        "readiness": False,
        "evidence": (),
        "citations": (),
        "final_citations": (),
        "historical_citations": (),
        "metadata_support": (),
        "structural_support": (),
        "trace_support": (),
        "viewer_refs": (),
        "context_pack": empty_context_pack(error_code),
        "answer_scope": "insufficient_evidence",
        "answer_type": "none",
        "answer": _ANSWER_TEMPLATES["insufficient"],
    }


def _has_resolved_legal_target(corpus_id: str, query: str) -> bool:
    parsed = parse_legal_reference(corpus_id, query, allow_roman_pasal=True)
    return any(parsed.values())


class LegalRuntimeService:
    def __init__(self, repo_root: Path | None = None):
        self.registry = CorpusRegistry(repo_root)
        self.repository = VerifiedCorpusRepository(self.registry)
        self._integrity_error: str | None = None
        self._store_cache: dict[str, EvidenceStore] = {}

    def _store(self, corpus_id: str):
        cached = self._store_cache.get(corpus_id)
        if cached is not None:
            self._integrity_error = None
            return cached
        try:
            config = self.repository.load(corpus_id).config
            self._integrity_error = None
        except CorpusIntegrityError as error:
            self._integrity_error = error.code
            return None
        store = EvidenceStore.shared(config)
        self._store_cache[corpus_id] = store
        return store

    def search(self, corpus_id: str, query: str, limit: int = 10, filters: dict | None = None) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return _integrity_failure(corpus_id, query, self._integrity_error) | {"results": ()}
        normalized_filters = normalize_filters(filters, config=store.config)
        if normalized_filters.get("_error"):
            return _catalog_search_response(
                corpus_id,
                query,
                (),
                "invalid_filter",
                normalized_filters["_error"],
                applied_filters=public_filters(normalized_filters),
                invalid_filters=normalized_filters.get("_invalid_filters", ()),
            )
        rows = _catalog_search(store, corpus_id, query, limit, normalized_filters)
        return _catalog_search_response(
            corpus_id,
            query,
            rows,
            "found" if rows else "no_results",
            None if rows else "document_not_found",
            applied_filters=public_filters(normalized_filters),
        )

    def citation(
        self,
        corpus_id: str,
        query: str,
        source_role: str | None = None,
        filters: dict | None = None,
    ) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return _integrity_failure(corpus_id, query, self._integrity_error) | _empty_citation_fields()
        scope = scope_guard_context(store, query)
        if scope:
            return {
                "status": "citation_not_found",
                "public_status": "insufficient_evidence",
                "route": scope["route"],
                "intent": scope["intent"],
                "matches": (),
                "reason": scope["reason"],
                "requested_function": scope.get("requested_function"),
                "target_reference": scope.get("target_reference"),
                "legal_domain": scope.get("legal_domain"),
                **_empty_citation_fields(),
            }
        metadata_filters = dict(filters or {})
        scope = resolve_source_scope(query, strategy=getattr(store.config, "query_strategy", "generic"), config=store.config)
        requested_role = source_role or (scope.role if scope.explicit else None)
        if requested_role is not None:
            metadata_filters["source_role"] = requested_role
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
        citations = tuple(_citation_with_authority(store, row) for row in context_pack["citation_payloads"])
        return routed | {
            "status": "found",
            "context_pack": context_pack,
            "citation_payloads": citations,
            "viewer_refs": context_pack["viewer_refs"],
            "validation_reasons": context_pack["validation_reasons"],
        }

    def viewer(
        self,
        corpus_id: str,
        evidence_id: str | None,
        *,
        relation_id: str | None = None,
        source_document_id: str | None = None,
        page_number: int | None = None,
        bbox_id: str | None = None,
        source_pdf_path: str | None = None,
    ) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return _integrity_failure(corpus_id, "", self._integrity_error)
        if evidence_id is None:
            source = _source_document_by_id(store, source_document_id)
            if source is None:
                return {"status": "not_found", "reason": "invalid_source", "corpus_id": corpus_id}
            return document_viewer_payload(
                store,
                corpus_id,
                source,
                page_number=page_number,
                source_pdf_path=source_pdf_path,
            )
        evidence = store.get(evidence_id)
        if evidence is None:
            evidence = _metadata_grounding_evidence(store, evidence_id)
        synthetic_bboxes: list[dict] | None = None
        if evidence is None:
            evidence, synthetic_bboxes = _source_conflict_viewer_evidence(store, evidence_id)
        if evidence is None:
            return {"status": "not_found", "reason": "invalid_evidence", "corpus_id": corpus_id}
        relation = _relation_for_evidence(store, evidence_id, relation_id)
        if relation is not None:
            evidence = evidence | {
                "bbox_refs": tuple(relation.get("bbox_refs") or ()),
                "quoted_text": relation.get("quoted_text") or evidence.get("quoted_text"),
            }
        bboxes = (
            synthetic_bboxes
            if synthetic_bboxes is not None
            else store.metadata_bboxes_for(evidence_id)
            if evidence.get("metadata_grounding")
            else store.bboxes_for_refs(tuple(relation.get("bbox_refs") or ())) if relation is not None else store.bboxes_for(evidence_id)
        )
        return _viewer_with_authority(
            store,
            evidence,
            viewer_payload(
                store,
                corpus_id,
                evidence,
                bboxes,
                source_document_id=source_document_id,
                page_number=page_number,
                bbox_id=bbox_id,
                source_pdf_path=source_pdf_path,
            ),
        )

    def pdf_access(
        self,
        corpus_id: str,
        evidence_id: str | None,
        *,
        relation_id: str | None = None,
        source_document_id: str,
        page_number: int,
        source_sha256: str | None = None,
        bbox_id: str | None = None,
        source_pdf_path: str | None = None,
    ) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return _integrity_failure(corpus_id, "", self._integrity_error)
        if evidence_id is None:
            source = _source_document_by_id(store, source_document_id)
            if source is None:
                return {"status": "not_found", "reason": "invalid_source", "corpus_id": corpus_id}
            return resolve_document_pdf_access(
                store,
                corpus_id,
                source,
                page_number=page_number,
                source_pdf_path=source_pdf_path,
            )
        evidence = store.get(evidence_id)
        if evidence is None:
            evidence = _metadata_grounding_evidence(store, evidence_id)
        synthetic_bboxes: list[dict] | None = None
        if evidence is None:
            evidence, synthetic_bboxes = _source_conflict_viewer_evidence(store, evidence_id)
        if evidence is None:
            return {"status": "not_found", "reason": "invalid_evidence", "corpus_id": corpus_id}
        relation = _relation_for_evidence(store, evidence_id, relation_id)
        if relation is not None:
            evidence = evidence | {
                "bbox_refs": tuple(relation.get("bbox_refs") or ()),
                "quoted_text": relation.get("quoted_text") or evidence.get("quoted_text"),
            }
        bboxes = (
            synthetic_bboxes
            if synthetic_bboxes is not None
            else store.metadata_bboxes_for(evidence_id)
            if evidence.get("metadata_grounding")
            else store.bboxes_for_refs(tuple(relation.get("bbox_refs") or ())) if relation is not None else store.bboxes_for(evidence_id)
        )
        return resolve_pdf_access(
            store,
            corpus_id,
            evidence,
            bboxes,
            source_document_id=source_document_id,
            page_number=page_number,
            bbox_id=bbox_id,
            source_sha256=source_sha256,
            source_pdf_path=source_pdf_path,
        )

    def capabilities(self, corpus_id: str) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return _integrity_failure(corpus_id, "", self._integrity_error) | {"capabilities": ()}
        return {
            "status": "ok",
            "corpus_id": corpus_id,
            "readiness": True,
            "manifest_digest": store.config.manifest_digest,
            "artifact_set_digest": store.config.artifact_set_digest,
            "capabilities": ("search", "ask", "citation", "viewer", "bookmarks"),
        }

    def bookmarks(self, corpus_id: str) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return _integrity_failure(corpus_id, "", self._integrity_error) | {"bookmarks": ()}
        with _BOOKMARK_LOCK:
            snapshot = tuple(row.copy() for row in _BOOKMARKS.values() if row["corpus_id"] == corpus_id)
        bookmarks = tuple(sorted((self._bookmark_status(row, store) for row in snapshot), key=lambda row: row["bookmark_id"]))
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
            return _integrity_failure(corpus_id, "", self._integrity_error)
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
        with _BOOKMARK_LOCK:
            _BOOKMARKS[bookmark["bookmark_id"]] = bookmark
        return {"status": "saved", "bookmark": bookmark}

    def _bookmark_status(self, bookmark: dict, store=None) -> dict:
        store = store or self._store(bookmark["corpus_id"])
        evidence = store.get(bookmark["evidence_id"]) if store else None
        status = "active" if evidence and evidence.get("status") == "final" else "unavailable"
        return bookmark | {"status": status}

    def ask(self, corpus_id: str, query: str, limit: int = 3, filters: dict | None = None) -> dict:
        store = self._store(corpus_id)
        if store is None:
            return _integrity_failure(corpus_id, query, self._integrity_error)
        anomaly = _source_anomaly_response(store, corpus_id, query)
        if anomaly:
            return anomaly
        document_relation = _document_relation_response(store, corpus_id, query)
        if document_relation:
            return document_relation
        source_document = _source_document_response(store, corpus_id, query)
        if source_document:
            return source_document
        # A resolved legal target has precedence over the instrument classifier.
        # Amendment wording then scopes the structured lookup to that source role.
        instrument = None if _has_resolved_legal_target(corpus_id, query) else _instrument_intent_context(store, query)
        if instrument:
            row, route, reason = instrument
            templates = _answer_templates(store)
            if row is None:
                context_pack = empty_context_pack(reason)
                return {
                    "status": "insufficient_evidence",
                    "route": route,
                    "intent": "instrument_unit_lookup",
                    "corpus_id": corpus_id,
                    "original_query": query,
                    "normalized_query": query.strip(),
                    "matches": (),
                    "reason": reason,
                    "answer_type": "none",
                    "answer": templates["insufficient"],
                    "context_pack": context_pack,
                    "evidence": (),
                    "citations": (),
                    "final_citations": (),
                    "historical_citations": (),
                    "metadata_support": (),
                    "structural_support": (),
                    "trace_support": (),
                    "viewer_refs": (),
                    "metadata_facts": (),
                    "legal_relations": (),
                    "answer_scope": "insufficient_evidence",
                    "warnings": (),
                    "insufficient_reasons": (reason,),
                }
            context_pack = assemble_context_pack(store, (row,))
            evidence = context_pack["answer_evidence"]
            if not evidence:
                return {
                    "status": "insufficient_evidence",
                    "route": route,
                    "intent": "instrument_unit_lookup",
                    "corpus_id": corpus_id,
                    "original_query": query,
                    "normalized_query": query.strip(),
                    "matches": (row,),
                    "reason": reason,
                    "answer_type": "none",
                    "answer": templates["insufficient"],
                    "context_pack": context_pack,
                    "evidence": (),
                    "citations": (),
                    "final_citations": (),
                    "historical_citations": context_pack.get("historical_citations", ()),
                    "metadata_support": context_pack.get("metadata_support", ()),
                    "structural_support": context_pack.get("structural_support", ()),
                    "trace_support": context_pack.get("trace_support", ()),
                    "viewer_refs": (),
                    "metadata_facts": (),
                    "legal_relations": (),
                    "answer_scope": "insufficient_evidence",
                    "warnings": (),
                    "insufficient_reasons": (reason,),
                }
            instrument_status = "answer_ready" if context_pack["citation_payloads"] else "limited_answer"
            return {
                "status": instrument_status,
                "route": route,
                "intent": "instrument_unit_lookup",
                "corpus_id": corpus_id,
                "original_query": query,
                "normalized_query": query.strip(),
                "matches": (row,),
                "reason": None,
                "answer_type": "quoted_evidence",
                "answer": self._answer_text(instrument_status, evidence, templates),
                "context_pack": context_pack,
                "evidence": evidence,
                "citations": tuple(_citation_with_authority(store, item) for item in context_pack["citation_payloads"]),
                "final_citations": tuple(_citation_with_authority(store, item) for item in context_pack["citation_payloads"]),
                "historical_citations": context_pack.get("historical_citations", ()),
                "metadata_support": context_pack.get("metadata_support", ()),
                "structural_support": context_pack.get("structural_support", ()),
                "trace_support": context_pack.get("trace_support", ()),
                "viewer_refs": context_pack["viewer_refs"] if context_pack["citation_payloads"] else (),
                "metadata_facts": (),
                "legal_relations": (),
                "answer_scope": "direct_evidence" if instrument_status == "answer_ready" else "limited_evidence",
                "warnings": (),
                "insufficient_reasons": (),
            }
        scope = scope_guard_context(store, query)
        if scope:
            templates = _answer_templates(store)
            context_pack = empty_context_pack(scope["reason"])
            return scope | {
                "status": "insufficient_evidence",
                "corpus_id": corpus_id,
                "original_query": query,
                "normalized_query": query.strip(),
                "matches": (),
                "answer_type": "none",
                "answer": templates["insufficient"],
                "context_pack": context_pack,
                "evidence": (),
                "citations": (),
                "final_citations": (),
                "historical_citations": context_pack.get("historical_citations", ()),
                "metadata_support": context_pack.get("metadata_support", ()),
                "structural_support": context_pack.get("structural_support", ()),
                "trace_support": context_pack.get("trace_support", ()),
                "viewer_refs": (),
                "metadata_facts": (),
                "legal_relations": (),
                "answer_scope": "insufficient_evidence",
                "warnings": (),
                "insufficient_reasons": (scope["reason"],),
            }
        routed = route_retrieval(corpus_id, query, store, limit=limit, metadata_filters=filters)
        ask_route = _ask_route(routed["route"])
        templates = _answer_templates(store)
        clarification = _metadata_scope_clarification(store, routed)
        if clarification:
            return routed | clarification
        if routed["status"] != "found":
            public_status = (
                "insufficient_evidence"
                if routed.get("route")
                in {"metadata_not_found", "relation_not_found", "structured_not_found", "scope_unresolved"}
                else routed["status"]
            )
            context_pack = empty_context_pack(routed.get("reason") or routed["status"])
            return routed | {
                "status": public_status,
                "route": ask_route,
                "answer_type": "none",
                "answer": templates["insufficient"],
                "context_pack": context_pack,
                "evidence": (),
                "citations": (),
                "viewer_refs": (),
                "metadata_facts": (),
                "legal_relations": (),
                "answer_scope": "insufficient_evidence",
                "warnings": (),
                "insufficient_reasons": (routed.get("reason") or routed["status"],),
            }
        answer_matches = routed["matches"]
        if ask_route == "lexical_fallback":
            # BM25 may rank several independently relevant rows, but one
            # answer may claim only one complete source-backed proposition.
            answer_matches = next(
                (row for row in answer_matches if validate_answer_candidate(store, row)[0]),
                None,
            )
            # Keep rejected lexical candidates in the diagnostic pack so a
            # fail-closed response still states why no answer was published.
            answer_matches = (answer_matches,) if answer_matches else routed["matches"]
        context_pack = assemble_context_pack(store, answer_matches)
        evidence = context_pack["answer_evidence"]
        if not evidence:
            return routed | {
                "status": "insufficient_evidence",
                "route": ask_route,
                "answer_type": "none",
                "answer": templates["insufficient"],
                "context_pack": context_pack,
                "evidence": (),
                "citations": (),
                "viewer_refs": (),
                "metadata_facts": (),
                "legal_relations": (),
                "answer_scope": "insufficient_evidence",
                "warnings": (),
                "insufficient_reasons": tuple(sorted(set(context_pack["validation_reasons"].values()))),
            }
        status = (
            "limited_answer"
            if ask_route == "lexical_fallback" or (context_pack["trace_support"] and not context_pack["citation_payloads"])
            else "answer_ready"
        )
        metadata_support = tuple(_metadata_support(store, row) for row in evidence if row.get("metadata_field"))
        citations: tuple[dict, ...]
        if metadata_support:
            citations = ()
            viewer_refs = ()
        else:
            citations = tuple(_citation_with_authority(store, row) for row in context_pack["citation_payloads"])
            viewer_refs = context_pack["viewer_refs"]
        deterministic_answer = self._answer_text(status, evidence, templates)
        answer = self._agent_answer(query, evidence, deterministic_answer)
        return routed | {
            "status": status,
            "route": ask_route,
            "answer_type": _answer_type(ask_route, status),
            "answer": answer,
            "context_pack": context_pack,
            "evidence": evidence,
            "citations": citations,
            "final_citations": citations,
            "historical_citations": context_pack.get("historical_citations", ()),
            "viewer_refs": viewer_refs,
            "metadata_facts": tuple(_metadata_fact(row) for row in evidence if row.get("metadata_field")),
            "metadata_support": metadata_support,
            "structural_support": context_pack.get("structural_support", ()),
            "trace_support": context_pack.get("trace_support", ()),
            "legal_relations": tuple(row["legal_relation"] for row in evidence if row.get("legal_relation")),
            "answer_scope": "direct_evidence" if status == "answer_ready" else "limited_evidence",
            "warnings": ("metadata_support_not_exact_highlightable",)
            if any(row.get("viewer_highlightable") is not True for row in metadata_support)
            else (),
            "insufficient_reasons": (),
        }

    def _agent_answer(self, query: str, evidence: tuple[dict, ...], fallback: str) -> str:
        return fallback

    def _answer_text(self, status: str, evidence: tuple[dict, ...], templates: dict[str, str]) -> str:
        if evidence[0].get("metadata_answer"):
            return templates["metadata"].format(answer=evidence[0]["metadata_answer"])
        if evidence[0].get("legal_relation"):
            return templates["legal_relation"]
        citation = evidence[0].get("label") or evidence[0].get("citation") or "evidence"
        if status == "answer_ready":
            return templates["citation"].format(citation=citation)
        return templates["limited"]


def _empty_citation_fields() -> dict:
    return {
        "final_citations": (),
        "historical_citations": (),
        "metadata_support": (),
        "structural_support": (),
        "trace_support": (),
        "citation_payloads": (),
        "viewer_refs": (),
        "validation_reasons": {},
    }


def _answer_templates(store) -> dict[str, str]:
    configured: dict = getattr(getattr(store, "config", None), "setting", lambda *args: {})("answer_templates", {})
    return _ANSWER_TEMPLATES | dict(configured or {})


def _metadata_scope_clarification(store, routed: dict) -> dict | None:
    if routed.get("route") not in {"metadata", "metadata_scope_unresolved"} or "source_role" in routed.get("applied_filters", {}):
        return None
    roles = tuple(routed.get("metadata_source_roles") or ())
    if not roles:
        roles = tuple(
            sorted(
                {
                    row.get("source_role")
                    for row in routed.get("matches", ())
                    if row.get("metadata_field") and row.get("source_role")
                }
            )
        )
    if len(roles) < 2:
        return None
    intent = intent_config_for(getattr(store.config, "query_strategy", "generic"), store.config)
    labels = intent.get("source_role_labels", {})
    titles = (store.config.setting("document_catalog", {}) or {}).get("titles", {})
    options = tuple({"source_role": role, "label": titles.get(role, labels.get(role, role))} for role in roles)
    answer = _answer_templates(store)["clarification"].format(options=", ".join(item["label"] for item in options))
    return {
        "status": "clarification_required",
        "route": "metadata_fact",
        "intent": "metadata_lookup",
        "reason": routed.get("reason") or "ambiguous_source_scope",
        "answer_type": "clarification",
        "answer": answer,
        "answer_scope": "clarification",
        "clarification_options": options,
        "context_pack": empty_context_pack(routed.get("reason") or "ambiguous_source_scope"),
        "evidence": (),
        "citations": (),
        "final_citations": (),
        "historical_citations": (),
        "metadata_support": (),
        "structural_support": (),
        "trace_support": (),
        "viewer_refs": (),
        "metadata_facts": (),
        "legal_relations": (),
        "warnings": ("source_scope_required",),
        "insufficient_reasons": ("ambiguous_source_scope",),
    }


def _authority_policy(store, row: dict, *, can_resolve: bool | None = None, conflict: dict | None = None) -> dict:
    owner = store.get(row.get("evidence_id")) if store is not None and row.get("evidence_id") else None
    source_row = {**(owner or {}), **row}
    authority_kind = _authority_kind(store, row, can_resolve=can_resolve, conflict=conflict)
    conflict_row = conflict or _source_conflict_by_evidence(store, row.get("evidence_id"))
    non_final_conflict = conflict_row is not None or row.get("source_conflict_id")
    citation_final = row.get("citation_final") if isinstance(row.get("citation_final"), bool) else authority_kind == "legal_citation"
    if non_final_conflict and authority_kind in {"source_anomaly", "source_conflict_provenance"}:
        citation_final = False
    payload = {
        "authority_kind": authority_kind,
        "authority_label": {
            "legal_citation": "Sitasi hukum",
            "metadata_source": "Metadata sumber",
            "metadata_trace": "Metadata trace",
            "source_conflict_provenance": "Jejak audit sumber",
            "source_anomaly": "Source anomaly",
            "structural_context": "Provenance struktural",
            "instrument_provenance": "Instrument provenance",
        }[authority_kind],
        "citation_final": citation_final,
        "source_url": row.get("source_url") or _source_url(store, row),
        "support_kind": "legal_unit" if source_row.get("evidence_owner_kind") == "legal_unit_source" and authority_kind == "legal_citation" else row.get("support_kind") or _support_kind_for_authority(authority_kind),
        "relevant_quote_eligible": source_row.get("relevant_quote_eligible") is True and authority_kind == "legal_citation",
        "display_text": row.get("display_text") or row.get("quoted_text") or "",
        "copy_text": _copy_text(row.get("copy_text") or row.get("quoted_text") or ""),
        "layout_lines": tuple(str(row.get("layout_lines") or row.get("quoted_text") or "").splitlines()),
        "viewer_target": row.get("viewer_ref") or {},
    }
    if conflict_row is not None or row.get("source_conflict_id"):
        payload |= _source_conflict_taxonomy_fields(conflict_row or row)
    return payload


def _copy_text(value: str) -> str:
    return "\n".join(line.lstrip(" \t") for line in str(value).replace("\r\n", "\n").replace("\r", "\n").split("\n")).strip()


def _support_kind_for_authority(authority_kind: str) -> str:
    return {
        "legal_citation": "legal_citation",
        "metadata_source": "metadata_source",
        "metadata_trace": "metadata_trace",
        "source_conflict_provenance": "source_anomaly_provenance",
        "source_anomaly": "source_anomaly_provenance",
        "structural_context": "structural_provenance",
        "instrument_provenance": "instrument_provenance",
    }[authority_kind]


def _source_url(store, row: dict) -> str | None:
    source_id = row.get("source_document_id")
    return next(
        (
            source.get("source_page_url") or source.get("final_download_url") or source.get("download_url")
            for source in getattr(store, "source_documents", ())
            if source.get("source_document_id") == source_id
        ),
        None,
    )


def _source_conflict_taxonomy_fields(conflict: dict | None) -> dict:
    if not conflict:
        return {}
    policy = conflict.get("source_anomaly_policy") or {}
    fields = {
        "source_anomaly_kind": conflict.get("source_anomaly_kind") or policy.get("anomaly_kind"),
        "source_mapping_kind": conflict.get("source_mapping_kind") or policy.get("mapping_kind"),
        "provenance_highlight_scope": conflict.get("provenance_highlight_scope") or policy.get("provenance_highlight_scope"),
        "finality_policy": policy.get("finality_policy"),
        "support_type": conflict.get("type"),
    }
    fields = {key: value for key, value in fields.items() if value is not None}
    if fields.get("finality_policy"):
        fields["support_kind"] = fields["finality_policy"]
    fields.update(_source_mapping_semantics(conflict))
    return fields


def _authority_kind(store, row: dict, *, can_resolve: bool | None = None, conflict: dict | None = None) -> str:
    viewer_resolvable = (
        can_resolve
        if can_resolve is not None
        else row.get("viewer_ref", {}).get("can_resolve") is True or row.get("viewer_highlightable") is True
    )
    if row.get("metadata_grounding") or row.get("metadata_field"):
        return "metadata_source" if viewer_resolvable else "metadata_trace"
    if row.get("authority_kind") == "source_anomaly_trace" and row.get("citation_final") is False:
        return "source_anomaly"
    if row.get("authority_kind") == "structural_context":
        return "structural_context"
    conflict_row = conflict or _source_conflict_by_evidence(store, row.get("evidence_id"))
    if conflict_row is not None or row.get("source_conflict_id"):
        return "source_anomaly" if _is_source_anomaly_conflict(conflict_row or row) else "source_conflict_provenance"
    if _row_is_historical_anomaly(store, row):
        return "source_anomaly"
    if _row_is_instrument_provenance(store, row):
        return "instrument_provenance"
    return "legal_citation"


def _source_conflict_by_evidence(store, evidence_id: object) -> dict | None:
    if store is None or not evidence_id:
        return None
    return next(
        (
            row
            for row in store.source_conflicts
            if evidence_id == row.get("source_conflict_id") or evidence_id in set(row.get("evidence_ids") or ())
        ),
        None,
    )


def _is_source_anomaly_conflict(row: dict) -> bool:
    if row.get("source_anomaly_kind") == "renumbering_provenance":
        return False
    if row.get("source_anomaly_kind") == "source_marker_sequence_anomaly":
        return True
    classification = str(row.get("classification") or "").casefold()
    return (
        row.get("provenance_exception_category") == "accepted_noncanonical_source_conflict_trace_only"
        or "anomaly" in classification
        or "typo" in classification
    )


def _row_is_instrument_provenance(store, row: dict) -> bool:
    candidate_type = str(row.get("candidate_type") or "")
    if candidate_type == "article_amendment_relation" or candidate_type.startswith("instrument_"):
        return True
    if store is None:
        return False
    try:
        units = store.legal_units
    except (KeyError, OSError, ValueError):
        return False
    unit: dict = next((item for item in units if item.get("legal_unit_id") == row.get("legal_unit_id")), {})
    return bool(unit) and _is_instrument_unit(store, unit)


def _row_is_historical_anomaly(store, row: dict) -> bool:
    if store is None:
        return False
    try:
        units = store.legal_units
    except (KeyError, OSError, ValueError):
        return False
    unit: dict = next((item for item in units if item.get("legal_unit_id") == row.get("legal_unit_id")), {})
    return unit.get("status") == "active_historical_record" and bool(unit.get("exclusion_ref"))


def _citation_with_authority(store, row: dict, *, conflict: dict | None = None) -> dict:
    return row | _authority_policy(store, row, conflict=conflict)


def _viewer_with_authority(store, evidence: dict, payload: dict) -> dict:
    return payload | _authority_policy(store, evidence, can_resolve=payload.get("viewer_highlightable") is True)


def _catalog_search(store, corpus_id: str, query: str, limit: int, filters: dict) -> tuple[dict, ...]:
    query_text = normalize_intent_text(query)
    if not query_text:
        return ()
    catalog = store.config.setting("document_catalog", {}) or {}
    if not contains_intent_phrase(query, tuple(catalog.get("document_terms") or ())):
        return ()
    rows = [
        _document_result(store, corpus_id, source, _document_search_score(store, source, query_text))
        for source in store.source_documents
        if _document_matches_filters(source, filters)
    ]
    rows = [row for row in rows if row["_score"] > 0]
    rows.sort(key=lambda row: (-row["_score"], row["title"]))
    return tuple(({key: value for key, value in row.items() if key != "_score"}) for row in rows[:limit])


def _document_search_score(store, source: dict, query_text: str) -> int:
    haystack = normalize_intent_text(
        " ".join(
            str(item or "")
            for item in (
                _document_title(store, source),
                source.get("filename"),
                source.get("source_role"),
                source.get("temporal_context"),
            )
        )
    )
    return sum(1 for token in query_text.split() if token in haystack)


def _document_result(store, corpus_id: str, source: dict, score: int) -> dict:
    page_count = int(source.get("page_count") or 0)
    return {
        "_score": score,
        "corpus_id": corpus_id,
        "source_document_id": source.get("source_document_id"),
        "document_id": source.get("source_document_id"),
        "title": _document_title(store, source),
        "document_title": _document_title(store, source),
        "snippet": f"Dokumen sumber terverifikasi: {source.get('filename')} ({page_count} halaman).",
        "source_url": source.get("source_page_url") or source.get("final_download_url") or source.get("download_url"),
        "source_role": source.get("source_role"),
        "temporal_context": source.get("temporal_context"),
        "page_numbers": (1,),
        "bbox_count": 0,
        "viewer_ref": {
            "action": "viewer",
            "source_document_id": source.get("source_document_id"),
            "page_numbers": (1,),
            "bbox_count": 0,
            "can_resolve": True,
        },
        "status": "document",
    }


def _document_title(store, source: dict) -> str:
    catalog = store.config.setting("document_catalog", {}) or {}
    return (catalog.get("titles") or {}).get(source.get("source_role")) or source.get("filename") or source["source_document_id"]


def _document_matches_filters(source: dict, filters: dict) -> bool:
    return all(
        source.get(key) == value for key, value in filters.items() if key in {"source_role", "temporal_context"} and value is not None
    )


def _catalog_search_response(
    corpus_id: str,
    query: str,
    rows: tuple[dict, ...],
    status: str,
    reason: str | None,
    *,
    applied_filters: dict | None = None,
    invalid_filters: tuple[str, ...] = (),
) -> dict:
    return {
        "status": status,
        "public_status": "found" if rows else status,
        "route": "document_catalog" if status != "unsupported_corpus" else "unsupported_corpus",
        "intent": "document_catalog_search",
        "corpus_id": corpus_id,
        "original_query": query,
        "normalized_query": query.strip(),
        "matches": (),
        "reason": reason,
        "required_corpus": None,
        "applied_filters": applied_filters or {},
        "invalid_filters": invalid_filters,
        "results": rows,
        "context_pack": empty_context_pack(reason),
    }


def _search_result(row: dict, routed: dict, context_pack: dict) -> dict:
    viewer_ref = row.get("viewer_ref") or {}
    return {
        "corpus_id": routed["corpus_id"],
        "legal_unit_id": row.get("legal_unit_id"),
        "evidence_id": row["evidence_id"],
        "citation_id": row["evidence_id"],
        "viewer_ref_id": viewer_ref.get("evidence_id"),
        "source_document_id": row.get("source_document_id"),
        "title": row.get("label") or row.get("citation") or row.get("legal_unit_id") or routed["corpus_id"].upper(),
        "snippet": row.get("quoted_text"),
        "document_title": row.get("document_title"),
        "source_url": row.get("source_url"),
        "citation": row.get("citation"),
        "label": row.get("label"),
        "source_role": row.get("source_role"),
        "temporal_context": row.get("temporal_context"),
        "page_numbers": tuple(row.get("page_numbers") or ()),
        "bbox_count": row.get("bbox_count"),
        "viewer_ref": viewer_ref,
        "retrieval_method": routed["route"],
        "reasons": context_pack["validation_reasons"].get(row["evidence_id"]),
        "status": "evidence",
    }


def _ask_route(route: str) -> str:
    return {
        "exact": "legal_reference",
        "structured": "legal_reference",
        "structural_navigation": "structural_navigation",
        "metadata": "metadata_fact",
        "metadata_not_found": "metadata_fact",
        "metadata_scope_unresolved": "metadata_fact",
        "relation": "legal_relation",
        "relation_not_found": "legal_relation",
        "citation_not_found": "legal_reference",
        "structured_not_found": "legal_reference",
        "scope_unresolved": "legal_reference",
        "bm25": "lexical_fallback",
    }.get(route, route)


def _answer_type(route: str, status: str) -> str:
    if status != "answer_ready":
        return "limited_evidence_summary"
    return {
        "metadata_fact": "metadata_fact",
        "legal_relation": "legal_relation",
    }.get(route, "quoted_evidence")


def _source_document_response(store, corpus_id: str, query: str) -> dict | None:
    """Open one explicitly scoped verified source without inventing a citation."""
    config = getattr(store, "config", None)
    strategy = getattr(config, "query_strategy", "generic")
    intent = intent_config_for(strategy, config)
    instrument_decision = resolve_instrument_intent(query, intent, corpus=corpus_id)
    scope = resolve_source_scope(query, strategy=strategy, config=config)
    if not scope.explicit or _has_resolved_legal_target(corpus_id, query):
        return None
    if has_relation_target(query, strategy=strategy, config=config):
        return None
    if instrument_decision.role_family is not None or instrument_decision.target_status == "instrument_resolved_fail_closed":
        return None
    relation_config = intent.get("document_relation", {})
    if not contains_intent_phrase(query, relation_config.get("target_document_terms", ())):
        return None
    if contains_intent_phrase(query, intent.get("instrument_analysis_signals", ())) or contains_intent_phrase(
        query, intent.get("instrument_effect_signals", ())
    ):
        return None
    if has_metadata_target(query, strategy=strategy, config=config, store=store) and _has_metadata_field_target(query, intent):
        return None
    metadata_rows = metadata_lookup(store, query, 1)
    if metadata_rows and metadata_rows[0].get("metadata_field") != "official_title":
        return None
    source = next((row for row in store.source_documents if row.get("source_role") == scope.role), None)
    templates = _answer_templates(store)
    if source is None:
        reason = "source_document_not_found"
        return {
            "status": "insufficient_evidence",
            "route": "source_document",
            "intent": "source_document_lookup",
            "corpus_id": corpus_id,
            "original_query": query,
            "normalized_query": query.strip(),
            "reason": reason,
            "answer_type": "none",
            "answer": templates["insufficient"],
            "document_source": None,
            "citations": (),
            "final_citations": (),
            "historical_citations": (),
            "metadata_support": (),
            "structural_support": (),
            "trace_support": (),
            "viewer_refs": (),
            "metadata_facts": (),
            "evidence": (),
            "warnings": (),
            "insufficient_reasons": (reason,),
        }
    title = _document_title(store, source)
    document_source = {
        "source_document_id": source.get("source_document_id"),
        "source_role": source.get("source_role"),
        "temporal_context": source.get("temporal_context"),
        "document_title": title,
        "viewer_target": {
            "action": "open_document",
            "source_document_id": source.get("source_document_id"),
        },
    }
    return {
        "status": "answer_ready",
        "route": "source_document",
        "intent": "source_document_lookup",
        "corpus_id": corpus_id,
        "original_query": query,
        "normalized_query": query.strip(),
        "reason": None,
        "answer_type": "source_document",
        "answer": f"Naskah sumber terverifikasi: {title}.",
        "document_source": document_source,
        "citations": (),
        "final_citations": (),
        "historical_citations": (),
        "metadata_support": (),
        "structural_support": (),
        "trace_support": (),
        "viewer_refs": (),
        "metadata_facts": (),
        "evidence": (),
        "warnings": ("document_source_has_no_legal_citation",),
        "insufficient_reasons": (),
    }


def _has_metadata_field_target(query: str, intent: dict) -> bool:
    return any(
        field != "official_title" and contains_intent_phrase(query, aliases)
        for field, aliases in intent.get("metadata_fields", {}).items()
    )


def _metadata_fact(row: dict) -> dict:
    return {
        "field": row.get("metadata_field"),
        "answer": row.get("metadata_answer"),
        "evidence_id": row.get("evidence_id"),
    }


def _metadata_support(store, row: dict) -> dict:
    can_resolve = row.get("viewer_ref", {}).get("can_resolve") is True
    authority = _authority_policy(store, row, can_resolve=can_resolve)
    viewer_ref = ((row.get("viewer_ref") or {}) | authority) if can_resolve else None
    return {
        "support_class": "exact_metadata_citation" if can_resolve else "metadata_trace",
        "field": row.get("metadata_field"),
        "answer": row.get("metadata_answer"),
        "evidence_id": row.get("evidence_id"),
        "source_document_id": row.get("source_document_id"),
        "source_role": row.get("source_role"),
        "page_numbers": tuple(row.get("page_numbers") or ()),
        "citation_available": can_resolve,
        "viewer_highlightable": can_resolve,
        "viewer_ref": viewer_ref,
    } | authority


def _metadata_grounding_evidence(store, metadata_grounding_id: str | None) -> dict | None:
    for row in store.metadata_grounding:
        if row.get("metadata_grounding_id") == metadata_grounding_id:
            return {
                "evidence_id": row["metadata_grounding_id"],
                "metadata_grounding": True,
                "legal_unit_id": None,
                "citation": f"Metadata {row.get('source_role')}: {row.get('metadata_field') or 'block'}",
                "hierarchy": (),
                "quoted_text": row.get("quoted_text"),
                "bbox_refs": tuple(row.get("bbox_refs") or ()),
                "bbox_precision": row.get("bbox_precision"),
                "viewer_highlightable": row.get("viewer_highlightable"),
                "page_numbers": tuple(row.get("page_numbers") or ()),
                "source_document_id": row.get("source_document_id"),
                "source_pdf_path": row.get("source_pdf_path"),
                "source_sha256": row.get("source_sha256"),
                "source_role": row.get("source_role"),
                "temporal_context": row.get("temporal_context"),
            }
    return None


def _document_relation_response(store, corpus_id: str, query: str) -> dict | None:
    if store is None:
        return None
    target = _document_relation_target(store, query)
    if target["mode"] is None:
        return None
    templates = _answer_templates(store)
    if target["mode"] == "article":
        return _article_relation_response(store, corpus_id, query, target, templates)
    if target["mode"] == "unsupported":
        return _relation_not_promoted(corpus_id, query, templates)
    support = _document_relation_support(store, target["role"])
    if not support:
        reason = "document_relation_not_found"
        return {
            "status": "insufficient_evidence",
            "route": "document_relation",
            "intent": "document_amendment_relation",
            "corpus_id": corpus_id,
            "original_query": query,
            "normalized_query": query.strip(),
            "matches": (),
            "reason": reason,
            "answer_type": "none",
            "answer": templates["insufficient"],
            "context_pack": empty_context_pack(reason),
            "evidence": (),
            "citations": (),
            "final_citations": (),
            "historical_citations": (),
            "metadata_support": (),
            "structural_support": (),
            "viewer_refs": (),
            "metadata_facts": (),
            "legal_relations": (),
            "document_relations": (),
            "answer_scope": "insufficient_evidence",
            "warnings": (),
            "insufficient_reasons": (reason,),
        }
    relations = tuple(_public_document_relation(row) for row in support)
    return {
        "status": "answer_ready",
        "route": "document_relation",
        "intent": "document_amendment_relation",
        "corpus_id": corpus_id,
        "original_query": query,
        "normalized_query": query.strip(),
        "matches": support,
        "reason": None,
        "answer_type": "document_relation",
        "answer": _document_relation_answer(store, relations),
        "context_pack": empty_context_pack("document_relation_source_role_trace"),
        "evidence": (),
        "citations": (),
        "viewer_refs": (),
        "metadata_facts": (),
        "legal_relations": (),
        "document_relations": relations,
        "answer_scope": "source_role_document_relation",
        "warnings": ("document_relation_not_exact_highlightable",),
        "insufficient_reasons": (),
    }


def _article_relation_response(store, corpus_id: str, query: str, target: dict, templates: dict[str, str]) -> dict:
    support = _article_relation_support(store, target)
    if not support:
        if target.get("target_citation"):
            return _relation_not_promoted(corpus_id, query, templates, reason="relation_target_not_found")
        return _relation_not_promoted(corpus_id, query, templates)
    exact_support = tuple(row for row in support if _is_exact_article_relation(row))
    exact_targets = {row.get("target_legal_unit_id") for row in exact_support}
    trace_support = tuple(row for row in support if not _is_exact_article_relation(row) and row.get("target_legal_unit_id") not in exact_targets)
    public_relations = tuple(_public_article_relation(row) for row in (*exact_support, *trace_support))
    answer_evidence = tuple(row for row in (_article_relation_evidence(store, row) for row in exact_support) if row)
    if not answer_evidence:
        if not trace_support:
            return _relation_not_promoted(corpus_id, query, templates)
        return {
            "status": "limited_answer",
            "route": "document_relation",
            "intent": "document_amendment_relation",
            "corpus_id": corpus_id,
            "original_query": query,
            "normalized_query": query.strip(),
            "matches": support,
            "reason": "relation_trace_only",
            "answer_type": "article_amendment_relation",
            "answer": _article_relation_answer(store, (), trace_support),
            "context_pack": empty_context_pack("relation_trace_only"),
            "evidence": (),
            "citations": (),
            "final_citations": (),
            "historical_citations": (),
            "metadata_support": (),
            "structural_support": (),
            "viewer_refs": (),
            "metadata_facts": (),
            "legal_relations": (),
            "document_relations": (),
            "article_amendment_relations": public_relations,
            "trace_support": tuple(_public_article_relation(row) for row in trace_support),
            "answer_scope": "trace_article_relation",
            "warnings": ("article_relation_trace_only_not_citable",),
            "insufficient_reasons": (),
        }
    citations = _deduplicated_article_relation_citations(store, answer_evidence)
    final_citations = tuple(row for row in citations if row.get("citation_final") is True)
    historical_citations = tuple(row for row in citations if row.get("citation_final") is False)
    viewer_refs = tuple(row["viewer_ref"] for row in answer_evidence if row.get("citation_final") is True)
    partial = bool(trace_support)
    public_evidence = answer_evidence
    public_viewer_refs = viewer_refs
    context_pack = {
        "answer_evidence": public_evidence,
        "supporting_context": (),
        "excluded_results": (),
        "citation_payloads": final_citations,
        "historical_citations": historical_citations,
        "viewer_refs": public_viewer_refs,
        "validation_reasons": {row["evidence_id"]: "article_amendment_relation_exact_source_text" for row in public_evidence},
    }
    return {
        "status": "limited_answer" if partial else "answer_ready",
        "route": "document_relation",
        "intent": "document_amendment_relation",
        "corpus_id": corpus_id,
        "original_query": query,
        "normalized_query": query.strip(),
        "matches": support,
        "reason": None,
        "answer_type": "article_amendment_relation",
        "answer": _article_relation_answer(store, exact_support, trace_support),
        "context_pack": context_pack,
        "evidence": public_evidence,
        "citations": final_citations,
        "final_citations": final_citations,
        "historical_citations": historical_citations,
        "metadata_support": (),
        "structural_support": (),
        "viewer_refs": public_viewer_refs,
        "metadata_facts": (),
        "legal_relations": (),
        "document_relations": (),
        "article_amendment_relations": public_relations,
        "trace_support": tuple(_public_article_relation(row) for row in trace_support),
        "answer_scope": "partial_exact_article_relation" if partial else "exact_article_relation",
        "warnings": ("article_relation_exact_support_partial_trace_omitted",) if trace_support else (),
        "insufficient_reasons": (),
    }


def _relation_not_promoted(corpus_id: str, query: str, templates: dict[str, str], *, reason: str = "relation_not_promoted") -> dict:
    return {
        "status": "insufficient_evidence",
        "route": "document_relation",
        "intent": "document_amendment_relation",
        "corpus_id": corpus_id,
        "original_query": query,
        "normalized_query": query.strip(),
        "matches": (),
        "reason": reason,
        "answer_type": "none",
        "answer": templates["insufficient"],
        "context_pack": empty_context_pack(reason),
        "evidence": (),
        "citations": (),
        "final_citations": (),
        "historical_citations": (),
        "metadata_support": (),
        "structural_support": (),
        "trace_support": (),
        "viewer_refs": (),
        "metadata_facts": (),
        "legal_relations": (),
        "document_relations": (),
        "article_amendment_relations": (),
        "answer_scope": "insufficient_evidence",
        "warnings": (),
        "insufficient_reasons": (reason,),
    }


def _document_relation_target(store, query: str) -> dict:
    config = getattr(store, "config", None)
    intent = intent_config_for(getattr(config, "query_strategy", "generic"), config)
    relation_config = intent.get("document_relation", {})
    normalized = normalize_intent_text(query)
    if not normalized:
        return {"mode": None}
    relation_intent = classify_relation_intent(store, query)
    relation_family = (relation_config.get("relation_families") or {}).get(relation_intent.relation_type, {})
    relation_types = tuple(relation_family.get("relation_types") or ())
    source_scope = resolve_source_scope(query, strategy=getattr(config, "query_strategy", "generic"), config=config)
    references = parse_legal_references(getattr(config, "corpus_id", ""), query)
    relation_signal = bool(relation_family) or contains_intent_phrase(query, relation_config.get("change_terms", ()))
    add_signal = contains_intent_phrase(query, relation_config.get("add_terms", ()))
    if source_scope.explicit and len(references) == 1 and not relation_signal and not add_signal:
        return {"mode": None}
    mentioned_roles = source_roles_for_query(query, strategy=getattr(config, "query_strategy", "generic"), config=config)
    amendment_role = next((role for role in mentioned_roles if role.startswith("amendment_")), None)
    amendment_signal = amendment_role in set(getattr(config, "source_roles", ()) or ()) or contains_intent_phrase(
        query, relation_config.get("source_terms", ())
    )
    if relation_intent.relation_type == "RENAME_PROVISION":
        amendment_signal = True
    target_original = contains_intent_phrase(query, relation_config.get("target_document_terms", ()))
    article_detail = contains_intent_phrase(query, relation_config.get("article_detail_terms", ()))
    source_less_delete = relation_intent.relation_type == "DELETE_OR_REMOVE_PROVISION" and article_detail
    if relation_intent.relation_type == "DELETE_OR_REMOVE_PROVISION" and not article_detail:
        return {"mode": None}
    if not references and amendment_role and "original_historical" in mentioned_roles:
        return {"mode": "document", "role": amendment_role}
    if not (relation_signal or add_signal) or (not amendment_signal and not source_less_delete):
        return {"mode": None}
    target_citation = _article_relation_target_citation(
        getattr(config, "corpus_id", None), query, prefer_last=relation_intent.relation_type == "RENAME_PROVISION"
    )
    if add_signal:
        return {
            "mode": "article",
            "role": amendment_role,
            "relation_types": tuple(
                relation_type for relation_type in relation_config.get("schema_only_relation_types", ()) if relation_type not in {"RENAMES", "RENUMBERED_TO"}
            ),
            "target_citation": target_citation,
        }
    if relation_intent.relation_type == "RENAME_PROVISION":
        return {
            "mode": "article",
            "role": amendment_role,
                "relation_types": ("RENAMES", "RENUMBERED_TO"),
            "target_citation": target_citation,
        }
    if contains_intent_phrase(query, relation_config.get("unsupported_detail_terms", ())):
        return {"mode": "unsupported"}
    if article_detail:
        if relation_intent.relation_type == "MODIFY_PROVISION":
            relation_types = ("MODIFIES",)
        return {
            "mode": "article",
            "role": amendment_role,
            "relation_types": relation_types or ("MODIFIES", "DELETES"),
            "target_citation": target_citation,
        }
    if amendment_role and amendment_role.startswith("amendment_"):
        return {"mode": "document", "role": amendment_role}
    if target_original:
        return {"mode": "document", "role": "original_historical"}
    return {"mode": None}


def _document_relation_support(store, target: str) -> tuple[dict, ...]:
    if target == "original_historical":
        return tuple(
            row | {"route_sources": ("document_relation",)}
            for row in store.document_relations
            if row.get("relation_type") == "AMENDED_BY" and row.get("source_role") == "original_historical"
        )
    return tuple(
        row | {"route_sources": ("document_relation",)}
        for row in store.document_relations
        if row.get("relation_type") == "AMENDS" and row.get("source_role") == target
    )


def _article_relation_support(store, target: dict) -> tuple[dict, ...]:
    role = target.get("role")
    relation_types = set(target.get("relation_types") or ())
    roles = {role} if role else {row for row in getattr(store.config, "source_roles", ()) if str(row).startswith("amendment_")}
    if not roles:
        return ()
    return tuple(
        row | {"route_sources": ("article_amendment_relation",)}
        for row in store.article_amendment_relations
        if row.get("source_role") in roles
        and row.get("relation_type") in relation_types
        and row.get("runtime_loadable") is True
        and _article_relation_matches_target(store, row, target.get("target_citation"))
    )


def _article_relation_target_citation(corpus_id: str | None, query: str, *, prefer_last: bool = False) -> str | None:
    if not corpus_id:
        return None
    if prefer_last:
        references = parse_legal_references(corpus_id, query)
        if references:
            matches = list(re.finditer(r"\bpasal\s*([0-9]+[A-Za-z]?)(?:\s*ayat\s*\(\s*(\d+)\s*\))?", query or "", re.IGNORECASE))
            if matches:
                match = matches[-1]
                reference = f"Pasal {match.group(1)}"
                return f"{reference} ayat ({match.group(2)})" if match.group(2) else reference
            return str(references[-1]["reference"])
    ref = parse_legal_reference(corpus_id, query)
    pasal = ref.get("pasal")
    ayat = ref.get("ayat")
    return f"{pasal} ayat {ayat}" if pasal and ayat else pasal


def _article_relation_matches_target(store, row: dict, target_citation: str | None) -> bool:
    if not target_citation:
        return True
    target = _normalize_article_target(target_citation)
    citation = _normalize_article_target(row.get("target_reference") or row.get("new_reference") or row.get("target_citation"))
    if target == citation:
        return True
    unit: dict = next((unit for unit in store.legal_units if unit.get("legal_unit_id") == row.get("target_legal_unit_id")), {})
    labels = [unit.get("unit_label"), *(unit.get("hierarchy") or ())]
    return target in {_normalize_article_target(label) for label in labels}


def _normalize_article_target(value: object) -> str:
    return " ".join(str(value or "").casefold().replace("(", "").replace(")", "").split())


def _article_relation_evidence(store, relation: dict) -> dict | None:
    if not _is_exact_article_relation(relation):
        return None
    row = store.get(relation["evidence_id"])
    if row is None:
        return None
    if store.lineage_error(row):
        return None
    proof_bbox_refs = tuple(relation.get("bbox_refs") or row.get("bbox_refs") or ())
    proof_text_span_ids = tuple(relation.get("text_span_ids") or row.get("text_span_ids") or ())
    target_bbox_refs = tuple(relation.get("target_bbox_refs") or ())
    proof_bboxes = store.bboxes_for_refs(proof_bbox_refs)
    if not proof_bboxes or not set(proof_bbox_refs) <= {bbox["bbox_id"] for bbox in proof_bboxes}:
        return None
    if row.get("bbox_precision") != "exact" or row.get("viewer_highlightable") is not True:
        return None
    return {
        **row,
        "bbox_refs": proof_bbox_refs,
        "text_span_ids": proof_text_span_ids,
        "relation_source_proof_bbox_refs": proof_bbox_refs,
        "relation_source_proof_text_span_ids": proof_text_span_ids,
        "relation_target_bbox_refs": target_bbox_refs,
        "relation_target_text_span_ids": tuple(relation.get("target_text_span_ids") or ()),
        "quoted_text": relation.get("quoted_text") or row.get("quoted_text"),
        "bbox_count": len(proof_bboxes),
        "route_sources": ("article_amendment_relation",),
        "article_amendment_relation": relation,
        "viewer_ref": {
            "action": "viewer",
            "evidence_id": row["evidence_id"],
            "source_document_id": row.get("source_document_id"),
            "page_numbers": tuple(row.get("page_numbers") or ()),
            "text_span_ids": proof_text_span_ids,
            "bbox_count": len(proof_bboxes),
            "bbox_refs": proof_bbox_refs,
            "source_proof_text_span_ids": proof_text_span_ids,
            "source_proof_bbox_refs": proof_bbox_refs,
            "target_text_span_ids": tuple(relation.get("target_text_span_ids") or ()),
            "target_bbox_refs": target_bbox_refs,
            "can_resolve": True,
        },
    }


def _relation_for_evidence(store, evidence_id: str | None, relation_id: str | None = None) -> dict | None:
    if not evidence_id:
        return None
    return next(
        (
            row
            for row in getattr(store, "article_amendment_relations", ())
            if row.get("evidence_id") == evidence_id and (relation_id is None or row.get("relation_id") == relation_id)
        ),
        None,
    )


def _is_exact_article_relation(row: dict) -> bool:
    return (
        row.get("support_class") == "exact_article_relation"
        and row.get("grounding_level") == "exact_source_text"
        and row.get("bbox_precision") == "exact"
        and row.get("viewer_highlightable") is True
        and row.get("citation_available") is True
    )


def _public_document_relation(row: dict) -> dict:
    return {
        "relation_id": row.get("relation_id"),
        "relation_type": row.get("relation_type"),
        "source_document_id": row.get("source_document_id"),
        "source_role": row.get("source_role"),
        "target_source_role": row.get("target_source_role"),
        "target_document_id": row.get("target_document_id"),
        "support_type": row.get("support_type"),
        "reason": row.get("reason"),
        "highlightable": row.get("viewer_highlightable") is True,
    }


def public_article_relation(row: dict) -> dict:
    return {
        "relation_id": row.get("relation_id"),
        "relation_type": row.get("relation_type"),
        "source_document_id": row.get("source_document_id"),
        "source_role": row.get("source_role"),
        "source_legal_unit_id": row.get("source_legal_unit_id"),
        "source_legal_unit_role": row.get("source_legal_unit_role"),
        "source_label": row.get("source_label"),
        "source_reference": row.get("old_reference") or row.get("source_reference"),
        "source_reference_range": row.get("old_reference_range") or row.get("source_reference_range"),
        "source_reference_range_kind": row.get("old_reference_range_kind") or row.get("source_reference_range_kind"),
        "target_legal_unit_id": row.get("target_legal_unit_id"),
        "target_label": row.get("target_label") or row.get("target_citation"),
        "target_citation": row.get("target_citation"),
        "target_reference": row.get("new_reference") or row.get("target_reference"),
        "target_reference_range": row.get("new_reference_range") or row.get("target_reference_range"),
        "target_reference_range_kind": row.get("new_reference_range_kind") or row.get("target_reference_range_kind"),
        "target_source_role": row.get("target_source_role"),
        "evidence_id": row.get("evidence_id"),
        "bbox_refs": tuple(row.get("bbox_refs") or ()),
        "source_proof_text_span_ids": tuple(row.get("text_span_ids") or ()),
        "source_proof_bbox_refs": tuple(row.get("bbox_refs") or ()),
        "target_bbox_refs": tuple(row.get("target_bbox_refs") or ()),
        "target_precision": row.get("target_precision"),
        "source_support_exact": row.get("source_support_exact") is True,
        "text_span_ids": tuple(row.get("text_span_ids") or ()),
        "target_text_span_ids": tuple(row.get("target_text_span_ids") or ()),
        "support_class": row.get("support_class"),
        "grounding_level": row.get("grounding_level"),
        "authority_kind": row.get("authority_kind"),
        "citation_final": row.get("citation_final") is True,
        "recovery_capability": row.get("recovery_capability"),
        "recovery_status": row.get("recovery_status"),
        "target_geometry_method": row.get("target_geometry_method"),
        "trace_only_reason": row.get("trace_only_reason"),
        "citation_available": row.get("citation_available") is True,
        "viewer_highlightable": row.get("viewer_highlightable") is True,
    }


_public_article_relation = public_article_relation


def _article_relation_citation(row: dict) -> dict:
    return {
        "corpus_id": row.get("corpus_id"),
        "evidence_id": row["evidence_id"],
        "legal_unit_id": row.get("legal_unit_id"),
        "source_document_id": row.get("source_document_id"),
        "citation": row.get("citation"),
        "label": row.get("citation"),
        "hierarchy": tuple(row.get("hierarchy") or ()),
        "quoted_text": row.get("quoted_text"),
        "source_role": row.get("source_role"),
        "temporal_context": row.get("temporal_context"),
        "source_pdf_path": row.get("source_pdf_path"),
        "source_sha256": row.get("source_sha256"),
        "page_numbers": tuple(row.get("page_numbers") or ()),
        "bbox_count": row.get("bbox_count"),
        "viewer_ref": row.get("viewer_ref"),
        "evidence_status": row.get("status"),
    }


def _deduplicated_article_relation_citations(store, rows: tuple[dict, ...]) -> tuple[dict, ...]:
    grouped: dict[object, dict] = {}
    for row in rows:
        citation = _article_relation_citation(row)
        key = citation.get("evidence_id")
        grouped.setdefault(key, citation)
    return tuple(_citation_with_authority(store, row) for row in grouped.values())


def _document_relation_answer(store, relations: tuple[dict, ...]) -> str:
    intent = store.config.setting("intent_config", {}) or {}
    relation_config = intent.get("document_relation", {}) or {}
    labels = intent.get("source_role_labels", {}) or {}
    prefix = str(relation_config.get("source_role_label_prefix", ""))
    amendment_roles = [_document_relation_amendment_role(row) for row in relations]
    amendment_roles = [role for role in amendment_roles if role]
    names = [f"{prefix}{labels.get(role, role)}" for role in amendment_roles]
    if len(names) > 1:
        listed = ", ".join(names[:-1]) + f", dan {names[-1]}"
        return str(relation_config.get("document_answer_template", "{relations}")).format(relations=listed)
    name = names[0] if names else "Perubahan"
    return str(relation_config.get("single_document_answer_template", "{relation}")).format(relation=name)


def _document_relation_amendment_role(row: dict) -> str | None:
    for key in ("source_role", "target_source_role"):
        role = str(row.get(key) or "")
        if role.startswith("amendment_"):
            return role
    return None


def _article_relation_answer(store, relations: tuple[dict, ...], trace_support: tuple[dict, ...]) -> str:
    relation_config = (store.config.setting("intent_config", {}) or {}).get("document_relation", {}) or {}

    def labels_for(rows: tuple[dict, ...]) -> list[str]:
        by_target: dict[str, set[str]] = {}
        for row in rows:
            target = str(row.get("new_reference") or row.get("target_citation") or "")
            if target:
                by_target.setdefault(target, set()).add(str(row.get("relation_type") or ""))
        labels = []
        for target in sorted(by_target, key=_legal_reference_sort_key):
            types = by_target[target]
            suffix = " / ".join(relation_labels[relation] for relation in ("DELETES", "MODIFIES", "RENAMES", "RENUMBERED_TO") if relation in types)
            labels.append(f"{target} ({suffix})" if suffix else target)
        return labels

    relation_labels = {
        "DELETES": "dihapus",
        "MODIFIES": "diubah",
        "RENAMES": "dinomori ulang",
        "RENUMBERED_TO": "dinomori ulang",
    }
    exact_labels = labels_for(tuple(relations))
    trace_labels = labels_for(tuple(trace_support))
    if not exact_labels and not trace_labels:
        listed = "relasi yang diminta"
        return str(relation_config.get("article_trace_answer_template", "Relasi trace: {relations}. ")).format(relations=listed)
    if exact_labels and trace_labels:
        return str(
            relation_config.get("article_mixed_answer_template", "Relasi exact: {exact_relations}. Relasi trace: {trace_relations}. ")
        ).format(exact_relations=", ".join(exact_labels), trace_relations=", ".join(trace_labels))
    if trace_labels:
        return str(relation_config.get("article_trace_answer_template", "Relasi trace: {relations}. ")).format(
            relations=", ".join(trace_labels)
        )
    return str(relation_config.get("article_exact_answer_template", "Relasi exact: {relations}. ")).format(
        relations=", ".join(exact_labels)
    )


def _legal_reference_sort_key(value: str) -> tuple[int, str]:
    match = re.search(r"Pasal\s*(\d+)([A-Za-z]?)", value)
    if not match:
        return (10**9, value.casefold())
    return (int(match.group(1)), match.group(2).casefold())


def _instrument_intent_context(store, query: str) -> tuple[dict | None, str, str] | None:
    if store is None:
        return None
    config = getattr(store, "config", None)
    intent = intent_config_for(getattr(config, "structured_strategy", "generic"), config)
    decision = resolve_instrument_intent(query, intent, corpus=getattr(config, "corpus_id", ""))
    if metadata_lookup(store, query, 1) and decision.reason not in {"analysis_metadata_conflict", "unsupported_analysis_intent"}:
        return None
    if decision.target_status == "not_instrument":
        return None
    if decision.target_status == "instrument_unresolved":
        return None, "instrument_unresolved", decision.reason
    row = next(
        (
            item
            for item in store.evidence
            if item.get("source_role") == decision.amendment and item.get("citation") == decision.target_citation
        ),
        None,
    )
    if row is None:
        return None, "instrument_unresolved", decision.reason
    row = row | {
        "route_sources": ("structured",),
        "candidate_type": f"instrument_{decision.role_family}_candidate",
    }
    accepted, _ = validate_answer_candidate(store, row)
    if accepted:
        return row, "instrument_resolved_answerable", "answer_evidence"
    return (
        row | {"forced_rejection_reason": "instrument_resolved_fail_closed"},
        "instrument_resolved_fail_closed",
        "instrument_resolved_fail_closed",
    )


def _is_instrument_unit(store, unit: dict) -> bool:
    schema: dict = getattr(getattr(store, "config", None), "setting", lambda *args: {})("schema", {}) or {}
    return unit.get("unit_type") in set(schema.get("instrument_unit_types") or ())


def _source_anomaly_response(store, corpus_id: str, query: str) -> dict | None:
    if store is None:
        return None
    if not _is_source_anomaly_query(store, query):
        return None
    conflict = _matched_source_conflict(store, query)
    if conflict is None:
        return _source_anomaly_fallback()
    support = _source_conflict_support(store, conflict)
    reasons = _source_conflict_reasons(store, query)
    exact_provenance = bool(support["evidence"])
    trace_only = bool(support["trace_support"]) and not exact_provenance
    answer = _source_anomaly_answer(store, conflict, query, exact_provenance=exact_provenance, trace_only=trace_only)
    return {
        "status": "limited_answer" if exact_provenance or trace_only else "insufficient_evidence",
        "route": "source_anomaly_explanation",
        "intent": "structured_lookup",
        "answer_type": "source_conflict_provenance" if exact_provenance or trace_only else "none",
        "answer": answer,
        "context_pack": support["context_pack"],
        "evidence": support["evidence"],
        "citations": (),
        "final_citations": (),
        "historical_citations": (),
        "metadata_support": (),
        "structural_support": (),
        "viewer_refs": (),
        "metadata_facts": (),
        "legal_relations": (),
        "trace_support": support["trace_support"],
        "answer_scope": support["answer_scope"],
        "warnings": support["warnings"],
        "insufficient_reasons": tuple(dict.fromkeys(reasons)) if not exact_provenance else (),
        "source_conflict": _public_source_conflict(conflict),
    }


def _matched_source_conflict(store, query: str) -> dict | None:
    folded = (query or "").casefold()
    if not _is_source_anomaly_query(store, query):
        return None
    intent = _source_conflict_intent(store)
    matches = [(score, row) for row in store.source_conflicts if (score := _source_conflict_match_score(store, row, folded, intent)) > 0]
    if not matches:
        return None
    matches.sort(key=lambda item: (-item[0], item[1].get("source_conflict_id") or ""))
    return matches[0][1]


def _is_source_anomaly_query(store, query: str) -> bool:
    folded = (query or "").casefold()
    intent = _source_conflict_intent(store)
    terms = tuple(str(term).casefold() for term in intent.get("query_terms") or ())
    if not any(_query_contains_term(folded, term) for term in terms):
        return False
    unresolved_terms = tuple(str(term).casefold() for term in intent.get("unresolved_query_terms") or ())
    if any(_query_contains_term(folded, term) for term in unresolved_terms):
        return True
    return False


def _source_anomaly_fallback() -> dict:
    return {
        "status": "insufficient_evidence",
        "route": "source_anomaly_explanation",
        "intent": "structured_lookup",
        "answer_type": "none",
        "answer": "Bukti tidak cukup untuk mengaitkan pertanyaan ini dengan catatan konflik sumber tertentu.",
        "context_pack": empty_context_pack("source_anomaly_unresolved"),
        "evidence": (),
        "citations": (),
        "viewer_refs": (),
        "metadata_facts": (),
        "legal_relations": (),
        "answer_scope": "insufficient_evidence",
        "warnings": (),
        "insufficient_reasons": ("source_anomaly", "source_anomaly_unresolved"),
        "source_conflict": None,
    }


def _source_conflict_reasons(store, query: str) -> list[str]:
    intent = _source_conflict_intent(store)
    folded = (query or "").casefold()
    reasons = ["source_anomaly", "canonical_conflict"]
    for rule in intent.get("reason_rules") or ():
        terms = tuple(str(term).casefold() for term in rule.get("query_terms") or ())
        if any(_query_contains_term(folded, term) for term in terms):
            reasons.extend(str(reason) for reason in rule.get("reasons") or ())
            return reasons
    reasons.extend(str(reason) for reason in intent.get("default_reasons") or ())
    return reasons


def _source_conflict_match_score(store, conflict: dict, folded_query: str, intent: dict) -> int:
    exclusions = tuple(str(term).casefold() for term in conflict.get("query_exclusion_terms") or ())
    if any(_query_contains_term(folded_query, term) for term in exclusions):
        return 0
    required = tuple(str(term).casefold() for term in conflict.get("query_required_terms") or ())
    anchors = {str(term).casefold() for term in conflict.get("query_anchor_terms") or ()}
    source_role = str(_source_document_meta(store, conflict.get("source_document_id")).get("source_role") or "")
    role_label = str((intent.get("role_labels") or {}).get(source_role) or source_role).casefold()
    role_anchor_match = _query_contains_term(folded_query, role_label) and any(
        _query_contains_term(folded_query, anchor) for anchor in anchors
    )
    source_marker_context = conflict.get("source_anomaly_kind") == "source_marker_sequence_anomaly" and role_anchor_match
    if required and not any(_query_contains_term(folded_query, term) for term in required) and not source_marker_context:
        return 0
    explicit_anchor_match = any(
        len(anchor.split()) > 1 and _query_contains_term(folded_query, anchor) for anchor in anchors
    )
    semantic_required = tuple(term for term in required if term not in anchors or "pasal" not in term)
    marker_context = role_anchor_match or any(_query_contains_term(folded_query, term) for term in semantic_required)
    if (
        semantic_required
        and not any(_query_contains_term(folded_query, term) for term in semantic_required)
        and not role_anchor_match
        and not (
            conflict.get("source_anomaly_kind") == "source_marker_sequence_anomaly"
            and (explicit_anchor_match or role_anchor_match)
        )
    ):
        return 0
    if conflict.get("source_anomaly_kind") == "source_marker_sequence_anomaly" and not marker_context:
        return 0
    score = 0
    if _query_contains_term(folded_query, role_label):
        score += 4
    for token in (str(value).casefold() for value in (conflict.get("query_anchor_terms") or conflict.get("anchor_terms") or ())):
        if _query_contains_term(folded_query, token):
            score += 3
    if score:
        return score
    haystack = (
        " ".join(
            str(value or "")
            for value in (
                conflict.get("source_conflict_id"),
                conflict.get("type"),
                conflict.get("source_anomaly_kind"),
                conflict.get("source_mapping_kind"),
                conflict.get("classification"),
                conflict.get("source_document_id"),
                role_label,
            )
        )
        .replace("_", " ")
        .casefold()
    )
    query_tokens = _meaningful_conflict_tokens(folded_query, intent)
    conflict_tokens = {token for token in re.findall(r"[a-z0-9]+", haystack) if len(token) > 2}
    overlap = query_tokens & conflict_tokens
    return len(overlap) if len(overlap) >= 2 else 0


def _query_contains_term(query: str, term: str) -> bool:
    """Match policy terms on token boundaries so Pasal suffixes cannot alias."""
    if not term:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", query) is not None


def _source_document_meta(store, source_document_id: object) -> dict:
    return next(
        (row for row in store.source_documents if row.get("source_document_id") == source_document_id),
        {},
    )


def _source_document_by_id(store, source_document_id: object) -> dict | None:
    return next((row for row in store.source_documents if row.get("source_document_id") == source_document_id), None)


def _meaningful_conflict_tokens(text: str, intent: dict) -> set[str]:
    generic = set(intent.get("generic_tokens") or ())
    return {token for token in re.findall(r"[a-z0-9]+", text) if len(token) > 2 and token not in generic}


def _source_conflict_intent(store) -> dict:
    return store.config.setting("source_conflict_intent", {})


def _public_source_conflict(conflict: dict) -> dict:
    return {
        "source_conflict_id": conflict.get("source_conflict_id"),
        "type": conflict.get("type"),
        "classification": conflict.get("classification"),
        "source_document_id": conflict.get("source_document_id"),
        "status": conflict.get("status"),
    } | _source_conflict_contract_fields(conflict)


def _source_conflict_support(store, conflict: dict) -> dict:
    evidence_rows = tuple(
        row | {"route_sources": ("exact",), "candidate_type": "source_conflict_provenance"}
        for evidence_id in conflict.get("evidence_ids") or ()
        if (row := store.get(evidence_id)) is not None
    )
    context_pack = assemble_context_pack(store, evidence_rows) if evidence_rows else empty_context_pack("source_anomaly")
    evidence = context_pack["answer_evidence"]
    synthetic_support = _synthetic_source_conflict_support(store, conflict) if not evidence else None
    trace_support: tuple[dict, ...] = ()
    answer_scope = "insufficient_evidence"
    if evidence:
        trace_support = tuple(_citation_with_authority(store, row, conflict=conflict) for row in evidence)
        answer_scope = "source_conflict_exact_provenance"
    elif synthetic_support is not None:
        trace_support = tuple(synthetic_support["citations"])
        answer_scope = "source_conflict_exact_provenance"
    else:
        trace_support = (_source_conflict_trace_support(store, conflict, context_pack["validation_reasons"]),)
        answer_scope = "source_conflict_trace" if trace_support else "insufficient_evidence"
    viewer_refs = (
        tuple(
            ref | _authority_policy(store, evidence[0], can_resolve=ref.get("can_resolve") is True, conflict=conflict)
            for ref in context_pack["viewer_refs"]
        )
        if evidence
        else synthetic_support["viewer_refs"]
        if synthetic_support is not None
        else ()
    )
    citations = ()
    public_context_pack = (
        context_pack | {"citation_payloads": (), "viewer_refs": ()}
        if evidence
        else synthetic_support["context_pack"]
        if synthetic_support is not None
        else context_pack
    )
    return {
        "context_pack": public_context_pack,
        "evidence": synthetic_support["evidence"] if synthetic_support is not None else evidence,
        "citations": citations,
        "viewer_refs": viewer_refs,
        "trace_support": trace_support,
        "answer_scope": answer_scope,
        "warnings": (("source_conflict_not_final_legal_authority",) if evidence or synthetic_support is not None or trace_support else ()),
    }


def _source_conflict_trace_support(store, conflict: dict, validation_reasons: dict) -> dict:
    first_reason = next(iter(validation_reasons.values()), None)
    authority = _authority_policy(store, conflict, can_resolve=False, conflict=conflict)
    return (
        {
            "support_class": authority.get("support_kind") or "source_conflict_trace",
            "source_conflict_id": conflict.get("source_conflict_id"),
            "type": conflict.get("type"),
            "classification": conflict.get("classification"),
            "source_document_id": conflict.get("source_document_id"),
            "page_numbers": tuple(conflict.get("page_numbers") or conflict.get("affected_pages") or ()),
            "text_span_ids": tuple(conflict.get("text_span_ids") or ()),
            "evidence_ids": tuple(conflict.get("evidence_ids") or ()),
            "bbox_ids": tuple(conflict.get("bbox_ids") or ()),
            "citation_available": False,
            "viewer_highlightable": False,
            "viewer_ref": None,
            "failure_reason": conflict.get("failure_reason") or first_reason or "source_conflict_trace_only",
        }
        | _source_conflict_contract_fields(conflict)
        | authority
    )


def _source_anomaly_answer(store, conflict: dict, query: str, *, exact_provenance: bool, trace_only: bool) -> str:
    intent = _source_conflict_intent(store)
    decision = conflict.get("resolution_decision") or {}
    folded = (query or "").casefold()
    classification = conflict.get("classification") or "source_conflict_recorded"
    summary = str(conflict.get("provenance_summary") or classification).strip()
    authority_policy = str(
        conflict.get("final_authority_policy")
        or "Sistem menampilkan provenance sumber ini sebagai jejak audit, bukan kesimpulan hukum final."
    ).strip()
    role_label = _source_conflict_role_label(store, conflict)
    reviewer_suffix = _source_conflict_reviewer_suffix(decision.get("reviewer_decision"))
    policy = conflict.get("source_anomaly_policy") or {}
    if exact_provenance:
        provenance_note = _source_conflict_provenance_note(conflict)
        return _source_anomaly_policy_answer(
            policy,
            role_label=role_label,
            summary=summary,
            authority_policy=authority_policy,
            provenance_note=provenance_note,
            reviewer_suffix=reviewer_suffix,
        )
    if trace_only:
        return _source_anomaly_policy_answer(
            policy,
            role_label=role_label,
            summary=summary,
            authority_policy=authority_policy,
            provenance_note="Jejak sumber tersedia, tetapi belum memenuhi syarat sitasi atau highlight exact.",
            reviewer_suffix=reviewer_suffix,
        )
    values = {
        "classification": classification,
        "reviewer_decision": decision.get("reviewer_decision") or "Reviewer decision unavailable",
    }
    for rule in intent.get("answer_rules") or ():
        terms = tuple(str(term).casefold() for term in rule.get("query_terms") or ())
        types = tuple(str(item) for item in rule.get("conflict_types") or ())
        if (terms and any(term in folded for term in terms)) or (types and conflict.get("type") in types):
            return str(rule.get("template") or "").format_map(values)
    return str(intent.get("default_answer_template") or "").format_map(values)


def _source_anomaly_policy_answer(
    policy: dict,
    *,
    role_label: str,
    summary: str,
    authority_policy: str,
    provenance_note: str,
    reviewer_suffix: str,
) -> str:
    values = {
        "anomaly_kind": policy.get("anomaly_kind") or "source_anomaly_provenance",
        "mapping_kind": policy.get("mapping_kind") or "source_anomaly_provenance",
        "role_label": role_label,
        "summary": summary,
        "authority_policy": authority_policy,
        "provenance_note": provenance_note,
        "reviewer_suffix": reviewer_suffix,
    }
    template = str(
        policy.get("public_wording_template")
        or "Catatan provenance sumber ({anomaly_kind}) pada {role_label}: {summary}. {authority_policy} {provenance_note}{reviewer_suffix}"
    )
    return template.format_map(values)


def _source_conflict_role_label(store, conflict: dict) -> str:
    source = _source_document_meta(store, conflict.get("source_document_id"))
    source_role = str(source.get("source_role") or "")
    labels = _source_conflict_intent(store).get("role_labels") or {}
    return str(labels.get(source_role) or source_role or conflict.get("source_document_id") or "sumber historis")


def _source_conflict_reviewer_suffix(reviewer_decision: object) -> str:
    text = str(reviewer_decision or "").strip()
    if not text:
        return ""
    return f" Reviewer decision: {text}."


def _source_conflict_contract_fields(conflict: dict) -> dict:
    raw_bbox_ids = tuple(conflict.get("raw_provenance_bbox_ids") or ())
    raw_text_span_ids = tuple(conflict.get("raw_provenance_text_span_ids") or ())
    blocked_text_span_ids = tuple(conflict.get("blocked_raw_provenance_text_span_ids") or ())
    fields = {
        "final_evidence_available": bool(conflict.get("final_evidence_available")),
        "source_anomaly_kind": conflict.get("source_anomaly_kind"),
        "source_mapping_kind": conflict.get("source_mapping_kind"),
        "provenance_bbox_status": conflict.get("provenance_bbox_status"),
        "provenance_highlight_scope": conflict.get("provenance_highlight_scope"),
        "raw_provenance_bbox_count": len(raw_bbox_ids),
        "raw_provenance_text_span_count": len(raw_text_span_ids),
        "blocked_raw_provenance_text_span_count": len(blocked_text_span_ids),
        "blocked_raw_provenance_reason": conflict.get("blocked_raw_provenance_reason"),
    }
    fields.update(_source_mapping_semantics(conflict))
    return fields


def _source_mapping_semantics(conflict: dict) -> dict:
    if conflict.get("source_anomaly_kind") != "renumbering_provenance":
        return {}
    return {
        "relation_type": "renumbered_to",
        "substantive_change": False,
        "anomaly": False,
        "source_conflict": False,
    }


def _source_conflict_provenance_note(conflict: dict) -> str:
    if conflict.get("provenance_highlight_scope") == "all_relevant_spans":
        return "Highlight viewer tersedia untuk semua span provenance exact yang relevan."
    if conflict.get("provenance_highlight_scope") == "anchor_span_only":
        return "Highlight viewer saat ini terbatas pada span anchor exact yang tersedia; span anomali lain tetap tercatat sebagai trace tanpa highlight palsu."
    return "Viewer tidak menampilkan highlight exact karena belum ada span/BBox provenance yang aman."


def _source_conflict_viewer_evidence(store, evidence_id: str | None) -> tuple[dict | None, list[dict] | None]:
    if store is None or not evidence_id:
        return None, None
    conflict = next((row for row in store.source_conflicts if row.get("source_conflict_id") == evidence_id), None)
    if conflict is None:
        return None, None
    synthetic = _synthetic_source_conflict_support(store, conflict)
    if synthetic is None:
        return None, None
    return synthetic["evidence"][0], list(synthetic["bboxes"])


def _synthetic_source_conflict_support(store, conflict: dict) -> dict | None:
    bboxes = tuple(store.exact_bboxes_for_text_spans(tuple(conflict.get("text_span_ids") or ())))
    if not bboxes:
        return None
    evidence = _synthetic_source_conflict_evidence(store, conflict, bboxes)
    viewer_ref = {
        "action": "viewer",
        "evidence_id": evidence["evidence_id"],
        "source_document_id": evidence.get("source_document_id"),
        "page_numbers": evidence["page_numbers"],
        "bbox_count": len(bboxes),
        "can_resolve": True,
    } | _authority_policy(store, evidence, can_resolve=True, conflict=conflict)
    citation = evidence | {
        "label": conflict.get("classification"),
        "document_title": _document_title(store, _source_document_meta(store, conflict.get("source_document_id"))),
        "viewer_ref": viewer_ref,
        "bbox_count": len(bboxes),
        "evidence_status": conflict.get("status"),
    }
    return {
        "context_pack": {
            "answer_evidence": (evidence,),
            "supporting_context": (),
            "excluded_results": (),
            "citation_payloads": (),
            "viewer_refs": (viewer_ref,),
            "validation_reasons": {evidence["evidence_id"]: "source_conflict_exact_span_bbox"},
        },
        "evidence": (evidence,),
        "citations": (_citation_with_authority(store, citation, conflict=conflict),),
        "viewer_refs": (viewer_ref,),
        "bboxes": bboxes,
    }


def _synthetic_source_conflict_evidence(store, conflict: dict, bboxes: tuple[dict, ...]) -> dict:
    source = _source_document_meta(store, conflict.get("source_document_id"))
    pages = tuple(dict.fromkeys(int(row["page_number"]) for row in bboxes if row.get("page_number")))
    quoted_text = "\n".join(dict.fromkeys(str(row.get("text") or "").strip() for row in bboxes if str(row.get("text") or "").strip()))
    return {
        "evidence_id": conflict.get("source_conflict_id"),
        "legal_unit_id": None,
        "citation": f"Source anomaly: {conflict.get('classification')}",
        "quoted_text": quoted_text,
        "bbox_refs": tuple(row["bbox_id"] for row in bboxes if row.get("bbox_id")),
        "bbox_precision": "exact",
        "viewer_highlightable": True,
        "page_numbers": pages,
        "source_document_id": conflict.get("source_document_id"),
        "source_pdf_path": bboxes[0].get("source_pdf_path"),
        "source_sha256": bboxes[0].get("source_sha256"),
        "source_role": source.get("source_role"),
        "temporal_context": source.get("temporal_context"),
    }
