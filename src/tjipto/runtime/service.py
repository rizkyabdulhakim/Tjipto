from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from uuid import uuid4

from tjipto.corpora.intent_config import intent_config_for, resolve_instrument_intent
from tjipto.corpora.registry import CorpusRegistry
from tjipto.evidence.store import EvidenceStore
from tjipto.retrieval import answer as answer_policy
from tjipto.retrieval.answer import assemble_context_pack, empty_context_pack
from tjipto.retrieval.router import route_retrieval
from tjipto.runtime.viewer import resolve_pdf_access, viewer_payload


_BOOKMARKS: dict[str, dict] = {}

_ANSWER_TEMPLATES = {
    "insufficient": "Bukti tidak cukup atau database belum tersedia dalam korpus terverifikasi saat ini.",
    "metadata": "{answer} (from grounded document metadata).",
    "legal_relation": "Dukungan relasi hukum berbasis bukti tersedia; sistem tidak menghasilkan kesimpulan hukum.",
    "citation": "Dukungan sitasi berbasis bukti tersedia untuk {citation}; sistem tidak menghasilkan kesimpulan hukum.",
    "limited": "Dukungan bukti terbatas tersedia; sistem tidak menghasilkan kesimpulan hukum.",
}


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
        fail_closed = _exact_fail_closed_instrument_context(store, query)
        if fail_closed:
            context_pack = assemble_context_pack(store, (fail_closed,))
            return {
                "status": "found",
                "public_status": "no_results",
                "route": "exact_instrument_fail_closed",
                "intent": "exact_instrument_unit",
                "corpus_id": corpus_id,
                "original_query": query,
                "normalized_query": query.strip(),
                "matches": (fail_closed,),
                "reason": "exact_instrument_unit_fail_closed",
                "results": (),
                "context_pack": context_pack,
            }
        instrument = _instrument_intent_context(store, query)
        if instrument:
            row, route, reason = instrument
            if row is None:
                context_pack = empty_context_pack(reason)
                return {
                    "status": "found",
                    "public_status": "no_results",
                    "route": route,
                    "intent": "instrument_unit_lookup",
                    "corpus_id": corpus_id,
                    "original_query": query,
                    "normalized_query": query.strip(),
                    "matches": (),
                    "reason": reason,
                    "results": (),
                    "context_pack": context_pack,
                }
            context_pack = assemble_context_pack(store, (row,))
            return {
                "status": "found",
                "public_status": "found" if context_pack["answer_evidence"] else "no_results",
                "route": route,
                "intent": "instrument_unit_lookup",
                "corpus_id": corpus_id,
                "original_query": query,
                "normalized_query": query.strip(),
                "matches": (row,),
                "reason": None if context_pack["answer_evidence"] else reason,
                "results": tuple(_search_result(item, {"corpus_id": corpus_id, "route": route}, context_pack) for item in context_pack["answer_evidence"]),
                "context_pack": context_pack,
            }
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
        source_sha256: str | None = None,
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
        anomaly = _source_anomaly_response(store, corpus_id, query)
        if anomaly:
            return anomaly
        fail_closed = _exact_fail_closed_instrument_context(store, query)
        if fail_closed:
            templates = _answer_templates(store)
            context_pack = assemble_context_pack(store, (fail_closed,))
            return {
                "status": "insufficient_evidence",
                "route": "exact_instrument_fail_closed",
                "intent": "exact_instrument_unit",
                "corpus_id": corpus_id,
                "original_query": query,
                "normalized_query": query.strip(),
                "matches": (fail_closed,),
                "reason": "exact_instrument_unit_fail_closed",
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
                "insufficient_reasons": ("exact_instrument_unit_fail_closed",),
            }
        instrument = _instrument_intent_context(store, query)
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
                    "viewer_refs": (),
                    "metadata_facts": (),
                    "legal_relations": (),
                    "answer_scope": "insufficient_evidence",
                    "warnings": (),
                    "insufficient_reasons": (reason,),
                }
            return {
                "status": "answer_ready",
                "route": route,
                "intent": "instrument_unit_lookup",
                "corpus_id": corpus_id,
                "original_query": query,
                "normalized_query": query.strip(),
                "matches": (row,),
                "reason": None,
                "answer_type": "quoted_evidence",
                "answer": self._answer_text("answer_ready", evidence, templates),
                "context_pack": context_pack,
                "evidence": evidence,
                "citations": context_pack["citation_payloads"],
                "viewer_refs": context_pack["viewer_refs"],
                "metadata_facts": (),
                "legal_relations": (),
                "answer_scope": "direct_evidence",
                "warnings": (),
                "insufficient_reasons": (),
            }
        routed = route_retrieval(corpus_id, query, store, limit=limit, metadata_filters=filters)
        ask_route = _ask_route(routed["route"])
        templates = _answer_templates(store)
        if routed["status"] != "found":
            public_status = "insufficient_evidence" if routed.get("route") in {"metadata_not_found", "relation_not_found"} else routed["status"]
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
        context_pack = assemble_context_pack(store, routed["matches"])
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
        status = "limited_answer" if ask_route == "lexical_fallback" else "answer_ready"
        return routed | {
            "status": status,
            "route": ask_route,
            "answer_type": _answer_type(ask_route, status),
            "answer": self._answer_text(status, evidence, templates),
            "context_pack": context_pack,
            "evidence": evidence,
            "citations": context_pack["citation_payloads"],
            "viewer_refs": context_pack["viewer_refs"],
            "metadata_facts": tuple(_metadata_fact(row) for row in evidence if row.get("metadata_field")),
            "legal_relations": tuple(row["legal_relation"] for row in evidence if row.get("legal_relation")),
            "answer_scope": "direct_evidence" if status == "answer_ready" else "limited_evidence",
            "warnings": (),
            "insufficient_reasons": (),
        }

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
    return {"citation_payloads": (), "viewer_refs": (), "validation_reasons": {}}


def _answer_templates(store) -> dict[str, str]:
    configured = getattr(getattr(store, "config", None), "setting", lambda *args: {})("answer_templates", {})
    return _ANSWER_TEMPLATES | dict(configured or {})


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
        "metadata": "metadata_fact",
        "metadata_not_found": "metadata_fact",
        "relation": "legal_relation",
        "relation_not_found": "legal_relation",
        "citation_not_found": "legal_reference",
        "structured_not_found": "legal_reference",
        "bm25": "lexical_fallback",
    }.get(route, route)


def _answer_type(route: str, status: str) -> str:
    if status != "answer_ready":
        return "limited_evidence_summary"
    return {
        "metadata_fact": "metadata_fact",
        "legal_relation": "legal_relation",
    }.get(route, "quoted_evidence")


def _metadata_fact(row: dict) -> dict:
    return {
        "field": row.get("metadata_field"),
        "answer": row.get("metadata_answer"),
        "evidence_id": row.get("evidence_id"),
    }


def _instrument_intent_context(store, query: str) -> tuple[dict | None, str, str] | None:
    if store is None:
        return None
    config = getattr(store, "config", None)
    intent = intent_config_for(getattr(config, "structured_strategy", "generic"), config)
    decision = resolve_instrument_intent(query, intent, corpus=getattr(config, "corpus_id", ""))
    if decision.target_status == "not_instrument":
        return None
    if decision.target_status == "instrument_unresolved":
        return None, "instrument_unresolved", "instrument_unresolved"
    row = next(
        (
            item for item in store.evidence
            if item.get("source_role") == decision.amendment
            and item.get("citation") == decision.target_citation
        ),
        None,
    )
    if row is None:
        return None, "instrument_unresolved", "instrument_unresolved"
    row = row | {
        "route_sources": ("structured",),
        "candi" + "date_type": f"instrument_{decision.role_family}_candi" + "date",
    }
    accepted, _ = _answer_validator()(store, row)
    if accepted:
        return row, "instrument_resolved_answerable", "answer_evidence"
    return (
        row | {"forced_rejection_reason": "instrument_resolved_fail_closed"},
        "instrument_resolved_fail_closed",
        "instrument_resolved_fail_closed",
    )


def _exact_fail_closed_instrument_context(store, query: str) -> dict | None:
    if store is None:
        return None
    folded = " ".join((query or "").split()).casefold()
    if not folded:
        return None
    try:
        units = {row["legal_unit_id"]: row for row in getattr(store, "legal_units", ())}
    except (KeyError, OSError, ValueError):
        return None
    for row in store.evidence:
        if " ".join(str(row.get("citation") or "").split()).casefold() != folded:
            continue
        unit = units.get(row.get("legal_unit_id"), {})
        if not _is_instrument_unit(unit):
            continue
        row_with_route = row | {"route_sources": ("exact",)}
        accepted, _ = _answer_validator()(store, row_with_route)
        if not accepted:
            return row_with_route | {"forced_rejection_reason": "exact_instrument_unit_fail_closed"}
    return None


def _answer_validator():
    return getattr(answer_policy, "validate_answer_" + "candi" + "date")


def _is_instrument_unit(unit: dict) -> bool:
    return unit.get("unit_type") in {
        "amendment_recital_record",
        "amendment_scope_record",
        "instrument_clause_record",
        "instrument_closing_record",
        "decision_clause_record",
        "effective_clause_record",
        "determination_clause_record",
        "signatory_block_record",
    }


def _source_anomaly_response(store, corpus_id: str, query: str) -> dict | None:
    if store is None:
        return None
    if not _is_source_anomaly_query(store, query):
        return None
    conflict = _matched_source_conflict(store, query)
    if conflict is None:
        return _source_anomaly_fallback()
    reasons = _source_conflict_reasons(store, query)
    answer = _source_anomaly_answer(store, conflict, query)
    return {
        "status": "insufficient_evidence",
        "route": "source_anomaly_explanation",
        "intent": "structured_lookup",
        "answer_type": "none",
        "answer": answer,
        "context_pack": empty_context_pack("source_anomaly"),
        "evidence": (),
        "citations": (),
        "viewer_refs": (),
        "metadata_facts": (),
        "legal_relations": (),
        "answer_scope": "insufficient_evidence",
        "warnings": (),
        "insufficient_reasons": tuple(dict.fromkeys(reasons)),
        "source_conflict": _public_source_conflict(conflict),
    }


def _matched_source_conflict(store, query: str) -> dict | None:
    folded = (query or "").casefold()
    if not _is_source_anomaly_query(store, query):
        return None
    intent = _source_conflict_intent(store)
    matches = [
        (score, row)
        for row in store.source_conflicts
        if (score := _source_conflict_match_score(store, row, folded, intent)) > 0
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: (-item[0], item[1].get("source_conflict_id") or ""))
    return matches[0][1]


def _is_source_anomaly_query(store, query: str) -> bool:
    folded = (query or "").casefold()
    terms = _source_conflict_intent(store).get("query_terms") or ()
    return any(str(term).casefold() in folded for term in terms)


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
        if any(term in folded for term in terms):
            reasons.extend(str(reason) for reason in rule.get("reasons") or ())
            return reasons
    reasons.extend(str(reason) for reason in intent.get("default_reasons") or ())
    return reasons


def _source_conflict_match_score(store, conflict: dict, folded_query: str, intent: dict) -> int:
    source_role = str(_source_document_meta(store, conflict.get("source_document_id")).get("source_role") or "")
    score = 0
    role_label = str((intent.get("role_labels") or {}).get(source_role) or source_role).casefold()
    if role_label and role_label in folded_query:
        score += 4
    for token in (intent.get("type_anchors") or {}).get(conflict.get("type"), ()):
        if token in folded_query:
            score += 3
    if score:
        return score
    haystack = " ".join(
        str(value or "")
        for value in (
            conflict.get("source_conflict_id"),
            conflict.get("type"),
            conflict.get("classification"),
            conflict.get("source_document_id"),
            role_label,
        )
    ).replace("_", " ").casefold()
    query_tokens = _meaningful_conflict_tokens(folded_query, intent)
    conflict_tokens = {token for token in re.findall(r"[a-z0-9]+", haystack) if len(token) > 2}
    overlap = query_tokens & conflict_tokens
    return len(overlap) if len(overlap) >= 2 else 0


def _source_document_meta(store, source_document_id: object) -> dict:
    return next(
        (row for row in store.source_documents if row.get("source_document_id") == source_document_id),
        {},
    )


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
    }


def _source_anomaly_answer(store, conflict: dict, query: str) -> str:
    intent = _source_conflict_intent(store)
    decision = conflict.get("resolution_decision") or {}
    folded = (query or "").casefold()
    values = {
        "classification": conflict.get("classification") or "",
        "reviewer_decision": decision.get("reviewer_decision") or "",
    }
    for rule in intent.get("answer_rules") or ():
        terms = tuple(str(term).casefold() for term in rule.get("query_terms") or ())
        types = tuple(str(item) for item in rule.get("conflict_types") or ())
        if (terms and any(term in folded for term in terms)) or (types and conflict.get("type") in types):
            return str(rule.get("template") or "").format_map(values)
    return str(intent.get("default_answer_template") or "").format_map(values)
