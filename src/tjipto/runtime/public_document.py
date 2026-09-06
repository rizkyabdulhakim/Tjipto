from __future__ import annotations

from collections.abc import Callable, Iterable
import re

from tjipto.catalog import CatalogDocument
from tjipto.corpora.intent_config import contains_intent_phrase, normalize_intent_text
from tjipto.retrieval.answer import assemble_context_pack, empty_context_pack
from tjipto.retrieval.requirements import research_requirements_for_ask
from tjipto.retrieval.document_research import document_summary_rows, version_comparison_rows
from tjipto.retrieval.sufficiency import assess_sufficiency, collect_evidence_set
from tjipto.runtime.answer_arbitration import (
    _answer_templates,
    _document_summary_answer,
    _document_title,
    _version_comparison_answer,
)
from tjipto.runtime.response import (
    AnswerDecision,
    _article_relation_evidence,
    _public_document_relation,
    compose_research_answer,
    project_response,
)
from tjipto.runtime.viewer import _authority_policy, _citation_with_authority, _copy_text
from tjipto.contracts.legal_information import FieldState, LifecycleKind, RelationKind, VerifiedValue


_RELATION_LABELS = {
    RelationKind.AMENDS: "Mengubah",
    RelationKind.AMENDED_BY: "Diubah oleh",
    RelationKind.CONSOLIDATES: "Menggabungkan",
    RelationKind.CONSOLIDATED_BY: "Digabungkan dalam",
    RelationKind.DERIVED_FROM: "Berasal dari",
    RelationKind.DERIVES: "Menjadi dasar bagi",
    RelationKind.REVOKES: "Mencabut",
    RelationKind.REVOKED_BY: "Dicabut oleh",
}

_EFFECT_LABELS = {
    "MODIFIES": "Ketentuan yang diubah",
    "RENAMES": "Penomoran menjadi",
    "RENUMBERED_TO": "Penomoran menjadi",
    "DELETES": "Ketentuan yang dihapus",
    "ADDS": "Ketentuan yang ditambahkan",
    "SUPPLEMENTS": "Ketentuan yang dilengkapi",
    "AMBIGUOUS_OPERATION": "Ketentuan yang diubah dan/atau ditambahkan",
}


def enrich_document_summary(store, response: dict) -> dict:
    roles = tuple(response.get("source_scopes") or ()) or (getattr(store.config, "preferred_source_role", None),)
    rows = document_summary_rows(store, tuple(role for role in roles if role))
    if not rows:
        return response
    support = tuple(_citation_with_authority(store, row) for row in rows)
    answer = _document_summary_answer(store, rows)
    if not answer.strip():
        return response
    return response | {
        "answer": answer,
        "summary_support": support,
        "structural_support": tuple(response.get("structural_support") or support),
    }


def enrich_version_comparison(store, response: dict) -> dict:
    roles = tuple(response.get("source_scopes") or ())
    references = tuple(response.get("legal_references") or ())
    if len(roles) < 2 and len(references) < 2:
        return response
    rows = version_comparison_rows(store, roles, references=references)
    if not rows:
        return response
    support = tuple(_citation_with_authority(store, row) for row in rows)
    answer = _version_comparison_answer(store, roles, rows, references=references)
    return response | {"answer": answer, "comparison_support": support} if answer.strip() else response


def historical_pre_change_response(
    corpus_id: str,
    query: str,
    store,
    semantics,
    *,
    route_retrieval: Callable[..., dict],
) -> dict:
    """Resolve historical text and its change provenance as one evidence set."""
    requirements = research_requirements_for_ask(store, semantics, query)
    relation_route = route_retrieval(
        corpus_id,
        query,
        store,
        limit=10,
        allow_navigation=False,
        allow_relation=True,
    )
    projections = tuple(
        edge.get("relation_projection") or {}
        for edge in relation_route.get("matches", ())
        if edge.get("relation_projection")
    )
    projection = next(iter(projections), {})
    normative_role = next(
        (requirement.source_role for requirement in requirements if requirement.source_role),
        str(projection.get("source_role") or "") or None,
    )
    provenance_role = next(
        (
            requirement.source_role
            for requirement in requirements
            if requirement.requirement_id == "deletion_provenance" and requirement.source_role
        ),
        str(projection.get("source_role") or "") or None,
    )
    target = next(
        (requirement.legal_target for requirement in requirements if requirement.requirement_id == "historical_normative_text"),
        str(projection.get("target_citation") or "") or None,
    )
    source_label = str(projection.get("source_label") or "") or next(
        (
            requirement.retrieval_query
            for requirement in requirements
            if requirement.requirement_id == "deletion_provenance"
        ),
        query,
    )
    normative_route = route_retrieval(
        corpus_id,
        target or query,
        store,
        limit=10,
        metadata_filters={"source_role": normative_role} if normative_role else None,
        allow_navigation=False,
        allow_relation=False,
    )
    provenance_route = route_retrieval(
        corpus_id,
        source_label or query,
        store,
        limit=10,
        metadata_filters={"source_role": provenance_role} if provenance_role else None,
        allow_navigation=False,
        allow_relation=False,
    )
    marked: list[dict] = []
    normative_requirement = next(
        (requirement for requirement in requirements if requirement.requirement_id == "historical_normative_text"),
        None,
    )
    if normative_requirement is not None:
        marked.extend(
            dict(row) | {"_requirement_ids": (normative_requirement.requirement_id,)}
            for row in normative_route.get("matches", ())
        )
    relation_evidence = _article_relation_evidence(store, projection)
    if relation_evidence is not None and relation_evidence.get("citation_final") is True:
        marked.append(relation_evidence | {"_requirement_ids": ("deletion_provenance",), "route_sources": ("relation",)})
    else:
        provenance_requirement = next(
            (requirement for requirement in requirements if requirement.requirement_id == "deletion_provenance"),
            None,
        )
        if provenance_requirement is not None:
            marked.extend(
                dict(row) | {"_requirement_ids": (provenance_requirement.requirement_id,)}
                for row in provenance_route.get("matches", ())
            )
    matches = tuple({str(row["evidence_id"]): row for row in marked if row.get("evidence_id")}.values())
    evidence_set = collect_evidence_set(store, matches, requirements)
    assessment = assess_sufficiency(evidence_set, requirements)
    public_matches = tuple({key: value for key, value in row.items() if not str(key).startswith("_")} for row in matches)
    routed = {
        "status": "found" if public_matches else "no_results",
        "route": "historical_pre_change",
        "intent": "historical_text",
        "corpus_id": corpus_id,
        "original_query": query,
        "normalized_query": query.strip(),
        "operation": semantics.operation,
        "source_scopes": semantics.source_scopes,
        "temporal_scope": semantics.temporal_scope,
        "matches": public_matches,
        "reason": None if public_matches else "historical_normative_text_and_deletion_provenance_required",
        "evidence_set": {
            "support_ids": tuple(str(row.get("evidence_id")) for row in evidence_set.supports),
            "assignments": evidence_set.assignments,
            "missing_requirement_ids": evidence_set.missing_requirement_ids,
            "missing_reasons": evidence_set.missing_reasons,
        },
        "sufficiency": {
            "status": assessment.status,
            "fulfilled_requirement_ids": assessment.fulfilled_requirement_ids,
            "missing_requirement_ids": assessment.missing_requirement_ids,
            "missing_reasons": assessment.missing_reasons,
            "retry_allowed": assessment.retry_allowed,
        },
    }
    templates = _answer_templates(store)
    if assessment.status != "complete":
        reason = "historical_normative_text_and_deletion_provenance_required"
        return project_response(
            routed | {"status": "no_results", "reason": reason},
            AnswerDecision(
                "insufficient_evidence",
                "historical_pre_change",
                "none",
                templates["insufficient"],
                empty_context_pack(reason),
                insufficient_reasons=(reason,),
                reason_code=reason,
            ),
        )
    evidence = tuple({key: value for key, value in row.items() if not str(key).startswith("_")} for row in evidence_set.supports)
    context_pack = assemble_context_pack(store, evidence)
    answer = compose_research_answer(
        evidence,
        evidence_set,
        requirements,
        assessment,
        preferred_source_role=getattr(store.config, "preferred_source_role", None),
    )
    citations = tuple(_citation_with_authority(store, row) for row in context_pack["citation_payloads"])
    return project_response(
        routed,
        AnswerDecision(
            "answer_ready",
            "historical_pre_change",
            "quoted_evidence",
            answer,
            context_pack,
            evidence=evidence,
            citations=citations,
            final_citations=tuple(row for row in citations if row.get("citation_final") is True),
            historical_citations=tuple(row for row in citations if row.get("citation_final") is False),
            viewer_refs=tuple(
                row["viewer_ref"]
                for row in citations
                if row.get("citation_final") is True and row.get("viewer_ref")
            ),
            answer_scope="historical_evidence",
        ),
    )


def project_legal_document(
    document: CatalogDocument,
    documents: Iterable[CatalogDocument],
    *,
    viewer_target: dict | None = None,
) -> dict:
    """Build the one allowlisted legal-document representation used publicly."""
    by_id = {item.stable_id: item for item in documents}
    lifecycle = {event.kind: _public_value(event.value) for event in document.lifecycle}
    projection = {
        "title": _display_value(document.identity.official_title),
        "legal_identity": _identity_display(document),
        "legal_status": _status_display(document.legal_status.status),
        "legal_status_scope": _status_scope_display(document.legal_status.scope),
        "document_role": document.document_role_label,
        "issuer": _display_value(document.identity.issuer),
        "signatories": _public_value(document.signatories) if document.signatories else None,
        "establishment_date": lifecycle.get(LifecycleKind.ESTABLISHMENT),
        "establishment_place": _public_value(document.establishment_place) if document.establishment_place else None,
        "promulgation_date": lifecycle.get(LifecycleKind.PROMULGATION),
        "effective_date": lifecycle.get(LifecycleKind.EFFECTIVENESS),
        "publication": _public_value(document.publication),
        "official_url": document.official_url,
        "relations": tuple(_public_relation(document, relation, by_id) for relation in document.relations),
        "provision_effects": tuple(_public_effect(effect, by_id) for effect in document.provision_effects),
        "source_annotations": tuple(
            {
                "label": f"Catatan sumber {annotation.marker}",
                "text": f"{annotation.marker} berarti {annotation.meaning}.",
                "source_reference": document.official_url,
                "page_number": annotation.page_number,
            }
            for annotation in document.source_annotations
        ),
    }
    conflict = _public_conflict(document.identity.official_title)
    if conflict is not None:
        projection["official_title_conflict"] = conflict
    if viewer_target is not None:
        projection["viewer_target"] = viewer_target
    return projection


def consolidated_definition_response(corpus_id: str, query: str, store, semantics) -> dict | None:
    """Explain a consolidated document from persisted document relations."""
    normalized = normalize_intent_text(query)
    if (
        semantics.operation != "search"
        or semantics.source_role != getattr(store.config, "preferred_source_role", None)
        or semantics.legal_references
        or not re.search(r"\bapa\s+itu\b|\byang\s+dimaksud\b|\bjelaskan\b", normalized)
        or not contains_intent_phrase(query, tuple(store.config.setting("document_catalog", {}).get("document_terms", ())))
    ):
        return None
    current_role = str(semantics.source_role)
    relations = tuple(
        edge.get("relation_projection") or {}
        for edge in store.graph_edges
        if (edge.get("relation_projection") or {}).get("source_role") == current_role
        and (edge.get("relation_projection") or {}).get("target_source_role")
        and (edge.get("relation_projection") or {}).get("article_level") is not True
    )
    target_roles = tuple(dict.fromkeys(str(row.get("target_source_role")) for row in relations))
    if not relations or not target_roles:
        return None
    catalog = store.config.setting("document_catalog", {}) or {}
    titles = catalog.get("titles", {}) if isinstance(catalog, dict) else {}
    current_title = str(titles.get(current_role) or current_role)
    original_role = next((role for role in target_roles if role.startswith("original_")), None)
    amendment_roles = tuple(role for role in target_roles if role.startswith("amendment_"))
    source_title = str(titles.get(original_role) or original_role) if original_role else None
    amendment_titles = tuple(str(titles.get(role) or role) for role in amendment_roles)
    clauses = [f"{current_title} merupakan naskah konsolidasi"]
    if source_title:
        clauses.append(f"yang diturunkan dari {source_title}")
    if amendment_titles:
        clauses.append(f"dan mencakup hubungan konsolidasi dengan {', '.join(amendment_titles)}")
    answer = " ".join(clauses) + "."
    public_relations = tuple(_public_document_relation(row) for row in relations)
    return project_response(
        {
            "status": "found",
            "route": "document_relation",
            "intent": "document_relation_lookup",
            "corpus_id": corpus_id,
            "original_query": query,
            "normalized_query": normalized,
            "matches": relations,
            "reason": None,
        },
        AnswerDecision(
            "limited_answer",
            "document_relation",
            "document_relation",
            answer,
            empty_context_pack("consolidated_document_provenance"),
            evidence=public_relations,
            document_relations=public_relations,
            trace_support=public_relations,
            answer_scope="consolidated_document_provenance",
            warnings=("document_relation_trace_only",),
        ),
    )


def _identity_display(document: CatalogDocument) -> str | None:
    identity = document.identity
    title = _display_value(identity.official_title)
    number = _display_value(identity.number)
    year = _display_value(identity.year)
    document_type = _display_value(identity.document_type)
    if number and document_type:
        return " ".join(part for part in (document_type, f"Nomor {number}", f"Tahun {year}" if year else None) if part)
    return title


def _status_display(value: VerifiedValue) -> str:
    if value.state is FieldState.VERIFIED and value.display_value:
        return value.display_value
    if value.state is FieldState.CONFLICTING_SOURCES:
        return "Konflik Sumber"
    return "Belum Diverifikasi"


def _status_scope_display(scope: str) -> str:
    return {
        "document_record": "Record peraturan",
        "parent_record": "Record induk sumber",
    }.get(scope, "Cakupan sumber tidak tersedia")


def _public_value(value: VerifiedValue) -> str | None:
    return value.display_value if value.state in {FieldState.VERIFIED, FieldState.CONFLICTING_SOURCES} else None


def _display_value(value: VerifiedValue) -> str | None:
    return value.display_value if value.display_value else None


def _public_relation(document: CatalogDocument, relation, by_id: dict[str, CatalogDocument]) -> dict:
    source = by_id.get(relation.source_document_id)
    target = by_id.get(relation.target_document_id)
    return {
        "label": _RELATION_LABELS[relation.relation],
        "relation_type": _RELATION_LABELS[relation.relation],
        "source": _identity_display(source) if source else "Dokumen sumber belum tersedia",
        "target": _identity_display(target) if target else "Dokumen target belum tersedia",
        "direction": "Naskah sumber ke naskah terkait",
        "verification_state": "Terverifikasi",
        "source_reference": relation.provenance.reference,
    }


def _public_effect(effect, by_id: dict[str, CatalogDocument]) -> dict:
    target = by_id.get(effect.target_document_id)
    return {
        "label": _EFFECT_LABELS[effect.operation],
        "target": effect.exact_target,
        "document": _identity_display(target) if target else None,
        "verification_state": "Ruang lingkup naskah" if effect.operation == "AMBIGUOUS_OPERATION" else "Terverifikasi",
        "source_reference": effect.provenance.reference,
        "page_number": effect.provenance.page_number,
    }


def _public_conflict(value: VerifiedValue) -> dict | None:
    if value.state is not FieldState.CONFLICTING_SOURCES or value.resolution is None:
        return None
    return {
        "state": "Terselesaikan" if value.resolution.selected_value is not None else "Belum Terselesaikan",
        "kind": "Perbedaan Nilai Sumber Resmi",
        "values": tuple(
            {
                "value": candidate.display_value,
                "source_authority": candidate.provenance.source_authority,
                "source_reference": candidate.provenance.reference,
                "verified_at": candidate.provenance.verified_at.isoformat(),
            }
            for candidate in value.conflicting_values
        ),
        "reviewer_decision": value.resolution.reviewer_decision,
        "legal_basis": value.resolution.legal_basis,
    }


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


def _metadata_fact(row: dict) -> dict:
    return {
        "field": row.get("metadata_field"),
        "answer": row.get("metadata_answer"),
        "evidence_id": row.get("evidence_id"),
    }


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


def _document_matches_filters(source: dict, filters: dict) -> bool:
    return all(
        source.get(key) == value for key, value in filters.items() if key in {"source_role", "temporal_context"} and value is not None
    )


def _metadata_support(store, row: dict) -> dict:
    can_resolve = row.get("viewer_ref", {}).get("can_resolve") is True
    authority = _authority_policy(store, row, can_resolve=can_resolve)
    viewer_ref = ((row.get("viewer_ref") or {}) | authority) if can_resolve else None
    field = str(row.get("metadata_field") or "")
    labels = {
        "signatories": "Penandatangan",
        "penetapan": "Tanggal Penetapan",
        "institution": "Lembaga",
        "place": "Tempat Penetapan",
    }
    source: dict = next(
        (item for item in store.source_documents if item.get("source_document_id") == row.get("source_document_id")),
        {},
    )
    document_title = _document_title(store, source)
    name = str(row.get("printed_name") or "").strip()
    role = str(row.get("printed_role") or "").strip()
    institution = str(row.get("institution") or "").strip()
    date_context = str(row.get("date_context") or "").strip()
    display_text = str(row.get("display_text") or row.get("answer") or "").strip()
    if name and role:
        display_text = f"{name} tercantum sebagai {role} dalam {document_title}."
    return authority | {
        "support_class": "exact_metadata_citation" if can_resolve else "metadata_trace",
        "field": row.get("metadata_field"),
        "answer": row.get("metadata_answer"),
        "fact_kind": row.get("fact_kind") or ("source_fact" if field else "metadata"),
        "display_label": labels.get(field, "Sumber Dokumen"),
        "display_text": display_text,
        "copy_text": _copy_text(display_text),
        "printed_name": name or None,
        "entity_identity": row.get("entity_identity"),
        "printed_name_alias": row.get("printed_name_alias"),
        "printed_role": role or None,
        "institution": institution or None,
        "date_context": date_context or None,
        "evidence_id": row.get("evidence_id"),
        "source_document_id": row.get("source_document_id"),
        "source_role": row.get("source_role"),
        "page_numbers": tuple(row.get("page_numbers") or ()),
        "page_query": row.get("page_query") is True,
        "citation_available": can_resolve,
        "viewer_highlightable": can_resolve,
        "viewer_ref": viewer_ref,
    }
