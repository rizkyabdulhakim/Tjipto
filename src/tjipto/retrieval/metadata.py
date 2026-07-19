from __future__ import annotations

import re
from dataclasses import dataclass

from tjipto.corpora.intent_config import contains_intent_phrase, intent_config_for, normalize_intent_text
from tjipto.corpora.parser_dispatch import DEFAULT_CORPUS_ID, normalize_metadata_intent, parse_legal_reference


@dataclass(frozen=True)
class SourceScopeDecision:
    role: str | None
    state: str

    @property
    def explicit(self) -> bool:
        return self.state == "explicit_resolved"

    @property
    def unresolved(self) -> bool:
        return self.state == "unresolved"


def normalize_filters(filters: dict | None = None, *, config=None, **kwargs) -> dict:
    if filters is not None and not isinstance(filters, dict):
        return {
            "status": "final",
            "_error": "invalid_filter",
            "_invalid_filters": ("metadata_filters",),
        }
    raw = dict(filters or {})
    raw.update({key: value for key, value in kwargs.items() if value is not None})
    normalized: dict = {}
    invalid = []
    for key in raw:
        if key not in {"source_role", "temporal_context"}:
            invalid.append(key)
    source_roles = set(getattr(config, "source_roles", ()) or ())
    temporal_contexts = set(getattr(config, "temporal_contexts", ()) or ())
    source_role = raw.get("source_role")
    temporal_context = raw.get("temporal_context")

    if source_role is not None:
        if source_role not in source_roles:
            invalid.append("source_role")
        else:
            normalized["source_role"] = source_role
    if temporal_context is not None:
        if temporal_context not in temporal_contexts:
            invalid.append("temporal_context")
        else:
            normalized["temporal_context"] = temporal_context
    normalized["status"] = "final"
    if invalid:
        normalized["_error"] = "invalid_filter"
        normalized["_invalid_filters"] = tuple(invalid)
    if "source_role" in normalized and "temporal_context" in normalized and normalized["source_role"] != normalized["temporal_context"]:
        normalized["_error"] = "conflicting_filters"
    return normalized


def filter_evidence(rows: tuple[dict, ...], filters: dict) -> tuple[dict, ...]:
    if filters.get("_error"):
        return ()
    return tuple(
        row
        for row in rows
        if row.get("status") == filters.get("status", "final")
        and ("source_role" not in filters or row.get("source_role") == filters["source_role"])
        and ("temporal_context" not in filters or row.get("temporal_context") == filters["temporal_context"])
    )


def source_role_for_query(query: str, *, strategy: str = "generic", config=None) -> str | None:
    decision = resolve_source_scope(query, strategy=strategy, config=config)
    return decision.role if decision.explicit else None


def source_roles_for_query(query: str, *, strategy: str = "generic", config=None) -> tuple[str, ...]:
    """Return every artifact-declared temporal role mentioned in the query."""
    return tuple(
        role
        for role, pattern in intent_config_for(strategy, config)["metadata_roles"]
        if pattern.search(query or "")
    )


def resolve_source_scope(query: str, *, strategy: str = "generic", config=None) -> SourceScopeDecision:
    explicit_role = next(iter(source_roles_for_query(query, strategy=strategy, config=config)), None)
    if explicit_role is not None:
        return SourceScopeDecision(explicit_role, "explicit_resolved")
    intent = intent_config_for(strategy, config)
    if any(pattern.search(query or "") for pattern in intent["unresolved_source_scope_patterns"]):
        return SourceScopeDecision(None, "unresolved")
    if contains_intent_phrase(query, intent.get("instrument_source_signals", ())):
        return SourceScopeDecision(None, "unresolved")
    return SourceScopeDecision(getattr(config, "preferred_source_role", None), "unscoped")


def public_filters(filters: dict) -> dict:
    return {key: value for key, value in filters.items() if not key.startswith("_")}


def metadata_lookup(store, query: str, limit: int = 10) -> tuple[dict, ...]:
    config = getattr(store, "config", None)
    strategy = getattr(config, "query_strategy", "generic")
    if _has_legal_reference(query, config):
        return ()
    intent = intent_config_for(strategy, config)
    field = _metadata_field(query, strategy=strategy, config=config)
    if field is None and not _matching_signatories(store, query):
        return ()
    scope = resolve_source_scope(query, strategy=strategy, config=config)
    role = scope.role if scope.explicit else None
    requires_penetapan = _asks_enactment_context((query or "").casefold(), intent)
    rows = []
    grounding_by_id = {row["metadata_grounding_id"]: row for row in store.metadata_grounding}
    for row in store.document_metadata:
        if role is not None and row.get("source_role") != role:
            continue
        if requires_penetapan and row.get("field_statuses", {}).get("penetapan") != "grounded":
            continue
        selected_signatories = _matching_signatories(store, query, row)
        selected_field = "signatories" if selected_signatories else field
        if row.get("field_statuses", {}).get(selected_field) != "grounded":
            continue
        refs = tuple(row.get("grounded_fields", {}).get(selected_field) or ())
        if not refs:
            continue
        signatory = selected_signatories[0] if selected_signatories else None
        grounding = _signatory_grounding(grounding_by_id, refs, signatory) if signatory else grounding_by_id.get(refs[0])
        if grounding is None:
            continue
        result = _metadata_result(store, row, grounding, selected_field, value=signatory.get("name_text") if signatory else None)
        if result:
            rows.append(result)
    return tuple(rows[:limit])


def has_metadata_target(query: str, *, strategy: str = "generic", config=None, store=None) -> bool:
    if _has_legal_reference(query, config):
        return False
    return _metadata_field(query, strategy=strategy, config=config) is not None or bool(
        store is not None and _matching_signatories(store, query)
    )


def _has_legal_reference(query: str, config) -> bool:
    try:
        return any(parse_legal_reference(getattr(config, "corpus_id", DEFAULT_CORPUS_ID), query).values())
    except ValueError:
        return False


def _metadata_field(query: str, *, strategy: str, config=None) -> str | None:
    corpus_id = getattr(config, "corpus_id", DEFAULT_CORPUS_ID)
    folded = normalize_metadata_intent(corpus_id, query)
    intent = intent_config_for(strategy, config)
    intent = intent | {
        "document_target_words": tuple(normalize_metadata_intent(corpus_id, value) for value in intent["document_target_words"]),
        "metadata_fields": {
            field: tuple(normalize_metadata_intent(corpus_id, value) for value in values)
            for field, values in intent["metadata_fields"].items()
        },
        "metadata_rules": {
            rule: tuple(normalize_metadata_intent(corpus_id, value) for value in values)
            for rule, values in intent["metadata_rules"].items()
        },
    }
    patterns = intent["metadata_fields"]
    if not patterns:
        return None
    if not any(word in folded for word in intent["document_target_words"]):
        return None
    if _asks_promulgation(folded, intent):
        return "promulgation"
    if _asks_revocation(folded, intent):
        return "revocation"
    if _asks_signatories(folded, intent):
        return "signatories"
    if _asks_effective_rule(folded, intent):
        return "effective_rule"
    if _asks_decision_date(folded, intent):
        return "decision_date"
    if _asks_decision_session(folded, intent):
        return "decision_session"
    if _asks_enactment_place(folded, intent):
        return "place"
    if _asks_institution(folded, intent):
        return "institution"
    if _asks_enactment_date(folded, intent):
        return "penetapan"
    for field, field_patterns in patterns.items():
        if any(pattern in folded for pattern in field_patterns):
            return field
    return None


def _matching_signatories(store, query: str, row: dict | None = None) -> tuple[dict, ...]:
    """Find source-declared signatory names without treating BM25 as proof."""
    query_tokens = normalize_intent_text(query).split()
    if not query_tokens:
        return ()
    rows = (row,) if row is not None else store.document_metadata
    matches = []
    for metadata in rows:
        for signatory in metadata.get("signatories") or ():
            name_tokens = normalize_intent_text(signatory.get("name_text")).split()
            if _contains_name_tokens(query_tokens, name_tokens):
                matches.append(signatory)
    return tuple(matches)


def _contains_name_tokens(query_tokens: list[str], name_tokens: list[str]) -> bool:
    if not name_tokens:
        return False
    for start in range(len(name_tokens)):
        candidate = name_tokens[start:]
        if len(candidate) > len(query_tokens):
            continue
        if any(query_tokens[index : index + len(candidate)] == candidate for index in range(len(query_tokens) - len(candidate) + 1)):
            return True
    return False


def _signatory_grounding(grounding_by_id: dict[str, dict], refs: tuple[str, ...], signatory: dict) -> dict | None:
    expected = normalize_intent_text(signatory.get("name_text"))
    return next(
        (
            grounding
            for grounding_id in refs
            if (grounding := grounding_by_id.get(grounding_id)) is not None
            and normalize_intent_text(grounding.get("quoted_text")) == expected
        ),
        None,
    )


def _asks_any(folded: str, intent: dict, rule: str) -> bool:
    return any(pattern in folded for pattern in intent["metadata_rules"].get(rule, ()))


def _asks_token(folded: str, intent: dict, rule: str) -> bool:
    return any(re.search(rf"\b{re.escape(pattern)}\b", folded) for pattern in intent["metadata_rules"].get(rule, ()))


def _asks_promulgation(folded: str, intent: dict) -> bool:
    return _asks_any(folded, intent, "promulgation")


def _asks_revocation(folded: str, intent: dict) -> bool:
    return _asks_any(folded, intent, "revocation")


def _asks_signatories(folded: str, intent: dict) -> bool:
    return _asks_any(folded, intent, "signatories")


def _asks_enactment_place(folded: str, intent: dict) -> bool:
    return (_asks_any(folded, intent, "place_context") and _asks_any(folded, intent, "enactment_context")) or (
        _asks_any(folded, intent, "enactment_verbs") and _asks_any(folded, intent, "place_question")
    )


def _asks_effective_rule(folded: str, intent: dict) -> bool:
    return _asks_any(folded, intent, "effective_rule") and _asks_any(folded, intent, "date_question")


def _asks_decision_date(folded: str, intent: dict) -> bool:
    return _asks_any(folded, intent, "decision_context") and _asks_any(folded, intent, "date_question")


def _asks_decision_session(folded: str, intent: dict) -> bool:
    return _asks_any(folded, intent, "decision_context") and _asks_any(folded, intent, "session_question")


def _asks_institution(folded: str, intent: dict) -> bool:
    return (
        _asks_any(folded, intent, "institution")
        or _asks_any(folded, intent, "institution_question")
        or _asks_token(folded, intent, "institution_tokens")
        or (_asks_any(folded, intent, "who_question") and _asks_any(folded, intent, "enactment_context"))
    )


def _asks_enactment_date(folded: str, intent: dict) -> bool:
    return _asks_any(folded, intent, "enactment_date") or (
        _asks_any(folded, intent, "enactment_context") and _asks_any(folded, intent, "date_question")
    )


def _asks_enactment_context(folded: str, intent: dict) -> bool:
    return _asks_any(folded, intent, "enactment_context") or _asks_token(folded, intent, "institution_tokens")


def _metadata_result(store, row: dict, grounding: dict, field: str, *, value: str | None = None) -> dict | None:
    value = value or _field_value(row, field)
    if not value:
        return None
    bboxes = store.metadata_bboxes_for(grounding["metadata_grounding_id"])
    return {
        "corpus_id": row.get("corpus_id"),
        "evidence_id": grounding["metadata_grounding_id"],
        "legal_unit_id": None,
        "source_document_id": grounding.get("source_document_id"),
        "citation": _metadata_citation(row, field),
        "hierarchy": (_metadata_citation(row, field),),
        "quoted_text": grounding.get("quoted_text"),
        "metadata_answer": value,
        "metadata_field": field,
        "source_pdf_path": grounding.get("source_pdf_path"),
        "source_sha256": grounding.get("source_sha256"),
        "source_role": row.get("source_role"),
        "temporal_context": row.get("temporal_context"),
        "page_numbers": tuple(grounding.get("page_numbers") or ()),
        "bbox_refs": tuple(grounding.get("bbox_refs") or ()),
        "bbox_ids": tuple(grounding.get("bbox_ids") or ()),
        "bbox_precision": grounding.get("bbox_precision"),
        "bbox_count": len(bboxes),
        "text_span_ids": tuple(grounding.get("text_span_ids") or ()),
        "viewer_highlightable": grounding.get("viewer_highlightable"),
        "status": "final",
        "route_sources": ("metadata",),
        "route_score": 900.0,
        "route_scores": {"metadata": 900.0},
        "rank_reasons": ("metadata", "answer_evidence"),
        "metadata_grounding": True,
        "metadata_viewer_resolvable": (
            grounding.get("bbox_precision") == "exact"
            and grounding.get("viewer_highlightable") is True
            and _bboxes_have_coordinates(bboxes)
        ),
    }


def _field_value(row: dict, field: str) -> str | None:
    if field == "penetapan":
        value = row.get("penetapan")
        return value.get("date_text") if isinstance(value, dict) else None
    if field in {"promulgation", "revocation"}:
        return None
    if field in {"effective_rule", "decision_date", "decision_session", "source_anomaly_status"}:
        value = row.get(field)
        return str(value) if value else None
    value = row.get(field)
    if field == "signatories":
        signatories = row.get("signatories") or ()
        names = [item.get("name_text") for item in signatories if item.get("name_text")]
        return ", ".join(names) if names else None
    if isinstance(value, dict):
        return value.get("title_text") or value.get("date_text") or value.get("institution") or value.get("place")
    return str(value) if value else None


def _metadata_citation(row: dict, field: str) -> str:
    return f"Metadata {row.get('source_role')}: {field}"


def _bboxes_have_coordinates(bboxes: list[dict]) -> bool:
    return bool(bboxes) and all(box.get(key) is not None for box in bboxes for key in ("x0", "y0", "x1", "y1"))
