from __future__ import annotations

import re

from tjipto.corpora.intent_config import intent_config_for


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
    if (
        "source_role" in normalized
        and "temporal_context" in normalized
        and normalized["source_role"] != normalized["temporal_context"]
    ):
        normalized["_error"] = "conflicting_filters"
    return normalized


def filter_evidence(rows: tuple[dict, ...], filters: dict) -> tuple[dict, ...]:
    if filters.get("_error"):
        return ()
    return tuple(
        row for row in rows
        if row.get("status") == filters.get("status", "final")
        and (
            "source_role" not in filters
            or row.get("source_role") == filters["source_role"]
        )
        and (
            "temporal_context" not in filters
            or row.get("temporal_context") == filters["temporal_context"]
        )
    )


def public_filters(filters: dict) -> dict:
    return {key: value for key, value in filters.items() if not key.startswith("_")}


def metadata_lookup(store, query: str, limit: int = 10) -> tuple[dict, ...]:
    config = getattr(store, "config", None)
    strategy = getattr(config, "query_strategy", "generic")
    intent = intent_config_for(strategy, config)
    field = _metadata_field(query, strategy=strategy, config=config)
    if field is None:
        return ()
    role = _source_role(query, strategy=strategy, config=config)
    requires_penetapan = _asks_enactment_context((query or "").casefold(), intent)
    rows = []
    grounding_by_id = {row["metadata_grounding_id"]: row for row in store.metadata_grounding}
    for row in store.document_metadata:
        if role is not None and row.get("source_role") != role:
            continue
        if requires_penetapan and row.get("field_statuses", {}).get("penetapan") != "grounded":
            continue
        if row.get("field_statuses", {}).get(field) != "grounded":
            continue
        refs = tuple(row.get("grounded_fields", {}).get(field) or ())
        if not refs:
            continue
        grounding = grounding_by_id.get(refs[0])
        if grounding is None:
            continue
        result = _metadata_result(store, row, grounding, field)
        if result:
            rows.append(result)
    return tuple(rows[:limit])


def has_metadata_target(query: str, *, strategy: str = "generic", config=None) -> bool:
    return _metadata_field(query, strategy=strategy, config=config) is not None


def _metadata_field(query: str, *, strategy: str, config=None) -> str | None:
    folded = (query or "").casefold()
    intent = intent_config_for(strategy, config)
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
    return (
        _asks_any(folded, intent, "place_context")
        and _asks_any(folded, intent, "enactment_context")
    ) or (
        _asks_any(folded, intent, "enactment_verbs")
        and _asks_any(folded, intent, "place_question")
    )


def _asks_effective_rule(folded: str, intent: dict) -> bool:
    return (
        _asks_any(folded, intent, "effective_rule")
        and _asks_any(folded, intent, "date_question")
    )


def _asks_decision_date(folded: str, intent: dict) -> bool:
    return (
        _asks_any(folded, intent, "decision_context")
        and _asks_any(folded, intent, "date_question")
    )


def _asks_decision_session(folded: str, intent: dict) -> bool:
    return (
        _asks_any(folded, intent, "decision_context")
        and _asks_any(folded, intent, "session_question")
    )


def _asks_institution(folded: str, intent: dict) -> bool:
    return (
        _asks_any(folded, intent, "institution")
        or _asks_any(folded, intent, "institution_question")
        or _asks_token(folded, intent, "institution_tokens")
    )


def _asks_enactment_date(folded: str, intent: dict) -> bool:
    return (
        _asks_any(folded, intent, "enactment_date")
        or (
            _asks_any(folded, intent, "enactment_context")
            and _asks_any(folded, intent, "date_question")
        )
    )


def _asks_enactment_context(folded: str, intent: dict) -> bool:
    return _asks_any(folded, intent, "enactment_context") or _asks_token(folded, intent, "institution_tokens")


def _source_role(query: str, *, strategy: str, config=None) -> str | None:
    for role, pattern in intent_config_for(strategy, config)["metadata_roles"]:
        if pattern.search(query or ""):
            return role
    return None


def _metadata_result(store, row: dict, grounding: dict, field: str) -> dict | None:
    value = _field_value(row, field)
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
        "bbox_count": len(bboxes),
        "status": "final",
        "route_sources": ("metadata",),
        "route_score": 900.0,
        "route_scores": {"metadata": 900.0},
        "rank_reasons": ("metadata", "answer_evidence"),
        "metadata_grounding": True,
        "metadata_viewer_resolvable": _bboxes_have_coordinates(bboxes),
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
    return bool(bboxes) and all(
        box.get(key) is not None
        for box in bboxes
        for key in ("x0", "y0", "x1", "y1")
    )
