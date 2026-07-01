from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class InstrumentIntentDecision:
    corpus: str
    normalized_query: str
    role_family: str | None
    amendment: str | None
    target_status: str
    fallback_permission: bool
    reason: str
    target_citation: str | None = None


_GENERIC = {
    "document_target_words": (),
    "metadata_fields": {},
    "metadata_rules": {},
    "metadata_roles": (),
    "relation_words": (),
    "direct_relation_words": (),
    "pasal_parent_words": (),
    "relation_child_words": (),
    "relation_routes": {},
    "unsupported_relation_context_words": (),
    "instrument_scope_queries": (),
    "instrument_deletion_words": (),
    "instrument_deletion_evidence_words": (),
    "instrument_change_context_words": (),
    "instrument_citation_templates": {},
    "instrument_role_queries": {},
    "instrument_intent_blocking_queries": (),
    "instrument_intent_matrix": {},
    "source_role_labels": {},
    "structured_sections": (),
    "structured_lookup_enabled": False,
}


def intent_config_for(strategy: str | None, config=None) -> dict:
    raw = config.setting("intent_config") if config is not None else None
    if not raw:
        return _GENERIC
    return {
        "document_target_words": tuple(raw.get("document_target_words") or ()),
        "metadata_fields": {
            key: tuple(value)
            for key, value in (raw.get("metadata_fields") or {}).items()
        },
        "metadata_rules": {
            key: tuple(value)
            for key, value in (raw.get("metadata_rules") or {}).items()
        },
        "metadata_roles": tuple(
            (row["role"], re.compile(row["pattern"], re.IGNORECASE))
            for row in raw.get("metadata_roles", ())
        ),
        "relation_words": tuple(raw.get("relation_words") or ()),
        "direct_relation_words": tuple(raw.get("direct_relation_words") or ()),
        "pasal_parent_words": tuple(raw.get("pasal_parent_words") or ()),
        "relation_child_words": tuple(raw.get("relation_child_words") or ()),
        "relation_routes": dict(raw.get("relation_routes") or {}),
        "unsupported_relation_context_words": tuple(raw.get("unsupported_relation_context_words") or ()),
        "instrument_scope_queries": tuple(raw.get("instrument_scope_queries") or ()),
        "instrument_deletion_words": tuple(raw.get("instrument_deletion_words") or ()),
        "instrument_deletion_evidence_words": tuple(raw.get("instrument_deletion_evidence_words") or ()),
        "instrument_change_context_words": tuple(raw.get("instrument_change_context_words") or ()),
        "instrument_citation_templates": dict(raw.get("instrument_citation_templates") or {}),
        "instrument_role_queries": {
            key: tuple(value)
            for key, value in (raw.get("instrument_role_queries") or {}).items()
        },
        "instrument_intent_blocking_queries": tuple(raw.get("instrument_intent_blocking_queries") or ()),
        "instrument_intent_matrix": dict(raw.get("instrument_intent_matrix") or {}),
        "source_role_labels": dict(raw.get("source_role_labels") or {}),
        "structured_sections": tuple(raw.get("structured_sections") or ()),
        "structured_lookup_enabled": bool(raw.get("structured_lookup_enabled")),
    }


def normalize_intent_text(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"(?<=\bke)-(?=\d+\b)", " ", text)
    text = re.sub(r"[^0-9a-z]+", " ", text)
    return " ".join(text.split())


def contains_intent_phrase(text: str, aliases: tuple[str, ...] | list[str]) -> bool:
    haystack = f" {normalize_intent_text(text)} "
    return any(f" {normalize_intent_text(alias)} " in haystack for alias in aliases)


def resolve_instrument_intent(query: str, intent: dict, *, corpus: str = "") -> InstrumentIntentDecision:
    normalized = normalize_intent_text(query)
    if not normalized or contains_intent_phrase(query, intent.get("instrument_intent_blocking_queries", ())):
        return InstrumentIntentDecision(corpus, normalized, None, None, "not_instrument_intent", True, "not_instrument_intent")
    role = next(
        (key for key, aliases in intent.get("instrument_role_queries", {}).items() if contains_intent_phrase(query, aliases)),
        None,
    )
    amendment = next((source_role for source_role, pattern in intent.get("metadata_roles", ()) if pattern.search(query or "")), None)
    if role is None or amendment is None:
        return InstrumentIntentDecision(corpus, normalized, role, amendment, "not_instrument_intent", True, "not_instrument_intent")
    citation = _instrument_citation(intent, amendment, role, query)
    if not citation:
        return InstrumentIntentDecision(corpus, normalized, role, amendment, "instrument_like_unresolved", False, "instrument_like_unresolved")
    return InstrumentIntentDecision(corpus, normalized, role, amendment, "resolved_target_fail_closed", False, "instrument_target_fail_closed", citation)


def _instrument_citation(intent: dict, role: str, key: str, query: str) -> str:
    template = intent.get("instrument_citation_templates", {}).get(key, "")
    if not template:
        return ""
    values = {"ordinal": intent.get("source_role_labels", {}).get(role, "")}
    if "{clause}" in template:
        match = re.search(r"\b(?:clause|klausul|huruf)\s*\(?([a-e])\)?|\(([a-e])\)", query or "", re.IGNORECASE)
        if not match:
            return ""
        values["clause"] = (match.group(1) or match.group(2)).lower()
    return template.format(**values)
