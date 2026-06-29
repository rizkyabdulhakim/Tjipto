from __future__ import annotations

import re


_GENERIC = {
    "document_target_words": (),
    "metadata_fields": {},
    "metadata_rules": {},
    "metadata_roles": (),
    "relation_words": (),
    "direct_relation_words": (),
    "pasal_parent_words": (),
    "relation_child_words": (),
    "unsupported_relation_context_words": (),
    "instrument_scope_queries": (),
    "instrument_deletion_words": (),
    "instrument_change_context_words": (),
    "source_role_labels": {},
    "structured_sections": (),
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
        "unsupported_relation_context_words": tuple(raw.get("unsupported_relation_context_words") or ()),
        "instrument_scope_queries": tuple(raw.get("instrument_scope_queries") or ()),
        "instrument_deletion_words": tuple(raw.get("instrument_deletion_words") or ()),
        "instrument_change_context_words": tuple(raw.get("instrument_change_context_words") or ()),
        "source_role_labels": dict(raw.get("source_role_labels") or {}),
        "structured_sections": tuple(raw.get("structured_sections") or ()),
    }
