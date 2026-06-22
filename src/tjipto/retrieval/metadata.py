from __future__ import annotations


DEFAULT_TEMPORAL_CONTEXTS = {
    "current_consolidated",
    "original_historical",
    "amendment_1_historical",
    "amendment_2_historical",
    "amendment_3_historical",
    "amendment_4_historical",
}

DEFAULT_SOURCE_ROLES = DEFAULT_TEMPORAL_CONTEXTS


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
    source_roles = set(getattr(config, "source_roles", ()) or DEFAULT_SOURCE_ROLES)
    temporal_contexts = set(getattr(config, "temporal_contexts", ()) or DEFAULT_TEMPORAL_CONTEXTS)
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
