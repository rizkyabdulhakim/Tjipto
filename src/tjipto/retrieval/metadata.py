from __future__ import annotations


SOURCE_ROLES = {
    "current_consolidated",
    "original_historical",
    "amendment_1_historical",
    "amendment_2_historical",
    "amendment_3_historical",
    "amendment_4_historical",
}


def normalize_filters(filters: dict | None = None, **kwargs) -> dict:
    raw = dict(filters or {})
    raw.update({key: value for key, value in kwargs.items() if value is not None})
    normalized: dict = {}
    if raw.get("source_role") in SOURCE_ROLES:
        normalized["source_role"] = raw["source_role"]
    normalized["status"] = "final"
    return normalized


def filter_evidence(rows: tuple[dict, ...], filters: dict) -> tuple[dict, ...]:
    return tuple(
        row for row in rows
        if row.get("status") == filters.get("status", "final")
        and (
            "source_role" not in filters
            or row.get("source_role") == filters["source_role"]
        )
    )
