from __future__ import annotations


def bbox_is_accepted(row: dict) -> bool:
    return row.get("status") == "accepted" and all(row.get(key) is not None for key in ("page_number", "x0", "y0", "x1", "y1"))
