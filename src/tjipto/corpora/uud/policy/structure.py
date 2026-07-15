from __future__ import annotations

from tjipto.corpora.uud.specs import UUD_INSERTED_BAB_PREDECESSORS


def apply_uud_parent_policy(units: list[dict]) -> None:
    by_id = {row["legal_unit_id"]: row for row in units}
    for row in units:
        predecessors = set(UUD_INSERTED_BAB_PREDECESSORS.get(row.get("unit_label"), ()))
        if not predecessors:
            continue
        row["parent_legal_unit_ids"] = [
            parent_id
            for parent_id in row.get("parent_legal_unit_ids") or ()
            if by_id.get(parent_id, {}).get("unit_label") not in predecessors
        ]
