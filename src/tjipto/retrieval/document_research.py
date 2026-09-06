from __future__ import annotations

import re
from typing import Any

from tjipto.corpora.parser_dispatch import parse_ayat_reference, parse_pasal_reference
from tjipto.evidence.store import EvidenceStore


def document_summary_rows(
    store: EvidenceStore,
    source_roles: tuple[str, ...],
    *,
    per_role: int | None = None,
) -> tuple[dict, ...]:
    """Return verified document-level coverage for a summary request."""
    units = {str(unit.get("legal_unit_id")): unit for unit in store.legal_units}
    instrument_types, normative_types = _unit_type_sets(store)
    top_level_normative_ids = {
        unit_id
        for unit_id, unit in units.items()
        if unit.get("unit_type") in normative_types and not unit.get("parent_legal_unit_ids")
    }
    rows: list[dict] = []
    for role in dict.fromkeys(str(value) for value in source_roles if value):
        amendment = role.startswith("amendment_")
        amendment_targets = _amendment_target_ids(store, role) if amendment else set()
        candidates = [
            row
            for row in store.evidence
            if row.get("source_role") == role
            and row.get("status") == "final"
            and store.bboxes_for(row.get("evidence_id"))
            and (
                str(row.get("legal_unit_id") or "") in top_level_normative_ids
                or (
                    amendment
                    and (
                        units.get(str(row.get("legal_unit_id") or ""), {}).get("unit_type") in instrument_types
                        or str(row.get("legal_unit_id") or "") in amendment_targets
                    )
                )
            )
        ]
        if not candidates:
            candidates = [
                row
                for row in store.evidence
                if row.get("source_role") == role
                and row.get("status") == "final"
                and store.bboxes_for(row.get("evidence_id"))
            ]
        candidates.sort(key=lambda row: _summary_row_order(row, units))
        rows.extend(candidates if per_role is None else candidates[: max(1, per_role)])
    return tuple(rows)


def version_comparison_rows(
    store: EvidenceStore,
    source_roles: tuple[str, ...],
    *,
    per_role: int | None = None,
    references: tuple[str, ...] = (),
) -> tuple[dict, ...]:
    """Return source-backed normative propositions for version comparison."""
    units = {str(unit.get("legal_unit_id")): unit for unit in store.legal_units}
    requested = {str(value).casefold().strip() for value in references if value}
    _, normative_types = _unit_type_sets(store)
    rows: list[dict] = []
    for role in dict.fromkeys(str(value) for value in source_roles if value):
        candidates = [
            row
            for row in store.evidence
            if row.get("source_role") == role
            and (not requested or _row_matches_reference(store, row, requested))
            and row.get("status") == "final"
            and units.get(str(row.get("legal_unit_id")), {}).get("unit_type") in normative_types
            and row.get("authority_kind") == "normative_legal_text"
            and row.get("citation_eligibility") == "eligible"
            and row.get("relevant_quote_eligible") is True
            and store.bboxes_for(row.get("evidence_id"))
        ]
        candidates.sort(key=lambda row: _summary_row_order(row, units))
        rows.extend(candidates if per_role is None else candidates[: max(1, per_role)])
    return tuple(rows)


def _unit_type_sets(store: EvidenceStore) -> tuple[set[str], set[str]]:
    config = getattr(store, "config", None)
    raw_schema = config.setting("schema", {}) if config is not None else {}
    schema: dict[str, Any] = raw_schema if isinstance(raw_schema, dict) else {}
    hierarchy = {str(value) for value in schema.get("unit_hierarchy", ()) if value}
    instrument = {str(value) for value in schema.get("instrument_unit_types", ()) if value}
    return instrument, hierarchy - instrument


def _amendment_target_ids(store: EvidenceStore, source_role: str) -> set[str]:
    return {
        str(target_id)
        for edge in store.graph_edges
        if (relation := edge.get("relation_projection") or {}).get("source_role") == source_role
        and relation.get("target_source_role") == source_role
        and (target_id := relation.get("target_legal_unit_id"))
    }


def _row_matches_reference(store: EvidenceStore, row: dict, requested: set[str]) -> bool:
    labels = {
        _normalise_reference_label(value)
        for value in (row.get("citation"), *(row.get("hierarchy") or ()))
        if value
    }
    corpus_id = str(getattr(store.config, "corpus_id", ""))
    for reference in requested:
        parts = tuple(
            value
            for value in (
                parse_pasal_reference(corpus_id, reference, allow_roman=True, config=store.config),
                parse_ayat_reference(corpus_id, reference, config=store.config),
            )
            if value
        )
        if {_normalise_reference_label(value) for value in (parts or (reference,))} <= labels:
            return True
    return False


def _normalise_reference_label(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold().strip())


def _summary_row_order(row: dict, units: dict[str, dict]) -> tuple[int, int, str]:
    unit = units.get(str(row.get("legal_unit_id")), {})
    page = unit.get("page_start") or row.get("page_number") or (row.get("page_numbers") or [0])[0]
    return int(page or 0), int(unit.get("sibling_order") or 0), str(row.get("evidence_id") or "")
