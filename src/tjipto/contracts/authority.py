from __future__ import annotations

from typing import Final


FINAL_AUTHORITY_KINDS: Final = {"normative_legal_text"}
NONFINAL_AUTHORITY_KINDS: Final = {
    "structural_context",
    "exact_relation_support",
    "deterministic_structure",
    "endpoint_provenance",
    "instrument_provenance",
    "historical_mapping",
    "source_anomaly_trace",
    "metadata",
    "page_only",
    "rejected",
    "nonlegal",
}
AUTHORITY_KINDS: Final = FINAL_AUTHORITY_KINDS | NONFINAL_AUTHORITY_KINDS
EXACTNESS_VALUES: Final = {"exact", "not_applicable", "page_only", "rejected"}


def authority_decision(
    *,
    authority_kind: str,
    citable: bool,
    citation_final: bool,
    exactness: str,
    evidence_exists: bool,
    reason_code: str,
) -> dict:
    error = authority_state_error(
        authority_kind=authority_kind,
        citable=citable,
        citation_final=citation_final,
        exactness=exactness,
        evidence_exists=evidence_exists,
        reason_code=reason_code,
    )
    if error:
        raise ValueError(error)
    return {
        "authority_kind": authority_kind,
        "citable_status": "citable_exact" if citable else "not_citable",
        "citable": citable,
        "citation_final": citation_final,
        "citation_finality_reason": reason_code,
        "exactness": exactness,
        "evidence_exists": evidence_exists,
        "reason_code": reason_code,
    }


def authority_state_error(
    *,
    authority_kind: object,
    citable: object,
    citation_final: object,
    exactness: object,
    evidence_exists: object,
    reason_code: object,
) -> str | None:
    if authority_kind not in AUTHORITY_KINDS:
        return "authority_kind_unknown"
    if not isinstance(citable, bool) or not isinstance(citation_final, bool) or not isinstance(evidence_exists, bool):
        return "authority_boolean_invalid"
    if exactness not in EXACTNESS_VALUES:
        return "authority_exactness_unknown"
    if not isinstance(reason_code, str) or not reason_code:
        return "authority_reason_missing"
    if citation_final and authority_kind not in FINAL_AUTHORITY_KINDS:
        return "authority_nonlegal_final"
    if citation_final and (not citable or exactness != "exact" or not evidence_exists):
        return "authority_final_without_exact_evidence"
    if citable and (authority_kind not in FINAL_AUTHORITY_KINDS or exactness != "exact" or not evidence_exists):
        return "authority_citable_without_exact_evidence"
    if authority_kind in NONFINAL_AUTHORITY_KINDS and (citable or citation_final):
        return "authority_nonfinal_promoted"
    return None
