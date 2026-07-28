"""Deterministic, clause-grounded substantive-claim gate."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from tjipto.runtime.query_semantics import PropositionClaim, QuerySemantics


@dataclass(frozen=True)
class ClaimSupport:
    claim_id: str
    claim_text: str
    status: str
    support_evidence_ids: tuple[str, ...]
    source_role: str | None
    temporal_context: str | None
    validation_method: str
    reason_code: str | None
    predicate: str
    polarity: str
    modality: str
    text_span_ids: tuple[str, ...] = ()
    support_segments: tuple[dict[str, Any], ...] = ()

    def public(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "status": self.status,
            "support_evidence_ids": self.support_evidence_ids,
            "source_role": self.source_role,
            "temporal_context": self.temporal_context,
            "validation_method": self.validation_method,
            "reason_code": self.reason_code,
            "predicate": self.predicate,
            "polarity": self.polarity,
            "modality": self.modality,
            "text_span_ids": self.text_span_ids,
            "support_segments": self.support_segments,
        }


def verify_claims(semantics: QuerySemantics, evidence: tuple[dict, ...], store) -> tuple[ClaimSupport, ...]:
    if semantics.requested_function != "proposition_verification":
        return ()
    claim = semantics.requested_proposition
    if claim is None:
        return ()
    propositions = _grounded_propositions(store, evidence)
    support = tuple(proposition for proposition in propositions if _supports(claim, proposition))
    first = support[0] if support else (propositions[0] if propositions else None)
    # Absence is never an opposite proposition.  A future corpus parser may
    # emit an explicit opposite ClauseProposition; this conservative baseline
    # fails closed until it does.
    status = "supported" if support else "insufficient"
    support_ids = tuple(str(proposition["evidence_id"]) for proposition in support)
    return (
        ClaimSupport(
            claim_id="proposition:0",
            claim_text=claim.object,
            status=status,
            support_evidence_ids=support_ids,
            source_role=first.get("source_role") if first else claim.source_role,
            temporal_context=first.get("temporal_context") if first else claim.temporal_context,
            validation_method="artifact_backed_atomic_proposition",
            reason_code=None if support else "claim_support_insufficient",
            predicate=claim.predicate,
            polarity=claim.polarity,
            modality=claim.modality,
            text_span_ids=tuple(str(item) for item in support[0].get("text_span_ids") or ()) if support else (),
            support_segments=tuple(_public_segment(item) for item in support),
        ),
    )


def all_supported(claims: tuple[ClaimSupport, ...]) -> bool:
    return all(claim.status == "supported" for claim in claims)


def _grounded_propositions(store, evidence: tuple[dict, ...]) -> tuple[dict, ...]:
    units = {row.get("legal_unit_id") for row in evidence if row.get("legal_unit_id")}
    children = {
        row["legal_unit_id"]
        for row in store.legal_units
        if row.get("parent_legal_unit_id") in units and row.get("unit_type") == "ayat_record"
    }
    target_units = children or units
    return tuple(
        row
        for row in store.propositions
        if row.get("legal_unit_id") in target_units
        and row.get("evidence_id")
        and row.get("bbox_refs")
        and row.get("text_span_ids")
    )


def _public_segment(proposition: dict) -> dict[str, Any]:
    spans = tuple(str(item) for item in proposition.get("text_span_ids") or ())
    return {
        "segment_id": proposition.get("text_segment_id"),
        "exact_quote": proposition.get("exact_quote"),
        "start_selector": spans[0] if spans else None,
        "end_selector": spans[-1] if spans else None,
        "text_span_ids": spans,
        "bbox_refs": tuple(str(item) for item in proposition.get("bbox_refs") or ()),
        "page_numbers": tuple(proposition.get("page_numbers") or ()),
        "source_document_id": proposition.get("source_document_id"),
        "terminal_boundary": proposition.get("terminal_boundary"),
    }


def _supports(claim: PropositionClaim, proposition: dict) -> bool:
    # Normative parsing is deliberately not inferred from article words. Until
    # a strategy publishes a compatible structured proposition, it fails closed.
    return (
        claim.predicate == proposition.get("predicate")
        and claim.polarity == proposition.get("polarity")
        and claim.modality == proposition.get("modality")
        and claim.modality == "textual"
        and _contains_tokens(_tokens(claim.object), _tokens(str(proposition.get("object") or "")))
    )


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.casefold()))


def _contains_tokens(needle: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
    return bool(needle) and any(haystack[index : index + len(needle)] == needle for index in range(len(haystack)))
