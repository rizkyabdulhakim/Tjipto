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
    conditions: tuple[str, ...] = ()
    exceptions: tuple[str, ...] = ()
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
            "conditions": self.conditions,
            "exceptions": self.exceptions,
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
    contradiction = () if support else tuple(proposition for proposition in propositions if _contradicts(claim, proposition))
    selected = support or contradiction
    first = selected[0] if selected else (propositions[0] if propositions else None)
    status = "supported" if support else "contradicted" if contradiction else "insufficient"
    support_ids = tuple(str(proposition["evidence_id"]) for proposition in selected)
    return (
        ClaimSupport(
            claim_id="proposition:0",
            claim_text=claim.object,
            status=status,
            support_evidence_ids=support_ids,
            source_role=first.get("source_role") if first else claim.source_role,
            temporal_context=first.get("temporal_context") if first else claim.temporal_context,
            validation_method="artifact_backed_atomic_proposition",
            reason_code=None if support else "claim_explicitly_contradicted" if contradiction else "claim_support_insufficient",
            predicate=claim.predicate,
            polarity=claim.polarity,
            modality=claim.modality,
            conditions=claim.conditions,
            exceptions=claim.exceptions,
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
        if row.get("parent_legal_unit_id") in units and row.get("structural_role") == "subprovision"
    }
    target_units = children or units
    return tuple(
        row
        for row in store.propositions
        if row.get("legal_unit_id") in target_units
        and row.get("evidence_id")
        and row.get("bbox_refs")
        and row.get("text_span_ids")
        and row.get("source_selectors")
    )


def _public_segment(proposition: dict) -> dict[str, Any]:
    spans = tuple(str(item) for item in proposition.get("text_span_ids") or ())
    return {
        "proposition_id": proposition.get("proposition_id"),
        "evidence_id": proposition.get("evidence_id"),
        "legal_unit_id": proposition.get("legal_unit_id"),
        "segment_id": proposition.get("text_segment_id"),
        "exact_quote": proposition.get("exact_quote"),
        "start_selector": spans[0] if spans else None,
        "end_selector": spans[-1] if spans else None,
        "text_span_ids": spans,
        "bbox_refs": tuple(str(item) for item in proposition.get("bbox_refs") or ()),
        "page_numbers": tuple(proposition.get("page_numbers") or ()),
        "source_document_id": proposition.get("source_document_id"),
        "terminal_boundary": proposition.get("terminal_boundary"),
        "viewer_overlay": proposition.get("viewer_overlay"),
    }


def _supports(claim: PropositionClaim, proposition: dict) -> bool:
    return _compatible(claim, proposition) and _content_matches(claim, proposition)


def _contradicts(claim: PropositionClaim, proposition: dict) -> bool:
    if not _compatible(claim, proposition, allow_opposite=True) or not _content_matches(claim, proposition):
        return False
    return (claim.predicate, proposition.get("predicate")) in {
        ("requires", "prohibits"),
        ("prohibits", "requires"),
    } or claim.polarity != proposition.get("polarity")


def _compatible(claim: PropositionClaim, proposition: dict, *, allow_opposite: bool = False) -> bool:
    predicate_matches = claim.predicate == proposition.get("predicate")
    if allow_opposite:
        predicate_matches = predicate_matches or (claim.predicate, proposition.get("predicate")) in {
            ("requires", "prohibits"),
            ("prohibits", "requires"),
        }
    modality_matches = claim.modality == proposition.get("modality")
    if allow_opposite and {claim.modality, proposition.get("modality")} == {"obligation", "prohibition"}:
        modality_matches = True
    if not predicate_matches or not modality_matches:
        return False
    if not allow_opposite and claim.polarity != proposition.get("polarity"):
        return False
    if claim.source_role and claim.source_role != proposition.get("source_role"):
        return False
    if claim.temporal_context and claim.temporal_context != proposition.get("temporal_context"):
        return False
    if tuple(proposition.get("conditions") or ()) != claim.conditions:
        return False
    if tuple(proposition.get("exceptions") or ()) != claim.exceptions:
        return False
    return not claim.subject or _contains_tokens(_tokens(claim.subject), _tokens(str(proposition.get("subject") or "")))


def _content_matches(claim: PropositionClaim, proposition: dict) -> bool:
    query_tokens = _tokens(claim.object)
    if not query_tokens:
        return False
    source_text = str(proposition.get("object") or "")
    if proposition.get("claim_type") == "normative_proposition":
        source_text = f"{proposition.get('subject') or ''} {source_text}"
        return _ordered_subsequence(query_tokens, _tokens(source_text))
    return _contains_tokens(query_tokens, _tokens(source_text))


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.casefold()))


def _contains_tokens(needle: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
    return bool(needle) and any(haystack[index : index + len(needle)] == needle for index in range(len(haystack)))


def _ordered_subsequence(needle: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
    """Allow omitted modifiers without reversing a grounded proposition's roles."""
    iterator = iter(haystack)
    return bool(needle) and all(any(token == candidate for candidate in iterator) for token in needle)
