"""Deterministic substantive-claim gate; candidate proximity is not authority."""

from __future__ import annotations

from dataclasses import dataclass
import re

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
        }


def verify_claims(semantics: QuerySemantics, evidence: tuple[dict, ...]) -> tuple[ClaimSupport, ...]:
    if semantics.requested_function != "proposition_verification":
        return ()
    claim = semantics.requested_proposition
    if claim is None:
        return ()
    support = tuple(
        row for row in evidence
        if _supports(claim, row)
    )
    contrary = tuple(
        row for row in evidence
        if claim.polarity == "negative" and _supports(claim, row, polarity="positive")
    )
    first = support[0] if support else (evidence[0] if evidence else {})
    status = "supported" if support else "contradicted" if contrary else "insufficient"
    support_ids = tuple(row["evidence_id"] for row in support or contrary)
    return (
        ClaimSupport(
            claim_id="proposition:0",
            claim_text=claim.object,
            status=status,
            support_evidence_ids=support_ids,
            source_role=first.get("source_role"),
            temporal_context=first.get("temporal_context"),
            validation_method="normative_text_term_containment",
            reason_code=None if support else "claim_support_contradicted" if contrary else "claim_support_insufficient",
            predicate=claim.predicate,
            polarity=claim.polarity,
            modality=claim.modality,
        ),
    )


def all_supported(claims: tuple[ClaimSupport, ...]) -> bool:
    return all(claim.status == "supported" for claim in claims)


def _terms(text: str) -> set[str]:
    ignored = {"apa", "aturan", "tentang", "yang", "dan", "atau", "dengan", "dalam", "itu", "ini"}
    return {token for token in re.findall(r"[a-z0-9]+", text.casefold()) if len(token) > 2 and token not in ignored}


def _supports(claim: PropositionClaim, row: dict, *, polarity: str | None = None) -> bool:
    text_terms = _terms(str(row.get("copy_text") or row.get("quoted_text") or ""))
    if not _terms(claim.object).issubset(text_terms):
        return False
    if (polarity or claim.polarity) == "negative":
        return False
    return claim.modality == "textual" or bool(set(claim.evidence_terms) & text_terms)
